"""
本地化 LGBM 多因子选股 + Backtrader 回测
"""
import logging
import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import backtrader as bt
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tushare as ts
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

TUSHARE_TOKEN = os.environ.get("TUSHARE_TOKEN", "your_tushare_token")
TUSHARE_HTTP_URL = os.environ.get("TUSHARE_HTTP_URL", "your_tushare_http_url")

LOG_FILE = Path(__file__).with_suffix(".txt")

_log_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_log_formatter)
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(_log_formatter)

logging.basicConfig(
	level=logging.INFO,
	handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
	backtest_start: str
	backtest_end: str
	index_code: str
	stock_count: int
	topk: int
	hold_buffer_size: int
	min_rebalance_threshold: float
	hold_days: int
	retrain_interval: int
	retrain_lookback_days: int
	label_horizon: int
	min_train_samples: int
	enable_grid_search: bool
	grid_hold_days: Tuple[int, ...]
	grid_label_horizons: Tuple[int, ...]
	grid_eval_points: int


def init_backtest_config(
	backtest_start: str = "2024-01-01",
	backtest_end: str = "2026-01-01",
	index_code: str = "000300.SH",
	stock_count: int = 100,
	topk: int = 20,
	hold_buffer_size: int = 5,
	min_rebalance_threshold: float = 0.02,
	hold_days: int = 20, # monthly rebalance
	retrain_interval: int = 60,
	retrain_lookback_days: int = 365 * 2,
	label_horizon: int = 20, # match with hold_days
	min_train_samples: int = 300,
	enable_grid_search: bool = True,
	grid_hold_days: Tuple[int, ...] = (10, 15, 20, 30),
	grid_label_horizons: Tuple[int, ...] = (10, 15, 20, 30),
	grid_eval_points: int = 80,
) -> BacktestConfig:
	"""config initialization"""
	return BacktestConfig(
		backtest_start=backtest_start,
		backtest_end=backtest_end,
		index_code=index_code,
		stock_count=stock_count,
		topk=topk,
		hold_buffer_size=hold_buffer_size,
		min_rebalance_threshold=min_rebalance_threshold,
		hold_days=hold_days,
		retrain_interval=retrain_interval,
		retrain_lookback_days=retrain_lookback_days,
		label_horizon=label_horizon,
		min_train_samples=min_train_samples,
		enable_grid_search=enable_grid_search,
		grid_hold_days=grid_hold_days,
		grid_label_horizons=grid_label_horizons,
		grid_eval_points=grid_eval_points,
	)

def _to_tushare_date(date_str: str) -> str:
	return pd.Timestamp(date_str).strftime("%Y%m%d")

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
	delta = series.diff()
	gain = delta.clip(lower=0.0)
	loss = -delta.clip(upper=0.0)
	avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
	avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
	rs = avg_gain / avg_loss.replace(0, np.nan)
	return 100 - (100 / (1 + rs))


def ic_scorer(estimator, x_test, y_test):
	y_pred = estimator.predict_proba(x_test)[:, 1]
	ic, _ = spearmanr(y_test, y_pred)
	return 0.0 if np.isnan(ic) else float(ic)


class DateCleaner(BaseEstimator, TransformerMixin):
	def __init__(self, handle_outliers: bool = True, outlier_std: float = 3.0):
		self.handle_outliers = handle_outliers
		self.outlier_std = outlier_std
		self.outlier_bounds_: Dict[int, Dict[str, float]] = {}

	def fit(self, x, y=None):
		"""Compute per-feature outlier bounds from the training data."""
		if self.handle_outliers:
			x_df = pd.DataFrame(x)
			for col in x_df.columns:
				mean = float(x_df[col].mean())
				std = float(x_df[col].std())
				self.outlier_bounds_[int(col)] = {
					"lower": mean - self.outlier_std * std,
					"upper": mean + self.outlier_std * std,
				}
		return self

	def transform(self, x):
		x_df = pd.DataFrame(x).ffill().bfill()
		if self.handle_outliers and self.outlier_bounds_:
			for col in x_df.columns:
				if int(col) in self.outlier_bounds_:
					bounds = self.outlier_bounds_[int(col)]
					x_df[col] = x_df[col].clip(bounds["lower"], bounds["upper"])
		x_df = x_df.replace([np.inf, -np.inf], np.nan).fillna(0)
		return x_df.values


class FeatureEngineer(BaseEstimator, TransformerMixin):
	def __init__(self, create_interactions: bool = True, create_ratio: bool = True):
		self.create_interactions = create_interactions
		self.create_ratio = create_ratio

	def fit(self, x, y=None):
		return self

	def transform(self, x):
		x_df = pd.DataFrame(x)
		new_features = [x_df.values]
		cap = min(5, x_df.shape[1])
		if self.create_interactions:
			for i in range(cap):
				for j in range(i + 1, cap):
					new_features.append((x_df.iloc[:, i] * x_df.iloc[:, j]).values.reshape(-1, 1))
		if self.create_ratio:
			for i in range(cap):
				for j in range(cap):
					if i != j:
						ratio = x_df.iloc[:, i] / (x_df.iloc[:, j] + 1e-8)
						new_features.append(ratio.values.reshape(-1, 1))
		return np.hstack(new_features)


class FeatureSelector(BaseEstimator, TransformerMixin):
	"select the top n features of the most importance"
	def __init__(self, n_features: int = 25):
		self.n_features = n_features
		self.selected_features_: Optional[np.ndarray] = None

	def fit(self, x, y):
		model = lgb.LGBMClassifier(
			objective="binary",
			n_estimators=300,
			learning_rate=0.07,
			max_depth=5,
			num_leaves=31,
			subsample=0.8,
			colsample_bytree=0.7,
			min_child_samples=25,
			reg_lambda=1.0,
			reg_alpha=0.5,
			scale_pos_weight=2.5,
			random_state=42,
			n_jobs=1,
			verbosity=-1,
		)
		model.fit(x, y)
		importance_idx = np.argsort(model.feature_importances_)[::-1]
		self.selected_features_ = importance_idx[: self.n_features]
		return self

	def transform(self, x):
		if self.selected_features_ is not None:
			return x[:, self.selected_features_]
		return x


class QuantPipeline:
	def __init__(self):
		self.pipeline = None

	def create_pipeline(self):
		steps = [
			("cleaner", DateCleaner()),
			("engineer", FeatureEngineer()),
			("scaler", StandardScaler()),
			("selector", FeatureSelector()),
			(
				"model",
				lgb.LGBMClassifier(
					objective="binary",
					metric="auc",
					boosting_type="gbdt",
					subsample=0.8,
					colsample_bytree=0.7,
					reg_lambda=1.0,
					reg_alpha=0.5,
					random_state=42,
					verbosity=-1,
					n_jobs=1,
				),
			),
		]
		pipe = Pipeline(steps=steps)
		param_grid = {
			"model__num_leaves": [31, 63],
			"model__learning_rate": [0.07],
			"model__min_child_samples": [20, 30],
			"model__max_depth": [5],
			"model__n_estimators": [200, 400],
		}
		cv = TimeSeriesSplit(n_splits=2)
		self.pipeline = GridSearchCV(
			estimator=pipe,
			param_grid=param_grid,
			scoring={"roc_auc": "roc_auc", "ic": ic_scorer},
			cv=cv,
			n_jobs=1,
			verbose=0,
			refit="ic",
		)
		return self.pipeline


@dataclass
class StrategyState:
	hold_days: int
	topk: int
	retrain_interval: int
	retrain_lookback_days: int
	last_rebalance_date: Optional[pd.Timestamp] = None
	last_retrain_date: Optional[pd.Timestamp] = None


def get_index_constituents(pro, index_code: str, end_date: str) -> List[str]:
	"get the latest constituents of the index as of end_date"
	end_ts = _to_tushare_date(end_date)
	start_ts = _to_tushare_date((pd.Timestamp(end_date) - timedelta(days=45)).strftime("%Y-%m-%d"))
	idx = pro.index_weight(index_code=index_code, start_date=start_ts, end_date=end_ts)
	if idx.empty:
		raise RuntimeError("未获取到指数成分股")
	idx = idx.sort_values("trade_date").drop_duplicates("con_code", keep="last")
	return idx["con_code"].tolist()

def get_data(
	pro,
	index_code: str,
	start_date: str,
	end_date: str,
	stock_count: int = 60,
) -> pd.DataFrame:
	all_codes = get_index_constituents(pro, index_code=index_code, end_date=end_date)
	start_ts = _to_tushare_date(start_date)
	end_ts = _to_tushare_date(end_date)
	daily_basic = pro.daily_basic(
		ts_code="",
		start_date=start_ts,
		end_date=end_ts,
		fields="ts_code,trade_date,turnover_rate,pe,pb,total_mv",
	)
	if daily_basic.empty:
		raise RuntimeError("daily_basic 返回为空，无法筛选样本")
	liquid = (
		daily_basic[daily_basic["ts_code"].isin(all_codes)]
		.groupby("ts_code")["turnover_rate"]
		.mean()
		.sort_values(ascending=False)
		.head(stock_count)
	)
	selected = liquid.index.tolist()

	frames = []
	for code in selected:
		price = pro.daily(
			ts_code=code,
			start_date=start_ts,
			end_date=end_ts,
			fields="ts_code,trade_date,open,high,low,close,vol,amount",
		)
		if price.empty:
			continue
		frames.append(price)

	if not frames:
		raise RuntimeError("样本股票没有拉取到有效行情")

	df = pd.concat(frames, ignore_index=True) # 合并多只股票的拉取结果并重索引
	df = df.rename(columns={"ts_code": "code", "trade_date": "date", "vol": "volume"})
	df["date"] = pd.to_datetime(df["date"])
	df = df.sort_values(["code", "date"]).reset_index(drop=True)
	return df[["date", "code", "open", "high", "low", "close", "volume", "amount"]]


def get_benchmark_close(pro, index_code: str, start_date: str, end_date: str) -> pd.DataFrame:
	"""Fetch benchmark index close series for comparison plotting."""
	bench = pro.index_daily(
		ts_code=index_code,
		start_date=_to_tushare_date(start_date),
		end_date=_to_tushare_date(end_date),
		fields="ts_code,trade_date,close",
	)
	if bench.empty:
		return pd.DataFrame(columns=["date", "benchmark_close"])
	bench = bench.rename(columns={"trade_date": "date", "close": "benchmark_close"})
	bench["date"] = pd.to_datetime(bench["date"])
	bench = bench.sort_values("date").drop_duplicates("date", keep="last")
	return bench[["date", "benchmark_close"]]


def generate_advanced_features(df: pd.DataFrame, label_horizon: int = 20) -> pd.DataFrame:
	df = df.sort_values(["code", "date"]).reset_index(drop=True)
	df["returns"] = df.groupby("code")["close"].pct_change()
	for p in [5, 10, 20, 30]:
		df[f"ema{p}"] = df.groupby("code")["close"].transform(lambda x: x.ewm(span=p, adjust=False).mean())
		df[f"ma{p}"] = df.groupby("code")["close"].transform(lambda x: x.rolling(p, min_periods=1).mean())
		df[f"std{p}"] = df.groupby("code")["returns"].transform(lambda x: x.rolling(p, min_periods=1).std())

	df["high_low_ratio"] = df.groupby("code")["close"].transform(lambda x: x / x.rolling(20, min_periods=1).max())
	df["turnover"] = df["amount"].fillna(df["close"] * df["volume"])
	df["turnover_ma20"] = df.groupby("code")["turnover"].transform(lambda x: x.rolling(20, min_periods=1).mean())
	df["volume_ma5"] = df.groupby("code")["volume"].transform(lambda x: x.rolling(5, min_periods=1).mean())
	df["price_skew"] = df.groupby("code")["returns"].transform(lambda x: x.rolling(20, min_periods=1).skew())
	df["price_kurt"] = df.groupby("code")["returns"].transform(lambda x: x.rolling(20, min_periods=1).kurt())
	df["volume_std"] = df.groupby("code")["volume"].transform(lambda x: x.rolling(20, min_periods=1).std())
	df["volume_zscore"] = (df["volume"] - df["volume_ma5"]) / df["volume_std"].replace(0, np.nan)

	market_ret = df.groupby("date")["returns"].mean().rename("market_return")
	df = df.merge(market_ret.reset_index(), on="date", how="left")
	df["relative_strength"] = df["returns"] - df["market_return"]

	for p in [6, 14]:
		df[f"rsi{p}"] = df.groupby("code")["close"].transform(lambda x: calc_rsi(x, period=p))

	def _indicator_block(g, code_value):
		g = g.copy()
		g["code"] = code_value
		close = g["close"]
		high = g["high"]
		low = g["low"]
		vol = g["volume"]

		ema12 = close.ewm(span=12, adjust=False).mean()
		ema26 = close.ewm(span=26, adjust=False).mean()
		g["macd"] = ema12 - ema26
		g["macd_signal"] = g["macd"].ewm(span=9, adjust=False).mean()
		g["macd_hist"] = g["macd"] - g["macd_signal"]

		ll = low.rolling(14, min_periods=1).min()
		hh = high.rolling(14, min_periods=1).max()
		k = ((close - ll) / (hh - ll).replace(0, np.nan) * 100).clip(0, 100)
		d = k.rolling(3, min_periods=1).mean()
		g["stoch_k"] = k
		g["stoch_d"] = d
		g["stoch_j"] = 3 * k - 2 * d

		bb_m = close.rolling(20, min_periods=1).mean()
		bb_std = close.rolling(20, min_periods=1).std()
		g["bb_upper"] = bb_m + 2 * bb_std
		g["bb_middle"] = bb_m
		g["bb_lower"] = bb_m - 2 * bb_std
		g["bb_bw"] = (g["bb_upper"] - g["bb_lower"]) / bb_m.replace(0, np.nan)
		g["bb_percB"] = (close - g["bb_lower"]) / (g["bb_upper"] - g["bb_lower"]).replace(0, np.nan)

		tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
		atr_14 = tr.rolling(14, min_periods=1).mean()
		g["atr_14"] = atr_14
		g["atr_14_pct"] = atr_14 / close.replace(0, np.nan)

		direction = np.sign(close.diff().fillna(0))
		g["obv"] = (direction * vol).cumsum()
		g["obv_chg_5"] = g["obv"].pct_change(5)

		up_move = high.diff()
		down_move = -low.diff()
		plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
		minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
		tr14 = tr.rolling(14, min_periods=1).sum().replace(0, np.nan)
		plus_di = 100 * pd.Series(plus_dm, index=g.index).rolling(14, min_periods=1).sum() / tr14
		minus_di = 100 * pd.Series(minus_dm, index=g.index).rolling(14, min_periods=1).sum() / tr14
		dx = (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan) * 100
		g["adx_14"] = dx.rolling(14, min_periods=1).mean()
		return g

	indicator_frames = []
	for code_value, group in df.groupby("code", sort=False):
		indicator_frames.append(_indicator_block(group, code_value))
	df = pd.concat(indicator_frames, ignore_index=True)

	h = int(label_horizon)
	df["future_return"] = df.groupby("code")["close"].transform(lambda x: x.shift(-h) / x - 1)
	df["label"] = df.groupby("date")["future_return"].transform(lambda x: (x > x.quantile(0.7)).astype(float))
	return df


class PandasDataExtend(bt.feeds.PandasData):
	lines = ("code",)
	params = (("datetime", None), ("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("volume", "volume"), ("openinterest", None), ("code", "code_int"))


class MyStrategy(bt.Strategy):
	params = dict(
		hold_days=20,
		topk=10,
		hold_buffer_size=5,
		min_rebalance_threshold=0.02,
		retrain_interval=60,
		retrain_lookback_days=365 * 2,
		min_train_samples=500,
		feature_df=None,
		feature_cols=None,
		code_map=None,
	)

	def __init__(self):
		self.state = StrategyState(
			hold_days=self.p.hold_days,
			topk=self.p.topk,
			retrain_interval=self.p.retrain_interval,
			retrain_lookback_days=self.p.retrain_lookback_days,
		)
		self.feature_df = self.p.feature_df.copy()
		self.feature_cols = self.p.feature_cols
		self.code_map = self.p.code_map or {}
		self.pipeline = None
		self.pending_order_count = 0

	def notify_order(self, order):
		if order.status in [order.Submitted, order.Accepted]:
			return

		status_name = order.getstatusname()
		code = None
		if getattr(order, "data", None) is not None:
			code = self.code_map.get(int(order.data.code[0]), None)

		if order.status in [order.Completed]:
			self.pending_order_count = max(self.pending_order_count - 1, 0)
			action = "买入" if order.isbuy() else "卖出"
			logger.info(
				"订单执行: %s | 股票: %s | 成交价: %.2f | 数量: %s | 成交金额: %.2f | 状态: %s",
				action,
				code or "UNKNOWN",
				order.executed.price,
				order.executed.size,
				order.executed.value,
				status_name,
			)
		elif order.status in [order.Canceled, order.Margin, order.Rejected]:
			self.pending_order_count = max(self.pending_order_count - 1, 0)
			action = "买入" if order.isbuy() else "卖出"
			logger.warning(
				"订单被拒绝/取消/资金不足 | %s | 股票: %s | 状态码: %s",
				action,
				code or "UNKNOWN",
				status_name,
			)
		else:
			logger.info("订单异常状态 | 股票: %s | 状态码: %s", code or "UNKNOWN", status_name)

	def _current_date(self) -> pd.Timestamp:
		return pd.Timestamp(self.datas[0].datetime.date(0))

	def _trade_dates_between(self, start_date: pd.Timestamp, end_date: pd.Timestamp) -> int:
		"count the number of unique trade dates"
		all_dates = self.feature_df["date"].drop_duplicates().sort_values()
		mask = (all_dates >= start_date) & (all_dates <= end_date)
		return int(mask.sum())
	

	def _need_retrain(self, now: pd.Timestamp) -> bool:
		"judge whether need to retrain model on fixed rolling schedule"
		if self.pipeline is None:
			return True
		if self.state.last_retrain_date is None:
			return True
		return self._trade_dates_between(self.state.last_retrain_date, now) - 1 >= self.state.retrain_interval

	def _slice_train_df(self, now: pd.Timestamp) -> pd.DataFrame:
		lookback_start = now - timedelta(days=self.state.retrain_lookback_days)
		train_df = self.feature_df[(self.feature_df["date"] >= lookback_start) & (self.feature_df["date"] < now)].copy()
		train_df[self.feature_cols] = train_df[self.feature_cols].replace([np.inf, -np.inf], np.nan)
		train_df[self.feature_cols] = train_df[self.feature_cols].fillna(0.0)
		train_df = train_df.dropna(subset=["label"])
		return train_df

	def _fit_model(self, now: pd.Timestamp, reason: str = "rolling") -> bool:
		"main model training logic"
		train_df = self._slice_train_df(now)
		if len(train_df) < self.p.min_train_samples:
			logger.info("跳过训练: 样本不足 | now=%s | size=%d | min=%d", now.date(), len(train_df), self.p.min_train_samples)
			return False
		if train_df["label"].nunique() < 2:
			logger.info("跳过训练: 标签单一 | now=%s", now.date())
			return False
		
		# create and fit the pipeline 
		pipeline = QuantPipeline().create_pipeline()
		pipeline.fit(train_df[self.feature_cols].values, train_df["label"].values)
		self.pipeline = pipeline
		self.state.last_retrain_date = now # update the last retrain date
		logger.info("模型训练完成 | now=%s | reason=%s | 样本=%d", now.date(), reason, len(train_df))
		return True

	def _today_cross_section(self, now: pd.Timestamp) -> pd.DataFrame:
		"get the cross section data of today for prediction"
		cross = self.feature_df[self.feature_df["date"] == now].copy()
		if cross.empty:
			logger.warning("Warning: No data for date %s to predict.", now.date())
			return cross
		cross[self.feature_cols] = cross[self.feature_cols].replace([np.inf, -np.inf], np.nan)
		cross[self.feature_cols] = cross[self.feature_cols].fillna(0.0)
		return cross

	def _rebalance(self, now: pd.Timestamp):
		"main rebalance logic"
		if self.pipeline is None:
			return
		cross = self._today_cross_section(now)
		if cross.empty:
			return

		probs = self.pipeline.predict_proba(cross[self.feature_cols].values)[:, 1]
		ranked_all = cross.assign(prob=probs).sort_values("prob", ascending=False)
		buy_targets = set(ranked_all.head(self.state.topk)["code"].tolist())
		buffer_universe = set(ranked_all.head(self.state.topk + self.p.hold_buffer_size)["code"].tolist())

		current_holds = set()
		for data in self.datas:
			code = self.code_map.get(int(data.code[0]))
			if code and self.getposition(data).size != 0:
				current_holds.add(code)

		# 持仓缓冲带: 当前持仓如果仍在 topk + buffer 内，则继续持有，降低边界抖动换手。
		targets = buy_targets | (current_holds & buffer_universe)
		total_value = max(float(self.broker.getvalue()), 1.0)

		def current_weight_of(data_obj) -> float:
			pos = self.getposition(data_obj)
			if pos.size == 0:
				return 0.0
			px = float(data_obj.close[0])
			if not np.isfinite(px) or px <= 0:
				return 0.0
			return float(pos.size * px / total_value)

		# 先清仓非目标持仓。
		for data in self.datas:
			code = self.code_map.get(int(data.code[0]))
			if code and code not in targets:
				if abs(current_weight_of(data) - 0.0) < self.p.min_rebalance_threshold:
					continue
				self.pending_order_count += 1
				self.order_target_percent(data=data, target=0.0)

		# 再等权买入目标持仓
		# 预留少量现金（2%）以应对可能的订单执行失败和市场波动
		w = 0.98 / max(len(targets), 1)
		for data in self.datas:
			code = self.code_map.get(int(data.code[0]))
			if code and code in targets:
				if abs(current_weight_of(data) - w) < self.p.min_rebalance_threshold:
					continue
				self.pending_order_count += 1
				self.order_target_percent(data=data, target=w)

		self.state.last_rebalance_date = now

	def next(self):
		now = self._current_date()
		if self._need_retrain(now):
			self._fit_model(now)

		if self.state.last_rebalance_date is not None:
			days_passed = self._trade_dates_between(self.state.last_rebalance_date, now) - 1
			if days_passed < self.state.hold_days:
				return
		self._rebalance(now)


class EquityCurveAnalyzer(bt.Analyzer):
	def start(self):
		self.records = []

	def next(self):
		dt = self.strategy.datas[0].datetime.datetime(0)
		self.records.append(
			{
				"date": pd.Timestamp(dt),
				"portfolio_value": float(self.strategy.broker.getvalue()),
				"cash": float(self.strategy.broker.getcash()),
			}
		)

	def get_analysis(self):
		return self.records


def save_equity_curve_plot(
	equity_records: List[Dict[str, float]],
	benchmark_df: pd.DataFrame,
	output_png: Path,
	output_csv: Path,
	sharpe: float = np.nan,
	max_drawdown: float = np.nan,
) -> Dict[str, float]:
	if not equity_records:
		logger.warning("没有可视化数据，未生成净值曲线")
		return {}
	curve_df = pd.DataFrame(equity_records).drop_duplicates(subset=["date"]).sort_values("date")
	curve_df["strategy_nav"] = curve_df["portfolio_value"] / curve_df["portfolio_value"].iloc[0]

	if benchmark_df is not None and not benchmark_df.empty:
		curve_df = curve_df.merge(benchmark_df, on="date", how="left")
		curve_df["benchmark_close"] = curve_df["benchmark_close"].ffill().bfill()
		if curve_df["benchmark_close"].notna().any():
			first_close = curve_df["benchmark_close"].dropna().iloc[0]
			curve_df["benchmark_nav"] = curve_df["benchmark_close"] / first_close
		else:
			curve_df["benchmark_nav"] = np.nan
	else:
		curve_df["benchmark_nav"] = np.nan

	curve_df["excess_nav"] = curve_df["strategy_nav"] - curve_df["benchmark_nav"].fillna(1.0)
	strategy_total = float(curve_df["strategy_nav"].iloc[-1] - 1.0)
	day_count = max(len(curve_df) - 1, 1)
	annualized = float(curve_df["strategy_nav"].iloc[-1] ** (252 / day_count) - 1.0)
	benchmark_total = float(curve_df["benchmark_nav"].iloc[-1] - 1.0) if curve_df["benchmark_nav"].notna().any() else np.nan
	excess_total = float(strategy_total - benchmark_total) if np.isfinite(benchmark_total) else np.nan

	daily_s = curve_df["strategy_nav"].pct_change().dropna()
	profit_count = int((daily_s > 0).sum())
	loss_count = int((daily_s < 0).sum())
	win_rate = float((daily_s > 0).mean()) if len(daily_s) > 0 else np.nan

	alpha = np.nan
	beta = np.nan
	info_ratio = np.nan
	if curve_df["benchmark_nav"].notna().any():
		daily_b = curve_df["benchmark_nav"].pct_change().dropna()
		aligned = pd.concat([daily_s.rename("s"), daily_b.rename("b")], axis=1).dropna()
		if len(aligned) > 2:
			b_var = float(np.var(aligned["b"]))
			if b_var > 1e-12:
				beta = float(np.cov(aligned["s"], aligned["b"])[0, 1] / b_var)
				alpha = float((aligned["s"].mean() - beta * aligned["b"].mean()) * 252)
			excess_daily = aligned["s"] - aligned["b"]
			ex_std = float(excess_daily.std())
			if ex_std > 1e-12:
				info_ratio = float(excess_daily.mean() / ex_std * np.sqrt(252))

	metrics = {
		"strategy_return": strategy_total,
		"annualized_return": annualized,
		"benchmark_return": benchmark_total,
		"excess_return": excess_total,
		"alpha": alpha,
		"beta": beta,
		"win_rate": win_rate,
		"profit_count": float(profit_count),
		"loss_count": float(loss_count),
		"information_ratio": info_ratio,
		"sharpe": float(sharpe) if sharpe is not None else np.nan,
		"max_drawdown": float(max_drawdown) if max_drawdown is not None else np.nan,
	}

	curve_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

	plt.figure(figsize=(12, 5))
	plt.plot(curve_df["date"], curve_df["strategy_nav"], linewidth=1.8, color="#1f77b4", label="策略收益")
	if curve_df["benchmark_nav"].notna().any():
		plt.plot(curve_df["date"], curve_df["benchmark_nav"], linewidth=1.5, color="#b22222", label="沪深300")
	plt.plot(curve_df["date"], curve_df["excess_nav"], linewidth=1.2, color="#cc00ff90", alpha=0.9, label="超额收益")
	plt.title("策略净值对比")
	plt.xlabel("日期")
	plt.ylabel("净值")
	ann_text = (
		f"策略收益: {metrics['strategy_return']:.2%}  年化: {metrics['annualized_return']:.2%}\n"
		f"基准收益: {metrics['benchmark_return']:.2%}  超额: {metrics['excess_return']:.2%}\n"
		f"年化Sharpe: {metrics['sharpe']:.3f}  最大回撤: {metrics['max_drawdown']:.2f}%\n"
		f"Alpha: {metrics['alpha']:.3f}  Beta: {metrics['beta']:.3f}"
	)
	plt.gca().text(
		0.01,
		0.95,
		ann_text,
		transform=plt.gca().transAxes,
		fontsize=9,
		verticalalignment="top",
		bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="#bbbbbb"),
	)
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_png, dpi=160)
	plt.close()
	return metrics


def build_backtrader_feeds(
	feature_df: pd.DataFrame,
	start: str,
	end: str,
	max_start_delay_days: int = 5,
):
	feeds = []
	int_to_code: Dict[int, str] = {}
	skipped_late = []
	start_ts = pd.Timestamp(start)
	codes = sorted(feature_df["code"].unique())
	for i, code in enumerate(codes, start=1):
		int_to_code[i] = code
		one = feature_df[feature_df["code"] == code].copy()
		first_date = one["date"].min()
		# Backtrader multi-data mode can effectively start near the latest first-date among feeds.
		if pd.notna(first_date) and first_date > start_ts + pd.Timedelta(days=max_start_delay_days):
			skipped_late.append((code, first_date))
			continue
		one = one[(one["date"] >= pd.Timestamp(start)) & (one["date"] <= pd.Timestamp(end))]
		if one.empty:
			continue
		one = one[["date", "open", "high", "low", "close", "volume"]].copy()
		one["code_int"] = i
		one = one.sort_values("date")
		one = one.set_index("date")
		data = PandasDataExtend(dataname=one)
		feeds.append((code, data))

	if skipped_late:
		show = ", ".join([f"{c}:{d.strftime('%Y-%m-%d')}" for c, d in skipped_late[:8]])
		logger.warning(
			"为避免回测起点被晚上市标的拖后，已跳过 %d 只股票（示例 %s）",
			len(skipped_late),
			show,
		)
	return feeds, int_to_code


def _fit_fast_lgbm(x: np.ndarray, y: np.ndarray):
	"""Fast estimator used only inside coarse grid search."""
	model = lgb.LGBMClassifier(
		objective="binary",
		n_estimators=220,
		learning_rate=0.07,
		max_depth=5,
		num_leaves=31,
		subsample=0.8,
		colsample_bytree=0.7,
		min_child_samples=25,
		reg_lambda=1.0,
		reg_alpha=0.5,
		random_state=42,
		n_jobs=1,
		verbosity=-1,
	)
	model.fit(x, y)
	return model


def run_hold_horizon_grid_search(raw_df: pd.DataFrame, config: BacktestConfig) -> Dict[str, int]:
	"""Grid search over (hold_days, label_horizon) using pre-backtest history only."""
	start_bt = pd.Timestamp(config.backtest_start)
	pre_raw = raw_df[raw_df["date"] < start_bt].copy()
	if pre_raw.empty:
		logger.warning("网格搜索跳过: 回测起点前无可用历史数据")
		return {"hold_days": config.hold_days, "label_horizon": config.label_horizon}

	close_pivot = pre_raw.pivot_table(index="date", columns="code", values="close").sort_index().ffill()
	results = []
	best = {"score": -np.inf, "hold_days": config.hold_days, "label_horizon": config.label_horizon}

	for horizon in config.grid_label_horizons:
		feat = generate_advanced_features(raw_df, label_horizon=int(horizon))
		feat = feat[feat["date"] < start_bt].copy()
		if feat.empty:
			continue
		feature_cols = [c for c in feat.columns if c not in {"date", "code", "future_return", "label"}]
		cand_dates = sorted(feat["date"].drop_duplicates())
		if len(cand_dates) < 20:
			continue
		eval_dates = cand_dates[-config.grid_eval_points :]

		for hold in config.grid_hold_days:
			forward_returns = []
			ics = []
			for i, now in enumerate(eval_dates):
				if i % max(int(hold), 1) != 0:
					continue

				lookback_start = now - timedelta(days=config.retrain_lookback_days)
				train = feat[(feat["date"] >= lookback_start) & (feat["date"] < now)].copy()
				train[feature_cols] = train[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
				train = train.dropna(subset=["label"])
				if len(train) < config.min_train_samples or train["label"].nunique() < 2:
					continue

				cross = feat[feat["date"] == now].copy()
				if cross.empty:
					continue
				cross[feature_cols] = cross[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

				model = _fit_fast_lgbm(train[feature_cols].values, train["label"].values)
				probs = model.predict_proba(cross[feature_cols].values)[:, 1]
				top_codes = cross.assign(prob=probs).sort_values("prob", ascending=False).head(config.topk)["code"].tolist()
				if not top_codes:
					continue

				if cross["future_return"].notna().sum() >= 5:
					ic, _ = spearmanr(cross["future_return"], probs)
					if not np.isnan(ic):
						ics.append(float(ic))

				if now not in close_pivot.index:
					continue
				start_idx = close_pivot.index.get_loc(now)
				end_idx = min(start_idx + int(hold), len(close_pivot.index) - 1)
				end_date = close_pivot.index[end_idx]
				r = (close_pivot.loc[end_date, top_codes] / close_pivot.loc[now, top_codes] - 1.0)
				r = r.replace([np.inf, -np.inf], np.nan).dropna()
				if not r.empty:
					forward_returns.append(float(r.mean()))

			if len(forward_returns) < 5:
				continue
			avg_ret = float(np.mean(forward_returns))
			ret_std = float(np.std(forward_returns))
			avg_ic = float(np.mean(ics)) if ics else 0.0
			score = avg_ret / (ret_std + 1e-8) + 0.2 * avg_ic
			results.append(
				{
					"hold_days": int(hold),
					"label_horizon": int(horizon),
					"samples": int(len(forward_returns)),
					"avg_forward_return": avg_ret,
					"std_forward_return": ret_std,
					"avg_ic": avg_ic,
					"score": float(score),
				}
			)
			if score > best["score"]:
				best = {"score": score, "hold_days": int(hold), "label_horizon": int(horizon)}

	if results:
		res_df = pd.DataFrame(results).sort_values("score", ascending=False)
		res_path = Path(__file__).with_name("LGBM_grid_search_results.csv")
		res_df.to_csv(res_path, index=False, encoding="utf-8-sig")
		logger.info(
			"网格搜索完成: best hold_days=%d, label_horizon=%d, score=%.6f (详情见 %s)",
			best["hold_days"],
			best["label_horizon"],
			best["score"],
			res_path.name,
		)
	else:
		logger.warning("网格搜索未得到有效组合，保留原配置")

	return {"hold_days": best["hold_days"], "label_horizon": best["label_horizon"]}


def run_backtest(token: str, config: BacktestConfig):
	if token == "your_tushare_token" or not token:
		raise ValueError(
			"请配置有效的 Tushare Token。"
			"可通过环境变量 TUSHARE_TOKEN 设置，或直接修改代码中的 TUSHARE_TOKEN 常量。"
		)
	if TUSHARE_HTTP_URL == "your_tushare_http_url" or not TUSHARE_HTTP_URL:
		raise ValueError(
			"请配置有效的 Tushare HTTP URL。"
			"可通过环境变量 TUSHARE_HTTP_URL 设置，或直接修改代码中的 TUSHARE_HTTP_URL 常量。"
		)
	pro = ts.pro_api(token)
	pro._DataApi__http_url = TUSHARE_HTTP_URL

	data_start = (pd.Timestamp(config.backtest_start) - timedelta(days=365 * 2)).strftime("%Y-%m-%d")
	raw_df = get_data(
		pro=pro,
		index_code=config.index_code,
		start_date=data_start,
		end_date=config.backtest_end,
		stock_count=config.stock_count,
	)

	selected_hold = config.hold_days
	selected_horizon = config.label_horizon
	if config.enable_grid_search:
		best = run_hold_horizon_grid_search(raw_df, config)
		selected_hold = int(best["hold_days"])
		selected_horizon = int(best["label_horizon"])
		logger.info("采用网格搜索参数: hold_days=%d, label_horizon=%d", selected_hold, selected_horizon)

	feature_df = generate_advanced_features(raw_df, label_horizon=selected_horizon)

	feature_cols = [
		c
		for c in feature_df.columns
		if c not in {"date", "code", "future_return", "label"}
	]

	cerebro = bt.Cerebro(stdstats=False)
	feeds, int_to_code = build_backtrader_feeds(feature_df, config.backtest_start, config.backtest_end)
	for code, feed in feeds:
		cerebro.adddata(feed, name=code)

	cerebro.broker.setcash(1_000_000.0)
	cerebro.broker.setcommission(commission=0.0003)

	strategy_df = feature_df[
		(feature_df["date"] >= pd.Timestamp(config.backtest_start))
		& (feature_df["date"] <= pd.Timestamp(config.backtest_end))
	].copy()
	cerebro.addstrategy(
		MyStrategy,
		feature_df=strategy_df,
		feature_cols=feature_cols,
		code_map=int_to_code,
		hold_days=selected_hold,
		topk=config.topk,
		hold_buffer_size=config.hold_buffer_size,
		min_rebalance_threshold=config.min_rebalance_threshold,
		retrain_interval=config.retrain_interval,
		retrain_lookback_days=config.retrain_lookback_days,
		min_train_samples=config.min_train_samples,
	)
	cerebro.addanalyzer(
		bt.analyzers.SharpeRatio,
		_name="sharpe",
		timeframe=bt.TimeFrame.Days,
		annualize=True,
		factor=252,
	)
	cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
	cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
	cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity")

	results = cerebro.run()
	strat = results[0]
	final_value = cerebro.broker.getvalue()
	benchmark_df = get_benchmark_close(pro, config.index_code, config.backtest_start, config.backtest_end)
	logger.info("%s", "=" * 70)
	logger.info("Backtest Range: %s -> %s", config.backtest_start, config.backtest_end)
	logger.info("Final Portfolio Value: %.2f", final_value)
	logger.info("Total Return: %.4f", strat.analyzers.returns.get_analysis().get("rtot", np.nan))
	sharpe_value = strat.analyzers.sharpe.get_analysis().get("sharperatio", np.nan)
	logger.info("Sharpe(年化): %s", sharpe_value)
	dd = strat.analyzers.drawdown.get_analysis()
	logger.info("Max DrawDown(%%): %.2f", dd.max.drawdown)
	metrics = save_equity_curve_plot(
		strat.analyzers.equity.get_analysis(),
		benchmark_df,
		Path(__file__).with_name("LGBM_backtrader_demo_equity_curve.png"),
		Path(__file__).with_name("LGBM_backtrader_demo_equity_curve.csv"),
		sharpe=sharpe_value,
		max_drawdown=dd.max.drawdown,
	)
	if metrics:
		logger.info("基准收益: %.2f%%", 100 * metrics.get("benchmark_return", np.nan))
		logger.info("年化收益: %.2f%%", 100 * metrics.get("annualized_return", np.nan))
		logger.info("超额收益: %.2f%%", 100 * metrics.get("excess_return", np.nan))
		logger.info("阿尔法(alpha): %.4f", metrics.get("alpha", np.nan))
		logger.info("贝塔(beta): %.4f", metrics.get("beta", np.nan))
		logger.info("胜率: %.2f%%", 100 * metrics.get("win_rate", np.nan))
		logger.info("盈利次数: %d", int(metrics.get("profit_count", 0)))
		logger.info("亏损次数: %d", int(metrics.get("loss_count", 0)))
		logger.info("信息比率: %.4f", metrics.get("information_ratio", np.nan))
	logger.info("%s", "=" * 70)


def main():
	config = init_backtest_config()
	run_backtest(TUSHARE_TOKEN, config)

if __name__ == "__main__":
	main()