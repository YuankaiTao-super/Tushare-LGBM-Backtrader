# 策略简介

多因子机器学习选股回测学习项目：

- 用 Tushare 拉取指数成分股和日线行情
- 构建技术指标特征并生成未来收益标签
- 用 LightGBM 训练并预测股票上涨概率
- 在 Backtrader 中按周期调仓并输出绩效

## 基本流程

1. 数据准备：指数成分筛选 + 流动性筛选 + 行情拉取
2. 特征与标签：生成技术指标，按截面未来收益打标签
3. 模型训练：滚动窗口训练，定期重训
4. 策略调仓：买入预测概率 TopK，等权持仓
5. 结果评估：输出收益、回撤、Sharpe 与净值曲线

## 快速运行

1. 在代码中填写 Tushare 配置：
	- TUSHARE_TOKEN
	- TUSHARE_HTTP_URL
2. 安装依赖：

```bash
pip install pandas numpy tushare lightgbm backtrader scikit-learn scipy matplotlib
```

3. 执行回测：

```bash
python LGBM_backtrader_demo.py
```

