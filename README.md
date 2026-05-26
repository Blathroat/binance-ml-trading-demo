# ccxt + Binance 现货行情 Demo

这是一个面向《机器学习》课程作业的最小可运行 Demo：

- 使用 `ccxt` 连接 Binance 公开现货行情
- 默认以 **15 分钟 K 线** 抓取和分析数据
- 拉取历史 K 线数据
- 计算基础技术指标、动量特征和滚动特征
- 生成简单交易信号
- 导出 CSV 结果

另外提供一个监督学习版本：
- 用 Logistic Regression、Random Forest、Extra Trees、HistGradientBoosting 和软投票集成做涨跌预测
- 采用按时间切分的训练集 / 验证集 / 测试集
- 输出测试集预测结果和评估指标
- 默认优先读取本地 `data/raw` 中已保存的原始数据
- 加 `--fetch-online` 后才会联网抓取，并自动保存原始数据到本地
- 如果联网抓取失败，会自动回退到本地 `data/raw` 继续训练
- 默认预测未来 **1 根 K 线** 的涨跌方向
- 默认扩大训练窗口，优先使用更长时间范围的数据
- 默认加入收益阈值（0.15%），过滤微小波动噪声后再判断涨跌
- 可通过 `--target-horizon` 和 `--target-threshold` 继续调参，进一步平衡准确率和稳定性
- 训练完成后会自动把最佳模型保存为 `joblib` 文件，后续可直接加载复用

还提供一个基于 **Gradio** 的网页前端：
- 可在网页里直接选择交易对、K 线周期、训练参数和数据来源
- 可在网页里直接选择模型，或者保持自动多模型对比
- 可配置最近输出保留条数，训练结束后自动清理更旧的 outputs
- 训练完成后可在线查看指标、预览预测结果并下载文件

## 文件说明

- `技术规划文档.md`：技术规划文档，不是开题报告
- `demo/binance_spot_demo.py`：Demo 主程序
- `demo/binance_ml_demo.py`：监督学习涨跌预测 Demo
- `demo/gradio_app.py`：Gradio 网页前端
- `requirements.txt`：依赖
- `outputs/`：运行后的输出目录
- `outputs/models/`：训练好的模型文件目录

## 默认能力

- 默认交易对：`BTC/USDT`
- 默认周期：`15m`
- 默认数据源：Binance 现货公开行情
- 支持指定 `--days-back` 抓取最近一段历史数据
- 监督学习版默认使用更长的历史区间来扩大训练集

## 安装依赖

```powershell
pip install -r requirements.txt
```

## 运行 Demo

```powershell
python .\demo\binance_spot_demo.py
```

## 自定义参数

```powershell
python .\demo\binance_spot_demo.py --symbol BTC/USDT --timeframe 15m --limit 2000 --days-back 30
```

## 运行涨跌预测 Demo

```powershell
python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 15m --raw-dir .\data\raw
```

如果要联网抓取并自动保存到本地：

```powershell
python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 15m --days-back 180 --limit 8000 --fetch-online
```

如果要手动指定某个原始数据文件：

```powershell
python .\demo\binance_ml_demo.py --raw-file .\outputs\raw_BTC_USDT_15m_20260525_102618.csv
```

## 启动 Gradio 网页前端

```powershell
python .\demo\gradio_app.py
```

## 输出结果

程序会在 `outputs/` 下生成：
- 原始行情 CSV
- 特征 CSV
- 信号 CSV
- 监督学习特征 CSV
- 监督学习预测结果 CSV
- 模型评估指标 CSV
- 模型汇总 JSON
- 训练好的模型文件 `outputs/models/*.joblib`

联网抓取时，原始数据会额外保存到 `data/raw/`，方便后续默认离线读取。
每次训练结束后，程序会自动检查并删除超过“最近输出保留条数”的旧 outputs，减少目录堆积。

## 模型保存与加载

训练监督学习版本时，程序会自动把当前最优模型保存到 `outputs/models/`，文件格式为 `joblib`。

你可以在 Python 代码里直接加载使用：

```python
from demo.binance_ml_demo import load_trained_model

artifact = load_trained_model(r"outputs/models/model_BTC_USDT_15m_20260525_120000.joblib")
model = artifact["model"]
threshold = artifact["best_threshold"]
feature_columns = artifact["feature_columns"]
```

加载后，`model` 就是已经训练好的完整 `Pipeline`，可以直接调用 `predict` / `predict_proba`。

## 说明

这个 Demo 不涉及真实交易下单，仅用于课程展示、特征工程和信号生成。
监督学习版本的目标变量是“未来若干根 K 线是否上涨”。

