# ccxt + Binance 现货行情 Demo

这是一个面向《机器学习》课程作业的最小可运行 Demo：

- 使用 `ccxt` 连接 Binance 公开现货行情
- 拉取历史 K 线数据
- 计算基础技术指标与滚动特征
- 生成简单交易信号
- 导出 CSV 结果

另外提供一个监督学习版本：
- 用 Logistic Regression 和 Random Forest 做下一根 K 线涨跌预测
- 采用按时间切分的训练集 / 验证集 / 测试集
- 输出测试集预测结果和评估指标
- 默认优先读取本地 `data/raw` 中已保存的原始数据
- 加 `--fetch-online` 后才会联网抓取，并自动保存原始数据到本地
- 如果联网抓取失败，会自动回退到本地 `data/raw` 继续训练

## 文件说明

- `技术规划文档.md`：技术规划文档，不是开题报告
- `demo/binance_spot_demo.py`：Demo 主程序
- `requirements.txt`：依赖
- `outputs/`：运行后的输出目录

## 默认能力

- 默认交易对：`BTC/USDT`
- 默认周期：`1h`
- 默认数据源：Binance 现货公开行情
- 支持指定 `--days-back` 抓取最近一段历史数据

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
python .\demo\binance_spot_demo.py --symbol BTC/USDT --timeframe 1h --limit 500 --days-back 30
```

## 运行涨跌预测 Demo

```powershell
python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 1h --raw-dir .\data\raw
```

如果要联网抓取并自动保存到本地：

```powershell
python .\demo\binance_ml_demo.py --symbol BTC/USDT --timeframe 1h --days-back 30 --limit 500 --fetch-online
```

如果要手动指定某个原始数据文件：

```powershell
python .\demo\binance_ml_demo.py --raw-file .\outputs\raw_BTC_USDT_1h_20260506_111742.csv
```

## 输出结果

程序会在 `outputs/` 下生成：
- 原始行情 CSV
- 特征 CSV
- 信号 CSV
- 监督学习预测结果 CSV
- 模型评估指标 CSV

联网抓取时，原始数据会额外保存到 `data/raw/`，方便后续默认离线读取。

## 说明

这个 Demo 不涉及真实交易下单，仅用于课程展示、特征工程和信号生成。
监督学习版本的目标变量是“下一根 K 线是否上涨”。

