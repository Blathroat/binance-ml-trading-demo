from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import ccxt
import numpy as np
import pandas as pd


@dataclass
class DemoConfig:
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    limit: int = 500
    days_back: int = 30
    exchange_id: str = "binance"
    output_dir: Path = Path("outputs")
    raw_dir: Path = Path("data/raw")


RAW_REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(exchange_id: str = "binance"):
    if not hasattr(ccxt, exchange_id):
        raise ValueError(f"Unsupported exchange: {exchange_id}")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    return exchange


def fetch_ohlcv_data(exchange, symbol: str, timeframe: str, days_back: int, limit: int) -> pd.DataFrame:
    if days_back <= 0:
        raise ValueError("days_back 必须大于 0")
    if limit <= 0:
        raise ValueError("limit 必须大于 0")

    since_dt = datetime.now(timezone.utc) - pd.Timedelta(days=days_back)
    since_ms = int(since_dt.timestamp() * 1000)
    all_rows: list[list[float]] = []
    batch_size = min(500, limit)

    while len(all_rows) < limit:
        batch = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=min(batch_size, limit - len(all_rows)),
        )
        if not batch:
            break

        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(datetime.now(timezone.utc).timestamp() * 1000):
            break
        since_ms = last_ts + 1

        if len(batch) < min(batch_size, limit - len(all_rows) + len(batch)):
            break

    df = pd.DataFrame(
        all_rows,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    if df.empty:
        raise RuntimeError("未获取到任何 OHLCV 数据，请检查交易对、周期或网络状态")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    df["symbol"] = symbol
    df["exchange"] = exchange.id
    df["timeframe"] = timeframe
    return df


def normalize_symbol_token(symbol: str) -> str:
    return symbol.replace("/", "_").replace("-", "_")


def resolve_latest_raw_file(
    symbol: str,
    timeframe: str,
    search_dirs: Iterable[str | Path] = (Path("data/raw"), Path("outputs")),
    raw_file: str | Path | None = None,
) -> Path:
    if raw_file is not None:
        candidate = Path(raw_file)
        if not candidate.exists():
            raise FileNotFoundError(f"指定的原始数据文件不存在：{candidate}")
        return candidate

    token = normalize_symbol_token(symbol)
    pattern = f"raw_{token}_{timeframe}_*.csv"
    candidates: list[Path] = []
    for base in search_dirs:
        base_path = Path(base)
        if base_path.exists():
            candidates.extend(sorted(base_path.glob(pattern)))

    if not candidates:
        raise FileNotFoundError(
            f"未找到可用的原始数据文件，期望模式类似：{pattern}。请先运行联网抓取，或使用 --fetch-online。"
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_local_raw_data(
    raw_path: str | Path,
    symbol: str | None = None,
    timeframe: str | None = None,
    exchange_id: str | None = None,
) -> pd.DataFrame:
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"原始数据文件不存在：{path}")

    df = pd.read_csv(path)
    missing = [col for col in RAW_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"原始数据缺少必要列：{missing}，请检查 CSV 是否为 K 线数据")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"]).drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if "symbol" not in df.columns:
        if symbol is None:
            raise ValueError("原始数据缺少 symbol 列，且未提供 symbol 参数")
        df["symbol"] = symbol
    if "exchange" not in df.columns:
        if exchange_id is None:
            raise ValueError("原始数据缺少 exchange 列，且未提供 exchange_id 参数")
        df["exchange"] = exchange_id
    if "timeframe" not in df.columns:
        if timeframe is None:
            raise ValueError("原始数据缺少 timeframe 列，且未提供 timeframe 参数")
        df["timeframe"] = timeframe

    return df


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values("timestamp").reset_index(drop=True)

    out["return_1"] = out["close"].pct_change()
    ratio = out["close"] / out["close"].shift(1)
    out["log_return_1"] = np.log(ratio.where(ratio > 0))
    out["range_pct"] = (out["high"] - out["low"]) / out["close"]
    out["ma_7"] = out["close"].rolling(7).mean()
    out["ma_21"] = out["close"].rolling(21).mean()
    out["volatility_7"] = out["return_1"].rolling(7).std()
    out["volume_ma_7"] = out["volume"].rolling(7).mean()
    out["rsi_14"] = compute_rsi(out["close"], window=14)
    out["close_vs_ma7"] = out["close"] / out["ma_7"] - 1
    out["close_vs_ma21"] = out["close"] / out["ma_21"] - 1

    out["day_of_week"] = out["timestamp"].dt.day_name()
    out["hour"] = out["timestamp"].dt.hour
    out["trend_state"] = "neutral"
    out.loc[out["ma_7"] > out["ma_21"], "trend_state"] = "bull"
    out.loc[out["ma_7"] < out["ma_21"], "trend_state"] = "bear"

    return out.dropna().reset_index(drop=True)


def generate_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signal"] = "hold"

    buy_mask = (out["ma_7"] > out["ma_21"]) & (out["rsi_14"] < 70)
    sell_mask = (out["ma_7"] < out["ma_21"]) & (out["rsi_14"] > 30)

    out.loc[buy_mask, "signal"] = "buy"
    out.loc[sell_mask, "signal"] = "sell"
    return out


def summarize(df: pd.DataFrame) -> str:
    latest = df.iloc[-1]
    summary = [
        f"symbol={latest['symbol']}",
        f"exchange={latest['exchange']}",
        f"timeframe={latest['timeframe']}",
        f"latest_timestamp={latest['timestamp']}",
        f"latest_close={latest['close']:.4f}",
        f"latest_signal={latest['signal']}",
        f"latest_rsi={latest['rsi_14']:.2f}",
        f"rows={len(df)}",
    ]
    return " | ".join(summary)


def save_outputs(raw_df: pd.DataFrame, feature_df: pd.DataFrame, config: DemoConfig) -> tuple[Path, Path, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.raw_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    raw_path = config.output_dir / f"raw_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    raw_local_path = config.raw_dir / f"raw_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    feat_path = config.output_dir / f"features_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"
    signal_path = config.output_dir / f"signals_{config.symbol.replace('/', '_')}_{config.timeframe}_{stamp}.csv"

    raw_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    raw_df.to_csv(raw_local_path, index=False, encoding="utf-8-sig")
    feature_df.to_csv(feat_path, index=False, encoding="utf-8-sig")
    feature_df[["timestamp", "symbol", "exchange", "timeframe", "close", "rsi_14", "signal", "trend_state"]].to_csv(
        signal_path, index=False, encoding="utf-8-sig"
    )
    return raw_local_path, feat_path, signal_path


def run_demo(config: DemoConfig) -> None:
    exchange = make_exchange(config.exchange_id)
    raw_df = fetch_ohlcv_data(exchange, config.symbol, config.timeframe, config.days_back, config.limit)

    feature_df = add_features(raw_df)
    feature_df = generate_signal(feature_df)
    raw_path, feat_path, signal_path = save_outputs(raw_df, feature_df, config)

    print(f"数据源：{exchange.id}")
    print(f"交易对：{config.symbol} | 周期：{config.timeframe} | 回看天数：{config.days_back} | 目标K线数：{config.limit}")
    print(f"抓取完成：{len(raw_df)} 条 OHLCV，已计算特征并生成信号")
    print(f"最新状态：{summarize(feature_df)}")
    print(f"原始数据已保存到本地：{raw_path}")
    print(f"特征数据已保存：{feat_path}")
    print(f"信号数据已保存：{signal_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于 ccxt 接入 Binance 现货行情的 Demo")
    parser.add_argument("--symbol", default="BTC/USDT", help="交易对，例如 BTC/USDT")
    parser.add_argument("--timeframe", default="1h", help="K线周期，例如 1h、15m、4h")
    parser.add_argument("--limit", type=int, default=500, help="最多抓取的 K 线条数")
    parser.add_argument("--days-back", type=int, default=30, help="回看最近多少天的数据")
    parser.add_argument("--exchange", default="binance", help="ccxt 支持的交易所 id")
    parser.add_argument("--output-dir", default="outputs", help="结果保存目录")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    config = DemoConfig(
        symbol=args.symbol,
        timeframe=args.timeframe,
        limit=args.limit,
        days_back=args.days_back,
        exchange_id=args.exchange,
        output_dir=Path(args.output_dir),
    )
    run_demo(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

