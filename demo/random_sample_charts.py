from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.binance_ml_demo import (
    CATEGORICAL_FEATURES,
    InferenceConfig,
    DEFAULT_OUTPUT_DIR,
    NUMERIC_FEATURES,
    build_dataset,
    find_best_threshold,
    load_trained_model,
    resolve_model_feature_columns,
    run_inference,
    simulate_long_only_backtest,
    time_split,
)
from demo.binance_spot_demo import ROOT_DIR, resolve_project_path


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


@dataclass(frozen=True)
class SelectedModel:
    label: str
    summary_path: Path


SELECTED_MODELS = [
    SelectedModel("15m", DEFAULT_OUTPUT_DIR / "ml_summary_BTC_USDT_15m_20260603_130217.json"),
    SelectedModel("1h", DEFAULT_OUTPUT_DIR / "ml_summary_BTC_USDT_1h_20260603_130145.json"),
    SelectedModel("4h", DEFAULT_OUTPUT_DIR / "ml_summary_BTC_USDT_4h_20260603_125733.json"),
]


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def to_repo_relative(path: Path) -> str:
    """把路径转成相对仓库根的 POSIX 路径，避免把本机绝对路径写入结果文件。"""
    try:
        return path.resolve().relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(path)


def raw_row_count(raw_path: Path) -> int:
    return len(pd.read_csv(raw_path, usecols=["timestamp"]))


def sample_offsets(total_rows: int, base_window: int, samples: int, seed: int) -> list[int]:
    max_offset = max(0, total_rows - base_window)
    if samples <= 1 or max_offset == 0:
        return [0]
    series = pd.Series(range(max_offset + 1)).sample(n=min(samples, max_offset + 1), random_state=seed)
    return sorted(int(value) for value in series.tolist())


def sample_window_size(total_rows: int, base_window: int, samples: int) -> int:
    if samples <= 1:
        return min(base_window, total_rows)
    return min(base_window, max(220, total_rows - samples + 1))


def make_inference_config(summary: dict, raw_offset: int, raw_window: int, output_dir: Path) -> InferenceConfig:
    return InferenceConfig(
        symbol=str(summary["symbol"]),
        timeframe=str(summary["timeframe"]),
        output_dir=output_dir,
        raw_file=str(resolve_project_path(summary["raw_file"])),
        raw_offset=raw_offset,
        raw_window=raw_window,
        target_horizon=int(summary["target_horizon"]),
        target_threshold=float(summary["target_threshold"]),
        keep_recent_runs=0,
    )


def save_equity_plot(result, label: str, sample_index: int, output_dir: Path) -> Path:
    detail = result["detail"]
    equity = detail["equity_curve"].astype(float).reset_index(drop=True)
    path = output_dir / f"{label}_equity_sample_{sample_index}.png"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(equity.index, equity.values, linewidth=1.8, label="equity")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0)
    ax.set_title(f"{label} sample {sample_index} equity curve")
    ax.set_xlabel("test sample index")
    ax.set_ylabel("normalized equity")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def evaluate_sample(summary: dict, raw_offset: int, raw_window: int, output_dir: Path) -> dict:
    config = make_inference_config(summary, raw_offset, raw_window, output_dir)
    df, _, _ = build_dataset(config)
    _, val_df, test_df = time_split(df, config.train_ratio, config.val_ratio)
    artifact = load_trained_model(summary["model_path"])
    model = artifact["model"]
    feature_columns = resolve_model_feature_columns(artifact, df)
    val_prob = model.predict_proba(val_df[feature_columns])[:, 1]
    threshold, _, _ = find_best_threshold(
        val_df["future_return_horizon"],
        val_prob,
        config.fee_rate,
        config.timeframe,
        min_trades=2,
        selection_metric="total_return",
    )
    test_prob = model.predict_proba(test_df[feature_columns])[:, 1]
    metrics, detail = simulate_long_only_backtest(
        test_df["future_return_horizon"],
        test_prob,
        threshold,
        config.fee_rate,
        config.timeframe,
    )
    return {"metrics": metrics, "detail": detail, "threshold": threshold}


def save_summary_charts(metrics: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    return_boxplot = output_dir / "random_sample_return_boxplot.png"
    sharpe_bar = output_dir / "random_sample_sharpe_bar.png"

    fig, ax = plt.subplots(figsize=(8, 4))
    metrics.boxplot(column="test_total_return", by="timeframe", ax=ax)
    ax.set_title("Random sample test total return")
    ax.set_xlabel("timeframe")
    ax.set_ylabel("total return")
    fig.suptitle("")
    fig.tight_layout()
    fig.savefig(return_boxplot, dpi=150)
    plt.close(fig)

    sharpe = metrics.groupby("timeframe", as_index=False)["test_sharpe"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(sharpe["timeframe"], sharpe["test_sharpe"], color="#1f77b4")
    ax.set_title("Average Sharpe by timeframe")
    ax.set_xlabel("timeframe")
    ax.set_ylabel("average Sharpe")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(sharpe_bar, dpi=150)
    plt.close(fig)

    return return_boxplot, sharpe_bar


def generate(samples: int, seed: int, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for selected in SELECTED_MODELS:
        summary = load_summary(selected.summary_path)
        raw_path = resolve_project_path(summary["raw_file"])
        total_rows = raw_row_count(raw_path)
        base_window = sample_window_size(total_rows, int(summary["raw_window"] or total_rows), samples)
        for sample_index, offset in enumerate(sample_offsets(total_rows, base_window, samples, seed), start=1):
            result = evaluate_sample(summary, offset, base_window, output_dir)
            chart_path = save_equity_plot(result, selected.label, sample_index, output_dir)
            metrics = result["metrics"]
            rows.append(
                {
                    "timeframe": selected.label,
                    "sample_index": sample_index,
                    "raw_file": to_repo_relative(raw_path),
                    "raw_offset": offset,
                    "raw_window": base_window,
                    "model_path": summary["model_path"],
                    "target_horizon": summary["target_horizon"],
                    "best_threshold": result["threshold"],
                    "test_total_return": metrics["total_return"],
                    "test_sharpe": metrics["sharpe"],
                    "test_max_drawdown": metrics["max_drawdown"],
                    "test_win_rate": metrics["win_rate"],
                    "test_trade_count": metrics["trade_count"],
                    "test_exposure": metrics["exposure"],
                    "equity_chart": to_repo_relative(chart_path),
                }
            )

    metrics = pd.DataFrame(rows)
    metrics_path = output_dir / "random_sample_metrics.csv"
    metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    save_summary_charts(metrics, output_dir)
    return metrics_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="为选定模型生成随机连续样本回测图表")
    parser.add_argument("--samples", type=int, default=5, help="每个模型抽取多少个随机连续窗口")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "random_sample_charts"), help="图表输出目录")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.samples <= 0:
        raise ValueError("samples 必须大于 0")
    metrics_path = generate(args.samples, args.seed, resolve_project_path(args.output_dir))
    print(f"随机样本指标已保存：{metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
