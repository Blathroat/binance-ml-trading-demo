from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from demo.binance_ml_demo import InferenceConfig, MLConfig, run_batch_training as run_batch_training_backend, run_inference
from demo.binance_spot_demo import DEFAULT_RAW_DIR, resolve_project_path


def _format_summary(result, title: str = "训练完成") -> str:
    test = result.test_metrics
    class_balance = result.class_balance
    summary = [
        f"# {title}",
        "",
        f"- 数据来源：{result.data_source}",
        f"- 原始文件：{result.raw_path}",
        f"- 特征文件：{result.feature_path}",
        f"- 模型文件：{result.model_path}",
        f"- 最优模型：{result.best_model_name}",
        f"- 模型选择：{result.config.fixed_model_name}",
        f"- 最佳概率阈值：{result.best_threshold:.3f}",
        f"- 验证集回测：{result.validation_backtest_metrics}",
        f"- 测试集回测：{result.test_backtest_metrics}",
        f"- 样本总数：{len(result.feature_df)}",
        f"- 训练集 / 验证集 / 测试集：{result.train_rows} / {result.val_rows} / {result.test_rows}",
        f"- 标签分布：{class_balance}",
        f"- 收益阈值：{result.config.target_threshold}",
        f"- 测试集 Accuracy：{test['accuracy']:.4f}",
        f"- 测试集 Precision：{test['precision']:.4f}",
        f"- 测试集 Recall：{test['recall']:.4f}",
        f"- 测试集 F1：{test['f1']:.4f}",
        f"- 测试集 ROC AUC：{test['roc_auc']:.4f}",
        "",
        f"- 预测结果文件：{result.pred_path}",
        f"- 评估指标文件：{result.metrics_path}",
        f"- 汇总文件：{result.summary_path}",
        f"- 最近输出保留条数：{result.config.keep_recent_runs}",
    ]
    return "\n".join(summary)


def _list_existing_model_choices() -> list[tuple[str, str]]:
    model_dir = ROOT_DIR / "outputs" / "models"
    if not model_dir.exists():
        return []
    model_paths = sorted(
        [path for path in model_dir.glob("*.joblib") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [(path.name, str(path)) for path in model_paths]


def _format_metrics_table(result):
    metrics_df = result.metrics_df.copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].round(4)
    return metrics_df


def _format_prediction_preview(result):
    columns = [
        "timestamp",
        "symbol",
        "exchange",
        "timeframe",
        "close",
        "future_return_horizon",
        "target_up",
        "pred_target_up",
        "pred_proba_up",
    ]
    preview = result.pred_df[columns].copy()
    return preview.head(30)


def _format_feature_preview(result):
    preview_cols = [
        "timestamp",
        "close",
        "volume",
        "ma_13",
        "ma_34",
        "macd_hist",
        "rsi_14",
        "trend_state",
    ]
    available = [col for col in preview_cols if col in result.feature_df.columns]
    return result.feature_df[available].head(30)


def _build_equity_plot(result):
    detail = result.test_backtest_detail
    fig, ax = plt.subplots(figsize=(9, 4))
    if detail is None or detail.empty or "equity_curve" not in detail.columns:
        ax.text(0.5, 0.5, "暂无回测净值数据", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.tight_layout()
        return fig

    equity = detail["equity_curve"].astype(float).reset_index(drop=True)
    ax.plot(equity.index, equity.values, color="#1f77b4", linewidth=1.8, label="总资产")
    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=1.0, label="初始资产")
    ax.set_title("仿真的总资产波动折线图")
    ax.set_xlabel("测试集样本索引")
    ax.set_ylabel("归一化总资产")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def _build_result_outputs(result, title: str = "训练完成", include_summary: bool = True):
    outputs = []
    if include_summary:
        outputs.append(_format_summary(result, title=title))
    outputs.extend(
        [
            _format_metrics_table(result),
            _format_prediction_preview(result),
            _format_feature_preview(result),
            _build_equity_plot(result),
            str(result.raw_path),
            str(result.feature_path),
            str(result.model_path),
            str(result.pred_path),
            str(result.metrics_path),
            str(result.summary_path),
        ]
    )
    return tuple(outputs)


def _format_batch_summary(results, summary_df) -> str:
    lines = [
        "# 批量训练完成",
        "",
        f"- 共训练分时：{len(results)} 个",
        f"- 排序依据：测试集总收益 / Sharpe / 年化收益",
        "",
        "## 分时结果",
    ]
    for _, row in summary_df.iterrows():
        lines.append(
            f"- {row['timeframe']} | {row['best_model']} | 总收益 {row['test_total_return']:.4f} | 年化 {row['test_annual_return']:.4f} | Sharpe {row['test_sharpe']:.2f}"
        )
    return "\n".join(lines)


def run_batch_training(
    symbol,
    timeframes,
    fetch_online,
    days_back,
    limit,
    target_horizon,
    target_threshold,
    fee_rate,
    use_gpu,
    gpu_backend,
    fixed_model_name,
    selection_metric,
    min_trades,
    keep_recent_runs,
    raw_dir,
    raw_file,
    raw_offset,
    raw_window,
    train_ratio,
    val_ratio,
):
    raw_file = raw_file.strip() if isinstance(raw_file, str) else raw_file
    raw_file = raw_file or None
    if not timeframes:
        raise ValueError("请至少选择一个分时周期")
    base_config = MLConfig(
        symbol=symbol,
        timeframe=timeframes[0],
        limit=int(limit),
        days_back=int(days_back),
        raw_dir=resolve_project_path(raw_dir),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        fetch_online=bool(fetch_online),
        raw_file=str(resolve_project_path(raw_file)) if raw_file else None,
        raw_offset=int(raw_offset),
        raw_window=int(raw_window),
        target_horizon=int(target_horizon),
        target_threshold=float(target_threshold),
        fee_rate=float(fee_rate),
        use_gpu=bool(use_gpu),
        gpu_backend=str(gpu_backend),
        fixed_model_name=str(fixed_model_name),
        selection_metric=str(selection_metric),
        min_trades=int(min_trades),
        keep_recent_runs=int(keep_recent_runs),
    )
    results, summary_df, best_result = run_batch_training_backend(base_config, list(timeframes), verbose=False)
    return (
        _format_batch_summary(results, summary_df),
        summary_df,
        *_build_result_outputs(best_result, title="训练完成", include_summary=False),
    )


def run_inference_ui(
    symbol,
    timeframe,
    fetch_online,
    days_back,
    limit,
    target_horizon,
    target_threshold,
    fee_rate,
    selected_model_path,
    raw_dir,
    raw_file,
    raw_offset,
    raw_window,
    train_ratio,
    val_ratio,
    keep_recent_runs,
):
    raw_file = raw_file.strip() if isinstance(raw_file, str) else raw_file
    raw_file = raw_file or None
    if not selected_model_path:
        raise ValueError("请选择一个已有的 joblib 模型文件")
    config = InferenceConfig(
        symbol=symbol,
        timeframe=timeframe,
        limit=int(limit),
        days_back=int(days_back),
        raw_dir=resolve_project_path(raw_dir),
        train_ratio=float(train_ratio),
        val_ratio=float(val_ratio),
        fetch_online=bool(fetch_online),
        raw_file=str(resolve_project_path(raw_file)) if raw_file else None,
        raw_offset=int(raw_offset),
        raw_window=int(raw_window),
        target_horizon=int(target_horizon),
        target_threshold=float(target_threshold),
        fee_rate=float(fee_rate),
        keep_recent_runs=int(keep_recent_runs),
    )
    result = run_inference(config, selected_model_path, verbose=False)
    return _build_result_outputs(result, title="推理完成")


def build_app():
    import gradio as gr

    with gr.Blocks(title="Binance 量化交易 Demo") as demo:
        gr.Markdown("""
# Binance 量化交易 Demo

上半部分用于 **批量训练多个分时模型**，下半部分用于 **单独推理**，并且推理使用独立的数据配置。
""")

        with gr.Tabs():
            with gr.Tab("训练"):
                with gr.Row():
                    with gr.Column(scale=1):
                        symbol = gr.Dropdown(["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"], value="BTC/USDT", label="交易对")
                        timeframes = gr.CheckboxGroup(["15m", "1h", "4h"], value=["15m", "1h", "4h"], label="批量训练分时（可多选）")
                        fetch_online = gr.Checkbox(value=False, label="联网抓取 Binance 数据并保存到本地")
                        days_back = gr.Slider(30, 365, value=180, step=1, label="回看天数")
                        limit = gr.Slider(500, 20000, value=8000, step=100, label="最多抓取 K 线数")
                        target_horizon = gr.Slider(1, 12, value=1, step=1, label="预测未来 K 线根数")
                        target_threshold = gr.Slider(0.0, 0.005, value=0.0015, step=0.0001, label="收益阈值（过滤微小波动）")
                        fee_rate = gr.Slider(0.0, 0.005, value=0.001, step=0.0001, label="单边手续费率")
                        use_gpu = gr.Checkbox(value=False, label="启用 GPU 训练选项（需安装 xgboost）")
                        gpu_backend = gr.Dropdown(["auto", "xgboost", "none"], value="auto", label="GPU 后端")
                        fixed_model_name = gr.Dropdown(
                            ["auto", "随机森林", "极端随机树", "HistGradientBoosting", "XGBoost_CPU", "XGBoost_GPU"],
                            value="auto",
                            label="训练时模型选择",
                        )
                        selection_metric = gr.Dropdown(["total_return", "annual_return", "sharpe"], value="total_return", label="训练时模型选择目标")
                        min_trades = gr.Slider(0, 50, value=5, step=1, label="验证集最少交易次数")
                        keep_recent_runs = gr.Slider(1, 50, value=20, step=1, label="最近输出保留条数")
                        raw_dir = gr.Textbox(value=str(DEFAULT_RAW_DIR), label="原始数据目录")
                        raw_file = gr.Textbox(value="", label="手动指定 raw 文件（可留空）")
                        raw_offset = gr.Number(value=0, precision=0, label="raw 起始行 offset（0 表示从头开始）")
                        raw_window = gr.Number(value=0, precision=0, label="raw 连续窗口行数（0 表示不截断）")
                        train_ratio = gr.Slider(0.5, 0.85, value=0.7, step=0.01, label="训练集比例")
                        val_ratio = gr.Slider(0.05, 0.3, value=0.15, step=0.01, label="验证集比例")
                        run_btn = gr.Button("开始批量训练", variant="primary")

                    with gr.Column(scale=2):
                        summary = gr.Markdown()
                        batch_table = gr.Dataframe(label="批量训练结果", interactive=False)
                        metrics_table = gr.Dataframe(label="最佳分时模型评估指标", interactive=False)
                        pred_preview = gr.Dataframe(label="最佳分时测试集预测预览", interactive=False)
                        feature_preview = gr.Dataframe(label="特征预览", interactive=False)
                        equity_plot = gr.Plot(label="仿真的总资产波动折线图")
                        with gr.Row():
                            raw_file_out = gr.File(label="原始数据文件")
                            feature_file_out = gr.File(label="特征文件")
                        with gr.Row():
                            model_file_out = gr.File(label="训练好的模型文件")
                            pred_file_out = gr.File(label="预测结果文件")
                            metrics_file_out = gr.File(label="评估指标文件")
                        summary_file_out = gr.File(label="汇总文件")

                run_btn.click(
                    fn=run_batch_training,
                    inputs=[
                        symbol,
                        timeframes,
                        fetch_online,
                        days_back,
                        limit,
                        target_horizon,
                        target_threshold,
                        fee_rate,
                        use_gpu,
                        gpu_backend,
                        fixed_model_name,
                        selection_metric,
                        min_trades,
                        keep_recent_runs,
                        raw_dir,
                        raw_file,
                        raw_offset,
                        raw_window,
                        train_ratio,
                        val_ratio,
                    ],
                    outputs=[
                        summary,
                        batch_table,
                        metrics_table,
                        pred_preview,
                        feature_preview,
                        equity_plot,
                        raw_file_out,
                        feature_file_out,
                        model_file_out,
                        pred_file_out,
                        metrics_file_out,
                        summary_file_out,
                    ],
                )

            with gr.Tab("推理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        existing_model_choices = _list_existing_model_choices()
                        selected_model_path = gr.Dropdown(
                            choices=existing_model_choices,
                            value=existing_model_choices[0][1] if existing_model_choices else "",
                            label="已有 joblib 模型文件",
                        )
                        symbol_inf = gr.Dropdown(["BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT"], value="BTC/USDT", label="交易对")
                        timeframe_inf = gr.Dropdown(["15m", "1h", "4h"], value="15m", label="K 线周期")
                        fetch_online_inf = gr.Checkbox(value=False, label="联网抓取 Binance 数据并保存到本地")
                        days_back_inf = gr.Slider(30, 365, value=180, step=1, label="回看天数")
                        limit_inf = gr.Slider(500, 20000, value=8000, step=100, label="最多抓取 K 线数")
                        target_horizon_inf = gr.Slider(1, 12, value=1, step=1, label="预测未来 K 线根数")
                        target_threshold_inf = gr.Slider(0.0, 0.005, value=0.0015, step=0.0001, label="收益阈值（过滤微小波动）")
                        fee_rate_inf = gr.Slider(0.0, 0.005, value=0.001, step=0.0001, label="单边手续费率")
                        raw_dir_inf = gr.Textbox(value=str(DEFAULT_RAW_DIR), label="推理原始数据目录")
                        raw_file_inf = gr.Textbox(value="", label="推理时指定 raw 文件（可留空）")
                        raw_offset_inf = gr.Number(value=0, precision=0, label="推理 raw 起始行 offset（0 表示从头开始）")
                        raw_window_inf = gr.Number(value=0, precision=0, label="推理 raw 连续窗口行数（0 表示不截断）")
                        train_ratio_inf = gr.Slider(0.5, 0.85, value=0.7, step=0.01, label="推理数据切分：训练集比例")
                        val_ratio_inf = gr.Slider(0.05, 0.3, value=0.15, step=0.01, label="推理数据切分：验证集比例")
                        keep_recent_runs_inf = gr.Slider(1, 50, value=20, step=1, label="最近输出保留条数")
                        infer_btn = gr.Button("开始推理", variant="primary")

                    with gr.Column(scale=2):
                        infer_summary = gr.Markdown()
                        infer_metrics_table = gr.Dataframe(label="推理评估指标", interactive=False)
                        infer_pred_preview = gr.Dataframe(label="推理预测预览", interactive=False)
                        infer_feature_preview = gr.Dataframe(label="推理特征预览", interactive=False)
                        infer_equity_plot = gr.Plot(label="推理回测净值曲线")
                        with gr.Row():
                            infer_raw_file_out = gr.File(label="原始数据文件")
                            infer_feature_file_out = gr.File(label="特征文件")
                        with gr.Row():
                            infer_model_file_out = gr.File(label="模型文件")
                            infer_pred_file_out = gr.File(label="预测结果文件")
                            infer_metrics_file_out = gr.File(label="评估指标文件")
                        infer_summary_file_out = gr.File(label="汇总文件")

                infer_btn.click(
                    fn=run_inference_ui,
                    inputs=[
                        symbol_inf,
                        timeframe_inf,
                        fetch_online_inf,
                        days_back_inf,
                        limit_inf,
                        target_horizon_inf,
                        target_threshold_inf,
                        fee_rate_inf,
                        selected_model_path,
                        raw_dir_inf,
                        raw_file_inf,
                        raw_offset_inf,
                        raw_window_inf,
                        train_ratio_inf,
                        val_ratio_inf,
                        keep_recent_runs_inf,
                    ],
                    outputs=[
                        infer_summary,
                        infer_metrics_table,
                        infer_pred_preview,
                        infer_feature_preview,
                        infer_equity_plot,
                        infer_raw_file_out,
                        infer_feature_file_out,
                        infer_model_file_out,
                        infer_pred_file_out,
                        infer_metrics_file_out,
                        infer_summary_file_out,
                    ],
                )

    return demo


def main():
    app = build_app()
    app.queue()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        allowed_paths=[str(ROOT_DIR / "outputs"), str(ROOT_DIR / "data")],
    )


if __name__ == "__main__":
    main()

