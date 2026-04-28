"""公共工具：将三次种子实验结果自动追加写入 result.md"""

from datetime import datetime
from pathlib import Path


def write_result_md(
    model_name: str,
    agg: dict,
    result_path: str | Path | None = None,
):
    """将单个模型的三次种子实验结果追加写入 result.md。

    Parameters
    ----------
    model_name : str
        模型名称，如 "BRITS"、"SAITS" 等。
    agg : dict
        聚合结果字典，需包含以下键：
        - dataset, mask_type, missing_rate, seeds
        - mae / rmse / mre / mape / n_points（各含 mean, std, display）
        - runs: list[dict]，每个元素含 seed, mae, rmse, mre, mape, n_points
    result_path : str | Path | None
        结果文件路径，默认为当前脚本所在目录下的 result.md。
    """
    if result_path is None:
        result_path = Path(__file__).resolve().parent / "result.md"
    md_path = Path(result_path)

    need_header = not md_path.exists() or md_path.stat().st_size == 0

    dataset = agg["dataset"]
    mask_type = agg["mask_type"]
    missing_rate = agg["missing_rate"]
    runs = agg["runs"]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: list[str] = []

    if need_header:
        lines.append("# Experiment Results\n")

    lines.append(
        f"## {model_name} | {dataset} | {mask_type} | Missing Rate: {missing_rate}\n"
    )
    lines.append(f"> Recorded at: {timestamp}\n")
    has_robust = "mape_capped" in runs[0]

    if has_robust:
        lines.append("| Seed | MAE | RMSE | MRE | MAPE | MAPE_cap | MAPE_trim | sMAPE | Outlier% | Points |")
        lines.append("|:----:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:------:|")
    else:
        lines.append("| Seed | MAE | RMSE | MRE | MAPE | Eval Points |")
        lines.append("|:----:|:-------:|:-------:|:-------:|:-------:|:-----------:|")

    for run in runs:
        if has_robust:
            lines.append(
                f"| {run['seed']} "
                f"| {run['mae']:.6f} "
                f"| {run['rmse']:.6f} "
                f"| {run['mre']:.6f} "
                f"| {run['mape']:.6f} "
                f"| {run['mape_capped']:.6f} "
                f"| {run['mape_trimmed']:.6f} "
                f"| {run['smape']:.6f} "
                f"| {run['mape_outlier_ratio']:.4f} "
                f"| {int(run['n_points'])} |"
            )
        else:
            lines.append(
                f"| {run['seed']} "
                f"| {run['mae']:.6f} "
                f"| {run['rmse']:.6f} "
                f"| {run['mre']:.6f} "
                f"| {run['mape']:.6f} "
                f"| {int(run['n_points'])} |"
            )

    a_mae = agg["mae"]
    a_rmse = agg["rmse"]
    a_mre = agg["mre"]
    a_mape = agg["mape"]
    a_np = agg["n_points"]

    if has_robust:
        a_mc = agg["mape_capped"]
        a_mt = agg["mape_trimmed"]
        a_sm = agg["smape"]
        a_or = agg["mape_outlier_ratio"]
        lines.append(
            f"| **Mean±Std** "
            f"| **{a_mae['mean']:.6f}±{a_mae['std']:.6f}** "
            f"| **{a_rmse['mean']:.6f}±{a_rmse['std']:.6f}** "
            f"| **{a_mre['mean']:.6f}±{a_mre['std']:.6f}** "
            f"| **{a_mape['mean']:.6f}±{a_mape['std']:.6f}** "
            f"| **{a_mc['mean']:.6f}±{a_mc['std']:.6f}** "
            f"| **{a_mt['mean']:.6f}±{a_mt['std']:.6f}** "
            f"| **{a_sm['mean']:.6f}±{a_sm['std']:.6f}** "
            f"| **{a_or['mean']:.4f}±{a_or['std']:.4f}** "
            f"| **{a_np['mean']:.0f}±{a_np['std']:.0f}** |"
        )
    else:
        lines.append(
            f"| **Mean±Std** "
            f"| **{a_mae['mean']:.6f}±{a_mae['std']:.6f}** "
            f"| **{a_rmse['mean']:.6f}±{a_rmse['std']:.6f}** "
            f"| **{a_mre['mean']:.6f}±{a_mre['std']:.6f}** "
            f"| **{a_mape['mean']:.6f}±{a_mape['std']:.6f}** "
            f"| **{a_np['mean']:.0f}±{a_np['std']:.0f}** |"
        )

    lines.append("\n---\n")

    with open(md_path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n[Result] {model_name} 的结果已追加写入 {md_path.resolve()}")
