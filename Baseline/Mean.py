import json
import random
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pygrinder import mar_logistic, mnar_x, mnar_t, calc_missing_rate
from result_writer import write_result_md
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.experiment_utils import load_main_dataset, summarize_metrics  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    choices=["electricity_transformer_temperature", "italy_air_quality"],
    default="electricity_transformer_temperature",
    help="选择数据集",
)
parser.add_argument(
    "--mask_type",
    choices=["mar", "mnar_x", "mnar_t", "mix"],
    default="mar",
    help="缺失机制（随数据集限制）",
)
parser.add_argument(
    "--missing_rate",
    type=float,
    choices=[0.2, 0.3, 0.4, 0.6],
    default=0.2,
    help="缺失率 (0.2/0.3/0.4/0.6)",
)
parser.add_argument("--mar_ratio", type=float, default=0.5, help="MAR fraction in mix scenario")
parser.add_argument(
    "--prep_n_steps",
    type=int,
    default=48,
    help="ETT/IAQ 预处理窗口长度 n_steps",
)
parser.add_argument(
    "--cuda_device",
    type=int,
    default=None,
    help="指定 CUDA GPU 编号，如 0 或 1；留空则按默认可见设备",
)
argv = [a for a in sys.argv[1:] if a != "--"]
args = parser.parse_args(argv)

# 设备选择（可选）
cuda_device = getattr(args, "cuda_device", None)
if cuda_device is not None:
    if torch.cuda.is_available():
        torch.cuda.set_device(cuda_device)
    else:
        raise ValueError("指定了 --cuda_device 但当前环境无可用 GPU")


# 数据准备
DATASET_NAME = args.dataset
prep_n_steps = args.prep_n_steps
data = load_main_dataset(DATASET_NAME, prep_n_steps)
train_X_raw, val_X_raw, test_X_raw = data["train_X"], data["val_X"], data["test_X"]

print(f"Base missing rate train: {calc_missing_rate(train_X_raw):.2%}")
print(f"Base missing rate val  : {calc_missing_rate(val_X_raw):.2%}")
print(f"Base missing rate test : {calc_missing_rate(test_X_raw):.2%}")

output_dir = Path("outputs/mean")
output_dir.mkdir(parents=True, exist_ok=True)
_, n_steps, n_features = train_X_raw.shape

run_seeds = [3407, 3408, 3409]
all_metrics = []
mask_type = args.mask_type
missing_rate = args.missing_rate
mix_missing_rates = {0.2, 0.3, 0.4}
mix_mnar_map = {
    "electricity_transformer_temperature": "mnar_t",
    "italy_air_quality": "mnar_x",
}
allowed = {
    "electricity_transformer_temperature": {"mar", "mnar_t", "mix"},
    "italy_air_quality": {"mar", "mnar_x", "mix"},
}
if mask_type not in allowed[DATASET_NAME]:
    raise ValueError(f"{DATASET_NAME} 不支持缺失机制 {mask_type}，可选 {allowed[DATASET_NAME]}")
if mask_type == "mix" and missing_rate not in mix_missing_rates:
    raise ValueError(f"mix 场景仅支持缺失率 {sorted(mix_missing_rates)}")

def build_filled_array(ts_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回原始缺失掩码与按特征均值填充后的数组。"""
    orig_nan = np.isnan(ts_array)
    filled = ts_array.copy()
    feat_mean = np.nanmean(filled, axis=(0, 1))
    feat_mean = np.nan_to_num(feat_mean, nan=0.0)
    idx = np.where(orig_nan)
    filled[idx] = np.take(feat_mean, idx[2])
    return orig_nan, filled

def unwrap(res):
    return res[0] if isinstance(res, tuple) else res

def build_candidate_mask(ts_array: np.ndarray, mechanism_type: str, mechanism_rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """生成候选新增缺失位置，再由后续步骤精确控制新增缺失数量。"""
    rng = np.random.default_rng(seed)
    orig_nan, filled = build_filled_array(ts_array)

    if mechanism_type == "mar":
        flat = filled.reshape(-1, filled.shape[2])
        masked_flat = unwrap(mar_logistic(flat, obs_rate=0.1, missing_rate=mechanism_rate))
        masked_full = masked_flat.reshape(ts_array.shape)
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_x":
        masked_full = unwrap(mnar_x(filled, offset=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_t":
        masked_full = unwrap(mnar_t(filled, cycle=20, pos=10, scale=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled

    raise ValueError(f"未知缺失机制 {mechanism_type}")

def choose_exact_missing_positions(
    obs_mask: np.ndarray,
    candidate_mask: np.ndarray,
    target_extra: int,
    seed: int,
    supplement_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """从候选位置中优先采样，并补齐到目标新增缺失数量。"""
    rng = np.random.default_rng(seed)
    obs_flat_idx = np.where(obs_mask.reshape(-1))[0]
    if target_extra <= 0 or len(obs_flat_idx) == 0:
        return np.zeros_like(obs_mask, dtype=bool)

    target_extra = min(target_extra, len(obs_flat_idx))
    selected_flat = np.zeros(obs_mask.size, dtype=bool)
    candidate_flat_idx = np.where(candidate_mask.reshape(-1))[0]

    if len(candidate_flat_idx) >= target_extra:
        chosen = rng.choice(candidate_flat_idx, size=target_extra, replace=False)
        selected_flat[chosen] = True
        return selected_flat.reshape(obs_mask.shape)

    if len(candidate_flat_idx) > 0:
        selected_flat[candidate_flat_idx] = True

    need = target_extra - int(selected_flat.sum())
    if need <= 0:
        return selected_flat.reshape(obs_mask.shape)

    remaining_flat_idx = obs_flat_idx[~np.isin(obs_flat_idx, candidate_flat_idx)]
    if len(remaining_flat_idx) == 0:
        return selected_flat.reshape(obs_mask.shape)

    if supplement_weights is not None:
        flat_weights = supplement_weights.reshape(-1)[remaining_flat_idx].astype(float)
        flat_weights = np.clip(flat_weights, a_min=0.0, a_max=None)
        if float(flat_weights.sum()) > 0:
            probs = flat_weights / flat_weights.sum()
            extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False, p=probs)
        else:
            extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    else:
        extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    selected_flat[extra] = True
    return selected_flat.reshape(obs_mask.shape)

def apply_single_mechanism(ts_array: np.ndarray, mechanism_type: str, target_extra: int, seed: int) -> np.ndarray:
    """在当前可观测位置上精确追加 target_extra 个缺失值。"""
    orig_nan = np.isnan(ts_array)
    obs_mask = ~orig_nan
    if target_extra <= 0 or not np.any(obs_mask):
        return ts_array.copy()

    obs_count = int(obs_mask.sum())
    mechanism_rate = target_extra / max(obs_count, 1)
    candidate_mask, filled = build_candidate_mask(ts_array, mechanism_type, mechanism_rate, seed)
    supplement_weights = np.abs(filled) + 1e-8 if mechanism_type == "mnar_x" else None
    selected_mask = choose_exact_missing_positions(obs_mask, candidate_mask, target_extra, seed, supplement_weights)
    result = ts_array.copy()
    result[selected_mask] = np.nan
    return result

def apply_mask_with_mechanism(ts_array: np.ndarray, mask_type: str, missing_rate: float, seed: int, *, mar_ratio: float = 0.5) -> np.ndarray:
    """仅在可观测位置追加缺失，原始缺失保持不变。"""
    obs_count = int((~np.isnan(ts_array)).sum())
    total_target = int(obs_count * missing_rate)

    if mask_type == "mix":
        mnar_type = mix_mnar_map[DATASET_NAME]
        mar_target = int(round(total_target * mar_ratio))
        mnar_target = total_target - mar_target
        mixed = apply_single_mechanism(ts_array, mnar_type, mnar_target, seed)
        mixed = apply_single_mechanism(mixed, "mar", mar_target, seed + 10_000)
        return mixed

    return apply_single_mechanism(ts_array, mask_type, total_target, seed)

def aggregate_metric(metric_name: str) -> dict:
    values = [m[metric_name] for m in all_metrics]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "std": std,
        "display": f"{mean:.6f} ± {std:.6f}",
    }

for seed in run_seeds:
    print(f"\n===== Run with seed {seed} =====")
    set_seed(seed)

    train_X = apply_mask_with_mechanism(train_X_raw, mask_type, missing_rate, seed, mar_ratio=args.mar_ratio)
    val_X = apply_mask_with_mechanism(val_X_raw, mask_type, missing_rate, seed, mar_ratio=args.mar_ratio)
    test_X = apply_mask_with_mechanism(test_X_raw, mask_type, missing_rate, seed, mar_ratio=args.mar_ratio)

    print(f"After {mask_type} masking train: {calc_missing_rate(train_X):.2%}")
    print(f"After {mask_type} masking val  : {calc_missing_rate(val_X):.2%}")
    print(f"After {mask_type} masking test : {calc_missing_rate(test_X):.2%}")

    # 均值填充：基于训练集按特征求均值
    feature_means = np.nanmean(train_X.reshape(-1, n_features), axis=0)
    imputation = np.copy(test_X)
    for i in range(n_features):
        imputation[..., i] = np.nan_to_num(imputation[..., i], nan=feature_means[i])

    test_X_ori = data["test_X_ori"]
    indicating_mask = np.isnan(test_X) ^ np.isnan(test_X_ori)

    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    metric_values = summarize_metrics(imputation, test_X_ori, indicating_mask)
    metric_values["seed"] = seed
    all_metrics.append(metric_values)

    print(f"MRE on masked ground truth (seed {seed}): {metric_values['mre']:.6f}")
    print(f"NRMSE on masked ground truth (seed {seed}): {metric_values['nrmse']:.6f}")
    print(f"MAE on masked ground truth (seed {seed}): {metric_values['mae']:.6f}")
    print(f"RMSE on masked ground truth (seed {seed}): {metric_values['rmse']:.6f}")
    print(f"Evaluated points (seed {seed}): {metric_values['n_points']}")

    np.save(seed_dir / "imputation.npy", imputation)
    (seed_dir / "metrics.json").write_text(json.dumps(metric_values, ensure_ascii=False, indent=2))

agg_mae = aggregate_metric("mae")
agg_rmse = aggregate_metric("rmse")
agg_mre = aggregate_metric("mre")
agg_nrmse = aggregate_metric("nrmse")
agg_n_points = aggregate_metric("n_points")

print("\n===== Aggregated over seeds =====")
print(f"MAE : {agg_mae['display']}")
print(f"RMSE: {agg_rmse['display']}")
print(f"MRE : {agg_mre['display']}")
print(f"NRMSE: {agg_nrmse['display']}")
print(f"Evaluated points: {agg_n_points['display']}")

agg = {
    "dataset": DATASET_NAME,
    "mask_type": mask_type,
    "missing_rate": missing_rate,
    "seeds": run_seeds,
    "mae": agg_mae,
    "rmse": agg_rmse,
    "mre": agg_mre,
    "nrmse": agg_nrmse,
    "n_points": agg_n_points,
    "runs": all_metrics,
}
(output_dir / "metrics_avg.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
write_result_md("Mean", agg)