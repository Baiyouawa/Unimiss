from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import tsdb
from benchpots.datasets import preprocess_ett, preprocess_italy_air_quality
from pygrinder import calc_missing_rate, mar_logistic, mnar_t, mnar_x


RUN_SEEDS = [3407, 3408, 3409]

ETT = "electricity_transformer_temperature"
IAQ = "italy_air_quality"

MISSING_LABEL_NONE = 0
MISSING_LABEL_MAR = 1
MISSING_LABEL_MNAR = 2


@dataclass
class AggregatedMetric:
    mean: float
    std: float
    display: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "Data"


def ensure_cache_dir() -> Path:
    cache_dir = DATA_ROOT
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        import tsdb.data_processing as tsdb_data_processing

        tsdb_data_processing.CACHED_DATASET_DIR = str(cache_dir.resolve())
    except Exception:
        pass
    return cache_dir.resolve()


def ensure_tsdb_cache(dataset_name: str) -> Path:
    cache_dir = ensure_cache_dir()
    dataset_dir = cache_dir / dataset_name
    if not dataset_dir.exists():
        legacy_cache = Path.home() / ".pypots" / "tsdb"
        if legacy_cache.exists():
            try:
                tsdb.migrate_cache(str(cache_dir))
            except FileExistsError:
                pass
    return dataset_dir


def load_main_dataset(dataset_name: str, prep_n_steps: int) -> dict:
    ensure_tsdb_cache(dataset_name)
    if dataset_name == ETT:
        return preprocess_ett(subset="ETTm2", rate=0.01, n_steps=prep_n_steps, pattern="point")
    if dataset_name == IAQ:
        return preprocess_italy_air_quality(rate=0.01, n_steps=prep_n_steps, pattern="point")
    raise ValueError(f"Unsupported dataset for current CIKM scope: {dataset_name}")


def supported_masks(dataset_name: str) -> set[str]:
    if dataset_name == ETT:
        return {"mar", "mnar_t", "mix"}
    if dataset_name == IAQ:
        return {"mar", "mnar_x", "mix"}
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def validate_main_scope(dataset_name: str, mask_type: str, missing_rate: float) -> None:
    if mask_type not in supported_masks(dataset_name):
        raise ValueError(f"{dataset_name} does not support mask_type={mask_type}")
    if missing_rate not in {0.2, 0.3, 0.4}:
        raise ValueError("Current CIKM scope only supports missing rates 0.2, 0.3, 0.4")


def build_filled_array(ts_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    orig_nan = np.isnan(ts_array)
    filled = ts_array.copy()
    feat_mean = np.nanmean(filled, axis=(0, 1))
    feat_mean = np.nan_to_num(feat_mean, nan=0.0)
    idx = np.where(orig_nan)
    filled[idx] = np.take(feat_mean, idx[2])
    return orig_nan, filled


def unwrap_masking_result(result):
    return result[0] if isinstance(result, tuple) else result


def build_candidate_mask(
    ts_array: np.ndarray,
    mechanism_type: str,
    mechanism_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    orig_nan, filled = build_filled_array(ts_array)
    if mechanism_type == "mar":
        flat = filled.reshape(-1, filled.shape[2])
        masked_flat = unwrap_masking_result(mar_logistic(flat, obs_rate=0.1, missing_rate=mechanism_rate))
        masked_full = masked_flat.reshape(ts_array.shape)
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_x":
        masked_full = unwrap_masking_result(mnar_x(filled, offset=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    if mechanism_type == "mnar_t":
        masked_full = unwrap_masking_result(mnar_t(filled, cycle=20, pos=10, scale=mechanism_rate))
        candidate_mask = np.isnan(masked_full) & (~orig_nan)
        return candidate_mask, filled
    raise ValueError(f"Unknown mechanism_type: {mechanism_type}")


def choose_exact_missing_positions(
    obs_mask: np.ndarray,
    candidate_mask: np.ndarray,
    target_extra: int,
    seed: int,
    supplement_weights: Optional[np.ndarray] = None,
) -> np.ndarray:
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
            extra = rng.choice(
                remaining_flat_idx,
                size=min(need, len(remaining_flat_idx)),
                replace=False,
                p=probs,
            )
        else:
            extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    else:
        extra = rng.choice(remaining_flat_idx, size=min(need, len(remaining_flat_idx)), replace=False)
    selected_flat[extra] = True
    return selected_flat.reshape(obs_mask.shape)


def _make_labels(mask: np.ndarray, label_value: int) -> np.ndarray:
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[mask] = label_value
    return labels


def apply_single_mechanism_with_labels(
    ts_array: np.ndarray,
    mechanism_type: str,
    target_extra: int,
    seed: int,
    label_value: int,
) -> tuple[np.ndarray, np.ndarray]:
    orig_nan = np.isnan(ts_array)
    obs_mask = ~orig_nan
    if target_extra <= 0 or not np.any(obs_mask):
        return ts_array.copy(), np.zeros_like(obs_mask, dtype=np.int64)

    obs_count = int(obs_mask.sum())
    mechanism_rate = target_extra / max(obs_count, 1)
    candidate_mask, filled = build_candidate_mask(ts_array, mechanism_type, mechanism_rate)
    supplement_weights = np.abs(filled) + 1e-8 if mechanism_type == "mnar_x" else None
    selected_mask = choose_exact_missing_positions(obs_mask, candidate_mask, target_extra, seed, supplement_weights)
    result = ts_array.copy()
    result[selected_mask] = np.nan
    return result, _make_labels(selected_mask, label_value)


def apply_mask_with_labels(
    ts_array: np.ndarray,
    dataset_name: str,
    mask_type: str,
    missing_rate: float,
    seed: int,
    *,
    mar_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    obs_count = int((~np.isnan(ts_array)).sum())
    total_target = int(obs_count * missing_rate)

    if mask_type == "mar":
        return apply_single_mechanism_with_labels(
            ts_array, "mar", total_target, seed, MISSING_LABEL_MAR
        )
    if mask_type == "mnar_t":
        return apply_single_mechanism_with_labels(
            ts_array, "mnar_t", total_target, seed, MISSING_LABEL_MNAR
        )
    if mask_type == "mnar_x":
        return apply_single_mechanism_with_labels(
            ts_array, "mnar_x", total_target, seed, MISSING_LABEL_MNAR
        )
    if mask_type == "mix":
        mnar_type = "mnar_t" if dataset_name == ETT else "mnar_x"
        mar_target = int(round(total_target * mar_ratio))
        mnar_target = total_target - mar_target
        mixed, mnar_labels = apply_single_mechanism_with_labels(
            ts_array, mnar_type, mnar_target, seed, MISSING_LABEL_MNAR
        )
        mixed, mar_labels = apply_single_mechanism_with_labels(
            mixed, "mar", mar_target, seed + 10_000, MISSING_LABEL_MAR
        )
        labels = np.where(mnar_labels > 0, mnar_labels, mar_labels)
        return mixed, labels
    raise ValueError(f"Unknown mask_type: {mask_type}")


def summarize_metrics(
    imputation_array: np.ndarray,
    gt_array: np.ndarray,
    mask: np.ndarray,
) -> dict:
    _EMPTY = {
        "mae": 0.0,
        "rmse": 0.0,
        "mre": 0.0,
        "nrmse": 0.0,
        "n_points": 0,
    }
    gt = np.nan_to_num(gt_array)
    pred = np.nan_to_num(imputation_array)
    n_points = int(mask.sum())
    if n_points == 0:
        return _EMPTY
    masked_gt = gt[mask]
    masked_pred = pred[mask]
    diff = masked_pred - masked_gt
    abs_diff = np.abs(diff)

    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))

    eps = 1e-8
    mean_gt = float(np.mean(np.abs(masked_gt)))
    mre = float(mae / max(mean_gt, eps))
    rms_gt = float(np.sqrt(np.mean(masked_gt ** 2)))
    nrmse = float(rmse / max(rms_gt, eps))

    return {
        "mae": mae,
        "rmse": rmse,
        "mre": mre,
        "nrmse": nrmse,
        "n_points": n_points,
    }


def aggregate_metric(runs: list[dict], key: str) -> AggregatedMetric:
    values = [float(run[key]) for run in runs]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return AggregatedMetric(mean=mean, std=std, display=f"{mean:.6f} +- {std:.6f}")


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"param_count": int(total), "trainable_param_count": int(trainable)}


def scenario_id(dataset: str, mask_type: str, missing_rate: float) -> str:
    rate_tag = str(missing_rate).replace(".", "")
    return f"{dataset}-{mask_type}-mr{rate_tag}"


def build_output_dir(
    output_root: Path | str,
    model_name: str,
    exp_group: str,
    dataset: str,
    mask_type: str,
    missing_rate: float,
    variant: str,
    seed: Optional[int] = None,
) -> Path:
    path = (
        Path(output_root)
        / model_name
        / exp_group
        / dataset
        / mask_type
        / f"mr_{missing_rate}"
        / variant
    )
    if seed is not None:
        path = path / f"seed_{seed}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path | str, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_result_markdown(result_path: Path | str, title: str, payload: dict) -> None:
    result_path = Path(result_path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if not result_path.exists():
        lines.append("# Experiment Results")
        lines.append("")
    lines.append(f"## {title}")
    lines.append(
        f"- scenario: `{payload['scenario_id']}`"
    )
    lines.append(
        f"- mae: `{payload['mae']['display']}` | rmse: `{payload['rmse']['display']}` | "
        f"mre: `{payload['mre']['display']}` | nrmse: `{payload['nrmse']['display']}`"
    )
    lines.append(
        f"- params: `{payload.get('param_count', 0)}` total / `{payload.get('trainable_param_count', 0)}` trainable"
    )
    lines.append("")
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def masked_average(values: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    weights = mask.float()
    numer = (values * weights).sum(dim=dim, keepdim=keepdim)
    denom = weights.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return numer / denom


def local_pool_1d(values: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return values
    pad = kernel_size // 2
    bsz, seq_len, n_features = values.shape
    data = values.permute(0, 2, 1).reshape(bsz * n_features, 1, seq_len)
    pooled = F.avg_pool1d(data, kernel_size=kernel_size, stride=1, padding=pad)
    return pooled.reshape(bsz, n_features, seq_len).permute(0, 2, 1)


def current_time_seconds() -> float:
    return time.perf_counter()
