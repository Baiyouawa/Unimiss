import json
import random
import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
import sys
import json as _json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from result_writer import write_result_md
from pygrinder import mar_logistic, mnar_x, mnar_t, calc_missing_rate
from common.experiment_utils import load_main_dataset, summarize_metrics  # noqa: E402

# ??? FGTI ???
FGTI_ROOT = PROJECT_ROOT / "FGTI24" / "Code"
if FGTI_ROOT.exists():
    sys.path.append(str(FGTI_ROOT))
from models import main_model  # noqa: E402

# #region agent log helper
_DEBUG_LOG_PATH = "/home/liwei/backup/.cursor/debug.log"
_SESSION_ID = "debug-session"
def _dbg_log(run_id: str, hypothesis_id: str, location: str, message: str, data: dict):
    try:
        payload = {
            "sessionId": _SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(torch.tensor([]).new_full((1,), torch.cuda.Event().elapsed_time if torch.cuda.is_available() else 0).cpu().numpy()[0]) if False else int(__import__("time").time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(_json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass
# #endregion agent log helper


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
parser.add_argument(
    "--epochs",
    type=int,
    default=80,
    help=0,
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="批大小",
)
parser.add_argument(
    "--eval_samples",
    type=int,
    default=10,
    help="评估时采样条数（默认 10，过大会显著拖慢/耗显存）",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="学习率",
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

# mask 机制限制
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

def build_candidate_mask(ts_array: np.ndarray, mechanism_type: str, mechanism_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """生成候选新增缺失位置，再由后续步骤精确控制新增缺失数量。"""
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
    candidate_mask, filled = build_candidate_mask(ts_array, mechanism_type, mechanism_rate)
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


class FGTIDataset(torch.utils.data.Dataset):
    def __init__(self, orig: np.ndarray, masked: np.ndarray):
        """
        orig: 原始数据 [N, L, K]（含历史缺失）
        masked: 在可观测基础上新增缺失后的数据（含 NaN）
        """
        self.orig = orig
        self.masked = masked
        self.L = orig.shape[1]

    def __len__(self):
        return self.orig.shape[0]

    def __getitem__(self, idx):
        orig_data = self.orig[idx]          # 原始（含历史缺失）
        masked_data = self.masked[idx]      # 新增缺失后的
        orig_mask = ~np.isnan(orig_data)
        obs_mask = ~np.isnan(masked_data)

        # observed_data：新增缺失后的输入，缺失置 0
        observed = np.nan_to_num(masked_data, nan=0.0)
        observed_dataf = observed.copy()    # 简化：直接复用
        observed_tp = np.arange(self.L, dtype=np.float32)

        return (
            torch.from_numpy(observed).float(),                # observed_data
            torch.from_numpy(observed_dataf).float(),          # observed_dataf
            torch.from_numpy(obs_mask.astype(np.float32)).float(),  # observed_mask
            torch.from_numpy(observed_tp).float(),             # observed_tp
            torch.from_numpy(orig_mask.astype(np.float32)).float(), # gt_mask（原始可观测）
            torch.from_numpy(orig_data).float(),               # orig_data（含 NaN，用于评估）
        )


def build_loaders(train_X, val_X, test_X, batch_size: int):
    train_ds = FGTIDataset(train_X_raw, train_X)
    val_ds = FGTIDataset(val_X_raw, val_X)
    test_ds = FGTIDataset(test_X_raw, test_X)
    return (
        torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


run_seeds = [3407, 3408, 3409]
all_metrics = []

output_dir = Path("outputs/fgti")
output_dir.mkdir(parents=True, exist_ok=True)

# 形状依据原始数据（未掩码）
_, n_steps, n_features = train_X_raw.shape

for seed in run_seeds:
    print(f"\n===== Run with seed {seed} =====")
    set_seed(seed)

    train_X = apply_mask_with_mechanism(train_X_raw, mask_type, missing_rate, seed, mar_ratio=args.mar_ratio)
    val_X = apply_mask_with_mechanism(val_X_raw, mask_type, missing_rate, seed + 1000, mar_ratio=args.mar_ratio)
    test_X = apply_mask_with_mechanism(test_X_raw, mask_type, missing_rate, seed + 2000, mar_ratio=args.mar_ratio)

    print(f"[seed {seed}] After {mask_type} masking train: {calc_missing_rate(train_X):.2%}")
    print(f"[seed {seed}] After {mask_type} masking val  : {calc_missing_rate(val_X):.2%}")
    print(f"[seed {seed}] After {mask_type} masking test : {calc_missing_rate(test_X):.2%}")

    train_loader, val_loader, test_loader = build_loaders(train_X, val_X, test_X, args.batch_size)

    cfg = SimpleNamespace(
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch=args.batch_size,
        dataset=DATASET_NAME,
        missing_rate=missing_rate,
        seed=seed,
        seq_len=n_steps,
        enc_in=n_features,
        c_out=n_features,
        d_model=224,
        e_layers=4,
        diffusion_step_num=50,
        timeemb=64,
        featureemb=16,
        nheads=8,
        channel=32,
        proj_t=32,
        residual_layers=4,
        schedule="quad",
        beta_start=0.0001,
        beta_end=0.2,
        epoch_diff=args.epochs,
        learning_rate_diff=args.lr,
        flimit=0.3,
        topf=10,
    )

    model = main_model.FGTI(cfg).to(cfg.device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate_diff, weight_decay=1e-6)
    p1 = int(0.75 * cfg.epoch_diff)
    p2 = int(0.9 * cfg.epoch_diff)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[p1, p2], gamma=0.1)

    # 训练
    model.train()
    for epoch in range(cfg.epoch_diff):
        loss_list = []
        for observed_data, observed_dataf, observed_mask, observed_tp, gt_mask, _orig in train_loader:
            observed_data = observed_data.to(cfg.device)
            observed_dataf = observed_dataf.to(cfg.device)
            observed_mask = observed_mask.to(cfg.device)
            observed_tp = observed_tp.to(cfg.device)
            gt_mask = gt_mask.to(cfg.device)

            optim.zero_grad()
            loss = model(observed_data, observed_dataf, observed_mask, observed_tp, gt_mask)
            loss.backward()
            optim.step()
            loss_list.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[seed {seed}] epoch {epoch+1}/{cfg.epoch_diff}, loss {np.mean(loss_list):.6f}")

    # 推理
    model.eval()
    batch_metrics = []
    with torch.no_grad():
        for observed_data, observed_dataf, observed_mask, observed_tp, gt_mask, orig_data in test_loader:
            observed_data = observed_data.to(cfg.device)
            observed_dataf = observed_dataf.to(cfg.device)
            observed_mask = observed_mask.to(cfg.device)
            observed_tp = observed_tp.to(cfg.device)
            gt_mask = gt_mask.to(cfg.device)
            orig_data = orig_data.to(cfg.device)

            imputed_samples, c_target, eval_points, observed_points, _ = model.evaluate(
                observed_data, observed_dataf, observed_mask, observed_tp, gt_mask, n_samples=args.eval_samples
            )
            # 对齐形状：若模型输出为 [B, S, L, K] 则无需转置；若为 [B, S, K, L] 则转成 [B, S, L, K]
            if imputed_samples.shape[2] == n_steps:  # [B, S, L, K]
                imputed = imputed_samples.median(dim=1).values                   # [B, L, K]
                evalmask = eval_points                                           # [B, L, K] (若为 [B,K,L] 会在对齐函数处理)
                obs_pts = observed_points                                        # [B, L, K] 同上
            else:  # 视为 [B, S, K, L]
                imputed = imputed_samples.permute(0, 1, 3, 2).median(dim=1).values  # [B, L, K]
                evalmask = eval_points.permute(0, 2, 1)                              # [B, L, K]
                obs_pts = observed_points.permute(0, 2, 1)                           # [B, L, K]

            def align_to_LK(t: torch.Tensor) -> torch.Tensor:
                if t.dim() != 3:
                    if t.numel() == imputed.shape[0] * n_steps * n_features:
                        return t.reshape(imputed.shape[0], n_steps, n_features)
                    return t
                b, d1, d2 = t.shape
                if (d1, d2) == (n_steps, n_features):
                    return t
                if (d1, d2) == (n_features, n_steps):
                    return t.permute(0, 2, 1)
                if t.numel() == b * n_steps * n_features:
                    return t.reshape(b, n_steps, n_features)
                return t.expand(b, n_steps, n_features)

            imputed = align_to_LK(imputed)
            obs_pts = align_to_LK(obs_pts)
            evalmask = align_to_LK(evalmask)

            imputed_data = obs_pts * observed_data + (1 - obs_pts) * imputed

            gt_np = align_to_LK(orig_data.permute(0, 2, 1)).cpu().numpy()  # [B, L, K]
            imputed_np = align_to_LK(imputed_data).cpu().numpy()           # [B, L, K]
            eval_np = align_to_LK(evalmask).cpu().numpy().astype(bool)     # [B, L, K]
            _dbg_log(
                run_id="fgti-eval",
                hypothesis_id="H-shape",
                location="FGTI.py:evaluate:before_metrics",
                message="shapes before metrics",
                data={
                    "seed": seed,
                    "imputed_shape": list(imputed_np.shape),
                    "gt_shape": list(gt_np.shape),
                    "eval_shape": list(eval_np.shape),
                    "eval_true": int(eval_np.sum()),
                },
            )
            batch_metrics.append(summarize_metrics(imputed_np, gt_np, eval_np))

    # 聚合
    seed_metrics = {
        k: float(np.mean([m[k] for m in batch_metrics])) for k in ["mae", "rmse", "mre", "nrmse", "n_points"]
    }
    seed_metrics["seed"] = seed
    all_metrics.append(seed_metrics)
    print(
        f"Seed {seed}: MAE {seed_metrics['mae']:.6f}, RMSE {seed_metrics['rmse']:.6f}, "
        f"MRE {seed_metrics['mre']:.6f}, NRMSE {seed_metrics['nrmse']:.6f}"
    )

    # 保存
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / "metrics.json").write_text(json.dumps(seed_metrics, ensure_ascii=False, indent=2))

def aggregate_metric(metric_name: str) -> dict:
    values = [m[metric_name] for m in all_metrics]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "std": std,
        "display": f"{mean:.6f} ± {std:.6f}",
    }

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
write_result_md("FGTI", agg)
