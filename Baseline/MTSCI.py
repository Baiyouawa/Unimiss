import json
import random
import argparse
from pathlib import Path
from typing import Optional
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from pygrinder import mar_logistic, mnar_x, mnar_t, calc_missing_rate
from benchpots.datasets import preprocess_ett, preprocess_italy_air_quality
from result_writer import write_result_md

# ??? MTSCI ???
MTSCI_REPO_ROOT = PROJECT_ROOT / "MTSCI"
for extra_path in [MTSCI_REPO_ROOT / "dataloader", MTSCI_REPO_ROOT / "utils"]:
    if extra_path.exists():
        sys.path.append(str(extra_path))

model_search_paths = [
    PROJECT_ROOT / "models",           # current in-repo shared layout
    MTSCI_REPO_ROOT / "models",        # upstream MTSCI layout, if present
    PROJECT_ROOT / "Code" / "models",  # legacy layout, if present
]
for model_path in model_search_paths:
    if (model_path / "model.py").exists():
        sys.path.append(str(model_path))
        break
else:
    searched = ", ".join(str(p) for p in model_search_paths)
    raise ModuleNotFoundError(f"Cannot find MTSCI model.py. Searched: {searched}")

from model import MTSCI  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def summarize_metrics(
    imputation_array: np.ndarray,
    gt_array: np.ndarray,
    mask: np.ndarray,
    *,
    mape_cap: float = 10.0,
    mape_trim_ratio: float = 0.05,
) -> dict:
    """计算在掩码位置上的指标."""
    gt = np.nan_to_num(gt_array)
    pred = np.nan_to_num(imputation_array)
    n_points = int(mask.sum())
    if n_points == 0:
        return {
            "mae": 0.0, "rmse": 0.0, "mre": 0.0,
            "mape": 0.0, "mape_capped": 0.0, "mape_trimmed": 0.0, "smape": 0.0,
            "mape_outlier_ratio": 0.0, "n_points": 0,
        }
    masked_gt = gt[mask]
    masked_pred = pred[mask]
    diff = masked_pred - masked_gt
    abs_diff = np.abs(diff)
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    eps = 1e-8
    mean_gt = float(np.mean(np.abs(masked_gt)))
    mre = float(mae / max(mean_gt, eps))
    abs_gt = np.abs(masked_gt)
    point_denom = np.maximum(abs_gt, eps)
    per_point_ape = abs_diff / point_denom
    mape = float(np.mean(per_point_ape))
    capped_ape = np.minimum(per_point_ape, mape_cap)
    mape_capped = float(np.mean(capped_ape))
    if n_points >= 20:
        k = max(1, int(np.ceil(n_points * mape_trim_ratio)))
        sorted_ape = np.sort(per_point_ape)
        mape_trimmed = float(np.mean(sorted_ape[:-k]))
    else:
        mape_trimmed = mape
    smape_denom = np.maximum(np.abs(masked_pred) + abs_gt, eps)
    smape = float(np.mean(2.0 * abs_diff / smape_denom))
    gt_threshold = max(mean_gt * 0.01, eps)
    mape_outlier_ratio = float(np.mean(abs_gt < gt_threshold))
    return {
        "mae": mae,
        "rmse": rmse,
        "mre": mre,
        "mape": mape,
        "mape_capped": mape_capped,
        "mape_trimmed": mape_trimmed,
        "smape": smape,
        "mape_outlier_ratio": mape_outlier_ratio,
        "n_points": n_points,
    }


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
    default=60,
    help="MTSCI 训练轮数（默认 60，按需调大）",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=16,
    help="批大小",
)
parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="学习率",
)
parser.add_argument(
    "--eval_samples",
    type=int,
    default=10,
    help="评估时采样条数（默认 10，过大会显著拖慢/耗显存）",
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
cache_dir = (Path(__file__).resolve().parent.parent / "Datasets").resolve()
cache_dir.mkdir(parents=True, exist_ok=True)
if DATASET_NAME == "electricity_transformer_temperature":
    data = preprocess_ett(subset="ETTm1", rate=0.01, n_steps=prep_n_steps, pattern="point")
elif DATASET_NAME == "italy_air_quality":
    data = preprocess_italy_air_quality(rate=0.01, n_steps=prep_n_steps, pattern="point")
else:
    raise ValueError(f"未知数据集 {DATASET_NAME}")
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


def make_tensors(original: np.ndarray, masked: np.ndarray):
    """构造与官方 MTSCI dataloader 语义一致的张量."""
    # 原始观测掩码（历史缺失=0）
    orig_mask = ~np.isnan(original)
    # 掩码后观测掩码（新增缺失后）
    obs_mask = ~np.isnan(masked)
    # 人工新增缺失位置（用于训练/评估目标）
    indicating = orig_mask & (~obs_mask)

    # 官方含义:
    # X_tensor: 输入(新增缺失后)
    # mask_tensor: 输入可观测掩码
    # eval_mask: 人工缺失位置
    # X_Tilde_tensor: GT（新增缺失前）
    # X_Tilde_mask: GT可观测掩码（原始掩码）
    x_in = np.nan_to_num(masked, nan=0.0).astype(np.float32)
    x_in_mask = obs_mask.astype(np.float32)
    eval_mask = indicating.astype(np.float32)
    x_gt = np.nan_to_num(original, nan=0.0).astype(np.float32)
    x_gt_mask = orig_mask.astype(np.float32)

    # 训练分支的 pred/pred_mask 在当前封装中无序列移位信息，退化为当前步 GT。
    pred_gt = x_gt
    pred_gt_mask = x_gt_mask

    return x_in, x_in_mask, eval_mask, x_gt, x_gt_mask, pred_gt, pred_gt_mask


class MTSCIDatasetTrain(torch.utils.data.Dataset):
    def __init__(self, original: np.ndarray, masked: np.ndarray):
        tensors = [make_tensors(original[i], masked[i]) for i in range(original.shape[0])]
        (
            self.X_in,
            self.X_in_mask,
            self.eval_mask,
            self.X_gt,
            self.X_gt_mask,
            self.pred_gt,
            self.pred_gt_mask,
        ) = map(
            lambda *xs: np.stack(xs, axis=0), *tensors
        )

    def __len__(self):
        return self.X_in.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_in[idx]),          # X_tensor
            torch.from_numpy(self.X_in_mask[idx]),     # mask_tensor
            torch.from_numpy(self.eval_mask[idx]),     # indicating_mask_tensor
            torch.from_numpy(self.X_gt[idx]),          # X_Tilde_tensor
            torch.from_numpy(self.X_gt_mask[idx]),     # X_Tilde_mask_tensor
            torch.from_numpy(self.pred_gt[idx]),       # pred_tensor
            torch.from_numpy(self.pred_gt_mask[idx]),  # pred_mask_tensor
        )


class MTSCIDatasetEval(torch.utils.data.Dataset):
    def __init__(self, original: np.ndarray, masked: np.ndarray):
        tensors = [make_tensors(original[i], masked[i]) for i in range(original.shape[0])]
        self.X_in, self.X_in_mask, self.eval_mask, self.X_gt, self.X_gt_mask, _, _ = map(
            lambda *xs: np.stack(xs, axis=0), *tensors
        )

    def __len__(self):
        return self.X_in.shape[0]

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.X_in[idx]),          # X_tensor
            torch.from_numpy(self.X_in_mask[idx]),     # mask_tensor
            torch.from_numpy(self.X_gt[idx]),          # X_Tilde_tensor (GT)
            torch.from_numpy(self.X_gt_mask[idx]),     # X_Tilde_mask_tensor
            torch.from_numpy(self.eval_mask[idx]),     # indicating_mask
        )


def evaluate_mtsci(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str,
    n_samples: int,
    n_steps: int,
    n_features: int,
) -> dict:
    def align_to_lk(t: torch.Tensor, bsz: int) -> torch.Tensor:
        if t.dim() != 3:
            if t.numel() == bsz * n_steps * n_features:
                return t.reshape(bsz, n_steps, n_features)
            return t
        b, d1, d2 = t.shape
        if (d1, d2) == (n_steps, n_features):
            return t
        if (d1, d2) == (n_features, n_steps):
            return t.permute(0, 2, 1)
        if t.numel() == b * n_steps * n_features:
            return t.reshape(b, n_steps, n_features)
        return t

    model.eval()
    batch_metrics = []
    total_eval_points = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = [b.to(device).float() for b in batch]
            # Eval dataloader layout:
            # [X_in, X_in_mask, X_gt, X_gt_mask, eval_mask]
            # Use X_in_mask as the true observed mask at model input.
            input_obs_mask = batch[1]
            samples, c_target, eval_points, observed_points, _ = model.evaluate(batch, n_samples=n_samples)
            if samples.shape[2] == n_steps:
                imputed = samples.median(dim=1).values
                evalmask = eval_points
                target = c_target
            else:
                imputed = samples.permute(0, 1, 3, 2).median(dim=1).values
                evalmask = eval_points.permute(0, 2, 1)
                target = c_target.permute(0, 2, 1)

            bsz = imputed.shape[0]
            imputed = align_to_lk(imputed, bsz)
            input_obs_mask = align_to_lk(input_obs_mask, bsz)
            evalmask = align_to_lk(evalmask, bsz)
            target = align_to_lk(target, bsz)

            # Keep observed entries from input, only fill missing entries with model outputs.
            imputed_data = input_obs_mask * target + (1 - input_obs_mask) * imputed

            gt_np = target.cpu().numpy()
            imputed_np = imputed_data.cpu().numpy()
            eval_np = evalmask.cpu().numpy().astype(bool)
            total_eval_points += int(eval_np.sum())
            batch_metrics.append(summarize_metrics(imputed_np, gt_np, eval_np))

    if total_eval_points == 0:
        raise RuntimeError("MTSCI evaluate_mtsci: eval mask has 0 points, please check masking pipeline.")

    return {
        k: float(np.mean([m[k] for m in batch_metrics]))
        for k in ["mae", "rmse", "mre", "mape", "mape_capped", "mape_trimmed", "smape", "mape_outlier_ratio", "n_points"]
    }


run_seeds = [3407, 3408, 3409]
all_metrics = []

output_dir = Path("outputs/mtsci")
output_dir.mkdir(parents=True, exist_ok=True)

_, n_steps, n_features = train_X_raw.shape

for seed in run_seeds:
    print(f"\n===== Run with seed {seed} =====")
    set_seed(seed)

    train_masked = apply_mask_with_mechanism(train_X_raw, mask_type, missing_rate, seed, mar_ratio=args.mar_ratio)
    val_masked = apply_mask_with_mechanism(val_X_raw, mask_type, missing_rate, seed + 1000, mar_ratio=args.mar_ratio)
    test_masked = apply_mask_with_mechanism(test_X_raw, mask_type, missing_rate, seed + 2000, mar_ratio=args.mar_ratio)

    print(f"After {mask_type} masking train: {calc_missing_rate(train_masked):.2%}")
    print(f"After {mask_type} masking val  : {calc_missing_rate(val_masked):.2%}")
    print(f"After {mask_type} masking test : {calc_missing_rate(test_masked):.2%}")

    train_ds = MTSCIDatasetTrain(train_X_raw, train_masked)
    val_ds = MTSCIDatasetEval(val_X_raw, val_masked)
    test_ds = MTSCIDatasetEval(test_X_raw, test_masked)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    config = {
        "train": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "alpha": 1,
            "beta": 5,
        },
        "diffusion": {
            "seqlen": n_steps,
            "layers": 4,
            "channels": 64,
            "nheads": 8,
            "diffusion_embedding_dim": 128,
            "beta_start": 0.0001,
            "beta_end": 0.2,
            "num_steps": 50,
            "schedule": "quad",
            "side_dim": None,  # 占位，模型内部会设置
        },
        "model": {"timeemb": 128, "featureemb": 16},
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MTSCI(config, device=device, target_dim=n_features, seq_len=n_steps).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
    p1 = int(0.5 * config["train"]["epochs"])
    p2 = int(0.75 * config["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[p1, p2], gamma=0.1)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_metric = float("inf")
    best_epoch = -1

    # 训练
    model.train()
    for epoch in range(config["train"]["epochs"]):
        loss_list = []
        for batch in train_loader:
            batch = [b.to(device).float() for b in batch]
            optimizer.zero_grad()
            loss_noise, loss_cons = model(batch, is_train=1)
            loss = loss_noise + loss_cons
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        scheduler.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"[seed {seed}] epoch {epoch+1}/{config['train']['epochs']}, loss {np.mean(loss_list):.6f}")
        val_metrics = evaluate_mtsci(
            model=model,
            data_loader=val_loader,
            device=device,
            n_samples=args.eval_samples,
            n_steps=n_steps,
            n_features=n_features,
        )
        if val_metrics["mae"] < best_metric:
            best_metric = val_metrics["mae"]
            best_epoch = epoch + 1
            torch.save(
                {
                    "epoch": best_epoch,
                    "seed": seed,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "args": vars(args),
                },
                ckpt_dir / "best.pt",
            )

    torch.save(
        {
            "epoch": config["train"]["epochs"],
            "seed": seed,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "val_best_epoch": best_epoch,
            "val_best_mae": best_metric,
            "args": vars(args),
        },
        ckpt_dir / "last.pt",
    )

    best_ckpt_path = ckpt_dir / "best.pt"
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
    else:
        print(f"[seed {seed}] best checkpoint not found, using last epoch model.")

    seed_metrics = evaluate_mtsci(
        model=model,
        data_loader=test_loader,
        device=device,
        n_samples=args.eval_samples,
        n_steps=n_steps,
        n_features=n_features,
    )
    seed_metrics["seed"] = seed
    seed_metrics["best_val_epoch"] = best_epoch
    seed_metrics["best_val_mae"] = best_metric
    all_metrics.append(seed_metrics)
    print(
        f"Seed {seed}: MAE {seed_metrics['mae']:.6f}, RMSE {seed_metrics['rmse']:.6f}, "
        f"MRE {seed_metrics['mre']:.6f}, MAPE {seed_metrics['mape']:.6f}"
    )

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
agg_mape = aggregate_metric("mape")
agg_mape_capped = aggregate_metric("mape_capped")
agg_mape_trimmed = aggregate_metric("mape_trimmed")
agg_smape = aggregate_metric("smape")
agg_mape_outlier = aggregate_metric("mape_outlier_ratio")
agg_n_points = aggregate_metric("n_points")

print("\n===== Aggregated over seeds =====")
print(f"MAE : {agg_mae['display']}")
print(f"RMSE: {agg_rmse['display']}")
print(f"MRE : {agg_mre['display']}")
print(f"MAPE: {agg_mape['display']}")
print(f"Evaluated points: {agg_n_points['display']}")

agg = {
    "dataset": DATASET_NAME,
    "mask_type": mask_type,
    "missing_rate": missing_rate,
    "seeds": run_seeds,
    "mae": agg_mae,
    "rmse": agg_rmse,
    "mre": agg_mre,
    "mape": agg_mape,
    "mape_capped": agg_mape_capped,
    "mape_trimmed": agg_mape_trimmed,
    "smape": agg_smape,
    "mape_outlier_ratio": agg_mape_outlier,
    "n_points": agg_n_points,
    "runs": all_metrics,
}
(output_dir / "metrics_avg.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
write_result_md("MTSCI", agg)
