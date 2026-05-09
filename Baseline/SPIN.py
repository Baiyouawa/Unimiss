import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from result_writer import write_result_md
from pygrinder import mar_logistic, mnar_x, mnar_t, calc_missing_rate
from common.experiment_utils import load_main_dataset, summarize_metrics  # noqa: E402

# SPIN model is unified under repository-level models/layers modules.
from models.spin_model import SPINModel  # noqa: E402


DATASET_NAME = ""


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




def build_phase_encoding(batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
    pos = torch.arange(seq_len, device=device).float()
    # SPIN positional encoder expects [B, L, C]
    sin = torch.sin(2 * math.pi * pos / max(seq_len, 1))
    cos = torch.cos(2 * math.pi * pos / max(seq_len, 1))
    u = torch.stack([sin, cos], dim=-1)
    return u.unsqueeze(0).expand(batch_size, -1, -1)


def build_edge_index(train_x_raw: np.ndarray, topk: int) -> torch.Tensor:
    _, _, n_features = train_x_raw.shape
    if n_features <= 1:
        return torch.zeros((2, 0), dtype=torch.long)

    x = np.nan_to_num(train_x_raw.reshape(-1, n_features), nan=0.0)
    corr = np.corrcoef(x, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(corr, 0.0)

    edges = set()
    k = min(max(int(topk), 1), max(n_features - 1, 1))
    for i in range(n_features):
        idx = np.argpartition(-np.abs(corr[i]), kth=min(k, n_features - 1) - 1)[:k]
        for j in idx:
            if i != j:
                edges.add((i, int(j)))
                edges.add((int(j), i))

    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()
    return edge_index


def build_filled_array(ts_array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    orig_nan, filled = build_filled_array(ts_array)

    if mechanism_type == "mar":
        flat = filled.reshape(-1, filled.shape[2])
        try:
            masked_flat = unwrap(mar_logistic(flat, obs_rate=0.1, missing_rate=mechanism_rate))
            masked_full = masked_flat.reshape(ts_array.shape)
        except Exception:
            rng = np.random.default_rng(seed)
            candidate_flat = np.zeros(orig_nan.size, dtype=bool)
            obs_idx = np.where((~orig_nan).reshape(-1))[0]
            target = int(len(obs_idx) * mechanism_rate)
            if target > 0:
                chosen = rng.choice(obs_idx, size=min(target, len(obs_idx)), replace=False)
                candidate_flat[chosen] = True
            return candidate_flat.reshape(ts_array.shape), filled
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
    obs_count = int((~np.isnan(ts_array)).sum())
    total_target = int(obs_count * missing_rate)
    mix_mnar_map = {
        "electricity_transformer_temperature": "mnar_t",
        "italy_air_quality": "mnar_x",
    }

    if mask_type == "mix":
        mnar_type = mix_mnar_map[DATASET_NAME]
        mar_target = int(round(total_target * mar_ratio))
        mnar_target = total_target - mar_target
        mixed = apply_single_mechanism(ts_array, mnar_type, mnar_target, seed)
        mixed = apply_single_mechanism(mixed, "mar", mar_target, seed + 10_000)
        return mixed

    return apply_single_mechanism(ts_array, mask_type, total_target, seed)


class SPINDataset(torch.utils.data.Dataset):
    def __init__(self, orig: np.ndarray, masked: np.ndarray):
        self.orig = orig
        self.masked = masked

    def __len__(self):
        return self.orig.shape[0]

    def __getitem__(self, idx):
        orig_data = self.orig[idx]
        masked_data = self.masked[idx]
        orig_mask = ~np.isnan(orig_data)
        obs_mask = ~np.isnan(masked_data)
        observed = np.nan_to_num(masked_data, nan=0.0)
        return (
            torch.from_numpy(observed).float(),
            torch.from_numpy(obs_mask.astype(np.float32)).float(),
            torch.from_numpy(orig_data).float(),
            torch.from_numpy(orig_mask.astype(np.float32)).float(),
        )


def evaluate_model(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    edge_index: torch.Tensor,
) -> dict:
    model.eval()
    all_gt = []
    all_imputed = []
    all_mask = []
    total_points = 0
    with torch.no_grad():
        for observed, obs_mask, orig_data, orig_mask in data_loader:
            observed = observed.to(device)
            obs_mask = obs_mask.to(device)
            orig_data = orig_data.to(device)
            orig_mask = orig_mask.to(device)

            x = observed.unsqueeze(-1)
            mask = obs_mask.unsqueeze(-1)
            u = build_phase_encoding(observed.shape[0], observed.shape[1], device)
            x_hat, _ = model(x=x, u=u, mask=mask, edge_index=edge_index)
            x_hat = x_hat.squeeze(-1)

            imputed = obs_mask * observed + (1 - obs_mask) * x_hat
            target_mask = (orig_mask > 0.5) & (obs_mask < 0.5)

            gt_np = orig_data.cpu().numpy()
            imputed_np = imputed.cpu().numpy()
            mask_np = target_mask.cpu().numpy().astype(bool)
            total_points += int(mask_np.sum())
            all_gt.append(gt_np)
            all_imputed.append(imputed_np)
            all_mask.append(mask_np)

    if total_points == 0:
        raise RuntimeError("SPIN evaluate_model: target eval points are 0, please check masking pipeline.")

    gt_all = np.concatenate(all_gt, axis=0)
    imputed_all = np.concatenate(all_imputed, axis=0)
    mask_all = np.concatenate(all_mask, axis=0)
    return summarize_metrics(imputed_all, gt_all, mask_all)


def aggregate_metric(all_metrics: list[dict], metric_name: str) -> dict:
    values = [m[metric_name] for m in all_metrics]
    mean = float(np.mean(values))
    std = float(np.std(values))
    return {
        "mean": mean,
        "std": std,
        "display": f"{mean:.6f} ± {std:.6f}",
    }


def main():
    global DATASET_NAME
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["electricity_transformer_temperature", "italy_air_quality"],
        default="electricity_transformer_temperature",
    )
    parser.add_argument(
        "--mask_type",
        choices=["mar", "mnar_x", "mnar_t", "mix"],
        default="mar",
    )
    parser.add_argument("--missing_rate", type=float, choices=[0.2, 0.3, 0.4, 0.6], default=0.2)
    parser.add_argument("--mar_ratio", type=float, default=0.5, help="MAR fraction in mix scenario")
    parser.add_argument("--prep_n_steps", type=int, default=48)
    parser.add_argument("--cuda_device", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=48)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--eta", type=int, default=3)
    parser.add_argument("--message_layers", type=int, default=1)
    parser.add_argument("--graph_topk", type=int, default=3)
    parser.add_argument("--whiten_prob", type=float, default=0.3)
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    if args.cuda_device is not None:
        if torch.cuda.is_available():
            torch.cuda.set_device(args.cuda_device)
        else:
            raise ValueError("指定了 --cuda_device 但当前环境无可用 GPU")

    dataset_name = args.dataset
    DATASET_NAME = dataset_name
    data = load_main_dataset(dataset_name, args.prep_n_steps)

    train_X_raw, val_X_raw, test_X_raw = data["train_X"], data["val_X"], data["test_X"]
    mix_missing_rates = {0.2, 0.3, 0.4}
    allowed = {
        "electricity_transformer_temperature": {"mar", "mnar_t", "mix"},
        "italy_air_quality": {"mar", "mnar_x", "mix"},
    }
    if args.mask_type not in allowed[dataset_name]:
        raise ValueError(f"{dataset_name} 不支持缺失机制 {args.mask_type}，可选 {allowed[dataset_name]}")
    if args.mask_type == "mix" and args.missing_rate not in mix_missing_rates:
        raise ValueError(f"mix 场景仅支持缺失率 {sorted(mix_missing_rates)}")

    run_seeds = [3407, 3408, 3409]
    all_metrics = []
    output_dir = Path("outputs/spin")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, n_steps, n_features = train_X_raw.shape
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_index = build_edge_index(train_X_raw, topk=args.graph_topk).to(device)

    for seed in run_seeds:
        print(f"\n===== Run with seed {seed} =====")
        set_seed(seed)

        train_X = apply_mask_with_mechanism(train_X_raw, args.mask_type, args.missing_rate, seed, mar_ratio=args.mar_ratio)
        val_X = apply_mask_with_mechanism(val_X_raw, args.mask_type, args.missing_rate, seed + 1000, mar_ratio=args.mar_ratio)
        test_X = apply_mask_with_mechanism(test_X_raw, args.mask_type, args.missing_rate, seed + 2000, mar_ratio=args.mar_ratio)

        train_loader = torch.utils.data.DataLoader(
            SPINDataset(train_X_raw, train_X), batch_size=args.batch_size, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            SPINDataset(val_X_raw, val_X), batch_size=args.batch_size, shuffle=False
        )
        test_loader = torch.utils.data.DataLoader(
            SPINDataset(test_X_raw, test_X), batch_size=args.batch_size, shuffle=False
        )

        model = SPINModel(
            input_size=1,
            hidden_size=args.hidden_size,
            n_nodes=n_features,
            u_size=2,
            output_size=1,
            n_layers=args.n_layers,
            eta=args.eta,
            message_layers=args.message_layers,
        ).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = seed_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_metric = float("inf")
        best_epoch = -1

        for epoch in range(args.epochs):
            model.train()
            loss_list = []
            for observed, obs_mask, orig_data, orig_mask in train_loader:
                observed = observed.to(device)
                obs_mask = obs_mask.to(device)
                orig_data = orig_data.to(device)
                orig_mask = orig_mask.to(device)

                # SPIN-style training augmentation: additionally hide a subset
                # of currently observed points to improve denoising robustness.
                if args.whiten_prob > 0:
                    rand_keep = (torch.rand_like(obs_mask) > args.whiten_prob).float()
                    train_mask = obs_mask * rand_keep
                else:
                    train_mask = obs_mask

                x_in = observed * train_mask
                x = x_in.unsqueeze(-1)
                mask = train_mask.unsqueeze(-1)
                u = build_phase_encoding(observed.shape[0], observed.shape[1], device)
                x_hat, _ = model(x=x, u=u, mask=mask, edge_index=edge_index)
                x_hat = x_hat.squeeze(-1)

                target_mask = (orig_mask > 0.5) & (train_mask < 0.5)
                if target_mask.sum() > 0:
                    # Original SPIN uses L1-style objective in training.
                    loss = torch.mean(torch.abs(x_hat[target_mask] - orig_data[target_mask]))
                else:
                    loss = torch.zeros((), device=device)

                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_list.append(loss.item())

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"[seed {seed}] epoch {epoch+1}/{args.epochs}, loss {np.mean(loss_list):.6f}")

            val_metrics = evaluate_model(model, val_loader, device, edge_index)
            if val_metrics["mae"] < best_metric:
                best_metric = val_metrics["mae"]
                best_epoch = epoch + 1
                torch.save(
                    {
                        "epoch": best_epoch,
                        "seed": seed,
                        "model_state": model.state_dict(),
                        "optim_state": optim.state_dict(),
                        "val_metrics": val_metrics,
                        "args": vars(args),
                    },
                    ckpt_dir / "best.pt",
                )

        torch.save(
            {
                "epoch": args.epochs,
                "seed": seed,
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "val_best_epoch": best_epoch,
                "val_best_mae": best_metric,
                "args": vars(args),
            },
            ckpt_dir / "last.pt",
        )

        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            state = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(state["model_state"])

        seed_metrics = evaluate_model(model, test_loader, device, edge_index)
        seed_metrics["seed"] = seed
        seed_metrics["best_val_epoch"] = best_epoch
        seed_metrics["best_val_mae"] = best_metric
        all_metrics.append(seed_metrics)
        print(
            f"Seed {seed}: MAE {seed_metrics['mae']:.6f}, RMSE {seed_metrics['rmse']:.6f}, "
            f"MRE {seed_metrics['mre']:.6f}, NRMSE {seed_metrics['nrmse']:.6f}"
        )

        (seed_dir / "metrics.json").write_text(json.dumps(seed_metrics, ensure_ascii=False, indent=2))

    agg_mae = aggregate_metric(all_metrics, "mae")
    agg_rmse = aggregate_metric(all_metrics, "rmse")
    agg_mre = aggregate_metric(all_metrics, "mre")
    agg_nrmse = aggregate_metric(all_metrics, "nrmse")
    agg_n_points = aggregate_metric(all_metrics, "n_points")
    print("\n===== Aggregated over seeds =====")
    print(f"MAE : {agg_mae['display']}")
    print(f"RMSE: {agg_rmse['display']}")
    print(f"MRE : {agg_mre['display']}")
    print(f"NRMSE: {agg_nrmse['display']}")
    print(f"Evaluated points: {agg_n_points['display']}")

    agg = {
        "dataset": dataset_name,
        "mask_type": args.mask_type,
        "missing_rate": args.missing_rate,
        "seeds": run_seeds,
        "mae": agg_mae,
        "rmse": agg_rmse,
        "mre": agg_mre,
        "nrmse": agg_nrmse,
        "n_points": agg_n_points,
        "runs": all_metrics,
    }
    (output_dir / "metrics_avg.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    write_result_md("SPIN", agg)


if __name__ == "__main__":
    main()
