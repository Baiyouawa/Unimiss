import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_ROOT = PROJECT_ROOT / "Baseline"
for extra_path in [PROJECT_ROOT, BASELINE_ROOT]:
    if str(extra_path) not in sys.path:
        sys.path.append(str(extra_path))

from common.experiment_utils import (  # noqa: E402
    MISSING_LABEL_MAR,
    MISSING_LABEL_MNAR,
    apply_mask_with_labels,
    load_main_dataset,
    summarize_metrics,
)
from models.unimiss_model import UniMissModel  # noqa: E402
from pygrinder import calc_missing_rate  # noqa: E402
from result_writer import write_result_md  # noqa: E402

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - visualization is optional
    plt = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_seeds(seed_arg: str) -> list[int]:
    return [int(s.strip()) for s in seed_arg.split(",") if s.strip()]


def load_dataset(dataset_name: str, prep_n_steps: int) -> dict:
    return load_main_dataset(dataset_name, prep_n_steps)


class SequenceDataset(Dataset):
    def __init__(self, masked_x: np.ndarray, raw_x: np.ndarray, mech_labels: np.ndarray, period_len: int):
        self.masked_filled = torch.from_numpy(np.nan_to_num(masked_x, nan=0.0)).float()
        self.raw_filled = torch.from_numpy(np.nan_to_num(raw_x, nan=0.0)).float()
        self.obs_mask = torch.from_numpy((~np.isnan(masked_x)).astype(np.float32))
        self.eval_mask = torch.from_numpy((np.isnan(masked_x) & ~np.isnan(raw_x)).astype(np.float32))
        self.mech_labels = torch.from_numpy(mech_labels.astype(np.int64))
        self.seq_len = masked_x.shape[1]
        self.n_features = masked_x.shape[2]
        self.phase = build_phase_matrix(self.seq_len, period_len).unsqueeze(0).repeat(masked_x.shape[0], 1, 1)
        density = self.obs_mask.mean(dim=1)
        self.density = density.float()
        self.time_index = torch.arange(self.seq_len).long()

    def __len__(self) -> int:
        return self.masked_filled.shape[0]

    def __getitem__(self, idx: int) -> dict:
        return {
            "x": self.masked_filled[idx],
            "raw_x": self.raw_filled[idx],
            "obs_mask": self.obs_mask[idx],
            "target_mask": self.eval_mask[idx],
            "mech_labels": self.mech_labels[idx],
            "phase": self.phase[idx],
            "density": self.density[idx],
            "time_index": self.time_index,
        }


def build_phase_matrix(seq_len: int, period_len: int) -> torch.Tensor:
    pos = torch.arange(seq_len).float()
    period = max(int(period_len), 1)
    return torch.stack(
        [
            torch.sin(2 * np.pi * pos / period),
            torch.cos(2 * np.pi * pos / period),
        ],
        dim=-1,
    )


def reconstruction_loss(x_hat: torch.Tensor, raw_x: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    denom = target_mask.sum().clamp_min(1.0)
    return (((x_hat - raw_x) ** 2) * target_mask).sum() / denom


def gate_regulation_loss(
    beta_om: torch.Tensor,
    beta_mm: torch.Tensor,
    mech_labels: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    label_mask = target_mask.float()
    mar_target = ((mech_labels == MISSING_LABEL_MAR).float() * label_mask)
    mnar_target = ((mech_labels == MISSING_LABEL_MNAR).float() * label_mask)
    supervise = mar_target + mnar_target
    denom = supervise.sum().clamp_min(1.0)
    beta_om = beta_om.clamp_min(1e-8)
    beta_mm = beta_mm.clamp_min(1e-8)
    loss = -(mar_target * torch.log(beta_om) + mnar_target * torch.log(beta_mm)).sum() / denom
    return loss


def count_parameters(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": int(total), "trainable": int(trainable)}


def resolve_model_scale(args: argparse.Namespace) -> dict:
    d_model = args.d_model
    n_heads = args.n_heads
    n_layers = args.n_layers
    d_ff = args.d_ff
    if args.lightweight_level in {"small", "lite_s"}:
        d_model = max(96, d_model // 2)
        n_heads = max(4, n_heads // 2)
        n_layers = max(3, n_layers - 1)
        d_ff = max(128, d_ff // 2)
    elif args.lightweight_level == "lite_m":
        d_model = max(128, (d_model * 3) // 4)
        n_heads = max(4, n_heads // 2)
        n_layers = max(3, n_layers - 1)
        d_ff = max(160, (d_ff * 3) // 4)
    elif args.lightweight_level == "tiny":
        d_model = max(64, d_model // 3)
        n_heads = max(2, n_heads // 4)
        n_layers = max(2, n_layers - 2)
        d_ff = max(96, d_ff // 3)
    return {"d_model": d_model, "n_heads": n_heads, "n_layers": n_layers, "d_ff": d_ff}


def infer_run_name(args: argparse.Namespace) -> str:
    if args.run_name != "default":
        return args.run_name

    tags = []
    if args.experiment_group == "hyperparameter":
        tags.extend(
            [
                f"d{args.d_model}",
                f"layers{args.n_layers}",
                f"period{args.period_len}",
                f"gateT{str(args.gate_temperature).replace('.', 'p')}",
            ]
        )
    if args.lightweight_level != "base":
        tags.append(args.lightweight_level)
    if not args.use_oo:
        tags.append("no_oo")
    if not args.use_om:
        tags.append("no_om")
    if not args.use_mm:
        tags.append("no_mm")
    if not args.use_stage2_gate:
        tags.append("no_gate")
    if not args.use_topology_expert:
        tags.append("no_topology")
    if not args.use_periodic_expert:
        tags.append("no_periodic")
    if not args.use_extreme_expert:
        tags.append("no_extreme")
    return "__".join(tags) if tags else "full"


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def visualize_sample(seed_dir: Path, payload: dict) -> None:
    viz_path = seed_dir / "visualization.json"
    save_json(viz_path, payload)
    if plt is None:
        return
    if not payload["samples"]:
        return
    sample = payload["samples"][0]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(sample["ground_truth"], label="ground_truth")
    ax.plot(sample["prediction"], label="prediction")
    ax.plot(sample["observed"], label="observed", alpha=0.7)
    ax.set_title("UniMiss visualization sample")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(seed_dir / "visualization.png", dpi=200)
    plt.close(fig)


def train_one_seed(
    args: argparse.Namespace,
    model_kwargs: dict,
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    seed: int,
    output_dir: Path,
) -> dict:
    set_seed(seed)
    train_masked, train_labels = apply_mask_with_labels(train_raw, args.dataset, args.mask_type, args.missing_rate, seed, mar_ratio=args.mar_ratio)
    val_masked, val_labels = apply_mask_with_labels(val_raw, args.dataset, args.mask_type, args.missing_rate, seed + 1000, mar_ratio=args.mar_ratio)
    test_masked, test_labels = apply_mask_with_labels(test_raw, args.dataset, args.mask_type, args.missing_rate, seed + 2000, mar_ratio=args.mar_ratio)

    print(f"[seed {seed}] After {args.mask_type} masking train: {calc_missing_rate(train_masked):.2%}")
    print(f"[seed {seed}] After {args.mask_type} masking val  : {calc_missing_rate(val_masked):.2%}")
    print(f"[seed {seed}] After {args.mask_type} masking test : {calc_missing_rate(test_masked):.2%}")

    train_ds = SequenceDataset(train_masked, train_raw, train_labels, args.period_len)
    val_ds = SequenceDataset(val_masked, val_raw, val_labels, args.period_len)
    test_ds = SequenceDataset(test_masked, test_raw, test_labels, args.period_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if args.cuda_device is not None:
            torch.cuda.set_device(args.cuda_device)
        torch.cuda.reset_peak_memory_stats()

    model = UniMissModel(n_features=train_raw.shape[-1], **model_kwargs).to(device)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    save_json(seed_dir / "config.json", {"seed": seed, **vars(args), **model_kwargs, "param_count": count_parameters(model)})

    if args.experiment_group == "param_count":
        counts = count_parameters(model)
        save_json(seed_dir / "param_count.json", counts)
        return {"seed": seed, "mae": 0.0, "rmse": 0.0, "mre": 0.0, "nrmse": 0.0, "n_points": 0}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_state = None
    best_val = float("inf")
    train_wall_begin = time.perf_counter()

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(
                x=batch["x"],
                mask=batch["obs_mask"],
                phase=batch["phase"],
                density=batch["density"],
                raw_x=batch["raw_x"],
                target_mask=batch["target_mask"],
                time_index=batch["time_index"],
            )
            loss_recon = reconstruction_loss(outputs["x_hat"], batch["raw_x"], batch["target_mask"])
            loss_gate = gate_regulation_loss(
                outputs["beta_om"],
                outputs["beta_mm"],
                batch["mech_labels"],
                batch["target_mask"],
            )
            loss = (
                loss_recon
                + args.lambda_sep * outputs["sep_loss"]
                + args.lambda_gate * loss_gate
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_preds = []
        val_targets = []
        val_masks = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(
                    x=batch["x"],
                    mask=batch["obs_mask"],
                    phase=batch["phase"],
                    density=batch["density"],
                    raw_x=batch["raw_x"],
                    target_mask=batch["target_mask"],
                    time_index=batch["time_index"],
                )
                val_preds.append(outputs["x_hat"].cpu().numpy())
                val_targets.append(batch["raw_x"].cpu().numpy())
                val_masks.append(batch["target_mask"].cpu().numpy().astype(bool))
        val_pred = np.concatenate(val_preds, axis=0)
        val_target = np.concatenate(val_targets, axis=0)
        val_mask = np.concatenate(val_masks, axis=0)
        val_metrics = summarize_metrics(val_pred, val_target, val_mask)
        if val_metrics["rmse"] < best_val:
            best_val = val_metrics["rmse"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"[seed {seed}] epoch={epoch + 1}/{args.epochs} "
            f"train_loss={running_loss / max(len(train_loader), 1):.6f} "
            f"val_rmse={val_metrics['rmse']:.6f}"
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_time_sec = time.perf_counter() - train_wall_begin

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), seed_dir / "best.pt")

    model.eval()
    test_preds = []
    test_targets = []
    test_masks = []
    viz_samples = []
    gate_om_values = []
    gate_mm_values = []
    gate_label_values = []
    infer_begin = time.perf_counter()
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                x=batch["x"],
                mask=batch["obs_mask"],
                phase=batch["phase"],
                density=batch["density"],
                raw_x=batch["raw_x"],
                target_mask=batch["target_mask"],
                time_index=batch["time_index"],
            )
            pred = outputs["x_hat"].cpu().numpy()
            raw = batch["raw_x"].cpu().numpy()
            target_mask = batch["target_mask"].cpu().numpy().astype(bool)
            observed = batch["x"].cpu().numpy()
            mech_labels = batch["mech_labels"].cpu().numpy()
            beta_om = outputs["beta_om"].cpu().numpy()
            beta_mm = outputs["beta_mm"].cpu().numpy()

            test_preds.append(pred)
            test_targets.append(raw)
            test_masks.append(target_mask)
            gate_om_values.append(beta_om[target_mask])
            gate_mm_values.append(beta_mm[target_mask])
            gate_label_values.append(mech_labels[target_mask])

            if args.experiment_group == "visualization" and batch_idx == 0:
                max_samples = min(args.visualization_samples, pred.shape[0])
                for i in range(max_samples):
                    viz_samples.append(
                        {
                            "sample_id": int(i),
                            "feature_id": 0,
                            "ground_truth": raw[i, :, 0].tolist(),
                            "prediction": pred[i, :, 0].tolist(),
                            "observed": observed[i, :, 0].tolist(),
                            "target_mask": target_mask[i, :, 0].astype(int).tolist(),
                            "mechanism_label": mech_labels[i, :, 0].astype(int).tolist(),
                            "beta_om": beta_om[i, :, 0].tolist(),
                            "beta_mm": beta_mm[i, :, 0].tolist(),
                            "gate_prompt_norm": outputs["gate_prompt"][i, :, 0].norm(dim=-1).cpu().tolist(),
                        }
                    )
    infer_time_sec = time.perf_counter() - infer_begin

    test_pred = np.concatenate(test_preds, axis=0)
    test_target = np.concatenate(test_targets, axis=0)
    test_mask = np.concatenate(test_masks, axis=0)
    metrics = summarize_metrics(test_pred, test_target, test_mask)
    metrics["train_time_sec"] = train_time_sec
    metrics["infer_time_sec"] = infer_time_sec
    metrics["peak_gpu_mem_mb"] = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if device.type == "cuda"
        else 0.0
    )

    save_json(seed_dir / "metrics.json", metrics)
    if viz_samples:
        gate_om_concat = np.concatenate(gate_om_values) if gate_om_values else np.array([], dtype=float)
        gate_mm_concat = np.concatenate(gate_mm_values) if gate_mm_values else np.array([], dtype=float)
        gate_label_concat = np.concatenate(gate_label_values) if gate_label_values else np.array([], dtype=int)
        visualize_sample(
            seed_dir,
            {
                "dataset": args.dataset,
                "mask_type": args.mask_type,
                "missing_rate": args.missing_rate,
                "gate_distribution": {
                    "beta_om_mean": float(gate_om_concat.mean()) if gate_om_concat.size else 0.0,
                    "beta_mm_mean": float(gate_mm_concat.mean()) if gate_mm_concat.size else 0.0,
                    "mar_like_beta_om_mean": float(gate_om_concat[gate_label_concat == MISSING_LABEL_MAR].mean())
                    if np.any(gate_label_concat == MISSING_LABEL_MAR)
                    else 0.0,
                    "mnar_like_beta_mm_mean": float(gate_mm_concat[gate_label_concat == MISSING_LABEL_MNAR].mean())
                    if np.any(gate_label_concat == MISSING_LABEL_MNAR)
                    else 0.0,
                },
                "samples": viz_samples,
            },
        )
    return {"seed": seed, **metrics}


def aggregate_runs(args: argparse.Namespace, runs: list[dict], output_dir: Path) -> dict:
    def stat(key: str) -> dict:
        values = np.array([run[key] for run in runs], dtype=float)
        return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    agg = {
        "dataset": args.dataset,
        "mask_type": args.mask_type,
        "missing_rate": args.missing_rate,
        "seeds": [run["seed"] for run in runs],
        "runs": runs,
        "mae": stat("mae"),
        "rmse": stat("rmse"),
        "mre": stat("mre"),
        "nrmse": stat("nrmse"),
        "n_points": stat("n_points"),
        "train_time_sec": stat("train_time_sec") if "train_time_sec" in runs[0] else None,
        "infer_time_sec": stat("infer_time_sec") if "infer_time_sec" in runs[0] else None,
        "peak_gpu_mem_mb": stat("peak_gpu_mem_mb") if "peak_gpu_mem_mb" in runs[0] else None,
        "experiment_group": args.experiment_group,
        "run_name": args.run_name,
    }
    save_json(output_dir / "metrics_avg.json", agg)
    write_result_md("UniMiss", agg, result_path=output_dir / "result.md")
    return agg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["electricity_transformer_temperature", "italy_air_quality"],
        default="electricity_transformer_temperature",
    )
    parser.add_argument("--mask_type", choices=["mar", "mnar_x", "mnar_t", "mix"], default="mar")
    parser.add_argument("--missing_rate", type=float, choices=[0.2, 0.3, 0.4], default=0.2)
    parser.add_argument("--mar_ratio", type=float, default=0.5,
                        help="MAR fraction in mix scenario: 0.2=MNAR-dominant, 0.5=balanced, 0.8=MAR-dominant")
    parser.add_argument("--prep_n_steps", type=int, default=48)
    parser.add_argument("--cuda_device", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--period_len", type=int, default=24)
    parser.add_argument("--lambda_sep", type=float, default=0.05)
    parser.add_argument("--lambda_gate", type=float, default=0.1)
    parser.add_argument("--gate_temperature", type=float, default=1.0)
    parser.add_argument("--lightweight_level", choices=["base", "small", "lite_s", "lite_m", "tiny"], default="base")
    parser.add_argument(
        "--experiment_group",
        choices=["main", "ablation", "hyperparameter", "visualization", "scaling", "param_count"],
        default="main",
    )
    parser.add_argument("--run_name", type=str, default="default")
    parser.add_argument("--seeds", type=str, default="3407,3408,3409")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--visualization_samples", type=int, default=2)
    parser.add_argument("--use-oo", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-om", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-mm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-stage2-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-sep-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-srne", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-topology-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-periodic-expert", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-extreme-expert", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args([a for a in sys.argv[1:] if a != "--"])

    allowed = {
        "electricity_transformer_temperature": {"mar", "mnar_t", "mix"},
        "italy_air_quality": {"mar", "mnar_x", "mix"},
    }
    if args.mask_type not in allowed[args.dataset]:
        raise ValueError(f"{args.dataset} does not support {args.mask_type}; choose from {allowed[args.dataset]}")

    data = load_dataset(args.dataset, args.prep_n_steps)
    train_raw, val_raw, test_raw = data["train_X"], data["val_X"], data["test_X"]
    print(f"Base missing rate train: {calc_missing_rate(train_raw):.2%}")
    print(f"Base missing rate val  : {calc_missing_rate(val_raw):.2%}")
    print(f"Base missing rate test : {calc_missing_rate(test_raw):.2%}")

    scale_kwargs = resolve_model_scale(args)
    args.run_name = infer_run_name(args)
    model_kwargs = {
        **scale_kwargs,
        "phase_dim": 2,
        "period_len": args.period_len,
        "dropout": args.dropout,
        "use_oo": args.use_oo,
        "use_om": args.use_om,
        "use_mm": args.use_mm,
        "use_stage2_gate": args.use_stage2_gate,
        "use_sep_loss": args.use_sep_loss,
        "use_srne": args.use_srne,
        "use_topology_expert": args.use_topology_expert,
        "use_periodic_expert": args.use_periodic_expert,
        "use_extreme_expert": args.use_extreme_expert,
        "gate_temperature": args.gate_temperature,
    }

    mar_ratio_tag = f"_mar{int(args.mar_ratio * 100)}" if args.mask_type == "mix" and args.mar_ratio != 0.5 else ""
    default_output = (
        PROJECT_ROOT
        / "outputs"
        / "unimiss"
        / args.experiment_group
        / args.dataset
        / f"{args.mask_type}{mar_ratio_tag}"
        / f"mr_{args.missing_rate}"
        / args.run_name
    )
    output_dir = Path(args.output_dir) if args.output_dir else default_output
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_manifest.json", {**vars(args), **model_kwargs})

    runs = []
    for seed in parse_seeds(args.seeds):
        run = train_one_seed(args, model_kwargs, train_raw, val_raw, test_raw, seed, output_dir)
        runs.append(run)

    aggregate_runs(args, runs, output_dir)


if __name__ == "__main__":
    main()
