import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def build_model():
    import torch
    from torch import nn

    class Tiny3DUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(
                nn.Conv3d(7, 12, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(12, 12, 3, padding=1),
                nn.ReLU(),
            )
            self.down = nn.MaxPool3d(2)
            self.enc2 = nn.Sequential(
                nn.Conv3d(12, 24, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(24, 24, 3, padding=1),
                nn.ReLU(),
            )
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.dec = nn.Sequential(
                nn.Conv3d(36, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv3d(16, 2, 1),
            )

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            u = self.up(e2)
            if u.shape[-3:] != e1.shape[-3:]:
                u = u[..., : e1.shape[-3], : e1.shape[-2], : e1.shape[-1]]
            return self.dec(torch.cat([u, e1], dim=1))

    return Tiny3DUNet()


def r2_score(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metric_row(y_true, y_pred):
    err = y_pred - y_true
    abs_err = np.abs(err)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(abs_err)),
        "r2": r2_score(y_true, y_pred),
        "p95_abs_error": float(np.percentile(abs_err, 95)),
        "max_abs_error": float(np.max(abs_err)),
    }


def plume_metrics(true_sg, pred_sg, threshold=0.10):
    true_mask = true_sg > threshold
    pred_mask = pred_sg > threshold
    inter = np.logical_and(true_mask, pred_mask).sum()
    union = np.logical_or(true_mask, pred_mask).sum()
    true_count = true_mask.sum()
    pred_count = pred_mask.sum()
    dice = 2.0 * inter / (true_count + pred_count) if true_count + pred_count else 1.0
    iou = inter / union if union else 1.0
    return {
        "threshold": threshold,
        "true_voxels": int(true_count),
        "pred_voxels": int(pred_count),
        "voxel_count_error": int(pred_count - true_count),
        "relative_voxel_count_error": float((pred_count - true_count) / true_count)
        if true_count
        else float("nan"),
        "iou": float(iou),
        "dice": float(dice),
    }


def plot_field(true_dp, pred_dp, true_sg, pred_sg, case_id, day, out_path):
    # The first two axes are used as the horizontal plan-view axes in this pilot.
    dp_true = true_dp[:, :, true_dp.shape[2] // 2]
    dp_pred = pred_dp[:, :, pred_dp.shape[2] // 2]
    dp_err = dp_pred - dp_true
    sg_true = np.max(true_sg, axis=2)
    sg_pred = np.max(pred_sg, axis=2)
    sg_err = sg_pred - sg_true

    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    panels = [
        (axes[0, 0], dp_true, "True ΔP (MPa)", "viridis"),
        (axes[0, 1], dp_pred, "Predicted ΔP (MPa)", "viridis"),
        (axes[0, 2], dp_err, "ΔP error (MPa)", "coolwarm"),
        (axes[0, 3], sg_true, "True max SG", "magma"),
        (axes[0, 4], sg_pred, "Predicted max SG", "magma"),
        (axes[1, 0], sg_err, "Max SG error", "coolwarm"),
        (axes[1, 1], true_sg[:, :, true_sg.shape[2] // 2], "True SG slice", "magma"),
        (axes[1, 2], pred_sg[:, :, pred_sg.shape[2] // 2], "Predicted SG slice", "magma"),
        (axes[1, 3], true_sg.max(axis=(0, 1)), "True SG by layer", "viridis"),
        (axes[1, 4], pred_sg.max(axis=(0, 1)), "Predicted SG by layer", "viridis"),
    ]
    for ax, field, title, cmap in panels:
        field = np.asarray(field)
        if field.ndim == 1:
            ax.plot(np.arange(field.size), field, color="#1f5a85", linewidth=1.5)
            ax.set_xlim(0, max(1, field.size - 1))
            ax.grid(True, alpha=0.25)
        else:
            image = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cmap)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Field prediction diagnostic | case {case_id} | day {day:.1f}", fontsize=12)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"F:\CODE1\field_dl_baseline_A_20260814")
    args = ap.parse_args()

    root = Path(args.root)
    dataset_path = root / "pilot_field_dataset_pressure_sg.npz"
    model_path = root / "model" / "tiny3d_unet_pressure_sg.pt"
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(dataset_path)
    x = data["X"].astype(np.float32)
    y = data["Y"].astype(np.float32)
    yn = data["Yn"].astype(np.float32)
    y_mean = data["y_mean"].astype(np.float32)
    y_std = data["y_std"].astype(np.float32)
    case_ids = data["case_ids"].astype(int)
    days = data["days"].astype(float)
    sand = x[:, 2] > 0.05
    cases = np.array(sorted(set(case_ids)))
    n_val = max(1, int(round(len(cases) * 0.25)))
    val_cases = set(cases[-n_val:].tolist())
    val_mask = np.array([c in val_cases for c in case_ids])

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x)).cpu().numpy()
    pred = pred_norm * y_std + y_mean
    pred[:, 1] = np.clip(pred[:, 1], 0.0, 1.0)

    records = []
    for idx in np.where(val_mask)[0]:
        row = {
            "sample_index": int(idx),
            "case_id": int(case_ids[idx]),
            "day": float(days[idx]),
            "split": "validation",
            "delta_pressure_all": metric_row(y[idx, 0], pred[idx, 0]),
            "gas_saturation_all": metric_row(y[idx, 1], pred[idx, 1]),
            "gas_saturation_plume": plume_metrics(y[idx, 1], pred[idx, 1]),
        }
        active = sand[idx]
        row["delta_pressure_sand_cells"] = metric_row(y[idx, 0][active], pred[idx, 0][active])
        row["gas_saturation_sand_cells"] = metric_row(y[idx, 1][active], pred[idx, 1][active])
        records.append(row)

    summary = {}
    for key in [
        "delta_pressure_all",
        "gas_saturation_all",
        "delta_pressure_sand_cells",
        "gas_saturation_sand_cells",
    ]:
        summary[key] = {}
        for metric in ["rmse", "mae", "r2", "p95_abs_error", "max_abs_error"]:
            vals = [r[key][metric] for r in records]
            summary[key][metric] = float(np.nanmean(vals))
    for key in ["iou", "dice", "relative_voxel_count_error"]:
        vals = [r["gas_saturation_plume"][key] for r in records]
        summary[f"plume_{key}"] = float(np.nanmean(vals))

    out = {
        "dataset": str(dataset_path),
        "model": str(model_path),
        "all_cases": cases.tolist(),
        "validation_cases": sorted(val_cases),
        "validation_samples": int(val_mask.sum()),
        "summary_mean_over_validation_samples": summary,
        "per_sample": records,
        "interpretation": {
            "pressure_units": "MPa",
            "gas_saturation_threshold_for_plume": 0.10,
            "note": "This is a pilot, low-resolution, CPU-trained baseline; it is not a final generalization result.",
        },
    }
    (root / "pilot_eval_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Plot the latest validation sample and one earlier validation time if available.
    val_indices = np.where(val_mask)[0]
    selected = [val_indices[0], val_indices[-1]] if len(val_indices) > 1 else [val_indices[0]]
    for idx in selected:
        plot_field(
            y[idx, 0],
            pred[idx, 0],
            y[idx, 1],
            pred[idx, 1],
            case_ids[idx],
            days[idx],
            figure_dir / f"case_{case_ids[idx]:04d}_day_{days[idx]:.0f}_diagnostic.png",
        )

    print(json.dumps(out["summary_mean_over_validation_samples"], indent=2))


if __name__ == "__main__":
    main()
