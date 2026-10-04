import argparse
import json
import os
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt


NI, NJ, NK = 268, 203, 200


def read_property_txt(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        vals = [float(line.split()[0]) for line in f if line.strip()]
    arr = np.asarray(vals[: NI * NJ * NK], dtype=np.float32)
    return arr.reshape((NI, NJ, NK), order="F")


def normalize01(a: np.ndarray, mask=None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if mask is None:
        mask = np.isfinite(a)
    vals = a[mask & np.isfinite(a)]
    if vals.size == 0:
        return np.zeros_like(a, dtype=np.float32)
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo:
        return np.zeros_like(a, dtype=np.float32)
    return np.clip((a - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize3(a: np.ndarray, shape, order=1) -> np.ndarray:
    zoom = [shape[i] / a.shape[i] for i in range(3)]
    return ndimage.zoom(a, zoom=zoom, order=order).astype(np.float32)


def build_full_sandbody(out_dir: Path, prop_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "full_sandbody_high_quality_268x203x200.npz"
    out_json = out_dir / "full_sandbody_high_quality_summary.json"
    if out_npz.exists():
        return out_npz

    perm = read_property_txt(prop_dir / "property_260421_all_perm.txt")
    poro = read_property_txt(prop_dir / "property_260421_all_poro.txt")
    sand_mask = (perm >= 4.0) & (poro >= 0.04)
    structure = ndimage.generate_binary_structure(3, 1)
    labels, nlab = ndimage.label(sand_mask, structure=structure)
    counts = np.bincount(labels.ravel())
    top = sorted([int(x) for x in counts[1:]], reverse=True)[:20]

    np.savez_compressed(
        out_npz,
        labels=labels.astype(np.int32),
        sand_mask=sand_mask.astype(np.uint8),
        perm=perm.astype(np.float32),
        poro=poro.astype(np.float32),
        rule=np.array(["perm>=4mD and poro>=0.04"]),
        shape=np.array([NI, NJ, NK], dtype=np.int32),
    )
    out_json.write_text(
        json.dumps(
            {
                "shape": [NI, NJ, NK],
                "rule": "permeability >= 4 mD and porosity >= 0.04",
                "components": int(nlab),
                "sand_cells": int(sand_mask.sum()),
                "largest_components": top,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_npz


def list_sr3_cases(root: Path, max_cases: int):
    sr3s = sorted(root.rglob("opt_001.sr3"))
    rows = []
    for p in sr3s:
        if p.stat().st_size < 5 * 1024**3:
            continue
        m = re.search(r"case_(\d+)", str(p))
        case_id = int(m.group(1)) if m else len(rows)
        dat = p.with_suffix(".dat")
        if dat.exists():
            rows.append((case_id, p, dat))
    rows = sorted(rows, key=lambda x: x[0])
    return rows[:max_cases]


def nearest_groups(f, target_days):
    mtt = f["General/MasterTimeTable"][()]
    groups = sorted([g for g in f["SpatialProperties"].keys() if g.isdigit()], key=lambda x: int(x))
    ids = np.asarray([int(g) for g in groups], dtype=int)
    days = mtt["Offset in days"][ids].astype(float)
    out = []
    for d in target_days:
        idx = int(np.argmin(np.abs(days - d)))
        out.append((groups[idx], float(days[idx])))
    return out


def full_from_sr3_group(f, group, name):
    grid = f["SpatialProperties/000000/GRID"]
    ipstcs = grid["IPSTCS"][()].astype(np.int64)
    full0 = ipstcs - 1
    vals = f[f"SpatialProperties/{group}/{name}"][()].astype(np.float32)
    arr = np.zeros(NI * NJ * NK, dtype=np.float32)
    arr[full0] = vals
    return arr.reshape((NI, NJ, NK), order="F")


def parse_perf_channels(dat_path: Path, shape):
    inj = np.zeros(shape, dtype=np.float32)
    prod = np.zeros(shape, dtype=np.float32)
    current = None
    in_perf = False
    with dat_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            m = re.match(r"WELL\s+'([^']+)'", s)
            if m:
                current = m.group(1)
                in_perf = False
                continue
            if s.startswith("PERF"):
                in_perf = True
                continue
            if in_perf:
                parts = s.split()
                if len(parts) < 3 or not parts[0].isdigit():
                    if s and not s.startswith("**"):
                        in_perf = False
                    continue
                i, j, k = int(parts[0]) - 1, int(parts[1]) - 1, int(parts[2]) - 1
                if 0 <= i < NI and 0 <= j < NJ and 0 <= k < NK:
                    ii = min(shape[0] - 1, int(round(i * (shape[0] - 1) / (NI - 1))))
                    jj = min(shape[1] - 1, int(round(j * (shape[1] - 1) / (NJ - 1))))
                    kk = min(shape[2] - 1, int(round(k * (shape[2] - 1) / (NK - 1))))
                    if current and current.startswith("INJ"):
                        inj[ii, jj, kk] = 1.0
                    elif current and current.startswith("PRD"):
                        prod[ii, jj, kk] = 1.0
    inj = ndimage.gaussian_filter(inj, sigma=1.0)
    prod = ndimage.gaussian_filter(prod, sigma=1.0)
    if inj.max() > 0:
        inj /= inj.max()
    if prod.max() > 0:
        prod /= prod.max()
    return inj.astype(np.float32), prod.astype(np.float32)


def build_coordinate_channels(shape):
    xs = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)[:, None, None]
    ys = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)[None, :, None]
    zs = np.linspace(0.0, 1.0, shape[2], dtype=np.float32)[None, None, :]
    x = np.broadcast_to(xs, shape)
    y = np.broadcast_to(ys, shape)
    z = np.broadcast_to(zs, shape)
    return x.astype(np.float32), y.astype(np.float32), z.astype(np.float32)


def build_dataset(args, static_npz: Path):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "a3_dataset.npz"
    if dataset_path.exists() and not args.overwrite:
        return dataset_path

    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    target_days = np.asarray([float(x) for x in args.target_days.split(",")], dtype=float)
    outputs = [s.strip() for s in args.outputs.split(",") if s.strip()]

    stat = np.load(static_npz)
    perm = stat["perm"]
    poro = stat["poro"]
    sand = stat["sand_mask"].astype(np.float32)
    perm_ch = resize3(normalize01(np.log10(np.maximum(perm, 1e-9))), target_shape, order=1)
    poro_ch = resize3(normalize01(poro), target_shape, order=1)
    sand_ch = resize3(sand, target_shape, order=0)
    x_ch, y_ch, z_ch = build_coordinate_channels(target_shape)

    rows = list_sr3_cases(Path(args.sr3_root), args.max_cases)
    X_list, Y_list, plume_list, case_ids, days_used = [], [], [], [], []

    p0_norm = None
    for n, (case_id, sr3_path, dat_path) in enumerate(rows, 1):
        print(f"[dataset] case {case_id} ({n}/{len(rows)})")
        inj_ch, prod_ch = parse_perf_channels(dat_path, target_shape)
        with h5py.File(sr3_path, "r") as f:
            groups = nearest_groups(f, target_days)
            if p0_norm is None:
                p0 = full_from_sr3_group(f, "000000", "PRES") / 1000.0
                p0_norm = resize3(normalize01(p0), target_shape, order=1)
            p0_full = full_from_sr3_group(f, "000000", "PRES")
            for group, day in groups:
                fields = {}
                for name in outputs:
                    if name == "PRES":
                        fields[name] = full_from_sr3_group(f, group, "PRES") / 1000.0
                    else:
                        fields[name] = full_from_sr3_group(f, group, name)

                y_items = []
                for name in outputs:
                    arr = fields[name]
                    if name == "PRES":
                        arr = arr - p0_full / 1000.0
                    if name in {"SG", "SW"}:
                        arr = np.clip(arr, 0.0, 1.0)
                    y_items.append(resize3(arr, target_shape, order=1))

                y = np.stack(y_items, axis=0).astype(np.float32)
                plume = (y[outputs.index("SG")] > 0.10).astype(np.float32) if "SG" in outputs else np.zeros(target_shape, dtype=np.float32)
                time_ch = np.full(target_shape, day / target_days.max(), dtype=np.float32)
                x = np.stack(
                    [perm_ch, poro_ch, sand_ch, p0_norm, inj_ch, prod_ch, x_ch, y_ch, z_ch, time_ch],
                    axis=0,
                ).astype(np.float32)
                X_list.append(x)
                Y_list.append(y)
                plume_list.append(plume)
                case_ids.append(case_id)
                days_used.append(day)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    P = np.stack(plume_list, axis=0)
    y_mean = Y.mean(axis=(0, 2, 3, 4), keepdims=True)
    y_std = Y.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-6
    Yn = (Y - y_mean) / y_std
    np.savez_compressed(
        dataset_path,
        X=X,
        Y=Y,
        Yn=Yn.astype(np.float32),
        plume=P.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        case_ids=np.asarray(case_ids, dtype=np.int32),
        days=np.asarray(days_used, dtype=np.float32),
        channels=np.array(["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "x", "y", "z", "time"]),
        targets=np.array(outputs),
        target_shape=np.asarray(target_shape, dtype=np.int32),
    )
    (out_dir / "a3_dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "samples": int(X.shape[0]),
                "cases": sorted(set(int(x) for x in case_ids)),
                "input_shape": list(X.shape),
                "target_shape": list(Y.shape),
                "target_days_requested": target_days.tolist(),
                "channels": ["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "x", "y", "z", "time"],
                "targets": outputs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


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


def build_model(in_ch, out_ch):
    import torch
    from torch import nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.GroupNorm(4 if cout >= 4 else 1, cout),
            nn.SiLU(),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.GroupNorm(4 if cout >= 4 else 1, cout),
            nn.SiLU(),
        )

    class ResidualUNet3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = block(in_ch, 32)
            self.down1 = nn.Conv3d(32, 48, 3, stride=2, padding=1)
            self.enc2 = block(48, 48)
            self.down2 = nn.Conv3d(48, 64, 3, stride=2, padding=1)
            self.enc3 = block(64, 64)
            self.down3 = nn.Conv3d(64, 96, 3, stride=2, padding=1)
            self.mid = block(96, 96)
            self.up3 = nn.ConvTranspose3d(96, 64, 2, stride=2)
            self.dec3 = block(128, 64)
            self.up2 = nn.ConvTranspose3d(64, 48, 2, stride=2)
            self.dec2 = block(96, 48)
            self.up1 = nn.ConvTranspose3d(48, 32, 2, stride=2)
            self.dec1 = block(64, 32)
            self.out_field = nn.Conv3d(32, out_ch, 1)
            self.out_plume = nn.Conv3d(32, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down1(e1))
            e3 = self.enc3(self.down2(e2))
            m = self.mid(self.down3(e3))
            d3 = self.up3(m)
            if d3.shape[-3:] != e3.shape[-3:]:
                d3 = d3[..., : e3.shape[-3], : e3.shape[-2], : e3.shape[-1]]
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            if d2.shape[-3:] != e2.shape[-3:]:
                d2 = d2[..., : e2.shape[-3], : e2.shape[-2], : e2.shape[-1]]
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            if d1.shape[-3:] != e1.shape[-3:]:
                d1 = d1[..., : e1.shape[-3], : e1.shape[-2], : e1.shape[-1]]
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out_field(d1), self.out_plume(d1)

    return ResidualUNet3D()


def make_splits(case_ids, val_frac=0.2):
    cases = np.array(sorted(set(int(x) for x in case_ids)))
    n_val = max(1, int(round(len(cases) * val_frac)))
    val_cases = set(cases[-n_val:].tolist())
    val_mask = np.array([int(c) in val_cases for c in case_ids])
    return val_mask, val_cases


def train(dataset_path: Path, out_dir: Path, epochs: int, batch_size: int, lr: float, weighted_loss: bool):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    Yn = data["Yn"].astype(np.float32)
    plume = data["plume"].astype(np.float32)
    y_mean = data["y_mean"].astype(np.float32)
    y_std = data["y_std"].astype(np.float32)
    case_ids = data["case_ids"].astype(int)
    targets = [str(x) for x in data["targets"]]

    val_mask, val_cases = make_splits(case_ids)
    train_mask = ~val_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(X.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_ds = TensorDataset(
        torch.from_numpy(X[train_mask]),
        torch.from_numpy(Yn[train_mask]),
        torch.from_numpy(Y[train_mask]),
        torch.from_numpy(plume[train_mask]),
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_x = torch.from_numpy(X[val_mask]).to(device)
    val_y = torch.from_numpy(Yn[val_mask]).to(device)
    val_y_raw = torch.from_numpy(Y[val_mask]).to(device)
    val_plume = torch.from_numpy(plume[val_mask]).to(device)

    field_weights = torch.ones(Y.shape[1], dtype=torch.float32, device=device)
    if weighted_loss:
        for i, name in enumerate(targets):
            if name == "SG":
                field_weights[i] = 8.0
            elif name == "PRES":
                field_weights[i] = 2.0
            elif name == "SW":
                field_weights[i] = 1.0
            else:
                field_weights[i] = 1.0

    train_plume = plume[train_mask]
    pos = float(train_plume.sum())
    neg = float(train_plume.size - train_plume.sum())
    pos_weight = min(200.0, max(1.0, neg / max(pos, 1.0)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))
    mse = nn.MSELoss(reduction="none")
    history = []
    best_val = float("inf")
    best_epoch = 0
    patience = 6
    wait = 0
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []
        for xb, yb, yraw, pmb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yraw = yraw.to(device, non_blocking=True)
            pmb = pmb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred_field, pred_plume_logit = model(xb)
                field_loss = mse(pred_field, yb)
                if weighted_loss:
                    weights = torch.ones_like(field_loss)
                    for i, name in enumerate(targets):
                        if name == "PRES":
                            weights[:, i] = 1.0 + 2.0 * (torch.abs(yraw[:, i]) > 0.2).float() + 4.0 * (torch.abs(yraw[:, i]) > 0.5).float()
                        elif name == "SG":
                            weights[:, i] = 1.0 + 25.0 * (yraw[:, i] > 0.02).float() + 80.0 * (yraw[:, i] > 0.10).float()
                        elif name == "SW":
                            weights[:, i] = 1.0 + 4.0 * (torch.abs(yraw[:, i] - 1.0) > 0.02).float()
                        else:
                            weights[:, i] = 1.0
                    field_loss = (field_loss * weights).mean(dim=(1, 2, 3, 4)).mean()
                else:
                    field_loss = field_loss.mean()
                plume_loss = bce(pred_plume_logit, pmb.unsqueeze(1))
                loss = field_loss + 0.5 * plume_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            pred_field, pred_plume_logit = model(val_x)
            val_field_loss = float(nn.MSELoss()(pred_field, val_y).cpu())
            val_plume_loss = float(bce(pred_plume_logit, val_plume.unsqueeze(1)).cpu())
            val_loss = val_field_loss + 0.5 * val_plume_loss

        row = {
            "epoch": ep,
            "train_loss": float(np.mean(train_losses)),
            "val_field_mse_norm": val_field_loss,
            "val_plume_bce": val_plume_loss,
            "val_loss": val_loss,
            "weighted_loss": bool(weighted_loss),
            "device": str(device),
        }
        history.append(row)
        print("[train]", row, flush=True)

        ckpt = {
            "epoch": ep,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "y_mean": y_mean,
            "y_std": y_std,
            "targets": targets,
            "val_cases": sorted(val_cases),
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            wait = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
            save_validation_preview(
                out_dir,
                model,
                val_x,
                val_y_raw,
                y_mean,
                y_std,
                val_cases,
                case_ids[val_mask],
                data["days"][val_mask],
                targets,
            )
        else:
            wait += 1
            if wait >= patience:
                print(f"[early-stop] no improvement for {patience} epochs", flush=True)
                break

    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "device": str(device),
                "targets": targets,
                "validation_cases": sorted(val_cases),
                "plume_bce_pos_weight": pos_weight,
                "samples_total": int(X.shape[0]),
                "samples_train": int(train_mask.sum()),
                "samples_val": int(val_mask.sum()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_validation_preview(out_dir, model, val_x, val_y_raw, y_mean, y_std, val_cases, val_case_ids, val_days, targets):
    import torch

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        pred_field, pred_plume_logit = model(val_x)
        pred_field = pred_field.cpu().numpy() * y_std + y_mean
        pred_plume = torch.sigmoid(pred_plume_logit).cpu().numpy()
        true = val_y_raw.cpu().numpy()

    # save two representative samples: first and last validation sample
    idxs = [0] if true.shape[0] == 1 else [0, true.shape[0] - 1]
    for idx in idxs:
        sample = true[idx]
        pred = pred_field[idx] * 1.0
        sg_idx = targets.index("SG") if "SG" in targets else None
        pres_idx = targets.index("PRES") if "PRES" in targets else None
        sg_true = sample[sg_idx] if sg_idx is not None else sample[0]
        sg_pred = pred[sg_idx] if sg_idx is not None else pred[0]
        pres_true = sample[pres_idx] if pres_idx is not None else sample[0]
        pres_pred = pred[pres_idx] if pres_idx is not None else pred[0]

        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        panels = [
            (axes[0, 0], pres_true[:, :, pres_true.shape[2] // 2], "True ΔP slice", "viridis"),
            (axes[0, 1], pres_pred[:, :, pres_pred.shape[2] // 2], "Pred ΔP slice", "viridis"),
            (axes[0, 2], (pres_pred - pres_true)[:, :, pres_true.shape[2] // 2], "ΔP error", "coolwarm"),
            (axes[0, 3], pred_plume[idx, 0, :, :, pred_plume.shape[-1] // 2], "Plume logit slice", "magma"),
            (axes[1, 0], np.max(sg_true, axis=2), "True SG max-z", "magma"),
            (axes[1, 1], np.max(sg_pred, axis=2), "Pred SG max-z", "magma"),
            (axes[1, 2], np.max(pred_plume[idx, 0], axis=2), "Plume prob max-z", "magma"),
            (axes[1, 3], np.max(np.abs(sg_pred - sg_true), axis=2), "SG abs error max-z", "coolwarm"),
        ]
        for ax, field, title, cmap in panels:
            im = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        case_lab = int(val_case_ids[idx])
        fig.suptitle(f"A3 validation preview | case {case_lab} | day {float(val_days[idx]):.1f}", fontsize=12)
        fig.savefig(fig_dir / f"preview_case_{case_lab:04d}_day_{float(val_days[idx]):.0f}.png", dpi=180)
        plt.close(fig)


def evaluate(root: Path):
    import torch

    data = np.load(root / "a3_dataset.npz")
    X = data["X"].astype(np.float32)
    Y = data["Y"].astype(np.float32)
    case_ids = data["case_ids"].astype(int)
    days = data["days"].astype(float)
    targets = [str(x) for x in data["targets"]]
    y_mean = data["y_mean"].astype(np.float32)
    y_std = data["y_std"].astype(np.float32)
    val_mask, val_cases = make_splits(case_ids)

    ckpt = torch.load(root / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    model = build_model(X.shape[1], Y.shape[1])
    model.load_state_dict(ckpt["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    with torch.no_grad():
        pred_field, pred_plume_logit = model(torch.from_numpy(X[val_mask]).to(device))
        pred_field = pred_field.cpu().numpy() * y_std + y_mean
        pred_plume = torch.sigmoid(pred_plume_logit).cpu().numpy()

    pred = pred_field * 1.0
    true = Y[val_mask]
    records = []
    for i in range(true.shape[0]):
        row = {
            "sample_index": int(np.where(val_mask)[0][i]),
            "case_id": int(case_ids[val_mask][i]),
            "day": float(days[val_mask][i]),
            "delta_pressure": metric_row(true[i, targets.index("PRES")], pred[i, targets.index("PRES")]),
            "gas_saturation": metric_row(true[i, targets.index("SG")], pred[i, targets.index("SG")]),
            "plume": plume_metrics(true[i, targets.index("SG")], pred[i, targets.index("SG")]),
        }
        if "SW" in targets:
            row["water_saturation"] = metric_row(true[i, targets.index("SW")], pred[i, targets.index("SW")])
        if "W(1)" in targets:
            row["W1"] = metric_row(true[i, targets.index("W(1)")], pred[i, targets.index("W(1)")])
        if "Z(1)" in targets:
            row["Z1"] = metric_row(true[i, targets.index("Z(1)")], pred[i, targets.index("Z(1)")])
        records.append(row)

    summary = {}
    for key in ["delta_pressure", "gas_saturation", "water_saturation", "W1", "Z1"]:
        if key in records[0]:
            summary[key] = {}
            for metric in ["rmse", "mae", "r2", "p95_abs_error", "max_abs_error"]:
                vals = [r[key][metric] for r in records]
                summary[key][metric] = float(np.nanmean(vals))
    summary["plume_iou"] = float(np.nanmean([r["plume"]["iou"] for r in records]))
    summary["plume_dice"] = float(np.nanmean([r["plume"]["dice"] for r in records]))
    summary["plume_relative_voxel_count_error"] = float(
        np.nanmean([r["plume"]["relative_voxel_count_error"] for r in records])
    )

    out = {
        "dataset": str(root / "a3_dataset.npz"),
        "checkpoint": str(root / "checkpoints" / "best.pt"),
        "validation_cases": sorted(val_cases),
        "validation_samples": int(val_mask.sum()),
        "summary_mean_over_validation_samples": summary,
        "per_sample": records,
        "note": "Targets are CMG dynamic fields as stored in SR3; W(1) and Z(1) are retained as raw CMG fields because their exact physical interpretation was not yet confirmed from local docs.",
    }
    (root / "a3_eval_metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop-dir", default=r"F:\CODE1\property_CMG")
    ap.add_argument("--sr3-root", default=r"F:\CODE1\opt_runs_lhs_doe_export")
    ap.add_argument("--out-dir", default=r"F:\CODE1\field_recon_A3_20260814")
    ap.add_argument("--max-cases", type=int, default=80)
    ap.add_argument("--target-shape", default="64,48,48")
    ap.add_argument("--target-days", default="365,1826,3652,7305,10958,18262,29219")
    ap.add_argument("--outputs", default="PRES,SG,SW,W(1),Z(1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_npz = build_full_sandbody(out_dir / "static", Path(args.prop_dir))
    dataset_path = build_dataset(args, static_npz)
    if not args.eval_only:
        train(dataset_path, out_dir, args.epochs, args.batch_size, args.lr, args.weighted_loss)
    evaluate(out_dir)


if __name__ == "__main__":
    main()
