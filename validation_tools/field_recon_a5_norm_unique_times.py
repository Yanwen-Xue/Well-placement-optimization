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


def build_coordinate_channels(shape):
    xs = np.linspace(0.0, 1.0, shape[0], dtype=np.float32)[:, None, None]
    ys = np.linspace(0.0, 1.0, shape[1], dtype=np.float32)[None, :, None]
    zs = np.linspace(0.0, 1.0, shape[2], dtype=np.float32)[None, None, :]
    return (
        np.broadcast_to(xs, shape).astype(np.float32),
        np.broadcast_to(ys, shape).astype(np.float32),
        np.broadcast_to(zs, shape).astype(np.float32),
    )


def build_full_sandbody(out_dir: Path, prop_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "full_sandbody_high_quality_268x203x200.npz"
    out_json = out_dir / "full_sandbody_high_quality_summary.json"
    if out_npz.exists():
        return out_npz

    perm = read_property_txt(prop_dir / "property_260421_all_perm.txt")
    poro = read_property_txt(prop_dir / "property_260421_all_poro.txt")
    sand_mask = (perm >= 4.0) & (poro >= 0.04)
    labels, nlab = ndimage.label(sand_mask, structure=ndimage.generate_binary_structure(3, 1))
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
    seen = set()
    for d in target_days:
        idx = int(np.argmin(np.abs(days - d)))
        if groups[idx] in seen:
            continue
        seen.add(groups[idx])
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


def build_case_targets(f, group, out_names, p0_full, target_shape):
    y_items = []
    for name in out_names:
        if name == "PRES":
            arr = full_from_sr3_group(f, group, "PRES") / 1000.0
        else:
            arr = full_from_sr3_group(f, group, name)
        if name == "PRES":
            arr = arr - p0_full / 1000.0
        if name in {"SG", "SW"}:
            arr = np.clip(arr, 0.0, 1.0)
        y_items.append(resize3(arr, target_shape, order=1))
    return np.stack(y_items, axis=0).astype(np.float16)


def build_h5_dataset(args, static_npz: Path):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / "a4_dataset.h5"
    summary_path = out_dir / "a4_dataset_summary.json"
    if h5_path.exists() and summary_path.exists() and not args.overwrite:
        return h5_path

    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    target_days = np.asarray([float(x) for x in args.target_days.split(",")], dtype=float)
    out_names = [s.strip() for s in args.outputs.split(",") if s.strip()]

    stat = np.load(static_npz)
    perm = stat["perm"]
    poro = stat["poro"]
    sand = stat["sand_mask"].astype(np.float32)
    perm_ch = resize3(normalize01(np.log10(np.maximum(perm, 1e-9))), target_shape, order=1)
    poro_ch = resize3(normalize01(poro), target_shape, order=1)
    sand_ch = resize3(sand, target_shape, order=0)
    x_ch, y_ch, z_ch = build_coordinate_channels(target_shape)

    rows = list_sr3_cases(Path(args.sr3_root), args.max_cases)
    if not rows:
        raise RuntimeError("No valid SR3 cases found.")
    with h5py.File(rows[0][1], "r") as f0:
        actual_target_days = [float(day) for _, day in nearest_groups(f0, target_days)]
    n_samples = len(rows) * len(actual_target_days)

    h5 = h5py.File(h5_path, "w")
    xds = h5.create_dataset(
        "X",
        shape=(n_samples, 10, *target_shape),
        dtype="float16",
        chunks=(1, 10, *target_shape),
        compression="lzf",
    )
    yds = h5.create_dataset(
        "Y",
        shape=(n_samples, len(out_names), *target_shape),
        dtype="float16",
        chunks=(1, len(out_names), *target_shape),
        compression="lzf",
    )
    pds = h5.create_dataset(
        "plume",
        shape=(n_samples, *target_shape),
        dtype="uint8",
        chunks=(1, *target_shape),
        compression="lzf",
    )
    cids = h5.create_dataset("case_ids", shape=(n_samples,), dtype="int32")
    days = h5.create_dataset("days", shape=(n_samples,), dtype="float32")
    h5.attrs["targets"] = json.dumps(out_names)
    h5.attrs["channels"] = json.dumps(["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "x", "y", "z", "time"])

    sum_y = np.zeros(len(out_names), dtype=np.float64)
    sum_y2 = np.zeros(len(out_names), dtype=np.float64)
    count_vox = float(n_samples * np.prod(target_shape))

    idx = 0
    case_list = []
    for n, (case_id, sr3_path, dat_path) in enumerate(rows, 1):
        print(f"[dataset] case {case_id} ({n}/{len(rows)})", flush=True)
        inj_ch, prod_ch = parse_perf_channels(dat_path, target_shape)
        with h5py.File(sr3_path, "r") as f:
            groups = nearest_groups(f, target_days)
            p0_full = full_from_sr3_group(f, "000000", "PRES")
            p0_ch = resize3(normalize01(p0_full / 1000.0), target_shape, order=1)
            for group, day in groups:
                y = build_case_targets(f, group, out_names, p0_full, target_shape)
                x = np.stack(
                    [perm_ch, poro_ch, sand_ch, p0_ch, inj_ch, prod_ch, x_ch, y_ch, z_ch, np.full(target_shape, day / target_days.max(), dtype=np.float32)],
                    axis=0,
                ).astype(np.float16)
                plume = (y[out_names.index("SG")] > 0.10).astype(np.uint8) if "SG" in out_names else np.zeros(target_shape, dtype=np.uint8)
                xds[idx] = x
                yds[idx] = y
                pds[idx] = plume
                cids[idx] = case_id
                days[idx] = day
                y64 = y.astype(np.float64)
                sum_y += y64.sum(axis=(1, 2, 3))
                sum_y2 += (y64**2).sum(axis=(1, 2, 3))
                case_list.append(case_id)
                idx += 1

    mean = sum_y / count_vox
    std = np.sqrt(np.maximum(sum_y2 / count_vox - mean**2, 1e-12))
    h5.attrs["y_mean"] = json.dumps(mean.tolist())
    h5.attrs["y_std"] = json.dumps(std.tolist())
    h5.close()

    summary_path.write_text(
        json.dumps(
            {
                "dataset": str(h5_path),
                "samples": int(n_samples),
                "cases": sorted(set(int(x) for x in case_list)),
                "input_shape": [int(n_samples), 10, *target_shape],
                "target_shape": [int(n_samples), len(out_names), *target_shape],
                "target_days_requested": target_days.tolist(),
                "target_days_used_unique": actual_target_days,
                "channels": ["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "x", "y", "z", "time"],
                "targets": out_names,
                "y_mean": mean.tolist(),
                "y_std": std.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return h5_path


class H5FieldDataset:
    def __init__(self, h5_path: Path):
        self.h5 = h5py.File(h5_path, "r")
        self.X = self.h5["X"]
        self.Y = self.h5["Y"]
        self.plume = self.h5["plume"]
        self.case_ids = self.h5["case_ids"][:]
        self.days = self.h5["days"][:]
        self.y_mean = np.asarray(json.loads(self.h5.attrs["y_mean"]), dtype=np.float32)
        self.y_std = np.asarray(json.loads(self.h5.attrs["y_std"]), dtype=np.float32)
        self.targets = json.loads(self.h5.attrs["targets"])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (
            np.asarray(self.X[idx]),
            np.asarray(self.Y[idx]),
            np.asarray(self.plume[idx]),
            int(self.case_ids[idx]),
            float(self.days[idx]),
        )


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
        "relative_voxel_count_error": float((pred_count - true_count) / true_count) if true_count else float("nan"),
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


def save_preview(out_dir, model, x, y, plume, days, case_ids, targets, y_mean, y_std, device):
    import torch

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    idxs = [0] if x.shape[0] == 1 else [0, x.shape[0] - 1]
    with torch.no_grad():
        # The HDF5 dataset uses float16 to reduce storage; the model runs in float32 here.
        preview_input = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        pred_field, pred_plume_logit = model(preview_input)
        pred_field = pred_field.float().cpu().numpy() * y_std[None, :, None, None, None] + y_mean[None, :, None, None, None]
        pred_plume = torch.sigmoid(pred_plume_logit).cpu().numpy()
    for idx in idxs:
        pres_idx = targets.index("PRES")
        sg_idx = targets.index("SG")
        true_pres = y[idx, pres_idx]
        pred_pres = pred_field[idx, pres_idx]
        true_sg = y[idx, sg_idx]
        pred_sg = pred_field[idx, sg_idx]
        fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
        panels = [
            (axes[0, 0], true_pres[:, :, true_pres.shape[2] // 2], "True PRES slice", "viridis"),
            (axes[0, 1], pred_pres[:, :, pred_pres.shape[2] // 2], "Pred PRES slice", "viridis"),
            (axes[0, 2], (pred_pres - true_pres)[:, :, true_pres.shape[2] // 2], "PRES error", "coolwarm"),
            (axes[0, 3], pred_plume[idx, 0, :, :, pred_plume.shape[-1] // 2], "Plume logit slice", "magma"),
            (axes[1, 0], np.max(true_sg, axis=2), "True SG max-z", "magma"),
            (axes[1, 1], np.max(pred_sg, axis=2), "Pred SG max-z", "magma"),
            (axes[1, 2], np.max(pred_plume[idx, 0], axis=2), "Plume prob max-z", "magma"),
            (axes[1, 3], np.max(np.abs(pred_sg - true_sg), axis=2), "SG abs error max-z", "coolwarm"),
        ]
        for ax, field, title, cmap in panels:
            im = ax.imshow(field.T, origin="lower", aspect="auto", cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        fig.suptitle(f"A5 validation preview | case {case_ids[idx]} | day {days[idx]:.1f}", fontsize=12)
        fig.savefig(fig_dir / f"preview_case_{case_ids[idx]:04d}_day_{days[idx]:.0f}.png", dpi=180)
        plt.close(fig)


def train_and_eval(dataset_path: Path, out_dir: Path, epochs: int, batch_size: int, lr: float, weighted_loss: bool):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader

    ds = H5FieldDataset(dataset_path)
    x0, y0, _, _, _ = ds[0]
    val_mask, val_cases = make_splits(ds.case_ids)
    train_mask = ~val_mask

    train_idx = np.where(train_mask)[0]
    val_idx = np.where(val_mask)[0]

    class SubsetDataset:
        def __init__(self, ds, idxs):
            self.ds = ds
            self.idxs = list(map(int, idxs))

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, i):
            return self.ds[self.idxs[i]]

    train_ds = SubsetDataset(ds, train_idx)
    val_ds = SubsetDataset(ds, val_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(x0.shape[0], y0.shape[0]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    mse = nn.MSELoss(reduction="none")

    train_plume = ds.plume[train_idx]
    pos = float(train_plume.sum())
    neg = float(train_plume.size - train_plume.sum())
    pos_weight = min(200.0, max(1.0, neg / max(pos, 1.0)))
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], dtype=torch.float32, device=device))

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    y_mean = ds.y_mean
    y_std = ds.y_std
    targets = ds.targets
    y_mean_t = torch.tensor(y_mean, dtype=torch.float32, device=device)[None, :, None, None, None]
    y_std_t = torch.tensor(y_std, dtype=torch.float32, device=device)[None, :, None, None, None]
    history = []
    best_val = float("inf")
    best_epoch = 0
    patience = 6
    wait = 0
    ckpt_dir = out_dir / "checkpoints"
    fig_dir = out_dir / "figures"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    target_index = {name: i for i, name in enumerate(targets)}

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, pmb, _, _ in loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).float()
            yb_norm = (yb - y_mean_t) / y_std_t
            pmb = pmb.to(device, non_blocking=True).float()
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred_field, pred_plume_logit = model(xb)
                field_loss = mse(pred_field, yb_norm)
                if weighted_loss:
                    weights = torch.ones_like(field_loss)
                    for name, i in target_index.items():
                        if name == "PRES":
                            weights[:, i] = 1.5
                        elif name == "SG":
                            weights[:, i] = 3.0 + 20.0 * (yb[:, i] > 0.02).float() + 100.0 * (yb[:, i] > 0.10).float()
                        elif name == "SW":
                            weights[:, i] = 1.0
                        elif name in {"W(1)", "Z(1)"}:
                            weights[:, i] = 2.5
                        else:
                            weights[:, i] = 1.0
                    field_loss = (field_loss * weights).mean()
                else:
                    field_loss = field_loss.mean()
                plume_loss = bce(pred_plume_logit, pmb.unsqueeze(1))
                loss = field_loss + 0.5 * plume_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb, pmb, _, _ in val_loader:
                xb = xb.to(device).float()
                yb = yb.to(device).float()
                yb_norm = (yb - y_mean_t) / y_std_t
                pmb = pmb.to(device).float()
                with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                    pred_field, pred_plume_logit = model(xb)
                    field_loss = mse(pred_field, yb_norm).mean()
                    plume_loss = bce(pred_plume_logit, pmb.unsqueeze(1))
                    vloss = float((field_loss + 0.5 * plume_loss).cpu())
                val_losses.append(vloss)

        val_loss = float(np.mean(val_losses))
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
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
            "targets": targets,
            "y_mean": y_mean,
            "y_std": y_std,
            "val_cases": sorted(val_cases),
            "history": history,
        }
        torch.save(ckpt, ckpt_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = ep
            wait = 0
            torch.save(ckpt, ckpt_dir / "best.pt")
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
                "samples_total": int(len(ds)),
                "samples_train": int(len(train_idx)),
                "samples_val": int(len(val_idx)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # evaluate on validation set
    ckpt = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    records = []
    with torch.no_grad():
        for local_idx, (xb, yb, pmb, cid, day) in enumerate(val_loader):
            xb = xb.to(device).float()
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pred_field, pred_plume_logit = model(xb)
            pred_field = pred_field.float().cpu().numpy()[0] * y_std[:, None, None, None] + y_mean[:, None, None, None]
            true = yb.numpy()[0]
            row = {
                "sample_index": int(val_idx[local_idx]),
                "case_id": int(cid.numpy()[0]),
                "day": float(day.numpy()[0]),
            }
            for name, i in target_index.items():
                row[name] = metric_row(true[i], pred_field[i])
            row["plume"] = plume_metrics(true[target_index["SG"]], pred_field[target_index["SG"]])
            records.append(row)

    summary = {}
    for name in targets:
        summary[name] = {}
        for metric in ["rmse", "mae", "r2", "p95_abs_error", "max_abs_error"]:
            summary[name][metric] = float(np.nanmean([r[name][metric] for r in records]))
    summary["plume_iou"] = float(np.nanmean([r["plume"]["iou"] for r in records]))
    summary["plume_dice"] = float(np.nanmean([r["plume"]["dice"] for r in records]))
    summary["plume_relative_voxel_count_error"] = float(np.nanmean([r["plume"]["relative_voxel_count_error"] for r in records]))

    (out_dir / "a4_eval_metrics.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "checkpoint": str(ckpt_dir / "best.pt"),
                "validation_cases": sorted(val_cases),
                "validation_samples": int(len(val_idx)),
                "summary_mean_over_validation_samples": summary,
                "per_sample": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # preview images
    preview_x, preview_y, preview_p, preview_c, preview_d = [], [], [], [], []
    for i in [0, len(val_idx) - 1]:
        xb, yb, pmb, cid, day = val_ds[i]
        preview_x.append(xb)
        preview_y.append(yb)
        preview_p.append(pmb)
        preview_c.append(cid)
        preview_d.append(day)
    preview_x = np.stack(preview_x, axis=0)
    preview_y = np.stack(preview_y, axis=0)
    preview_p = np.stack(preview_p, axis=0)
    preview_c = np.asarray(preview_c)
    preview_d = np.asarray(preview_d)
    save_preview(out_dir, model, preview_x, preview_y, preview_p, preview_d, preview_c, targets, y_mean, y_std, device)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop-dir", default=r"F:\CODE1\property_CMG")
    ap.add_argument("--sr3-root", default=r"F:\CODE1\opt_runs_lhs_doe_export")
    ap.add_argument("--out-dir", default=r"F:\CODE1\field_recon_A4_sitelevel_20260815")
    ap.add_argument("--max-cases", type=int, default=200)
    ap.add_argument("--target-shape", default="64,48,48")
    ap.add_argument("--target-days", default="0,30,90,180,365,730,1095,1826,3652,5478,7305,10958,14610,18262,29219")
    ap.add_argument("--outputs", default="PRES,SG,SW,W(1),Z(1)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    static_npz = build_full_sandbody(out_dir / "static", Path(args.prop_dir))
    dataset_path = build_h5_dataset(args, static_npz)
    train_and_eval(dataset_path, out_dir, args.epochs, args.batch_size, args.lr, args.weighted_loss)


if __name__ == "__main__":
    main()
