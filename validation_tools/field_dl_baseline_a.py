import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import ndimage


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

    print("[static] reading perm/poro")
    perm = read_property_txt(prop_dir / "property_260421_all_perm.txt")
    poro = read_property_txt(prop_dir / "property_260421_all_poro.txt")
    sand_mask = (perm >= 4.0) & (poro >= 0.04)

    print("[static] connected components")
    structure = ndimage.generate_binary_structure(3, 1)
    labels, nlab = ndimage.label(sand_mask, structure=structure)
    labels = labels.astype(np.int32)
    counts = np.bincount(labels.ravel())
    top = sorted([int(x) for x in counts[1:]], reverse=True)[:20]

    np.savez_compressed(
        out_npz,
        labels=labels,
        sand_mask=sand_mask,
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


def full_from_sr3_group(f, group, name):
    grid = f["SpatialProperties/000000/GRID"]
    ipstcs = grid["IPSTCS"][()].astype(np.int64)
    full0 = ipstcs - 1
    vals = f[f"SpatialProperties/{group}/{name}"][()].astype(np.float32)
    arr = np.zeros(NI * NJ * NK, dtype=np.float32)
    arr[full0] = vals
    return arr.reshape((NI, NJ, NK), order="F")


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


def build_dataset(args, static_npz: Path):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "pilot_field_dataset_pressure_sg.npz"
    if dataset_path.exists() and not args.overwrite:
        return dataset_path

    target_shape = tuple(int(x) for x in args.target_shape.split(","))
    target_days = np.asarray([float(x) for x in args.target_days.split(",")], dtype=float)

    stat = np.load(static_npz)
    perm = stat["perm"]
    poro = stat["poro"]
    sand = stat["sand_mask"].astype(np.float32)
    static_perm = resize3(normalize01(np.log10(np.maximum(perm, 1e-9))), target_shape, order=1)
    static_poro = resize3(normalize01(poro), target_shape, order=1)
    static_sand = resize3(sand, target_shape, order=0)

    rows = list_sr3_cases(Path(args.sr3_root), args.max_cases)
    X_list, Y_list, case_ids, days_used = [], [], [], []

    p0_low = None
    for n, (case_id, sr3_path, dat_path) in enumerate(rows, 1):
        print(f"[dataset] case {case_id} ({n}/{len(rows)})")
        inj_ch, prod_ch = parse_perf_channels(dat_path, target_shape)
        with h5py.File(sr3_path, "r") as f:
            groups = nearest_groups(f, target_days)
            if p0_low is None:
                p0 = full_from_sr3_group(f, "000000", "PRES") / 1000.0
                p0_low = resize3(normalize01(p0), target_shape, order=1)
            p0_full = full_from_sr3_group(f, "000000", "PRES")
            for group, day in groups:
                pres = full_from_sr3_group(f, group, "PRES")
                sg = full_from_sr3_group(f, group, "SG")
                dp = (pres - p0_full) / 1000.0
                dp_low = resize3(dp, target_shape, order=1)
                sg_low = resize3(np.clip(sg, 0.0, 1.0), target_shape, order=1)
                time_ch = np.full(target_shape, day / target_days.max(), dtype=np.float32)
                x = np.stack([static_perm, static_poro, static_sand, p0_low, inj_ch, prod_ch, time_ch], axis=0)
                y = np.stack([dp_low, sg_low], axis=0)
                X_list.append(x.astype(np.float32))
                Y_list.append(y.astype(np.float32))
                case_ids.append(case_id)
                days_used.append(day)

    X = np.stack(X_list, axis=0)
    Y = np.stack(Y_list, axis=0)
    y_mean = Y.mean(axis=(0, 2, 3, 4), keepdims=True)
    y_std = Y.std(axis=(0, 2, 3, 4), keepdims=True) + 1e-6
    Yn = (Y - y_mean) / y_std
    np.savez_compressed(
        dataset_path,
        X=X,
        Y=Y,
        Yn=Yn.astype(np.float32),
        y_mean=y_mean.astype(np.float32),
        y_std=y_std.astype(np.float32),
        case_ids=np.asarray(case_ids, dtype=np.int32),
        days=np.asarray(days_used, dtype=np.float32),
        channels=np.array(["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "time"]),
        targets=np.array(["delta_pressure_mpa", "gas_saturation"]),
        target_shape=np.asarray(target_shape, dtype=np.int32),
    )
    (out_dir / "pilot_field_dataset_summary.json").write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "samples": int(X.shape[0]),
                "cases": sorted(set(int(x) for x in case_ids)),
                "input_shape": list(X.shape),
                "target_shape": list(Y.shape),
                "target_days_requested": target_days.tolist(),
                "channels": ["logperm", "poro", "sand_mask", "p0", "inj_perf", "prod_perf", "time"],
                "targets": ["delta_pressure_mpa", "gas_saturation"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


def train_baseline(dataset_path: Path, out_dir: Path, epochs: int, batch_size: int, weighted_loss: bool):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    data = np.load(dataset_path)
    X = data["X"].astype(np.float32)
    Y = data["Yn"].astype(np.float32)
    Y_raw = data["Y"].astype(np.float32)
    case_ids = data["case_ids"]
    cases = np.array(sorted(set(int(x) for x in case_ids)))
    n_val_cases = max(1, int(round(len(cases) * 0.25)))
    val_cases = set(cases[-n_val_cases:].tolist())
    val_mask = np.array([int(c) in val_cases for c in case_ids])
    train_mask = ~val_mask

    class Tiny3DUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv3d(7, 12, 3, padding=1), nn.ReLU(), nn.Conv3d(12, 12, 3, padding=1), nn.ReLU())
            self.down = nn.MaxPool3d(2)
            self.enc2 = nn.Sequential(nn.Conv3d(12, 24, 3, padding=1), nn.ReLU(), nn.Conv3d(24, 24, 3, padding=1), nn.ReLU())
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
            self.dec = nn.Sequential(nn.Conv3d(36, 16, 3, padding=1), nn.ReLU(), nn.Conv3d(16, 2, 1))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.down(e1))
            u = self.up(e2)
            if u.shape[-3:] != e1.shape[-3:]:
                u = u[..., : e1.shape[-3], : e1.shape[-2], : e1.shape[-1]]
            return self.dec(torch.cat([u, e1], dim=1))

    device = torch.device("cpu")
    model = Tiny3DUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(
        torch.from_numpy(X[train_mask]),
        torch.from_numpy(Y[train_mask]),
        torch.from_numpy(Y_raw[train_mask]),
    )
    val_x = torch.from_numpy(X[val_mask]).to(device)
    val_y = torch.from_numpy(Y[val_mask]).to(device)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = []
    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        for xb, yb, yraw in loader:
            xb, yb, yraw = xb.to(device), yb.to(device), yraw.to(device)
            opt.zero_grad()
            pred = model(xb)
            if weighted_loss:
                # SG plume cells are sparse; without weighting the model can minimize loss by predicting near-zero SG everywhere.
                dp_weight = 1.0 + 2.0 * (torch.abs(yraw[:, 0:1]) > 0.2).float()
                sg_weight = 1.0 + 80.0 * (yraw[:, 1:2] > 0.02).float() + 200.0 * (yraw[:, 1:2] > 0.10).float()
                dp_loss = (((pred[:, 0:1] - yb[:, 0:1]) ** 2) * dp_weight).mean()
                sg_loss = (((pred[:, 1:2] - yb[:, 1:2]) ** 2) * sg_weight).mean()
                loss = dp_loss + sg_loss
            else:
                loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).cpu())
        row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)),
            "val_mse_norm": val_loss,
            "weighted_loss": bool(weighted_loss),
        }
        history.append(row)
        print("[train]", row)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "tiny3d_unet_pressure_sg.pt")
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prop-dir", default=r"F:\CODE1\property_CMG")
    ap.add_argument("--sr3-root", default=r"F:\CODE1\opt_runs_lhs_doe_export")
    ap.add_argument("--out-dir", default=r"F:\CODE1\field_dl_baseline_A_20260814")
    ap.add_argument("--max-cases", type=int, default=6)
    ap.add_argument("--target-shape", default="48,36,36")
    ap.add_argument("--target-days", default="365,1826,3652,7305,10958,18262,29219")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--weighted-loss", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    static_npz = build_full_sandbody(out_dir / "static", Path(args.prop_dir))
    dataset_path = build_dataset(args, static_npz)
    train_baseline(dataset_path, out_dir / "model", args.epochs, args.batch_size, args.weighted_loss)


if __name__ == "__main__":
    main()
