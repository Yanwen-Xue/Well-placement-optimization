from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation_tools.train_surrogate import build_feature_extractor

NI, NJ, NK = 268, 203, 200
INJ_K_BOT = 41
INJ_K_TOP = 80


def _find_sr3(run_dir: str) -> Path | None:
    path = Path(str(run_dir))
    if not path.exists():
        return None
    files = sorted(path.glob("*.sr3"))
    return files[0] if files else None


def _target_group(f: h5py.File, target_day: float = 30.0 * 365.25) -> str | None:
    if "SpatialProperties" not in f:
        return None
    groups = sorted([g for g in f["SpatialProperties"].keys() if g.isdigit()], key=lambda x: int(x))
    if not groups:
        return None
    mtt = f["General/MasterTimeTable"][()]
    days = mtt["Offset in days"].astype(float)
    ids = np.array([int(g) for g in groups], dtype=int)
    group_days = days[ids]
    return groups[int(np.argmin(np.abs(group_days - target_day)))]


def compute_block_f3(sr3_path: Path, nbi: int, nbj: int) -> dict[str, float]:
    with h5py.File(sr3_path, "r") as f:
        gname = _target_group(f)
        if gname is None:
            raise RuntimeError("No spatial-property time group found")
        if "SpatialProperties/000000/PRES" not in f or f"SpatialProperties/{gname}/PRES" not in f:
            raise RuntimeError("PRES not found")

        grid = f["SpatialProperties/000000/GRID"]
        ipstcs = grid["IPSTCS"][()].astype(np.int64)
        pv = grid["BLOCKPVOL"][()].astype(np.float64)
        full0 = ipstcs - 1
        ii = full0 % NI
        jj = (full0 // NI) % NJ
        kk = full0 // (NI * NJ)
        inj = (
            (kk >= INJ_K_BOT - 1)
            & (kk <= INJ_K_TOP - 1)
            & np.isfinite(pv)
            & (pv > 0.0)
        )
        idx = np.where(inj)[0]
        if idx.size == 0:
            raise RuntimeError("No active cells in injection interval")

        p0 = f["SpatialProperties/000000/PRES"][()][idx].astype(np.float64)
        pt = f[f"SpatialProperties/{gname}/PRES"][()][idx].astype(np.float64)
        pvi = pv[idx]
        dp = (pt - p0) / 1000.0
        finite = np.isfinite(dp) & np.isfinite(pvi) & (pvi > 0.0)
        if not np.any(finite):
            raise RuntimeError("No finite pressure data")

        ii = ii[idx][finite]
        jj = jj[idx][finite]
        pvi = pvi[finite]
        dp = dp[finite]

        bi = np.minimum((ii.astype(np.int64) * nbi) // NI, nbi - 1)
        bj = np.minimum((jj.astype(np.int64) * nbj) // NJ, nbj - 1)
        bid = bi * nbj + bj
        nb = nbi * nbj

        wsum = np.bincount(bid, weights=pvi, minlength=nb)
        dpsum = np.bincount(bid, weights=pvi * dp, minlength=nb)
        active = wsum > 0.0
        block_mean = np.full(nb, np.nan, dtype=float)
        block_mean[active] = dpsum[active] / wsum[active]

        w = wsum[active]
        m = block_mean[active]
        mean_dp = float(np.sum(w * m) / np.sum(w))
        var_block = float(np.sum(w * (m - mean_dp) ** 2) / np.sum(w))
        std_block = float(np.sqrt(max(var_block, 0.0)))
        f3_block = float(std_block / max(mean_dp, 1.0e-12))

        return {
            "block_mean_mpa": mean_dp,
            "block_std_mpa": std_block,
            "block_cv": f3_block,
            "n_active_blocks": int(np.sum(active)),
            "time_group": int(gname),
        }


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
    }


def train_eval(Xf: np.ndarray, y: np.ndarray, seed: int = 42) -> dict:
    params = {
        "n_estimators": 800,
        "max_depth": 2,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 5.0,
        "min_child_weight": 3.0,
        "n_jobs": 16,
        "tree_method": "hist",
        "device": "cpu",
        "random_state": seed,
    }
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.2, random_state=seed, shuffle=True)
    model = XGBRegressor(**params)
    model.fit(Xf[train_idx], y[train_idx])
    out = {
        "heldout": _metrics(y[test_idx], model.predict(Xf[test_idx])),
        "range": {
            "min": float(np.min(y)),
            "max": float(np.max(y)),
            "mean": float(np.mean(y)),
            "std": float(np.std(y)),
            "range": float(np.max(y) - np.min(y)),
        },
    }
    fold_metrics = []
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(Xf):
        m = XGBRegressor(**params)
        m.fit(Xf[tr], y[tr])
        fold_metrics.append(_metrics(y[te], m.predict(Xf[te])))
    out["cv5"] = {
        k: {
            "mean": float(np.mean([fm[k] for fm in fold_metrics])),
            "std": float(np.std([fm[k] for fm in fold_metrics])),
        }
        for k in fold_metrics[0]
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--blocks", nargs="+", default=["4x4", "5x5", "8x8"])
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.results, allow_pickle=True)
    success = np.asarray(data["success_mask"], dtype=bool)
    sample_index = np.asarray(data["sample_index"], dtype=int)
    rows = np.where(success)[0]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    block_specs = []
    for spec in args.blocks:
        a, b = spec.lower().split("x")
        block_specs.append((spec, int(a), int(b)))

    values = {spec: np.full(len(rows), np.nan, dtype=float) for spec, _, _ in block_specs}
    details = []
    for out_pos, row in enumerate(rows):
        sr3 = _find_sr3(str(data["run_dir"][row]))
        item = {"row": int(row), "sample_index": int(sample_index[row]), "sr3": str(sr3) if sr3 else None}
        if sr3 is None:
            item["error"] = "sr3_not_found"
            details.append(item)
            continue
        for spec, nbi, nbj in block_specs:
            try:
                res = compute_block_f3(sr3, nbi, nbj)
                values[spec][out_pos] = res["block_cv"]
                item[f"{spec}_block_cv"] = res["block_cv"]
                item[f"{spec}_n_active_blocks"] = res["n_active_blocks"]
            except Exception as exc:
                item[f"{spec}_error"] = str(exc)
        details.append(item)
        if (out_pos + 1) % 20 == 0:
            print(f"processed {out_pos + 1}/{len(rows)}")

    extractor, feature_report = build_feature_extractor(Path(args.seeds), "prescreen_geometry_local")
    X = np.asarray(data["X"][rows], dtype=float)
    Xf = np.hstack([X, np.asarray(extractor(X), dtype=float)])
    report = {
        "results": str(args.results),
        "n_rows_requested": int(len(rows)),
        "feature_report": feature_report,
        "blocks": {},
        "details": details,
    }
    for spec, vals in values.items():
        finite = np.isfinite(vals)
        if np.sum(finite) >= 40:
            report["blocks"][spec] = train_eval(Xf[finite], vals[finite])
            report["blocks"][spec]["n_finite"] = int(np.sum(finite))
        else:
            report["blocks"][spec] = {"n_finite": int(np.sum(finite)), "error": "too_few_finite_values"}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report["blocks"].items()}, ensure_ascii=False, indent=2))
    print(f"Saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
