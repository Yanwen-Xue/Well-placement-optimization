\
\
\
\
\
\
\
\
\

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import objectives


DEFAULT_LHS_NPZ = ROOT / "opt_seeds_lhs.npz"
DEFAULT_RESULTS_NPZ = ROOT / "opt_runs_lhs_doe" / "results_current.npz"
DEFAULT_OUT_DIR = ROOT / "well_set_maps"
PENALTY_THRESH = 1e8
SURF_ELEV_M = 1200.0


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = finite_mask(x, y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return {"r": float("nan"), "p": float("nan"), "n": int(len(x))}
    r, p = spearmanr(x, y)
    return {"r": float(r), "p": float(p), "n": int(len(x))}


def mann_whitney(a: np.ndarray, b: np.ndarray, higher_is_better: bool) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return {
            "n_low": int(len(a)),
            "n_high": int(len(b)),
            "median_low": float("nan"),
            "median_high": float("nan"),
            "delta_high_minus_low": float("nan"),
            "p": float("nan"),
            "direction_ok": False,
        }
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    med_low = float(np.median(a))
    med_high = float(np.median(b))
    delta = med_high - med_low
    return {
        "n_low": int(len(a)),
        "n_high": int(len(b)),
        "median_low": med_low,
        "median_high": med_high,
        "delta_high_minus_low": float(delta),
        "p": float(p),
        "direction_ok": bool(delta > 0 if higher_is_better else delta < 0),
    }


def values_at_wells(field: np.ndarray, wells: np.ndarray) -> np.ndarray:
    vals = []
    for wi, wj in np.asarray(wells, dtype=int):
        i = int(np.clip(wi, 0, field.shape[0] - 1))
        j = int(np.clip(wj, 0, field.shape[1] - 1))
        vals.append(float(field[i, j]))
    return np.asarray(vals, dtype=float)


def rank01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    mask = np.isfinite(x)
    if mask.sum() <= 1:
        return out
    order = np.argsort(np.argsort(x[mask], kind="mergesort"), kind="mergesort")
    out[mask] = order / float(mask.sum() - 1)
    return out


def top_threshold(field: np.ndarray, bound_mask: np.ndarray, top_frac: float) -> float:
    vals = field[bound_mask & np.isfinite(field)]
    return float(np.quantile(vals, 1.0 - top_frac))


def safe_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def compute_bhp_metrics(inj_wells: np.ndarray, grid_top_2d: np.ndarray, bhp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ratios = []
    margins = []
    for (wi, wj), b in zip(inj_wells, bhp):
        if not np.isfinite(b):
            ratios.append(float("nan"))
            margins.append(float("nan"))
            continue
        tvdss = float(grid_top_2d[int(wi), int(wj)])
        depth = SURF_ELEV_M + tvdss
        pp = objectives.pore_pressure_at_tvdss_kPa(tvdss)
        pf = objectives.eaton_fracture_kPa(depth, pp)
        ratios.append(float(b / pf))
        margins.append(float(objectives.BHP_SAFETY_COEFF * pf - b))
    return np.asarray(ratios, dtype=float), np.asarray(margins, dtype=float)


def group_summary(name: str, low_mask: np.ndarray, high_mask: np.ndarray, data: dict) -> dict:
    valid = data["valid"]
    low = low_mask & valid
    high = high_mask & valid
    return {
        "name": name,
        "n_low_valid": int(low.sum()),
        "n_high_valid": int(high.sum()),
        "success_ratio_low_all": safe_mean(data["success"][low_mask].astype(float)),
        "success_ratio_high_all": safe_mean(data["success"][high_mask].astype(float)),
        "F1_pressure_rise": mann_whitney(data["F1"][low], data["F1"][high], higher_is_better=False),
        "dissolution_ratio": mann_whitney(data["dissolution"][low], data["dissolution"][high], higher_is_better=True),
        "F3_brine_produced": mann_whitney(data["F3"][low], data["F3"][high], higher_is_better=True),
        "max_bhp_ratio": mann_whitney(data["max_bhp_ratio"][low_mask], data["max_bhp_ratio"][high_mask], higher_is_better=False),
        "min_bhp_margin_kpa": mann_whitney(data["min_bhp_margin_kpa"][low_mask], data["min_bhp_margin_kpa"][high_mask], higher_is_better=True),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate prescreen-map utility beyond direct objective correlation.")
    parser.add_argument("--lhs", default=str(DEFAULT_LHS_NPZ))
    parser.add_argument("--results", default=str(DEFAULT_RESULTS_NPZ))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--tag", default="utility_current")
    parser.add_argument("--top-frac", type=float, default=0.20)
    args = parser.parse_args()

    lhs_path = Path(args.lhs)
    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lhs = np.load(lhs_path, allow_pickle=True)
    res = np.load(results_path, allow_pickle=True)
    j_inj = np.asarray(lhs["J_inj_2d"], dtype=float)
    j_prod = np.asarray(lhs["J_prod_2d"], dtype=float)
    bound_mask = np.asarray(lhs["bound_mask"], dtype=bool)
    grid_top_2d = np.asarray(lhs["grid_top_2d"], dtype=float)

    sample_index = np.asarray(res["sample_index"], dtype=int)
    objs = np.asarray(res["objectives"], dtype=float)
    success = np.asarray(res["success_mask"], dtype=bool)
    failure_reason = np.asarray(res["failure_reason"].astype(str))
    inj_wells = np.asarray(res["inj_wells"], dtype=int)
    prod_wells = np.asarray(res["prod_wells"], dtype=int)
    bhp = np.asarray(res["inj_bhp_max_per_well"], dtype=float)

    valid_obj = success & np.isfinite(objs).all(axis=1) & (np.abs(objs).max(axis=1) < PENALTY_THRESH)
    n = len(sample_index)

    inj_vals = np.array([values_at_wells(j_inj, inj_wells[k]) for k in range(n)])
    prod_vals = np.array([values_at_wells(j_prod, prod_wells[k]) for k in range(n)])
    inj_mean = np.nanmean(inj_vals, axis=1)
    inj_min = np.nanmin(inj_vals, axis=1)
    prod_mean = np.nanmean(prod_vals, axis=1)
    prod_min = np.nanmin(prod_vals, axis=1)

    inj_top_thr = top_threshold(j_inj, bound_mask, float(args.top_frac))
    prod_top_thr = top_threshold(j_prod, bound_mask, float(args.top_frac))
    inj_top_frac = np.mean(inj_vals >= inj_top_thr, axis=1)
    prod_top_frac = np.mean(prod_vals >= prod_top_thr, axis=1)

    bhp_ratio = np.full((n, 5), np.nan, dtype=float)
    bhp_margin = np.full((n, 5), np.nan, dtype=float)
    for k in range(n):
        bhp_ratio[k], bhp_margin[k] = compute_bhp_metrics(inj_wells[k], grid_top_2d, bhp[k])
    max_bhp_ratio = np.nanmax(bhp_ratio, axis=1)
    min_bhp_margin = np.nanmin(bhp_margin, axis=1)

    F1 = objs[:, 0]
    dissolution = -objs[:, 1]
    F3 = objs[:, 2]
    layout_prior = 0.45 * rank01(inj_mean) + 0.45 * rank01(prod_mean) + 0.05 * inj_top_frac + 0.05 * prod_top_frac

    data = {
        "valid": valid_obj,
        "success": success,
        "F1": F1,
        "dissolution": dissolution,
        "F3": F3,
        "max_bhp_ratio": max_bhp_ratio,
        "min_bhp_margin_kpa": min_bhp_margin,
    }

    top_frac = float(args.top_frac)
    qlo = top_frac
    qhi = 1.0 - top_frac
    masks = {
        "inj_mean": (inj_mean <= np.quantile(inj_mean, qlo), inj_mean >= np.quantile(inj_mean, qhi)),
        "prod_mean": (prod_mean <= np.quantile(prod_mean, qlo), prod_mean >= np.quantile(prod_mean, qhi)),
        "layout_prior": (layout_prior <= np.nanquantile(layout_prior, qlo), layout_prior >= np.nanquantile(layout_prior, qhi)),
        "inj_top_coverage": (inj_top_frac <= np.quantile(inj_top_frac, qlo), inj_top_frac >= np.quantile(inj_top_frac, qhi)),
        "prod_top_coverage": (prod_top_frac <= np.quantile(prod_top_frac, qlo), prod_top_frac >= np.quantile(prod_top_frac, qhi)),
    }

    well_j = inj_vals.reshape(-1)
    well_ratio = bhp_ratio.reshape(-1)
    well_margin = bhp_margin.reshape(-1)

    correlations = {
        "well_J_inj_vs_BHP_ratio": spearman(well_j, well_ratio),
        "well_J_inj_vs_BHP_margin_kPa": spearman(well_j, well_margin),
        "layout_J_inj_mean_vs_max_BHP_ratio": spearman(inj_mean, max_bhp_ratio),
        "layout_J_inj_mean_vs_min_BHP_margin": spearman(inj_mean, min_bhp_margin),
        "layout_J_inj_mean_vs_dissolution_valid": spearman(inj_mean[valid_obj], dissolution[valid_obj]),
        "layout_J_inj_mean_vs_F1_valid": spearman(inj_mean[valid_obj], F1[valid_obj]),
        "layout_J_prod_mean_vs_F3_valid": spearman(prod_mean[valid_obj], F3[valid_obj]),
        "layout_J_prod_mean_vs_F1_valid": spearman(prod_mean[valid_obj], F1[valid_obj]),
        "F3_vs_F1_valid": spearman(F3[valid_obj], F1[valid_obj]),
        "layout_prior_vs_F1_valid": spearman(layout_prior[valid_obj], F1[valid_obj]),
        "layout_prior_vs_dissolution_valid": spearman(layout_prior[valid_obj], dissolution[valid_obj]),
        "layout_prior_vs_F3_valid": spearman(layout_prior[valid_obj], F3[valid_obj]),
    }

    groups = {name: group_summary(name, low, high, data) for name, (low, high) in masks.items()}
    failure_counts = {
        str(k): int(v)
        for k, v in zip(*np.unique(failure_reason.astype(str), return_counts=True))
    }

    rows_path = out_dir / f"prescreen_utility_rows_{args.tag}.csv"
    with open(rows_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_index",
                "success",
                "failure_reason",
                "inj_mean",
                "inj_min",
                "prod_mean",
                "prod_min",
                "inj_top_frac",
                "prod_top_frac",
                "layout_prior",
                "max_bhp_ratio",
                "min_bhp_margin_kpa",
                "F1",
                "dissolution",
                "F3",
            ],
        )
        writer.writeheader()
        for k in range(n):
            writer.writerow(
                {
                    "sample_index": int(sample_index[k]),
                    "success": bool(success[k]),
                    "failure_reason": str(failure_reason[k]),
                    "inj_mean": inj_mean[k],
                    "inj_min": inj_min[k],
                    "prod_mean": prod_mean[k],
                    "prod_min": prod_min[k],
                    "inj_top_frac": inj_top_frac[k],
                    "prod_top_frac": prod_top_frac[k],
                    "layout_prior": layout_prior[k],
                    "max_bhp_ratio": max_bhp_ratio[k],
                    "min_bhp_margin_kpa": min_bhp_margin[k],
                    "F1": F1[k],
                    "dissolution": dissolution[k],
                    "F3": F3[k],
                }
            )

    report = {
        "lhs_npz": str(lhs_path),
        "results_npz": str(results_path),
        "n_samples": int(n),
        "n_success": int(success.sum()),
        "n_valid_objectives": int(valid_obj.sum()),
        "failure_reason_counts": failure_counts,
        "top_fraction": top_frac,
        "top_thresholds": {"J_inj": inj_top_thr, "J_prod": prod_top_thr},
        "correlations": correlations,
        "quantile_group_tests": groups,
        "interpretation": {
            "intended_use": "Prescreen maps are local physics priors for initialization and search guidance, not direct global-response predictors.",
            "strong_evidence_patterns": [
                "J_prod should correlate positively with brine production and negatively with F1 if pressure relief dominates.",
                "J_inj should correlate with BHP safety metrics more directly than with global F1.",
                "layout_prior/top-zone groups should show better local safety or pressure-relief indicators than low-prior groups.",
            ],
        },
    }

    json_path = out_dir / f"prescreen_utility_validation_{args.tag}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    valid = valid_obj

    axes[0, 0].scatter(well_j, well_ratio, s=18, alpha=0.45, c="#3b73a8", edgecolors="none")
    axes[0, 0].set_title(f"Well J_inj vs BHP/Pf\nr={correlations['well_J_inj_vs_BHP_ratio']['r']:+.3f}")
    axes[0, 0].set_xlabel("J_inj at injection well")
    axes[0, 0].set_ylabel("max BHP / Pf")

    axes[0, 1].scatter(prod_mean[valid], F3[valid] / 1e6, s=30, alpha=0.65, c="#2f8f62", edgecolors="white")
    axes[0, 1].set_title(f"J_prod vs Brine Production\nr={correlations['layout_J_prod_mean_vs_F3_valid']['r']:+.3f}")
    axes[0, 1].set_xlabel("Mean J_prod at production wells")
    axes[0, 1].set_ylabel("F3 water produced (Mm3)")

    axes[0, 2].scatter(prod_mean[valid], F1[valid] * 100, s=30, alpha=0.65, c="#b85b38", edgecolors="white")
    axes[0, 2].set_title(f"J_prod vs Pressure Rise\nr={correlations['layout_J_prod_mean_vs_F1_valid']['r']:+.3f}")
    axes[0, 2].set_xlabel("Mean J_prod at production wells")
    axes[0, 2].set_ylabel("F1 pressure rise (%)")

    axes[1, 0].scatter(layout_prior[valid], F1[valid] * 100, s=30, alpha=0.65, c="#6b5aa6", edgecolors="white")
    axes[1, 0].set_title(f"Layout Prior vs Pressure\nr={correlations['layout_prior_vs_F1_valid']['r']:+.3f}")
    axes[1, 0].set_xlabel("Layout prior score")
    axes[1, 0].set_ylabel("F1 pressure rise (%)")

    labels = ["prod low", "prod high"]
    low, high = masks["prod_mean"]
    box = [F3[low & valid] / 1e6, F3[high & valid] / 1e6]
    axes[1, 1].boxplot(box, labels=labels, patch_artist=True)
    axes[1, 1].set_title("F3 by Production Potential Quantile")
    axes[1, 1].set_ylabel("F3 water produced (Mm3)")

    labels = ["prior low", "prior high"]
    low, high = masks["layout_prior"]
    box = [F1[low & valid] * 100, F1[high & valid] * 100]
    axes[1, 2].boxplot(box, labels=labels, patch_artist=True)
    axes[1, 2].set_title("F1 by Layout Prior Quantile")
    axes[1, 2].set_ylabel("F1 pressure rise (%)")

    for ax in axes.ravel():
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        f"Prescreen Utility Validation: {n} samples, {int(valid.sum())} valid objectives",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    fig_path = out_dir / f"prescreen_utility_validation_{args.tag}.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Rows saved: {rows_path}")
    print(f"Report saved: {json_path}")
    print(f"Figure saved: {fig_path}")
    print("\nKey correlations:")
    for key in [
        "well_J_inj_vs_BHP_ratio",
        "well_J_inj_vs_BHP_margin_kPa",
        "layout_J_prod_mean_vs_F3_valid",
        "layout_J_prod_mean_vs_F1_valid",
        "layout_prior_vs_F1_valid",
    ]:
        item = correlations[key]
        print(f"  {key}: r={item['r']:+.3f}, p={item['p']:.3g}, n={item['n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
