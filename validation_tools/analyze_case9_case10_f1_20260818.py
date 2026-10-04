from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patheffects as patheffects
from matplotlib.colors import Normalize, PowerNorm
from scipy.ndimage import label as connected_components
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

ROOT = Path(r"F:\CODE1")
CASE_ROOT = ROOT / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817\final_analysis_cases_18\compare"
OUT = ROOT / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817\final_analysis_cases_18\case9_case10_F1_analysis_20260818"
OUT.mkdir(parents=True, exist_ok=True)

NI, NJ, NK = 268, 203, 200
INJ = (40, 80)  # zero-based, upper bound excluded
PRD = (0, 15)
TARGET_DAY = 30.0 * 365.25
FIGURE_OUT = Path(r"C:\Users\admin\Desktop\paper_figure\writen\figure")
FIGURE_OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "none",
        "figure.facecolor": "none",
    }
)

sys.path.insert(0, str(ROOT))
import cmg_runner  # noqa: E402
import objectives  # noqa: E402


CASES = {
    9: CASE_ROOT / r"case1_F1_low\run_001\opt_001.sr3",
    10: CASE_ROOT / r"case2_F1_high\run_001\opt_001.sr3",
}
DISPLAY_CASE = {9: 1, 10: 2}
WELLS = {
    9: {
        "inj": np.array([[97, 143], [87, 124], [96, 81], [76, 127], [185, 169]], dtype=int),
        "prod": np.array([[183, 176], [38, 162], [163, 75], [80, 160], [73, 101]], dtype=int),
    },
    10: {
        "inj": np.array([[100, 127], [201, 165], [108, 127], [95, 82], [92, 151]], dtype=int),
        "prod": np.array([[85, 83], [70, 157], [146, 73], [193, 182], [74, 176]], dtype=int),
    },
}


def load_property(path: Path) -> np.ndarray:
    return cmg_runner._read_property_txt(str(path), NI, NJ, NK)


def physical_xy(seeds: dict[str, np.ndarray], wells: np.ndarray) -> np.ndarray:
    cx = seeds["cx_3d"][:, :, 40]
    cy = seeds["cy_3d"][:, :, 40]
    return np.asarray([[cx[i, j], cy[i, j]] for i, j in wells], dtype=float)


def run_length_metrics(values: np.ndarray, thickness: np.ndarray, threshold: float = 4.0) -> tuple[float, int]:
    mask = (~np.isfinite(values)) | (values < threshold)
    longest = 0.0
    count = 0
    current = 0.0
    for flag, h in zip(mask, thickness):
        if flag and np.isfinite(h) and h > 0:
            current += float(h)
        else:
            if current > 0:
                longest = max(longest, current)
                count += 1
            current = 0.0
    if current > 0:
        longest = max(longest, current)
        count += 1
    return longest, count


def column_metrics(i: int, j: int, interval: tuple[int, int], perm: np.ndarray, poro: np.ndarray,
                   thick: np.ndarray, sand: np.ndarray, object_id: np.ndarray) -> dict[str, float]:
    k0, k1 = interval
    k = np.arange(k0, k1)
    h = np.asarray(thick[i, j, k], dtype=float)
    kk = np.asarray(perm[i, j, k], dtype=float)
    ph = np.asarray(poro[i, j, k], dtype=float)
    sm = np.asarray(sand[k, i, j], dtype=bool)
    valid = np.isfinite(h) & (h > 0) & np.isfinite(kk) & (kk > 0) & np.isfinite(ph)
    h = np.where(valid, h, 0.0)
    kk = np.where(valid, kk, np.nan)
    ph = np.where(valid, ph, np.nan)
    gross = float(h.sum())
    net = float(h[sm & valid].sum())
    kh = float(np.nansum(kk * h))
    phih = float(np.nansum(ph * h))
    longest, low_runs = run_length_metrics(kk, h)
    high = np.isfinite(kk) & (kk >= 4.0)
    high_weight = np.where(high, kk * h, 0.0)
    centroid = float(np.sum((k - k0 + 0.5) * high_weight) / high_weight.sum()) if high_weight.sum() > 0 else np.nan
    mid = len(k) // 2
    top_kh = float(np.nansum(kk[:mid] * h[:mid]))
    bot_kh = float(np.nansum(kk[mid:] * h[mid:]))
    ids = np.unique(object_id[k0:k1, i, j][sm & valid])
    ids = ids[ids > 0]
    return {
        "gross_thickness_m": gross,
        "net_thickness_m": net,
        "net_to_gross": net / gross if gross > 0 else np.nan,
        "kh_mD_m": kh,
        "kh_mean_mD": kh / gross if gross > 0 else np.nan,
        "phih_m": phih,
        "poro_h_weighted": phih / gross if gross > 0 else np.nan,
        "log10k_h_weighted": float(np.nansum(np.log10(kk) * h) / gross) if gross > 0 else np.nan,
        "lowk_longest_run_m": longest,
        "lowk_run_count": low_runs,
        "highk_centroid_layer_relative": centroid,
        "top_to_bottom_kh_ratio": top_kh / bot_kh if bot_kh > 0 else np.nan,
        "object_count": int(ids.size),
        "object_ids": ";".join(str(int(x)) for x in ids),
    }


def spatial_metrics(points: np.ndarray) -> dict[str, float]:
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    tri = d[np.triu_indices(len(points), 1)]
    graph_w = np.exp(-d / 2000.0)
    np.fill_diagonal(graph_w, 0.0)
    lap = np.diag(graph_w.sum(axis=1)) - graph_w
    eig = np.linalg.eigvalsh(lap)
    hull_area = float(ConvexHull(points).volume) if len(points) >= 3 else np.nan
    return {
        "pair_distance_mean_m": float(tri.mean()),
        "pair_distance_min_m": float(tri.min()),
        "pair_distance_max_m": float(tri.max()),
        "pair_distance_cv": float(tri.std() / tri.mean()),
        "close_pair_count_below_1000m": int(np.count_nonzero(tri < 1000.0)),
        "inverse_distance_interference": float(np.sum(1.0 / np.maximum(tri, 1.0))),
        "exp_distance_overlap_index": float(np.sum(np.exp(-tri / 2000.0))),
        "convex_hull_area_m2": hull_area,
        "proximity_graph_lambda2": float(eig[1]) if len(eig) > 1 else np.nan,
    }


def cross_group_metrics(inj_xy: np.ndarray, prod_xy: np.ndarray, prod_volume: np.ndarray) -> dict[str, float]:
    d = np.linalg.norm(inj_xy[:, None, :] - prod_xy[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(d)
    weights = prod_volume / max(float(prod_volume.sum()), 1.0)
    relief = np.sum(weights[None, :] * np.exp(-d / 3000.0), axis=1)
    return {
        "inj_prod_distance_mean_m": float(d.mean()),
        "inj_prod_distance_min_m": float(d.min()),
        "nearest_prod_distance_mean_m": float(d.min(axis=1).mean()),
        "nearest_prod_distance_max_m": float(d.min(axis=1).max()),
        "hungarian_inj_prod_distance_m": float(d[rows, cols].mean()),
        "production_weighted_relief_coupling_mean": float(relief.mean()),
        "production_weighted_relief_coupling_min": float(relief.min()),
        "production_weighted_relief_coupling_max": float(relief.max()),
    }


def pressure_buildup_map(path: Path, target_day: float) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        days = np.asarray(f["General/MasterTimeTable"]["Offset in days"], dtype=float)
        groups = sorted((name for name in f["SpatialProperties"] if name.isdigit()), key=int)
        group_days = days[np.asarray([int(name) for name in groups], dtype=int)]
        group = groups[int(np.argmin(np.abs(group_days - target_day)))]
        grid = f["SpatialProperties/000000/GRID"]
        ipstcs = np.asarray(grid["IPSTCS"], dtype=np.int64) - 1
        pv = np.asarray(grid["BLOCKPVOL"], dtype=float)
        ii = ipstcs % NI
        jj = (ipstcs // NI) % NJ
        kk = ipstcs // (NI * NJ)
        p0 = np.asarray(f["SpatialProperties/000000/PRES"], dtype=float)
        pt = np.asarray(f[f"SpatialProperties/{group}/PRES"], dtype=float)
        valid = (
            (kk >= INJ[0])
            & (kk < INJ[1])
            & np.isfinite(pv)
            & (pv > 0)
            & np.isfinite(p0)
            & np.isfinite(pt)
        )
        num = np.zeros((NI, NJ), dtype=float)
        den = np.zeros((NI, NJ), dtype=float)
        dp_mpa = (pt[valid] - p0[valid]) / 1000.0
        np.add.at(num, (ii[valid], jj[valid]), dp_mpa * pv[valid])
        np.add.at(den, (ii[valid], jj[valid]), pv[valid])
        result = np.full((NI, NJ), np.nan, dtype=float)
        np.divide(num, den, out=result, where=den > 0)
        return result, float(days[int(group)])


def pressure_field_metrics(field: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(field)
    positive = np.where(valid, np.maximum(field, 0.0), 0.0)
    cell_area_km2 = 0.0025
    effective_area = (
        cell_area_km2 * positive.sum() ** 2 / np.sum(positive ** 2)
        if np.sum(positive ** 2) > 0
        else np.nan
    )
    metrics = {
        "pressure_map_mean_mpa": float(np.nanmean(field)),
        "pressure_map_std_mpa": float(np.nanstd(field)),
        "pressure_map_max_mpa": float(np.nanmax(field)),
        "pressure_effective_area_km2": float(effective_area),
    }
    structure = np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
    for threshold in (1.0, 2.0, 3.0, 4.0):
        mask = valid & (field >= threshold)
        labels, count = connected_components(mask, structure=structure)
        sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=int)
        stem = str(threshold).replace(".", "p")
        metrics[f"pressure_area_ge_{stem}mpa_km2"] = float(mask.sum() * cell_area_km2)
        metrics[f"pressure_components_ge_{stem}mpa"] = int(count)
        metrics[f"pressure_largest_component_share_ge_{stem}mpa"] = (
            float(sizes.max() / mask.sum()) if sizes.size and mask.sum() else 0.0
        )
    return metrics


def local_interval_map(interval: tuple[int, int], thick: np.ndarray, sand: np.ndarray) -> np.ndarray:
    k0, k1 = interval
    h = np.asarray(thick[:, :, k0:k1], dtype=float)
    sm = np.transpose(sand[k0:k1], (1, 2, 0))
    valid = np.isfinite(h) & (h > 0)
    gross = np.sum(np.where(valid, h, 0.0), axis=2)
    net = np.sum(np.where(valid & sm, h, 0.0), axis=2)
    out = np.full((NI, NJ), np.nan, dtype=float)
    np.divide(net, gross, out=out, where=gross > 0)
    return out


def label_wells(ax, points: np.ndarray, prefix: str, color: str,
                all_points: np.ndarray, used_positions: list[np.ndarray]) -> None:
    candidates = [(10, 10), (10, -10), (-10, 10), (-10, -10),
                  (13, 0), (-13, 0), (0, 13), (0, -13)]
    for number, point in enumerate(points, 1):
        best = None
        best_score = -np.inf
        for dx, dy in candidates:
            trial = point + np.asarray([dx * 68.0, dy * 55.0])
            other = all_points[np.any(np.abs(all_points - point) > 1e-8, axis=1)]
            well_clearance = np.min(np.linalg.norm(other - trial, axis=1)) if other.size else 1.0e9
            label_clearance = (
                np.min(np.linalg.norm(np.asarray(used_positions) - trial, axis=1))
                if used_positions else 1.0e9
            )
            score = min(well_clearance, label_clearance)
            if score > best_score:
                best_score = score
                best = (dx, dy, trial)
        dx, dy, trial = best
        used_positions.append(trial)
        ax.annotate(
            f"{prefix}{number}", point, xytext=(dx, dy), textcoords="offset points",
            ha="left" if dx > 0 else "right" if dx < 0 else "center",
            va="bottom" if dy > 0 else "top" if dy < 0 else "center",
            fontsize=7.2, color=color, zorder=8,
        )


def add_wells(ax, seeds: dict[str, np.ndarray], case: int, labels: bool = True) -> None:
    inj_xy = physical_xy(seeds, WELLS[case]["inj"])
    prod_xy = physical_xy(seeds, WELLS[case]["prod"])
    ax.scatter(inj_xy[:, 0], inj_xy[:, 1], s=30, marker="o", color="#3C5488", edgecolor="white", linewidth=0.7, zorder=6, label="Injection well")
    ax.scatter(prod_xy[:, 0], prod_xy[:, 1], s=30, marker="s", color="#7A8793", edgecolor="white", linewidth=0.7, zorder=6, label="Production well")
    if labels:
        all_points = np.vstack([inj_xy, prod_xy])
        used_positions: list[np.ndarray] = []
        label_wells(ax, inj_xy, "I", "#3C5488", all_points, used_positions)
        label_wells(ax, prod_xy, "P", "#5E6870", all_points, used_positions)


def plot_dynamic_mechanism(seeds: dict[str, np.ndarray], summaries: pd.DataFrame, ts: pd.DataFrame,
                           pressure_maps: dict[int, np.ndarray]) -> None:
    # Muted blue-grey and warm ochre retain contrast in print without high saturation.
    colors = {9: "#55738F", 10: "#B48758"}
    markers = {9: "o", 10: "s"}
    line_styles = {9: (0, (4.5, 1.8, 1.2, 1.8)), 10: (0, (6.0, 2.0))}
    marker_styles = {9: "o", 10: "s"}
    cx = seeds["cx_3d"][:, :, 40]
    cy = seeds["cy_3d"][:, :, 40]
    bound = np.asarray(seeds["bound_mask"], dtype=bool)
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0))
    fig.subplots_adjust(left=0.09, right=0.90, bottom=0.09, top=0.90, hspace=0.32, wspace=0.28)
    ax = axes[0, 0]
    for case in (9, 10):
        d = ts[ts.case == case]
        markevery = max(len(d) // 12, 1)
        ax.plot(d.time_years, 100.0 * d.F1_relative_to_initial, color=colors[case], lw=1.55,
                linestyle=line_styles[case], marker=marker_styles[case], markevery=markevery,
                ms=3.2, markerfacecolor="white", markeredgewidth=0.75,
                label=f"Case {DISPLAY_CASE[case]}")
        row = summaries[summaries.case == case].iloc[0]
        ax.scatter(row.pavg_peak_year, 100.0 * row.F1, s=28, color=colors[case], edgecolor="white", linewidth=0.6, zorder=4)
        ax.annotate(
            f"peak ({row.pavg_peak_year:.1f} yr)",
            (row.pavg_peak_year, 100.0 * row.F1),
            xytext=(5, -12 if case == 9 else 8), textcoords="offset points",
            fontsize=7.2, color=colors[case],
            ha="left", va="top" if case == 9 else "bottom",
        )
    ax.axvline(30.0, color="#777777", lw=0.8, ls=(0, (2.5, 2.0)), zorder=1)
    ax.text(30.5, 0.97, "injection/production stop", transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="left", fontsize=7.0, color="#666666")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Average pressure increase (%)")
    ax.text(0.02, 0.96, "(a)", transform=ax.transAxes, va="top", fontsize=9.5)

    ax = axes[0, 1]
    for case in (9, 10):
        d = ts[ts.case == case]
        markevery = max(len(d) // 12, 1)
        ax.plot(d.time_years, d.brine_total_m3 / 1e6, color=colors[case], lw=1.55,
                linestyle=line_styles[case], marker=marker_styles[case], markevery=markevery,
                ms=3.2, markerfacecolor="white", markeredgewidth=0.75,
                label=f"Case {DISPLAY_CASE[case]}")
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Cumulative brine production (10⁶ m³)")
    ax.text(0.02, 0.96, "(b)", transform=ax.transAxes, va="top", fontsize=9.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper center", bbox_to_anchor=(0.50, 0.985),
               ncol=2, columnspacing=1.2, handlelength=2.2, borderaxespad=0.0)

    vmax = max(float(np.nanpercentile(pressure_maps[c], 99.5)) for c in (9, 10))
    norm = Normalize(vmin=0.0, vmax=vmax)
    pressure_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pressure_nature_blue", ["#F5F6F4", "#DCE5EA", "#AFC5D3", "#7096B3", "#3C6287", "#233B5D"]
    )
    for panel, case, ax in zip(("(c)", "(d)"), (9, 10), axes[1]):
        field = pressure_maps[case]
        mesh = ax.contourf(cx, cy, field, levels=np.linspace(0.0, vmax, 18), cmap=pressure_cmap, norm=norm, extend="neither")
        mesh.set_rasterized(False)
        ax.contour(cx, cy, bound.astype(float), levels=[0.5], colors=["#6e7478"], linewidths=0.7, linestyles="--")
        add_wells(ax, seeds, case)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Case {DISPLAY_CASE[case]}", fontsize=9.5)
        ax.text(0.02, 0.96, panel, transform=ax.transAxes, va="top", fontsize=9.5, color="white")
    lower_positions = [ax.get_position() for ax in axes[1]]
    lower_y0 = min(pos.y0 for pos in lower_positions)
    lower_y1 = max(pos.y1 for pos in lower_positions)
    cax = fig.add_axes([0.875, lower_y0, 0.016, lower_y1 - lower_y0])
    cbar = fig.colorbar(mesh, cax=cax, extend="neither")
    cbar.set_label("30-year pressure buildup (MPa)\n(K41--K80, PV-weighted)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)
    for ax in axes.ravel():
        ax.tick_params(labelsize=8, width=0.7, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    for ext in ("png", "pdf", "svg"):
        target = FIGURE_OUT / f"case9_case10_F1_dynamic_mechanism.{ext}"
        fig.savefig(target, dpi=900 if ext == "png" else None, bbox_inches="tight", transparent=True)
        fig.savefig(OUT / target.name, dpi=900 if ext == "png" else None, bbox_inches="tight", transparent=True)
    plt.close(fig)


def plot_geological_mechanism(seeds: dict[str, np.ndarray], prod_ntg: np.ndarray, wells: pd.DataFrame,
                              dynamic_records: list[dict], perm: np.ndarray) -> None:
    cx = seeds["cx_3d"][:, :, 40]
    cy = seeds["cy_3d"][:, :, 40]
    bound = np.asarray(seeds["bound_mask"], dtype=bool)
    records = {int(x["case"]): x for x in dynamic_records}
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.0))
    fig.subplots_adjust(left=0.08, right=0.89, bottom=0.10, top=0.95, hspace=0.32, wspace=0.42)
    ntg_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ntg_shared_soft_ochre",
        ["#FCFAF4", "#F6F0DF", "#EEE3C8", "#E4D3AA", "#D8C294", "#CBB17F", "#BDA16F"],
    )
    # Restore the original paired colors used for the producer comparison.
    case_colors = {9: "#B8CBD8", 10: "#D8C3A5"}
    line_colors = {9: "#466F87", 10: "#A66F3F"}
    for panel, case, ax in zip(("(a)", "(b)"), (9, 10), axes[0]):
        mesh = ax.contourf(
            cx, cy, prod_ntg, levels=np.linspace(0, 1, 21), cmap=ntg_cmap,
            norm=PowerNorm(gamma=1.75, vmin=0.0, vmax=1.0),
        )
        mesh.set_rasterized(False)
        ax.contour(cx, cy, bound.astype(float), levels=[0.5], colors=["#6e7478"], linewidths=0.7, linestyles="--")
        add_wells(ax, seeds, case)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Case {DISPLAY_CASE[case]}: production interval K1--K15", fontsize=9.5)
        ax.text(0.02, 0.96, panel, transform=ax.transAxes, va="top", fontsize=9.5)
    cax = fig.add_axes([0.92, 0.565, 0.017, 0.34])
    cbar = fig.colorbar(mesh, cax=cax, extend="neither")
    cbar.set_label("Production-interval net-to-gross\n(K1--K15)", fontsize=8.5)
    cbar.ax.tick_params(labelsize=8)

    ax = axes[1, 0]
    x = np.arange(1, 6)
    width = 0.34
    for offset, case, color in ((-width / 2, 9, case_colors[9]), (width / 2, 10, case_colors[10])):
        subset = wells[(wells.case == case) & (wells.group == "prod")].sort_values("well")
        ax.bar(x + offset, subset.kh_mD_m, width=width, color=color,
               edgecolor="#5C5A52", linewidth=0.45, label=rf"Case {DISPLAY_CASE[case]} $kh$")
    ax.set_xlabel("Production well")
    ax.set_ylabel(r"Transmissivity proxy, $kh$ (mD m)")
    ax.set_xticks(x, [f"P{i}" for i in x])
    ax2 = ax.twinx()
    for case, color, marker in ((9, line_colors[9], "o"), (10, line_colors[10], "s")):
        q = np.array([records[case]["prod_final_m3"][f"PRD-{i}"] for i in x]) / 1e6
        ax2.plot(x, q, color=color, marker=marker, lw=1.35, ms=4.8,
                 markeredgecolor="white", markeredgewidth=0.5, label=f"Case {DISPLAY_CASE[case]} brine")
    ax2.set_ylabel(r"Cumulative brine production ($10^6$ m$^3$)", labelpad=3)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=6.7, ncol=2,
              loc="upper right", bbox_to_anchor=(0.99, 0.99),
              columnspacing=0.8, handletextpad=0.35, borderaxespad=0.0)
    ax.text(-0.08, 1.04, "(c)", transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9.5, fontweight="bold", clip_on=False)

    ax = axes[1, 1]
    profile_rows = []
    labels = []
    for case in (9, 10):
        for number, (i, j) in enumerate(WELLS[case]["prod"], 1):
            profile_rows.append(np.log10(np.maximum(perm[i, j, PRD[0]:PRD[1]], 1e-3)))
            labels.append(f"Case {DISPLAY_CASE[case]}--P{number}")
    profiles = np.asarray(profile_rows)
    permeability_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "permeability_blue_gray",
        ["#F5F7F8", "#DCE5E9", "#B9CBD4", "#91AAB8", "#6F8C9E", "#4D6B80", "#2F4D63"],
    )
    im = ax.imshow(profiles, aspect="auto", cmap=permeability_cmap,
                   vmin=-2.5, vmax=1, interpolation="nearest")
    ax.axhline(4.5, color="white", lw=2.2)
    ax.axhline(4.5, color="#37434A", lw=0.65)
    ax.set_title("Layer-wise permeability at production wells", fontsize=9.2, pad=4)
    ax.set_xlabel("Production layer (K1--K15)")
    ax.set_xticks(np.arange(15), np.arange(1, 16), fontsize=7)
    ax.set_yticks(np.arange(10), labels, fontsize=7)
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.text(-0.10, 1.04, "(d)", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, fontweight="bold", color="#20252A", clip_on=False)
    cax2 = fig.add_axes([0.92, 0.105, 0.017, 0.34])
    cbar2 = fig.colorbar(im, cax=cax2, extend="neither")
    cbar2.set_label(r"$\log_{10} k$ (mD)" + "\n(raw CMG permeability)", fontsize=8.5)
    cbar2.ax.tick_params(labelsize=8)
    for ax0 in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        ax0.tick_params(labelsize=8, width=0.7, length=3)
        for spine in ax0.spines.values():
            spine.set_linewidth(0.8)
    for ext in ("png", "pdf", "svg"):
        target = FIGURE_OUT / f"case9_case10_F1_geological_mechanism.{ext}"
        fig.savefig(target, dpi=900 if ext == "png" else None, bbox_inches="tight", transparent=True)
        fig.savefig(OUT / target.name, dpi=900 if ext == "png" else None, bbox_inches="tight", transparent=True)
    plt.close(fig)


def parse_dynamic(case: int, path: Path) -> tuple[dict, pd.DataFrame]:
    sr3 = cmg_runner.parse_sr3(str(path))
    pavg = np.asarray(sr3["pavg"], dtype=float)
    t = np.asarray(sr3["time_days_sp"], dtype=float)
    peak = int(np.nanargmax(pavg))
    prod_names = sorted(x for x in sr3["watvolsc"] if x.startswith("PRD"))
    prod_final = np.asarray([sr3["watvolsc"][x][-1] for x in prod_names], dtype=float)
    inj_names = sorted(x for x in sr3["bhp"] if x.startswith("INJ"))
    bhp = np.asarray([np.nanmax(sr3["bhp"][x]) for x in inj_names], dtype=float)
    gas = np.asarray([sr3["gasvolsc"][x][-1] for x in sorted(sr3["gasvolsc"]) if x.startswith("INJ")], dtype=float)
    f3 = objectives.compute_F3(sr3.get("pressure_heterogeneity_injend_std_mpa"), sr3.get("pressure_heterogeneity_injend_mean_mpa"))
    summary = {
        "case": case,
        "F1": objectives.compute_F1(pavg),
        "F2_km2": sr3.get("plume_area_sg0p1_km2"),
        "F3": f3,
        "pavg_initial_kpa": float(pavg[0]),
        "pavg_peak_kpa": float(pavg[peak]),
        "pavg_final_kpa": float(pavg[-1]),
        "pavg_peak_day": float(t[peak]),
        "pavg_peak_year": float(t[peak] / 365.25),
        "pressure_mean_mpa": sr3.get("pressure_heterogeneity_injend_mean_mpa"),
        "pressure_std_mpa": sr3.get("pressure_heterogeneity_injend_std_mpa"),
        "total_brine_production_m3": float(prod_final.sum()),
        "total_injected_gas_surface_m3": float(gas.sum()),
        "max_inj_bhp_kpa": float(bhp.max()),
        "prod_final_m3": {n: float(v) for n, v in zip(prod_names, prod_final)},
        "inj_bhp_max_kpa": {n: float(v) for n, v in zip(inj_names, bhp)},
        "inj_gas_surface_m3": {n: float(v) for n, v in zip(sorted(sr3["gasvolsc"]), gas)},
    }
    ts = pd.DataFrame({"time_days": t, "time_years": t / 365.25, "pavg_kpa": pavg})
    ts["case"] = case
    ts["F1_relative_to_initial"] = pavg / pavg[0] - 1.0
    well_time = np.asarray(sr3["time_days"], dtype=float)
    brine_series = []
    for name in prod_names:
        values = np.asarray(sr3["watvolsc"][name], dtype=float)
        source_time = well_time[-len(values):]
        brine_series.append(np.interp(t, source_time, values, left=values[0], right=values[-1]))
    ts["brine_total_m3"] = np.sum(brine_series, axis=0)
    return summary, ts


def main() -> None:
    seeds = dict(np.load(ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz", allow_pickle=True))
    prop_dir = ROOT / "property_CMG"
    perm = load_property(prop_dir / "property_260421_all_perm.txt")
    poro = load_property(prop_dir / "property_260421_all_poro.txt")
    thick = np.asarray(seeds["thick_3d"], dtype=float)
    sand = np.asarray(np.load(ROOT / r"_organized_active_work_20260816\04_verified_sandbody_analysis\verified_sandbody_grid_268x203x200.npz", allow_pickle=True)["sand_mask_kij"], dtype=bool)
    objects = np.asarray(np.load(ROOT / r"_organized_active_work_20260816\04_verified_sandbody_analysis\verified_sandbody_grid_268x203x200.npz", allow_pickle=True)["object_id_kij"], dtype=np.int64)
    # Raw CMG properties are I,J,K; verified sand-body fields are K,I,J.
    layout_rows = []
    topology_rows = []
    dynamic_summaries = []
    time_series = []
    pressure_maps = {}
    for case, path in CASES.items():
        dynamic, ts = parse_dynamic(case, path)
        dynamic_summaries.append(dynamic)
        time_series.append(ts)
        pressure_maps[case], dynamic["pressure_map_day"] = pressure_buildup_map(path, TARGET_DAY)
        dynamic.update(pressure_field_metrics(pressure_maps[case]))
        inj_xy = physical_xy(seeds, WELLS[case]["inj"])
        prod_xy = physical_xy(seeds, WELLS[case]["prod"])
        prod_vol = np.asarray([dynamic["prod_final_m3"][f"PRD-{n}"] for n in range(1, 6)])
        topo = {"case": case, "group": "injection", **spatial_metrics(inj_xy), **cross_group_metrics(inj_xy, prod_xy, prod_vol)}
        topology_rows.append(topo)
        for group, wells in (("inj", WELLS[case]["inj"]), ("prod", WELLS[case]["prod"])):
            interval = INJ if group == "inj" else PRD
            for number, (i, j) in enumerate(wells, 1):
                m = column_metrics(i, j, interval, perm, poro, thick, sand, objects)
                brine = dynamic["prod_final_m3"].get(f"PRD-{number}", np.nan) if group == "prod" else np.nan
                layout_rows.append({"case": case, "group": group, "well": f"{group.upper()}-{number}", "i_1based": int(i + 1), "j_1based": int(j + 1), "cumulative_brine_m3": brine, **m})
    summary = pd.DataFrame(dynamic_summaries)
    ts = pd.concat(time_series, ignore_index=True)
    wells = pd.DataFrame(layout_rows)
    topo = pd.DataFrame(topology_rows)
    for case in (9, 10):
        inj_xy = physical_xy(seeds, WELLS[case]["inj"])
        prod_xy = physical_xy(seeds, WELLS[case]["prod"])
        distance = np.linalg.norm(inj_xy[:, None, :] - prod_xy[None, :, :], axis=2)
        prod_rows = wells[(wells.case == case) & (wells.group == "prod")].sort_values("well")
        prod_kh = prod_rows.kh_mD_m.to_numpy(float)
        prod_q = prod_rows.cumulative_brine_m3.to_numpy(float)
        idx = topo.case == case
        topo.loc[idx, "kh_distance_relief_index"] = np.mean(np.sum(prod_kh[None, :] / distance, axis=1))
        topo.loc[idx, "production_distance_relief_index"] = np.mean(np.sum(prod_q[None, :] / distance, axis=1))
    group_static = wells.groupby(["case", "group"]).agg(
        net_to_gross_mean=("net_to_gross", "mean"),
        net_thickness_sum_m=("net_thickness_m", "sum"),
        kh_sum_mD_m=("kh_mD_m", "sum"),
        kh_mean_mD=("kh_mean_mD", "mean"),
        porosity_mean=("poro_h_weighted", "mean"),
        longest_lowk_run_mean_m=("lowk_longest_run_m", "mean"),
        object_count_mean=("object_count", "mean"),
    ).reset_index()
    balance_rows = []
    for case in (9, 10):
        inj_row = group_static[(group_static.case == case) & (group_static.group == "inj")].iloc[0]
        prod_row = group_static[(group_static.case == case) & (group_static.group == "prod")].iloc[0]
        prod_values = np.asarray([next(x for x in dynamic_summaries if x["case"] == case)["prod_final_m3"][f"PRD-{n}"] for n in range(1, 6)])
        shares = prod_values / prod_values.sum()
        balance_rows.append(
            {
                "case": case,
                "inj_kh_sum_mD_m": inj_row.kh_sum_mD_m,
                "prod_kh_sum_mD_m": prod_row.kh_sum_mD_m,
                "source_sink_kh_imbalance": inj_row.kh_sum_mD_m / prod_row.kh_sum_mD_m,
                "effective_producer_count": 1.0 / np.sum(shares ** 2),
                "normalized_production_entropy": -np.sum(shares * np.log(shares)) / np.log(5.0),
            }
        )
    balance = pd.DataFrame(balance_rows)
    producer_rows = wells[wells.group == "prod"].copy()
    correlation_rows = []
    for feature in ("net_to_gross", "net_thickness_m", "kh_mD_m", "kh_mean_mD",
                    "poro_h_weighted", "lowk_longest_run_m", "lowk_run_count"):
        correlation_rows.append({
            "feature": feature,
            "pearson_r": producer_rows[feature].corr(producer_rows.cumulative_brine_m3, method="pearson"),
            "spearman_rho": producer_rows[feature].corr(producer_rows.cumulative_brine_m3, method="spearman"),
            "n_wells": len(producer_rows),
        })
    producer_correlations = pd.DataFrame(correlation_rows)
    sand_pair_rows = []
    for case in (9, 10):
        inj_rows = wells[(wells.case == case) & (wells.group == "inj")].sort_values("well")
        xy = physical_xy(seeds, WELLS[case]["inj"])
        object_sets = []
        for value in inj_rows.object_ids:
            ids = {int(x) for x in str(value).split(";") if x}
            object_sets.append(ids)
        for a in range(len(object_sets)):
            for b in range(a + 1, len(object_sets)):
                shared = len(object_sets[a] & object_sets[b])
                union = len(object_sets[a] | object_sets[b])
                sand_pair_rows.append(
                    {
                        "case": case,
                        "well_a": f"I{a + 1}",
                        "well_b": f"I{b + 1}",
                        "distance_m": float(np.linalg.norm(xy[a] - xy[b])),
                        "shared_sand_objects": shared,
                        "sand_object_jaccard": shared / union if union else 0.0,
                    }
                )
    sand_pair_metrics = pd.DataFrame(sand_pair_rows)
    summary.to_json(OUT / "case9_case10_dynamic_summary.json", orient="records", indent=2)
    summary.drop(columns=[c for c in summary if isinstance(summary[c].iloc[0], dict)], errors="ignore").to_csv(OUT / "case9_case10_dynamic_summary.csv", index=False)
    wells.drop(columns=["object_ids"]).to_csv(OUT / "case9_case10_well_petrophysics_architecture.csv", index=False)
    group_static.to_csv(OUT / "case9_case10_group_static_summary.csv", index=False)
    balance.to_csv(OUT / "case9_case10_source_sink_balance.csv", index=False)
    topo.to_csv(OUT / "case9_case10_layout_topology.csv", index=False)
    producer_correlations.to_csv(OUT / "case9_case10_producer_correlations.csv", index=False)
    sand_pair_metrics.to_csv(OUT / "case9_case10_injection_sand_pair_metrics.csv", index=False)
    ts.to_csv(OUT / "case9_case10_pressure_brine_timeseries.csv", index=False)
    with open(OUT / "case9_case10_analysis_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"cases": {str(k): str(v) for k, v in CASES.items()}, "intervals_zero_based": {"inj": INJ, "prod": PRD}, "low_k_threshold_mD": 4.0, "notes": "Static petrophysics are extracted from CMG property files; sand objects are used as geometric descriptors, not as hydraulic-isolation labels."}, f, indent=2)
    prod_ntg = local_interval_map(PRD, thick, sand)
    plot_dynamic_mechanism(seeds, summary, ts, pressure_maps)
    plot_geological_mechanism(seeds, prod_ntg, wells, dynamic_summaries, perm)
    print(summary[["case", "F1", "F2_km2", "F3", "total_brine_production_m3", "pavg_peak_kpa", "pavg_peak_year"]].to_string(index=False))
    print(wells.groupby(["case", "group"])[["net_to_gross", "kh_mean_mD", "poro_h_weighted", "lowk_longest_run_m", "lowk_run_count", "top_to_bottom_kh_ratio", "object_count"]].mean().to_string())
    print(balance.to_string(index=False))
    print(topo.to_string(index=False))


if __name__ == "__main__":
    main()
