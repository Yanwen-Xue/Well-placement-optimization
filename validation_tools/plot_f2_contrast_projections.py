from __future__ import annotations

import json
import re
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


ROOT = Path(r"F:\CODE1")
OUT = Path(r"C:\Users\admin\Desktop\paper_figure\writen\figure")
SEEDS = np.load(ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz", allow_pickle=True)
CASE_ROOT = ROOT / "_organized_active_work_20260817" / "05_nsga3_current_0_683_20260817" / "final_analysis_cases_18"
COMPARE_ROOT = CASE_ROOT / "compare"
DISPLAY_CASE = {7: 3, 8: 4, 255: 5, 312: 6}
NI, NJ, NK = 268, 203, 200
INJ_K0, INJ_K1 = 40, 80
TARGET_DAY = 30.0 * 365.25

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "none",
        "figure.facecolor": "none",
    }
)


def sr3_path(case_number: int) -> Path:
    directory = {
        7: "case3_F2_clustered_low",
        8: "case4_F2_clustered_high",
        255: "case5_F2_distributed_high",
        312: "case6_F2_distributed_low",
    }.get(case_number)
    if directory is None:
        raise ValueError(case_number)
    return COMPARE_ROOT / directory / "run_001" / "opt_001.sr3"


def wells(case_number: int) -> tuple[np.ndarray, np.ndarray]:
    if case_number in {7, 8}:
        manifest = json.loads((CASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
        item = next(x for x in manifest["cases"] if x["case_id"] == case_number)
        return np.asarray(item["inj_wells_0based"], dtype=int), np.asarray(item["prod_wells_0based"], dtype=int)
    dat_path = sr3_path(case_number).with_suffix(".dat")
    text = dat_path.read_text(encoding="utf-8", errors="replace")
    result = {"INJ": [], "PRD": []}
    for prefix in ("INJ", "PRD"):
        for number in range(1, 6):
            match = re.search(
                rf"PERF\s+GEOA\s+'{prefix}-{number}'\s*\n"
                rf"\*\*[^\n]*\n\s*(\d+)\s+(\d+)\s+\d+",
                text,
            )
            if match is None:
                raise ValueError(f"Cannot find {prefix}-{number} coordinates in {dat_path}")
            result[prefix].append((int(match.group(1)) - 1, int(match.group(2)) - 1))
    return np.asarray(result["INJ"], dtype=int), np.asarray(result["PRD"], dtype=int)


def projected_sg(path: Path) -> tuple[np.ndarray, float]:
    with h5py.File(path, "r") as f:
        days = np.asarray(f["General/MasterTimeTable"]["Offset in days"], dtype=float)
        groups = sorted(
            (name for name in f["SpatialProperties"].keys() if name.isdigit()),
            key=lambda name: int(name),
        )
        group_days = days[np.asarray([int(name) for name in groups], dtype=int)]
        group = groups[int(np.argmin(np.abs(group_days - TARGET_DAY)))]
        grid = f["SpatialProperties/000000/GRID"]
        ipstcs = np.asarray(grid["IPSTCS"], dtype=np.int64) - 1
        pv = np.asarray(grid["BLOCKPVOL"], dtype=float)
        full_i = ipstcs % NI
        full_j = (ipstcs // NI) % NJ
        full_k = ipstcs // (NI * NJ)
        valid = (
            (full_k >= INJ_K0)
            & (full_k < INJ_K1)
            & np.isfinite(pv)
            & (pv > 0.0)
        )
        sg = np.asarray(f[f"SpatialProperties/{group}/SG"], dtype=float)
        projected = np.full((NI, NJ), np.nan, dtype=float)
        for k in range(INJ_K0, INJ_K1):
            pick = valid & (full_k == k)
            if np.any(pick):
                ii, jj = full_i[pick], full_j[pick]
                values = sg[pick]
                old = projected[ii, jj]
                projected[ii, jj] = np.fmax(np.where(np.isfinite(old), old, -np.inf), values)
        return projected, float(days[int(group)])


def physical_well_xy(well_array: np.ndarray) -> np.ndarray:
    layer = 40
    cx = SEEDS["cx_3d"][:, :, layer]
    cy = SEEDS["cy_3d"][:, :, layer]
    return np.asarray([[cx[i, j], cy[i, j]] for i, j in well_array], dtype=float)


def label_wells(
    ax,
    points: np.ndarray,
    prefix: str,
    color: str,
    occupied: list[np.ndarray],
    used_label_points: list[np.ndarray],
) -> None:
    candidates = [(11, 11), (11, -11), (-11, 11), (-11, -11), (13, 0), (-13, 0), (0, 13), (0, -13)]
    all_points = np.vstack(occupied) if occupied else np.empty((0, 2))
    for number, point in enumerate(points, start=1):
        best = None
        best_score = -np.inf
        for dx, dy in candidates:
            # Approximate the offset in data units for a label-clearance score.
            trial = point + np.array([dx * 68.0, dy * 55.0])
            other_wells = all_points[np.any(np.abs(all_points - point) > 1e-8, axis=1)] if all_points.size else all_points
            well_clearance = np.min(np.linalg.norm(other_wells - trial, axis=1)) if other_wells.size else 1.0e9
            label_clearance = np.min(np.linalg.norm(np.asarray(used_label_points) - trial, axis=1)) if used_label_points else 1.0e9
            edge_clearance = min(trial[0], 15500.0 - trial[0], trial[1] - 4800.0, 22300.0 - trial[1])
            score = min(well_clearance, label_clearance) + 0.25 * edge_clearance
            if score > best_score:
                best_score = score
                best = (dx, dy, trial)
        dx, dy, trial = best
        used_label_points.append(trial)
        ha = "left" if dx > 0 else "right" if dx < 0 else "center"
        va = "bottom" if dy > 0 else "top" if dy < 0 else "center"
        ax.annotate(
            f"{prefix}{number}",
            xy=point,
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7.0,
            color=color,
            fontweight="bold",
            zorder=8,
        )


def plot_pair(left_case: int, right_case: int, output_stem: str) -> None:
    left_field, left_day = projected_sg(sr3_path(left_case))
    right_field, right_day = projected_sg(sr3_path(right_case))
    left_inj, left_prod = wells(left_case)
    right_inj, right_prod = wells(right_case)
    left_f2 = float(np.count_nonzero(np.isfinite(left_field) & (left_field >= 0.10)) * 0.0025)
    right_f2 = float(np.count_nonzero(np.isfinite(right_field) & (right_field >= 0.10)) * 0.0025)

    cx = SEEDS["cx_3d"][:, :, 40]
    cy = SEEDS["cy_3d"][:, :, 40]
    bound = np.asarray(SEEDS["bound_mask"], dtype=bool)
    outer = np.array([
        [cx[0, 0], cy[0, 0]], [cx[-1, 0], cy[-1, 0]],
        [cx[-1, -1], cy[-1, -1]], [cx[0, -1], cy[0, -1]],
    ])
    x_flat, y_flat = cx.ravel(), cy.ravel()
    valid_flat = np.isfinite(x_flat) & np.isfinite(y_flat)
    # Keep the common monitoring scale anchored at zero even though the
    # displayed plume mask starts at the 0.10 saturation threshold.
    norm = Normalize(vmin=0.0, vmax=0.8, clip=True)
    cmap = mpl.colormaps["Blues"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.55), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.895, bottom=0.17, top=0.84, wspace=0.18)
    specs = [
        (axes[0], left_case, left_field, left_f2, left_inj, left_prod),
        (axes[1], right_case, right_field, right_f2, right_inj, right_prod),
    ]
    for ax, case, field, f2, inj, prod in specs:
        ax.fill(outer[:, 0], outer[:, 1], color="#f3f7fb", alpha=1.0, zorder=0)
        display = np.ma.masked_where(~(np.isfinite(field) & (field >= 0.10)), field)
        plume = ax.pcolormesh(cx, cy, display, cmap=cmap, norm=norm,
                               shading="auto", alpha=1.0, rasterized=False,
                               linewidth=0.0, edgecolors="none", antialiased=False,
                               zorder=2)
        ax.contour(cx, cy, np.where(np.isfinite(field), field, np.nan), levels=[0.10], colors=["#1f2933"], linewidths=0.40, zorder=4)
        ax.contour(cx, cy, bound.astype(float), levels=[0.5], colors=["#8a8f94"], linewidths=0.9, linestyles="--", zorder=3)
        inj_xy = physical_well_xy(inj)
        prod_xy = physical_well_xy(prod)
        ax.scatter(inj_xy[:, 0], inj_xy[:, 1], s=54, c="#c64040", edgecolors="white", linewidths=0.9, zorder=7, label="Injection well")
        ax.scatter(prod_xy[:, 0], prod_xy[:, 1], s=54, c="#2d6b54", marker="s", edgecolors="white", linewidths=0.9, zorder=7, label="Production well")
        used_label_points: list[np.ndarray] = []
        label_wells(ax, inj_xy, "I", "#8f2424", [inj_xy, prod_xy], used_label_points)
        label_wells(ax, prod_xy, "P", "#20543f", [inj_xy, prod_xy], used_label_points)
        ax.set_title(f"Case {DISPLAY_CASE[case]}  |  F₂ = {f2:.3f} km²", fontsize=9.2, pad=7.0)
        ax.set_xlim(-350, 15700)
        ax.set_ylim(4700, 22300)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)", fontsize=8.5)
        ax.set_ylabel("Y (m)", fontsize=8.5)
        ax.tick_params(labelsize=8.0, width=0.7, length=3.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.2, bbox_to_anchor=(0.48, 0.035), handletextpad=0.5, columnspacing=1.4)
    cax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Maximum CO₂ gas saturation, max(Sg)", fontsize=8.5, labelpad=7)
    cbar.ax.tick_params(labelsize=8.0, width=0.7, length=3.5)
    fig.suptitle("Thirty-year CO₂ plume footprint in the inclined reservoir", fontsize=10.0, y=0.94)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{output_stem}.png", dpi=900, bbox_inches="tight", transparent=True)
    fig.savefig(OUT / f"{output_stem}.pdf", bbox_inches="tight", transparent=True)
    fig.savefig(OUT / f"{output_stem}.svg", bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(json.dumps({"output": output_stem, "cases": [left_case, right_case], "F2_km2": [left_f2, right_f2], "days": [left_day, right_day]}, ensure_ascii=False))


if __name__ == "__main__":
    plot_pair(7, 8, "case7_case8_F2_max_vertical_projection")
    plot_pair(255, 312, "case255_case312_F2_max_vertical_projection")
