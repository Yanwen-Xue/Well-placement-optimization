from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import matplotlib as mpl
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm, TwoSlopeNorm
from matplotlib.patches import Polygon
from scipy.ndimage import label
from scipy.spatial import ConvexHull

ROOT = Path(r"F:\CODE1")
CASE_ROOT = ROOT / r"_organized_active_work_20260817\05_nsga3_current_0_683_20260817\final_analysis_cases_18"
RUN_ROOT = CASE_ROOT / r"machine_1_final_analysis_09\lhs_doe"
COMPARE_ROOT = CASE_ROOT / "compare"
CASE_PATHS = {
    5: COMPARE_ROOT / r"case7_F3_low\run_001\opt_001.sr3",
    6: COMPARE_ROOT / r"case8_F3_high\run_001\opt_001.sr3",
}
OUT = CASE_ROOT / "case5_case6_F3_analysis_20260819"
OUT.mkdir(parents=True, exist_ok=True)

NI, NJ, NK = 268, 203, 200
INJ = (40, 80)
PRD = (0, 15)
TARGET_DAY = 30.0 * 365.25
CELL_AREA_KM2 = 0.0025

sys.path.insert(0, str(ROOT))
import cmg_runner  # noqa: E402
import objectives  # noqa: E402
from validation_tools.analyze_case9_case10_f1_20260818 import (  # noqa: E402
    column_metrics,
    cross_group_metrics,
    local_interval_map,
    physical_xy,
    spatial_metrics,
)

FIGURE_OUT = Path(r"C:\Users\admin\Desktop\paper_figure\writen\figure")
FIGURE_OUT.mkdir(parents=True, exist_ok=True)
DISPLAY_CASE = {5: 7, 6: 8}

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


def load_property(path: Path) -> np.ndarray:
    return cmg_runner._read_property_txt(str(path), NI, NJ, NK)


def read_target_pressure(sr3_path: Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    with h5py.File(sr3_path, "r") as f:
        days = np.asarray(f["General/MasterTimeTable"]["Offset in days"], dtype=float)
        groups = sorted((x for x in f["SpatialProperties"] if x.isdigit()), key=int)
        ids = np.asarray([int(x) for x in groups], dtype=int)
        group = groups[int(np.argmin(np.abs(days[ids] - TARGET_DAY)))]
        grid = f["SpatialProperties/000000/GRID"]
        ipstcs = np.asarray(grid["IPSTCS"], dtype=np.int64) - 1
        pv = np.asarray(grid["BLOCKPVOL"], dtype=float)
        i = ipstcs % NI
        j = (ipstcs // NI) % NJ
        k = ipstcs // (NI * NJ)
        p0 = np.asarray(f["SpatialProperties/000000/PRES"], dtype=float)
        pt = np.asarray(f[f"SpatialProperties/{group}/PRES"], dtype=float)
        valid = (
            (k >= INJ[0]) & (k < INJ[1]) & np.isfinite(pv) & (pv > 0)
            & np.isfinite(p0) & np.isfinite(pt)
        )
        dp = (pt - p0) / 1000.0
        i, j, k, pv, dp = i[valid], j[valid], k[valid], pv[valid], dp[valid]
        layer_mean = []
        layer_std = []
        layer_pv = []
        for layer_id in range(INJ[0], INJ[1]):
            mask = k == layer_id
            w = pv[mask]
            x = dp[mask]
            layer_pv.append(float(w.sum()))
            layer_mean.append(float(np.sum(w * x) / w.sum()) if w.sum() else np.nan)
            layer_std.append(float(np.sqrt(np.sum(w * (x - layer_mean[-1]) ** 2) / w.sum())) if w.sum() else np.nan)
        order = np.argsort(dp)
        cumulative = np.cumsum(pv[order]) / pv.sum()
        quantiles = {f"pressure_q{int(q * 100):02d}_mpa": float(np.interp(q, cumulative, dp[order])) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
        pressure_map = np.full((NI, NJ), np.nan, dtype=float)
        numerator = np.zeros((NI, NJ), dtype=float)
        denominator = np.zeros((NI, NJ), dtype=float)
        np.add.at(numerator, (i, j), dp * pv)
        np.add.at(denominator, (i, j), pv)
        np.divide(numerator, denominator, out=pressure_map, where=denominator > 0)
        structure = np.ones((3, 3), dtype=int)
        field = {
            "pressure_map_mpa": pressure_map,
            "layer_mean_mpa": np.asarray(layer_mean),
            "layer_std_mpa": np.asarray(layer_std),
            "pressure_cells_mpa": dp,
            "pressure_cell_pv": pv,
        }
        summary = {
            "pressure_day": float(days[int(group)]),
            "pressure_mean_mpa": float(np.sum(pv * dp) / pv.sum()),
            "pressure_std_mpa": float(np.sqrt(np.sum(pv * (dp - np.sum(pv * dp) / pv.sum()) ** 2) / pv.sum())),
            **quantiles,
            "layer_mean_min_mpa": float(np.nanmin(layer_mean)),
            "layer_mean_max_mpa": float(np.nanmax(layer_mean)),
            "layer_mean_range_mpa": float(np.nanmax(layer_mean) - np.nanmin(layer_mean)),
            "layer_std_mean_mpa": float(np.nanmean(layer_std)),
            "layer_std_max_mpa": float(np.nanmax(layer_std)),
        }
        for threshold in (0.25, 0.5, 1.0, 2.0):
            mask = np.isfinite(pressure_map) & (pressure_map >= threshold)
            labels, count = label(mask, structure=structure)
            sizes = np.bincount(labels.ravel())[1:] if count else np.asarray([])
            summary[f"pressure_area_ge_{str(threshold).replace('.', 'p')}_mpa_km2"] = float(mask.sum() * CELL_AREA_KM2)
            summary[f"pressure_components_ge_{str(threshold).replace('.', 'p')}_mpa"] = int(count)
            summary[f"pressure_largest_share_ge_{str(threshold).replace('.', 'p')}_mpa"] = float(sizes.max() / mask.sum()) if sizes.size and mask.sum() else 0.0
        return field, summary


def parse_dynamic(sr3_path: Path) -> dict[str, float]:
    sr3 = cmg_runner.parse_sr3(str(sr3_path))
    pavg = np.asarray(sr3["pavg"], dtype=float)
    prod_names = sorted(x for x in sr3["watvolsc"] if x.startswith("PRD"))
    gas_names = sorted(x for x in sr3["gasvolsc"] if x.startswith("INJ"))
    result = {
        "F1": float(objectives.compute_F1(pavg)),
        "F2_km2": float(sr3["plume_area_sg0p1_km2"]),
        "F3": float(sr3["pressure_heterogeneity_injend_std_mpa"] / sr3["pressure_heterogeneity_injend_mean_mpa"]),
        "pressure_mean_mpa": float(sr3["pressure_heterogeneity_injend_mean_mpa"]),
        "pressure_std_mpa": float(sr3["pressure_heterogeneity_injend_std_mpa"]),
        "total_brine_m3": float(sum(sr3["watvolsc"][x][-1] for x in prod_names)),
        "total_injected_gas_m3": float(sum(sr3["gasvolsc"][x][-1] for x in gas_names)),
    }
    result.update({f"{name}_brine_m3": float(sr3["watvolsc"][name][-1]) for name in prod_names})
    return result


def add_wells(ax, xy: np.ndarray, prefix: str, color: str, marker: str) -> None:
    ax.scatter(xy[:, 0], xy[:, 1], s=30, marker=marker, color=color,
               edgecolor="white", linewidth=1.0, zorder=6)
    for number, (x, y) in enumerate(xy, 1):
        ax.annotate(f"{prefix}{number}", (x, y), xytext=(5, 5), textcoords="offset points",
                    fontsize=7.2, color="#20252A", zorder=7)


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "svg", "png"):
        kwargs = {"dpi": 900} if suffix == "png" else {"dpi": 600}
        fig.savefig(FIGURE_OUT / f"{stem}.{suffix}", bbox_inches="tight",
                    transparent=True, **kwargs)
    plt.close(fig)


def plot_pressure_diagnostics(fields: dict[int, dict[str, np.ndarray]], seeds: dict[str, np.ndarray],
                              wells_by_case: dict[int, dict[str, np.ndarray]]) -> None:
    cx = seeds["cx_3d"][:, :, INJ[0]]
    cy = seeds["cy_3d"][:, :, INJ[0]]
    bound = np.asarray(seeds["bound_mask"], dtype=bool)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "pressure_nature_muted",
        ["#315C78", "#87AFC3", "#F3F1EB", "#DDA77B", "#A85B4B"],
    )
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=2.2)
    colors = {5: "#3D6684", 6: "#A96C5A"}
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7))
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.10, top=0.95, hspace=0.30, wspace=0.26)
    for ax, case in zip(axes[0], (5, 6)):
        mesh = ax.pcolormesh(
            cx, cy, fields[case]["pressure_map_mpa"], cmap=cmap, norm=norm,
            shading="auto", linewidth=0.0, edgecolors="none", antialiased=False,
        )
        mesh.set_rasterized(False)
        ax.contour(cx, cy, bound.astype(float), levels=[0.5], colors="#5E666D", linewidths=0.7, linestyles="--")
        add_wells(ax, physical_xy(seeds, wells_by_case[case]["inj"]), "I", "#3F718C", "o")
        add_wells(ax, physical_xy(seeds, wells_by_case[case]["prod"]), "P", "#7A7F83", "s")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"Case {DISPLAY_CASE[case]}: PV-weighted mean over K41--K80")
    cax = fig.add_axes([0.925, 0.555, 0.016, 0.36])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Pressure buildup (MPa)")
    cbar.ax.tick_params(labelsize=8)

    ax = axes[1, 0]
    layers = np.arange(INJ[0] + 1, INJ[1] + 1)
    for case in (5, 6):
        ax.plot(layers, fields[case]["layer_mean_mpa"], color=colors[case], lw=1.6,
                marker="o", markersize=2.6, markevery=4, label=f"Case {DISPLAY_CASE[case]}")
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("PV-weighted mean pressure buildup (MPa)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#D7DADD", linewidth=0.5)

    ax = axes[1, 1]
    for case in (5, 6):
        values = fields[case]["pressure_cells_mpa"]
        weights = fields[case]["pressure_cell_pv"]
        order = np.argsort(values)
        cdf = np.cumsum(weights[order]) / np.sum(weights)
        ax.plot(values[order], cdf, color=colors[case], lw=1.7, label=f"Case {DISPLAY_CASE[case]}")
    ax.axvline(0.0, color="#71777B", lw=0.8, ls="--")
    ax.set_xlabel("Pressure buildup (MPa)")
    ax.set_ylabel("Pore-volume fraction")
    ax.set_xlim(-0.6, 2.25)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(color="#D7DADD", linewidth=0.5)
    for label_text, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(-0.13, 1.045, label_text, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=9.5, fontweight="bold", clip_on=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
    save_figure(fig, "case5_case6_F3_pressure_diagnostics")


def plot_geological_context(seeds: dict[str, np.ndarray], wells_by_case: dict[int, dict[str, np.ndarray]],
                            thick: np.ndarray, sand: np.ndarray) -> None:
    cx = seeds["cx_3d"][:, :, INJ[0]]
    cy = seeds["cy_3d"][:, :, INJ[0]]
    bound = np.asarray(seeds["bound_mask"], dtype=bool)
    maps = {"inj": local_interval_map(INJ, thick, sand), "prod": local_interval_map(PRD, thick, sand)}
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "ntg_shared_soft_ochre",
        ["#FCFAF4", "#F6F0DF", "#EEE3C8", "#E4D3AA", "#D8C294", "#CBB17F", "#BDA16F"],
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.09, top=0.95, hspace=0.28, wspace=0.25)
    for row, group in enumerate(("inj", "prod")):
        for col, case in enumerate((5, 6)):
            ax = axes[row, col]
            mesh = ax.pcolormesh(
                cx, cy, maps[group], cmap=cmap,
                norm=PowerNorm(gamma=1.75, vmin=0.0, vmax=1.0), shading="auto",
                linewidth=0.0, edgecolors="none", antialiased=False,
            )
            mesh.set_rasterized(False)
            ax.contour(cx, cy, bound.astype(float), levels=[0.5], colors="#62686C", linewidths=0.7, linestyles="--")
            xy = physical_xy(seeds, wells_by_case[case][group])
            if len(xy) >= 3:
                hull = ConvexHull(xy)
                polygon = Polygon(xy[hull.vertices], closed=True, fill=False, lw=1.65,
                                  ls=(0, (4.0, 2.0)), edgecolor="#253B44", zorder=5)
                ax.add_patch(polygon)
            add_wells(ax, xy, "I" if group == "inj" else "P",
                      "#3F718C" if group == "inj" else "#7A7F83", "o" if group == "inj" else "s")
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            interval_text = "K41--K80" if group == "inj" else "K1--K15"
            ax.set_title(
                f"Case {DISPLAY_CASE[case]}: {'injection' if group == 'inj' else 'production'} interval {interval_text}"
            )
    cax = fig.add_axes([0.925, 0.18, 0.016, 0.64])
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label("Net-to-gross ratio")
    cbar.ax.tick_params(labelsize=8)
    for label_text, ax in zip(("(a)", "(b)", "(c)", "(d)"), axes.ravel()):
        ax.text(-0.13, 1.045, label_text, transform=ax.transAxes, va="bottom", ha="left",
                fontsize=9.5, fontweight="bold", clip_on=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.75)
    save_figure(fig, "case5_case6_F3_geological_context")


def main() -> None:
    manifest = json.loads((CASE_ROOT / "manifest.json").read_text(encoding="utf-8"))
    cases = {int(row["case_id"]): row for row in manifest["cases"] if int(row["case_id"]) in (5, 6)}
    seeds = dict(np.load(ROOT / "opt_seeds_0_683_current_mixed_active60_20260813.npz", allow_pickle=True))
    perm = load_property(ROOT / r"property_CMG\property_260421_all_perm.txt")
    poro = load_property(ROOT / r"property_CMG\property_260421_all_poro.txt")
    thick = np.asarray(seeds["thick_3d"], dtype=float)
    verified = np.load(ROOT / r"_organized_active_work_20260816\04_verified_sandbody_analysis\verified_sandbody_grid_268x203x200.npz", allow_pickle=True)
    sand = np.asarray(verified["sand_mask_kij"], dtype=bool)
    objects = np.asarray(verified["object_id_kij"], dtype=np.int64)
    rows = []
    group_rows = []
    layout_rows = []
    pressure_rows = []
    topology_rows = []
    fields = {}
    wells_by_case = {}
    for case, item in cases.items():
        wells = {"inj": np.asarray(item["inj_wells_0based"], dtype=int), "prod": np.asarray(item["prod_wells_0based"], dtype=int)}
        wells_by_case[case] = wells
        sr3_path = CASE_PATHS[case]
        dynamic = parse_dynamic(sr3_path)
        field, pressure = read_target_pressure(sr3_path)
        fields[case] = field
        rows.append({"case": case, **dynamic, **pressure})
        inj_xy = physical_xy(seeds, wells["inj"])
        prod_xy = physical_xy(seeds, wells["prod"])
        topology_rows.append({"case": case, "group": "inj", **spatial_metrics(inj_xy)})
        topology_rows.append({"case": case, "group": "prod", **spatial_metrics(prod_xy)})
        topology_rows.append({"case": case, "group": "cross", **cross_group_metrics(inj_xy, prod_xy, np.ones(5))})
        for group, coords, interval in (("inj", wells["inj"], INJ), ("prod", wells["prod"], PRD)):
            metrics = []
            for number, (i, j) in enumerate(coords, 1):
                metric = column_metrics(i, j, interval, perm, poro, thick, sand, objects)
                metric.update({"case": case, "group": group, "well": f"{group.upper()}-{number}", "i_1based": i + 1, "j_1based": j + 1})
                metrics.append(metric)
                layout_rows.append(metric)
            group_rows.append({"case": case, "group": group, **pd.DataFrame(metrics).select_dtypes(include=[np.number]).mean().to_dict(), "net_thickness_sum_m": float(sum(x["net_thickness_m"] for x in metrics)), "kh_sum_mD_m": float(sum(x["kh_mD_m"] for x in metrics)), "object_count_sum": float(sum(x["object_count"] for x in metrics))})
        for layer, mean, std in zip(range(INJ[0] + 1, INJ[1] + 1), field["layer_mean_mpa"], field["layer_std_mpa"]):
            pressure_rows.append({"case": case, "layer": layer, "pressure_mean_mpa": mean, "pressure_std_mpa": std})
    summary = pd.DataFrame(rows)
    groups = pd.DataFrame(group_rows)
    wells = pd.DataFrame(layout_rows)
    layers = pd.DataFrame(pressure_rows)
    topology = pd.DataFrame(topology_rows)
    summary.to_csv(OUT / "case5_case6_pressure_summary.csv", index=False)
    groups.to_csv(OUT / "case5_case6_geology_group_summary.csv", index=False)
    wells.to_csv(OUT / "case5_case6_well_geology.csv", index=False)
    layers.to_csv(OUT / "case5_case6_layer_pressure.csv", index=False)
    topology.to_csv(OUT / "case5_case6_layout_topology.csv", index=False)
    (OUT / "analysis_notes.txt").write_text(
        "F3 = pore-volume-weighted pressure standard deviation / pore-volume-weighted pressure mean at the 30-year output nearest to 10957 days.\n"
        "Pressure fields are restricted to K41-K80, matching the objective definition. Geological metrics use the verified sand mask and raw CMG permeability/porosity.\n",
        encoding="ascii",
    )
    plot_pressure_diagnostics(fields, seeds, wells_by_case)
    plot_geological_context(seeds, wells_by_case, thick, sand)
    print(summary.to_string(index=False))
    print(groups.to_string(index=False))


if __name__ == "__main__":
    main()
