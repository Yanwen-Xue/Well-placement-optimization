from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

NI, NJ, NK = 268, 203, 200
INJ_K_START, INJ_K_END = 41, 80
PRD_K_START, PRD_K_END = 1, 15


def cell_edges_from_centers_2d(xc: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(xc, dtype=float)
    y = np.asarray(yc, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("center grids must be two-dimensional and have the same shape")

    def pad_edges(a: np.ndarray) -> np.ndarray:
        p = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=float)
        p[1:-1, 1:-1] = a
        p[0, 1:-1] = 2.0 * a[0, :] - a[1, :]
        p[-1, 1:-1] = 2.0 * a[-1, :] - a[-2, :]
        p[1:-1, 0] = 2.0 * a[:, 0] - a[:, 1]
        p[1:-1, -1] = 2.0 * a[:, -1] - a[:, -2]
        p[0, 0] = 2.0 * p[0, 1] - p[0, 2]
        p[0, -1] = 2.0 * p[0, -2] - p[0, -3]
        p[-1, 0] = 2.0 * p[-1, 1] - p[-1, 2]
        p[-1, -1] = 2.0 * p[-1, -2] - p[-1, -3]
        return 0.25 * (p[:-1, :-1] + p[1:, :-1] + p[:-1, 1:] + p[1:, 1:])

    return pad_edges(x), pad_edges(y)


def truncated_cmap(name: str, lo: float = 0.03, hi: float = 0.92):
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(name)
    colors = base(np.linspace(lo, hi, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_{lo:.2f}_{hi:.2f}", colors)
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    return cmap


def robust_range(arr: np.ndarray, valid: np.ndarray, qlo: float = 3.0, qhi: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(arr, dtype=float)[valid & np.isfinite(arr)]
    vals = vals[vals > 0.0]
    if vals.size < 10:
        return 0.0, 1.0
    lo = float(np.percentile(vals, qlo))
    hi = float(np.percentile(vals, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def soft_contours(ax, arr: np.ndarray, x: np.ndarray, z: np.ndarray, valid: np.ndarray, color: str) -> None:
    vals = arr[valid & np.isfinite(arr)]
    vals = vals[vals > 0.0]
    if vals.size < 10:
        return
    lo = float(np.percentile(vals, 15.0))
    hi = float(np.percentile(vals, 96.0))
    if hi <= lo:
        return
    ax.contour(
        x.T,
        z.T,
        np.where(valid, arr, np.nan).T,
        levels=np.linspace(lo, hi, 7),
        colors=[color],
        linewidths=0.48,
        alpha=0.22,
    )


def get_section(
    field: np.ndarray,
    valid: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    zmid: np.ndarray,
    well_i: int,
    well_j: int,
    direction: str,
    k_start: int,
    k_end: int,
) -> dict:
    i = int(np.clip(round(well_i), 0, NI - 1))
    j = int(np.clip(round(well_j), 0, NJ - 1))
    sl = slice(max(0, k_start - 1), min(NK, k_end))
    if direction == "xz":
        return {
            "x": cx[:, j, sl] / 1000.0,
            "z": zmid[:, j, sl],
            "data": field[:, j, sl],
            "valid": valid[:, j, sl],
            "axis_label": "X (km)",
            "well_axis_value": float(cx[i, j, k_start - 1]) / 1000.0,
            "section_label": f"J={j}",
        }
    if direction == "yz":
        return {
            "x": cy[i, :, sl] / 1000.0,
            "z": zmid[i, :, sl],
            "data": field[i, :, sl],
            "valid": valid[i, :, sl],
            "axis_label": "Y (km)",
            "well_axis_value": float(cy[i, j, k_start - 1]) / 1000.0,
            "section_label": f"I={i}",
        }
    raise ValueError(direction)


def plot_one(
    section: dict,
    out_path: Path,
    title: str,
    cmap_name: str,
    contour_color: str,
    target_k_range: tuple[int, int],
    vertical_exaggeration: float = 20.0,
) -> None:
    data = np.asarray(section["data"], dtype=float)
    valid = np.asarray(section["valid"], dtype=bool) & np.isfinite(data)
    smoothed = gaussian_filter(data, sigma=(6.0, 2.0))
    smoothed = np.where(valid, smoothed, np.nan)

    x = np.asarray(section["x"], dtype=float)
    z = np.asarray(section["z"], dtype=float)
    z_plot = z * vertical_exaggeration
    xedge, zedge = cell_edges_from_centers_2d(x, z)
    zedge_plot = zedge * vertical_exaggeration
    vmin, vmax = robust_range(smoothed, valid)

    fig, ax = plt.subplots(figsize=(3.45, 2.35), dpi=600, constrained_layout=True)
    im = ax.pcolormesh(
        xedge.T,
        zedge_plot.T,
        smoothed.T,
        cmap=truncated_cmap(cmap_name),
        shading="flat",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    soft_contours(ax, smoothed, x, z_plot, valid, contour_color)
    ax.contour(
        x.T,
        z_plot.T,
        valid.T.astype(float),
        levels=[0.5],
        colors=["#3f3f3f"],
        linewidths=0.65,
        alpha=0.95,
    )
    ax.axvline(section["well_axis_value"], color="#1f1f1f", linewidth=0.85, linestyle="--", alpha=0.8)

    k0, k1 = target_k_range
    kk = np.arange(k0 - 1, k1)
    target_mask = np.zeros_like(valid, dtype=bool)
    target_mask[:, (k0 - 1):(k1)] = True
    target_vals = z_plot[target_mask & np.isfinite(z_plot)]
    if target_vals.size > 0:
        ax.axhspan(float(np.nanmin(target_vals)), float(np.nanmax(target_vals)), color="#111111", alpha=0.055, linewidth=0)

    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(direction="out", length=2.8, width=0.65, labelsize=7.8, pad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#404040")
    ax.set_title(title, fontsize=8.8, fontweight="normal", pad=2.2)
    ax.set_xlabel(section["axis_label"], fontsize=8.4, labelpad=1.8)
    ax.set_ylabel("TVDSS (m)", fontsize=8.4, labelpad=1.8)
    ticks = ax.get_yticks()
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t / vertical_exaggeration:.0f}" for t in ticks])
    ax.invert_yaxis()

    good_xy = valid & np.isfinite(x) & np.isfinite(z_plot)
    if np.any(good_xy):
        xmin = float(np.nanmin(x[good_xy]))
        xmax = float(np.nanmax(x[good_xy]))
        zmin = float(np.nanmin(z_plot[good_xy]))
        zmax = float(np.nanmax(z_plot[good_xy]))
        xpad = max((xmax - xmin) * 0.025, 0.02)
        zpad = max((zmax - zmin) * 0.045, 25.0)
        ax.set_xlim(xmin - xpad, xmax + xpad)
        ax.set_ylim(zmax + zpad, zmin - zpad)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.018, aspect=23)
    cbar.set_label("Normalized static prescreen score", fontsize=7.8)
    ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{v:.2f}" for v in ticks])
    cbar.ax.tick_params(labelsize=7.0, length=2.0, width=0.55)
    cbar.outline.set_linewidth(0.55)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot well-conditioned XZ/YZ prescreen sections.")
    parser.add_argument("--seeds", default=r"F:\CODE1\opt_seeds.npz")
    parser.add_argument("--static", default=r"F:\CODE1\prescreen_maps\static_sideview_fields_no_pressure_threshold.npz")
    parser.add_argument("--out-dir", default=r"C:\Users\admin\Desktop\paper_figure\writen\figure")
    parser.add_argument("--inj-i", type=int, default=105)
    parser.add_argument("--inj-j", type=int, default=98)
    parser.add_argument("--prod-i", type=int, default=104)
    parser.add_argument("--prod-j", type=int, default=70)
    parser.add_argument("--k-start", type=int, default=1)
    parser.add_argument("--k-end", type=int, default=120)
    args = parser.parse_args()

    seeds = np.load(args.seeds, allow_pickle=False)
    static = np.load(args.static, allow_pickle=False)
    cx = np.asarray(seeds["cx_3d"], dtype=float)
    cy = np.asarray(seeds["cy_3d"], dtype=float)
    zmid = np.asarray(seeds["top_3d"], dtype=float) + 0.5 * np.asarray(seeds["thick_3d"], dtype=float)
    valid = np.asarray(static["static_valid_3d"], dtype=bool)
    inj = np.asarray(static["static_inj_3d"], dtype=float)
    prod = np.asarray(static["static_prod_3d"], dtype=float)

    out_dir = Path(args.out_dir)
    specs = [
        ("injection", "xz", inj, args.inj_i, args.inj_j, "Injection-side XZ section", "YlOrRd", "#6b1f14", (INJ_K_START, INJ_K_END)),
        ("injection", "yz", inj, args.inj_i, args.inj_j, "Injection-side YZ section", "YlOrRd", "#6b1f14", (INJ_K_START, INJ_K_END)),
        ("production", "xz", prod, args.prod_i, args.prod_j, "Production-side XZ section", "Blues", "#15374a", (PRD_K_START, PRD_K_END)),
        ("production", "yz", prod, args.prod_i, args.prod_j, "Production-side YZ section", "Blues", "#15374a", (PRD_K_START, PRD_K_END)),
    ]

    outputs = {}
    for group, direction, field, wi, wj, title, cmap, contour, target_k in specs:
        sec = get_section(field, valid, cx, cy, zmid, wi, wj, direction, args.k_start, args.k_end)
        stem = f"prescreen_{group}_{direction}_well_I{wi}_J{wj}_sigma6"
        out_path = out_dir / f"{stem}.png"
        plot_one(sec, out_path, f"{title}; {sec['section_label']}", cmap, contour, target_k)
        outputs[stem] = str(out_path)

    summary = {
        "definition": "well-conditioned static XZ/YZ prescreen sections; sigma=6 contours only; no pressure-threshold gating",
        "injection_well_feature_indices": [int(args.inj_i), int(args.inj_j)],
        "production_well_feature_indices": [int(args.prod_i), int(args.prod_j)],
        "k_plot_range": [int(args.k_start), int(args.k_end)],
        "injection_interval": [INJ_K_START, INJ_K_END],
        "production_interval": [PRD_K_START, PRD_K_END],
        "outputs": outputs,
    }
    summary_path = out_dir / "prescreen_well_conditioned_sections_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
