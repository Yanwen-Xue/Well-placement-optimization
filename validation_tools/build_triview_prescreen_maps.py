from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

ROOT = Path(__file__).resolve().parent.parent
PROPERTY_DIR = ROOT / "property_CMG"
OUT_DIR = ROOT / "opt_runs_lhs_doe" / "triview_prescreen_20260813"

NI, NJ, NK = 268, 203, 200
DX_M = 50.0
DY_M = 50.0
PRD_K_START = 1
PRD_K_END = 15
INJ_K_START = 41
INJ_K_END = 80


def find_property_file(folder: Path, keyword: str) -> Path:
    matches = sorted(folder.glob(f"*{keyword}*.txt"))
    if not matches:
        raise FileNotFoundError(f"missing property file containing {keyword!r} under {folder}")
    return matches[0]


def read_cmg_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    data = np.zeros((ni, nj, nk), dtype=np.float32)
    current_k = None
    current_j = None
    values: list[float] = []

    def flush() -> None:
        nonlocal values, current_k, current_j
        if current_k is None or current_j is None or not values:
            return
        arr = np.asarray(values[:ni], dtype=np.float32)
        if arr.size == ni:
            data[:, current_j - 1, current_k - 1] = arr
        values = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("**") and "K =" in line and "J =" in line:
                flush()
                try:
                    parts = line.replace(",", " ").split()
                    current_k = int(parts[parts.index("K") + 2])
                    current_j = int(parts[parts.index("J") + 2])
                except Exception:
                    current_k = None
                    current_j = None
                continue
            if line.startswith("**") or line.startswith("RESULTS"):
                continue
            try:
                values.extend(float(v) for v in line.split())
            except ValueError:
                continue
    flush()
    return data


def read_property_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    vals: list[float] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip().split()
        if len(first) < 3:
            raise ValueError(f"unexpected property header in {path}")
        for raw in f:
            parts = raw.strip().split()
            if parts:
                vals.append(float(parts[0]))
    total = ni * nj * nk
    arr = np.asarray(vals[:total], dtype=np.float32)
    if arr.size != total:
        raise ValueError(f"{path} has {arr.size} values, expected {total}")
    return arr.reshape((ni, nj, nk), order="F")


def robust_norm(arr: np.ndarray, valid: np.ndarray | None = None, qlo: float = 2.0, qhi: float = 98.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(x) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(x))
    vals = x[mask]
    vals = vals[vals > 0]
    if vals.size < 10:
        return np.zeros_like(x, dtype=np.float32)
    lo = float(np.percentile(vals, qlo))
    hi = float(np.percentile(vals, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1e-12
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def neighbour_fraction(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sand = np.asarray(mask, dtype=np.float32)
    x_count = np.zeros_like(sand)
    y_count = np.zeros_like(sand)
    z_count = np.zeros_like(sand)
    x_den = np.zeros_like(sand)
    y_den = np.zeros_like(sand)
    z_den = np.zeros_like(sand)

    x_count[1:] += sand[:-1]
    x_den[1:] += 1
    x_count[:-1] += sand[1:]
    x_den[:-1] += 1

    y_count[:, 1:] += sand[:, :-1]
    y_den[:, 1:] += 1
    y_count[:, :-1] += sand[:, 1:]
    y_den[:, :-1] += 1

    z_count[:, :, 1:] += sand[:, :, :-1]
    z_den[:, :, 1:] += 1
    z_count[:, :, :-1] += sand[:, :, 1:]
    z_den[:, :, :-1] += 1

    conn_x = np.divide(x_count, np.maximum(x_den, 1.0), dtype=np.float32)
    conn_y = np.divide(y_count, np.maximum(y_den, 1.0), dtype=np.float32)
    conn_z = np.divide(z_count, np.maximum(z_den, 1.0), dtype=np.float32)
    return conn_x, conn_y, conn_z


def aggregate_layers(score: np.ndarray, axis: int) -> np.ndarray:
    mean = np.nanmean(score, axis=axis)
    p90 = np.nanpercentile(score, 90.0, axis=axis)
    return (0.55 * mean + 0.45 * p90).astype(np.float32)


def _robust_vrange(arr: np.ndarray, valid: np.ndarray | None = None, qlo: float = 3.0, qhi: float = 99.0) -> tuple[float, float]:
    x = np.asarray(arr, dtype=np.float32)
    mask = np.isfinite(x) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(x))
    vals = x[mask]
    vals = vals[vals > 0]
    if vals.size == 0:
        return 0.0, 1.0
    lo = float(np.percentile(vals, qlo))
    hi = float(np.percentile(vals, qhi))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
    if hi <= lo:
        hi = lo + 1e-12
    return lo, hi


def _style_ax(ax) -> None:
    ax.set_facecolor("#f7f4ef")
    ax.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#3a332c")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#5b5248")


def _style_paper_map_ax(ax, xlabel: bool = True, ylabel: bool = False) -> None:
    ax.set_aspect("equal")
    ax.set_facecolor("white")
    ax.grid(False)
    ax.tick_params(direction="out", length=2.5, width=0.6, labelsize=7, pad=1.5)
    ax.set_xlabel("X (m)" if xlabel else "", fontsize=8, labelpad=1.5)
    ax.set_ylabel("Y (m)" if ylabel else "", fontsize=8, labelpad=1.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#404040")


def _transparent_cmap(name: str):
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def _truncated_cmap(name: str, lo: float = 0.05, hi: float = 0.92):
    from matplotlib.colors import LinearSegmentedColormap

    base = plt.get_cmap(name)
    colors = base(np.linspace(lo, hi, 256))
    cmap = LinearSegmentedColormap.from_list(f"{name}_{lo:.2f}_{hi:.2f}", colors)
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def _mean_grid_step(cx_2d: np.ndarray, cy_2d: np.ndarray) -> float:
    ni, nj = cx_2d.shape
    di = np.mean(np.sqrt(np.diff(cx_2d[:, nj // 2]) ** 2 + np.diff(cy_2d[:, nj // 2]) ** 2))
    dj = np.mean(np.sqrt(np.diff(cx_2d[ni // 2, :]) ** 2 + np.diff(cy_2d[ni // 2, :]) ** 2))
    return float((di + dj) / 2.0)


def _soft_contours(ax, data2d: np.ndarray, xgrid: np.ndarray, ygrid: np.ndarray, valid: np.ndarray | None = None, levels: int = 7, color: str = "#2c241f", alpha: float = 0.18, linewidth: float = 0.55) -> None:
    arr = np.asarray(data2d, dtype=float)
    mask = np.isfinite(arr) if valid is None else (np.asarray(valid, dtype=bool) & np.isfinite(arr))
    vals = arr[mask]
    vals = vals[vals > 0]
    if vals.size < 10:
        return
    lo = float(np.percentile(vals, 12))
    hi = float(np.percentile(vals, 96))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    cs = np.linspace(lo, hi, levels)
    ax.contour(xgrid, ygrid, np.where(mask, arr, np.nan), levels=cs, colors=[color], linewidths=linewidth, alpha=alpha)


def _cell_edges_from_centers_2d(xc: np.ndarray, yc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build quadrilateral cell corners from 2-D CMG cell-center coordinates."""
    x = np.asarray(xc, dtype=np.float64)
    y = np.asarray(yc, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x/y center grids must be 2-D arrays with the same shape")

    def pad_edges(a: np.ndarray) -> np.ndarray:
        p = np.empty((a.shape[0] + 2, a.shape[1] + 2), dtype=np.float64)
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


def _paper_vrange(arr2d: np.ndarray, mask: np.ndarray, qlo: float = 3.0, qhi: float = 99.0) -> tuple[float, float]:
    vals = np.asarray(arr2d)[mask & np.isfinite(arr2d)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.percentile(vals, qlo)), float(np.percentile(vals, qhi))


def _plot_official_xy_prescreen(payload: dict, seeds_file: Path, out_dir: Path) -> dict | None:
    if not seeds_file.exists():
        return None

    z = np.load(seeds_file)
    required = ["J_inj_2d_raw", "J_prod_2d_raw", "J_inj_2d_smooth", "J_prod_2d_smooth", "bound_mask"]
    if any(key not in z.files for key in required):
        missing = [key for key in required if key not in z.files]
        raise KeyError(f"{seeds_file} missing official prescreen arrays: {missing}")

    inj_raw = z["J_inj_2d_raw"]
    prod_raw = z["J_prod_2d_raw"]
    inj_s6 = z["J_inj_2d_smooth"]
    prod_s6 = z["J_prod_2d_smooth"]
    inj_s2 = gaussian_filter(inj_raw, sigma=2)
    prod_s2 = gaussian_filter(prod_raw, sigma=2)
    bound_mask = z["bound_mask"].astype(bool)

    cx_2d = payload.get("cx_mid_2d", payload["cx_2d"])
    cy_2d = payload.get("cy_mid_2d", payload["cy_2d"])
    xedge, yedge = _cell_edges_from_centers_2d(cx_2d, cy_2d)
    dxy = _mean_grid_step(cx_2d, cy_2d)

    inj_v = _paper_vrange(inj_s6, bound_mask, 3.0, 99.0)
    prod_v = _paper_vrange(prod_s6, bound_mask, 3.0, 99.0)
    arrays = [[inj_raw, inj_s2, inj_s6], [prod_raw, prod_s2, prod_s6]]
    cmaps = [_transparent_cmap("hot_r"), _transparent_cmap("Blues")]
    vranges = [inj_v, prod_v]
    labels = ["Normalized CO$_2$ Injection Score", "Normalized Productivity Score"]
    contour_colors = ["#2b2018", "#17313f"]
    titles = ["Raw", rf"$\sigma=2$ (~{2 * dxy:.0f} m)", rf"$\sigma=6$ (~{6 * dxy:.0f} m)"]

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.0), dpi=600, constrained_layout=True)
    fig.patch.set_facecolor("white")
    row_images = []
    for row in range(2):
        row_im = None
        for col in range(3):
            ax = axes[row, col]
            data = arrays[row][col]
            row_im = ax.pcolormesh(
                xedge.T,
                yedge.T,
                np.where(bound_mask, data, np.nan).T,
                cmap=cmaps[row],
                shading="flat",
                vmin=vranges[row][0],
                vmax=vranges[row][1],
            )
            ax.contour(
                cx_2d.T,
                cy_2d.T,
                bound_mask.T.astype(float),
                levels=[0.5],
                colors=["#333333"],
                linewidths=0.60,
                alpha=0.90,
            )
            _soft_contours(
                ax,
                data,
                cx_2d,
                cy_2d,
                valid=bound_mask,
                levels=8,
                color=contour_colors[row],
                alpha=0.16,
                linewidth=0.42,
            )
            _style_paper_map_ax(ax, xlabel=True, ylabel=(col == 0))
            if row == 0:
                ax.set_title(titles[col], fontsize=8, fontweight="bold", pad=2.5)
        row_images.append(row_im)

    for row, im in enumerate(row_images):
        cb = fig.colorbar(im, ax=axes[row, :], shrink=0.88, pad=0.015, aspect=30)
        cb.set_label(labels[row], fontsize=8)
        ticks = np.linspace(vranges[row][0], vranges[row][1], 4)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{v:.2f}" for v in ticks])
        cb.ax.tick_params(labelsize=7, length=2.2, width=0.55)
        cb.outline.set_linewidth(0.6)

    out_png = out_dir / "official_prescreen_xy_raw_sigma2_sigma6.png"
    out_pdf = out_dir / "official_prescreen_xy_raw_sigma2_sigma6.pdf"
    out_svg = out_dir / "official_prescreen_xy_raw_sigma2_sigma6.svg"
    for path in (out_png, out_pdf, out_svg):
        fig.savefig(path, dpi=600 if path.suffix == ".png" else None, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    separate_specs = [
        ("official_prescreen_xy_injection_raw_sigma2_sigma6", arrays[0], cmaps[0], inj_v, labels[0], contour_colors[0]),
        ("official_prescreen_xy_production_raw_sigma2_sigma6", arrays[1], cmaps[1], prod_v, labels[1], contour_colors[1]),
    ]
    for stem, row_arrays, cmap, vrange, cbar_label, contour_color in separate_specs:
        fig, axes_row = plt.subplots(1, 3, figsize=(7.0, 2.1), dpi=600, constrained_layout=True)
        last_im = None
        for ax, data, title in zip(axes_row, row_arrays, titles):
            last_im = ax.pcolormesh(
                xedge.T,
                yedge.T,
                np.where(bound_mask, data, np.nan).T,
                cmap=cmap,
                shading="flat",
                vmin=vrange[0],
                vmax=vrange[1],
            )
            ax.contour(cx_2d.T, cy_2d.T, bound_mask.T.astype(float), levels=[0.5], colors=["#333333"], linewidths=0.60)
            _soft_contours(ax, data, cx_2d, cy_2d, valid=bound_mask, levels=8, color=contour_color, alpha=0.16, linewidth=0.42)
            _style_paper_map_ax(ax, xlabel=True, ylabel=(ax is axes_row[0]))
            ax.set_title(title, fontsize=8, fontweight="bold", pad=2.5)
        cb = fig.colorbar(last_im, ax=axes_row, shrink=0.88, pad=0.015, aspect=30)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7, length=2.2, width=0.55)
        fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.04)
        fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    stats = {
        "source": str(seeds_file),
        "combined_png": str(out_png),
        "injection_sigma6_minmax_inside_boundary": [
            float(np.nanmin(inj_s6[bound_mask])),
            float(np.nanmax(inj_s6[bound_mask])),
        ],
        "production_sigma6_minmax_inside_boundary": [
            float(np.nanmin(prod_s6[bound_mask])),
            float(np.nanmax(prod_s6[bound_mask])),
        ],
        "injection_colorbar_vrange": [float(inj_v[0]), float(inj_v[1])],
        "production_colorbar_vrange": [float(prod_v[0]), float(prod_v[1])],
        "boundary_cells": int(bound_mask.sum()),
        "dxy_m": float(dxy),
    }
    (out_dir / "official_prescreen_xy_ranges.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats


def _plot_side_view_comparison(
    data: np.ndarray,
    xgrid: np.ndarray,
    ygrid: np.ndarray,
    valid: np.ndarray,
    title_prefix: str,
    xlabel: str,
    ylabel: str,
    out_png: Path,
    cmap: str = "hot_r",
    depth_like: bool = False,
    cbar_label: str = "normalized geological descriptor",
    contour_color: str = "#2b2018",
    vertical_exaggeration: float = 20.0,
) -> None:
    sigmas = [0, 2, 6]
    labels = ["Raw", r"$\sigma=2$", r"$\sigma=6$"]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.45), constrained_layout=True)
    if data.ndim != 2:
        raise ValueError("data must be 2-D")
    smoothed_all = []
    for sigma in sigmas:
        if sigma == 0:
            smoothed_all.append(data.copy())
        else:
            if depth_like:
                smoothed_all.append(gaussian_filter(data, sigma=(sigma, max(0.5, sigma / 3.0))).astype(np.float32))
            else:
                smoothed_all.append(gaussian_filter(data, sigma=sigma).astype(np.float32))
    vmin, vmax = _robust_vrange(np.stack(smoothed_all, axis=0), valid, qlo=5.0, qhi=97.5)
    im = None
    xedge, yedge = _cell_edges_from_centers_2d(xgrid, ygrid)
    y_for_plot = ygrid * vertical_exaggeration
    yedge_plot = yedge * vertical_exaggeration
    for ax, arr, lab in zip(axes, smoothed_all, labels):
        arr_plot = np.where(valid, arr, np.nan)
        im = ax.pcolormesh(
            xedge.T,
            yedge_plot.T,
            arr_plot.T,
            cmap=_truncated_cmap(cmap, 0.04, 0.88),
            shading="flat",
            vmin=vmin,
            vmax=vmax,
        )
        if lab == r"$\sigma=6$":
            _soft_contours(ax, arr, xgrid, y_for_plot, valid=valid, levels=8, color=contour_color, alpha=0.16, linewidth=0.45)
        ax.contour(xgrid.T, y_for_plot.T, valid.T.astype(float), levels=[0.5], colors=["#404040"], linewidths=0.65, alpha=0.90)
        ax.set_facecolor("white")
        ax.grid(False)
        ax.tick_params(direction="out", length=2.5, width=0.6, labelsize=7, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#404040")
        ax.set_title(lab, fontsize=7.8, fontweight="bold", pad=2.0)
        ax.set_xlabel(xlabel, fontsize=8, labelpad=1.5)
        ax.set_ylabel(ylabel, fontsize=8, labelpad=1.5)
        ticks = ax.get_yticks()
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t / vertical_exaggeration:.0f}" for t in ticks])
        if depth_like:
            ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=axes, shrink=0.88, pad=0.02)
    cbar.set_label(cbar_label, fontsize=7.6)
    cb_ticks = np.linspace(vmin, vmax, 4)
    cbar.set_ticks(cb_ticks)
    cbar.set_ticklabels([f"{v:.2f}" for v in cb_ticks])
    cbar.ax.tick_params(labelsize=7, length=2.2, width=0.55)
    cbar.outline.set_linewidth(0.6)
    fig.suptitle(f"{title_prefix}; vertical exaggeration x{vertical_exaggeration:g}", fontsize=8.8, fontweight="bold")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def build_maps(args: argparse.Namespace) -> dict:
    prop_dir = Path(args.property_dir)
    pressure = read_cmg_txt(find_property_file(prop_dir, "Pressure")) * 1e-3
    thickness = read_cmg_txt(find_property_file(prop_dir, "Grid Thickness"))
    top = read_cmg_txt(find_property_file(prop_dir, "Grid Top"))
    cx = read_cmg_txt(find_property_file(prop_dir, "Grid Centroid X"))
    cy = read_cmg_txt(find_property_file(prop_dir, "Grid Centroid Y"))
    mu_water = read_cmg_txt(find_property_file(prop_dir, "Water Viscosity"))
    perm = read_property_txt(Path(args.perm_file) if args.perm_file else prop_dir / "property_260421_all_perm.txt")
    poro = read_property_txt(Path(args.poro_file) if args.poro_file else prop_dir / "property_260421_all_poro.txt")

    k0 = max(0, int(args.k_start) - 1)
    k1 = min(NK - 1, int(args.k_end) - 1)
    sl = slice(k0, k1 + 1)

    valid = (
        np.isfinite(perm)
        & np.isfinite(poro)
        & np.isfinite(thickness)
        & np.isfinite(mu_water)
        & np.isfinite(pressure)
        & (perm > 0.0)
        & (poro > 0.0)
        & (thickness > 0.0)
        & (mu_water > 1e-10)
        & (pressure > 0.0)
    )
    if float(args.sand_perm_threshold) > 0:
        sand_threshold = float(args.sand_perm_threshold)
    else:
        pos = perm[valid & (perm > 0)]
        sand_threshold = float(np.percentile(pos, float(args.sand_quantile)))
    sand = valid & (perm >= sand_threshold)

    conn_x, conn_y, conn_z = neighbour_fraction(sand)
    lateral_conn = 0.5 * (conn_x + conn_y)
    all_conn = 0.60 * lateral_conn + 0.25 * conn_z + 0.15 * sand.astype(np.float32)

    mobility = np.nan_to_num(perm * poro * thickness / np.maximum(mu_water, 1e-10), nan=0.0, posinf=0.0, neginf=0.0)
    mobility_n = robust_norm(np.log1p(np.maximum(mobility, 0.0)), valid)
    pressure_n = robust_norm(pressure, valid)

    # Prescreen-style score: rock conductance is the base, sand-body continuity controls transfer,
    # and pressure state provides the static driving-potential modifier.
    score_3d = mobility_n * (0.35 + 0.65 * all_conn) * (0.70 + 0.30 * pressure_n)
    score_3d = np.where(valid, score_3d, 0.0).astype(np.float32)

    focus = score_3d[:, :, sl]
    cx_focus = cx[:, :, sl]
    cy_focus = cy[:, :, sl]
    z_focus = top[:, :, sl] + 0.5 * thickness[:, :, sl]
    xy = aggregate_layers(focus, axis=2)
    xz = aggregate_layers(focus, axis=1)
    yz = aggregate_layers(focus, axis=0)
    cx_2d = np.nanmean(cx_focus, axis=2)
    cy_2d = np.nanmean(cy_focus, axis=2)
    x_xz = np.nanmean(cx_focus, axis=1)
    y_yz = np.nanmean(cy_focus, axis=0)
    z_xz = np.nanmean(z_focus, axis=1)
    z_yz = np.nanmean(z_focus, axis=0)

    sigma = float(args.smooth_sigma)
    if sigma > 0:
        xy = gaussian_filter(xy, sigma=sigma).astype(np.float32)
        xz = gaussian_filter(xz, sigma=(sigma, max(0.5, sigma / 3.0))).astype(np.float32)
        yz = gaussian_filter(yz, sigma=(sigma, max(0.5, sigma / 3.0))).astype(np.float32)

    xy = robust_norm(xy, xy > 0)
    xz = robust_norm(xz, xz > 0)
    yz = robust_norm(yz, yz > 0)

    return {
        "score_3d_focus": focus.astype(np.float32),
        "J_conn_xy": xy,
        "J_conn_xz": xz,
        "J_conn_yz": yz,
        "cx_2d": cx_2d.astype(np.float32),
        "cy_2d": cy_2d.astype(np.float32),
        "cx_mid_2d": cx[:, :, NK // 2].astype(np.float32),
        "cy_mid_2d": cy[:, :, NK // 2].astype(np.float32),
        "x_xz": x_xz.astype(np.float32),
        "y_yz": y_yz.astype(np.float32),
        "z_xz": z_xz.astype(np.float32),
        "z_yz": z_yz.astype(np.float32),
        "sand_mask_focus": sand[:, :, sl],
        "valid_mask_focus": valid[:, :, sl],
        "k_start": np.array(k0 + 1, dtype=np.int32),
        "k_end": np.array(k1 + 1, dtype=np.int32),
        "sand_perm_threshold_md": np.array(sand_threshold, dtype=np.float32),
        "dx_m": np.array(DX_M, dtype=np.float32),
        "dy_m": np.array(DY_M, dtype=np.float32),
    }


def _aggregate_interval_side_views(
    score_3d: np.ndarray,
    valid_3d: np.ndarray,
    cx: np.ndarray,
    cy: np.ndarray,
    zmid: np.ndarray,
    k_start: int,
    k_end: int,
) -> dict:
    k0 = max(0, k_start - 1)
    k1 = min(NK - 1, k_end - 1)
    sl = slice(k0, k1 + 1)
    j_mid = NJ // 2
    i_mid = NI // 2
    xz_raw = score_3d[:, j_mid, sl]
    yz_raw = score_3d[i_mid, :, sl]
    xz_valid = valid_3d[:, j_mid, sl]
    yz_valid = valid_3d[i_mid, :, sl]
    return {
        "xz": robust_norm(xz_raw, xz_valid),
        "yz": robust_norm(yz_raw, yz_valid),
        "xz_valid": xz_valid,
        "yz_valid": yz_valid,
        "x_xz": cx[:, j_mid, sl],
        "z_xz": zmid[:, j_mid, sl],
        "y_yz": cy[i_mid, :, sl],
        "z_yz": zmid[i_mid, :, sl],
        "i_mid": np.array(i_mid + 1, dtype=np.int32),
        "j_mid": np.array(j_mid + 1, dtype=np.int32),
    }


def build_side_view_prescreen_fields(seeds_file: Path) -> dict:
    z = np.load(seeds_file)
    required = [
        "J_inj_3d",
        "J_prod_3d",
        "J_global_max_inj",
        "J_global_max_prod",
        "cx_3d",
        "cy_3d",
        "top_3d",
        "thick_3d",
    ]
    missing = [key for key in required if key not in z.files]
    if missing:
        raise KeyError(
            f"{seeds_file} does not contain layer-resolved prescreen fields {missing}. "
            "Run F:\\CODE1\\injectivity_prescreening_map.py once to refresh opt_seeds.npz."
        )

    j_inj = np.asarray(z["J_inj_3d"], dtype=np.float32) / max(float(z["J_global_max_inj"]), 1e-30)
    j_prod = np.asarray(z["J_prod_3d"], dtype=np.float32) / max(float(z["J_global_max_prod"]), 1e-30)
    cx = np.asarray(z["cx_3d"], dtype=np.float32)
    cy = np.asarray(z["cy_3d"], dtype=np.float32)
    zmid = np.asarray(z["top_3d"], dtype=np.float32) + 0.5 * np.asarray(z["thick_3d"], dtype=np.float32)
    inj_valid = j_inj > 0
    prod_valid = j_prod > 0

    return {
        "injection": _aggregate_interval_side_views(j_inj, inj_valid, cx, cy, zmid, INJ_K_START, INJ_K_END),
        "production": _aggregate_interval_side_views(j_prod, prod_valid, cx, cy, zmid, PRD_K_START, PRD_K_END),
    }


def _plot_combined_side_view(
    side_payload: dict,
    direction: str,
    out_dir: Path,
    vertical_exaggeration: float = 20.0,
) -> Path:
    if direction == "xz":
        x_label = "X (m)"
        x_inj = side_payload["injection"]["x_xz"]
        z_inj = side_payload["injection"]["z_xz"]
        d_inj = side_payload["injection"]["xz"]
        v_inj = side_payload["injection"]["xz_valid"]
        x_prod = side_payload["production"]["x_xz"]
        z_prod = side_payload["production"]["z_xz"]
        d_prod = side_payload["production"]["xz"]
        v_prod = side_payload["production"]["xz_valid"]
        section_label = f"J={int(side_payload['injection']['j_mid'])}"
        filename = "official_prescreen_xz_raw_sigma2_sigma6"
        title = f"Side-view prescreen maps (XZ, {section_label})"
    else:
        x_label = "Y (m)"
        x_inj = side_payload["injection"]["y_yz"]
        z_inj = side_payload["injection"]["z_yz"]
        d_inj = side_payload["injection"]["yz"]
        v_inj = side_payload["injection"]["yz_valid"]
        x_prod = side_payload["production"]["y_yz"]
        z_prod = side_payload["production"]["z_yz"]
        d_prod = side_payload["production"]["yz"]
        v_prod = side_payload["production"]["yz_valid"]
        section_label = f"I={int(side_payload['injection']['i_mid'])}"
        filename = "official_prescreen_yz_raw_sigma2_sigma6"
        title = f"Front-view prescreen maps (YZ, {section_label})"

    rows = [
        {
            "data": d_inj,
            "valid": v_inj,
            "x": x_inj,
            "z": z_inj,
            "cmap": _truncated_cmap("hot_r", 0.04, 0.88),
            "label": "Normalized CO$_2$ Injection Score",
            "contour": "#2b2018",
        },
        {
            "data": d_prod,
            "valid": v_prod,
            "x": x_prod,
            "z": z_prod,
            "cmap": _truncated_cmap("Blues", 0.04, 0.88),
            "label": "Normalized Productivity Score",
            "contour": "#17313f",
        },
    ]
    sigma_values = [0, 2, 6]
    sigma_labels = ["Raw", r"$\sigma=2$", r"$\sigma=6$"]
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.5), dpi=600, constrained_layout=True)
    fig.patch.set_facecolor("white")
    row_images = []

    for row_idx, row in enumerate(rows):
        xedge, zedge = _cell_edges_from_centers_2d(row["x"], row["z"])
        z_plot = row["z"] * vertical_exaggeration
        zedge_plot = zedge * vertical_exaggeration
        arrays = []
        for sigma in sigma_values:
            if sigma == 0:
                arrays.append(row["data"].copy())
            else:
                arrays.append(gaussian_filter(row["data"], sigma=(sigma, max(0.5, sigma / 3.0))).astype(np.float32))
        valid = row["valid"]
        vmin, vmax = _robust_vrange(np.stack(arrays, axis=0), valid, qlo=3.0, qhi=99.0)
        row_im = None
        for col, (ax, arr, sigma_label) in enumerate(zip(axes[row_idx, :], arrays, sigma_labels)):
            row_im = ax.pcolormesh(
                xedge.T,
                zedge_plot.T,
                np.where(valid, arr, np.nan).T,
                cmap=row["cmap"],
                shading="flat",
                vmin=vmin,
                vmax=vmax,
            )
            # Only the selected sigma=6 panel receives value contours.
            if col == 2:
                _soft_contours(
                    ax,
                    arr,
                    row["x"],
                    z_plot,
                    valid=valid,
                    levels=8,
                    color=row["contour"],
                    alpha=0.16,
                    linewidth=0.45,
                )
            ax.contour(
                row["x"].T,
                z_plot.T,
                valid.T.astype(float),
                levels=[0.5],
                colors=["#404040"],
                linewidths=0.6,
                alpha=0.9,
            )
            ax.set_facecolor("white")
            ax.grid(False)
            ax.tick_params(direction="out", length=2.2, width=0.55, labelsize=6.7, pad=1.2)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#404040")
            ax.set_title(sigma_label, fontsize=7.8, fontweight="bold", pad=2.0)
            ax.set_xlabel(x_label, fontsize=7.5, labelpad=1.0)
            ax.set_ylabel("TVDSS (m)" if col == 0 else "", fontsize=7.5, labelpad=1.0)
            ticks = ax.get_yticks()
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{tick / vertical_exaggeration:.0f}" for tick in ticks])
            ax.invert_yaxis()
        row_images.append((row_im, vmin, vmax, row["label"]))

    for row_idx, (im, vmin, vmax, label) in enumerate(row_images):
        cb = fig.colorbar(im, ax=axes[row_idx, :], shrink=0.88, pad=0.018, aspect=30)
        cb.set_label(label, fontsize=7.5)
        ticks = np.linspace(vmin, vmax, 4)
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{value:.2f}" for value in ticks])
        cb.ax.tick_params(labelsize=6.5, length=2.0, width=0.5)
        cb.outline.set_linewidth(0.55)

    fig.suptitle(f"{title}; vertical exaggeration x{vertical_exaggeration:g}", fontsize=8.8, fontweight="bold")
    out_png = out_dir / f"{filename}.png"
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return out_png


def plot_side_views(side_payload: dict, out_dir: Path) -> dict:
    outputs = {}
    for direction in ("xz", "yz"):
        output = _plot_combined_side_view(side_payload, direction, out_dir)
        outputs[output.stem] = str(output)
    return outputs


def plot_maps(payload: dict, out_png: Path) -> None:
    xy = payload["J_conn_xy"]
    xz = payload["J_conn_xz"]
    yz = payload["J_conn_yz"]
    cx_2d = payload["cx_2d"]
    cy_2d = payload["cy_2d"]
    x_xz = payload["x_xz"]
    y_yz = payload["y_yz"]
    z_xz = payload["z_xz"]
    z_yz = payload["z_yz"]
    k_start = int(payload["k_start"])
    k_end = int(payload["k_end"])
    sand_mask = payload["sand_mask_focus"]
    valid_mask = payload["valid_mask_focus"]
    xy_valid = np.any(valid_mask, axis=2)
    xz_valid = np.any(valid_mask, axis=1)
    yz_valid = np.any(valid_mask, axis=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    xx_xy = cx_2d / 1000.0
    yy_xy = cy_2d / 1000.0
    xx_xy_edge, yy_xy_edge = _cell_edges_from_centers_2d(xx_xy, yy_xy)
    xy_sm = gaussian_filter(xy, sigma=2.0)
    vmin_xy, vmax_xy = _robust_vrange(xy_sm, xy_valid, qlo=3.0, qhi=99.0)
    im0 = axes[0].pcolormesh(xx_xy_edge.T, yy_xy_edge.T, np.where(xy_valid, xy_sm, np.nan).T, cmap="hot_r", shading="flat", vmin=vmin_xy, vmax=vmax_xy)
    _soft_contours(axes[0], xy_sm, xx_xy, yy_xy, valid=xy_valid, levels=8, color="#2b2018", alpha=0.16, linewidth=0.5)
    axes[0].contour(xx_xy.T, yy_xy.T, xy_valid.T.astype(float), levels=[0.5], colors=["#4a4037"], linewidths=0.8, alpha=0.85)
    _style_ax(axes[0])
    axes[0].set_title("XY connectivity prescreen")
    axes[0].set_xlabel("X (km)")
    axes[0].set_ylabel("Y (km)")

    xx_xz = x_xz / 1000.0
    zz_xz = z_xz
    xx_xz_edge, zz_xz_edge = _cell_edges_from_centers_2d(xx_xz, zz_xz)
    xz_sm = gaussian_filter(xz, sigma=(2.0, 1.2))
    vmin_xz, vmax_xz = _robust_vrange(xz_sm, xz_valid, qlo=3.0, qhi=99.0)
    im1 = axes[1].pcolormesh(xx_xz_edge.T, zz_xz_edge.T, np.where(xz_valid, xz_sm, np.nan).T, cmap="viridis", shading="flat", vmin=vmin_xz, vmax=vmax_xz)
    _soft_contours(axes[1], xz_sm, xx_xz, zz_xz, valid=xz_valid, levels=8, color="#17313f", alpha=0.16, linewidth=0.5)
    axes[1].contour(xx_xz.T, zz_xz.T, xz_valid.T.astype(float), levels=[0.5], colors=["#4a4037"], linewidths=0.8, alpha=0.85)
    axes[1].invert_yaxis()
    _style_ax(axes[1])
    axes[1].set_title("XZ connectivity prescreen")
    axes[1].set_xlabel("X (km)")
    axes[1].set_ylabel("TVDSS / grid depth (m)")

    yy_yz = y_yz / 1000.0
    zz_yz = z_yz
    yy_yz_edge, zz_yz_edge = _cell_edges_from_centers_2d(yy_yz, zz_yz)
    yz_sm = gaussian_filter(yz, sigma=(2.0, 1.2))
    vmin_yz, vmax_yz = _robust_vrange(yz_sm, yz_valid, qlo=3.0, qhi=99.0)
    im2 = axes[2].pcolormesh(yy_yz_edge.T, zz_yz_edge.T, np.where(yz_valid, yz_sm, np.nan).T, cmap="viridis", shading="flat", vmin=vmin_yz, vmax=vmax_yz)
    _soft_contours(axes[2], yz_sm, yy_yz, zz_yz, valid=yz_valid, levels=8, color="#17313f", alpha=0.16, linewidth=0.5)
    axes[2].contour(yy_yz.T, zz_yz.T, yz_valid.T.astype(float), levels=[0.5], colors=["#4a4037"], linewidths=0.8, alpha=0.85)
    axes[2].invert_yaxis()
    _style_ax(axes[2])
    axes[2].set_title("YZ connectivity prescreen")
    axes[2].set_xlabel("Y (km)")
    axes[2].set_ylabel("TVDSS / grid depth (m)")

    cbar = fig.colorbar(im2, ax=axes, shrink=0.86, pad=0.015)
    cbar.set_label("normalized connectivity-transfer potential")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    # Separate publication-style figures for top, side, and front views.
    out_dir = out_png.parent
    _plot_view_comparison(
        xy,
        xx_xy,
        yy_xy,
        xy_valid,
        "Top-view prescreen map (XY)",
        "X (km)",
        "Y (km)",
        out_dir / "prescreen_xy_raw_sigma2_sigma6.png",
        cmap="hot_r",
        depth_like=False,
    )
    _plot_view_comparison(
        xz,
        xx_xz,
        zz_xz,
        xz_valid,
        "Side-view prescreen map (XZ)",
        "X (km)",
        "TVDSS / grid depth (m)",
        out_dir / "prescreen_xz_raw_sigma2_sigma6.png",
        cmap="hot_r",
        depth_like=True,
    )
    _plot_view_comparison(
        yz,
        yy_yz,
        zz_yz,
        yz_valid,
        "Front-view prescreen map (YZ)",
        "Y (km)",
        "TVDSS / grid depth (m)",
        out_dir / "prescreen_yz_raw_sigma2_sigma6.png",
        cmap="hot_r",
        depth_like=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build tri-view prescreen maps for sand-body connectivity and pressure transfer.")
    parser.add_argument("--property-dir", default=str(PROPERTY_DIR))
    parser.add_argument("--perm-file", default="")
    parser.add_argument("--poro-file", default="")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--k-start", type=int, default=1)
    parser.add_argument("--k-end", type=int, default=NK)
    parser.add_argument("--sand-perm-threshold", type=float, default=1.0)
    parser.add_argument("--sand-quantile", type=float, default=65.0)
    parser.add_argument("--smooth-sigma", type=float, default=2.0)
    parser.add_argument("--seeds-file", default=str(ROOT / "opt_seeds.npz"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_maps(args)
    official_xy = _plot_official_xy_prescreen(payload, Path(args.seeds_file), out_dir)
    side_payload = build_side_view_prescreen_fields(Path(args.seeds_file))
    side_outputs = plot_side_views(side_payload, out_dir)

    summary = {
        "official_xy_prescreen": official_xy,
        "side_view_prescreen_descriptors": side_outputs,
        "side_view_definition": {
            "section_type": "central sections, not full-width averaged projections",
            "xz_section": "J = 102",
            "yz_section": "I = 135",
            "injection_interval": f"K = {INJ_K_START}-{INJ_K_END}",
            "production_interval": f"K = {PRD_K_START}-{PRD_K_END}",
            "vertical_exaggeration": 20.0,
            "descriptor": "layer-resolved J_inj_3d/J_prod_3d fields normalized by their interval global maxima",
            "note": (
                "The XY maps use the official 2-D prescreen arrays from opt_seeds.npz. "
                "The side views use layer-resolved prescreen potentials exported by injectivity_prescreening_map.py."
            ),
        },
    }
    (out_dir / "official_prescreen_triview_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
