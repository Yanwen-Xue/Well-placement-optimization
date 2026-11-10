"""
Plot layer-wise pressure-difference fields used by the prescreen map.

Outputs:
- Injection interval: DeltaP_safe,k = 0.90 * Pf,k - P_BHP,k
- Production interval: DeltaP_prod,k = Pp,k - Pmin_prod

The script is intentionally independent from the main optimization workflow and
only creates figures under paper_materials/pressure_difference_layers.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titleweight"] = "semibold"
plt.rcParams["savefig.facecolor"] = "white"

BLUE_GREEN_RED = mcolors.LinearSegmentedColormap.from_list(
    "blue_green_red",
    ["#1f5aa6", "#24b6c8", "#7fd34e", "#f6d84a", "#d73027"],
    N=256,
)

RED_GREEN_BLUE = mcolors.LinearSegmentedColormap.from_list(
    "red_green_blue",
    ["#d73027", "#fdae61", "#7fd34e", "#24b6c8", "#1f5aa6"],
    N=256,
)

# Solid green for uniform brine viscosity – the single constant value maps to
# this green regardless of norm, instead of falling on the pale end of a gradient.
BRINE_GREEN = mcolors.LinearSegmentedColormap.from_list(
    "brine_green",
    ["#3a8c3f", "#3a8c3f"],
    N=256,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "property_CMG"
OUT_DIR = ROOT / "paper_materials" / "pressure_difference_layers"

NI, NJ, NK = 268, 203, 200

POISSON_NU = 0.29
RHO_OVERBURDEN = 2500.0
RHO_BRINE = 1015.0
RHO_CO2_COLUMN = 700.0
G = 9.81
EATON_COEFF = 3.0 * POISSON_NU / (2.0 * (1.0 - POISSON_NU))
BHP_SAFETY_COEFF = 0.90

REFPRES_KPA = 22929.0
REFDEPTH_TVDSS_M = 987.241211
H_SURFACE = 1200.0

INJ_K_START = 41
INJ_K_END = 80
Q_DAY = 288392.0
BG_CO2 = 0.003
RE_RW_RATIO = 500.0
LN_RE_RW = np.log(RE_RW_RATIO)
DARCY_CONV = 0.0853

P_MIN_PROD_MPA = 8.0
D_BOUND_M = 1000.0

T_SEA_LEVEL_C = 15.0
GEO_GRADIENT = 33.3

INJ_LAYERS = [41, 46, 51, 56, 61, 66, 71, 76, 80]
PRD_LAYERS = [1, 3, 5, 7, 9, 11, 13, 14, 15]

_PFX = "2026_0421_1day_property "
_SFX = " 2025-Jan-01.txt"
FILES = {
    "pressure": DATA_DIR / f"{_PFX}Pressure{_SFX}",
    "thickness": DATA_DIR / f"{_PFX}Grid Thickness{_SFX}",
    "top": DATA_DIR / f"{_PFX}Grid Top{_SFX}",
    "cx": DATA_DIR / f"{_PFX}Grid Centroid X{_SFX}",
    "cy": DATA_DIR / f"{_PFX}Grid Centroid Y{_SFX}",
    "mu_water": DATA_DIR / f"{_PFX}Water Viscosity{_SFX}",
    "perm": DATA_DIR / "property_260421_all_perm.txt",
}


def read_cmg_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    data = np.zeros((ni, nj, nk), dtype=np.float64)
    current_k = None
    current_j = None
    i_values: list[float] = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("**") and "K =" in line and "J =" in line:
                if current_k is not None and current_j is not None and i_values:
                    arr = np.asarray(i_values[:ni], dtype=np.float64)
                    if len(arr) == ni:
                        data[:, current_j - 1, current_k - 1] = arr
                try:
                    parts = line.replace(",", " ").split()
                    current_k = int(parts[parts.index("K") + 2])
                    current_j = int(parts[parts.index("J") + 2])
                    i_values = []
                except Exception:
                    current_k = None
                    current_j = None
                    i_values = []
            elif line.startswith("**") or line.startswith("RESULTS") or line.startswith("Grid"):
                continue
            else:
                try:
                    i_values.extend(float(v) for v in line.split())
                except Exception:
                    continue

    if current_k is not None and current_j is not None and i_values:
        arr = np.asarray(i_values[:ni], dtype=np.float64)
        if len(arr) == ni:
            data[:, current_j - 1, current_k - 1] = arr
    return data


def read_property_txt(path: Path, ni: int = NI, nj: int = NJ, nk: int = NK) -> np.ndarray:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        _ = f.readline()
        vals = [float(line.split()[0]) for line in f if line.strip()]
    data = np.asarray(vals, dtype=np.float64)
    total = ni * nj * nk
    if len(data) != total:
        data = data[:total]
    return data.reshape((ni, nj, nk), order="F")


def initial_pore_pressure_mpa(tvdss_m: np.ndarray) -> np.ndarray:
    return (REFPRES_KPA + (RHO_BRINE * G / 1000.0) * (tvdss_m - REFDEPTH_TVDSS_M)) * 1e-3


def compute_mu_co2_cp(pres_3d_mpa: np.ndarray, temp_3d_c: np.ndarray) -> np.ndarray:
    from CoolProp.CoolProp import PropsSI
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import distance_transform_edt

    p_min = pres_3d_mpa[pres_3d_mpa > 0].min() * 1e6
    p_max = pres_3d_mpa.max() * 1e6
    p_grid = np.linspace(max(1e6, p_min * 0.9), p_max * 1.1, 60)
    t_grid = np.linspace(temp_3d_c.min() + 273.15 - 3.0, temp_3d_c.max() + 273.15 + 3.0, 60)

    mu_table = np.zeros((len(p_grid), len(t_grid)), dtype=np.float64)
    for i, p_val in enumerate(p_grid):
        for j, t_val in enumerate(t_grid):
            try:
                mu_table[i, j] = PropsSI("V", "P", p_val, "T", t_val, "CO2")
            except Exception:
                mu_table[i, j] = np.nan

    if np.isnan(mu_table).any():
        nan_mask = np.isnan(mu_table)
        _, idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        mu_table = mu_table[tuple(idx)]

    interp = RegularGridInterpolator(
        (p_grid, t_grid),
        mu_table * 1e3,
        bounds_error=False,
        fill_value=None,
    )
    p_input = pres_3d_mpa * 1e6
    t_input = temp_3d_c + 273.15
    p_eff = np.where(p_input > 1e3, p_input, p_grid[0])
    pts = np.stack([p_eff.ravel(), t_input.ravel()], axis=-1)
    mu = interp(pts).reshape(pres_3d_mpa.shape)
    mu[p_input <= 1e3] = np.nan
    return mu


def fracture_pressure_mpa(top_3d: np.ndarray, thick_3d: np.ndarray) -> np.ndarray:
    tvdss_mid = top_3d + 0.5 * thick_3d
    depth_mid = np.maximum(H_SURFACE + tvdss_mid, 0.0)
    sigma_v = RHO_OVERBURDEN * G * depth_mid * 1e-6
    pp = initial_pore_pressure_mpa(tvdss_mid)
    return pp + (sigma_v - pp) * EATON_COEFF


def meters_to_grid_steps(distance_m: float, cx_2d: np.ndarray, cy_2d: np.ndarray) -> int:
    di = np.mean(
        np.sqrt(
            np.diff(cx_2d[:, NJ // 2]) ** 2
            + np.diff(cy_2d[:, NJ // 2]) ** 2
        )
    )
    dj = np.mean(
        np.sqrt(
            np.diff(cx_2d[NI // 2, :]) ** 2
            + np.diff(cy_2d[NI // 2, :]) ** 2
        )
    )
    return max(1, int(np.ceil(distance_m / ((di + dj) / 2.0))))


def boundary_mask(d_bound: int) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(NI), np.arange(NJ), indexing="ij")
    dist = np.minimum(np.minimum(ii, NI - 1 - ii), np.minimum(jj, NJ - 1 - jj))
    return dist >= d_bound


def style_map_axes(ax) -> None:
    ax.set_aspect("equal")
    ax.set_facecolor("#f7f4ef")
    ax.grid(False)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors="#3a332c")
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color("#5b5248")


def plot_domain_outline(ax, cx_2d: np.ndarray, cy_2d: np.ndarray, mask_2d: np.ndarray) -> None:
    ax.contour(
        cx_2d.T,
        cy_2d.T,
        mask_2d.T.astype(float),
        levels=[0.5],
        colors=["#4a4037"],
        linewidths=0.8,
        alpha=0.85,
    )


def plot_soft_contours(
    ax,
    field: np.ndarray,
    cx_2d: np.ndarray,
    cy_2d: np.ndarray,
    mask_2d: np.ndarray,
    color: str,
    alpha: float = 0.24,
    linewidth: float = 0.45,
) -> None:
    return
    vals = field[mask_2d & np.isfinite(field)]
    if vals.size < 10:
        return
    lo, hi = np.nanpercentile(vals, [8.0, 96.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return
    levels = np.linspace(lo, hi, 8)
    ax.contour(
        cx_2d.T,
        cy_2d.T,
        np.where(mask_2d, field, np.nan).T,
        levels=levels,
        colors=[color],
        linewidths=linewidth,
        alpha=alpha,
    )


def plot_single(
    field: np.ndarray,
    title: str,
    path: Path,
    cmap: str,
    norm,
    cbar_label: str,
    cx_2d: np.ndarray,
    cy_2d: np.ndarray,
    mask_2d: np.ndarray,
    contour_color: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    plot_field = np.where(mask_2d, field, np.nan)
    im = ax.pcolormesh(
        cx_2d.T,
        cy_2d.T,
        plot_field.T,
        shading="auto",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        snap=True,
        cmap=cmap,
        norm=norm,
    )
    plot_soft_contours(ax, field, cx_2d, cy_2d, mask_2d, contour_color)
    plot_domain_outline(ax, cx_2d, cy_2d, mask_2d)
    style_map_axes(ax)
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel(
    fields: dict[int, np.ndarray],
    title: str,
    path: Path,
    cmap: str,
    norm,
    cbar_label: str,
    cx_2d: np.ndarray,
    cy_2d: np.ndarray,
    mask_2d: np.ndarray,
    contour_color: str,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.5), constrained_layout=True)
    last_im = None
    for ax, (k, field) in zip(axes.ravel(), fields.items()):
        plot_field = np.where(mask_2d, field, np.nan)
        last_im = ax.pcolormesh(
            cx_2d.T,
            cy_2d.T,
            plot_field.T,
            shading="auto",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
            snap=True,
            cmap=cmap,
            norm=norm,
        )
        plot_soft_contours(ax, field, cx_2d, cy_2d, mask_2d, contour_color)
        plot_domain_outline(ax, cx_2d, cy_2d, mask_2d)
        style_map_axes(ax)
        ax.set_title(f"k={k}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    fig.suptitle(title, fontsize=16, fontweight="semibold")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label(cbar_label)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _old_plot_single(field: np.ndarray, title: str, path: Path, cmap: str, norm, cbar_label: str, extent) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    im = ax.imshow(
        field.T,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=extent,
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.88)
    cbar.set_label(cbar_label)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _old_plot_panel(fields: dict[int, np.ndarray], title: str, path: Path, cmap: str, norm, cbar_label: str, extent) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(13.0, 10.5), constrained_layout=True)
    last_im = None
    for ax, (k, field) in zip(axes.ravel(), fields.items()):
        last_im = ax.imshow(
            field.T,
            origin="lower",
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            aspect="auto",
        )
        ax.set_title(f"k={k}")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
    fig.suptitle(title, fontsize=16, fontweight="semibold")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82)
        cbar.set_label(cbar_label)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def robust_norm(fields: list[np.ndarray], diverging: bool, mask_2d: np.ndarray) -> mcolors.Normalize:
    values = np.concatenate([arr[mask_2d & np.isfinite(arr)].ravel() for arr in fields])
    if values.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    if diverging:
        vmax = float(np.nanpercentile(np.abs(values), 98.0))
        vmax = max(vmax, 1e-6)
        return mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    vmin, vmax = np.nanpercentile(values, [2.0, 98.0])
    if vmax <= vmin:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))


def percentile_norm(fields: list[np.ndarray], mask_2d: np.ndarray, qlo: float, qhi: float) -> mcolors.Normalize:
    values = np.concatenate([arr[mask_2d & np.isfinite(arr)].ravel() for arr in fields])
    if values.size == 0:
        return mcolors.Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = np.nanpercentile(values, [qlo, qhi])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin, vmax = np.nanmin(values), np.nanmax(values)
    if vmax <= vmin:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=float(vmin), vmax=float(vmax))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1] Loading pressure, top, thickness, viscosity, permeability, and coordinates...")
    pres_3d = read_cmg_txt(FILES["pressure"]) * 1e-3
    top_3d = read_cmg_txt(FILES["top"])
    thick_3d = read_cmg_txt(FILES["thickness"])
    cx_3d = read_cmg_txt(FILES["cx"])
    cy_3d = read_cmg_txt(FILES["cy"])
    mu_water_3d = read_cmg_txt(FILES["mu_water"])
    perm_3d = read_property_txt(FILES["perm"])
    perm_3d[perm_3d < 1e-6] = 0.0

    print("[2] Computing CO2 viscosity and well-level injection pressure...")
    temp_3d = T_SEA_LEVEL_C + GEO_GRADIENT * top_3d / 1000.0
    mu_co2_3d = compute_mu_co2_cp(pres_3d, temp_3d)

    inj_k0 = INJ_K_START - 1
    inj_k1 = INJ_K_END - 1
    kh_inj = np.sum(perm_3d[:, :, inj_k0 : inj_k1 + 1] * thick_3d[:, :, inj_k0 : inj_k1 + 1], axis=2)
    with np.errstate(invalid="ignore"):
        mu_inj = np.nanmean(mu_co2_3d[:, :, inj_k0 : inj_k1 + 1], axis=2)

    tvdss_perf_top = top_3d[:, :, inj_k1]
    pp_perf_top = initial_pore_pressure_mpa(tvdss_perf_top)
    dp_darcy = np.full((NI, NJ), np.nan)
    valid = (kh_inj > 1e-10) & np.isfinite(mu_inj) & (mu_inj > 1e-10)
    dp_darcy[valid] = (
        Q_DAY * BG_CO2 * mu_inj[valid] * LN_RE_RW
        / (2.0 * np.pi * kh_inj[valid] * DARCY_CONV)
    )
    p_inj = pp_perf_top + dp_darcy

    print("[3] Computing layer-resolved pressure differences...")
    pf_3d = fracture_pressure_mpa(top_3d, thick_3d)
    tvdss_mid_3d = top_3d + 0.5 * thick_3d

    inj_fields: dict[int, np.ndarray] = {}
    for k in INJ_LAYERS:
        kk = k - 1
        dz = tvdss_mid_3d[:, :, kk] - tvdss_perf_top
        p_bhp_k = p_inj + RHO_CO2_COLUMN * G * dz * 1e-6
        inj_fields[k] = BHP_SAFETY_COEFF * pf_3d[:, :, kk] - p_bhp_k

    prd_fields: dict[int, np.ndarray] = {}
    for k in PRD_LAYERS:
        kk = k - 1
        prd_fields[k] = pres_3d[:, :, kk] - P_MIN_PROD_MPA

    mu_co2_fields: dict[int, np.ndarray] = {}
    for k in INJ_LAYERS:
        mu_co2_fields[k] = mu_co2_3d[:, :, k - 1]

    mu_water_fields: dict[int, np.ndarray] = {}
    for k in PRD_LAYERS:
        mu_water_fields[k] = mu_water_3d[:, :, k - 1]

    cx = cx_3d[:, :, NK // 2]
    cy = cy_3d[:, :, NK // 2]
    bound_steps = meters_to_grid_steps(D_BOUND_M, cx, cy)
    bound_mask = boundary_mask(bound_steps)

    inj_norm = percentile_norm(list(inj_fields.values()), mask_2d=bound_mask, qlo=3.0, qhi=97.0)
    prd_norm = percentile_norm(list(prd_fields.values()), mask_2d=bound_mask, qlo=3.0, qhi=97.0)
    mu_co2_norm = percentile_norm(list(mu_co2_fields.values()), mask_2d=bound_mask, qlo=3.0, qhi=97.0)
    mu_water_norm = percentile_norm(list(mu_water_fields.values()), mask_2d=bound_mask, qlo=3.0, qhi=97.0)

    print("[4] Saving figures...")
    for k, field in inj_fields.items():
        plot_single(
            field,
            f"Injection layer k={k}: DeltaP_safe",
            OUT_DIR / f"injection_deltaP_safe_k{k:03d}.png",
            BLUE_GREEN_RED,
            inj_norm,
            "DeltaP_safe (MPa)",
            cx,
            cy,
            bound_mask,
            "#2b2018",
        )
    for k, field in prd_fields.items():
        plot_single(
            field,
            f"Production layer k={k}: DeltaP_prod",
            OUT_DIR / f"production_deltaP_prod_k{k:03d}.png",
            "YlGnBu",
            prd_norm,
            "DeltaP_prod (MPa)",
            cx,
            cy,
            bound_mask,
            "#17313f",
        )

    for k, field in mu_co2_fields.items():
        plot_single(
            field,
            f"Injection layer k={k}: CO2 viscosity",
            OUT_DIR / f"injection_mu_co2_k{k:03d}.png",
            RED_GREEN_BLUE,
            mu_co2_norm,
            "CO2 viscosity (cP)",
            cx,
            cy,
            bound_mask,
            "#17313f",
        )

    for k, field in mu_water_fields.items():
        plot_single(
            field,
            f"Production layer k={k}: brine viscosity",
            OUT_DIR / f"production_mu_brine_k{k:03d}.png",
            BRINE_GREEN,
            mu_water_norm,
            "Brine viscosity (cP)",
            cx,
            cy,
            bound_mask,
            "#17313f",
        )

    plot_panel(
        inj_fields,
        "Layer-resolved injection pressure-safety margin",
        OUT_DIR / "injection_deltaP_safe_3x3.png",
        BLUE_GREEN_RED,
        inj_norm,
        "DeltaP_safe (MPa)",
        cx,
        cy,
        bound_mask,
        "#2b2018",
    )
    plot_panel(
        prd_fields,
        "Layer-resolved production drawdown margin",
        OUT_DIR / "production_deltaP_prod_3x3.png",
        "YlGnBu",
        prd_norm,
        "DeltaP_prod (MPa)",
        cx,
        cy,
        bound_mask,
        "#17313f",
    )

    plot_panel(
        mu_co2_fields,
        "Layer-resolved CO2 viscosity in injection interval",
        OUT_DIR / "injection_mu_co2_3x3.png",
        RED_GREEN_BLUE,
        mu_co2_norm,
        "CO2 viscosity (cP)",
        cx,
        cy,
        bound_mask,
        "#17313f",
    )
    plot_panel(
        mu_water_fields,
        "Layer-resolved brine viscosity in production interval",
        OUT_DIR / "production_mu_brine_3x3.png",
        BRINE_GREEN,
        mu_water_norm,
        "Brine viscosity (cP)",
        cx,
        cy,
        bound_mask,
        "#17313f",
    )

    print(f"Saved figures to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
