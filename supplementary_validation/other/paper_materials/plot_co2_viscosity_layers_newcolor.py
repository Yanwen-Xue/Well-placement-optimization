from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from plot_pressure_difference_layers import (
    DATA_DIR,
    D_BOUND_M,
    FILES,
    GEO_GRADIENT,
    INJ_LAYERS,
    NI,
    NJ,
    NK,
    OUT_DIR,
    T_SEA_LEVEL_C,
    boundary_mask,
    compute_mu_co2_cp,
    meters_to_grid_steps,
    percentile_norm,
    plot_domain_outline,
    read_cmg_txt,
    style_map_axes,
)


plt.rcParams.update(
    {
        "font.family": "Times New Roman",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "savefig.facecolor": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


FIG_DIR = Path(r"C:\Users\admin\Desktop\paper_figure\writen\figure")

CO2_VISCOSITY_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "co2_viscosity_balanced_teal_yellow",
    ["#24415a", "#356f7d", "#4f9a8f", "#86bd96", "#c9d88f", "#f2e8a5"],
    N=256,
)


def load_fields() -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    print("[1] Loading pressure, top, and CMG grid coordinates...")
    pres_3d = read_cmg_txt(FILES["pressure"]) * 1e-3
    top_3d = read_cmg_txt(FILES["top"])
    cx_3d = read_cmg_txt(FILES["cx"])
    cy_3d = read_cmg_txt(FILES["cy"])

    print("[2] Computing CO2 viscosity from pressure-temperature fields...")
    temp_3d = T_SEA_LEVEL_C + GEO_GRADIENT * top_3d / 1000.0
    mu_co2_3d = compute_mu_co2_cp(pres_3d, temp_3d)

    mu_fields = {k: mu_co2_3d[:, :, k - 1] for k in INJ_LAYERS}
    cx = cx_3d[:, :, NK // 2]
    cy = cy_3d[:, :, NK // 2]
    bound_steps = meters_to_grid_steps(D_BOUND_M, cx, cy)
    mask = boundary_mask(bound_steps)
    return mu_fields, cx, cy, mask


def save_single(
    k: int,
    field: np.ndarray,
    norm: mcolors.Normalize,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    im = ax.pcolormesh(
        cx.T,
        cy.T,
        np.where(mask, field, np.nan).T,
        shading="auto",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        snap=True,
        cmap=CO2_VISCOSITY_CMAP,
        norm=norm,
    )
    plot_domain_outline(ax, cx, cy, mask)
    style_map_axes(ax)
    ax.set_title(f"CO$_2$ viscosity, k={k}")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label("CO$_2$ viscosity (mPa s)")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_panel(
    fields: dict[int, np.ndarray],
    norm: mcolors.Normalize,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17.0 / 2.54, 15.0 / 2.54), constrained_layout=True)
    last_im = None
    for ax, (k, field) in zip(axes.ravel(), fields.items()):
        last_im = ax.pcolormesh(
            cx.T,
            cy.T,
            np.where(mask, field, np.nan).T,
            shading="auto",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
            snap=True,
            cmap=CO2_VISCOSITY_CMAP,
            norm=norm,
        )
        plot_domain_outline(ax, cx, cy, mask)
        style_map_axes(ax)
        ax.set_title(f"k={k}")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.015)
        cbar.set_label("CO$_2$ viscosity (mPa s)")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_panel_clean(
    fields: dict[int, np.ndarray],
    norm: mcolors.Normalize,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(17.0 / 2.54, 14.0 / 2.54), constrained_layout=True)
    for ax, (k, field) in zip(axes.ravel(), fields.items()):
        ax.pcolormesh(
            cx.T,
            cy.T,
            np.where(mask, field, np.nan).T,
            shading="auto",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
            rasterized=True,
            snap=True,
            cmap=CO2_VISCOSITY_CMAP,
            norm=norm,
        )
        plot_domain_outline(ax, cx, cy, mask)
        ax.set_aspect("equal")
        ax.set_title(f"k={k}", fontsize=10, fontweight="semibold", pad=2)
        ax.set_axis_off()
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True, pad_inches=0.02)
    plt.close(fig)


def save_single_clean(
    field: np.ndarray,
    norm: mcolors.Normalize,
    cx: np.ndarray,
    cy: np.ndarray,
    mask: np.ndarray,
    out_png: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(4.0, 4.2), constrained_layout=True)
    ax.pcolormesh(
        cx.T,
        cy.T,
        np.where(mask, field, np.nan).T,
        shading="auto",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
        rasterized=True,
        snap=True,
        cmap=CO2_VISCOSITY_CMAP,
        norm=norm,
    )
    plot_domain_outline(ax, cx, cy, mask)
    ax.set_aspect("equal")
    ax.set_axis_off()
    fig.savefig(out_png, dpi=600, bbox_inches="tight", transparent=True, pad_inches=0.0)
    plt.close(fig)


def save_range_csv(fields: dict[int, np.ndarray], mask: np.ndarray, norm: mcolors.Normalize, path: Path) -> None:
    rows = ["layer,min_mPa_s,max_mPa_s,mean_mPa_s,colorbar_vmin_mPa_s,colorbar_vmax_mPa_s"]
    for k, field in fields.items():
        vals = field[mask & np.isfinite(field)]
        rows.append(
            f"{k},{np.nanmin(vals):.8f},{np.nanmax(vals):.8f},{np.nanmean(vals):.8f},"
            f"{norm.vmin:.8f},{norm.vmax:.8f}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fields, cx, cy, mask = load_fields()
    norm = percentile_norm(list(fields.values()), mask_2d=mask, qlo=2.0, qhi=98.0)

    print("[3] Saving recolored CO2 viscosity figures...")
    panel_png = OUT_DIR / "co2_viscosity_injection_layers_3x3_newcolor.png"
    save_panel(fields, norm, cx, cy, mask, panel_png)
    save_panel(
        fields,
        norm,
        cx,
        cy,
        mask,
        FIG_DIR / panel_png.name,
    )

    clean_panel_png = OUT_DIR / "co2_viscosity_injection_layers_3x3_newcolor_clean.png"
    save_panel_clean(fields, norm, cx, cy, mask, clean_panel_png)
    save_panel_clean(
        fields,
        norm,
        cx,
        cy,
        mask,
        FIG_DIR / clean_panel_png.name,
    )

    for k, field in fields.items():
        save_single(
            k,
            field,
            norm,
            cx,
            cy,
            mask,
            OUT_DIR / f"co2_viscosity_injection_k{k:03d}_newcolor.png",
        )
        save_single_clean(
            field,
            norm,
            cx,
            cy,
            mask,
            OUT_DIR / f"co2_viscosity_injection_k{k:03d}_clean_notext.png",
        )
        save_single_clean(
            field,
            norm,
            cx,
            cy,
            mask,
            FIG_DIR / f"co2_viscosity_injection_k{k:03d}_clean_notext.png",
        )

    save_range_csv(fields, mask, norm, OUT_DIR / "co2_viscosity_injection_newcolor_ranges.csv")
    save_range_csv(fields, mask, norm, FIG_DIR / "co2_viscosity_injection_newcolor_ranges.csv")
    print(f"Saved to: {OUT_DIR}")
    print(f"Copied panel to: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
