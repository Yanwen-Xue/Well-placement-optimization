from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon, Patch


SEEDS = Path(r"F:\CODE1\opt_seeds.npz")
GRID_CACHE = Path(r"C:\Users\admin\Desktop\paper_figure\figure\prescreen_grid_xy_cache.npz")
MANIFEST = Path(r"F:\CODE\compare\pareto_initial_vs_final\manifest.json")
OUT_DIR = Path(r"C:\Users\admin\Desktop\paper_figure\writen\Paper_Figure")
OUT_NAME = "pareto_initial_final_prescreen_mapping"


def robust_range(data: np.ndarray, mask: np.ndarray, qlo: float = 3.0, qhi: float = 99.0) -> tuple[float, float]:
    values = np.asarray(data)[mask & np.isfinite(data)]
    lo, hi = np.percentile(values, [qlo, qhi])
    return float(lo), float(hi)


def well_centres(wells: list[list[int]], cx: np.ndarray, cy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact center coordinates of the supplied zero-based model cells."""
    ij = np.asarray(wells, dtype=int)
    if ij.ndim != 2 or ij.shape[1] != 2:
        raise ValueError("well coordinates must be an n x 2 array")
    if np.any(ij[:, 0] < 0) or np.any(ij[:, 0] >= cx.shape[0]) or np.any(ij[:, 1] < 0) or np.any(ij[:, 1] >= cx.shape[1]):
        raise ValueError(f"well coordinate outside map grid: {wells}")
    return cx[ij[:, 0], ij[:, 1]], cy[ij[:, 0], ij[:, 1]]


def main() -> None:
    with MANIFEST.open(encoding="utf-8") as f:
        cases = {row["case_key"]: row for row in json.load(f)["cases"]}

    seeds = np.load(SEEDS)
    grid = np.load(GRID_CACHE)
    inj_map = np.asarray(seeds["J_inj_2d_smooth"], dtype=float)
    prod_map = np.asarray(seeds["J_prod_2d_smooth"], dtype=float)
    bound_mask = np.asarray(seeds["bound_mask"], dtype=bool)
    cx = np.asarray(grid["cx_2d"], dtype=float)
    cy = np.asarray(grid["cy_2d"], dtype=float)

    if inj_map.shape != cx.shape or prod_map.shape != cx.shape or bound_mask.shape != cx.shape:
        raise ValueError("prescreen fields and coordinate grid do not have the same shape")

    # Keep well-type colors fixed across generations; row labels identify the layout.
    colors = {"injection": "#b34b3c", "production": "#2f6685"}
    labels = {"Gen0_P060": "Gen0--P060", "Gen80_P060": "Gen80--P060"}
    maps = {"injection": inj_map, "production": prod_map}
    ranges = {
        "injection": robust_range(inj_map, bound_mask),
        "production": robust_range(prod_map, bound_mask),
    }

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.065, top=0.885,
                        hspace=0.24, wspace=0.16)
    validation = {"grid_shape": list(cx.shape), "cases": {}}

    for row, (case_key, case) in enumerate(cases.items()):
        validation["cases"][case_key] = {"injection": [], "production": []}
        for col, side in enumerate(("injection", "production")):
            ax = axes[row, col]
            data = maps[side]
            vmin, vmax = ranges[side]
            # Show the 1000 m model-edge buffer explicitly. Candidate wells are
            # drawn only inside the inner boundary represented by bound_mask.
            ax.pcolormesh(
                cx.T,
                cy.T,
                np.where(~bound_mask, 1.0, np.nan).T,
                cmap=ListedColormap(["#e9ecef"]),
                shading="auto",
                zorder=0,
            )
            ax.pcolormesh(
                cx.T,
                cy.T,
                np.where(bound_mask, data, np.nan).T,
                cmap="hot_r" if side == "injection" else "Blues",
                shading="auto",
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.contour(cx.T, cy.T, bound_mask.T.astype(float), levels=[0.5], colors=["#333333"], linewidths=0.55)
            outer_corners = [
                (cx[0, 0], cy[0, 0]),
                (cx[-1, 0], cy[-1, 0]),
                (cx[-1, -1], cy[-1, -1]),
                (cx[0, -1], cy[0, -1]),
            ]
            ax.add_patch(
                Polygon(
                    outer_corners,
                    closed=True,
                    fill=False,
                    edgecolor="#555555",
                    linewidth=0.8,
                    zorder=3,
                )
            )

            # Each map contains only its corresponding well type.
            wells = case["injectors_zero_based"] if side == "injection" else case["producers_zero_based"]
            ij = np.asarray(wells, dtype=int)
            if not np.all(bound_mask[ij[:, 0], ij[:, 1]]):
                raise ValueError(f"{case_key} contains a well outside the valid prescreen grid on {side} map")
            x, y = well_centres(wells, cx, cy)
            marker = "o" if side == "injection" else "s"
            color = colors[side]
            ax.scatter(x, y, s=20, marker=marker, color=color, edgecolor="white", linewidth=0.55, zorder=4)

            validation["cases"][case_key][side] = [
                {"i": int(i), "j": int(j), "x_m": float(xv), "y_m": float(yv)}
                for (i, j), xv, yv in zip(wells, x, y)
            ]

            ax.set_aspect("equal")
            ax.set_xlim(float(np.nanmin(cx)), float(np.nanmax(cx)))
            ax.set_ylim(float(np.nanmin(cy)), float(np.nanmax(cy)))
            ax.tick_params(direction="out", length=2.2, width=0.55, labelsize=7, pad=1.5)
            ax.set_xlabel("X (m)", fontsize=8, labelpad=1.5)
            ax.set_ylabel("Y (m)", fontsize=8, labelpad=1.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.55)
                spine.set_color("#444444")
            if row == 0:
                ax.set_title(f"{side.capitalize()} prescreen map", fontsize=9, pad=5)
            ax.text(
                0.03,
                0.04,
                labels[case_key],
                transform=ax.transAxes,
                fontsize=8.5,
                color="#222222",
                weight="bold",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=2.0),
            )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["injection"],
               markeredgecolor="white", markersize=6, label="Injector"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=colors["production"],
               markeredgecolor="white", markersize=6, label="Producer"),
        Patch(facecolor="#e9ecef", edgecolor="#555555", linewidth=0.5, label="1000 m edge buffer"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.975),
               ncol=3, frameon=False, fontsize=8.2, columnspacing=1.2, handletextpad=0.4)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{OUT_NAME}.{ext}", dpi=450 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (OUT_DIR / f"{OUT_NAME}_grid_validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
