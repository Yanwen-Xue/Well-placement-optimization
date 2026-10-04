from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(r"F:\CODE1")
SEEDS = ROOT / "opt_seeds.npz"
MANIFEST = Path(r"F:\CODE\compare\pareto_initial_vs_final\manifest.json")
OUT_DIR = Path(r"C:\Users\admin\Desktop\paper_figure\writen\Paper_Figure")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NAME = "pareto_initial_final_prescreen_mapping"


def physical_xy(cx: np.ndarray, cy: np.ndarray, wells: list[list[int]]) -> np.ndarray:
    return np.asarray([[cx[i, j], cy[i, j]] for i, j in wells], dtype=float)


def main() -> None:
    z = np.load(SEEDS, allow_pickle=True)
    inj_map = np.asarray(z["J_inj_2d_smooth"], dtype=float)
    prod_map = np.asarray(z["J_prod_2d_smooth"], dtype=float)
    bound = np.asarray(z["bound_mask"], dtype=bool)
    cx = np.asarray(z["cx_3d"][:, :, 40], dtype=float)
    cy = np.asarray(z["cy_3d"][:, :, 40], dtype=float)
    with MANIFEST.open(encoding="utf-8") as f:
        cases = {row["case_key"]: row for row in json.load(f)["cases"]}

    colors = {"Gen0_P060": "#4D718A", "Gen80_P060": "#B07A45"}
    labels = {"Gen0_P060": "Gen0--P060", "Gen80_P060": "Gen80--P060"}
    maps = [(inj_map, "Injection-side prescreen score"),
            (prod_map, "Production-side prescreen score")]
    values = [np.where(bound, field, np.nan) for field, _ in maps]
    levels = [np.linspace(np.nanpercentile(field, 2), np.nanpercentile(field, 98), 16)
              for field in values]

    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.0), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.91,
                        hspace=0.28, wspace=0.16)

    for row, (case_key, case) in enumerate(cases.items()):
        for col, (field, title) in enumerate(maps):
            ax = axes[row, col]
            value = values[col]
            mesh = ax.contourf(cx, cy, value, levels=levels[col], cmap="viridis")
            ax.contour(cx, cy, bound.astype(float), levels=[0.5],
                       colors=["#4b4b4b"], linewidths=0.55, linestyles="--")
            inj = physical_xy(cx, cy, case["injectors_zero_based"])
            prod = physical_xy(cx, cy, case["producers_zero_based"])
            color = colors[case_key]
            ax.scatter(inj[:, 0], inj[:, 1], s=28, marker="o", color=color,
                       edgecolor="white", linewidth=0.65, zorder=4)
            ax.scatter(prod[:, 0], prod[:, 1], s=31, marker="s", color=color,
                       edgecolor="white", linewidth=0.65, zorder=4)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{labels[case_key]}: {title}", fontsize=8.8)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")
            ax.tick_params(width=0.65, length=3)
            for spine in ax.spines.values():
                spine.set_linewidth(0.7)
            cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6.8, width=0.5, length=2)
            cbar.set_label("score", fontsize=7.5)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4D718A",
               markeredgecolor="white", markersize=6.5, label="Gen0--P060"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#B07A45",
               markeredgecolor="white", markersize=6.5, label="Gen80--P060"),
        Line2D([0], [0], marker="o", color="#555555", linestyle="None",
               markersize=5.5, label="Injector"),
        Line2D([0], [0], marker="s", color="#555555", linestyle="None",
               markersize=5.5, label="Producer"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.48, 0.975),
               ncol=4, frameon=False, fontsize=8, columnspacing=1.0,
               handletextpad=0.35)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(OUT_DIR / f"{OUT_NAME}.{ext}", dpi=450 if ext == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
