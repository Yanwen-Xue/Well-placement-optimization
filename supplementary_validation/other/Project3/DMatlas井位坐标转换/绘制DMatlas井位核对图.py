from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(r"F:\other\Project3")
OUT = ROOT / "DMatlas井位坐标转换" / "DMatlas井位坐标核对.png"

surface = np.loadtxt(ROOT / "VarThicknessSurferFile" / "TopName_1.dat")
x = np.unique(surface[:, 0])
y = np.unique(surface[:, 1])
z = surface[:, 2].reshape(len(y), len(x))

wells = [
    ("Case 7 INJ2", 19_410_459.770, 4_358_242.225, "#c64040", "o"),
    ("Case 8 INJ1", 19_407_005.230, 4_363_124.625, "#2f6f9f", "s"),
]

fig, ax = plt.subplots(figsize=(8.0, 6.4))
mesh = ax.pcolormesh(x, y, z, shading="auto", cmap="terrain")

for name, xw, yw, color, marker in wells:
    ax.scatter(xw, yw, s=62, marker=marker, color=color, edgecolor="white", linewidth=1.0, zorder=3)
    ax.annotate(name, (xw, yw), xytext=(7, 7), textcoords="offset points", fontsize=9, color=color)

ax.set_xlabel("DMatlas X (m)")
ax.set_ylabel("DMatlas Y (m)")
ax.set_aspect("equal", adjustable="box")
ax.ticklabel_format(style="plain", axis="both", useOffset=False)
fig.colorbar(mesh, ax=ax, label="Top-surface elevation (m)")
fig.tight_layout()
fig.savefig(OUT, dpi=300, facecolor="white")
plt.close(fig)

print(OUT)
