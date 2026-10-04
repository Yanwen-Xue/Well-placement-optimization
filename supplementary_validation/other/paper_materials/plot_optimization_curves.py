"""
绘制 NSGA-III 优化收敛曲线（HV / 前沿大小 / 各目标极值 / HV 单代增益）。

用法：
    python paper_materials/plot_optimization_curves.py                       # 自动用最新 ckpt
    python paper_materials/plot_optimization_curves.py path/to/ckpt_genXXXX.npz

输出：与 ckpt 同目录的 optimization_curves.png
"""
import os, sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parent.parent
OPT_RUNS_DIR = str(ROOT / "opt_runs")
HV_TOL_PCT   = 0.5   # 与 optimizer.py 的 HV_TOL=0.005 对齐


def find_latest_ckpt(opt_dir):
    cands = sorted(glob.glob(os.path.join(opt_dir, "ckpt_gen*.npz")))
    return cands[-1] if cands else None


def main():
    if len(sys.argv) > 1:
        ckpt = sys.argv[1]
    else:
        ckpt = find_latest_ckpt(OPT_RUNS_DIR)
        if ckpt is None:
            print(f"[ERR] {OPT_RUNS_DIR} 下找不到 ckpt_gen*.npz")
            sys.exit(1)

    print(f"读取检查点: {ckpt}")
    data = np.load(ckpt)

    if "hv_history" not in data.files:
        print("[ERR] 该 ckpt 没有 hv_history（旧版 optimizer 跑出的检查点）")
        print("      请用新版 optimizer.py 重新跑一次再画曲线")
        sys.exit(1)

    hv = np.asarray(data["hv_history"], dtype=float)
    fs = np.asarray(data["front_size_history"], dtype=float)
    om = np.asarray(data["obj_min_history"], dtype=float)   # (N, 3)
    n  = len(hv)
    gens = np.arange(n)   # 0=初始种群，i>=1 = 第 i 代演化后

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1. HV 收敛曲线
    ax = axes[0, 0]
    ax.plot(gens, hv, '-o', color='C0', markersize=3, lw=1.5)
    ax.set_xlabel("代数")
    ax.set_ylabel("Hypervolume")
    ax.set_title("HV 收敛曲线（越大越好）")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. 前沿大小
    ax = axes[0, 1]
    ax.plot(gens, fs, '-s', color='C1', markersize=3, lw=1.5)
    ax.set_xlabel("代数")
    ax.set_ylabel("|Front 0|")
    ax.set_title("Pareto 前沿大小")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 3. 各目标极值（归一化到 [0,1] 便于同图比较）
    obj_labels = ["F1: 压力增幅", "F2: -溶解封存比例", "F3: -咸水采出"]
    ax = axes[1, 0]
    for k in range(3):
        col = om[:, k]
        rng = col.max() - col.min()
        norm = (col - col.min()) / rng if rng > 1e-9 else np.zeros_like(col)
        ax.plot(gens, norm, '-', label=obj_labels[k], lw=1.5)
    ax.set_xlabel("代数")
    ax.set_ylabel("归一化目标极值（0=最优，1=最差）")
    ax.set_title("各目标极值演化")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 4. HV 单代相对增益（早停判据可视化）
    ax = axes[1, 1]
    if n > 1:
        rel_gain = np.diff(hv) / np.maximum(np.abs(hv[:-1]), 1e-9) * 100
        ax.plot(gens[1:], rel_gain, '-^', color='C2', markersize=3, lw=1.2)
        ax.axhline(HV_TOL_PCT, color='r', linestyle='--', lw=1, alpha=0.7,
                   label=f"HV_TOL = {HV_TOL_PCT}%")
        ax.axhline(0, color='gray', linestyle='-', lw=0.5, alpha=0.5)
        ax.set_xlabel("代数")
        ax.set_ylabel("HV 相对增益 (%/代)")
        ax.set_title("HV 单代增益（低于阈值线即满足收敛条件之一）")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle(f"优化收敛诊断 — {os.path.basename(ckpt)}  (n_records={n})",
                 fontsize=12, y=1.00)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(ckpt) or ".", "optimization_curves.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"[OK] 输出: {out}")
    print(f"     总记录数 = {n}（含初始种群=记录0）")
    print(f"     最终 HV = {hv[-1]:.4e}")
    print(f"     最终前沿大小 = {int(fs[-1])}")
    if n >= 16:
        recent = hv[-15:]
        gain = (recent[-1] - recent[0]) / max(abs(recent[0]), 1e-9) * 100
        print(f"     近 15 代 HV 相对增益 = {gain:+.2f}%")


if __name__ == "__main__":
    main()
