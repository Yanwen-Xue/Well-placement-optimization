from pathlib import Path

import numpy as np


# CMG 平面坐标到 DMatlas 全局米制坐标的平移量。
# 该数值由 8 个对应地层面和 48000 个有效控制点联合配准得到。
X_OFFSET_M = 19_398_899.270
Y_OFFSET_M = 4_345_621.425

SEEDS_FILE = Path(r"F:\CODE1\opt_seeds_0_683_current_mixed_active60_20260813.npz")


def cmg_grid_to_dmatlas(i_1based, j_1based):
    """把 CMG 的 1-based (I, J) 网格编号转换成 DMatlas 的 X/Y 米制坐标。"""
    data = np.load(SEEDS_FILE, allow_pickle=True)

    # Python 数组从 0 开始，所以 CMG 编号需要减 1。
    i = i_1based - 1
    j = j_1based - 1

    # X/Y 对所有垂向层相同，这里读取第 41 层的网格中心坐标。
    x_cmg = float(data["cx_3d"][i, j, 40])
    y_cmg = float(data["cy_3d"][i, j, 40])

    x_dmatlas = x_cmg + X_OFFSET_M
    y_dmatlas = y_cmg + Y_OFFSET_M
    return x_cmg, y_cmg, x_dmatlas, y_dmatlas


if __name__ == "__main__":
    wells = [
        ("Case 7 INJ2", 188, 160),
        ("Case 8 INJ1", 69, 149),
    ]

    for name, i, j in wells:
        x0, y0, x, y = cmg_grid_to_dmatlas(i, j)
        print(f"{name}: CMG=({x0:.3f}, {y0:.3f}) m, DMatlas=({x:.3f}, {y:.3f}) m")
