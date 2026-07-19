\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter
import os
import sys
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.titleweight'] = 'semibold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['savefig.facecolor'] = 'white'


POISSON_RATIO    = 0.29
RHO_OVERBURDEN   = 2500.0


C_SALINITY       = 1.07


INJ_K_START      = 41
INJ_K_END        = 80
PRD_K_START      = 1
PRD_K_END        = 15


Q_DAY            = 288392.0


RE_RW_RATIO      = 500.0
LN_RE_RW         = np.log(RE_RW_RATIO)


BG_CO2           = 0.003
BHP_SAFETY_COEFF = 0.90
RHO_CO2_COLUMN   = 700.0


T_SEA_LEVEL_C    = 15.0
GEO_GRADIENT     = 33.3


P_MIN_PROD       = 8.0


D_BOUND_M        = 1000.0


D_INJ_INJ_M      = 200.0

D_PROD_PROD_M    = 200.0

D_INJ_PROD_M     = 300.0


N_INJ_WELLS      = 5
N_PROD_WELLS     = 5
N_WELL_SETS      = 50


SAMPLING_ALPHA        = 2.0
SPATIAL_REPULSION_SIGMA = 30


DISTANCE_METHOD  = 'euclidean'


SMOOTH_SIGMA     = 6


CMG_MODEL_NAME   = os.environ.get("CMG_MODEL_NAME", "model")
DATA_DIR         = os.environ.get("CMG_PROPERTY_DIR", os.path.join(ROOT_DIR, "property_CMG"))
OUTPUT_DIR       = os.environ.get("PRESCREEN_OUTPUT_DIR", ROOT_DIR)


G                = 9.81
RHO_WATER        = 1000.0
UNIT_CONV        = 1e-6
DARCY_CONV       = 0.0853


_PFX = os.environ.get("CMG_PROPERTY_PREFIX", f"{CMG_MODEL_NAME}_property ")
_SFX = os.environ.get("CMG_PROPERTY_SUFFIX", "")
FILES = {
    'pressure'   : os.path.join(DATA_DIR, f"{_PFX}Pressure{_SFX}"),
    'thickness'  : os.path.join(DATA_DIR, f"{_PFX}Grid Thickness{_SFX}"),
    'top'        : os.path.join(DATA_DIR, f"{_PFX}Grid Top{_SFX}"),
    'bottom'     : os.path.join(DATA_DIR, f"{_PFX}Grid Bottom{_SFX}"),
    'kr_gas'     : os.path.join(DATA_DIR, f"{_PFX}Gas Relative Perm{_SFX}"),
    'mu_gas'     : os.path.join(DATA_DIR, f"{_PFX}Gas Viscosity{_SFX}"),
    'mu_water'   : os.path.join(DATA_DIR, f"{_PFX}Water Viscosity{_SFX}"),
    'sw'         : os.path.join(DATA_DIR, f"{_PFX}Water Saturation{_SFX}"),
    'cx'         : os.path.join(DATA_DIR, f"{_PFX}Grid Centroid X{_SFX}"),
    'cy'         : os.path.join(DATA_DIR, f"{_PFX}Grid Centroid Y{_SFX}"),
    'poro'       : os.environ.get("CMG_POROSITY_FILE", os.path.join(DATA_DIR, "porosity.txt")),
    'perm'       : os.environ.get("CMG_PERMEABILITY_FILE", os.path.join(DATA_DIR, "permeability.txt")),
}


def read_gslib(filepath, value_col=8):
\
\
\

    print(f"  读取: {os.path.basename(filepath)}")
    df = pd.read_csv(filepath, sep=r'\s+', skiprows=9, header=None,
                     names=['i','j','k','x','y','z','value'])

    expected = 268 * 203 * 200
    if len(df) != expected:
        print(f"  [警告] 行数 {len(df)} 不等于预期 {expected}")
    return df


def read_cmg_txt(filepath, NI=204, NJ=215, NK=197):
\
\
\
\

    print(f"  读取: {os.path.basename(filepath)}")
    data = np.zeros((NI, NJ, NK), dtype=np.float64)
    with open(filepath, 'r') as f:
        lines = f.readlines()

    current_k = None
    current_j = None
    i_values = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith('**') and 'K =' in line and 'J =' in line:

            if current_k is not None and current_j is not None and i_values:
                arr = np.array(i_values[:NI])
                if len(arr) == NI:
                    data[:, current_j-1, current_k-1] = arr

            try:
                parts = line.replace(',', ' ').split()
                k_idx = parts.index('K') + 2
                j_idx = parts.index('J') + 2
                current_k = int(parts[k_idx])
                current_j = int(parts[j_idx])
                i_values = []
            except:
                current_k = None
                current_j = None
        elif line.startswith('**') or line.startswith('RESULTS') or line.startswith('SW') or \
             line.startswith('Gas') or line.startswith('Water'):
            continue
        else:

            try:
                vals = [float(v) for v in line.split()]
                i_values.extend(vals)
            except:
                continue


    if current_k is not None and current_j is not None and i_values:
        arr = np.array(i_values[:NI])
        if len(arr) == NI:
            data[:, current_j-1, current_k-1] = arr

    print(f"    形状: {data.shape}, 范围: [{data.min():.4f}, {data.max():.4f}]")
    return data


def read_property_txt(filepath, NI, NJ, NK):
\
\
\

    print(f"  读取: {os.path.basename(filepath)}")
    with open(filepath, 'r') as f:
        first_line = f.readline().strip()
        dims = list(map(int, first_line.split()))
        print(f"    文件声明网格: {dims}")
        vals = []
        for line in f:
            parts = line.strip().split()
            if parts:
                vals.append(float(parts[0]))
    data = np.array(vals, dtype=np.float64)
    total = NI * NJ * NK
    if len(data) != total:
        print(f"  [警告] 数据量 {len(data)} != {total}")
        data = data[:total]
    return data.reshape((NI, NJ, NK), order='F')


def main():
    NI, NJ, NK = 268, 203, 200
    print("=" * 70)
    print("CO2 注入潜力计算程序")
    print("=" * 70)


    print("\n[步骤0] 读取CMG数据...")

    pres_3d    = read_cmg_txt(FILES['pressure'],  NI, NJ, NK) * 1e-3
    thick_3d   = read_cmg_txt(FILES['thickness'], NI, NJ, NK)
    top_3d     = read_cmg_txt(FILES['top'],        NI, NJ, NK)
    cx_3d      = read_cmg_txt(FILES['cx'],         NI, NJ, NK)
    cy_3d      = read_cmg_txt(FILES['cy'],         NI, NJ, NK)
    mu_co2_3d  = read_cmg_txt(FILES['mu_gas'],    NI, NJ, NK)
    mu_brine_3d= read_cmg_txt(FILES['mu_water'],  NI, NJ, NK)
    sw_3d      = read_cmg_txt(FILES['sw'],         NI, NJ, NK)

    print("\n  读取渗透率和孔隙度（268×203×200，与CMG同网格，直接读取）...")
    perm_3d = read_property_txt(FILES['perm'], NI, NJ, NK)
    poro_3d = read_property_txt(FILES['poro'], NI, NJ, NK)

    perm_3d[perm_3d < 1e-6] = 0.0
    poro_3d[poro_3d < 1e-6] = 0.0

    print(f"    渗透率范围: [{perm_3d.min():.4e}, {perm_3d.max():.4f}] mD")
    print(f"    孔隙度范围: [{poro_3d.min():.4f}, {poro_3d.max():.4f}]")


    if pres_3d.max() > 200 or pres_3d.max() < 1:
        print(f"[错误] 压力范围异常: {pres_3d.min():.2f}~{pres_3d.max():.2f} MPa，请检查数据")
        sys.exit(1)
    if mu_co2_3d.max() > 1.0 or mu_co2_3d.min() < 0:
        print(f"[错误] CO2粘度范围异常 (CMG输出): {mu_co2_3d.min():.4f}~{mu_co2_3d.max():.4f} cP")
        sys.exit(1)


    print("\n[步骤0.5] 计算物理温度场 + CO2 粘度 μ(P,T) [via CoolProp NIST 标准]...")


    T_3d = T_SEA_LEVEL_C + GEO_GRADIENT * top_3d / 1000.0

    inj_k0_tmp = INJ_K_START - 1
    inj_k1_tmp = INJ_K_END   - 1
    print(f"    全场 T_3d  范围: [{T_3d.min():.2f}, {T_3d.max():.2f}] °C")
    print(f"    INJ 段 T_3d 范围: [{T_3d[:,:,inj_k0_tmp:inj_k1_tmp+1].min():.2f}, "
          f"{T_3d[:,:,inj_k0_tmp:inj_k1_tmp+1].max():.2f}] °C  (k={INJ_K_START}~{INJ_K_END})")


    from CoolProp.CoolProp import PropsSI
    from scipy.interpolate import RegularGridInterpolator


    res_p_min = pres_3d[pres_3d > 0].min() * 1e6
    res_p_max = pres_3d.max() * 1e6
    P_grid_Pa = np.linspace(max(1e6, res_p_min*0.9), res_p_max*1.1, 60)
    T_grid_K  = np.linspace(T_3d.min()+273.15-3, T_3d.max()+273.15+3, 60)

    print(f"    构建 μ(P,T) 查表: P∈[{P_grid_Pa.min()/1e6:.1f},{P_grid_Pa.max()/1e6:.1f}] MPa "
          f"× T∈[{T_grid_K.min()-273.15:.1f},{T_grid_K.max()-273.15:.1f}] °C  "
          f"({len(P_grid_Pa)}×{len(T_grid_K)} 点)...")
    mu_table_Pas = np.zeros((len(P_grid_Pa), len(T_grid_K)))
    for i, p in enumerate(P_grid_Pa):
        for j, t in enumerate(T_grid_K):
            try:
                mu_table_Pas[i, j] = PropsSI('V', 'P', p, 'T', t, 'CO2')
            except Exception:
                mu_table_Pas[i, j] = np.nan

    if np.isnan(mu_table_Pas).any():
        from scipy.ndimage import distance_transform_edt
        nan_mask = np.isnan(mu_table_Pas)
        _, idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        mu_table_Pas = mu_table_Pas[tuple(idx)]
    mu_table_cP = mu_table_Pas * 1e3

    interp_mu = RegularGridInterpolator(
        (P_grid_Pa, T_grid_K), mu_table_cP,
        bounds_error=False, fill_value=None,
    )


    P_input_Pa = pres_3d * 1e6
    T_input_K  = T_3d   + 273.15

    valid_P = P_input_Pa > 1e3
    P_eff = np.where(valid_P, P_input_Pa, P_grid_Pa[0])
    pts   = np.stack([P_eff.flatten(), T_input_K.flatten()], axis=-1)
    mu_co2_phys_3d = interp_mu(pts).reshape(P_input_Pa.shape)
    mu_co2_phys_3d = np.where(valid_P, mu_co2_phys_3d, np.nan)

    inj_slab = mu_co2_phys_3d[:,:,inj_k0_tmp:inj_k1_tmp+1]
    print(f"    物理 μ_CO2 全场范围: [{np.nanmin(mu_co2_phys_3d):.4f}, "
          f"{np.nanmax(mu_co2_phys_3d):.4f}] cP")
    print(f"    物理 μ_CO2 INJ 段范围: [{np.nanmin(inj_slab):.4f}, "
          f"{np.nanmax(inj_slab):.4f}] cP  (均值 {np.nanmean(inj_slab):.4f})")
    print(f"    对比 CMG 输出（无气格子=0）: [{mu_co2_3d.min():.4f}, "
          f"{mu_co2_3d.max():.4f}] cP")


    mu_co2_3d = mu_co2_phys_3d


    print("\n[步骤1] 计算网格坐标和步长...")


    k_mid = NK // 2
    cx_2d = cx_3d[:, :, k_mid]
    cy_2d = cy_3d[:, :, k_mid]


    di_avg = np.mean(np.sqrt(np.diff(cx_2d[:, NJ//2])**2 + np.diff(cy_2d[:, NJ//2])**2))
    dj_avg = np.mean(np.sqrt(np.diff(cx_2d[NI//2, :])**2 + np.diff(cy_2d[NI//2, :])**2))
    dx_avg = di_avg
    dy_avg = dj_avg
    dxy_avg = (dx_avg + dy_avg) / 2.0

    print(f"    I方向平均步长 di = {dx_avg:.1f} m")
    print(f"    J方向平均步长 dj = {dy_avg:.1f} m")


    def m_to_grids(dist_m, dx, dy):
        """将距离[m]换算为网格数（用平均步长，取上整保守估计）"""
        avg = (dx + dy) / 2.0
        return max(1, int(np.ceil(dist_m / avg)))

    D_BOUND_G    = m_to_grids(D_BOUND_M,    dx_avg, dy_avg)
    D_INJ_INJ_G  = m_to_grids(D_INJ_INJ_M,  dx_avg, dy_avg)
    D_PROD_PROD_G= m_to_grids(D_PROD_PROD_M, dx_avg, dy_avg)
    D_INJ_PROD_G = m_to_grids(D_INJ_PROD_M,  dx_avg, dy_avg)

    print(f"    边界安全距离:   {D_BOUND_M}m → {D_BOUND_G} 网格")
    print(f"    注入井间距:     {D_INJ_INJ_M}m → {D_INJ_INJ_G} 网格")
    print(f"    采出井间距:     {D_PROD_PROD_M}m → {D_PROD_PROD_G} 网格")
    print(f"    注采隔离距离:   {D_INJ_PROD_M}m → {D_INJ_PROD_G} 网格")


    print("\n[步骤2] 计算盖层破裂压力（Eaton公式，作用在 INJ perf 顶）...")


    inj_k0  = INJ_K_START - 1
    inj_k1  = INJ_K_END   - 1
    prd_k0  = PRD_K_START - 1
    prd_k1  = PRD_K_END   - 1


    H_SURFACE = 1200.0


    D_perf_top  = H_SURFACE + top_3d[:, :, inj_k1]
    D_perf_top  = np.maximum(D_perf_top, 0.0)
    sigma_v_2d  = RHO_OVERBURDEN * G * D_perf_top * UNIT_CONV


    REFPRES_kPa = 22929.0
    REFDEPTH_TVDSS_m = 987.241211
    RHO_BRINE = 1015.0
    BRINE_GRAD_kPa_m = RHO_BRINE * G / 1000.0
    Pp_caprock = (REFPRES_kPa + BRINE_GRAD_kPa_m * (top_3d[:, :, inj_k1] - REFDEPTH_TVDSS_m)) * 1e-3


    mu = POISSON_RATIO
    Pf_caprock = Pp_caprock + (sigma_v_2d - Pp_caprock) * (3.0 * mu) / (2.0 * (1.0 - mu))

    print(f"    D_perf_top:    [{D_perf_top.min():.0f}, {D_perf_top.max():.0f}] m")
    print(f"    σv:            [{sigma_v_2d.min():.2f}, {sigma_v_2d.max():.2f}] MPa")
    print(f"    Pp:            [{Pp_caprock.min():.2f}, {Pp_caprock.max():.2f}] MPa")
    print(f"    Pf_caprock:    [{Pf_caprock.min():.2f}, {Pf_caprock.max():.2f}] MPa")

    if Pf_caprock.max() > 100 or Pf_caprock.min() < 0:
        print(f"[警告] 破裂压力范围异常，请检查盖层深度和密度参数")


    tvdss_mid_3d = top_3d + 0.5 * thick_3d
    D_mid_3d = H_SURFACE + tvdss_mid_3d
    sigma_v_mid_3d = RHO_OVERBURDEN * G * D_mid_3d * UNIT_CONV
    Pp_mid_3d = (REFPRES_kPa + BRINE_GRAD_kPa_m * (tvdss_mid_3d - REFDEPTH_TVDSS_m)) * 1e-3
    Pf_mid_3d = Pp_mid_3d + (sigma_v_mid_3d - Pp_mid_3d) * (3.0 * mu) / (2.0 * (1.0 - mu))
    pf_mid_inj = Pf_mid_3d[:, :, inj_k0:inj_k1+1]
    print(f"    Pf_mid (INJ):   [{np.nanmin(pf_mid_inj):.2f}, {np.nanmax(pf_mid_inj):.2f}] MPa")


    print("\n[步骤3] 计算有效渗透率...")


    kh_3d = perm_3d.copy()


    keff_co2_3d = kh_3d.copy()


    keff_brine_3d = kh_3d.copy()

    print(f"    keff_CO2 范围: [{keff_co2_3d.min():.4e}, {keff_co2_3d.max():.4f}] mD")


    print("\n[步骤4] 计算注入压力（在 INJ 段 k={}-{}）...".format(INJ_K_START, INJ_K_END))

    N_inj_layers = inj_k1 - inj_k0 + 1


    KH_inj = np.zeros((NI, NJ))
    for k in range(inj_k0, inj_k1 + 1):
        KH_inj += keff_co2_3d[:, :, k] * thick_3d[:, :, k]


    KH_inj_safe = np.where(KH_inj > 1e-10, KH_inj, np.nan)


    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mu_co2_avg = np.nanmean(mu_co2_3d[:, :, inj_k0:inj_k1+1], axis=2)


    Pp_inj = Pp_caprock.copy()


    try:
        rho_table = np.zeros((len(P_grid_Pa), len(T_grid_K)))
        for i, p in enumerate(P_grid_Pa):
            for j, t in enumerate(T_grid_K):
                try:
                    rho_table[i, j] = PropsSI('D', 'P', p, 'T', t, 'CO2')
                except Exception:
                    rho_table[i, j] = np.nan
        if np.isnan(rho_table).any():
            from scipy.ndimage import distance_transform_edt
            nan_mask = np.isnan(rho_table)
            _, idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
            rho_table = rho_table[tuple(idx)]
        rho_std = 1.977
        bg_table = rho_std / rho_table
        interp_bg = RegularGridInterpolator(
            (P_grid_Pa, T_grid_K), bg_table,
            bounds_error=False, fill_value=None,
        )
        bg_co2_phys_3d = interp_bg(pts).reshape(P_input_Pa.shape)
        bg_co2_phys_3d = np.where(valid_P, bg_co2_phys_3d, np.nan)
    except Exception:
        bg_co2_phys_3d = np.full_like(pres_3d, BG_CO2, dtype=np.float64)
        bg_co2_phys_3d[pres_3d <= 0] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bg_co2_avg = np.nanmean(bg_co2_phys_3d[:, :, inj_k0:inj_k1+1], axis=2)

    Pinj = Pp_inj + (Q_DAY * bg_co2_avg * mu_co2_avg * LN_RE_RW) / (2 * np.pi * KH_inj_safe * DARCY_CONV)


    dP_safe = BHP_SAFETY_COEFF * Pf_caprock - Pinj

    print(f"    KH_inj 范围:   [{np.nanmin(KH_inj):.4e}, {np.nanmax(KH_inj):.4f}] mD·m")
    print(f"    Pinj 范围:     [{np.nanmin(Pinj):.2f}, {np.nanmax(Pinj):.2f}] MPa")
    print(f"    ΔPsafe 范围:   [{np.nanmin(dP_safe):.2f}, {np.nanmax(dP_safe):.2f}] MPa")
    print(f"    可注入格点数:  {np.sum(dP_safe > 0)} / {NI*NJ}")

    if np.sum(dP_safe > 0) == 0:
        print("[错误] 所有格点均无法注入（ΔPsafe≤0），请检查盖层/储层范围参数")
        sys.exit(1)


    print("\n[步骤5] 计算三维注入潜力（仅 INJ 段）...")

    tvdss_top_perf = top_3d[:, :, inj_k1]
    J_inj_3d = np.zeros((NI, NJ, NK))
    bhp_at_k_min = np.inf
    bhp_at_k_max = -np.inf
    dpsafe_k_min = np.inf
    dpsafe_k_max = -np.inf
    for k in range(inj_k0, inj_k1 + 1):
        mu_co2_k = mu_co2_3d[:, :, k]
        mu_co2_k_safe = np.where(mu_co2_k > 1e-10, mu_co2_k, np.nan)
        dz_k = tvdss_mid_3d[:, :, k] - tvdss_top_perf
        BHP_at_k = Pinj + RHO_CO2_COLUMN * G * dz_k * UNIT_CONV
        dP_safe_k = BHP_SAFETY_COEFF * Pf_mid_3d[:, :, k] - BHP_at_k
        bhp_at_k_min = min(bhp_at_k_min, float(np.nanmin(BHP_at_k)))
        bhp_at_k_max = max(bhp_at_k_max, float(np.nanmax(BHP_at_k)))
        dpsafe_k_min = min(dpsafe_k_min, float(np.nanmin(dP_safe_k)))
        dpsafe_k_max = max(dpsafe_k_max, float(np.nanmax(dP_safe_k)))
        mask = dP_safe_k > 0
        h    = thick_3d[:, :, k]
        phi  = poro_3d[:, :, k]
        keff = keff_co2_3d[:, :, k]
        J_k = dP_safe_k * keff * h * phi / mu_co2_k_safe
        J_k = np.where(mask, J_k, 0.0)
        J_k = np.nan_to_num(J_k, nan=0.0)
        J_inj_3d[:, :, k] = J_k

    print(f"    J_inj_3D 范围: [{J_inj_3d.min():.4e}, {J_inj_3d.max():.4f}]")
    print(f"    BHP_at_k:      [{bhp_at_k_min:.2f}, {bhp_at_k_max:.2f}] MPa")
    print(f"    ΔPsafe_k:      [{dpsafe_k_min:.2f}, {dpsafe_k_max:.2f}] MPa")


    print("\n[步骤6] 计算三维采出潜力（仅 PRD 段）...")


    J_prod_3d = np.zeros((NI, NJ, NK))
    for k in range(prd_k0, prd_k1 + 1):
        dP_prod_k = pres_3d[:, :, k] - P_MIN_PROD
        mask_prod = dP_prod_k > 0
        if not mask_prod.any():
            continue
        h    = thick_3d[:, :, k]
        phi  = poro_3d[:, :, k]
        keff = keff_brine_3d[:, :, k]
        mu_b = mu_brine_3d[:, :, k]
        mu_b_safe = np.where(mu_b > 1e-10, mu_b, np.nan)

        J_k = dP_prod_k * keff * h * phi / mu_b_safe
        J_k = np.where(mask_prod, J_k, 0.0)
        J_k = np.nan_to_num(J_k, nan=0.0)
        J_prod_3d[:, :, k] = J_k


    dP_prod = np.max(pres_3d[:, :, prd_k0:prd_k1+1], axis=2) - P_MIN_PROD

    print(f"    J_prod_3D 范围: [{J_prod_3d.min():.4e}, {J_prod_3d.max():.4f}]")


    print("\n[步骤7] 二维归一化...")

    J_global_max_inj  = J_inj_3d[:, :, inj_k0:inj_k1+1].max()
    J_global_max_prod = J_prod_3d[:, :, prd_k0:prd_k1+1].max()

    if J_global_max_inj <= 0:
        print("[错误] 注入潜力全为0，请检查参数")
        sys.exit(1)

    J_inj_2d  = np.zeros((NI, NJ))
    J_prod_2d = np.zeros((NI, NJ))
    for k in range(inj_k0, inj_k1 + 1):
        J_inj_2d  += J_inj_3d[:, :, k]  / J_global_max_inj
    for k in range(prd_k0, prd_k1 + 1):
        J_prod_2d += J_prod_3d[:, :, k] / J_global_max_prod

    J_inj_2d  /= N_inj_layers
    J_prod_2d /= (prd_k1 - prd_k0 + 1)

    print(f"    J_inj_2D  范围: [{J_inj_2d.min():.4f}, {J_inj_2d.max():.4f}]")
    print(f"    J_prod_2D 范围: [{J_prod_2d.min():.4f}, {J_prod_2d.max():.4f}]")


    if SMOOTH_SIGMA > 0:
        J_inj_2d_viz  = gaussian_filter(J_inj_2d,  sigma=SMOOTH_SIGMA)
        J_prod_2d_viz = gaussian_filter(J_prod_2d, sigma=SMOOTH_SIGMA)
        print(f"    高斯平滑 sigma={SMOOTH_SIGMA} 格 (~{SMOOTH_SIGMA*dxy_avg:.0f}m)")
        print(f"    平滑后 J_inj_2D  范围: [{J_inj_2d_viz.min():.4f}, {J_inj_2d_viz.max():.4f}]")
        print(f"    平滑后 J_prod_2D 范围: [{J_prod_2d_viz.min():.4f}, {J_prod_2d_viz.max():.4f}]")
    else:
        J_inj_2d_viz  = J_inj_2d.copy()
        J_prod_2d_viz = J_prod_2d.copy()


    dP_safe_2d = np.nan_to_num(np.maximum(dP_safe, 0.0), nan=0.0, posinf=0.0, neginf=0.0)
    inj_capacity_2d = np.zeros((NI, NJ), dtype=np.float64)
    for k in range(inj_k0, inj_k1 + 1):
        mu_co2_k = mu_co2_3d[:, :, k]
        mu_co2_k_safe = np.where(mu_co2_k > 1e-10, mu_co2_k, np.nan)
        inj_capacity_2d += np.nan_to_num(
            keff_co2_3d[:, :, k] * thick_3d[:, :, k] * poro_3d[:, :, k] / mu_co2_k_safe,
            nan=0.0, posinf=0.0, neginf=0.0
        )

    def robust_vrange(arr2d, mask2d, qlo=2.0, qhi=98.0):
        vals = arr2d[mask2d & np.isfinite(arr2d)]
        if vals.size == 0:
            return 0.0, 1.0
        vmin = float(np.percentile(vals, qlo))
        vmax = float(np.percentile(vals, qhi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        return vmin, vmax

    def style_map_axes(ax):
        ax.set_aspect('equal')
        ax.set_facecolor('#f7f4ef')
        ax.grid(False)
        ax.tick_params(direction='out', length=3.5, width=0.8, colors='#3a332c')
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color('#5b5248')

    def plot_domain_outline(ax):
        ax.contour(
            cx_2d.T, cy_2d.T, bound_mask.T.astype(float),
            levels=[0.5], colors=['#4a4037'], linewidths=0.8, alpha=0.85
        )

    def plot_soft_contours(ax, data2d, levels=7, color='#2c241f', alpha=0.22, linewidth=0.55):
        vals = data2d[bound_mask & np.isfinite(data2d)]
        if vals.size < 10:
            return
        lo = float(np.percentile(vals, 12))
        hi = float(np.percentile(vals, 96))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        cs = np.linspace(lo, hi, levels)
        ax.contour(
            cx_2d.T, cy_2d.T, np.where(bound_mask, data2d, np.nan).T,
            levels=cs, colors=[color], linewidths=linewidth, alpha=alpha
        )

    def robust_vrange_many(fields, mask2d, qlo=1.0, qhi=99.0):
        vals_all = []
        for field in fields:
            vals = field[mask2d & np.isfinite(field)]
            vals = vals[vals > 0]
            if vals.size > 0:
                vals_all.append(vals)
        if not vals_all:
            return 0.0, 1.0
        vals = np.concatenate(vals_all)
        vmin = float(np.percentile(vals, qlo))
        vmax = float(np.percentile(vals, qhi))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin = float(np.nanmin(vals))
            vmax = float(np.nanmax(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        return vmin, vmax


    print(f"\n[步骤8] 概率采样+空间排斥布设 {N_WELL_SETS} 套井位（{N_INJ_WELLS}注{N_PROD_WELLS}采）...")

    def exclusion_mask(wi, wj, min_grids, NI, NJ, method='euclidean'):
        II, JJ = np.meshgrid(np.arange(NI), np.arange(NJ), indexing='ij')
        if method == 'euclidean':
            dist = np.sqrt((II - wi)**2 + (JJ - wj)**2)
        else:
            dist = np.abs(II - wi) + np.abs(JJ - wj)
        return dist < min_grids

    def boundary_mask(NI, NJ, d_bound):
        II, JJ = np.meshgrid(np.arange(NI), np.arange(NJ), indexing='ij')
        d = np.minimum(np.minimum(II, NI-1-II), np.minimum(JJ, NJ-1-JJ))
        return d >= d_bound

    def sample_well(J_work, avail, alpha):
        """从可用格点中按J^alpha概率权重随机采样一口井位。"""
        if not avail.any():
            return None
        scores = np.where(avail, J_work, 0.0)
        scores = np.maximum(scores, 0.0)
        min_v = scores[avail].min()
        scores = np.where(avail, (scores - min_v + 1e-10) ** alpha, 0.0)
        total = scores.sum()
        if total <= 0:
            return None
        flat = scores.flatten() / total
        idx = np.random.choice(len(flat), p=flat)
        return np.unravel_index(idx, J_work.shape)

    def apply_repulsion(J_work, wi, wj, sigma, NI, NJ, strength=0.85):
        """在(wi,wj)附近施加高斯衰减排斥，推动后续井位去向其他区域。"""
        II, JJ = np.meshgrid(np.arange(NI), np.arange(NJ), indexing='ij')
        dist_sq = (II - wi)**2 + (JJ - wj)**2
        repulsion = np.exp(-dist_sq / (2.0 * sigma**2))
        return J_work * (1.0 - strength * repulsion)

    print("    计算边界掩码...")
    bound_mask = boundary_mask(NI, NJ, D_BOUND_G)
    print(f"    边界约束后可用格点: {bound_mask.sum()} / {NI*NJ}")

    inj_base_mask  = (dP_safe > 0) & bound_mask & (J_inj_2d_viz > 0)
    prod_base_mask = (dP_prod > 0) & bound_mask & (J_prod_2d_viz > 0)
    print(f"    基础可注入格点: {inj_base_mask.sum()}")
    print(f"    基础可采出格点: {prod_base_mask.sum()}")

    all_well_sets = []

    for s in range(N_WELL_SETS):
        np.random.seed(s * 31 + 7)


        J_inj_work  = J_inj_2d_viz.copy()
        J_prod_work = J_prod_2d_viz.copy()

        inj_wells  = []
        prod_wells = []


        for m in range(N_INJ_WELLS):
            avail = inj_base_mask.copy()
            for (wi, wj) in inj_wells:
                avail &= ~exclusion_mask(wi, wj, D_INJ_INJ_G,  NI, NJ, DISTANCE_METHOD)
            for (wi, wj) in prod_wells:
                avail &= ~exclusion_mask(wi, wj, D_INJ_PROD_G, NI, NJ, DISTANCE_METHOD)

            best = sample_well(J_inj_work, avail, SAMPLING_ALPHA)
            if best is None:
                print(f"    [警告] 方案{s+1} 注入井{m+1}: 无可用格点")
                break
            inj_wells.append(best)

            J_inj_work = apply_repulsion(J_inj_work, best[0], best[1],
                                         SPATIAL_REPULSION_SIGMA, NI, NJ)


        for n in range(N_PROD_WELLS):
            avail = prod_base_mask.copy()
            for (wi, wj) in prod_wells:
                avail &= ~exclusion_mask(wi, wj, D_PROD_PROD_G, NI, NJ, DISTANCE_METHOD)
            for (wi, wj) in inj_wells:
                avail &= ~exclusion_mask(wi, wj, D_INJ_PROD_G,  NI, NJ, DISTANCE_METHOD)

            best = sample_well(J_prod_work, avail, SAMPLING_ALPHA)
            if best is None:
                print(f"    [警告] 方案{s+1} 采出井{n+1}: 无可用格点")
                break
            prod_wells.append(best)
            J_prod_work = apply_repulsion(J_prod_work, best[0], best[1],
                                          SPATIAL_REPULSION_SIGMA, NI, NJ)

        P_bound = bound_mask.astype(float)
        J_inj_result  = J_inj_2d_viz  * P_bound
        J_prod_result = J_prod_2d_viz * P_bound
        for (wi, wj) in inj_wells:
            J_inj_result[exclusion_mask(wi, wj, D_INJ_INJ_G,  NI, NJ, DISTANCE_METHOD)] = 0.0
            J_prod_result[exclusion_mask(wi, wj, D_INJ_PROD_G, NI, NJ, DISTANCE_METHOD)] = 0.0

        all_well_sets.append({
            'set_id'    : s + 1,
            'inj_wells' : inj_wells,
            'prod_wells': prod_wells,
            'J_inj'     : J_inj_result,
            'J_prod'    : J_prod_result,
        })


    print(f"\n    生成 {len(all_well_sets)} 套井位种子：")
    for ws in all_well_sets:
        inj_str  = ', '.join([f"({w[0]+1},{w[1]+1})" for w in ws['inj_wells']])
        prod_str = ', '.join([f"({w[0]+1},{w[1]+1})" for w in ws['prod_wells']])
        print(f"    Set{ws['set_id']:02d} INJ:[{inj_str}]  PRD:[{prod_str}]")


    N_VIZ_SETS = N_WELL_SETS
    WELLSET_DIR = os.path.join(OUTPUT_DIR, "well_set_maps")
    os.makedirs(WELLSET_DIR, exist_ok=True)
    print(f"\n[Step 9] Generating per-set heatmaps for all {N_VIZ_SETS} sets -> {WELLSET_DIR} ...")

    from matplotlib.lines import Line2D as _L2D

    _smooth_str = (f'Gaussian \u03c3={SMOOTH_SIGMA} grids \u2248 {SMOOTH_SIGMA*dxy_avg:.0f} m'
                   if SMOOTH_SIGMA > 0 else 'No smoothing')
    _legend_elems = [
        _L2D([0],[0], marker='^', color='w', markerfacecolor='royalblue',
             markersize=11, label='Injection Well'),
        _L2D([0],[0], marker='o', color='w', markerfacecolor='limegreen',
             markersize=10, label='Production Well'),
    ]

    vmin_inj, vmax_inj = robust_vrange(J_inj_2d_viz, bound_mask)
    vmin_prod, vmax_prod = robust_vrange(J_prod_2d_viz, bound_mask)

    for ws in all_well_sets[:N_VIZ_SETS]:
        sid = ws['set_id']

        fig, (ax_inj, ax_prod) = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(
            f'CO$_2$ Injection Potential Analysis  \u2013  Well Set {sid}\n'
            f'Shenhua-Ordos Basin  |  CMG Grid {NI}×{NJ}×{NK}  |  {_smooth_str}',
            fontsize=13, fontweight='bold'
        )


        J_inj_plot = np.where(bound_mask, ws['J_inj'], np.nan)
        im1 = ax_inj.pcolormesh(cx_2d.T, cy_2d.T, J_inj_plot.T, cmap='hot_r',
                                 vmin=vmin_inj, vmax=vmax_inj, shading='auto')
        cb1 = plt.colorbar(im1, ax=ax_inj, shrink=0.90, pad=0.02)
        cb1.set_label('Normalized CO$_2$ Injection Potential', fontsize=11)
        ax_inj.set_title('CO$_2$ Injection Potential  $J_{inj,2D}$', fontsize=12, fontweight='bold')
        ax_inj.set_xlabel('X (m)', fontsize=11)
        ax_inj.set_ylabel('Y (m)', fontsize=11)
        ax_inj.set_aspect('equal')

        for k, (wi, wj) in enumerate(ws['inj_wells']):
            ax_inj.plot(cx_2d[wi,wj], cy_2d[wi,wj], '^', color='royalblue', markersize=11,
                        markeredgecolor='white', linewidth=0.7, zorder=6)
            ax_inj.annotate(
                f'INJ-{k+1}\nGrid({wi+1},{wj+1})',
                (cx_2d[wi,wj], cy_2d[wi,wj]), textcoords='offset points', xytext=(6, 5),
                fontsize=7.5, color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='navy', alpha=0.82, lw=0)
            )
        for k, (wi, wj) in enumerate(ws['prod_wells']):
            ax_inj.plot(cx_2d[wi,wj], cy_2d[wi,wj], 'o', color='limegreen', markersize=9,
                        markeredgecolor='white', linewidth=0.7, zorder=6)
            ax_inj.annotate(
                f'PRD-{k+1}\nGrid({wi+1},{wj+1})',
                (cx_2d[wi,wj], cy_2d[wi,wj]), textcoords='offset points', xytext=(6, 5),
                fontsize=7.5, color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='darkgreen', alpha=0.82, lw=0)
            )
        ax_inj.legend(handles=_legend_elems, loc='lower right', fontsize=9, framealpha=0.9)


        J_prod_plot = np.where(bound_mask, ws['J_prod'], np.nan)
        im2 = ax_prod.pcolormesh(cx_2d.T, cy_2d.T, J_prod_plot.T, cmap='Blues',
                                  vmin=vmin_prod, vmax=vmax_prod, shading='auto')
        cb2 = plt.colorbar(im2, ax=ax_prod, shrink=0.90, pad=0.02)
        cb2.set_label('Normalized Brine Production Potential', fontsize=11)
        ax_prod.set_title('Brine Production Potential  $J_{prod,2D}$', fontsize=12, fontweight='bold')
        ax_prod.set_xlabel('X (m)', fontsize=11)
        ax_prod.set_ylabel('Y (m)', fontsize=11)
        ax_prod.set_aspect('equal')

        for k, (wi, wj) in enumerate(ws['inj_wells']):
            ax_prod.plot(cx_2d[wi,wj], cy_2d[wi,wj], '^', color='royalblue', markersize=11,
                         markeredgecolor='white', linewidth=0.7, zorder=6)
            ax_prod.annotate(
                f'INJ-{k+1}\nGrid({wi+1},{wj+1})',
                (cx_2d[wi,wj], cy_2d[wi,wj]), textcoords='offset points', xytext=(6, 5),
                fontsize=7.5, color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='navy', alpha=0.82, lw=0)
            )
        for k, (wi, wj) in enumerate(ws['prod_wells']):
            ax_prod.plot(cx_2d[wi,wj], cy_2d[wi,wj], 'o', color='limegreen', markersize=9,
                         markeredgecolor='white', linewidth=0.7, zorder=6)
            ax_prod.annotate(
                f'PRD-{k+1}\nGrid({wi+1},{wj+1})',
                (cx_2d[wi,wj], cy_2d[wi,wj]), textcoords='offset points', xytext=(6, 5),
                fontsize=7.5, color='white', fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.2', fc='darkgreen', alpha=0.82, lw=0)
            )
        ax_prod.legend(handles=_legend_elems, loc='lower right', fontsize=9, framealpha=0.9)

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        fig_path = os.path.join(WELLSET_DIR, f'injection_potential_set{sid:02d}.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {fig_path}")


    print("\n[Step 10] Saving well location report...")

    out_txt = os.path.join(OUTPUT_DIR, "well_locations.txt")
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("CO2 Injection Potential -- Well Location Coordinate Report\n")
        f.write(f"Grid         : {NI} x {NJ} x {NK}  (CMG model)\n")
        f.write(f"INJ section  : k={INJ_K_START} to {INJ_K_END} (shiqianfeng, 1-based)\n")
        f.write(f"PRD section  : k={PRD_K_START} to {PRD_K_END} (shihezi3)\n")
        f.write(f"Distance method       : {DISTANCE_METHOD}\n")
        f.write(f"Min. inj-inj spacing  : {D_INJ_INJ_M} m  ({D_INJ_INJ_G} grids)\n")
        f.write(f"Min. inj-prod spacing : {D_INJ_PROD_M} m  ({D_INJ_PROD_G} grids)\n")
        f.write(f"Gaussian smoothing    : sigma={SMOOTH_SIGMA} grids"
                f"  (~{SMOOTH_SIGMA*dxy_avg:.0f} m)\n")
        f.write("=" * 70 + "\n\n")

        for ws in all_well_sets:
            sid = ws['set_id']
            f.write(f"[Set {sid}]\n")
            f.write("-" * 50 + "\n")
            f.write("Injection Wells:\n")
            for k, (wi, wj) in enumerate(ws['inj_wells']):
                x = cx_2d[wi, wj]
                y = cy_2d[wi, wj]
                j_val = J_inj_2d[wi, wj]
                f.write(f"  INJ-{k+1}: Grid(i={wi+1}, j={wj+1})  "
                        f"X={x:.1f} m, Y={y:.1f} m  "
                        f"Norm.Potential={j_val:.4f}\n")
            f.write("Production Wells:\n")
            for k, (wi, wj) in enumerate(ws['prod_wells']):
                x = cx_2d[wi, wj]
                y = cy_2d[wi, wj]
                j_val = J_prod_2d[wi, wj]
                f.write(f"  PRD-{k+1}: Grid(i={wi+1}, j={wj+1})  "
                        f"X={x:.1f} m, Y={y:.1f} m  "
                        f"Norm.Potential={j_val:.4f}\n")
            f.write("\n")

    print(f"    Well location report saved: {out_txt}")


    print("\n[Step 11] Generating comparison and final figures (English)...")

    from matplotlib.lines import Line2D

    ws0      = all_well_sets[0]
    inj_ws0  = ws0['inj_wells']
    prod_ws0 = ws0['prod_wells']

    legend_elems = [
        Line2D([0],[0], marker='^', color='w', markerfacecolor='#3566d6',
               markeredgecolor='white', markeredgewidth=1.0, markersize=10, label='Injection Well'),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='#4dbb63',
               markeredgecolor='white', markeredgewidth=1.0, markersize=9,  label='Production Well'),
    ]

    def mark_wells(ax, inj_ws, prod_ws, cx_2d, cy_2d, label_size=6):
        for k,(wi,wj) in enumerate(inj_ws):
            ax.plot(cx_2d[wi,wj], cy_2d[wi,wj], '^', color='#3566d6', markersize=8.5,
                    markeredgecolor='white', markeredgewidth=1.0, linewidth=0.6, zorder=6)
            ax.text(cx_2d[wi,wj]+95, cy_2d[wi,wj]+95, f'I{k+1}', color='white', fontsize=label_size,
                    fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.14', fc='#27408b', alpha=0.82, lw=0))
        for k,(wi,wj) in enumerate(prod_ws):
            ax.plot(cx_2d[wi,wj], cy_2d[wi,wj], 'o', color='#4dbb63', markersize=7.5,
                    markeredgecolor='white', markeredgewidth=1.0, linewidth=0.6, zorder=6)
            ax.text(cx_2d[wi,wj]+95, cy_2d[wi,wj]+95, f'P{k+1}', color='white', fontsize=label_size,
                    fontweight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.14', fc='#1f6b39', alpha=0.82, lw=0))

    def style_map_axes(ax):
        ax.set_aspect('equal')
        ax.set_facecolor('#f7f4ef')
        ax.grid(False)
        ax.tick_params(direction='out', length=3.5, width=0.8, colors='#3a332c')
        for spine in ax.spines.values():
            spine.set_linewidth(0.9)
            spine.set_color('#5b5248')

    def plot_domain_outline(ax):
        ax.contour(
            cx_2d.T, cy_2d.T, bound_mask.T.astype(float),
            levels=[0.5], colors=['#4a4037'], linewidths=0.8, alpha=0.85
        )

    def plot_soft_contours(ax, data2d, levels=7, color='#2c241f', alpha=0.22, linewidth=0.55):
        vals = data2d[bound_mask & np.isfinite(data2d)]
        if vals.size < 10:
            return
        lo = float(np.percentile(vals, 12))
        hi = float(np.percentile(vals, 96))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return
        cs = np.linspace(lo, hi, levels)
        ax.contour(
            cx_2d.T, cy_2d.T, np.where(bound_mask, data2d, np.nan).T,
            levels=cs, colors=[color], linewidths=linewidth, alpha=alpha
        )


    sigmas_all = [0, 2, 6, 10]
    sigma_labels = [
        'Raw',
        f'\u03c3 = 2\n(~{2*dxy_avg:.0f} m)',
        f'\u03c3 = 6\n(~{6*dxy_avg:.0f} m)',
        f'\u03c3 = 10\n(~{10*dxy_avg:.0f} m)',
    ]

    data_inj_all = [J_inj_2d.copy() if sigma == 0 else gaussian_filter(J_inj_2d, sigma=sigma)
                    for sigma in sigmas_all]
    data_prod_all = [J_prod_2d.copy() if sigma == 0 else gaussian_filter(J_prod_2d, sigma=sigma)
                     for sigma in sigmas_all]
    cmp_vmin_inj, cmp_vmax_inj = robust_vrange_many(data_inj_all, bound_mask, qlo=1.0, qhi=99.0)
    cmp_vmin_prod, cmp_vmax_prod = robust_vrange_many(data_prod_all, bound_mask, qlo=1.0, qhi=99.0)

    fig_cmp, axes_cmp = plt.subplots(2, 4, figsize=(18, 9))
    fig_cmp.suptitle(
        'Progressive Gaussian Smoothing for Prescreen Visualization\n'
        'Supplementary comparison supporting the selection of \u03c3 = 6 grids',
        fontsize=13, fontweight='bold', y=0.985
    )

    im_i = None
    im_p = None
    for col, (sigma, slabel, data_inj, data_prod) in enumerate(zip(sigmas_all, sigma_labels, data_inj_all, data_prod_all)):
        ax_i = axes_cmp[0, col]
        plot_i = np.where(bound_mask, data_inj, np.nan)
        im_i = ax_i.pcolormesh(
            cx_2d.T, cy_2d.T, plot_i.T, cmap='hot_r', shading='auto',
            vmin=cmp_vmin_inj, vmax=cmp_vmax_inj
        )
        plot_domain_outline(ax_i)
        style_map_axes(ax_i)
        ax_i.set_title(slabel, fontsize=10, fontweight='bold', pad=4)
        ax_i.set_xlabel('X (m)', fontsize=8)
        ax_i.set_ylabel('Injectivity Score\nY (m)' if col == 0 else '', fontsize=9)
        for (wi, wj) in inj_ws0:
            ax_i.plot(cx_2d[wi,wj], cy_2d[wi,wj], '^', color='#3566d6', markersize=5.8,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=6)
        for (wi, wj) in prod_ws0:
            ax_i.plot(cx_2d[wi,wj], cy_2d[wi,wj], 'o', color='#4dbb63', markersize=5.2,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=6)

        ax_p = axes_cmp[1, col]
        plot_p = np.where(bound_mask, data_prod, np.nan)
        im_p = ax_p.pcolormesh(
            cx_2d.T, cy_2d.T, plot_p.T, cmap='Blues', shading='auto',
            vmin=cmp_vmin_prod, vmax=cmp_vmax_prod
        )
        plot_domain_outline(ax_p)
        style_map_axes(ax_p)
        ax_p.set_xlabel('X (m)', fontsize=8)
        ax_p.set_ylabel('Productivity Score\nY (m)' if col == 0 else '', fontsize=9)
        for (wi, wj) in inj_ws0:
            ax_p.plot(cx_2d[wi,wj], cy_2d[wi,wj], '^', color='#3566d6', markersize=5.8,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=6)
        for (wi, wj) in prod_ws0:
            ax_p.plot(cx_2d[wi,wj], cy_2d[wi,wj], 'o', color='#4dbb63', markersize=5.2,
                      markeredgecolor='white', markeredgewidth=0.8, zorder=6)

        if sigma == SMOOTH_SIGMA:
            for ax in (ax_i, ax_p):
                ax.text(
                    0.98, 0.98, 'Selected', transform=ax.transAxes,
                    ha='right', va='top', fontsize=8.5, fontweight='bold', color='#5c2510',
                    bbox=dict(boxstyle='round,pad=0.18', fc='#f8e4cf', ec='none', alpha=0.9)
                )

    cbar_i = fig_cmp.colorbar(im_i, ax=axes_cmp[0, :], shrink=0.92, pad=0.02, aspect=32)
    cbar_i.set_label('Normalized Injectivity Score', fontsize=10)
    cbar_i.outline.set_linewidth(0.8)
    cbar_p = fig_cmp.colorbar(im_p, ax=axes_cmp[1, :], shrink=0.92, pad=0.02, aspect=32)
    cbar_p.set_label('Normalized Productivity Score', fontsize=10)
    cbar_p.outline.set_linewidth(0.8)
    fig_cmp.legend(handles=legend_elems, loc='lower center', ncol=2, frameon=False,
                   bbox_to_anchor=(0.5, 0.02), fontsize=10)
    plt.tight_layout(rect=[0, 0.06, 1, 0.955])
    path_cmp = os.path.join(WELLSET_DIR, 'comparison_raw_vs_smoothed.png')
    plt.savefig(path_cmp, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Comparison figure saved: {path_cmp}")


    smooth_label_en = (f'Gaussian Smoothing  \u03c3={SMOOTH_SIGMA} grids '
                       f'\u2248 {SMOOTH_SIGMA*dxy_avg:.0f} m') if SMOOTH_SIGMA > 0 else 'No Smoothing'

    vmin_inj, vmax_inj = robust_vrange(J_inj_2d_viz, bound_mask, qlo=3.0, qhi=99.0)
    vmin_prod, vmax_prod = robust_vrange(J_prod_2d_viz, bound_mask, qlo=3.0, qhi=99.0)
    fig_final, (ax_inj, ax_prod) = plt.subplots(1, 2, figsize=(17.5, 7.8))

    J_inj_plot = np.where(bound_mask, J_inj_2d_viz, np.nan)
    im1 = ax_inj.pcolormesh(
        cx_2d.T, cy_2d.T, J_inj_plot.T, cmap='hot_r',
        shading='auto', vmin=vmin_inj, vmax=vmax_inj
    )
    plot_domain_outline(ax_inj)
    plot_soft_contours(ax_inj, J_inj_2d_viz, levels=8, color='#2b2018', alpha=0.18, linewidth=0.5)
    cb1 = plt.colorbar(im1, ax=ax_inj, shrink=0.88, pad=0.02)
    cb1.set_label('Normalized CO$_2$ Injection Potential', fontsize=11)
    cb1.outline.set_linewidth(0.8)
    ax_inj.set_title('CO$_2$ Injection Potential', fontsize=12)
    ax_inj.set_xlabel('X (m)', fontsize=11)
    ax_inj.set_ylabel('Y (m)', fontsize=11)
    style_map_axes(ax_inj)
    ax_inj.text(
        0.015, 0.985, smooth_label_en, transform=ax_inj.transAxes,
        ha='left', va='top', fontsize=9.2, color='#3f352d',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.82)
    )

    J_prod_plot = np.where(bound_mask, J_prod_2d_viz, np.nan)
    im2 = ax_prod.pcolormesh(
        cx_2d.T, cy_2d.T, J_prod_plot.T, cmap='Blues',
        shading='auto', vmin=vmin_prod, vmax=vmax_prod
    )
    plot_domain_outline(ax_prod)
    plot_soft_contours(ax_prod, J_prod_2d_viz, levels=8, color='#17313f', alpha=0.16, linewidth=0.5)
    cb2 = plt.colorbar(im2, ax=ax_prod, shrink=0.88, pad=0.02)
    cb2.set_label('Normalized Brine Production Potential', fontsize=11)
    cb2.outline.set_linewidth(0.8)
    ax_prod.set_title('Brine Production Potential', fontsize=12)
    ax_prod.set_xlabel('X (m)', fontsize=11)
    ax_prod.set_ylabel('Y (m)', fontsize=11)
    style_map_axes(ax_prod)
    ax_prod.text(
        0.015, 0.985, smooth_label_en, transform=ax_prod.transAxes,
        ha='left', va='top', fontsize=9.2, color='#3f352d',
        bbox=dict(boxstyle='round,pad=0.22', fc='white', ec='none', alpha=0.82)
    )

    fig_final.suptitle(
        'CO$_2$ Saline Aquifer Storage  |  Injection Potential Analysis\n'
        f'Shenhua-Ordos Basin    CMG Grid: {NI}×{NJ}×{NK}    '
        f'INJ: k={INJ_K_START}\u2013{INJ_K_END} (shiqianfeng)    '
        f'PRD: k={PRD_K_START}\u2013{PRD_K_END} (shihezi3)',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig_final.suptitle(
        'CO$_2$ Saline Aquifer Storage Prescreen\n'
        f'Shenhua-Ordos Basin  |  CMG Grid {NI}x{NJ}x{NK}  |  '
        f'INJ k={INJ_K_START}-{INJ_K_END}  |  PRD k={PRD_K_START}-{PRD_K_END}',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.935])
    path_final = os.path.join(WELLSET_DIR, 'final_injection_potential.png')
    plt.savefig(path_final, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Final figure saved: {path_final}")


    fig_diag, (ax_safe, ax_cap) = plt.subplots(1, 2, figsize=(18, 8))
    dP_safe_2d_viz = gaussian_filter(dP_safe_2d, sigma=2.0)
    inj_capacity_2d_viz = gaussian_filter(inj_capacity_2d, sigma=1.2)
    dp_plot = np.where(bound_mask, dP_safe_2d_viz, np.nan)
    cap_plot = np.where(bound_mask, inj_capacity_2d_viz, np.nan)
    vmin_dp, vmax_dp = robust_vrange(dP_safe_2d_viz, bound_mask, qlo=2.5, qhi=99.0)
    vmin_cap, vmax_cap = robust_vrange(inj_capacity_2d_viz, bound_mask, qlo=2.5, qhi=99.0)

    im_dp = ax_safe.pcolormesh(cx_2d.T, cy_2d.T, dp_plot.T, cmap='YlOrRd', shading='auto',
                               vmin=vmin_dp, vmax=vmax_dp)
    plot_domain_outline(ax_safe)
    plot_soft_contours(ax_safe, dP_safe_2d_viz, levels=7, color='#3b2919', alpha=0.18, linewidth=0.5)
    cb_dp = plt.colorbar(im_dp, ax=ax_safe, shrink=0.88, pad=0.02)
    cb_dp.set_label('Positive Safety Margin  max(0, 0.9Pf - Pinj) [MPa]', fontsize=11)
    cb_dp.outline.set_linewidth(0.8)
    ax_safe.set_title('Safety Margin', fontsize=12)
    ax_safe.set_xlabel('X (m)', fontsize=11)
    ax_safe.set_ylabel('Y (m)', fontsize=11)
    style_map_axes(ax_safe)
    mark_wells(ax_safe, inj_ws0, prod_ws0, cx_2d, cy_2d, label_size=7)

    im_cap = ax_cap.pcolormesh(cx_2d.T, cy_2d.T, cap_plot.T, cmap='YlGnBu', shading='auto',
                               vmin=vmin_cap, vmax=vmax_cap)
    plot_domain_outline(ax_cap)
    plot_soft_contours(ax_cap, inj_capacity_2d_viz, levels=7, color='#16333b', alpha=0.16, linewidth=0.5)
    cb_cap = plt.colorbar(im_cap, ax=ax_cap, shrink=0.88, pad=0.02)
    cb_cap.set_label(r'Layer-Integrated Capacity Proxy  $\sum (k_{eff} h \phi / \mu)$', fontsize=11)
    cb_cap.outline.set_linewidth(0.8)
    ax_cap.set_title('Capacity Proxy', fontsize=12)
    ax_cap.set_xlabel('X (m)', fontsize=11)
    ax_cap.set_ylabel('Y (m)', fontsize=11)
    style_map_axes(ax_cap)
    mark_wells(ax_cap, inj_ws0, prod_ws0, cx_2d, cy_2d, label_size=7)

    fig_diag.suptitle(
        'Injection Prescreen Decomposition\n'
        'Well-level safety versus layer-integrated reservoir capacity',
        fontsize=13, fontweight='bold'
    )
    plt.tight_layout(rect=[0, 0, 1, 0.935])
    path_diag = os.path.join(WELLSET_DIR, 'injection_prescreen_diagnostics.png')
    plt.savefig(path_diag, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"    Diagnostic figure saved: {path_diag}")


    print("\n[Step 12] Exporting seed data for optimizer...")
    seeds_path = os.path.join(OUTPUT_DIR, "opt_seeds.npz")


    n_sets  = len(all_well_sets)
    n_inj   = len(all_well_sets[0]['inj_wells'])
    n_prod  = len(all_well_sets[0]['prod_wells'])
    inj_arr  = np.array([[ws['inj_wells'][k]  for k in range(n_inj)]  for ws in all_well_sets], dtype=np.int32)
    prod_arr = np.array([[ws['prod_wells'][k] for k in range(n_prod)] for ws in all_well_sets], dtype=np.int32)


    np.savez_compressed(
        seeds_path,
        inj_wells   = inj_arr,
        prod_wells  = prod_arr,
        J_inj_2d    = J_inj_2d_viz,
        J_prod_2d   = J_prod_2d_viz,
        J_inj_2d_raw = J_inj_2d,
        J_prod_2d_raw = J_prod_2d,
        J_inj_2d_smooth = J_inj_2d_viz,
        J_prod_2d_smooth = J_prod_2d_viz,
        bound_mask  = bound_mask,

        grid_top_2d = top_3d[:, :, INJ_K_END - 1],
        grid_bot_2d = top_3d[:, :, 0] + thick_3d[:, :, 0],
        NI = np.array(NI), NJ = np.array(NJ), NK = np.array(NK),
        INJ_K_START = np.array(INJ_K_START),
        INJ_K_END   = np.array(INJ_K_END),
        PRD_K_START = np.array(PRD_K_START),
        PRD_K_END   = np.array(PRD_K_END),
        dx = np.array(dxy_avg), dy = np.array(dxy_avg),
    )
    print(f"    Seed file saved: {seeds_path}")
    print(f"    Shape: {n_sets} sets × {n_inj} inj + {n_prod} prod wells")

    print("\n[Done] All outputs complete.")
    return all_well_sets, J_inj_2d, J_prod_2d


if __name__ == '__main__':
    main()
