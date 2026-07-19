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


import math
import numpy as np


_POISSON_NU    = 0.29

_RHO_OB        = 2500.0
_G             = 9.81
_OB_GRAD_kPa_m = _RHO_OB * _G / 1000.0
_EATON_COEFF   = 3.0 * _POISSON_NU / (2.0 * (1.0 - _POISSON_NU))
_N_INJ         = 5


BHP_SAFETY_COEFF = 0.90


REFPRES_kPa      = 22929.0
REFDEPTH_TVDSS_m = 987.241211
RHO_BRINE        = 1015.0
_BRINE_GRAD_kPa_m = RHO_BRINE * _G / 1000.0


def pore_pressure_at_tvdss_kPa(tvdss_m):
    """从 CMG REFPRES + 盐水静水梯度递推任意 TVDSS 处的初始 Pp [kPa]。
    用于 BHP 约束判定与 BHP 上限计算（两端使用同一公式保证对齐）。"""
    return REFPRES_kPa + _BRINE_GRAD_kPa_m * (tvdss_m - REFDEPTH_TVDSS_m)


def eaton_fracture_kPa(tvd_from_surface_m, pore_pressure_kPa):
\
\
\
\
\
\
\

    sigma_v = _OB_GRAD_kPa_m * tvd_from_surface_m
    return pore_pressure_kPa + (sigma_v - pore_pressure_kPa) * _EATON_COEFF


P_FRACTURE_kPa = eaton_fracture_kPa(1200.0, 22000.0)

PENALTY_VALUE = 1e9

FAIL_NONE = "ok"
FAIL_BOUNDARY = "boundary"
FAIL_SPACING = "spacing"
FAIL_SIMULATION = "simulation"
FAIL_INJECTION = "injection"
FAIL_BHP = "bhp"
FAIL_OBJECTIVE = "objective"


def check_spacing(inj_wells, prod_wells,
                  d_inj_inj_m, d_prod_prod_m, d_inj_prod_m, cell_size_m):

    def dist(a, b):
        return cell_size_m * np.hypot(a[0] - b[0], a[1] - b[1])

    n_inj = len(inj_wells)
    for i in range(n_inj):
        for j in range(i + 1, n_inj):
            if dist(inj_wells[i], inj_wells[j]) < d_inj_inj_m:
                return False

    n_prod = len(prod_wells)
    for i in range(n_prod):
        for j in range(i + 1, n_prod):
            if dist(prod_wells[i], prod_wells[j]) < d_prod_prod_m:
                return False

    for wi in inj_wells:
        for wp in prod_wells:
            if dist(wi, wp) < d_inj_prod_m:
                return False

    return True


def check_boundary(inj_wells, prod_wells, bound_mask):

    for (wi, wj) in list(inj_wells) + list(prod_wells):
        if wi < 0 or wi >= bound_mask.shape[0]:
            return False
        if wj < 0 or wj >= bound_mask.shape[1]:
            return False
        if not bound_mask[wi, wj]:
            return False
    return True


def compute_F1(pavg_series):
\
\
\
\
\
\
\

    if pavg_series is None or len(pavg_series) == 0:
        return PENALTY_VALUE
    pavg_initial = pavg_series[0]
    if pavg_initial <= 0:
        return PENALTY_VALUE
    return float(np.max(pavg_series) / pavg_initial - 1.0)


def compute_F2(mobile_krg_fraction):
\
\
\
\
\
\

    value = float(mobile_krg_fraction) if mobile_krg_fraction is not None else float("nan")
    if not np.isfinite(value) or value < 0.0:
        return PENALTY_VALUE
    return value
def compute_F3(pressure_heterogeneity_injend_std_mpa):
\
\
\
\
\
\
\

    value = (
        float(pressure_heterogeneity_injend_std_mpa)
        if pressure_heterogeneity_injend_std_mpa is not None
        else float("nan")
    )
    if not np.isfinite(value) or value < 0.0:
        return PENALTY_VALUE
    return value


def check_bhp_constraint(inj_bhp_max_per_well, well_depths_m, surf_elev_m=1200.0):
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

    if inj_bhp_max_per_well is None or len(inj_bhp_max_per_well) < _N_INJ:
        return False, {"reason": "missing per-well BHP", "violators": []}
    if len(well_depths_m) < _N_INJ:
        return False, {"reason": "well_depths_m too short", "violators": []}

    per_well = []
    violators = []
    for k in range(_N_INJ):
        bhp = float(inj_bhp_max_per_well[k])
        D   = float(well_depths_m[k])
        if math.isnan(bhp) or bhp <= 0 or D <= 0:
            per_well.append({"k": k, "bhp": bhp, "pf": float("nan"),
                             "pp": float("nan"), "ratio": float("nan"),
                             "feasible": False})
            violators.append(k)
            continue

        tvdss = D - surf_elev_m
        pp    = pore_pressure_at_tvdss_kPa(tvdss)
        pf    = eaton_fracture_kPa(D, pp)
        ratio = bhp / pf
        ok    = bhp <= BHP_SAFETY_COEFF * pf
        per_well.append({"k": k, "bhp": bhp, "pp": pp, "pf": pf,
                         "ratio": ratio, "feasible": ok})
        if not ok:
            violators.append(k)

    feasible = len(violators) == 0
    return feasible, {"per_well": per_well, "violators": violators}


def assess_eval_result(eval_result, bound_mask, cell_size_m,
                       d_inj_inj_m=200.0, d_prod_prod_m=200.0, d_inj_prod_m=300.0):
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

    inj_wells = eval_result.get("inj_wells", [])
    prod_wells = eval_result.get("prod_wells", [])

    if not check_boundary(inj_wells, prod_wells, bound_mask):
        return {"feasible": False, "failure_reason": FAIL_BOUNDARY, "bhp_diag": {"violators": []}}

    if not check_spacing(inj_wells, prod_wells,
                         d_inj_inj_m, d_prod_prod_m, d_inj_prod_m, cell_size_m):
        return {"feasible": False, "failure_reason": FAIL_SPACING, "bhp_diag": {"violators": []}}

    if not eval_result.get("success", False):
        return {"feasible": False, "failure_reason": FAIL_SIMULATION, "bhp_diag": {"violators": []}}

    if not eval_result.get("dynamic_injection_feasible", True):
        return {"feasible": False, "failure_reason": FAIL_INJECTION, "bhp_diag": {"violators": []}}

    well_depths = eval_result.get("well_depths_m", np.array([]))
    inj_bhp_per_well = eval_result.get("inj_bhp_max_per_well", np.array([]))
    feasible_bhp, bhp_diag = check_bhp_constraint(inj_bhp_per_well, well_depths)
    if not feasible_bhp:
        return {"feasible": False, "failure_reason": FAIL_BHP, "bhp_diag": bhp_diag}

    F1 = compute_F1(eval_result.get("pavg"))
    F2 = compute_F2(eval_result.get("mobile_krg_fraction"))
    F3 = compute_F3(eval_result.get("pressure_heterogeneity_injend_std_mpa"))
    vals = np.array([F1, F2, F3], dtype=float)
    if (not np.isfinite(vals).all()) or (np.abs(vals).max() >= PENALTY_VALUE):
        return {"feasible": False, "failure_reason": FAIL_OBJECTIVE, "bhp_diag": bhp_diag}

    return {"feasible": True, "failure_reason": FAIL_NONE, "bhp_diag": bhp_diag}


def compute_objectives(eval_result, bound_mask, cell_size_m,
                       d_inj_inj_m=200.0, d_prod_prod_m=200.0, d_inj_prod_m=300.0):
\
\
\
\
\
\
\

    assess = assess_eval_result(
        eval_result, bound_mask, cell_size_m,
        d_inj_inj_m=d_inj_inj_m, d_prod_prod_m=d_prod_prod_m, d_inj_prod_m=d_inj_prod_m,
    )
    if not assess["feasible"]:
        return PENALTY_VALUE, PENALTY_VALUE, PENALTY_VALUE, False

    F1 = compute_F1(eval_result["pavg"])
    F2 = compute_F2(eval_result.get("mobile_krg_fraction"))
    F3 = compute_F3(eval_result.get("pressure_heterogeneity_injend_std_mpa"))
    return F1, F2, F3, True


def normalise_objectives(obj_matrix):
\
\
\
\

    obj_n = obj_matrix.copy().astype(float)
    for col in range(obj_matrix.shape[1]):
        lo = obj_matrix[:, col].min()
        hi = obj_matrix[:, col].max()
        if hi > lo:
            obj_n[:, col] = (obj_matrix[:, col] - lo) / (hi - lo)
    return obj_n
