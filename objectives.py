import math
import numpy as np


POISSON_NU = 0.29
RHO_OVERBURDEN = 2500.0
GRAVITY = 9.81
OVERBURDEN_GRAD_KPA_PER_M = RHO_OVERBURDEN * GRAVITY / 1000.0
EATON_COEFF = 3.0 * POISSON_NU / (2.0 * (1.0 - POISSON_NU))
N_INJ = 5

BHP_SAFETY_COEFF = 0.90

REFPRES_KPA = 22929.0
REFDEPTH_TVDSS_M = 987.241211
RHO_BRINE = 1015.0
BRINE_GRAD_KPA_PER_M = RHO_BRINE * GRAVITY / 1000.0

PRESSURE_CV_EPS = 1.0e-9
PENALTY_VALUE = 1.0e9

FAIL_NONE = "ok"
FAIL_BOUNDARY = "boundary"
FAIL_SPACING = "spacing"
FAIL_SIMULATION = "simulation"
FAIL_INJECTION = "injection"
FAIL_BHP = "bhp"
FAIL_OBJECTIVE = "objective"


def pore_pressure_at_tvdss_kPa(tvdss_m):
    """Estimate initial pore pressure at a TVDSS depth using the model reference pressure."""
    return REFPRES_KPA + BRINE_GRAD_KPA_PER_M * (tvdss_m - REFDEPTH_TVDSS_M)


def eaton_fracture_kPa(tvd_from_surface_m, pore_pressure_kPa):
    """Estimate fracture pressure using the Eaton fracture-gradient relation."""
    sigma_v = OVERBURDEN_GRAD_KPA_PER_M * tvd_from_surface_m
    return pore_pressure_kPa + (sigma_v - pore_pressure_kPa) * EATON_COEFF


P_FRACTURE_KPA = eaton_fracture_kPa(1200.0, 22000.0)


def check_spacing(
    inj_wells,
    prod_wells,
    d_inj_inj_m,
    d_prod_prod_m,
    d_inj_prod_m,
    cell_size_m,
):
    """Check minimum injection-injection, production-production, and injection-production spacing."""

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
    """Check whether all wells fall inside the feasible boundary mask."""
    for wi, wj in list(inj_wells) + list(prod_wells):
        if wi < 0 or wi >= bound_mask.shape[0]:
            return False
        if wj < 0 or wj >= bound_mask.shape[1]:
            return False
        if not bound_mask[wi, wj]:
            return False
    return True


def compute_F1(pavg_series):
    """Peak relative increase in average reservoir pressure."""
    if pavg_series is None or len(pavg_series) == 0:
        return PENALTY_VALUE
    pavg_initial = float(pavg_series[0])
    if pavg_initial <= 0:
        return PENALTY_VALUE
    return float(np.max(pavg_series) / pavg_initial - 1.0)


def compute_F2(plume_area_sg0p1_km2):
    """Horizontal projected CO2 plume footprint area in km2."""
    value = float(plume_area_sg0p1_km2) if plume_area_sg0p1_km2 is not None else float("nan")
    if not np.isfinite(value) or value < 0.0:
        return PENALTY_VALUE
    return value


def compute_F3(pressure_std_mpa, pressure_mean_mpa, eps=PRESSURE_CV_EPS):
    """Normalized pressure-buildup heterogeneity: PV-weighted std(DeltaP) / mean(DeltaP)."""
    std_value = float(pressure_std_mpa) if pressure_std_mpa is not None else float("nan")
    mean_value = float(pressure_mean_mpa) if pressure_mean_mpa is not None else float("nan")
    if (
        not np.isfinite(std_value)
        or not np.isfinite(mean_value)
        or std_value < 0.0
        or mean_value <= 0.0
    ):
        return PENALTY_VALUE
    return float(std_value / (mean_value + eps))


def check_bhp_constraint(inj_bhp_max_per_well, well_depths_m, surf_elev_m=1200.0):
    """Check the per-injection-well BHP limit against 90% of local Eaton fracture pressure."""
    if inj_bhp_max_per_well is None or len(inj_bhp_max_per_well) < N_INJ:
        return False, {"reason": "missing per-well BHP", "violators": []}
    if len(well_depths_m) < N_INJ:
        return False, {"reason": "well_depths_m too short", "violators": []}

    per_well = []
    violators = []
    for k in range(N_INJ):
        bhp = float(inj_bhp_max_per_well[k])
        depth = float(well_depths_m[k])
        if math.isnan(bhp) or bhp <= 0 or depth <= 0:
            per_well.append(
                {
                    "k": k,
                    "bhp": bhp,
                    "pf": float("nan"),
                    "pp": float("nan"),
                    "ratio": float("nan"),
                    "feasible": False,
                }
            )
            violators.append(k)
            continue

        tvdss = depth - surf_elev_m
        pp = pore_pressure_at_tvdss_kPa(tvdss)
        pf = eaton_fracture_kPa(depth, pp)
        ratio = bhp / pf
        ok = bhp <= BHP_SAFETY_COEFF * pf
        per_well.append({"k": k, "bhp": bhp, "pp": pp, "pf": pf, "ratio": ratio, "feasible": ok})
        if not ok:
            violators.append(k)

    return len(violators) == 0, {"per_well": per_well, "violators": violators}


def _objective_values(eval_result):
    f1 = compute_F1(eval_result.get("pavg"))
    f2 = compute_F2(eval_result.get("plume_area_sg0p1_km2"))
    f3 = compute_F3(
        eval_result.get("pressure_heterogeneity_injend_std_mpa"),
        eval_result.get("pressure_heterogeneity_injend_mean_mpa"),
    )
    return np.array([f1, f2, f3], dtype=float)


def assess_eval_result(
    eval_result,
    bound_mask,
    cell_size_m,
    d_inj_inj_m=200.0,
    d_prod_prod_m=200.0,
    d_inj_prod_m=300.0,
):
    """Assess geometric, simulation, BHP, and objective feasibility."""
    inj_wells = eval_result.get("inj_wells", [])
    prod_wells = eval_result.get("prod_wells", [])

    if not check_boundary(inj_wells, prod_wells, bound_mask):
        return {"feasible": False, "failure_reason": FAIL_BOUNDARY, "bhp_diag": {"violators": []}}

    if not check_spacing(inj_wells, prod_wells, d_inj_inj_m, d_prod_prod_m, d_inj_prod_m, cell_size_m):
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

    vals = _objective_values(eval_result)
    if (not np.isfinite(vals).all()) or (np.abs(vals).max() >= PENALTY_VALUE):
        return {"feasible": False, "failure_reason": FAIL_OBJECTIVE, "bhp_diag": bhp_diag}

    return {"feasible": True, "failure_reason": FAIL_NONE, "bhp_diag": bhp_diag}


def compute_objectives(
    eval_result,
    bound_mask,
    cell_size_m,
    d_inj_inj_m=200.0,
    d_prod_prod_m=200.0,
    d_inj_prod_m=300.0,
):
    """Return F1, F2, F3, and feasibility for a simulation result."""
    assess = assess_eval_result(
        eval_result,
        bound_mask,
        cell_size_m,
        d_inj_inj_m=d_inj_inj_m,
        d_prod_prod_m=d_prod_prod_m,
        d_inj_prod_m=d_inj_prod_m,
    )
    if not assess["feasible"]:
        return PENALTY_VALUE, PENALTY_VALUE, PENALTY_VALUE, False

    f1, f2, f3 = _objective_values(eval_result)
    return float(f1), float(f2), float(f3), True


def normalise_objectives(obj_matrix):
    """Min-max normalize objective columns."""
    obj_n = obj_matrix.copy().astype(float)
    for col in range(obj_matrix.shape[1]):
        lo = obj_matrix[:, col].min()
        hi = obj_matrix[:, col].max()
        if hi > lo:
            obj_n[:, col] = (obj_matrix[:, col] - lo) / (hi - lo)
    return obj_n
