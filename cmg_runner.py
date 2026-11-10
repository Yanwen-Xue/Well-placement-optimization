\
\
\


import os, subprocess, shutil, re, json, time, glob, warnings
from datetime import date
import numpy as np
import h5py

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


CMG_EXE      = os.environ.get("CMG_EXE", r"D:\CMG2024\GEM\2024.20\Win_x64\EXE\gm202420.exe")
TEMPLATE_DAT = os.environ.get("CMG_TEMPLATE_DAT", os.path.join(ROOT_DIR, "cmg", "2026_0421_1day.dat"))
SIP_FILE     = os.environ.get("CMG_SIP_FILE", os.path.join(ROOT_DIR, "cmg", "2026_0421.sip"))
OPT_RUNS_DIR = os.environ.get("OPT_RUNS_DIR", os.path.join(ROOT_DIR, "opt_runs"))
CMG_PARASOL  = int(os.environ.get("CMG_PARASOL", "10"))
USE_TUNED_NUMERICS = True


SIM_YEARS_FINAL  = 80
DTMAX_SIM        = 180.0


INJ_K_BOT  = 41
INJ_K_TOP  = 80
PRD_K_BOT  = 1
PRD_K_TOP  = 15


INJ_MAX_STG = 288392.0
USE_DYNAMIC_INJ_RATES = False
PRD_MIN_BHP = 8000.0
PRD_MAX_STW = 1200.0
CO2_BG_SURFACE = 0.003
CO2_DARCY_CONV = 0.0853
CO2_RE_RW_RATIO = 500.0
CO2_LN_RE_RW = np.log(CO2_RE_RW_RATIO)
CO2_T_SURFACE_C = 15.0
CO2_GEO_GRADIENT_C_PER_KM = 33.3


SURF_ELEV = 1200.0


_GRID_TOP_FILE = os.path.join(ROOT_DIR, "property_CMG", "2026_0421_1day_property Grid Top 2025-Jan-01.txt")
_top_3d_cache = None
_prop_cache = {}

def _load_top_3d(NI=268, NJ=203, NK=200):
    """加载 CMG 网格各层 TVDSS 顶面深度 [m]，返回 (NI, NJ, NK) array。
    用于精确计算各 (i,j) 处任意 k 层的真实深度（地层 dip 已纳入）。"""
    global _top_3d_cache
    if _top_3d_cache is not None:
        return _top_3d_cache
    vals = []
    with open(_GRID_TOP_FILE, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("**") or s.startswith("RESULTS") or s.startswith("Grid"):
                continue
            for tok in s.split():
                try: vals.append(float(tok))
                except ValueError: pass
    arr = np.array(vals).reshape(NK, NJ, NI).transpose(2, 1, 0)
    _top_3d_cache = arr
    return arr


def _read_cmg_txt(filepath, NI=268, NJ=203, NK=200):
    data = np.zeros((NI, NJ, NK), dtype=np.float64)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    current_k = None
    current_j = None
    i_values = []
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("**") and "K =" in s and "J =" in s:
            if current_k is not None and current_j is not None and i_values:
                arr = np.array(i_values[:NI], dtype=np.float64)
                if len(arr) == NI:
                    data[:, current_j - 1, current_k - 1] = arr
            parts = s.replace(",", " ").split()
            try:
                current_k = int(parts[parts.index("K") + 2])
                current_j = int(parts[parts.index("J") + 2])
                i_values = []
            except Exception:
                current_k = None
                current_j = None
        elif s.startswith("**") or s.startswith("RESULTS") or s.startswith("Grid"):
            continue
        else:
            try:
                i_values.extend(float(v) for v in s.split())
            except Exception:
                continue

    if current_k is not None and current_j is not None and i_values:
        arr = np.array(i_values[:NI], dtype=np.float64)
        if len(arr) == NI:
            data[:, current_j - 1, current_k - 1] = arr
    return data


def _read_property_txt(filepath, NI=268, NJ=203, NK=200):
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        _ = f.readline()
        vals = []
        for line in f:
            parts = line.strip().split()
            if parts:
                vals.append(float(parts[0]))
    arr = np.array(vals, dtype=np.float64)
    total = NI * NJ * NK
    if len(arr) != total:
        arr = arr[:total]
    return arr.reshape((NI, NJ, NK), order="F")


def _load_dynamic_injection_props():
    if _prop_cache:
        return _prop_cache

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "property_CMG")
    prefix = "2026_0421_1day_property "
    suffix = " 2025-Jan-01.txt"

    pressure = _read_cmg_txt(os.path.join(data_dir, f"{prefix}Pressure{suffix}")) * 1e-3
    thickness = _read_cmg_txt(os.path.join(data_dir, f"{prefix}Grid Thickness{suffix}"))
    top = _read_cmg_txt(os.path.join(data_dir, f"{prefix}Grid Top{suffix}"))
    perm = _read_property_txt(os.path.join(data_dir, "property_260421_all_perm.txt"))

    _prop_cache.update({
        "pressure_mpa": pressure,
        "thickness_m": thickness,
        "top_m": top,
        "perm_md": perm,
    })
    return _prop_cache


def _co2_viscosity_cp(pressure_mpa, temp_c):
    try:
        from CoolProp.CoolProp import PropsSI
        mu_pa_s = PropsSI("V", "P", float(pressure_mpa) * 1e6, "T", float(temp_c) + 273.15, "CO2")
        if np.isfinite(mu_pa_s) and mu_pa_s > 0:
            return float(mu_pa_s * 1e3)
    except Exception:
        pass
    return 0.05


def _co2_bg_surface(pressure_mpa, temp_c):
    try:
        from CoolProp.CoolProp import PropsSI
        rho_res = PropsSI("D", "P", float(pressure_mpa) * 1e6, "T", float(temp_c) + 273.15, "CO2")
        if np.isfinite(rho_res) and rho_res > 1e-10:
            rho_std = 1.977
            return float(rho_std / rho_res)
    except Exception:
        pass
    return CO2_BG_SURFACE


def _estimate_inj_capacity_nm3_day(wi, wj):
    props = _load_dynamic_injection_props()
    pressure = props["pressure_mpa"]
    thickness = props["thickness_m"]
    top = props["top_m"]
    perm = props["perm_md"]

    inj_k0 = INJ_K_BOT - 1
    inj_k1 = INJ_K_TOP - 1

    kh = float(np.sum(perm[wi, wj, inj_k0:inj_k1 + 1] * thickness[wi, wj, inj_k0:inj_k1 + 1]))
    if not np.isfinite(kh) or kh <= 1e-10:
        return 0.0

    tvdss_perf_top = float(top[wi, wj, inj_k1])
    import objectives as obj
    pp_mpa = float(obj.pore_pressure_at_tvdss_kPa(tvdss_perf_top) / 1000.0)
    temp_c = CO2_T_SURFACE_C + CO2_GEO_GRADIENT_C_PER_KM * tvdss_perf_top / 1000.0
    mu_cp = _co2_viscosity_cp(pp_mpa, temp_c)
    bg = _co2_bg_surface(pp_mpa, temp_c)
    if not np.isfinite(mu_cp) or mu_cp <= 1e-10:
        return 0.0
    if not np.isfinite(bg) or bg <= 1e-10:
        return 0.0

    bhp_cap_kpa = _per_well_max_bhp(wi, wj, top)
    pp_kpa = pp_mpa * 1000.0
    dp_max_mpa = max((bhp_cap_kpa - pp_kpa) / 1000.0, 0.0)
    if dp_max_mpa <= 0:
        return 0.0

    qmax = (
        dp_max_mpa * 2.0 * np.pi * kh * CO2_DARCY_CONV
        / (bg * mu_cp * CO2_LN_RE_RW)
    )
    if not np.isfinite(qmax):
        return 0.0
    return max(float(qmax), 0.0)


def _allocate_injection_rates(inj_wells, use_dynamic_inj_rates=None):
    if not inj_wells:
        return [], np.array([], dtype=float), 0.0

    q_caps = np.array([_estimate_inj_capacity_nm3_day(wi, wj) for (wi, wj) in inj_wells], dtype=float)
    q_caps = np.clip(q_caps, 0.0, None)
    rates = np.full(len(inj_wells), float(INJ_MAX_STG), dtype=float)
    shortfall = 0.0
    if use_dynamic_inj_rates:
        warnings.warn(
            "Dynamic injection-rate allocation has been retired; using fixed per-well INJ_MAX_STG rates."
        )
    return [float(q) for q in rates], q_caps, shortfall


def _per_well_max_bhp(wi, wj, top_3d):
\
\
\
\
\
\
\
\
\

    import objectives as obj
    tvdss_perf_top = float(top_3d[wi, wj, INJ_K_TOP - 1])
    D_perf_top     = SURF_ELEV + tvdss_perf_top
    pp = obj.pore_pressure_at_tvdss_kPa(tvdss_perf_top)
    pf = obj.eaton_fracture_kPa(D_perf_top, pp)
    return obj.BHP_SAFETY_COEFF * pf * 0.99


def _well_block(inj_wells, prod_wells, top_3d=None, use_dynamic_inj_rates=None):
\
\
\
\
\

    if top_3d is None:
        top_3d = _load_top_3d()
    lines = []
    inj_rates, q_caps, inj_shortfall = _allocate_injection_rates(
        inj_wells, use_dynamic_inj_rates=use_dynamic_inj_rates
    )

    for k, ((wi, wj), inj_rate) in enumerate(zip(inj_wells, inj_rates)):
        name = f"INJ-{k+1}"
        bhp_max_this_well = _per_well_max_bhp(wi, wj, top_3d)
        lines += [
            f"WELL  '{name}'",
            f"INJECTOR '{name}'",
            "INCOMP  SOLVENT  1.0  0.0",
            f"OPERATE  MAX  STG  {inj_rate:.1f}  CONT  REPEAT",
            f"OPERATE  MAX  BHP  {bhp_max_this_well:.1f}  CONT  REPEAT",
            "**          rad  geofac  wfrac  skin",
            "GEOMETRY  K  0.0762  0.37  1.0  0.0",
            f"PERF      GEOA  '{name}'",
            "** UBA                  ff          Status  Connection",
        ]
        for idx, kk in enumerate(range(INJ_K_TOP, INJ_K_BOT - 1, -1)):
            if idx == 0:
                lines.append(
                    f"  {wi+1}  {wj+1}  {kk}  1.0  OPEN  FLOW-FROM  'SURFACE'  REFLAYER"
                )
            else:
                lines.append(
                    f"  {wi+1}  {wj+1}  {kk}  1.0  OPEN  FLOW-FROM  {idx}"
                )
        lines.append("")

    for k, (wi, wj) in enumerate(prod_wells):
        name = f"PRD-{k+1}"
        lines += [
            f"WELL  '{name}'",
            f"PRODUCER '{name}'",
            f"OPERATE  MAX  STW  {PRD_MAX_STW:.1f}  CONT  REPEAT",
            f"OPERATE  MIN  BHP  {PRD_MIN_BHP:.1f}  CONT  REPEAT",
            "**          rad  geofac  wfrac  skin",
            "GEOMETRY  K  0.0762  0.37  1.0  0.0",
            f"PERF      GEOA  '{name}'",
            "** UBA                  ff          Status  Connection",
        ]
        for idx, kk in enumerate(range(PRD_K_TOP, PRD_K_BOT - 1, -1)):
            if idx == 0:
                lines.append(
                    f"  {wi+1}  {wj+1}  {kk}  1.0  OPEN  FLOW-FROM  'SURFACE'  REFLAYER"
                )
            else:
                lines.append(
                    f"  {wi+1}  {wj+1}  {kk}  1.0  OPEN  FLOW-FROM  {idx}"
                )
        lines.append("")

    return "\n".join(lines), inj_rates, q_caps, inj_shortfall

def make_run_dat(run_id, inj_wells, prod_wells, sim_years=None,
                 gen_subdir=None, local_rid=None, use_dynamic_inj_rates=None):
\
\
\
\
\
\

    if sim_years is None:
        sim_years = SIM_YEARS_FINAL

    if gen_subdir is not None:
        rid      = local_rid if local_rid is not None else run_id
        run_dir  = os.path.join(OPT_RUNS_DIR, gen_subdir, f"run_{rid:03d}")
        dat_stem = f"opt_{rid:03d}"
    else:
        run_dir  = os.path.join(OPT_RUNS_DIR, f"run_{run_id:05d}")
        dat_stem = f"opt_{run_id:05d}"
    os.makedirs(run_dir, exist_ok=True)

    with open(TEMPLATE_DAT, encoding="utf-8", errors="replace") as f:
        template = f.read()


    template = re.sub(
        r"FILENAMES\s+SIPDATA-IN\s+'[^']*'",
        f"FILENAMES SIPDATA-IN '{SIP_FILE.replace(chr(92), '/')}'"  ,
        template,
    )


    template = re.sub(r"DTMAX\s+[\d.]+", f"DTMAX {DTMAX_SIM:.1f}", template)


    if USE_TUNED_NUMERICS:
        tuned_lines = "CHECKRB ON\nSDEGREE 2\n"
        if "CHECKRB ON" not in template and "SDEGREE 2" not in template:
            template = re.sub(
                r"(NUMERICAL\s*\n)",
                r"\1" + tuned_lines,
                template,
                count=1,
            )


    template = re.sub(r"\nRESULTS SPEC.*?RESULTS SPEC STOP", "",
                      template, flags=re.DOTALL)


    if "GHGSOL" not in template:
        template = re.sub(
            r"OUTSRF\s+WELL\s+PAVG",
            "OUTSRF WELL PAVG\n            GHGTHY\n            GHGSCRIT\n            GHGSOL",
            template, count=1,
        )


    INJ_YEARS = 30
    end_year  = 2025 + sim_years
    well_text, inj_rates, q_caps, inj_shortfall = _well_block(
        inj_wells,
        prod_wells,
        use_dynamic_inj_rates=use_dynamic_inj_rates,
    )

    if sim_years > INJ_YEARS:

        inj_dates = "\n".join(
            f"DATE {yr} 1 1" for yr in range(2026, 2025 + INJ_YEARS + 1)
        )

        shutin_lines = (
            "\n".join(f"SHUTIN 'INJ-{k+1}'" for k in range(len(inj_wells)))
            + "\n" +
            "\n".join(f"SHUTIN 'PRD-{k+1}'" for k in range(len(prod_wells)))
        )

        monitor_dates = "\n".join(
            f"DATE {yr} 1 1"
            for yr in range(2025 + INJ_YEARS + 5, end_year + 1, 5)
        )
        date_steps = inj_dates + "\n" + shutin_lines + "\n" + monitor_dates
    else:

        date_steps = "\n".join(
            f"DATE {yr} 1 1" for yr in range(2026, end_year + 1)
        )

    new_run = (
        "RUN\n"
        "DATE 2025 1  1.00000\n"
        f"{well_text}\n"
        f"{date_steps}\n"
        "STOP\n"
    )

    template = re.sub(
        r"RUN\s*\n.*?STOP",
        new_run,
        template,
        flags=re.DOTALL,
    )

    dat_path = os.path.join(run_dir, dat_stem + ".dat")
    with open(dat_path, "w", encoding="utf-8") as f:
        f.write(template)

    return run_dir, dat_stem, inj_rates, q_caps, inj_shortfall


def run_sim(run_dir, dat_stem, timeout=172800):
\
\
\
\
\
\
\
\
\

    dat_path = os.path.join(run_dir, dat_stem + ".dat")
    sr3_path = os.path.join(run_dir, dat_stem + ".sr3")

    for stale in glob.glob(os.path.join(run_dir, "*.lck")) + \
                 glob.glob(os.path.join(run_dir, "*.nw.lck")):
        try:
            os.remove(stale)
        except OSError:
            pass

    try:
        result = subprocess.run(
            [CMG_EXE,
             "-f", dat_path,
             "-wd", run_dir,
             "-parasol", str(CMG_PARASOL),
             "-doms",
             "-wait"],
            cwd=run_dir,
            capture_output=True,
            timeout=timeout,
        )
        success = (result.returncode == 0) and os.path.exists(sr3_path)
    except subprocess.TimeoutExpired:
        success = False

    return sr3_path, success


def parse_sr3(sr3_path):
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
    NI, NJ, NK = 268, 203, 200
    result = {
        "time_days": None,
        "pavg": None,
        "ghgthy": None,
        "ghgscrit": None,
        "ghgsol": None,
        "well_names": [],
        "watvolsc": {},
        "gasvolsc": {},
        "bhp": {},


        "aqucommol_co2":   None,
        "gascommol_co2":   None,
        "gascmlisc_co2":   None,


        "mobile_krg_fraction": None,
        "plume_area_sg0p1_km2": None,
        "plume_area_sg0p1_m2": None,
        "plume_n_columns_sg0p1": None,
        "plume_day": None,


        "pressure_heterogeneity_injend_std_mpa": None,
        "pressure_heterogeneity_injend_mean_mpa": None,
    }

    with h5py.File(sr3_path, "r") as f:

        mtt = f["General/MasterTimeTable"][()]
        result["time_days"] = mtt["Offset in days"].astype(float)


        sp      = f["TimeSeries/SPECIAL HISTORY"]
        sp_vars = [v.decode() for v in sp["Variables"][()]]
        sp_data = sp["Data"][()]

        sp_ts   = sp["Timesteps"][()]

        def _sp(name):
            if name in sp_vars:
                idx = sp_vars.index(name)
                return sp_data[:, idx, 0]
            return np.zeros(len(sp_ts))

        result["time_days_sp"] = result["time_days"][sp_ts]
        result["pavg"]         = _sp("PAVG")
        result["ghgthy"]       = _sp("GHGTHY")
        result["ghgscrit"]     = _sp("GHGSCRIT")
        result["ghgsol"]       = _sp("GHGSOL")


        wells_grp = f["TimeSeries/WELLS"]
        well_vars = [v.decode() for v in wells_grp["Variables"][()]]

        if "Data" in wells_grp:
            w_data = wells_grp["Data"][()]
            n_wells_actual = w_data.shape[2]


            well_names = [
                o.decode().strip() if hasattr(o, "decode") else str(o).strip()
                for o in wells_grp["Origins"][()]
            ]
            result["well_names"] = well_names

            wat_idx = well_vars.index("WATVOLSC") if "WATVOLSC" in well_vars else None
            gas_idx = well_vars.index("GASVOLSC") if "GASVOLSC" in well_vars else None
            bhp_idx = well_vars.index("BHP")      if "BHP"      in well_vars else None

            for wi, wname in enumerate(well_names):
                if wi >= n_wells_actual:
                    break
                if wat_idx is not None and wat_idx < w_data.shape[1]:
                    result["watvolsc"][wname] = w_data[:, wat_idx, wi]
                if gas_idx is not None and gas_idx < w_data.shape[1]:
                    result["gasvolsc"][wname] = w_data[:, gas_idx, wi]
                if bhp_idx is not None and bhp_idx < w_data.shape[1]:
                    result["bhp"][wname] = w_data[:, bhp_idx, wi]


        if "TimeSeries/SECTORS" in f:
            sec_grp  = f["TimeSeries/SECTORS"]
            sec_vars = [v.decode() for v in sec_grp["Variables"][()]]
            sec_data = sec_grp["Data"][()]

            def _sec(name):
                if name in sec_vars:
                    return sec_data[:, sec_vars.index(name), 0]
                return None

            result["aqucommol_co2"] = _sec("AQUCOMMOL(1)")
            result["gascommol_co2"] = _sec("GASCOMMOL(1)")
            result["gascmlisc_co2"] = _sec("GASCMLISC(1)")


        if "SpatialProperties" in f and "000000" in f["SpatialProperties"]:
            groups = [name for name in f["SpatialProperties"].keys() if name.isdigit()]
            if groups:
                gname = max(groups, key=lambda s: int(s))
                root = f[f"SpatialProperties/{gname}"]
                groups_sorted = sorted(groups, key=lambda s: int(s))
                mtt_days = result["time_days"]
                target_day = 30.0 * 365.25
                group_ids = np.array([int(g) for g in groups_sorted], dtype=int)
                group_days = mtt_days[group_ids].astype(float)
                g_injend = groups_sorted[int(np.argmin(np.abs(group_days - target_day)))]
                result["plume_day"] = float(mtt_days[int(g_injend)])
                required = {"SG", "KRG", "MASDENG"}
                if required.issubset(set(root.keys())):
                    grid = f["SpatialProperties/000000/GRID"]
                    ipstcs = grid["IPSTCS"][()].astype(np.int64)
                    pv = grid["BLOCKPVOL"][()].astype(np.float64)
                    full0 = ipstcs - 1
                    i = full0 % NI
                    j = (full0 // NI) % NJ
                    k = full0 // (NI * NJ)
                    inj_interval = (
                        (k >= INJ_K_BOT - 1)
                        & (k <= INJ_K_TOP - 1)
                        & np.isfinite(pv)
                        & (pv > 0.0)
                    )
                    idx = np.where(inj_interval)[0]
                    if idx.size > 0:
                        sg = root["SG"][()][idx].astype(np.float64)
                        krg = root["KRG"][()][idx].astype(np.float64)
                        rhog = root["MASDENG"][()][idx].astype(np.float64)
                        pvi = pv[idx]
                        finite = (
                            np.isfinite(sg)
                            & np.isfinite(krg)
                            & np.isfinite(rhog)
                            & np.isfinite(pvi)
                            & (pvi > 0.0)
                        )
                        if np.any(finite):
                            sg = np.clip(sg[finite], 0.0, None)
                            krg = np.clip(krg[finite], 0.0, None)
                            rhog = np.clip(rhog[finite], 0.0, None)
                            pvi = pvi[finite]
                            gas_mass = pvi * sg * rhog
                            denom = float(np.sum(gas_mass))
                            if denom > 0.0:
                                result["mobile_krg_fraction"] = float(np.sum(gas_mass * krg) / denom)

                if "SG" in f[f"SpatialProperties/{g_injend}"]:
                    grid = f["SpatialProperties/000000/GRID"]
                    ipstcs = grid["IPSTCS"][()].astype(np.int64)
                    pv = grid["BLOCKPVOL"][()].astype(np.float64)
                    full0 = ipstcs - 1
                    i = full0 % NI
                    j = (full0 // NI) % NJ
                    k = full0 // (NI * NJ)
                    inj_interval = (
                        (k >= INJ_K_BOT - 1)
                        & (k <= INJ_K_TOP - 1)
                        & np.isfinite(pv)
                        & (pv > 0.0)
                    )
                    idx = np.where(inj_interval)[0]
                    if idx.size > 0:
                        sg = f[f"SpatialProperties/{g_injend}/SG"][()][idx].astype(np.float64)
                        finite_plume = np.isfinite(sg) & (sg >= 0.1)
                        if np.any(finite_plume):
                            cols = np.unique(np.stack([i[idx][finite_plume], j[idx][finite_plume]], axis=1), axis=0)
                            ncols = int(cols.shape[0])
                        else:
                            ncols = 0
                        area_m2 = float(ncols * 50.0 * 50.0)
                        result["plume_n_columns_sg0p1"] = ncols
                        result["plume_area_sg0p1_m2"] = area_m2
                        result["plume_area_sg0p1_km2"] = area_m2 / 1.0e6

                if "PRES" in f["SpatialProperties/000000"] and "PRES" in f[f"SpatialProperties/{g_injend}"]:
                    grid = f["SpatialProperties/000000/GRID"]
                    ipstcs = grid["IPSTCS"][()].astype(np.int64)
                    pv = grid["BLOCKPVOL"][()].astype(np.float64)
                    full0 = ipstcs - 1
                    k = full0 // (NI * NJ)
                    inj_interval = (
                        (k >= INJ_K_BOT - 1)
                        & (k <= INJ_K_TOP - 1)
                        & np.isfinite(pv)
                        & (pv > 0.0)
                    )
                    idx = np.where(inj_interval)[0]
                    if idx.size > 0:
                        p0 = f["SpatialProperties/000000/PRES"][()][idx].astype(np.float64)
                        pt = f[f"SpatialProperties/{g_injend}/PRES"][()][idx].astype(np.float64)
                        pvi = pv[idx]
                        dp_mpa = (pt - p0) / 1000.0
                        finite = np.isfinite(dp_mpa) & np.isfinite(pvi) & (pvi > 0.0)
                        if np.any(finite):
                            pvi = pvi[finite]
                            dp_mpa = dp_mpa[finite]
                            wsum = float(np.sum(pvi))
                            if wsum > 0.0:
                                mean_dp = float(np.sum(pvi * dp_mpa) / wsum)
                                var_dp = float(np.sum(pvi * (dp_mpa - mean_dp) ** 2) / wsum)
                                result["pressure_heterogeneity_injend_mean_mpa"] = mean_dp
                                result["pressure_heterogeneity_injend_std_mpa"] = float(np.sqrt(max(var_dp, 0.0)))

    return result


def expected_sim_days(sim_years):

    return float((date(2025 + int(sim_years), 1, 1) - date(2025, 1, 1)).days)


def reached_full_sim_duration(time_days, sim_years, tolerance_days=1.0):
\
\
\
\
\

    if time_days is None:
        return False
    arr = np.asarray(time_days, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return False
    return float(arr.max()) >= expected_sim_days(sim_years) - float(tolerance_days)


def evaluate(run_id, inj_wells, prod_wells, grid_top_2d,
             sim_years=None, cleanup=False, gen_subdir=None, local_rid=None,
             grid_bot_2d=None, use_dynamic_inj_rates=None):
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

    if sim_years is None:
        sim_years = SIM_YEARS_FINAL

    t0 = time.time()
    run_dir, dat_stem, inj_rates, inj_caps, inj_shortfall = make_run_dat(
        run_id,
        inj_wells,
        prod_wells,
        sim_years,
        gen_subdir,
        local_rid,
        use_dynamic_inj_rates=use_dynamic_inj_rates,
    )
    sr3_path, success = run_sim(run_dir, dat_stem)
    elapsed = time.time() - t0

    out = {
        "run_id":    run_id,
        "run_success": success,
        "success":   success,
        "run_dir":   run_dir,
        "elapsed_s": elapsed,
    }

    out["sim_years"] = sim_years
    out["inj_rates_surface_nm3_day"] = np.array(inj_rates, dtype=float)
    out["inj_total_surface_nm3_day"] = float(np.sum(inj_rates))
    out["inj_capacity_surface_nm3_day"] = np.array(inj_caps, dtype=float)
    out["inj_target_surface_nm3_day"] = float(INJ_MAX_STG * len(inj_wells))
    out["inj_shortfall_surface_nm3_day"] = float(inj_shortfall)
    out["dynamic_injection_feasible"] = bool(inj_shortfall <= 1e-6)
    out["full_duration_reached"] = False
    out["time_days_final"] = float("nan")
    out["failure_reason"] = ""

    if not success:
        out["failure_reason"] = "simulation_failed"
        out.update({"pavg": None, "ghgthy": None,
                    "watvolsc_total": np.nan, "gasvolsc_total": np.nan,
                    "prod_watvolsc_final_per_well": np.full(len(prod_wells), np.nan),
                    "max_inj_bhp": np.nan, "well_depths_m": np.nan,
                    "well_drill_depths_m": np.nan,
                    "ghgsol_final": 0.0, "ghgthy_final": 0.0,
                    "ghgscrit_final": 0.0,
                    "aqu_initial_mol": 0.0, "aqu_final_mol": 0.0,
                    "gas_final_mol": 0.0, "inj_cum_mol": 0.0,
                    "dissolved_net_mol": 0.0,
                    "mobile_krg_fraction": np.nan,
                    "pressure_heterogeneity_injend_std_mpa": np.nan,
                    "pressure_heterogeneity_injend_mean_mpa": np.nan,
                    "inj_bhp_max_per_well": np.array([])})
        return out

    sr3 = parse_sr3(sr3_path)
    out["time_days"] = sr3["time_days_sp"]
    out["pavg"]      = sr3["pavg"]
    out["ghgthy"]    = sr3["ghgthy"]
    out["time_days_final"] = float(np.max(sr3["time_days"])) if sr3["time_days"] is not None and len(sr3["time_days"]) > 0 else float("nan")
    out["full_duration_reached"] = reached_full_sim_duration(sr3["time_days"], sim_years)
    if not out["full_duration_reached"]:
        out["success"] = False
        out["failure_reason"] = "simulation_incomplete"


    ghgsol   = sr3.get("ghgsol")
    ghgthy   = sr3.get("ghgthy")
    ghgscrit = sr3.get("ghgscrit")
    out["ghgsol_final"]  = float(ghgsol[-1])   if (ghgsol   is not None and len(ghgsol)   > 0) else 0.0
    out["ghgthy_final"]  = float(ghgthy[-1])   if (ghgthy   is not None and len(ghgthy)   > 0) else 0.0
    out["ghgscrit_final"] = float(ghgscrit[-1]) if (ghgscrit is not None and len(ghgscrit) > 0) else 0.0


    aqu = sr3.get("aqucommol_co2")
    gas = sr3.get("gascommol_co2")
    inj = sr3.get("gascmlisc_co2")
    out["aqu_initial_mol"]   = float(aqu[0])  if (aqu is not None and len(aqu) > 0) else 0.0
    out["aqu_final_mol"]     = float(aqu[-1]) if (aqu is not None and len(aqu) > 0) else 0.0
    out["gas_final_mol"]     = float(gas[-1]) if (gas is not None and len(gas) > 0) else 0.0
    out["inj_cum_mol"]       = float(inj[-1]) if (inj is not None and len(inj) > 0) else 0.0
    out["dissolved_net_mol"] = out["aqu_final_mol"] - out["aqu_initial_mol"]
    out["mobile_krg_fraction"] = (
        float(sr3["mobile_krg_fraction"])
        if sr3.get("mobile_krg_fraction") is not None
        else np.nan
    )
    out["pressure_heterogeneity_injend_std_mpa"] = (
        float(sr3["pressure_heterogeneity_injend_std_mpa"])
        if sr3.get("pressure_heterogeneity_injend_std_mpa") is not None
        else np.nan
    )
    out["pressure_heterogeneity_injend_mean_mpa"] = (
        float(sr3["pressure_heterogeneity_injend_mean_mpa"])
        if sr3.get("pressure_heterogeneity_injend_mean_mpa") is not None
        else np.nan
    )


    prod_names = sorted(n for n in sr3["watvolsc"] if n.startswith("PRD"))
    if prod_names:
        out["prod_watvolsc_final_per_well"] = np.array(
            [float(sr3["watvolsc"][n][-1]) for n in prod_names],
            dtype=float,
        )
    else:
        out["prod_watvolsc_final_per_well"] = np.array([], dtype=float)
    out["watvolsc_total"] = float(np.sum(out["prod_watvolsc_final_per_well"]))
    out["gasvolsc_total"] = sum(
        v[-1] for v in sr3["gasvolsc"].values()
    ) if sr3["gasvolsc"] else 0.0


    inj_names = sorted(n for n in sr3["bhp"] if n.startswith("INJ"))
    if inj_names:
        out["inj_bhp_max_per_well"] = np.array(
            [float(sr3["bhp"][n].max()) for n in inj_names]
        )
        out["max_inj_bhp"] = float(out["inj_bhp_max_per_well"].max())
    else:
        out["inj_bhp_max_per_well"] = np.array([])
        out["max_inj_bhp"] = float(
            max(v.max() for v in sr3["bhp"].values())
        ) if sr3["bhp"] else np.nan

    all_wells = list(inj_wells) + list(prod_wells)


    top_3d = _load_top_3d()
    inj_depths = [max(1.0, SURF_ELEV + float(top_3d[wi, wj, INJ_K_TOP - 1])) for (wi, wj) in inj_wells]
    prd_depths = [max(1.0, SURF_ELEV + float(top_3d[wi, wj, PRD_K_TOP - 1])) for (wi, wj) in prod_wells]
    out["well_depths_m"] = np.array(inj_depths + prd_depths)


    if grid_bot_2d is not None:
        drill_depths = [max(1.0, SURF_ELEV + grid_bot_2d[wi, wj]) for (wi, wj) in all_wells]
    else:
        drill_depths = out["well_depths_m"]
    out["well_drill_depths_m"] = np.array(drill_depths)

    if cleanup:
        try:
            shutil.rmtree(run_dir)
        except Exception:
            pass

    return out


if __name__ == "__main__":
    print("cmg_runner.py self-test:")
    print(f"  CMG exe   : {CMG_EXE}")
    print(f"  Template  : {TEMPLATE_DAT}")
    print(f"  SIP file  : {SIP_FILE}")
    print(f"  Runs dir  : {OPT_RUNS_DIR}")
    print(f"  CMG found : {os.path.exists(CMG_EXE)}")
    print(f"  Template  : {os.path.exists(TEMPLATE_DAT)}")
    print(f"  SIP found : {os.path.exists(SIP_FILE)}")
    print()
    print("Testing dat generation (no simulation)...")
    inj  = [(38, 148), (205, 177), (127, 60), (200, 63), (245, 129)]
    prod = [(115, 122), (105, 75), (28, 78), (70, 133), (170, 50)]
    run_dir, dat_stem, inj_rates = make_run_dat(99999, inj, prod, sim_years=2)
    print(f"  Written: {os.path.join(run_dir, dat_stem+'.dat')}")
    print(f"  Injection rates [Nm3/day]: {[round(x, 1) for x in inj_rates]}")


    with open(os.path.join(run_dir, dat_stem + ".dat"), encoding="utf-8") as f:
        content = f.read()
    print(f"  Has INJ-1: {'INJ-1' in content}")
    print(f"  Has PRD-1: {'PRD-1' in content}")
    print(f"  Has DATE 2027: {'DATE 2027 1 1' in content}")
