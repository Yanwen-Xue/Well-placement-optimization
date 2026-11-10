from pathlib import Path
import json, sys, time
import h5py, numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import LinearRegression

ROOT = Path(r"F:/CODE1")
RESULTS = ROOT / r"_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz"
OUT = Path(r"C:/Users/admin/Desktop/paper_figure/writen/datasources/dynamic_connectivity_679.csv")
THRESHOLDS = (-0.05, -0.10, -0.20)  # MPa drawdown relative to initial pressure
NI, NJ, NK = 268, 203, 200
K0, K1 = 0, 80  # producer and injection reservoir intervals
TARGET_YEAR = 30.0

def pressure_metrics(path, wells):
    with h5py.File(path, 'r') as f:
        days = np.asarray(f['General/MasterTimeTable']['Offset in days'], dtype=float)
        names = sorted([n for n in f['SpatialProperties'] if n.isdigit()], key=int)
        ids = np.asarray([int(n) for n in names], dtype=int)
        target = int(np.argmin(np.abs(days[ids] / 365.25 - TARGET_YEAR)))
        g = names[target]
        grid = f['SpatialProperties/000000/GRID']
        ip = np.asarray(grid['IPSTCS'], dtype=np.int64) - 1
        pv = np.asarray(grid['BLOCKPVOL'], dtype=float)
        ii = ip % NI
        jj = (ip // NI) % NJ
        kk = ip // (NI * NJ)
        use = (kk >= K0) & (kk < K1) & np.isfinite(pv) & (pv > 0)
        idx = np.flatnonzero(use)
        p0 = np.asarray(f['SpatialProperties/000000/PRES'][idx], dtype=float)
        pt = np.asarray(f[f'SpatialProperties/{g}/PRES'][idx], dtype=float)
        vol = pv[idx]
        i = ii[idx].astype(float); j = jj[idx].astype(float)
        finite = np.isfinite(p0) & np.isfinite(pt) & np.isfinite(vol)
        dp = (pt[finite] - p0[finite]) / 1000.0
        vol = vol[finite]; i = i[finite]; j = j[finite]
        w = np.asarray(wells, dtype=float).reshape(5, 2)
        dist = np.sqrt((i[:, None] - w[None, :, 0]) ** 2 + (j[:, None] - w[None, :, 1]) ** 2).min(1) * 50.0
        out = {'sample_year': float(days[ids[target]] / 365.25), 'n_cells': int(len(dp))}
        for th in THRESHOLDS:
            mask = dp <= th
            key = str(abs(th)).replace('.', 'p')
            out[f'drawdown_volume_{key}mpa_m3'] = float(vol[mask].sum())
            out[f'drawdown_distance_p50_{key}mpa_m'] = float(np.quantile(dist[mask], .50)) if mask.any() else np.nan
            out[f'drawdown_distance_p90_{key}mpa_m'] = float(np.quantile(dist[mask], .90)) if mask.any() else np.nan
            out[f'drawdown_distance_max_{key}mpa_m'] = float(dist[mask].max()) if mask.any() else np.nan
        return out

def partial_rank(y, x, controls):
    def resid(v):
        z = rankdata(np.asarray(v, float))
        return z - LinearRegression().fit(controls, z).predict(controls)
    return float(np.corrcoef(resid(y), resid(x))[0, 1])

def main():
    d = np.load(RESULTS, allow_pickle=True)
    ok = d['success_mask'].astype(bool)
    rows = np.flatnonzero(ok)
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit > 0: rows = rows[:limit]
    X = np.asarray(d['X'][ok], float)
    sample = np.asarray(d['sample_index'][ok], int)
    out = []
    t0 = time.time()
    for n, (row, wells) in enumerate(zip(rows, np.asarray(d['prod_wells'][ok])), 1):
        sr3 = sorted(Path(str(d['run_dir'][row])).glob('*.sr3'))[0]
        rec = {'sample_index': int(sample[n-1])}
        try:
            rec.update(pressure_metrics(sr3, wells))
        except Exception as e:
            rec['error'] = str(e)
        out.append(rec)
        if n % 10 == 0: print(f'{n}/{len(rows)} elapsed={time.time()-t0:.1f}s', flush=True)
    pd.DataFrame(out).sort_values('sample_index').to_csv(OUT, index=False)
    print(OUT, flush=True)

if __name__ == '__main__': main()
