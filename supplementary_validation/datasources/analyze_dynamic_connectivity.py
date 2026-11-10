from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

ROOT = Path(r'C:\Users\admin\Desktop\paper_figure')
OUT = ROOT / 'writen/datasources/dynamic_connectivity_679_run'
dyn = pd.read_csv(OUT / 'dynamic_connectivity_679.csv')
static = pd.read_csv(r'F:\CODE1\_organized_active_work_20260817\06_verified_sandbody_analysis_0_683\petrophysical_connectivity_audit\layout_petrophysical_connectivity_metrics_0_683.csv')
resp = pd.read_csv(r'F:\CODE1\_organized_active_work_20260817\07_80year_three_target_analysis_0_683\three_targets_30y_80y_679.csv')
df = dyn.merge(static, on='sample_index').merge(resp[['sample_index','F1_peak_80y']], on='sample_index')
df = df[df.status.eq('ok')].copy()
local = 'prod_column_kh_mD_m_mean'
df['local_productivity'] = df[local]
df['target_F1'] = df['F1_peak_80y']

def sc(x, y):
    z = pd.concat([x, y], axis=1).dropna()
    r, p = spearmanr(z.iloc[:,0], z.iloc[:,1])
    return float(r), float(p)

rows = []
for col, label in [('local_productivity','producer integrated kh'), ('total_brine_production','cumulative brine production')]:
    r,p = sc(df[col], df.target_F1)
    rows.append({'metric':label, 'rho_spearman':r, 'p_value':p})
for th in ('0p1','0p5','1p0'):
    for col, label in [(f'connected_volume_{th}mpa_m3','connected volume'), (f'interference_distance_{th}mpa_p50_m','interference distance P50'), (f'interference_distance_{th}mpa_p90_m','interference distance P90')]:
        r,p = sc(df[col], df.target_F1)
        rows.append({'metric':f'{label} ({th} MPa)', 'rho_spearman':r, 'p_value':p})

geom = [c for c in ['inj_prod_contact_path_hops','inj_prod_contact_path_bottleneck_area_m2'] if c in df.columns]
base = ['local_productivity'] + geom
dc = ['connected_volume_0p5mpa_m3','interference_distance_0p5mpa_p90_m']
X = df[base + dc].replace([np.inf,-np.inf], np.nan)
y = df.target_F1.astype(float)
cv = KFold(n_splits=5, shuffle=True, random_state=2026)
def cv_r2(cols):
    vals=[]
    for tr,te in cv.split(X):
        m=make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), Ridge(alpha=1.0))
        m.fit(X.iloc[tr][cols], y.iloc[tr])
        vals.append(r2_score(y.iloc[te], m.predict(X.iloc[te][cols])))
    return float(np.mean(vals)), float(np.std(vals,ddof=1))

model_rows=[]
a,asd=cv_r2(base)
model_rows.append({'model':'A local productivity + geometry','cv_r2_mean':a,'cv_r2_sd':asd,'features':';'.join(base)})
b,bsd=cv_r2(geom+dc)
model_rows.append({'model':'B dynamic connectivity + geometry','cv_r2_mean':b,'cv_r2_sd':bsd,'features':';'.join(geom+dc)})
c,csd=cv_r2(base+dc)
model_rows.append({'model':'C local productivity + dynamic connectivity + geometry','cv_r2_mean':c,'cv_r2_sd':csd,'delta_vs_A':c-a,'features':';'.join(base+dc)})

pd.DataFrame(rows).to_csv(OUT/'dynamic_connectivity_F1_correlations.csv',index=False)
pd.DataFrame(model_rows).to_csv(OUT/'dynamic_connectivity_incremental_models.csv',index=False)
df.to_csv(OUT/'dynamic_connectivity_679_merged.csv',index=False)
meta={'n':int(len(df)),'local_column':local,'geometry_columns':geom,'dynamic_columns':dc,'target':'F1_peak_80y','peak_group_year_min':float(df.peak_group_year.min()),'peak_group_year_median':float(df.peak_group_year.median()),'peak_group_year_max':float(df.peak_group_year.max())}
(OUT/'dynamic_connectivity_analysis_meta.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(json.dumps(meta,indent=2))
print('CORRELATIONS')
print(pd.DataFrame(rows).to_string(index=False))
print('MODELS')
print(pd.DataFrame(model_rows).to_string(index=False))
