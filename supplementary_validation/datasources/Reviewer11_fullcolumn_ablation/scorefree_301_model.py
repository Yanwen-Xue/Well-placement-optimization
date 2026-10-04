from __future__ import annotations
import json, sys
from pathlib import Path
import joblib, numpy as np
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
CODE = Path(r'F:/CODE1')
sys.path.insert(0, str(CODE))
from component_f3_surrogate import ComponentF3Surrogate
from scorefree_301_extractor import ScoreFree301Extractor

RESULTS = CODE / r'_organized_active_work_20260817/training_data_0_683_333dim/results_0_683_current3obj_rebuilt_with_0603_0604_20260817.npz'
OFFICIAL = CODE / r'_organized_active_work_20260817/02_current_333dim_proxy_0_683/component_f3_model_0_683_static_well_triview_log_20260817.joblib'
PARAMS = dict(n_estimators=800,max_depth=2,learning_rate=0.03,subsample=0.9,colsample_bytree=0.9,reg_lambda=5.0,min_child_weight=3.0,n_jobs=16,tree_method='hist',device='cpu',random_state=42)

def main():
    with np.load(RESULTS, allow_pickle=False) as d:
        good=d['success_mask'].astype(bool); X=np.asarray(d['X'][good],dtype=np.float32)
        y=np.asarray(d['objectives'][good],dtype=float)
        std=np.asarray(d['pressure_heterogeneity_injend_std_mpa'][good],dtype=float)
        mean=np.asarray(d['pressure_heterogeneity_injend_mean_mpa'][good],dtype=float)
    base_model=joblib.load(OFFICIAL); base=base_model.feature_extractor
    names=list(base.feature_names)
    score_names=[n for n in names if ('J' in n or 'top20' in n or 'top30' in n)]
    score_names += ['inj_J_5x5_mean_mean','inj_J_5x5_mean_min','inj_J_center_minus_5x5_mean_mean','inj_J_center_minus_5x5_mean_min','inj_J_grad_mean_mean','inj_J_grad_max_max','prod_J_5x5_mean_mean','prod_J_center_minus_5x5_mean_mean','prod_J_grad_mean_mean','prod_J_grad_max_max']
    score_names=list(dict.fromkeys(score_names)); remove=[i for i,n in enumerate(names) if n in score_names]
    keep=[i for i in range(313) if i not in remove]
    out_names=[n for i,n in enumerate(names) if i in keep]
    ex=ScoreFree301Extractor(base,keep,out_names)
    M=np.hstack((X, ex(X)))
    models={}
    for key,target in [('F1',y[:,0]),('F2',y[:,1]),('pressure_std_mpa',np.log(np.maximum(std,1e-12))),('pressure_mean_mpa',np.log(np.maximum(mean,1e-12)))]:
        m=XGBRegressor(**PARAMS); m.fit(M,target); models[key]=m
    model=ComponentF3Surrogate(ex,models,log_components=True)
    path=HERE/'scorefree_301_model.joblib'; joblib.dump(model,path)
    (HERE/'scorefree_301_model_meta.json').write_text(json.dumps({'n_features':301,'removed':score_names,'model':str(path)},indent=2),encoding='utf-8')
    print(path, flush=True)
if __name__=='__main__': main()
