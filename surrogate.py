
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from collections import deque

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def _get_xgb_regressor():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for SurrogateModel. Install: pip install xgboost"
        ) from exc
    return XGBRegressor


def _get_xgb_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for SurrogateModel. Install: pip install xgboost"
        ) from exc
    return XGBClassifier


N_INJ = 5
N_PRD = 5
N_DIM = N_INJ * 2 + N_PRD * 2
TARGET_LABELS = ("F1", "F2", "F3")
COMPONENT_TARGET_LABELS = ("F1", "F2", "F3_pressure_std", "F3_pressure_mean")
_PENALTY_THRESH = 1e8
_MIN_CLASS_SAMPLES = 5


def _drop_penalty_rows(X, y):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.atleast_2d(np.asarray(y, dtype=float))
    bad = (np.abs(y).max(axis=1) >= _PENALTY_THRESH) | (~np.isfinite(y).all(axis=1))
    keep = ~bad
    return X[keep], y[keep], keep


def _failure_mask_from_targets(y):
    y = np.atleast_2d(np.asarray(y, dtype=float))
    return (np.abs(y).max(axis=1) >= _PENALTY_THRESH) | (~np.isfinite(y).all(axis=1))


def _summarize(values):
    if not values:
        return dict(mean=float("nan"), std=float("nan"))
    arr = np.asarray(values, dtype=float)
    return dict(mean=float(np.mean(arr)), std=float(np.std(arr)))


def _resolve_target_params(base_params, target_params=None, labels=TARGET_LABELS):
    base = dict(base_params)
    labels = tuple(labels)
    out = [dict(base) for _ in labels]
    if target_params is None:
        return out

    if isinstance(target_params, (list, tuple)):
        if len(target_params) != len(labels):
            raise ValueError(f"xgb_target_params list must have {len(labels)} entries")
        for k, override in enumerate(target_params):
            if override:
                out[k].update(dict(override))
        return out

    if isinstance(target_params, dict):
        for k, label in enumerate(labels):
            override = (
                target_params.get(label)
                or target_params.get(label.lower())
                or target_params.get(str(k))
                or target_params.get(str(k + 1))
            )
            if override:
                out[k].update(dict(override))
        return out

    raise TypeError("xgb_target_params must be None, a target-parameter list, or a dict")


def _align_one(x, ref):
    a = np.asarray(x, dtype=float).copy()

    t = ref[: N_INJ * 2].reshape(N_INJ, 2)
    s = a[: N_INJ * 2].reshape(N_INJ, 2)
    _, col = linear_sum_assignment(((t[:, None] - s[None]) ** 2).sum(axis=2))
    a[: N_INJ * 2] = s[col].flatten()

    off = N_INJ * 2
    t = ref[off:].reshape(N_PRD, 2)
    s = a[off:].reshape(N_PRD, 2)
    _, col = linear_sum_assignment(((t[:, None] - s[None]) ** 2).sum(axis=2))
    a[off:] = s[col].flatten()
    return a


def align_pop(X, ref):
    X = np.atleast_2d(np.asarray(X, dtype=float))
    return np.array([_align_one(x, ref) for x in X])


def _lex_sort_groups(ref):
    out = ref.copy()
    inj = out[: N_INJ * 2].reshape(N_INJ, 2)
    prd = out[N_INJ * 2 :].reshape(N_PRD, 2)
    out[: N_INJ * 2] = inj[np.lexsort((inj[:, 1], inj[:, 0]))].flatten()
    out[N_INJ * 2 :] = prd[np.lexsort((prd[:, 1], prd[:, 0]))].flatten()
    return out


def make_canonical_reference(X, mode="medoid"):
    if mode == "first":
        return np.asarray(X[0], dtype=float).copy()
    if mode == "lex":
        return _lex_sort_groups(np.asarray(X[0], dtype=float))
    if mode == "medoid":
        Xa = align_pop(X, X[0])
        centroid = Xa.mean(axis=0)
        idx = int(np.argmin(((Xa - centroid) ** 2).sum(axis=1)))
        return _lex_sort_groups(Xa[idx])
    raise ValueError(f"unknown ref mode: {mode}")


class SurrogateModel:
    DEFAULT_PARAMS = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=32,
        tree_method="hist",
    )
    DEFAULT_CLF_PARAMS = dict(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=32,
        tree_method="hist",
        eval_metric="logloss",
    )

    def __init__(
        self,
        feature_extractor=None,
        with_uncertainty: bool = False,
        with_failure_classifier: bool = True,
        ref_mode: str = "medoid",
        xgb_params: dict | None = None,
        xgb_target_params: dict | list | None = None,
        xgb_classifier_params: dict | None = None,
        early_stopping_rounds: int | None = None,
        validation_fraction: float = 0.20,
        f3_component_mode: bool = False,
    ):
        self.feature_extractor = feature_extractor
        self.with_uncertainty = with_uncertainty
        self.with_failure_classifier = with_failure_classifier
        self.ref_mode = ref_mode
        self.f3_component_mode = bool(f3_component_mode)
        self.target_labels = COMPONENT_TARGET_LABELS if self.f3_component_mode else TARGET_LABELS
        self.xgb_params = {**self.DEFAULT_PARAMS, **(xgb_params or {})}
        self.xgb_params_by_target = _resolve_target_params(
            self.xgb_params,
            xgb_target_params,
            labels=self.target_labels,
        )
        self.xgb_classifier_params = {
            **self.DEFAULT_CLF_PARAMS,
            **(xgb_classifier_params or {}),
        }
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = float(validation_fraction)

        if self.f3_component_mode and self.with_uncertainty:
            raise ValueError("with_uncertainty is not supported with f3_component_mode=True")

        XGBR = _get_xgb_regressor()
        self.mean_models = [XGBR(**params) for params in self.xgb_params_by_target]

        if with_uncertainty:
            self.lo_models = [
                XGBR(
                    **{
                        **params,
                        "objective": "reg:quantileerror",
                        "quantile_alpha": 0.10,
                    }
                )
                for params in self.xgb_params_by_target
            ]
            self.hi_models = [
                XGBR(
                    **{
                        **params,
                        "objective": "reg:quantileerror",
                        "quantile_alpha": 0.90,
                    }
                )
                for params in self.xgb_params_by_target
            ]
        else:
            self.lo_models = None
            self.hi_models = None

        if with_failure_classifier:
            XGBC = _get_xgb_classifier()
            self.fail_clf = XGBC(**self.xgb_classifier_params)
        else:
            self.fail_clf = None

        self.ref = None
        self.x_mean = None
        self.x_std = None
        self.n_features = None
        self.fitted = False
        self.failure_classifier_fitted = False
        self.failure_reason_ = None
        self.n_fail_ = 0
        self.n_succ_ = 0
        self.best_iterations_ = {}

    def _build_features(self, X_aligned):
        if self.feature_extractor is None:
            return X_aligned
        extra = np.atleast_2d(np.asarray(self.feature_extractor(X_aligned), dtype=float))
        if extra.shape[0] != X_aligned.shape[0]:
            raise ValueError(
                f"feature_extractor returned {extra.shape[0]} rows for {X_aligned.shape[0]} inputs"
            )
        return np.hstack([X_aligned, extra])

    def _transform(self, X):
        Xa = align_pop(X, self.ref)
        Xf = self._build_features(Xa)
        return (Xf - self.x_mean) / self.x_std

    def _fit_regressor(self, model, X_train, y_train, X_val=None, y_val=None):
        if self.early_stopping_rounds and X_val is not None and len(X_val) > 0:
            model.set_params(early_stopping_rounds=int(self.early_stopping_rounds))
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model

    def _fit_classifier(self, model, X_train, y_train, X_val=None, y_val=None):
        if self.early_stopping_rounds and X_val is not None and len(X_val) > 0:
            model.set_params(early_stopping_rounds=int(self.early_stopping_rounds))
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)
        return model

    def _as_objective_targets(self, y):
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if self.f3_component_mode and y.shape[1] == 4:
            f3 = y[:, 2] / np.maximum(y[:, 3], 1.0e-9)
            return np.column_stack([y[:, 0], y[:, 1], f3])
        return y

    def fit(self, X, y, failure_mask=None):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        if X.shape[1] != N_DIM:
            raise ValueError(f"X must have {N_DIM} cols, got {X.shape[1]}")
        expected_y_cols = 4 if self.f3_component_mode else 3
        if y.shape[1] != expected_y_cols:
            raise ValueError(f"y must have {expected_y_cols} cols ({self.target_labels}), got {y.shape[1]}")
        if len(X) != len(y):
            raise ValueError(f"len mismatch: X={len(X)} y={len(y)}")

        inferred_fail = _failure_mask_from_targets(y)
        if failure_mask is None:
            fail_mask = inferred_fail
        else:
            fail_mask = np.asarray(failure_mask, dtype=bool).reshape(-1)
            if len(fail_mask) != len(X):
                raise ValueError(
                    f"failure_mask len mismatch: X={len(X)} failure_mask={len(fail_mask)}"
                )
            fail_mask = fail_mask | inferred_fail

        n_in = len(X)
        X_reg, y_reg, keep = _drop_penalty_rows(X, y)
        n_drop = n_in - len(X_reg)
        if n_drop > 0:
            print(f"[surrogate] dropped {n_drop}/{n_in} infeasible/penalty rows before fit")
        if len(X_reg) < 5:
            raise ValueError(
                f"too few feasible training rows after PENALTY filter: {len(X_reg)}"
            )


        self.ref = make_canonical_reference(X, mode=self.ref_mode)
        Xa_all = align_pop(X, self.ref)
        Xf_all = self._build_features(Xa_all)
        Xf_reg = Xf_all[keep]
        self.x_mean = Xf_reg.mean(axis=0)
        self.x_std = Xf_reg.std(axis=0)
        self.x_std[self.x_std < 1e-6] = 1.0
        Xs_all = (Xf_all - self.x_mean) / self.x_std
        Xs_reg = Xs_all[keep]
        self.n_features = Xs_reg.shape[1]
        self.best_iterations_ = {}

        reg_fit = (Xs_reg, y_reg, None, None)
        use_reg_val = (
            self.early_stopping_rounds
            and 0.0 < self.validation_fraction < 0.5
            and len(Xs_reg) >= 20
        )
        if use_reg_val:
            reg_tr, reg_val = train_test_split(
                np.arange(len(Xs_reg)),
                test_size=self.validation_fraction,
                random_state=int(self.xgb_params.get("random_state", 42)),
                shuffle=True,
            )
            reg_fit = (Xs_reg[reg_tr], y_reg[reg_tr], Xs_reg[reg_val], y_reg[reg_val])
        for k in range(len(self.target_labels)):
            X_tr, y_tr, X_val, y_val = reg_fit
            self._fit_regressor(
                self.mean_models[k],
                X_tr,
                y_tr[:, k],
                X_val,
                None if y_val is None else y_val[:, k],
            )
            self.best_iterations_[self.target_labels[k]] = getattr(
                self.mean_models[k],
                "best_iteration",
                None,
            )
        if self.with_uncertainty:
            for k in range(len(self.target_labels)):
                self.lo_models[k].fit(Xs_reg, y_reg[:, k])
                self.hi_models[k].fit(Xs_reg, y_reg[:, k])

        self.n_fail_ = int(fail_mask.sum())
        self.n_succ_ = int(len(fail_mask) - self.n_fail_)
        self.failure_classifier_fitted = False
        self.failure_reason_ = None
        if self.with_failure_classifier and self.fail_clf is not None:
            if self.n_fail_ < _MIN_CLASS_SAMPLES or self.n_succ_ < _MIN_CLASS_SAMPLES:
                self.failure_reason_ = (
                    f"classifier skipped: success={self.n_succ_} fail={self.n_fail_}"
                )
                print(f"[surrogate] {self.failure_reason_}")
            else:
                clf_fit = (Xs_all, fail_mask.astype(int), None, None)
                use_clf_val = (
                    self.early_stopping_rounds
                    and 0.0 < self.validation_fraction < 0.5
                    and self.n_fail_ >= 2
                    and self.n_succ_ >= 2
                    and len(Xs_all) >= 20
                )
                if use_clf_val:
                    clf_tr, clf_val = train_test_split(
                        np.arange(len(Xs_all)),
                        test_size=self.validation_fraction,
                        random_state=int(self.xgb_classifier_params.get("random_state", 42)),
                        shuffle=True,
                        stratify=fail_mask.astype(int),
                    )
                    clf_fit = (
                        Xs_all[clf_tr],
                        fail_mask.astype(int)[clf_tr],
                        Xs_all[clf_val],
                        fail_mask.astype(int)[clf_val],
                    )
                X_tr, y_tr, X_val, y_val = clf_fit
                self._fit_classifier(self.fail_clf, X_tr, y_tr, X_val, y_val)
                self.best_iterations_["failure_classifier"] = getattr(
                    self.fail_clf,
                    "best_iteration",
                    None,
                )
                self.failure_classifier_fitted = True
                print(
                    f"[surrogate] failure classifier trained: "
                    f"success={self.n_succ_} fail={self.n_fail_}"
                )

        self.fitted = True
        return self

    def predict(self, X, return_std: bool = False):
        if not self.fitted:
            raise RuntimeError("call fit() first")
        Xs = self._transform(X)
        raw = np.column_stack([self.mean_models[k].predict(Xs) for k in range(len(self.target_labels))])
        if self.f3_component_mode:
            f3 = raw[:, 2] / np.maximum(raw[:, 3], 1.0e-9)
            mean = np.column_stack([raw[:, 0], raw[:, 1], f3])
        else:
            mean = raw
        if not return_std:
            return mean
        if self.f3_component_mode:
            raise RuntimeError("return_std is not supported with f3_component_mode=True")
        if not self.with_uncertainty:
            raise RuntimeError("surrogate not built with with_uncertainty=True")
        lo = np.column_stack([self.lo_models[k].predict(Xs) for k in range(3)])
        hi = np.column_stack([self.hi_models[k].predict(Xs) for k in range(3)])
        std = np.maximum(0.0, (hi - lo) / 2.563)
        return mean, std

    def predict_failure(self, X):
        if not self.fitted:
            raise RuntimeError("call fit() first")
        n = np.atleast_2d(np.asarray(X, dtype=float)).shape[0]
        if not self.failure_classifier_fitted or self.fail_clf is None:
            return np.zeros(n, dtype=float)
        Xs = self._transform(X)
        proba = self.fail_clf.predict_proba(Xs)
        if proba.ndim != 2 or proba.shape[1] < 2:
            return np.zeros(n, dtype=float)
        return np.clip(proba[:, 1].astype(float), 0.0, 1.0)

    def score(self, X, y, per_objective: bool = True):
        pred = self.predict(X)
        y = self._as_objective_targets(y)
        ss_res = ((pred - y) ** 2).sum(axis=0)
        ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
        ss_tot = np.where(ss_tot < 1e-12, 1.0, ss_tot)
        r2 = 1.0 - ss_res / ss_tot
        return r2 if per_objective else float(r2.mean())

    def metrics(self, X, y) -> dict:
        pred = self.predict(X)
        y = self._as_objective_targets(y)
        err = pred - y
        rmse = np.sqrt((err ** 2).mean(axis=0))
        y_safe = np.where(np.abs(y) < 1e-12, 1.0, y)
        mape = np.abs(err / y_safe).mean(axis=0)
        r2 = self.score(X, y, per_objective=True)
        labels = ["F1", "F2", "F3"]
        return {
            lab: dict(rmse=float(rmse[k]), mape=float(mape[k]), r2=float(r2[k]))
            for k, lab in enumerate(labels)
        }

    def cross_validate(self, X, y, failure_mask=None, n_splits: int = 5, seed: int = 42) -> dict:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.atleast_2d(np.asarray(y, dtype=float))
        inferred_fail = _failure_mask_from_targets(y)
        if failure_mask is None:
            fail_mask = inferred_fail
        else:
            fail_mask = np.asarray(failure_mask, dtype=bool).reshape(-1)
            if len(fail_mask) != len(X):
                raise ValueError(
                    f"failure_mask len mismatch: X={len(X)} failure_mask={len(fail_mask)}"
                )
            fail_mask = fail_mask | inferred_fail

        n_in = len(X)
        X_reg, _, _ = _drop_penalty_rows(X, y)
        n_drop = n_in - len(X_reg)
        if n_drop > 0:
            print(f"[surrogate.cv] dropped {n_drop}/{n_in} infeasible/penalty rows")
        if len(X_reg) < n_splits:
            raise ValueError(
                f"after PENALTY filter only {len(X_reg)} feasible rows remain, "
                f"cannot do {n_splits}-fold CV"
            )

        if self.with_failure_classifier and fail_mask.sum() >= n_splits * 2 and (~fail_mask).sum() >= n_splits * 2:
            split_iter = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X, fail_mask.astype(int))
        else:
            split_iter = KFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            ).split(X)

        agg = {lab: {k: [] for k in ("rmse", "mape", "r2")} for lab in ("F1", "F2", "F3")}
        clf_metrics = {k: [] for k in ("auc", "accuracy")}
        n_folds_with_clf = 0

        for tr, te in split_iter:
            tmp = SurrogateModel(
                feature_extractor=self.feature_extractor,
                with_uncertainty=False,
                with_failure_classifier=self.with_failure_classifier,
                ref_mode=self.ref_mode,
                xgb_params=self.xgb_params,
                xgb_target_params=self.xgb_params_by_target,
                xgb_classifier_params=self.xgb_classifier_params,
                early_stopping_rounds=self.early_stopping_rounds,
                validation_fraction=self.validation_fraction,
                f3_component_mode=self.f3_component_mode,
            )
            tmp.fit(X[tr], y[tr], failure_mask=fail_mask[tr])

            keep_te = ~_failure_mask_from_targets(y[te])
            if keep_te.sum() > 0:
                m = tmp.metrics(X[te][keep_te], y[te][keep_te])
                for lab in agg:
                    for key in agg[lab]:
                        agg[lab][key].append(m[lab][key])

            if tmp.failure_classifier_fitted and len(np.unique(fail_mask[te].astype(int))) == 2:
                p_fail = tmp.predict_failure(X[te])
                y_fail = fail_mask[te].astype(int)
                clf_metrics["auc"].append(float(roc_auc_score(y_fail, p_fail)))
                clf_metrics["accuracy"].append(
                    float(accuracy_score(y_fail, (p_fail >= 0.5).astype(int)))
                )
                n_folds_with_clf += 1

        out = {
            lab: {
                key: _summarize(vals)
                for key, vals in agg[lab].items()
            }
            for lab in agg
        }
        out["classifier"] = {
            "auc": _summarize(clf_metrics["auc"]),
            "accuracy": _summarize(clf_metrics["accuracy"]),
            "n_folds_with_clf": int(n_folds_with_clf),
        }
        return out

    def save(self, path):
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path):
        return joblib.load(Path(path))


def make_physics_extractor(KH_inj_2d, KH_prd_2d, Pf_2d, Pp_2d=None):
    KH_inj = np.asarray(KH_inj_2d, dtype=float)
    KH_prd = np.asarray(KH_prd_2d, dtype=float)
    Pf = np.asarray(Pf_2d, dtype=float)
    Pp = np.zeros_like(Pf) if Pp_2d is None else np.asarray(Pp_2d, dtype=float)

    def extractor(X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = X.shape[0]
        feats = np.empty((n, N_INJ * 3 + N_PRD * 2))
        for r in range(n):
            x = X[r]
            cnt = 0
            for k in range(N_INJ):
                i = int(np.clip(x[2 * k], 0, KH_inj.shape[0] - 1))
                j = int(np.clip(x[2 * k + 1], 0, KH_inj.shape[1] - 1))
                feats[r, cnt : cnt + 3] = [KH_inj[i, j], Pf[i, j], Pf[i, j] - Pp[i, j]]
                cnt += 3
            off = N_INJ * 2
            for k in range(N_PRD):
                i = int(np.clip(x[off + 2 * k], 0, KH_prd.shape[0] - 1))
                j = int(np.clip(x[off + 2 * k + 1], 0, KH_prd.shape[1] - 1))
                feats[r, cnt : cnt + 2] = [KH_prd[i, j], Pf[i, j]]
                cnt += 2
        return feats

    return extractor


def _field_values_at_wells(field, wells):
    vals = []
    for wi, wj in np.asarray(wells, dtype=float):
        i = int(np.clip(round(wi), 0, field.shape[0] - 1))
        j = int(np.clip(round(wj), 0, field.shape[1] - 1))
        vals.append(float(field[i, j]))
    return np.asarray(vals, dtype=float)


def _normalise_field(field, valid_mask):
    field = np.asarray(field, dtype=float)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(field)
    out = np.zeros_like(field, dtype=float)
    if not np.any(valid):
        return out
    vals = field[valid]
    lo = float(np.nanmin(vals))
    hi = float(np.nanmax(vals))
    if hi - lo < 1e-12:
        return out
    out[valid] = (field[valid] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


class PrescreenFeatureExtractor:


    def __init__(
        self,
        J_inj_2d,
        J_prod_2d,
        bound_mask=None,
        top_fracs=(0.20, 0.30),
        grid_top_2d=None,
        grid_bot_2d=None,
        include_geometry=False,
        include_local=False,
    ):
        j_inj = np.asarray(J_inj_2d, dtype=float)
        j_prod = np.asarray(J_prod_2d, dtype=float)
        if j_inj.shape != j_prod.shape:
            raise ValueError(f"J_inj/J_prod shape mismatch: {j_inj.shape} vs {j_prod.shape}")
        if bound_mask is None:
            valid_mask = np.isfinite(j_inj) & np.isfinite(j_prod)
        else:
            valid_mask = np.asarray(bound_mask, dtype=bool)
            if valid_mask.shape != j_inj.shape:
                raise ValueError(f"bound_mask shape mismatch: {valid_mask.shape} vs {j_inj.shape}")
            valid_mask = valid_mask & np.isfinite(j_inj) & np.isfinite(j_prod)

        self.j_inj_n = _normalise_field(j_inj, valid_mask & (j_inj > 0))
        self.j_prod_n = _normalise_field(j_prod, valid_mask & (j_prod > 0))
        self.valid_mask = valid_mask
        self.include_geometry = bool(include_geometry)
        self.include_local = bool(include_local)
        self.ni, self.nj = j_inj.shape
        self.boundary_dist = self._compute_boundary_distance(valid_mask)
        self.thickness_n = None
        if grid_top_2d is not None and grid_bot_2d is not None:
            top = np.asarray(grid_top_2d, dtype=float)
            bot = np.asarray(grid_bot_2d, dtype=float)
            if top.shape != j_inj.shape or bot.shape != j_inj.shape:
                raise ValueError("grid_top_2d/grid_bot_2d must match J map shape")
            thickness = np.abs(bot - top)
            self.thickness_n = _normalise_field(thickness, valid_mask & np.isfinite(thickness))
        self.top_fracs = tuple(float(v) for v in top_fracs)
        self.inj_thresholds = []
        self.prod_thresholds = []
        inj_valid = valid_mask & np.isfinite(self.j_inj_n)
        prod_valid = valid_mask & np.isfinite(self.j_prod_n)
        for frac in self.top_fracs:
            if not (0.0 < frac < 1.0):
                raise ValueError(f"top fraction must be in (0, 1), got {frac}")
            self.inj_thresholds.append(float(np.quantile(self.j_inj_n[inj_valid], 1.0 - frac)))
            self.prod_thresholds.append(float(np.quantile(self.j_prod_n[prod_valid], 1.0 - frac)))

        self.feature_names = (
            [f"inj_J_{k + 1}" for k in range(N_INJ)]
            + [f"prod_J_{k + 1}" for k in range(N_PRD)]
            + ["inj_J_mean", "inj_J_min", "inj_J_max", "inj_J_std"]
            + ["prod_J_mean", "prod_J_min", "prod_J_max", "prod_J_std"]
            + [f"inj_top{int(frac * 100)}_frac" for frac in self.top_fracs]
            + [f"prod_top{int(frac * 100)}_frac" for frac in self.top_fracs]
        )
        if self.include_geometry:
            self.feature_names += (
                "inj_boundary_dist_min",
                "inj_boundary_dist_mean",
                "prod_boundary_dist_min",
                "all_boundary_dist_min",
                "n_inj_boundary_le5",
                "n_all_boundary_le5",
                "min_inj_inj_dist",
                "mean_inj_inj_dist",
                "min_prod_prod_dist",
                "mean_prod_prod_dist",
                "min_inj_prod_dist",
                "mean_inj_prod_dist",
            )
        if self.include_local:
            self.feature_names += (
                "inj_J_5x5_mean_mean",
                "inj_J_5x5_mean_min",
                "inj_J_center_minus_5x5_mean_mean",
                "inj_J_center_minus_5x5_mean_min",
                "inj_J_grad_mean_mean",
                "inj_J_grad_max_max",
                "prod_J_5x5_mean_mean",
                "prod_J_center_minus_5x5_mean_mean",
                "prod_J_grad_mean_mean",
                "prod_J_grad_max_max",
            )
            if self.thickness_n is not None:
                self.feature_names += (
                    "inj_thickness_mean",
                    "inj_thickness_min",
                    "inj_thickness_std",
                    "inj_thickness_center_minus_5x5_mean_mean",
                )

    @staticmethod
    def _compute_boundary_distance(valid_mask):
        valid_mask = np.asarray(valid_mask, dtype=bool)
        ni, nj = valid_mask.shape
        dist = np.full((ni, nj), np.inf, dtype=float)
        q = deque()
        for i in range(ni):
            for j in range(nj):
                if not valid_mask[i, j]:
                    dist[i, j] = 0.0
                    q.append((i, j))
        for i in range(ni):
            for j in (0, nj - 1):
                if valid_mask[i, j] and dist[i, j] > 1.0:
                    dist[i, j] = 1.0
                    q.append((i, j))
        for j in range(nj):
            for i in (0, ni - 1):
                if valid_mask[i, j] and dist[i, j] > 1.0:
                    dist[i, j] = 1.0
                    q.append((i, j))
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni2, nj2 = i + di, j + dj
                if (
                    0 <= ni2 < ni
                    and 0 <= nj2 < nj
                    and valid_mask[ni2, nj2]
                    and dist[ni2, nj2] > dist[i, j] + 1.0
                ):
                    dist[ni2, nj2] = dist[i, j] + 1.0
                    q.append((ni2, nj2))
        max_dist = np.nanmax(dist[np.isfinite(dist)]) if np.any(np.isfinite(dist)) else 1.0
        if max_dist < 1.0:
            max_dist = 1.0
        return np.where(np.isfinite(dist), dist / max_dist, 0.0)

    @staticmethod
    def _summary(vals):
        vals = np.asarray(vals, dtype=float)
        return [
            float(np.mean(vals)),
            float(np.min(vals)),
            float(np.max(vals)),
            float(np.std(vals)),
        ]

    def _well_ij(self, well):
        i = int(np.clip(round(float(well[0])), 0, self.ni - 1))
        j = int(np.clip(round(float(well[1])), 0, self.nj - 1))
        return i, j

    def _field_window(self, field, i, j, radius=2):
        i0 = max(0, i - radius)
        i1 = min(self.ni, i + radius + 1)
        j0 = max(0, j - radius)
        j1 = min(self.nj, j + radius + 1)
        sub = field[i0:i1, j0:j1]
        mask = self.valid_mask[i0:i1, j0:j1] & np.isfinite(sub)
        vals = sub[mask]
        if len(vals) == 0:
            return np.asarray([float(field[i, j])], dtype=float)
        return vals

    def _field_gradients(self, field, i, j):
        center = float(field[i, j])
        vals = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < self.ni and 0 <= nj < self.nj and self.valid_mask[ni, nj]:
                vals.append(abs(center - float(field[ni, nj])))
        if not vals:
            vals = [0.0]
        return float(np.mean(vals)), float(np.max(vals))

    @staticmethod
    def _pair_distances(a, b=None):
        a = np.asarray(a, dtype=float)
        if b is None:
            vals = [
                float(np.linalg.norm(a[i] - a[j]))
                for i in range(len(a))
                for j in range(i + 1, len(a))
            ]
        else:
            b = np.asarray(b, dtype=float)
            vals = [
                float(np.linalg.norm(x - y))
                for x in a
                for y in b
            ]
        if not vals:
            vals = [0.0]
        return np.asarray(vals, dtype=float)

    def _boundary_values(self, wells):
        vals = []
        for well in wells:
            i, j = self._well_ij(well)
            vals.append(float(self.boundary_dist[i, j]))
        return np.asarray(vals, dtype=float)

    def _local_map_features(self, field, wells):
        neigh_mean = []
        center_minus = []
        grad_mean = []
        grad_max = []
        for well in wells:
            i, j = self._well_ij(well)
            center = float(field[i, j])
            vals = self._field_window(field, i, j, radius=2)
            mean = float(np.mean(vals))
            gm, gx = self._field_gradients(field, i, j)
            neigh_mean.append(mean)
            center_minus.append(center - mean)
            grad_mean.append(gm)
            grad_max.append(gx)
        return (
            np.asarray(neigh_mean, dtype=float),
            np.asarray(center_minus, dtype=float),
            np.asarray(grad_mean, dtype=float),
            np.asarray(grad_max, dtype=float),
        )

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feats = np.empty((X.shape[0], len(self.feature_names)), dtype=float)
        for r, x in enumerate(X):
            inj_wells = x[: N_INJ * 2].reshape(N_INJ, 2)
            prod_wells = x[N_INJ * 2 :].reshape(N_PRD, 2)
            inj_vals = _field_values_at_wells(self.j_inj_n, inj_wells)
            prod_vals = _field_values_at_wells(self.j_prod_n, prod_wells)

            row = []
            row.extend(inj_vals.tolist())
            row.extend(prod_vals.tolist())
            row.extend(self._summary(inj_vals))
            row.extend(self._summary(prod_vals))
            for thr in self.inj_thresholds:
                row.append(float(np.mean(inj_vals >= thr)))
            for thr in self.prod_thresholds:
                row.append(float(np.mean(prod_vals >= thr)))
            if self.include_geometry:
                inj_bd = self._boundary_values(inj_wells)
                prod_bd = self._boundary_values(prod_wells)
                all_bd = np.concatenate([inj_bd, prod_bd])
                inj_inj = self._pair_distances(inj_wells)
                prod_prod = self._pair_distances(prod_wells)
                inj_prod = self._pair_distances(inj_wells, prod_wells)
                row.extend([
                    float(np.min(inj_bd)),
                    float(np.mean(inj_bd)),
                    float(np.min(prod_bd)),
                    float(np.min(all_bd)),
                    float(np.mean(inj_bd <= 0.05)),
                    float(np.mean(all_bd <= 0.05)),
                    float(np.min(inj_inj) / max(self.ni, self.nj)),
                    float(np.mean(inj_inj) / max(self.ni, self.nj)),
                    float(np.min(prod_prod) / max(self.ni, self.nj)),
                    float(np.mean(prod_prod) / max(self.ni, self.nj)),
                    float(np.min(inj_prod) / max(self.ni, self.nj)),
                    float(np.mean(inj_prod) / max(self.ni, self.nj)),
                ])
            if self.include_local:
                inj_mean, inj_delta, inj_gm, inj_gx = self._local_map_features(
                    self.j_inj_n, inj_wells
                )
                prod_mean, prod_delta, prod_gm, prod_gx = self._local_map_features(
                    self.j_prod_n, prod_wells
                )
                row.extend([
                    float(np.mean(inj_mean)),
                    float(np.min(inj_mean)),
                    float(np.mean(inj_delta)),
                    float(np.min(inj_delta)),
                    float(np.mean(inj_gm)),
                    float(np.max(inj_gx)),
                    float(np.mean(prod_mean)),
                    float(np.mean(prod_delta)),
                    float(np.mean(prod_gm)),
                    float(np.max(prod_gx)),
                ])
                if self.thickness_n is not None:
                    thick_vals = _field_values_at_wells(self.thickness_n, inj_wells)
                    thick_mean, thick_delta, _, _ = self._local_map_features(
                        self.thickness_n, inj_wells
                    )
                    row.extend([
                        float(np.mean(thick_vals)),
                        float(np.min(thick_vals)),
                        float(np.std(thick_vals)),
                        float(np.mean(thick_delta)),
                    ])
            feats[r] = row
        return feats


def make_prescreen_extractor(
    J_inj_2d,
    J_prod_2d,
    bound_mask=None,
    top_fracs=(0.20, 0.30),
    grid_top_2d=None,
    grid_bot_2d=None,
    include_geometry=False,
    include_local=False,
):

    return PrescreenFeatureExtractor(
        J_inj_2d,
        J_prod_2d,
        bound_mask,
        top_fracs,
        grid_top_2d=grid_top_2d,
        grid_bot_2d=grid_bot_2d,
        include_geometry=include_geometry,
        include_local=include_local,
    )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 80
    X = rng.integers(20, 240, size=(n, N_DIM)).astype(float)
    inj_xy = X[:, : N_INJ * 2].reshape(n, N_INJ, 2)
    prd_xy = X[:, N_INJ * 2 :].reshape(n, N_PRD, 2)
    d_inj = np.linalg.norm(inj_xy[:, :, None] - inj_xy[:, None], axis=-1).mean(axis=(1, 2))
    d_prd = np.linalg.norm(prd_xy[:, :, None] - prd_xy[:, None], axis=-1).mean(axis=(1, 2))
    d_cross = np.linalg.norm(inj_xy[:, :, None] - prd_xy[:, None], axis=-1).mean(axis=(1, 2))
    y = np.column_stack(
        [
            0.05 + 0.001 * d_inj + 0.01 * rng.standard_normal(n),
            -0.10 + 0.0005 * d_cross + 0.005 * rng.standard_normal(n),
            1e7 * (1.0 + 0.01 * d_prd) + 1e5 * rng.standard_normal(n),
        ]
    )
    y[:12] = 1e9

    sm = SurrogateModel(with_uncertainty=True, with_failure_classifier=True)
    sm.fit(X[:60], y[:60])
    keep = ~_failure_mask_from_targets(y[60:])
    metrics = sm.metrics(X[60:][keep], y[60:][keep])
    cv = sm.cross_validate(X, y, n_splits=5)

    print("Held-out metrics:")
    for lab, m in metrics.items():
        print(f"  {lab}: rmse={m['rmse']:.4g}  mape={m['mape']:.3f}  r2={m['r2']:.3f}")
    print("\nClassifier sample:")
    print(sm.predict_failure(X[60:65]))
    print("\n5-fold CV summary:")
    for lab, m in cv.items():
        print(lab, m)
