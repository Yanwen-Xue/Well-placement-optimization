from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np


class ComponentF3Surrogate:
    """Surrogate that predicts F3 from pressure-heterogeneity components."""

    def __init__(self, feature_extractor, models: dict, log_components: bool = False):
        self.feature_extractor = feature_extractor
        self.models = dict(models)
        self.log_components = bool(log_components)

    def _features(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if self.feature_extractor is None:
            return X
        extra = np.atleast_2d(np.asarray(self.feature_extractor(X), dtype=float))
        if extra.shape[0] != X.shape[0]:
            raise ValueError(
                f"feature_extractor returned {extra.shape[0]} rows for {X.shape[0]} inputs"
            )
        return np.hstack([X, extra])

    def predict(self, X):
        Xf = self._features(X)
        f1 = np.asarray(self.models["F1"].predict(Xf), dtype=float)
        f2 = np.asarray(self.models["F2"].predict(Xf), dtype=float)
        std_pred = np.asarray(self.models["pressure_std_mpa"].predict(Xf), dtype=float)
        mean_pred = np.asarray(self.models["pressure_mean_mpa"].predict(Xf), dtype=float)
        if self.log_components:
            std = np.exp(std_pred)
            mean = np.exp(mean_pred)
        else:
            std = np.maximum(std_pred, 0.0)
            mean = np.maximum(mean_pred, 1e-9)
        f3 = std / mean
        return np.column_stack([f1, f2, f3])

    def save(self, path):
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path):
        return joblib.load(Path(path))
