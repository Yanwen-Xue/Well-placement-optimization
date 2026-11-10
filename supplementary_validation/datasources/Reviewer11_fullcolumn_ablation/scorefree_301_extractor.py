from __future__ import annotations
import numpy as np

class ScoreFree301Extractor:
    def __init__(self, base, keep_full, names):
        self.base = base
        self.keep_full = np.asarray(keep_full, dtype=int)
        self.feature_names = list(names)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        f = np.asarray(self.base(X), dtype=float)
        return f[:, self.keep_full]
