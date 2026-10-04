"""Score-free geological fields kept separate for the Reviewer 11 comparison."""

import numpy as np
from scipy.ndimage import uniform_filter


class PrimitiveFieldExtractor:
    """Existing phi/k/h/P statistics plus separate top-depth samples."""

    def __init__(self, base, top):
        self.base = base
        top = np.asarray(top, dtype=np.float32)
        if top.shape[:2] != (base.ni, base.nj):
            raise ValueError("top-depth grid does not match geological fields")
        self.depth = {}
        for group, layer in (("inj", 79), ("prod", 14)):
            self.depth[group, "top"] = top[:, :, layer]
            self.depth[group, "interval"] = top[:, :, 40:80].mean(2) if group == "inj" else top[:, :, :15].mean(2)
            for field in ("top", "interval"):
                self.depth[group, field + "_local_i"] = uniform_filter(self.depth[group, field], size=(7, 1), mode="nearest")
                self.depth[group, field + "_local_j"] = uniform_filter(self.depth[group, field], size=(1, 7), mode="nearest")
        self.feature_names = base.feature_names.copy()
        for group in ("inj", "prod"):
            for field in ("top", "interval", "top_local_i", "top_local_j"):
                self.feature_names.extend(f"{group}_depth_{field}_well_{w + 1}" for w in range(5))

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        extra = np.empty((len(X), 40), dtype=np.float32)
        for row, x in enumerate(X):
            values = []
            for group, coords in (("inj", x[:10]), ("prod", x[10:20])):
                wells = np.rint(coords).astype(int).reshape(5, 2)
                for field in ("top", "interval", "top_local_i", "top_local_j"):
                    values.extend(self.depth[group, field][wells[:, 0], wells[:, 1]].tolist())
            extra[row] = values
        return np.hstack((self.base(X), extra))
