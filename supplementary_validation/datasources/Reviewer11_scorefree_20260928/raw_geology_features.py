"""Score-free well-layout features sampled from the original geological fields."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

from feature_333 import LayoutSetFeatureExtractor


class RawGeologyFeatureExtractor:
    """Sample phi, kh, cell thickness, and initial pressure without J maps."""

    def __init__(self, poro, perm, thickness, pressure, bound_mask,
                 inj_k_range=(41, 80), prod_k_range=(1, 15), bridge_k_range=(16, 40),
                 radius=3):
        fields = {
            "poro": np.asarray(poro, dtype=np.float32),
            "perm": np.asarray(perm, dtype=np.float32),
            "thick": np.asarray(thickness, dtype=np.float32),
            "pressure": np.asarray(pressure, dtype=np.float32),
        }
        shapes = {field.shape for field in fields.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 3:
            raise ValueError("geological fields must share one 3-D grid")
        ni, nj, nk = fields["poro"].shape
        if np.asarray(bound_mask).shape != (ni, nj):
            raise ValueError("boundary mask must match the horizontal grid")
        self.ni, self.nj = ni, nj
        self.banks = {}
        self.feature_names = []
        layer_ranges = {
            "inj": {"target": inj_k_range, "other": prod_k_range, "bridge": bridge_k_range},
            "prod": {"target": prod_k_range, "other": inj_k_range, "bridge": bridge_k_range},
        }
        for group, bands in layer_ranges.items():
            for name, field in fields.items():
                for band, (first, last) in bands.items():
                    layers = slice(max(0, first - 1), min(nk, last))
                    block = field[:, :, layers]
                    if block.shape[2] == 0:
                        raise ValueError(f"empty {band} layer range")
                    column = np.stack((block.mean(2), block.min(2), block.max(2), block.std(2)), axis=2)
                    self.banks[group, name, band] = column
                    self.feature_names.extend(f"{group}_{name}_{band}_{stat}_well_mean"
                                              for stat in ("mean", "min", "max", "std"))
                    if band == "target":
                        local_i = uniform_filter(column[:, :, 0], size=(2 * radius + 1, 1), mode="nearest")
                        local_j = uniform_filter(column[:, :, 0], size=(1, 2 * radius + 1), mode="nearest")
                        self.banks[group, name, "local_i"] = local_i
                        self.banks[group, name, "local_j"] = local_j
                        for stat in ("mean", "std", "local_i", "local_j"):
                            self.feature_names.extend(f"{group}_{name}_{stat}_well_{w + 1}" for w in range(5))
        self.layout = LayoutSetFeatureExtractor(ni=ni, nj=nj, dx=50.0)
        self.feature_names.extend(f"layout_{name}" for name in self.layout.feature_names)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        out = np.empty((len(X), len(self.feature_names)), dtype=np.float32)
        for row_index, x in enumerate(X):
            row = []
            for group, wells in (("inj", x[:10]), ("prod", x[10:20])):
                wells = np.rint(wells).astype(int).reshape(5, 2)
                if np.any(wells[:, 0] < 0) or np.any(wells[:, 0] >= self.ni) or np.any(wells[:, 1] < 0) or np.any(wells[:, 1] >= self.nj):
                    raise ValueError("well coordinate outside geological grid")
                for name in ("poro", "perm", "thick", "pressure"):
                    for band in ("target", "other", "bridge"):
                        sampled = self.banks[group, name, band][wells[:, 0], wells[:, 1]]
                        row.extend(sampled.mean(0).tolist())
                        if band == "target":
                            row.extend(sampled[:, 0].tolist())
                            row.extend(sampled[:, 3].tolist())
                            for direction in ("local_i", "local_j"):
                                row.extend(self.banks[group, name, direction][wells[:, 0], wells[:, 1]].tolist())
            row.extend(self.layout(x).ravel().tolist())
            out[row_index] = row
        return out
