from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

N_INJ = N_PRD = 5


class CombinedFeatureExtractor:
    def __init__(self, extractors):
        self.extractors = [ex for ex in extractors if ex is not None]
        names = []
        for ex in self.extractors:
            names.extend(list(getattr(ex, "feature_names", [])))
        self.feature_names = names

    def __call__(self, X):
        if not self.extractors:
            return np.empty((np.atleast_2d(X).shape[0], 0), dtype=float)
        parts = [np.atleast_2d(np.asarray(ex(X), dtype=float)) for ex in self.extractors]
        return np.hstack(parts)


class LayoutSetFeatureExtractor:
    def __init__(self, ni=268, nj=203, dx=50.0, inj_rate_per_well=288392.0):
        self.ni = float(ni)
        self.nj = float(nj)
        self.dx = float(dx)
        self.inj_rate_per_well = float(inj_rate_per_well)
        group_stats = ("min", "p10", "mean", "p90", "max", "std")
        names = []
        for group in ("inj", "prod"):
            names.extend(
                [
                    f"{group}_centroid_i",
                    f"{group}_centroid_j",
                    f"{group}_spread_mean",
                    f"{group}_spread_max",
                    f"{group}_bbox_w",
                    f"{group}_bbox_h",
                    f"{group}_bbox_area",
                    f"{group}_bbox_aspect",
                ]
            )
            names.extend([f"{group}_pair_dist_{s}" for s in group_stats])
        names.extend([f"inj_prod_pair_dist_{s}" for s in group_stats])
        names.extend([f"inj_prod_nearest_dist_{s}" for s in group_stats])
        names.extend(
            [
                "inj_prod_centroid_dist",
                "inj_prod_centroid_di",
                "inj_prod_centroid_dj",
                "inj_prod_hungarian_mean_dist",
                "inj_prod_hungarian_max_dist",
                "inj_rate_total_norm",
                "inj_rate_per_well_norm",
            ]
        )
        self.feature_names = names

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
            vals = [float(np.linalg.norm(x - y)) for x in a for y in b]
        return np.asarray(vals if vals else [0.0], dtype=float)

    @staticmethod
    def _dist_stats(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.asarray([0.0], dtype=float)
        return [
            float(np.min(vals)),
            float(np.percentile(vals, 10)),
            float(np.mean(vals)),
            float(np.percentile(vals, 90)),
            float(np.max(vals)),
            float(np.std(vals)),
        ]

    def _group_features(self, wells):
        wells = np.asarray(wells, dtype=float)
        denom = np.asarray([max(self.ni, 1.0), max(self.nj, 1.0)], dtype=float)
        wn = wells / denom
        c = np.mean(wn, axis=0)
        spread = np.linalg.norm(wn - c, axis=1)
        width = float(np.max(wn[:, 0]) - np.min(wn[:, 0]))
        height = float(np.max(wn[:, 1]) - np.min(wn[:, 1]))
        return [
            float(c[0]),
            float(c[1]),
            float(np.mean(spread)),
            float(np.max(spread)),
            width,
            height,
            float(width * height),
            float(width / max(height, 1e-9)),
        ] + self._dist_stats(self._pair_distances(wn))

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feats = np.empty((X.shape[0], len(self.feature_names)), dtype=float)
        denom = np.asarray([max(self.ni, 1.0), max(self.nj, 1.0)], dtype=float)
        for r, x in enumerate(X):
            inj = x[: N_INJ * 2].reshape(N_INJ, 2)
            prod = x[N_INJ * 2 :].reshape(N_PRD, 2)
            inj_n = inj / denom
            prod_n = prod / denom
            row = []
            row.extend(self._group_features(inj))
            row.extend(self._group_features(prod))
            cross = self._pair_distances(inj_n, prod_n)
            row.extend(self._dist_stats(cross))
            dmat = np.linalg.norm(inj_n[:, None, :] - prod_n[None, :, :], axis=2)
            nearest = np.concatenate([np.min(dmat, axis=1), np.min(dmat, axis=0)])
            row.extend(self._dist_stats(nearest))
            c_inj = np.mean(inj_n, axis=0)
            c_prod = np.mean(prod_n, axis=0)
            _, col = linear_sum_assignment(dmat)
            match = dmat[np.arange(len(col)), col]
            total_rate = self.inj_rate_per_well * N_INJ
            row.extend(
                [
                    float(np.linalg.norm(c_inj - c_prod)),
                    float(c_inj[0] - c_prod[0]),
                    float(c_inj[1] - c_prod[1]),
                    float(np.mean(match)),
                    float(np.max(match)),
                    float(total_rate / 1.0e6),
                    float(self.inj_rate_per_well / 1.0e6),
                ]
            )
            feats[r] = row
        return feats


class StaticWellConditionedTriViewFeatureExtractor:
    """Per-well XZ/YZ static fields plus continuous initial-pressure context."""

    def __init__(
        self,
        static_inj_3d,
        static_prod_3d,
        pressure_initial_3d,
        inj_k_range=(41, 80),
        prod_k_range=(1, 15),
        bridge_k_range=(16, 40),
        radius=3,
    ):
        self.static_inj = np.asarray(static_inj_3d, dtype=float)
        self.static_prod = np.asarray(static_prod_3d, dtype=float)
        self.pressure = np.asarray(pressure_initial_3d, dtype=float)
        if (
            self.static_inj.shape != self.static_prod.shape
            or self.static_inj.shape != self.pressure.shape
            or self.static_inj.ndim != 3
        ):
            raise ValueError(
                "static side-view arrays must have the same 3-D shape: "
                f"{self.static_inj.shape}, {self.static_prod.shape}, {self.pressure.shape}"
            )
        self.ni, self.nj, self.nk = self.static_inj.shape
        self.inj_layers = self._layers(inj_k_range)
        self.prod_layers = self._layers(prod_k_range)
        self.bridge_layers = self._layers(bridge_k_range)
        self.radius = int(radius)

        stats = ("center", "local_mean", "local_min", "local_p10", "local_p90", "local_std")
        section_stats = ("section_mean", "section_p90")
        names = []
        for group in ("injw", "prodw"):
            for field_name in ("mobility", "pressure"):
                for view in ("xz", "yz"):
                    for band in ("target", "other", "bridge"):
                        names.extend(
                            [
                                f"{group}_static_{field_name}_{view}_{band}_{s}"
                                for s in stats
                            ]
                        )
                        names.extend(
                            [
                                f"{group}_static_{field_name}_{view}_{band}_{s}"
                                for s in section_stats
                            ]
                        )
            names.extend(
                [
                    f"{group}_static_xz_yz_target_mobility_balance",
                    f"{group}_static_target_bridge_mobility_contrast",
                    f"{group}_static_target_other_mobility_contrast",
                    f"{group}_static_xz_yz_target_pressure_balance",
                    f"{group}_static_target_bridge_pressure_contrast",
                    f"{group}_static_target_other_pressure_contrast",
                ]
            )
        pair_stats = ("mean", "min", "p10", "p90", "std")
        names.extend([f"inj_prod_static_target_path_{s}" for s in pair_stats])
        names.extend([f"inj_prod_static_bridge_path_{s}" for s in pair_stats])
        names.extend(
            [
                "static_inj_target_minus_prod_target_mobility",
                "static_inj_bridge_minus_prod_bridge_mobility",
                "static_inj_target_minus_prod_target_pressure",
                "static_inj_bridge_minus_prod_bridge_pressure",
            ]
        )
        self.feature_names = names

    def _layers(self, k_range):
        k0, k1 = int(k_range[0]) - 1, int(k_range[1]) - 1
        k0 = int(np.clip(k0, 0, self.nk - 1))
        k1 = int(np.clip(k1, 0, self.nk - 1))
        if k1 < k0:
            k0, k1 = k1, k0
        return np.arange(k0, k1 + 1, dtype=int)

    def _well_ij(self, well):
        i = int(np.clip(round(float(well[0])), 0, self.ni - 1))
        j = int(np.clip(round(float(well[1])), 0, self.nj - 1))
        return i, j

    def _i_slice(self, i):
        return slice(max(0, i - self.radius), min(self.ni, i + self.radius + 1))

    def _j_slice(self, j):
        return slice(max(0, j - self.radius), min(self.nj, j + self.radius + 1))

    @staticmethod
    def _stats(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            vals = np.asarray([0.0], dtype=float)
        return [
            float(np.mean(vals)),
            float(np.min(vals)),
            float(np.percentile(vals, 10)),
            float(np.percentile(vals, 90)),
            float(np.std(vals)),
        ]

    @staticmethod
    def _dist_stats(vals):
        return StaticWellConditionedTriViewFeatureExtractor._stats(vals)

    def _view_arrays(self, field, well, view, layers):
        i, j = self._well_ij(well)
        if view == "xz":
            section = field[:, j, :][:, layers]
            local = field[self._i_slice(i), j, :][:, layers]
        elif view == "yz":
            section = field[i, :, :][:, layers]
            local = field[i, self._j_slice(j), :][:, layers]
        else:
            raise ValueError(view)
        center = field[i, j, layers]
        return center, local.reshape(-1), section.reshape(-1)

    def _band_features(self, field, well, view, layers):
        center, local, section = self._view_arrays(field, well, view, layers)
        return [
            float(np.mean(center)),
            *self._stats(local),
            float(np.mean(section)),
            float(np.percentile(section, 90)),
        ]

    def _group_features(self, mobility, wells, target_layers, other_layers):
        row = []
        mob_target = []
        mob_bridge = []
        mob_other = []
        pressure_target = []
        pressure_bridge = []
        pressure_other = []
        xz_mob_target = []
        yz_mob_target = []
        xz_pressure_target = []
        yz_pressure_target = []
        for field, field_name, target, other in (
            (mobility, "mobility", mob_target, mob_other),
            (self.pressure, "pressure", pressure_target, pressure_other),
        ):
            for view in ("xz", "yz"):
                for layers in (target_layers, other_layers, self.bridge_layers):
                    per_well = np.asarray(
                        [self._band_features(field, w, view, layers) for w in wells],
                        dtype=float,
                    )
                    row.extend(np.mean(per_well, axis=0).tolist())
                    if field_name == "mobility":
                        if layers is target_layers:
                            mob_target.extend(per_well[:, 1].tolist())
                            (xz_mob_target if view == "xz" else yz_mob_target).extend(
                                per_well[:, 1].tolist()
                            )
                        elif layers is other_layers:
                            mob_other.extend(per_well[:, 1].tolist())
                        else:
                            mob_bridge.extend(per_well[:, 1].tolist())
                    else:
                        if layers is target_layers:
                            pressure_target.extend(per_well[:, 1].tolist())
                            (xz_pressure_target if view == "xz" else yz_pressure_target).extend(
                                per_well[:, 1].tolist()
                            )
                        elif layers is other_layers:
                            pressure_other.extend(per_well[:, 1].tolist())
                        else:
                            pressure_bridge.extend(per_well[:, 1].tolist())

        mob_target = np.asarray(mob_target if mob_target else [0.0])
        mob_other = np.asarray(mob_other if mob_other else [0.0])
        mob_bridge = np.asarray(mob_bridge if mob_bridge else [0.0])
        pressure_target = np.asarray(pressure_target if pressure_target else [0.0])
        pressure_other = np.asarray(pressure_other if pressure_other else [0.0])
        pressure_bridge = np.asarray(pressure_bridge if pressure_bridge else [0.0])
        row.extend(
            [
                float(np.mean(xz_mob_target) - np.mean(yz_mob_target)),
                float(np.mean(mob_target) - np.mean(mob_bridge)),
                float(np.mean(mob_target) - np.mean(mob_other)),
                float(np.mean(xz_pressure_target) - np.mean(yz_pressure_target)),
                float(np.mean(pressure_target) - np.mean(pressure_bridge)),
                float(np.mean(pressure_target) - np.mean(pressure_other)),
            ]
        )
        return row

    def _pair_path_values(self, inj_wells, prod_wells, layers):
        values = []
        for iw in np.asarray(inj_wells, dtype=float):
            ii, ij = self._well_ij(iw)
            for pw in np.asarray(prod_wells, dtype=float):
                pi, pj = self._well_ij(pw)
                i0, i1 = sorted((ii, pi))
                j0, j1 = sorted((ij, pj))
                pieces = [
                    self.static_inj[i0:i1 + 1, ij, :][:, layers],
                    self.static_inj[ii, j0:j1 + 1, :][:, layers],
                    self.static_prod[i0:i1 + 1, pj, :][:, layers],
                    self.static_prod[pi, j0:j1 + 1, :][:, layers],
                ]
                values.append(float(np.mean([np.mean(v) for v in pieces])))
        return np.asarray(values if values else [0.0], dtype=float)

    def __call__(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        feats = np.empty((X.shape[0], len(self.feature_names)), dtype=float)
        for r, x in enumerate(X):
            inj_wells = x[: N_INJ * 2].reshape(N_INJ, 2)
            prod_wells = x[N_INJ * 2 :].reshape(N_PRD, 2)
            inj_part = self._group_features(
                self.static_inj, inj_wells, self.inj_layers, self.prod_layers
            )
            prod_part = self._group_features(
                self.static_prod, prod_wells, self.prod_layers, self.inj_layers
            )
            row = list(inj_part) + list(prod_part)
            target_path = self._pair_path_values(inj_wells, prod_wells, self.inj_layers)
            bridge_path = self._pair_path_values(inj_wells, prod_wells, self.bridge_layers)
            row.extend(self._dist_stats(target_path))
            row.extend(self._dist_stats(bridge_path))
            row.extend(
                [
                    float(inj_part[1] - prod_part[1]),
                    float(inj_part[21] - prod_part[21]),
                    float(inj_part[61] - prod_part[61]),
                    float(inj_part[81] - prod_part[81]),
                ]
            )
            feats[r] = row
        return feats
