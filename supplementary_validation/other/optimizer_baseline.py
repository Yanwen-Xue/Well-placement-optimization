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


import os
import json
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import joblib

from cmg_runner import evaluate, SIM_YEARS_FINAL, OPT_RUNS_DIR
from objectives import compute_objectives, check_spacing, check_boundary, PENALTY_VALUE

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

BASELINE_DIR = os.environ.get("OPT_BASELINE_DIR", os.path.join(ROOT_DIR, "opt_runs_baseline"))
os.makedirs(BASELINE_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(BASELINE_DIR, "baseline.log"), mode="a", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


N_OBJ = 3
N_POP = 240
N_GEN = 180
N_DIM = 20
N_PARALLEL = 32
REF_P = 18

SBX_ETA_C = 20.0
SBX_P_C = 0.9
PM_ETA_M = 20.0
PM_P_M = 1.0 / N_DIM

N_INJ = 5
N_PROD = 5
D_INJ_INJ_M = 200.0
D_PROD_PROD_M = 200.0
D_INJ_PROD_M = 300.0
BOUND_MARGIN = 20

CKPT_INTERVAL = 5
NSGA3_FINAL_K = 32

USE_EARLY_STOP = True
MIN_GEN = 60
EARLY_STOP_PATIENCE = 20
HV_TOL = 0.01
FRONT_SIZE_TOL = 0.25
OBJ_MIN_TOL = 0.01
VALIDATION_WINDOW = 8
HV_REF_SLACK = 0.10

N_INJ = 5
N_PROD = 5
SURROGATE_DEFAULT = os.path.join(
    ROOT_DIR,
    "opt_runs_lhs_doe",
    "surrogate_model_0_343_f2_mobile_f3_hhi_68d.joblib",
)


def _eval_worker(args):
    rid, inj, prod, grid_top, grid_bot, sim_years, cleanup, bound_mask, dx, gen_subdir, local_rid = args
    result = evaluate(
        rid,
        inj,
        prod,
        grid_top,
        sim_years=sim_years,
        cleanup=cleanup,
        gen_subdir=gen_subdir,
        local_rid=local_rid,
        grid_bot_2d=grid_bot,
    )
    result["inj_wells"] = inj
    result["prod_wells"] = prod
    F1, F2, F3, _ = compute_objectives(
        result,
        bound_mask,
        dx,
        D_INJ_INJ_M,
        D_PROD_PROD_M,
        D_INJ_PROD_M,
    )
    return np.array([F1, F2, F3])


def _generate_reference_points(M, p):
    def _enum(remaining, n_vars, current):
        if n_vars == 1:
            yield current + [remaining]
            return
        for v in range(remaining + 1):
            yield from _enum(remaining - v, n_vars - 1, current + [v])

    return np.array([[v / p for v in combo] for combo in _enum(p, M, [])])


class PureNSGAIII:
    def __init__(
        self,
        seeds_path,
        run_id_start=0,
        cleanup_runs=False,
        use_early_stop=USE_EARLY_STOP,
        surrogate_path=None,
        ref_p=REF_P,
        final_k=NSGA3_FINAL_K,
        random_seed=None,
    ):
        self.seeds_path = seeds_path
        self.run_id = run_id_start
        self.sim_years = SIM_YEARS_FINAL
        self.cleanup = cleanup_runs
        self.use_early_stop = bool(use_early_stop)
        self.surrogate_path = surrogate_path
        self.surrogate_model = None if not surrogate_path else joblib.load(surrogate_path)
        self.ref_p = int(ref_p)
        self.final_k = int(final_k)
        self.random_seed = random_seed

        self.ref_points = _generate_reference_points(N_OBJ, p=self.ref_p)
        self._load_grid()

        self.hv_ref = None
        self.hv_history = []
        self.front_size_history = []
        self.obj_min_history = []

        log.info(
            "PureNSGAIII initialized: NI=%d NJ=%d dx=%.1fm valid_cells=%d sim_years=%d early_stop=%s ref_p=%d final_k=%d",
            self.NI,
            self.NJ,
            self.dx,
            len(self.valid_cells),
            self.sim_years,
            self.use_early_stop,
            self.ref_p,
            self.final_k,
        )

    def _load_grid(self):
        data = np.load(self.seeds_path, allow_pickle=False)
        self.bound_mask = data["bound_mask"].astype(bool)
        self.grid_top = data["grid_top_2d"]
        self.grid_bot = data["grid_bot_2d"] if "grid_bot_2d" in data else None
        self.NI = int(data["NI"])
        self.NJ = int(data["NJ"])
        self.dx = float(data["dx"])

        lo_i, hi_i = BOUND_MARGIN, self.NI - 1 - BOUND_MARGIN
        lo_j, hi_j = BOUND_MARGIN, self.NJ - 1 - BOUND_MARGIN
        ii, jj = np.indices(self.bound_mask.shape)
        rect_margin_mask = (
            (ii >= lo_i)
            & (ii <= hi_i)
            & (jj >= lo_j)
            & (jj <= hi_j)
        )
        self.bound_mask = self.bound_mask & rect_margin_mask
        self.valid_cells = np.argwhere(self.bound_mask)
        if len(self.valid_cells) < N_INJ + N_PROD:
            raise ValueError("not enough cells after boundary margin")

    def _decode(self, x):
        x = np.round(x).astype(int)
        inj = [(int(x[2 * k]), int(x[2 * k + 1])) for k in range(N_INJ)]
        prod = [
            (int(x[N_INJ * 2 + 2 * k]), int(x[N_INJ * 2 + 2 * k + 1]))
            for k in range(N_PROD)
        ]
        return inj, prod

    @staticmethod
    def _sort_pairs(pairs):
        return sorted((int(i), int(j)) for i, j in pairs)

    def _canonicalize_layout(self, x):
        x = np.round(np.asarray(x, dtype=float)).astype(float)
        inj, prod = self._decode(x)
        inj = self._sort_pairs(inj)
        prod = self._sort_pairs(prod)
        y = np.zeros(N_DIM, dtype=float)
        for k, (i, j) in enumerate(inj):
            y[2 * k], y[2 * k + 1] = float(i), float(j)
        for k, (i, j) in enumerate(prod):
            off = N_INJ * 2 + 2 * k
            y[off], y[off + 1] = float(i), float(j)
        return y

    def _layout_key(self, x):
        inj, prod = self._decode(x)
        inj = tuple(self._sort_pairs(inj))
        prod = tuple(self._sort_pairs(prod))
        return inj + prod

    def _clip(self, x):
        x = np.array(x, dtype=float)
        lo_i, hi_i = BOUND_MARGIN, self.NI - 1 - BOUND_MARGIN
        lo_j, hi_j = BOUND_MARGIN, self.NJ - 1 - BOUND_MARGIN
        for k in range(N_INJ + N_PROD):
            x[2 * k] = np.clip(x[2 * k], lo_i, hi_i)
            x[2 * k + 1] = np.clip(x[2 * k + 1], lo_j, hi_j)
        return x

    def _repair(self, x, rng):
        x = self._clip(x)
        for _ in range(50):
            inj, prod = self._decode(x)
            if check_boundary(inj, prod, self.bound_mask) and check_spacing(
                inj, prod, D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M, self.dx
            ):
                return self._canonicalize_layout(x)
            x = self._clip(x + rng.integers(-20, 21, size=N_DIM).astype(float))
        return self._canonicalize_layout(x)

    def _random_individual(self, rng):
        n_cells = len(self.valid_cells)
        for _ in range(300):
            chosen = self.valid_cells[rng.choice(n_cells, N_INJ + N_PROD, replace=False)]
            inj = [tuple(chosen[k]) for k in range(N_INJ)]
            prod = [tuple(chosen[k]) for k in range(N_INJ, N_INJ + N_PROD)]
            if check_spacing(inj, prod, D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M, self.dx):
                x = np.zeros(N_DIM, dtype=float)
                for k in range(N_INJ):
                    x[2 * k], x[2 * k + 1] = float(inj[k][0]), float(inj[k][1])
                for k in range(N_PROD):
                    x[N_INJ * 2 + 2 * k], x[N_INJ * 2 + 2 * k + 1] = float(prod[k][0]), float(prod[k][1])
                return self._canonicalize_layout(x)
        chosen = self.valid_cells[rng.choice(n_cells, N_INJ + N_PROD, replace=False)]
        x = np.zeros(N_DIM, dtype=float)
        for k in range(N_INJ + N_PROD):
            x[2 * k], x[2 * k + 1] = float(chosen[k, 0]), float(chosen[k, 1])
        return self._repair(x, rng)

    def _make_unique_batch(self, candidates, rng, existing_keys=None):
        seen = set(existing_keys or [])
        unique = []
        for x in candidates:
            x = self._canonicalize_layout(x)
            key = self._layout_key(x)
            if key in seen:
                continue
            seen.add(key)
            unique.append(x)
        while len(unique) < len(candidates):
            x = self._random_individual(rng)
            key = self._layout_key(x)
            if key in seen:
                continue
            seen.add(key)
            unique.append(x)
        return np.asarray(unique, dtype=float)

    def _sbx_crossover(self, x1, x2, rng):
        c1, c2 = x1.copy().astype(float), x2.copy().astype(float)
        if rng.random() > SBX_P_C:
            return c1, c2

        lo_i, hi_i = float(BOUND_MARGIN), float(self.NI - 1 - BOUND_MARGIN)
        lo_j, hi_j = float(BOUND_MARGIN), float(self.NJ - 1 - BOUND_MARGIN)
        for d in range(N_DIM):
            if rng.random() > 0.5:
                continue
            xl = lo_i if d % 2 == 0 else lo_j
            xu = hi_i if d % 2 == 0 else hi_j
            if abs(x1[d] - x2[d]) < 1e-9:
                continue
            y1, y2 = min(x1[d], x2[d]), max(x1[d], x2[d])
            u = rng.random()
            beta_l = 1.0 + 2.0 * (y1 - xl) / max(y2 - y1, 1e-10)
            alpha = 2.0 - (beta_l ** (-(SBX_ETA_C + 1)))
            if u <= 1.0 / alpha:
                betaq = (u * alpha) ** (1.0 / (SBX_ETA_C + 1))
            else:
                betaq = (1.0 / (2.0 - u * alpha)) ** (1.0 / (SBX_ETA_C + 1))
            c1[d] = 0.5 * ((x1[d] + x2[d]) - betaq * (x2[d] - x1[d]))
            c2[d] = 0.5 * ((x1[d] + x2[d]) + betaq * (x2[d] - x1[d]))
            c1[d] = np.clip(c1[d], xl, xu)
            c2[d] = np.clip(c2[d], xl, xu)
        return self._clip(c1), self._clip(c2)

    def _poly_mutate(self, x, rng):
        x = x.astype(float).copy()
        lo_i, hi_i = float(BOUND_MARGIN), float(self.NI - 1 - BOUND_MARGIN)
        lo_j, hi_j = float(BOUND_MARGIN), float(self.NJ - 1 - BOUND_MARGIN)
        for d in range(N_DIM):
            if rng.random() > PM_P_M:
                continue
            xl = lo_i if d % 2 == 0 else lo_j
            xu = hi_i if d % 2 == 0 else hi_j
            delta1 = (x[d] - xl) / max(xu - xl, 1.0)
            delta2 = (xu - x[d]) / max(xu - xl, 1.0)
            u = rng.random()
            if u <= 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (PM_ETA_M + 1))
                deltaq = val ** (1.0 / (PM_ETA_M + 1)) - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (PM_ETA_M + 1))
                deltaq = 1.0 - val ** (1.0 / (PM_ETA_M + 1))
            x[d] = np.clip(x[d] + deltaq * (xu - xl), xl, xu)
        return self._clip(x)

    def _check_layout_feasible(self, x):
        inj, prod = self._decode(x)
        return bool(
            check_boundary(inj, prod, self.bound_mask)
            and check_spacing(
                inj,
                prod,
                D_INJ_INJ_M,
                D_PROD_PROD_M,
                D_INJ_PROD_M,
                self.dx,
            )
        )

    def _surrogate_predict_pop(self, pop):
        pop = np.asarray(pop, dtype=float)
        objs = np.full((len(pop), N_OBJ), PENALTY_VALUE, dtype=float)
        feasible = np.array([self._check_layout_feasible(x) for x in pop], dtype=bool)
        if not np.any(feasible):
            return objs
        pred = np.asarray(self.surrogate_model.predict(pop[feasible]), dtype=float)
        if pred.shape != (int(feasible.sum()), N_OBJ):
            raise ValueError(
                f"surrogate prediction shape mismatch: got {pred.shape}, "
                f"expected {(int(feasible.sum()), N_OBJ)}"
            )
        bad = ~np.isfinite(pred).all(axis=1)
        pred[bad] = PENALTY_VALUE
        objs[feasible] = pred
        return objs

    def _eval_one(self, x, tag="", gen_subdir=None, local_rid=None):
        if self.surrogate_model is not None:
            obj = self._surrogate_predict_pop(np.asarray([x], dtype=float))[0]
            log.info(
                "[%s] surrogate  F1=%.4f F2=%.4f(sol=%.1f%%) F3_HHI=%.6f",
                tag,
                obj[0],
                obj[1],
                -obj[1] * 100,
                obj[2],
            )
            return obj
        inj, prod = self._decode(x)
        rid = self.run_id
        self.run_id += 1
        result = evaluate(
            rid,
            inj,
            prod,
            self.grid_top,
            sim_years=self.sim_years,
            cleanup=self.cleanup,
            gen_subdir=gen_subdir,
            local_rid=local_rid,
            grid_bot_2d=self.grid_bot,
        )
        result["inj_wells"] = inj
        result["prod_wells"] = prod
        F1, F2, F3, feas = compute_objectives(
            result,
            self.bound_mask,
            self.dx,
            D_INJ_INJ_M,
            D_PROD_PROD_M,
            D_INJ_PROD_M,
        )
        disp_id = f"{gen_subdir}/run_{local_rid:03d}" if gen_subdir else f"run_{rid:05d}"
        log.info(
            "[%s] %s  F1=%.4f F2=%.4f(diss=%.1f%%) F3_HHI=%.6f ok=%s  t=%.0fs",
            tag,
            disp_id,
            F1,
            F2,
            -F2 * 100,
            F3,
            feas,
            result["elapsed_s"],
        )
        return np.array([F1, F2, F3])

    def _select_representatives(self, objs, k):

        n = len(objs)
        if k is None or k <= 0 or n <= k:
            return list(range(n))

        objs = np.asarray(objs, dtype=float)
        mins = np.min(objs, axis=0)
        spans = np.ptp(objs, axis=0)
        norm = (objs - mins) / np.maximum(spans, 1e-12)

        selected = []
        for m in range(objs.shape[1]):
            idx = int(np.argmin(objs[:, m]))
            if idx not in selected:
                selected.append(idx)

        center = np.full(objs.shape[1], 0.5, dtype=float)
        idx_center = int(np.argmin(np.linalg.norm(norm - center, axis=1)))
        if idx_center not in selected:
            selected.append(idx_center)

        while len(selected) < min(k, n):
            d = np.min(
                np.linalg.norm(norm[:, None, :] - norm[selected][None, :, :], axis=2),
                axis=1,
            )
            d[selected] = -1.0
            selected.append(int(np.argmax(d)))

        return selected[:k]

    def _eval_pop(self, pop, tag="eval", gen_subdir=None):
        n = len(pop)
        objs = np.full((n, N_OBJ), PENALTY_VALUE)
        if self.surrogate_model is not None:
            objs = self._surrogate_predict_pop(pop)
            valid = int(np.sum(np.all(objs < PENALTY_VALUE / 2, axis=1)))
            log.info("[%s] surrogate batch evaluated: %d/%d feasible predictions", tag, valid, n)
            return objs

        if N_PARALLEL <= 1:
            for i, x in enumerate(pop):
                objs[i] = self._eval_one(x, tag=f"{tag}[{i+1:02d}]", gen_subdir=gen_subdir, local_rid=i + 1)
        else:
            base_rid = self.run_id
            self.run_id += n
            args_list = []
            for i, x in enumerate(pop):
                inj, prod = self._decode(x)
                args_list.append(
                    (
                        base_rid + i,
                        inj,
                        prod,
                        self.grid_top,
                        self.grid_bot,
                        self.sim_years,
                        self.cleanup,
                        self.bound_mask,
                        self.dx,
                        gen_subdir,
                        i + 1,
                    )
                )
            with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
                futs = {ex.submit(_eval_worker, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        objs[i] = fut.result()
                    except Exception as e:
                        log.warning("Parallel worker %d failed: %s", i, e)
        return objs

    @staticmethod
    def _dominates(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    @staticmethod
    def _nondominated_sort(objs):
        n = len(objs)
        dom_count = np.zeros(n, dtype=int)
        dom_set = [[] for _ in range(n)]
        fronts = [[]]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if PureNSGAIII._dominates(objs[i], objs[j]):
                    dom_set[i].append(j)
                elif PureNSGAIII._dominates(objs[j], objs[i]):
                    dom_count[i] += 1
            if dom_count[i] == 0:
                fronts[0].append(i)
        f = 0
        while fronts[f]:
            nf = []
            for i in fronts[f]:
                for j in dom_set[i]:
                    dom_count[j] -= 1
                    if dom_count[j] == 0:
                        nf.append(j)
            f += 1
            fronts.append(nf)
        return [fr for fr in fronts if fr]

    def _nsga3_select(self, combined_pop, combined_objs, n_keep):
        fronts = self._nondominated_sort(combined_objs)
        selected = []
        last_front = None
        for front in fronts:
            if len(selected) + len(front) <= n_keep:
                selected.extend(front)
            else:
                last_front = front
                break
        if last_front is None or len(selected) == n_keep:
            sel = np.array(selected[:n_keep])
            return combined_pop[sel].copy(), combined_objs[sel].copy()

        k = n_keep - len(selected)
        st_idx = np.array(selected + last_front)
        st_objs = combined_objs[st_idx]
        valid = np.all(st_objs < PENALTY_VALUE / 2, axis=1)
        if valid.sum() < N_OBJ:
            chosen = self._fallback_crowding(combined_objs, last_front, k)
        else:
            f_min = st_objs[valid].min(axis=0)
            f_shifted = st_objs - f_min
            weights_set = np.eye(N_OBJ) + 1e-6
            extreme = []
            for w in weights_set:
                asf = np.max(f_shifted * w, axis=1)
                extreme.append(st_idx[np.argmin(asf)])
            extreme = np.array(extreme)
            try:
                A = combined_objs[extreme] - f_min
                intercepts = np.linalg.solve(A, np.ones(N_OBJ))
                denom = 1.0 / intercepts
                denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
            except np.linalg.LinAlgError:
                vmax = st_objs[valid].max(axis=0)
                denom = np.where(vmax > f_min, vmax - f_min, 1.0)

            valid_global = np.all(combined_objs < PENALTY_VALUE / 2, axis=1)
            f_norm = np.full((len(combined_objs), N_OBJ), np.inf)
            f_norm[valid_global] = np.clip((combined_objs[valid_global] - f_min) / denom, 0, None)
            ref = self.ref_points
            ref_norm2 = np.sum(ref ** 2, axis=1, keepdims=True).T
            proj_scalar = f_norm @ ref.T / ref_norm2
            proj = proj_scalar[:, :, None] * ref[None, :, :]
            diff = f_norm[:, None, :] - proj
            dist_to_ref = np.sqrt(np.sum(diff ** 2, axis=2))
            assoc = np.argmin(dist_to_ref, axis=1)

            rho = np.zeros(len(ref), dtype=int)
            for idx in selected:
                rho[assoc[idx]] += 1

            lf_set = list(last_front)
            chosen = []
            while len(chosen) < k and lf_set:
                lf_assoc = [assoc[idx] for idx in lf_set]
                active_refs = list(set(lf_assoc))
                min_rho = min(rho[j] for j in active_refs)
                cands_ref = [j for j in active_refs if rho[j] == min_rho]
                chosen_ref = cands_ref[np.random.randint(len(cands_ref))]
                assoc_lf = [idx for idx in lf_set if assoc[idx] == chosen_ref]
                if rho[chosen_ref] == 0:
                    best = min(assoc_lf, key=lambda idx: dist_to_ref[idx, chosen_ref])
                else:
                    best = assoc_lf[np.random.randint(len(assoc_lf))]
                chosen.append(best)
                rho[chosen_ref] += 1
                lf_set.remove(best)

        final_idx = np.array(selected + chosen)
        return combined_pop[final_idx].copy(), combined_objs[final_idx].copy()

    def _fallback_crowding(self, objs, front, k):
        n = len(front)
        cd = np.zeros(n)
        if n <= 2:
            cd[:] = np.inf
        else:
            sub = objs[front]
            for m in range(sub.shape[1]):
                idx = np.argsort(sub[:, m])
                cd[idx[0]] = cd[idx[-1]] = np.inf
                span = sub[idx[-1], m] - sub[idx[0], m]
                if span == 0:
                    continue
                for kk in range(1, n - 1):
                    cd[idx[kk]] += (sub[idx[kk + 1], m] - sub[idx[kk - 1], m]) / span
        idx = np.argsort(-cd)
        return [front[i] for i in idx[:k]]

    def _hv_2d(self, points, ref):
        pts = points[np.all(points < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 0])]
        front, best_f2 = [], np.inf
        for p in pts:
            if p[1] < best_f2:
                front.append(p)
                best_f2 = p[1]
        front = np.array(front)
        hv = 0.0
        for i in range(len(front)):
            f1_next = front[i + 1, 0] if i + 1 < len(front) else ref[0]
            hv += (f1_next - front[i, 0]) * (ref[1] - front[i, 1])
        return float(hv)

    def _hv_3d(self, points, ref):
        pts = points[np.all(points < ref, axis=1)]
        if len(pts) == 0:
            return 0.0
        pts = pts[np.argsort(pts[:, 2])]
        hv = 0.0
        for i in range(len(pts)):
            f3_high = pts[i + 1, 2] if i + 1 < len(pts) else ref[2]
            slab = f3_high - pts[i, 2]
            if slab <= 0:
                continue
            hv += slab * self._hv_2d(pts[: i + 1, :2], ref[:2])
        return float(hv)

    def _compute_metrics(self, objs):
        valid = np.all(objs < PENALTY_VALUE / 2, axis=1)
        if valid.sum() == 0 or self.hv_ref is None:
            return 0.0, 0, np.full(N_OBJ, PENALTY_VALUE)
        valid_objs = objs[valid]
        fronts = self._nondominated_sort(valid_objs)
        front0_pts = valid_objs[fronts[0]]
        hv = self._hv_3d(front0_pts, self.hv_ref)
        return hv, len(fronts[0]), valid_objs.min(axis=0)

    def _check_early_stop(self):
        if len(self.hv_history) < EARLY_STOP_PATIENCE:
            return False, ""
        hv_window = self.hv_history[-EARLY_STOP_PATIENCE:]
        hv_base = max(abs(hv_window[0]), 1e-9)
        hv_gain = (hv_window[-1] - hv_window[0]) / hv_base

        fs_window = self.front_size_history[-EARLY_STOP_PATIENCE:]
        fs_mean = max(float(np.mean(fs_window)), 1.0)
        fs_swing = (max(fs_window) - min(fs_window)) / fs_mean

        om_window = np.array(self.obj_min_history[-EARLY_STOP_PATIENCE:])
        denom = np.maximum(np.abs(om_window[0]), 1e-9)
        obj_gain = (om_window[0] - om_window[-1]) / denom

        cond_hv = hv_gain < HV_TOL
        cond_fs = fs_swing < FRONT_SIZE_TOL
        cond_obj = bool(np.all(obj_gain < OBJ_MIN_TOL))
        diag = f"HV_gain={hv_gain*100:+.2f}% front_swing={fs_swing*100:.0f}% obj_gain_max={obj_gain.max()*100:+.2f}%"
        return cond_hv and cond_fs and cond_obj, diag

    def _save_checkpoint(self, gen, pop, objs):
        path = os.path.join(BASELINE_DIR, f"ckpt_gen{gen:04d}.npz")
        kw = dict(gen=gen, pop=pop, objs=objs, run_id=self.run_id)
        if self.hv_ref is not None:
            kw["hv_ref"] = self.hv_ref
        if self.hv_history:
            kw["hv_history"] = np.array(self.hv_history)
            kw["front_size_history"] = np.array(self.front_size_history)
            kw["obj_min_history"] = np.array(self.obj_min_history)
        np.savez(path, **kw)
        log.info("Checkpoint saved: %s", path)

    @staticmethod
    def load_checkpoint(path):
        data = np.load(path)
        extras = {
            k: data[k]
            for k in ("hv_ref", "hv_history", "front_size_history", "obj_min_history")
            if k in data.files
        }
        return int(data["gen"]), data["pop"], data["objs"], int(data["run_id"]), extras

    def run(self, resume_ckpt=None):
        log.info("=" * 70)
        log.info("Pure NSGA-III baseline start  seeds=%s", self.seeds_path)
        log.info(
            "  sim_years=%dyr  N_POP=%d  N_GEN=%d  N_PARALLEL=%d  early_stop=%s",
            self.sim_years,
            N_POP,
            N_GEN,
            N_PARALLEL,
            self.use_early_stop,
        )
        log.info("  evaluation=%s", "surrogate" if self.surrogate_model is not None else "high-fidelity simulation")
        log.info("=" * 70)

        rng = np.random.default_rng(self.random_seed)
        if resume_ckpt and os.path.exists(resume_ckpt):
            start_gen, pop, objs, self.run_id, extras = self.load_checkpoint(resume_ckpt)
            pop = pop.astype(float)
            if "hv_ref" in extras:
                self.hv_ref = extras["hv_ref"]
            if "hv_history" in extras:
                self.hv_history = list(extras["hv_history"])
                self.front_size_history = list(extras["front_size_history"])
                self.obj_min_history = [list(r) for r in extras["obj_min_history"]]
            log.info("Resumed from checkpoint: gen=%d run_id=%d", start_gen, self.run_id)
        else:
            start_gen = 0
            pop = np.array([self._random_individual(rng) for _ in range(N_POP)])
            pop = self._make_unique_batch(pop, rng)
            objs = self._eval_pop(pop, tag="init", gen_subdir="baseline_gen_000")

        if self.hv_ref is None:
            valid = np.all(objs < PENALTY_VALUE / 2, axis=1)
            if valid.sum() > 0:
                vmax = objs[valid].max(axis=0)
                vmin = objs[valid].min(axis=0)
                span = np.where(vmax - vmin > 1e-9, vmax - vmin, 1.0)
                self.hv_ref = vmax + HV_REF_SLACK * span
            else:
                self.hv_ref = np.full(N_OBJ, PENALTY_VALUE)

        if not self.hv_history:
            hv0, fs0, om0 = self._compute_metrics(objs)
            self.hv_history.append(hv0)
            self.front_size_history.append(fs0)
            self.obj_min_history.append(om0.tolist())

        gen = start_gen
        stop_pending_at = None
        stop_pending_hv = None
        while gen < N_GEN:
            t0 = time.time()
            gen_label = f"baseline_gen_{gen+1:03d}"

            offspring = []
            idx = rng.permutation(N_POP)
            for k in range(0, N_POP, 2):
                p1 = pop[idx[k]].copy()
                p2 = pop[idx[min(k + 1, N_POP - 1)]].copy()
                c1, c2 = self._sbx_crossover(p1, p2, rng)
                c1 = self._poly_mutate(c1, rng)
                c2 = self._poly_mutate(c2, rng)
                offspring.append(self._repair(c1, rng))
                offspring.append(self._repair(c2, rng))
            offspring = self._make_unique_batch(np.array(offspring[:N_POP]), rng, existing_keys={self._layout_key(x) for x in pop})

            offspring_objs = self._eval_pop(offspring, tag=f"SBX-g{gen+1}", gen_subdir=gen_label)
            pop, objs = self._nsga3_select(np.vstack([pop, offspring]), np.vstack([objs, offspring_objs]), N_POP)

            elapsed = time.time() - t0
            hv, fs, om = self._compute_metrics(objs)
            self.hv_history.append(hv)
            self.front_size_history.append(fs)
            self.obj_min_history.append(om.tolist())
            log.info(
                "Gen %3d/%d  t=%.0fs  |front0|=%d  HV=%.4e  F1=%.4f  F2=%.6f(diss=%.4f%%)  F3_HHI=%.6f",
                gen + 1,
                N_GEN,
                elapsed,
                fs,
                hv,
                om[0],
                om[1],
                -om[1] * 100.0,
                om[2],
            )

            if self.use_early_stop:
                if stop_pending_at is None:
                    if (gen + 1) >= MIN_GEN:
                        triggered, diag = self._check_early_stop()
                        if triggered:
                            stop_pending_at = gen
                            stop_pending_hv = hv
                            log.info("Early stop triggered at gen=%d -> %s", gen + 1, diag)
                else:
                    gens_since = gen - stop_pending_at
                    rebound = stop_pending_hv * (1.0 + HV_TOL)
                    if hv > rebound:
                        log.info("Validation window rebound detected; cancel early stop.")
                        stop_pending_at = None
                        stop_pending_hv = None
                    elif gens_since >= VALIDATION_WINDOW:
                        log.info("Validation window passed; stop at gen=%d", gen + 1)
                        if (gen + 1) % CKPT_INTERVAL != 0:
                            self._save_checkpoint(gen + 1, pop, objs)
                        gen += 1
                        break

            if (gen + 1) % CKPT_INTERVAL == 0:
                self._save_checkpoint(gen + 1, pop, objs)
            gen += 1

        fronts = self._nondominated_sort(objs)
        front0 = fronts[0]
        if len(front0) > self.final_k:
            idx = self._select_representatives(objs[front0], self.final_k)
            front0 = [front0[i] for i in idx]

        final_pop = pop[front0]
        final_objs = objs[front0]

        uniq_pop = []
        uniq_objs = []
        seen = set()
        for x, obj in zip(final_pop, final_objs):
            key = self._layout_key(x)
            if key in seen:
                continue
            seen.add(key)
            uniq_pop.append(x)
            uniq_objs.append(obj)
        if uniq_pop:
            final_pop = np.asarray(uniq_pop, dtype=float)
            final_objs = np.asarray(uniq_objs, dtype=float)
        self._save_checkpoint(gen, final_pop, final_objs)
        self._report(final_pop, final_objs)
        return final_pop, final_objs

    def _report(self, pop, objs):
        seen = set()
        results = []
        for k, (x, obj) in enumerate(zip(pop, objs)):
            key = self._layout_key(x)
            if key in seen:
                continue
            seen.add(key)
            inj, prod = self._decode(x)
            results.append(
                {
                    "rank": k + 1,
                    "F1_pressure_rise": float(obj[0]),
                    "pressure_rise_pct": float(obj[0] * 100),
                    "F2_neg_dissolution_ratio": float(obj[1]),
                    "dissolution_ratio": float(-obj[1]),
                    "F3_brine_concentration_index": float(obj[2]),
                    "inj_wells": [[int(i) + 1, int(j) + 1] for i, j in inj],
                    "prod_wells": [[int(i) + 1, int(j) + 1] for i, j in prod],
                }
            )
        out_path = os.path.join(BASELINE_DIR, "pareto_solutions_baseline.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Baseline results saved to %s", out_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pure NSGA-III baseline")
    parser.add_argument(
        "--seeds",
        default=os.path.join(ROOT_DIR, "opt_seeds.npz"),
        help="path to opt_seeds.npz",
    )
    parser.add_argument("--resume", default=None, help="checkpoint path")
    parser.add_argument("--run-id", type=int, default=0, help="starting run-id offset")
    parser.add_argument("--surrogate", default=SURROGATE_DEFAULT, help="joblib surrogate model path")
    parser.add_argument("--cleanup", action="store_true", help="remove intermediate CMG files")
    parser.add_argument("--parallel", type=int, default=None, help=f"parallel CMG workers (default {N_PARALLEL})")
    parser.add_argument("--pop", type=int, default=None, help=f"population size override (default {N_POP})")
    parser.add_argument("--gen", type=int, default=None, help=f"generation cap override (default {N_GEN})")
    parser.add_argument("--ref-p", type=int, default=None, help=f"NSGA-III Das-Dennis reference point divisions (default {REF_P})")
    parser.add_argument("--final-k", type=int, default=None, help=f"number of representative Pareto solutions (default {NSGA3_FINAL_K})")
    parser.add_argument("--seed", type=int, default=None, help="optional random seed for reproducible optimizer comparisons")
    parser.add_argument("--out-dir", default=None, help="output directory for checkpoints, log, and Pareto JSON")
    parser.add_argument("--use-early-stop", dest="use_early_stop", action="store_true", help="enable early stop")
    parser.add_argument("--no-early-stop", dest="use_early_stop", action="store_false", help="disable early stop")
    parser.set_defaults(use_early_stop=USE_EARLY_STOP)
    args = parser.parse_args()

    if args.out_dir is not None:
        BASELINE_DIR = os.path.abspath(args.out_dir)
        os.makedirs(BASELINE_DIR, exist_ok=True)
        for handler in list(log.handlers):
            if isinstance(handler, logging.FileHandler):
                log.removeHandler(handler)
                handler.close()
        log.addHandler(logging.FileHandler(os.path.join(BASELINE_DIR, "baseline.log"), mode="a", encoding="utf-8"))

    if args.parallel is not None:
        N_PARALLEL = args.parallel
    if args.pop is not None:
        N_POP = args.pop
    if args.gen is not None:
        N_GEN = args.gen
    if args.ref_p is not None:
        REF_P = args.ref_p
    if args.final_k is not None:
        NSGA3_FINAL_K = args.final_k

    opt = PureNSGAIII(
        seeds_path=args.seeds,
        run_id_start=args.run_id,
        cleanup_runs=args.cleanup,
        use_early_stop=args.use_early_stop,
        surrogate_path=args.surrogate,
        ref_p=REF_P,
        final_k=NSGA3_FINAL_K,
        random_seed=args.seed,
    )
    opt.run(resume_ckpt=args.resume)
