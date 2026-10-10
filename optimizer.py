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


import os, json, time, logging
import numpy as np
import joblib
from scipy.optimize import linear_sum_assignment
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations_with_replacement

from cmg_runner import (evaluate, OPT_RUNS_DIR, SIM_YEARS_FINAL)
from objectives import (compute_objectives, normalise_objectives,
                        check_spacing, check_boundary, PENALTY_VALUE)

# Make a clean checkout runnable before configuring the file logger.
os.makedirs(OPT_RUNS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(OPT_RUNS_DIR, "optimizer.log"),
            mode="a",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger(__name__)


N_OBJ = 3
N_POP = 80
N_GEN = 150
N_DIM = 20
N_PARALLEL = 32
REF_P = 10


DE_F_INIT = 0.5
DE_CR_INIT = 0.9
DE_C_ADAPT = 0.1
DE_P_BEST = 0.10


USE_DE = True
USE_OPT_PERM_ALIGN = True
USE_EARLY_STOP = True


CKPT_INTERVAL = int(os.environ.get("CKPT_INTERVAL", "5"))
NSGA3_FINAL_K = 16


MIN_GEN = 30
EARLY_STOP_PATIENCE = 15
HV_TOL = 0.005
FRONT_SIZE_TOL = 0.20
OBJ_MIN_TOL = 0.005
VALIDATION_WINDOW = 5
HV_REF_SLACK = 0.10
FINAL_POLISH = True


N_INJ = 5
N_PROD = 5
D_INJ_INJ_M = 200.0
D_PROD_PROD_M = 200.0
D_INJ_PROD_M = 300.0
BOUND_MARGIN = 20

SURROGATE_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "opt_runs_lhs_doe",
    "component_f3_model_0_623_official_log_20260813.joblib",
)


def _eval_worker(args):
    """Worker for parallel evaluation 鈥?top-level so ProcessPoolExecutor can pickle it."""
    rid, inj, prod, grid_top, grid_bot, sim_years, cleanup, bound_mask, dx, gen_subdir, local_rid = args
    result = evaluate(rid, inj, prod, grid_top,
                      sim_years=sim_years, cleanup=cleanup,
                      gen_subdir=gen_subdir, local_rid=local_rid,
                      grid_bot_2d=grid_bot)
    result["inj_wells"]  = inj
    result["prod_wells"] = prod
    F1, F2, F3, _ = compute_objectives(
        result, bound_mask, dx,
        D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
    )
    return np.array([F1, F2, F3])


def _generate_reference_points(M, p):
\
\
\
\
\
\
\
\

    def _enum(remaining, n_vars):
        if n_vars == 1:
            yield [remaining]
            return
        for v in range(remaining + 1):
            for rest in _enum(remaining - v, n_vars - 1):
                yield [v] + rest

    return np.array([[v / p for v in combo] for combo in _enum(p, M)])


def _hv_2d(points, ref):

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
    n  = len(front)
    hv = 0.0
    for i in range(n):
        f1_next = front[i+1, 0] if i+1 < n else ref[0]
        hv += (f1_next - front[i, 0]) * (ref[1] - front[i, 1])
    return float(hv)


def _hv_3d(points, ref):

    pts = points[np.all(points < ref, axis=1)]
    if len(pts) == 0:
        return 0.0
    pts = pts[np.argsort(pts[:, 2])]
    n  = len(pts)
    hv = 0.0
    for i in range(n):
        f3_high = pts[i+1, 2] if i+1 < n else ref[2]
        slab    = f3_high - pts[i, 2]
        if slab <= 0:
            continue
        hv += slab * _hv_2d(pts[:i+1, :2], ref[:2])
    return float(hv)

class HybridNSGAIII:


    def __init__(self, seeds_path,
                 run_id_start=0,
                 cleanup_runs=False,
                 surrogate_path=None,
                 use_de=USE_DE,
                 use_opt_perm_align=USE_OPT_PERM_ALIGN,
                 use_early_stop=USE_EARLY_STOP,
                 random_seed=None):
        self.seeds_path = seeds_path
        self.run_id = run_id_start
        self.sim_years = SIM_YEARS_FINAL
        self.cleanup = cleanup_runs
        self.surrogate_path = surrogate_path
        self.surrogate_model = None
        self.use_de = bool(use_de)
        self.use_opt_perm_align = bool(use_opt_perm_align)
        self.use_early_stop = bool(use_early_stop)
        self.random_seed = random_seed

        self.ref_points = _generate_reference_points(N_OBJ, p=REF_P)
        log.info("NSGA-III reference points: %d (M=%d, p=%d)", len(self.ref_points), N_OBJ, REF_P)

        os.makedirs(OPT_RUNS_DIR, exist_ok=True)
        self._load_seeds()
        self._load_surrogate()

        self.mu_F = DE_F_INIT
        self.mu_CR = DE_CR_INIT

        self.hv_ref = None
        self.hv_history = []
        self.front_size_history = []
        self.obj_min_history = []

        log.info(
            "Adaptive HybridNSGAIII initialized: NI=%d NJ=%d dx=%.1fm n_seeds=%d sim_years=%d",
            self.NI, self.NJ, self.dx, len(self.seeds), self.sim_years,
        )
        log.info(
            "Ablation flags: de=%s opt_perm_align=%s early_stop=%s",
            self.use_de, self.use_opt_perm_align, self.use_early_stop,
        )
        log.info(
            "JADE setup: mu_F=%.2f mu_CR=%.2f c=%.2f",
            self.mu_F, self.mu_CR, DE_C_ADAPT,
        )
        if self.surrogate_model is not None:
            log.info("Evaluation backend: surrogate model (%s)", self.surrogate_path)
        else:
            log.info("Evaluation backend: high-fidelity simulation")

    def _load_surrogate(self):
        if not self.surrogate_path:
            return
        if not os.path.exists(self.surrogate_path):
            raise FileNotFoundError(f"surrogate model not found: {self.surrogate_path}")
        self.surrogate_model = joblib.load(self.surrogate_path)


    def _align_to_target(self, target_x, source_x):
\
\
\
\
\
\
\
\

        s = np.asarray(source_x, dtype=float).copy()
        if not self.use_opt_perm_align:
            return s
        t = np.asarray(target_x, dtype=float)


        t_inj = t[:N_INJ*2].reshape(N_INJ, 2)
        s_inj = s[:N_INJ*2].reshape(N_INJ, 2)
        cost  = np.sum((t_inj[:, None, :] - s_inj[None, :, :]) ** 2, axis=2)
        _, col = linear_sum_assignment(cost)
        s[:N_INJ*2] = s_inj[col].flatten()


        off   = N_INJ * 2
        t_prd = t[off:].reshape(N_PROD, 2)
        s_prd = s[off:].reshape(N_PROD, 2)
        cost  = np.sum((t_prd[:, None, :] - s_prd[None, :, :]) ** 2, axis=2)
        _, col = linear_sum_assignment(cost)
        s[off:] = s_prd[col].flatten()

        return s


    def _load_seeds(self):
        data = np.load(self.seeds_path, allow_pickle=False)
        self.inj_seeds  = data["inj_wells"]
        self.prod_seeds = data["prod_wells"]
        self.bound_mask = data["bound_mask"].astype(bool)
        self.grid_top   = data["grid_top_2d"] if "grid_top_2d" in data else None


        self.grid_bot   = data["grid_bot_2d"] if "grid_bot_2d" in data else None
        self.NI         = int(data["NI"])
        self.NJ         = int(data["NJ"])
        self.dx         = float(data["dx"])
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
        if np.count_nonzero(self.bound_mask) < N_INJ + N_PROD:
            raise ValueError("not enough cells after boundary margin")

        n   = len(self.inj_seeds)
        pop = np.zeros((n, N_DIM), dtype=np.int32)
        for s in range(n):
            pop[s, :N_INJ*2] = self.inj_seeds[s].flatten()
            pop[s, N_INJ*2:] = self.prod_seeds[s].flatten()
        self.seeds = pop


    def _decode(self, x):
        x    = np.round(x).astype(int)
        inj  = [(int(x[2*k]),           int(x[2*k+1]))            for k in range(N_INJ)]
        prod = [(int(x[N_INJ*2 + 2*k]), int(x[N_INJ*2 + 2*k+1])) for k in range(N_PROD)]
        return inj, prod

    def _clip(self, x):
        x    = np.array(x, dtype=float)
        lo_i, hi_i = BOUND_MARGIN, self.NI - 1 - BOUND_MARGIN
        lo_j, hi_j = BOUND_MARGIN, self.NJ - 1 - BOUND_MARGIN
        for k in range(N_INJ + N_PROD):
            x[2*k]   = np.clip(x[2*k],   lo_i, hi_i)
            x[2*k+1] = np.clip(x[2*k+1], lo_j, hi_j)
        return x

    def _repair(self, x, rng):
        x = self._clip(x)
        for _ in range(50):
            inj, prod  = self._decode(x)
            ok_bound   = check_boundary(inj, prod, self.bound_mask)
            ok_spacing = check_spacing(inj, prod,
                                       D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
                                       self.dx)
            if ok_bound and ok_spacing:
                return x
            x = self._clip(x + rng.integers(-20, 21, size=N_DIM).astype(float))
        return x

    def _random_feasible_layout(self, rng):
        valid_cells = np.argwhere(self.bound_mask)
        if len(valid_cells) < N_INJ + N_PROD:
            raise ValueError("not enough valid cells to initialize a well layout")
        for _ in range(200):
            chosen = valid_cells[
                rng.choice(len(valid_cells), N_INJ + N_PROD, replace=False)
            ]
            x = np.zeros(N_DIM, dtype=float)
            for k in range(N_INJ + N_PROD):
                x[2 * k] = float(chosen[k, 0])
                x[2 * k + 1] = float(chosen[k, 1])
            x = self._repair(x, rng)
            inj, prod = self._decode(x)
            if (check_boundary(inj, prod, self.bound_mask) and
                    check_spacing(inj, prod,
                                  D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
                                  self.dx)):
                return x
        return x

    def _random_offspring(self, pop, rng):

        offspring = []
        idx = rng.permutation(len(pop))
        for k in range(len(pop)):
            base = pop[idx[k]].copy()
            noise = rng.integers(-15, 16, size=N_DIM).astype(float)
            child = self._repair(base + noise, rng)
            offspring.append(child)
        return np.array(offspring, dtype=float)


    def _check_layout_feasible(self, x):
        inj, prod = self._decode(x)
        ok_bound = check_boundary(inj, prod, self.bound_mask)
        ok_spacing = check_spacing(
            inj, prod,
            D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
            self.dx,
        )
        return bool(ok_bound and ok_spacing)

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

    def _eval_one(self, x, sim_years, tag="", gen_subdir=None, local_rid=None):
        if self.surrogate_model is not None:
            obj = self._surrogate_predict_pop(np.asarray([x], dtype=float))[0]
            disp_id = f"{gen_subdir}/run_{local_rid:03d}" if gen_subdir else "surrogate"
            log.info(
                "[%s] %s  F1=%.4f F2_plume_km2=%.4f F3_pressure_cv=%.6f  backend=surrogate",
                tag,
                disp_id,
                obj[0],
                obj[1],
                obj[2],
            )
            return obj

        if self.grid_top is None:
            raise ValueError("full-physics evaluation requires grid_top_2d in the seed file")
        inj, prod = self._decode(x)
        rid       = self.run_id
        self.run_id += 1

        result = evaluate(rid, inj, prod, self.grid_top,
                          sim_years=sim_years, cleanup=self.cleanup,
                          gen_subdir=gen_subdir, local_rid=local_rid,
                          grid_bot_2d=self.grid_bot)
        result["inj_wells"]  = inj
        result["prod_wells"] = prod
        F1, F2, F3, feas = compute_objectives(
            result, self.bound_mask, self.dx,
            D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
        )
        disp_id = f"{gen_subdir}/run_{local_rid:03d}" if gen_subdir else f"run_{rid:05d}"
        log.info("[%s] %s  F1=%.4f F2_plume_km2=%.4f F3_pressure_cv=%.6f ok=%s  t=%.0fs",
                 tag, disp_id, F1, F2, F3, feas, result["elapsed_s"])
        return np.array([F1, F2, F3])

    def _eval_pop(self, pop, sim_years, tag="eval", gen_subdir=None):
        n    = len(pop)
        objs = np.full((n, N_OBJ), PENALTY_VALUE)

        if self.surrogate_model is not None:
            objs = self._surrogate_predict_pop(pop)
            valid = int(np.sum(np.all(objs < PENALTY_VALUE / 2, axis=1)))
            log.info("[%s] surrogate batch evaluated: %d/%d feasible predictions", tag, valid, n)
            return objs

        if self.grid_top is None:
            raise ValueError("full-physics evaluation requires grid_top_2d in the seed file")
        if N_PARALLEL <= 1:
            for i, x in enumerate(pop):
                objs[i] = self._eval_one(x, sim_years, tag=f"{tag}[{i+1:02d}]",
                                          gen_subdir=gen_subdir, local_rid=i + 1)
        else:
            base_rid    = self.run_id
            self.run_id += n
            args_list   = []
            for i, x in enumerate(pop):
                inj, prod = self._decode(x)
                args_list.append((base_rid + i, inj, prod, self.grid_top,
                                  self.grid_bot, sim_years, self.cleanup,
                                  self.bound_mask, self.dx, gen_subdir, i + 1))
            with ProcessPoolExecutor(max_workers=N_PARALLEL) as ex:
                futs = {ex.submit(_eval_worker, a): i
                        for i, a in enumerate(args_list)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        objs[i] = fut.result()
                    except Exception as e:
                        log.warning("骞惰璇勪及Worker %d 寮傚父: %s", i, e)
        return objs


    @staticmethod
    def _dominates(a, b):
        return bool(np.all(a <= b) and np.any(a < b))

    @staticmethod
    def _nondominated_sort(objs):
        n         = len(objs)
        dom_count = np.zeros(n, dtype=int)
        dom_set   = [[] for _ in range(n)]
        fronts    = [[]]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if HybridNSGAIII._dominates(objs[i], objs[j]):
                    dom_set[i].append(j)
                elif HybridNSGAIII._dominates(objs[j], objs[i]):
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


    def _nsga3_select(self, combined_pop, combined_objs, n_keep, rng):
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

        fronts   = self._nondominated_sort(combined_objs)
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
            return combined_pop[sel].copy(), combined_objs[sel].copy(), sel


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
                b = np.ones(N_OBJ)
                intercepts = np.linalg.solve(A, b)
                denom = 1.0 / intercepts
                denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
            except np.linalg.LinAlgError:
                denom = np.where(
                    st_objs[valid].max(axis=0) > f_min,
                    st_objs[valid].max(axis=0) - f_min,
                    1.0,
                )


            valid_global = np.all(combined_objs < PENALTY_VALUE / 2, axis=1)
            f_norm = np.full((len(combined_objs), N_OBJ), np.inf)
            f_norm[valid_global] = np.clip(
                (combined_objs[valid_global] - f_min) / denom, 0, None
            )


            ref = self.ref_points


            ref_norm2 = np.sum(ref ** 2, axis=1, keepdims=True).T

            proj_scalar = f_norm @ ref.T / ref_norm2
            proj = proj_scalar[:, :, None] * ref[None, :, :]
            diff = f_norm[:, None, :] - proj
            dist_to_ref = np.sqrt(np.sum(diff ** 2, axis=2))

            assoc = np.argmin(dist_to_ref, axis=1)


            n_ref = len(ref)
            rho   = np.zeros(n_ref, dtype=int)
            for idx in selected:
                rho[assoc[idx]] += 1

            lf_set  = list(last_front)
            chosen  = []
            while len(chosen) < k and lf_set:

                lf_assoc = [assoc[idx] for idx in lf_set]
                active_refs = list(set(lf_assoc))
                min_rho = min(rho[j] for j in active_refs)
                cands_ref = [j for j in active_refs if rho[j] == min_rho]
                chosen_ref = cands_ref[rng.integers(len(cands_ref))]


                assoc_lf = [idx for idx in lf_set if assoc[idx] == chosen_ref]
                if rho[chosen_ref] == 0:

                    best = min(assoc_lf,
                               key=lambda idx: dist_to_ref[idx, chosen_ref])
                else:
                    best = assoc_lf[rng.integers(len(assoc_lf))]

                chosen.append(best)
                rho[chosen_ref] += 1
                lf_set.remove(best)

        final_idx = np.array(selected + chosen)
        return combined_pop[final_idx].copy(), combined_objs[final_idx].copy(), final_idx

    def _fallback_crowding(self, objs, front, k):

        n  = len(front)
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
                    cd[idx[kk]] += (sub[idx[kk+1], m] - sub[idx[kk-1], m]) / span
        idx = np.argsort(-cd)
        return [front[i] for i in idx[:k]]


    def _de_mutate(self, pop, objs, rng):

        n       = len(pop)
        n_pbest = max(1, int(n * DE_P_BEST))

        fronts      = self._nondominated_sort(objs)
        ranked      = []
        for front in fronts:
            ranked.extend(front)
            if len(ranked) >= n_pbest:
                break
        pbest_pool = ranked[:n_pbest]

        trials  = np.empty_like(pop, dtype=float)
        F_vals  = np.empty(n)
        CR_vals = np.empty(n)

        for i in range(n):

            F_i = float(np.clip(
                self.mu_F + 0.1 * np.tan(np.pi * (rng.random() - 0.5)),
                0.1, 1.0
            ))

            CR_i = float(np.clip(rng.normal(self.mu_CR, 0.1), 0.0, 1.0))
            F_vals[i]  = F_i
            CR_vals[i] = CR_i

            cands  = [j for j in range(n) if j != i]
            r1, r2 = rng.choice(cands, 2, replace=False)
            pb     = pbest_pool[int(rng.integers(len(pbest_pool)))]


            pb_a = self._align_to_target(pop[i], pop[pb])
            r1_a = self._align_to_target(pop[i], pop[r1])
            r2_a = self._align_to_target(pop[i], pop[r2])

            mutant = (pop[i]
                      + F_i * (pb_a - pop[i])
                      + F_i * (r1_a - r2_a))
            mutant = self._clip(mutant)

            trial = pop[i].astype(float).copy()
            jrand = int(rng.integers(N_DIM))
            for d in range(N_DIM):
                if d == jrand or rng.random() < CR_i:
                    trial[d] = mutant[d]

            trials[i] = self._clip(trial)
        return trials, F_vals, CR_vals

    def _update_de_params(self, S_F, S_CR):

        if len(S_F) < 1:
            return
        S_F  = np.array(S_F, dtype=float)
        S_CR = np.array(S_CR, dtype=float)
        lm_F  = float(np.sum(S_F ** 2) / np.sum(S_F))
        am_CR = float(np.mean(S_CR))
        self.mu_F  = (1 - DE_C_ADAPT) * self.mu_F  + DE_C_ADAPT * lm_F
        self.mu_CR = (1 - DE_C_ADAPT) * self.mu_CR + DE_C_ADAPT * am_CR
        self.mu_F  = float(np.clip(self.mu_F,  0.1, 0.9))
        self.mu_CR = float(np.clip(self.mu_CR, 0.1, 1.0))


    def _compute_metrics(self, objs):

        valid = np.all(objs < PENALTY_VALUE / 2, axis=1)
        if valid.sum() == 0 or self.hv_ref is None:
            return 0.0, 0, np.full(N_OBJ, PENALTY_VALUE)
        valid_objs = objs[valid]
        fronts     = self._nondominated_sort(valid_objs)
        front0_pts = valid_objs[fronts[0]]
        hv         = _hv_3d(front0_pts, self.hv_ref)
        return hv, len(fronts[0]), valid_objs.min(axis=0)

    def _check_early_stop(self):

        if len(self.hv_history) < EARLY_STOP_PATIENCE:
            return False, ""

        hv_window = self.hv_history[-EARLY_STOP_PATIENCE:]
        hv_base   = max(abs(hv_window[0]), 1e-9)
        hv_gain   = (hv_window[-1] - hv_window[0]) / hv_base

        if not self.use_early_stop:
            return False, ""

        fs_window = self.front_size_history[-EARLY_STOP_PATIENCE:]
        fs_mean   = max(float(np.mean(fs_window)), 1.0)
        fs_swing  = (max(fs_window) - min(fs_window)) / fs_mean

        om_window = np.array(self.obj_min_history[-EARLY_STOP_PATIENCE:])
        denom     = np.maximum(np.abs(om_window[0]), 1e-9)
        obj_gain  = (om_window[0] - om_window[-1]) / denom

        cond_hv  = hv_gain  < HV_TOL
        cond_fs  = fs_swing < FRONT_SIZE_TOL
        cond_obj = bool(np.all(obj_gain < OBJ_MIN_TOL))

        diag = (
            f"HV_gain={hv_gain*100:+.2f}%[tol={HV_TOL*100:.1f}%,ok={cond_hv}] "
            f"front_swing={fs_swing*100:.0f}%[tol={FRONT_SIZE_TOL*100:.0f}%,ok={cond_fs}] "
            f"obj_gain_max={obj_gain.max()*100:+.2f}%[tol={OBJ_MIN_TOL*100:.1f}%,ok={cond_obj}]"
        )
        return (cond_hv and cond_fs and cond_obj), diag


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

    def _save_checkpoint(self, gen, pop, objs):
        path = os.path.join(OPT_RUNS_DIR, f"ckpt_gen{gen:04d}.npz")
        kw = dict(
            gen=gen,
            pop=pop,
            objs=objs,
            run_id=self.run_id,
            surrogate_path=np.array("" if self.surrogate_path is None else self.surrogate_path),
        )
        if self.hv_ref is not None:
            kw["hv_ref"] = self.hv_ref
        if self.hv_history:
            kw["hv_history"]         = np.array(self.hv_history)
            kw["front_size_history"] = np.array(self.front_size_history)
            kw["obj_min_history"]    = np.array(self.obj_min_history)
        np.savez(path, **kw)
        log.info("妫€鏌ョ偣淇濆瓨: %s", path)

    @staticmethod
    def load_checkpoint(path):
        data   = np.load(path)
        gen    = int(data["gen"])
        pop    = data["pop"]
        objs   = data["objs"]
        run_id = int(data["run_id"])
        extras = {k: data[k] for k in ("hv_ref", "hv_history",
                                       "front_size_history", "obj_min_history")
                  if k in data.files}
        return gen, pop, objs, run_id, extras


    def run(self, resume_ckpt=None):
        log.info("=" * 70)
        log.info("CO2浜曚綅浼樺寲鍚姩锛圚ybrid NSGA-III锛? seeds=%s", self.seeds_path)
        log.info("  sim_years=%dyr  N_POP=%d  MAX_GEN=%d  N_PARALLEL=%d",
                 self.sim_years, N_POP, N_GEN, N_PARALLEL)
        log.info("  evaluation=%s",
                 "surrogate" if self.surrogate_model is not None else "high-fidelity simulation")
        log.info("  flags: de=%s opt_perm_align=%s early_stop=%s",
                 self.use_de, self.use_opt_perm_align, self.use_early_stop)
        log.info("  ref_points=%d  init_mu_F=%.2f  init_mu_CR=%.2f  DE_P_BEST=%.2f  [JADE adaptive]",
                 len(self.ref_points), self.mu_F, self.mu_CR, DE_P_BEST)
        log.info("  early_stop: MIN_GEN=%d PATIENCE=%d HV_TOL=%.1f%% FRONT_TOL=%.0f%% "
                 "OBJ_TOL=%.1f%% VALID=%d",
                 MIN_GEN, EARLY_STOP_PATIENCE,
                 HV_TOL*100, FRONT_SIZE_TOL*100, OBJ_MIN_TOL*100, VALIDATION_WINDOW)
        log.info("=" * 70)

        rng = np.random.default_rng(self.random_seed)

        if resume_ckpt and os.path.exists(resume_ckpt):
            start_gen, pop, objs, self.run_id, extras = self.load_checkpoint(resume_ckpt)
            pop = pop.astype(float)
            if "hv_ref" in extras:
                self.hv_ref = extras["hv_ref"]
            if "hv_history" in extras:
                self.hv_history         = list(extras["hv_history"])
                self.front_size_history = list(extras["front_size_history"])
                self.obj_min_history    = [list(r) for r in extras["obj_min_history"]]
            log.info("浠庢鏌ョ偣鎭㈠: gen=%d  run_id=%d  hv_history闀垮害=%d",
                     start_gen, self.run_id, len(self.hv_history))
        else:
            start_gen = 0
            pop = np.zeros((N_POP, N_DIM), dtype=float)
            for i in range(N_POP):
                pop[i] = self._random_feasible_layout(rng)
            log.info("Evaluating initial population: n=%d sim_years=%d", N_POP, self.sim_years)
            objs = self._eval_pop(pop, self.sim_years, tag="init",
                                  gen_subdir="gen_000")


        if self.hv_ref is None:
            valid = np.all(objs < PENALTY_VALUE / 2, axis=1)
            if valid.sum() > 0:
                vmax = objs[valid].max(axis=0)
                vmin = objs[valid].min(axis=0)
                span = np.where(vmax - vmin > 1e-9, vmax - vmin, 1.0)
                self.hv_ref = vmax + HV_REF_SLACK * span
            else:
                self.hv_ref = np.full(N_OBJ, PENALTY_VALUE)
            log.info("HV鍙傝€冪偣宸插浐瀹? %s",
                     np.array2string(self.hv_ref, precision=4))


        if not self.hv_history:
            hv0, fs0, om0 = self._compute_metrics(objs)
            self.hv_history.append(hv0)
            self.front_size_history.append(fs0)
            self.obj_min_history.append(om0.tolist())
            log.info("鍒濆 HV=%.4e  |鍓嶆部0|=%d", hv0, fs0)
            self._save_checkpoint(0, pop, objs)


        gen             = start_gen
        stop_pending_at = None
        stop_pending_hv = None

        while gen < N_GEN:
            t0        = time.time()
            gen_label = f"gen_{gen+1:03d}"

            if self.use_de:
                trials, F_vals, CR_vals = self._de_mutate(pop, objs, rng)
            else:
                trials = self._random_offspring(pop, rng)
                F_vals = np.full(len(pop), np.nan)
                CR_vals = np.full(len(pop), np.nan)
            repaired = []
            for t in trials:
                t = self._repair(t, rng)
                repaired.append(t)
            trials = np.array(repaired)
            trial_objs = self._eval_pop(trials, self.sim_years,
                                        tag=f"DE-g{gen+1}",
                                        gen_subdir=gen_label)

            combined_pop  = np.vstack([pop, trials])
            combined_objs = np.vstack([objs, trial_objs])
            pop, objs, sel_idx = self._nsga3_select(combined_pop, combined_objs, N_POP, rng)


            if self.use_de:
                trial_kept = [int(i) - N_POP for i in sel_idx if int(i) >= N_POP]
                S_F  = [F_vals[i]  for i in trial_kept]
                S_CR = [CR_vals[i] for i in trial_kept]
                self._update_de_params(S_F, S_CR)


            elapsed = time.time() - t0


            hv, fs, om = self._compute_metrics(objs)
            self.hv_history.append(hv)
            self.front_size_history.append(fs)
            self.obj_min_history.append(om.tolist())

            log.info("Gen %3d/%d  t=%.0fs  |front|=%d  HV=%.4e  "
                     "F1=%.4f  F2_plume_km2=%.6f  F3_pressure_cv=%.6f  "
                     "渭_F=%.3f 渭_CR=%.3f",
                     gen + 1, N_GEN, elapsed, fs, hv,
                     om[0], om[1], om[2],
                     self.mu_F, self.mu_CR)


            if stop_pending_at is None:
                if (gen + 1) >= MIN_GEN:
                    triggered, diag = self._check_early_stop()
                    if triggered:
                        stop_pending_at = gen
                        stop_pending_hv = hv
                        log.info("Early stop triggered at gen=%d -> %s", gen + 1, diag)
                        log.info("  entering validation window for %d generations",
                                 VALIDATION_WINDOW)
            else:
                gens_since = gen - stop_pending_at
                rebound    = stop_pending_hv * (1.0 + HV_TOL)
                if hv > rebound:
                    log.info("Validation window HV rebound (%.4e > %.4e); continue optimization",
                             hv, rebound)
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
        front0 = list(fronts[0])
        full_pop = pop[front0]
        full_objs = objs[front0]

        rep_idx = self._select_representatives(full_objs, NSGA3_FINAL_K)
        final_pop = full_pop[rep_idx]
        final_objs = full_objs[rep_idx]

        self._save_checkpoint(gen, pop, objs)
        np.savez(
            os.path.join(OPT_RUNS_DIR, "ckpt_final_population.npz"),
            gen=gen,
            pop=pop,
            objs=objs,
            full_pop=full_pop,
            full_objs=full_objs,
            representative_pop=final_pop,
            representative_objs=final_objs,
            surrogate_path=np.array("" if self.surrogate_path is None else self.surrogate_path),
        )
        self._report(full_pop, full_objs, filename="full_pareto_solutions.json")
        self._report(final_pop, final_objs, filename="pareto_solutions.json")
        return final_pop, final_objs

    def _report(self, pop, objs, filename="pareto_solutions.json"):
        results = []
        for k, (x, obj) in enumerate(zip(pop, objs)):
            inj, prod = self._decode(x)
            results.append(
                {
                    "rank": k + 1,
                    "F1_pressure_rise": float(obj[0]),
                    "pressure_rise_pct": float(obj[0] * 100),
                    "F2_plume_area_sg0p1_km2": float(obj[1]),
                    "F3_pressure_CV_std_over_mean": float(obj[2]),
                    "evaluation_backend": (
                        "surrogate" if self.surrogate_model is not None
                        else "high_fidelity_simulation"
                    ),
                    "surrogate_path": self.surrogate_path,
                    "inj_wells": [[int(i) + 1, int(j) + 1] for i, j in inj],
                    "prod_wells": [[int(i) + 1, int(j) + 1] for i, j in prod],
                }
            )
        out_path = os.path.join(OPT_RUNS_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        log.info("Pareto solutions saved to %s", out_path)

def _build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="CO2 well-placement hybrid NSGA-III optimizer"
    )
    parser.add_argument(
        "--seeds",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_seeds.npz"),
        help="path to opt_seeds.npz",
    )
    parser.add_argument("--resume", default=None, help="checkpoint path to resume from")
    parser.add_argument("--run-id", type=int, default=0, help="starting run-id offset")
    parser.add_argument(
        "--surrogate",
        default=SURROGATE_DEFAULT,
        help="joblib surrogate model path; use empty string to run high-fidelity simulations",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="delete intermediate CMG files after each evaluation",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=None,
        help=f"number of parallel CMG workers (default: {N_PARALLEL})",
    )
    parser.add_argument(
        "--pop",
        type=int,
        default=None,
        help=f"population size override (default: {N_POP})",
    )
    parser.add_argument(
        "--gen",
        type=int,
        default=None,
        help=f"generation cap override (default: {N_GEN})",
    )
    parser.add_argument(
        "--ref-p",
        type=int,
        default=None,
        help=f"NSGA-III Das-Dennis reference point divisions (default: {REF_P})",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=None,
        help=f"number of representative Pareto solutions in pareto_solutions.json (default: {NSGA3_FINAL_K})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional random seed for reproducible optimizer comparisons",
    )
    parser.add_argument("--use-de", dest="use_de", action="store_true", help="enable JADE DE mutation")
    parser.add_argument(
        "--no-de",
        dest="use_de",
        action="store_false",
        help="disable DE and use random repaired offspring",
    )
    parser.add_argument(
        "--use-opt-perm-align",
        dest="use_opt_perm_align",
        action="store_true",
        help="enable optimizer-side Hungarian alignment in DE",
    )
    parser.add_argument(
        "--no-opt-perm-align",
        dest="use_opt_perm_align",
        action="store_false",
        help="disable optimizer-side Hungarian alignment in DE",
    )
    parser.add_argument(
        "--use-early-stop",
        dest="use_early_stop",
        action="store_true",
        help="enable adaptive early stopping",
    )
    parser.add_argument(
        "--no-early-stop",
        dest="use_early_stop",
        action="store_false",
        help="disable adaptive early stopping",
    )
    parser.set_defaults(
        use_de=USE_DE,
        use_opt_perm_align=USE_OPT_PERM_ALIGN,
        use_early_stop=USE_EARLY_STOP,
    )
    return parser


def main():
    global N_PARALLEL, N_POP, N_GEN, REF_P, NSGA3_FINAL_K

    args = _build_arg_parser().parse_args()
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

    surrogate_path = args.surrogate if args.surrogate else None

    opt = HybridNSGAIII(
        seeds_path=args.seeds,
        run_id_start=args.run_id,
        cleanup_runs=args.cleanup,
        surrogate_path=surrogate_path,
        use_de=args.use_de,
        use_opt_perm_align=args.use_opt_perm_align,
        use_early_stop=args.use_early_stop,
        random_seed=args.seed,
    )
    opt.run(resume_ckpt=args.resume)


if __name__ == "__main__":
    main()
