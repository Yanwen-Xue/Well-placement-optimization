

from __future__ import annotations
import argparse
import time
import numpy as np
from scipy.stats.qmc import LatinHypercube
from objectives import check_spacing
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


D_BOUND_M       = 1000.0
D_INJ_INJ_M     = 200.0
D_PROD_PROD_M   = 200.0
D_INJ_PROD_M    = 300.0
N_INJ           = 5
N_PROD          = 5
N_WELLS_TOTAL   = N_INJ + N_PROD
N_DIM           = N_WELLS_TOTAL * 2

SEEDS_SRC_NPZ   = os.path.join(ROOT_DIR, "opt_seeds.npz")
SEEDS_OUT_NPZ   = os.path.join(ROOT_DIR, "opt_seeds_lhs.npz")


def build_bound_mask(NI: int, NJ: int, d_bound_grids: int) -> np.ndarray:
    """Use the same boundary-distance rule as the prescreening map builder."""
    II, JJ = np.meshgrid(np.arange(NI), np.arange(NJ), indexing="ij")
    d = np.minimum(np.minimum(II, NI - 1 - II), np.minimum(JJ, NJ - 1 - JJ))
    return d >= d_bound_grids


def sample_constraint_aware_lhs(n_target: int, seed: int, oversample: int,
                                NI: int, NJ: int, dx: float,
                                bound_mask: np.ndarray,
                                max_candidates: int = 5_000_000):

    inj_kept, prod_kept = [], []
    n_cand_total = 0
    n_rej_b = n_rej_s = 0
    batch_size = max(n_target * oversample, 1000)
    rng = np.random.default_rng(seed)

    while len(inj_kept) < n_target:
        if n_cand_total >= max_candidates:
            raise RuntimeError(
                f"hit max_candidates={max_candidates:,} with only "
                f"{len(inj_kept)}/{n_target} samples kept. "
                f"Check constraints (rejection rate too high)."
            )


        batch_seed = int(rng.integers(0, 2**31 - 1))
        sampler = LatinHypercube(d=N_DIM, seed=batch_seed)
        raw = sampler.random(n=batch_size)
        n_cand_total += batch_size

        for row in raw:
            wells = []
            ok_boundary = True
            for k in range(N_WELLS_TOTAL):
                i = int(row[2 * k]     * NI)
                j = int(row[2 * k + 1] * NJ)
                i = max(0, min(NI - 1, i))
                j = max(0, min(NJ - 1, j))
                if not bound_mask[i, j]:
                    ok_boundary = False
                    break
                wells.append((i, j))
            if not ok_boundary:
                n_rej_b += 1
                continue

            inj  = wells[:N_INJ]
            prod = wells[N_INJ:]
            if not check_spacing(inj, prod,
                                 D_INJ_INJ_M, D_PROD_PROD_M, D_INJ_PROD_M,
                                 cell_size_m=dx):
                n_rej_s += 1
                continue

            inj_kept.append(inj)
            prod_kept.append(prod)
            if len(inj_kept) >= n_target:
                break

    stats = dict(
        n_candidate            = n_cand_total,
        n_kept                 = len(inj_kept),
        n_rejected_boundary    = n_rej_b,
        n_rejected_spacing     = n_rej_s,
    )
    inj_arr  = np.array(inj_kept,  dtype=np.int32)
    prod_arr = np.array(prod_kept, dtype=np.int32)
    return inj_arr, prod_arr, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="target number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample", type=int, default=30,
                        help="LHS oversampling factor; increase it if the rejection rate is high")
    parser.add_argument("--src", default=SEEDS_SRC_NPZ,
                        help="source NPZ file containing grid metadata")
    parser.add_argument("--out", default=SEEDS_OUT_NPZ)
    args = parser.parse_args()

    print("=" * 70)
    print(f"Constraint-aware LHS sampler  (target={args.n}, seed={args.seed})")
    print("=" * 70)


    print(f"\n[1] Loading grid metadata: {args.src}")
    src = np.load(args.src)
    NI = int(src["NI"]);  NJ = int(src["NJ"]);  NK = int(src["NK"])
    dx = float(src["dx"]); dy = float(src["dy"])
    print(f"    NI x NJ x NK = {NI} x {NJ} x {NK}   dx={dx:.1f} m   dy={dy:.1f} m")


    d_bound_grids = max(1, int(np.ceil(D_BOUND_M / dx)))
    print(f"\n[2] Recomputing boundary mask: D_BOUND_M={D_BOUND_M} m -> {d_bound_grids} grid cells")
    bound_mask = build_bound_mask(NI, NJ, d_bound_grids)
    print(f"    Feasible cells: {bound_mask.sum():,} / {NI*NJ:,}  ({100*bound_mask.sum()/(NI*NJ):.1f}%)")


    print(f"\n[3] LHS sampling: n_target={args.n}, oversample={args.oversample}x")
    t0 = time.time()
    inj_arr, prod_arr, stats = sample_constraint_aware_lhs(
        n_target    = args.n,
        seed        = args.seed,
        oversample  = args.oversample,
        NI=NI, NJ=NJ, dx=dx, bound_mask=bound_mask,
    )
    dt = time.time() - t0
    keep_rate = stats["n_kept"] / stats["n_candidate"]
    print(f"    Candidates {stats['n_candidate']:,} -> kept {stats['n_kept']:,}"
          f"  (rate {keep_rate*100:.2f}%, {dt:.1f}s)")
    print(f"    Rejected by boundary: {stats['n_rejected_boundary']:,}")
    print(f"    Rejected by spacing : {stats['n_rejected_spacing']:,}")


    all_wells = np.concatenate([inj_arr.reshape(-1, 2),
                                prod_arr.reshape(-1, 2)], axis=0)
    print(f"\n[4] Spatial coverage statistics:")
    print(f"    All well i range [{all_wells[:,0].min()}, {all_wells[:,0].max()}]"
          f" / [0, {NI-1}]")
    print(f"    All well j range [{all_wells[:,1].min()}, {all_wells[:,1].max()}]"
          f" / [0, {NJ-1}]")

    bin_i = np.linspace(0, NI, 9).astype(int)
    bin_j = np.linspace(0, NJ, 9).astype(int)
    hist, _, _ = np.histogram2d(all_wells[:,0], all_wells[:,1], bins=[bin_i, bin_j])
    print(f"    8 x 8 bin histogram: min={int(hist.min())}, max={int(hist.max())}, "
          f"mean={hist.mean():.1f}  (more uniform layouts are closer to the mean)")


    print(f"\n[5] Saving: {args.out}")

    np.savez_compressed(
        args.out,
        inj_wells    = inj_arr,
        prod_wells   = prod_arr,
        J_inj_2d     = src["J_inj_2d"],
        J_prod_2d    = src["J_prod_2d"],
        bound_mask   = bound_mask,
        grid_top_2d  = src["grid_top_2d"],
        grid_bot_2d  = src["grid_bot_2d"],
        NI=np.array(NI), NJ=np.array(NJ), NK=np.array(NK),
        INJ_K_START=src["INJ_K_START"],
        INJ_K_END  =src["INJ_K_END"],
        PRD_K_START=src["PRD_K_START"],
        PRD_K_END  =src["PRD_K_END"],
        dx=np.array(dx), dy=np.array(dy),

        _source     = np.array("constraint_aware_lhs", dtype=object),
        _seed       = np.array(args.seed),
        _n_samples  = np.array(args.n),
    )
    print(f"    Done. {args.n} samples saved.")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
