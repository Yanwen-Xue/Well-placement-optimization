"""Build the compact surrogate-search seed from the released training arrays."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "public_data" / "reproduction_333"


def main() -> None:
    with np.load(DATA / "training_responses_684.npz", allow_pickle=False) as responses:
        layouts = np.asarray(responses["X"], dtype=float)
    if layouts.ndim != 2 or layouts.shape[1] != 20 or not np.array_equal(layouts, np.rint(layouts)):
        raise ValueError("expected integer grid coordinates for ten wells")

    with np.load(DATA / "training_prescreen_fields.npz", allow_pickle=False) as fields:
        mask = np.asarray(fields["bound_mask"], dtype=bool)
        ni, nj = int(fields["NI"]), int(fields["NJ"])
        dx = float(fields["dx"])
    if mask.shape != (ni, nj):
        raise ValueError("feasible-cell mask does not match grid dimensions")

    coords = layouts.astype(np.int32)
    output = DATA / "search_seed_geometry.npz"
    np.savez_compressed(
        output,
        inj_wells=coords[:, :10].reshape(-1, 5, 2),
        prod_wells=coords[:, 10:].reshape(-1, 5, 2),
        bound_mask=mask,
        NI=np.int64(ni),
        NJ=np.int64(nj),
        dx=np.float64(dx),
    )
    print(output)


if __name__ == "__main__":
    main()
