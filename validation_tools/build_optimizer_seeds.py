"""Combine published layout coordinates with locally generated prescreen maps."""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--maps", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.responses, allow_pickle=False) as responses, np.load(args.maps, allow_pickle=False) as maps:
        x = np.asarray(responses["X"], dtype=np.int32)
        if x.ndim != 2 or x.shape[1] != 20:
            raise ValueError(f"expected 20 layout coordinates, got {x.shape}")
        keys = ("J_inj_2d", "J_prod_2d", "bound_mask", "grid_top_2d", "grid_bot_2d", "NI", "NJ", "dx")
        missing = [key for key in keys if key not in maps.files]
        if missing:
            raise KeyError(f"missing prescreen inputs: {missing}")
        payload = {key: maps[key] for key in keys}
    payload["inj_wells"] = x[:, :10].reshape(-1, 5, 2)
    payload["prod_wells"] = x[:, 10:].reshape(-1, 5, 2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **payload)
    print(f"Saved {len(x)} layout seeds to {args.out}")


if __name__ == "__main__":
    main()
