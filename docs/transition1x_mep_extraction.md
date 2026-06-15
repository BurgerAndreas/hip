# Extracting Likely Final MEP Images From Transition1x

Transition1x stores three kinds of geometries for each reaction:

- `reactant`, `transition_state`, and `product` child groups, each with one final geometry.
- Top-level reaction arrays such as `positions`, `wB97x_6-31G(d).energy`, and `wB97x_6-31G(d).forces`.
- The top-level reaction arrays contain saved NEB/CINEB configurations from the reaction path optimization, not a single explicitly labeled final MEP trajectory.

The HDF5 file does not expose NEB iteration IDs or image IDs in the public loader. The final MEP must therefore be inferred from the frame order and convergence pattern.

## Frame Order Pattern

For the glycine proton-transfer example:

- split: `test`
- formula: `C2H5NO2`
- reaction: `rxn1961`
- top-level `positions` shape: `(266, 10, 3)`

The frame count factors as:

```text
266 = 10 + 32 * 8
```

This is consistent with:

- one initial 10-image path, including reactant and product endpoints
- followed by 32 saved NEB/CINEB path snapshots containing only the 8 internal images

Under this interpretation:

- frames `0..9` are the initial full path
- frames `10..17` are saved internal images for a later path snapshot
- frames `18..25` are the next saved internal-image path snapshot
- ...
- frames `258..265` are the last saved internal-image snapshot

The last internal-image block is the best candidate for the converged final MEP. Reconstruct the full path by adding the exact endpoint child-group geometries:

```text
[reactant] + positions[258:266] + [product]
```

For glycine, the exact endpoint geometries also appear in the top-level array:

- reactant matches frame `0`
- product matches frame `9`
- transition state matches frame `262`

## Glycine Candidate MEP

The likely final MEP indices for `test/C2H5NO2/rxn1961` are:

```text
[0, 258, 259, 260, 261, 262, 263, 264, 265, 9]
```

For the proton-transfer CVs using zero-based atom indices `N=4`, `O=3`, `H=9`:

```text
idx   role    qNH      qOH      xi       Erel_eV
0     R       1.0064   2.5185  -1.5121    0.000
258   img1    1.0050   2.4325  -1.4275    0.982
259   img2    1.0055   2.2299  -1.2244    5.702
260   img3    1.0268   1.9189  -0.8921   18.886
261   img4    1.1276   1.5964  -0.4688   45.448
262   TS      1.3393   1.2914   0.0479   65.167
263   img6    1.5888   1.0676   0.5212   51.058
264   img7    1.8314   0.9927   0.8387   37.145
265   img8    2.0637   0.9752   1.0884   29.591
9     P       2.2521   0.9697   1.2824   27.361
```

The last few saved internal-image blocks are nearly converged. For glycine, corresponding-image RMSDs to the last block are:

```text
block 242..249 vs 258..265: max 0.0024 A, mean 0.0012 A
block 250..257 vs 258..265: max 0.0009 A, mean 0.0005 A
```

That small change is the main evidence that `258..265` is the final saved MEP image set.

## General Extraction Recipe

Given a reaction group with `n_frames = len(group["positions"])`:

1. Count the number of images in the initial full path. For standard Transition1x NEB paths this is commonly `10`.
2. Compute the number of internal images as `n_internal = n_initial - 2`.
3. Check whether `(n_frames - n_initial) % n_internal == 0`.
4. If the divisibility check passes, take the final internal block:

```text
start = n_frames - n_internal
stop = n_frames
```

5. Reconstruct the likely final MEP as:

```text
reactant_child_geometry + positions[start:stop] + product_child_geometry
```

6. Validate the inference by checking that the last few internal-image blocks are nearly identical by corresponding-image RMSD.

## Minimal Python Snippet

```python
from pathlib import Path

import h5py
import numpy as np


def likely_final_mep(group, n_initial=10):
    positions = np.asarray(group["positions"])
    n_frames = positions.shape[0]
    n_internal = n_initial - 2

    if (n_frames - n_initial) % n_internal != 0:
        raise ValueError(
            f"Cannot infer fixed-size internal-image blocks from {n_frames=} "
            f"and {n_initial=}."
        )

    start = n_frames - n_internal
    internal = positions[start:n_frames]
    reactant = np.asarray(group["reactant"]["positions"])[0]
    product = np.asarray(group["product"]["positions"])[0]
    return np.concatenate([reactant[None], internal, product[None]], axis=0)


def corresponding_image_rmsd(block_a, block_b):
    return np.sqrt(np.mean((block_a - block_b) ** 2, axis=(1, 2)))


h5_path = Path("data/transition1x.h5")
with h5py.File(h5_path, "r") as handle:
    group = handle["test"]["C2H5NO2"]["rxn1961"]
    mep = likely_final_mep(group)

    positions = np.asarray(group["positions"])
    n_internal = 8
    final_block = positions[-n_internal:]
    prev_block = positions[-2 * n_internal : -n_internal]
    rmsd = corresponding_image_rmsd(prev_block, final_block)

print(mep.shape)
print(f"previous block vs final block: max RMSD = {rmsd.max():.4f} A")
```

## Caveats

- This is an inferred convention, not an explicit label in the HDF5 schema.
- The top-level reaction arrays include saved NEB/CINEB optimization configurations, not necessarily only the final MEP.
- Always verify the block pattern and convergence behavior for a new reaction before treating the extracted images as the final MEP.
- If a reaction uses a different number of NEB images, update `n_initial` accordingly.

With geodesic interpolation we created 73 frames.
- all-atom RMSD per frame: mean 0.0065 A, max 0.0127 A
- mean per-atom displacement: mean 0.0088 A, max 0.0191 A
- largest single-atom jump in any frame step: max 0.0457 A