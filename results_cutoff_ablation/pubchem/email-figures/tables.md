# Tables for email reply: HIP on PubChem 30–100 atoms

Units. `projected_hessian_mae_ev_a2_amu` and `eckart_eigval_mae_ev_a2` (despite its column name) are both computed on the **mass-weighted** Hessian, so both are in eV/(Å² amu). Definitions in `scripts/eval_projected_pubchem.py`: M^-1/2 H M^-1/2 is projected onto the reduced Eckart (vibrational) basis Q (translations/rotations removed); the eigenvalue MAE is mean_i |λ_i^pred − λ_i^ref| over all 3N−6 Eckart eigenvalues; projected Hessian MAE is the elementwise MAE of QᵀQ (H_pred − H_ref)_mw QᵀQ over all (3N)² entries; |cos v1| is the overlap of the lowest Eckart eigenvectors. Clean set: 710 minus 51 reference outliers (n = 659). Non-flat set: 710 minus the 8 z = 0 files (n = 702). All models use the backbone graph they were trained with (5 Å, or 12 Å for abl 12/12) with max_neighbors 1000; only the Hessian graph cutoff at inference varies.

## A. Inference Hessian-graph cutoff sweep (released checkpoints)

### hip_v2 (trained 5 / 5 Å)

| Hessian cutoff | median proj. Hessian MAE (clean) | mean proj. Hessian MAE (clean) | median eigval MAE (clean) | median abs cos v1 (clean) | mean eigval MAE (non-flat, n = 702) |
|---|---|---|---|---|---|
| 4 Å | 0.00225 | 0.00242 | 0.0217 | 0.897 | 0.0274 |
| 5 Å | 0.00162 | 0.00167 | 0.0228 | 0.925 | 0.0308 |
| 6 Å | 0.00215 | 0.00213 | 0.0278 | 0.762 | 0.0377 |
| 7 Å | 0.0204 | 0.0207 | 0.14 | 0.097 | 0.142 |
| 9 Å | 0.0448 | 0.045 | 0.32 | 0.052 | 0.312 |
| 12 Å | 0.0585 | 0.0557 | 0.412 | 0.058 | 0.393 |
| 16 Å | 0.0619 | 0.0587 | 0.447 | 0.069 | 0.417 |
| 32 Å | 0.0625 | 0.0593 | 0.447 | 0.072 | 0.425 |
| fully connected | 0.0625 | 0.0593 | 0.447 | 0.072 | 0.425 |

### hip_v3 (trained 5 Å / full)

| Hessian cutoff | median proj. Hessian MAE (clean) | mean proj. Hessian MAE (clean) | median eigval MAE (clean) | median abs cos v1 (clean) | mean eigval MAE (non-flat, n = 702) |
|---|---|---|---|---|---|
| 4 Å | 0.00207 | 0.00225 | 0.0162 | 0.960 | 0.0181 |
| 5 Å | 0.00139 | 0.00145 | 0.0101 | 0.986 | 0.0132 |
| 6 Å | 0.00127 | 0.00132 | 0.0107 | 0.986 | 0.0143 |
| 7 Å | 0.00125 | 0.00129 | 0.0119 | 0.986 | 0.0156 |
| 9 Å | 0.00147 | 0.00147 | 0.0232 | 0.954 | 0.0267 |
| 12 Å | 0.0462 | 0.0407 | 1.64 | 0.039 | 1.45 |
| 16 Å | 0.0498 | 0.0511 | 1.71 | 0.050 | 1.71 |
| 32 Å | 0.0498 | 0.055 | 1.71 | 0.053 | 1.78 |
| fully connected | 0.0498 | 0.055 | 1.71 | 0.053 | 1.78 |

For comparison, the external researcher reports mean Eckart eigenvalue MAE on 702 molecules of 1.8234 (default 100 Å cutoff) and 0.0131 (7.5 Å + 32 sorted neighbours). Units/normalisation of his metric should be checked against ours before comparing directly.

## B. Retraining ablation (backbone / Hessian graph, Å), each evaluated with its training graph, clean set (n = 659)

| backbone / Hessian | mean proj. Hessian MAE | median proj. Hessian MAE | mean eigval MAE | median eigval MAE | mean abs cos v1 | median abs cos v1 |
|---|---|---|---|---|---|---|
| 5 / 5 | 0.00162 | 0.00155 | 0.0233 | 0.0159 | 0.763 | 0.946 |
| 5 / 12 | 0.0159 | 0.0181 | 0.456 | 0.472 | 0.258 | 0.088 |
| 12 / 5 | 0.078 | 0.0884 | 3.79 | 3.9 | 0.192 | 0.015 |
| 12 / 12 | 0.131 | 0.159 | 9.25 | 10.6 | 0.175 | 0.020 |
| 5 / 5 + envelope | 0.00154 | 0.00146 | 0.0145 | 0.0138 | 0.850 | 0.975 |
| 12 / 12 + envelope | 0.00439 | 0.00349 | 0.28 | 0.165 | 0.436 | 0.285 |
| 5 / 5, Hessian RBF 0–5 Å | 0.00159 | 0.00152 | 0.0231 | 0.0179 | 0.815 | 0.975 |
| 5 / full | 0.0306 | 0.0213 | 0.696 | 0.398 | 0.246 | 0.080 |

All variants: 100 epochs with limit_train_batches=0.1 on T1x (269,500 steps), otherwise identical settings (see README). "envelope" = smooth cutoff envelope on Hessian edges; "Hessian RBF 0–5 Å" = Hessian-edge distance expansion restricted to 0–5 Å.

## C. T1x training set: pair-distance statistics (all training geometries)

| quantity | value |
|---|---|
| number of geometries | 1,725,362 |
| atoms per molecule (max / mean) | 23 / 13.92 |
| largest pair distance in the set | 12.210 Å |
| per-molecule max pair distance, percentiles 50 / 90 / 99 / 99.9 | 5.456 / 6.555 / 7.635 / 8.961 Å |
| fraction of molecules with any pair > 5 Å | 0.78040 (78 %) |
| fraction of molecules with any pair > 7.5 Å | 0.01349 (1.35 %) |
| fraction of molecules with any pair > 10 Å | 0.00008 (0.008 %) |
| fraction of molecules with any pair > 12 Å | 0.00001 (0.001 %) |
| fraction of molecules with any pair > 14.36 Å | 0.00000 (0 %) |

So a fully connected (or 100 Å) Hessian graph at inference produces edges far beyond any distance seen in training (> 12.21 Å never occurs; > 7.5 Å occurs in 1.3 % of molecules).

## D. The 8 flat (z = 0) PubChem files

| sample_name | natoms | in 51-outlier list |
|---|---|---|
| PubChem_5280344 | 51 | yes |
| PubChem_5280349 | 53 | yes |
| PubChem_5280489 | 96 | yes |
| PubChem_5280805 | 73 | yes |
| PubChem_5281673 | 53 | yes |
| PubChem_5281675 | 52 | yes |
| PubChem_5281792 | 42 | yes |
| PubChem_5281794 | 44 | yes |
