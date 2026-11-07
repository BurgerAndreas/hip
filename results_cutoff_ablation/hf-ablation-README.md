# HIP graph-cutoff ablation (epoch 100)

Three EquiformerV2-HIP models that differ only in their graph cutoffs. They were trained to test why
`ckpt/hip_v2.ckpt` generalises to large molecules better than `ckpt/hip_v3.ckpt`. Full write-up, code
links and per-molecule results:
https://github.com/BurgerAndreas/hip/tree/natcompsci-rebuttal/results_cutoff_ablation

| folder | backbone graph | Hessian graph | sha256 (final-epoch-100.ckpt) |
|---|---|---|---|
| `backbone5_hessian_all/` | 5 Å | fully connected | `49ee65b178963bf7fa1ca7ebf0fa30d670d5152fceff58206769c32646311e25` |
| `backbone5_hessian5/` | 5 Å | 5 Å | `e065547d7bb3afa053bd9b6e21881566c97b7085aaca4a3697d9bc0807d46099` |
| `backbone12_hessian12/` | 12 Å | 12 Å | `4ba078642cf603d997b9cf5915c952d04e2828bf4922c194bf820cc617f26ffe` |

All three use `max_neighbors=1000` (no neighbour cap), seed 20261102, and the hip_v3 optimiser:
Muon, lr 4.4e-4, betas (0.965, 0.965), weight decay 2e-5, Muon lr 1e-3, Muon weight decay 1e-3.
They also share batch size 64, `limit_train_batches=0.1`, and HORM Kaggle v5 `ts1x_hess_train_big.lmdb`.
Training ran 100 epochs (269,500 steps) on one L40S each. Each folder holds a Lightning checkpoint and
`final-validation.json` (T1x validation, `limit_val_batches=8`).

One checkpoint per configuration, so there is no seed-to-seed variance estimate.

## T1x validation (in distribution)

| variant | MAE E (eV) | MAE F (eV/Å) | MAE Hessian (eV/Å²) | MAE eigvals Eckart | abs cos v1 | neg eigvals match |
|---|---|---|---|---|---|---|
| backbone5_hessian_all | 0.0307 | 0.0090 | 0.0196 | 0.0140 | 0.977 | 0.901 |
| backbone5_hessian5 | 0.0327 | 0.0090 | 0.0202 | 0.0145 | 0.975 | 0.831 |
| backbone12_hessian12 | 0.0280 | 0.0091 | 0.0201 | 0.0166 | 0.978 | 0.861 |

## PubChem 30–100 atoms (`orca_wb97x_631gd_chno_30_100`, 710 molecules, out of distribution)

These are mass-weighted, Eckart-projected metrics, in eV/(Å²·amu). Each cell gives the mean with the
median in parentheses, after excluding the 51 flagged reference outliers (n = 659).

| variant | Hessian MAE | block RMSE | eigval MAE | abs cos v1 |
|---|---|---|---|---|
| backbone5_hessian_all | 0.0306 (0.0213) | 0.0394 (0.0266) | 0.696 (0.398) | 0.246 (0.080) |
| backbone5_hessian5 | 0.00163 (0.00155) | 0.00204 (0.00195) | 0.0233 (0.0159) | 0.763 (0.946) |
| backbone12_hessian12 | 0.131 (0.159) | 0.175 (0.212) | 9.25 (10.6) | 0.175 (0.020) |

## Use a 5 Å Hessian graph at inference

The Hessian graph used at **inference** matters more for large molecules than the one used in
training. All 5 Å backbone models, the released `ckpt/hip_v2.ckpt` and `ckpt/hip_v3.ckpt` included,
should be evaluated with a 5 Å radius Hessian graph (`model.fully_connected_hessian = False`,
`model.hessian_graph_cutoff = 5.0`). Current code defaults to a fully connected Hessian graph for
checkpoints that do not record this setting.

Same PubChem set, projected metrics as above (n = 659):

| model | Hessian graph at inference | Hessian MAE | eigval MAE | abs cos v1 |
|---|---|---|---|---|
| hip_v2 | fully connected | 0.0593 (0.0625) | 0.429 (0.447) | 0.120 (0.072) |
| hip_v3 | fully connected | 0.0550 (0.0498) | 1.81 (1.71) | 0.206 (0.053) |
| hip_v2 | 5 Å | 0.00167 (0.00162) | 0.0306 (0.0228) | 0.698 (0.925) |
| hip_v3 | 5 Å | **0.00145** (0.00139) | **0.0130** (0.0101) | **0.877** (0.986) |

A sweep over inference cutoffs (4–32 Å) is in progress; see the GitHub write-up.
