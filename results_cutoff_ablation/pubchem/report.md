# PubChem size-generalization results

All three epoch-100 checkpoint continuations were evaluated against 710 ORCA ωB97X/6-31G(d) CHNO reference Hessians: 10 molecules for each atom count from 30 to 100. All 2,130 model/sample evaluations succeeded. The evaluation follows `natcompsci-rebuttal` commit `666463211257c5ae3d345fae6882ae22f718b505` and reference dataset revision `f869cabd92ebcc66223891d0b3e815766c9bba0e`.

The model is the exact retraining code used for the checkpoints; the reference evaluator and frequency analysis follow the rebuttal branch. Checkpoint weights and graph cutoffs were checked, and the unmodified evaluator agreed on six smoke-test evaluations to within 5.6e-7 in the reported error metrics.

## Overall metrics

| Backbone / Hessian | Hessian MAE mean, all | Hessian MAE median, all | Hessian MAE mean, outliers removed | Eckart eigenvalue MAE mean, all | Negative-count agreement, all |
|---|---:|---:|---:|---:|---:|
| 5 Å / all-to-all | 0.20074 | 0.06009 | 0.08629 | 5.03299 | 6.20% |
| 5 Å / 5 Å | 0.11880 | 0.00520 | 0.00547 | 4.36733 | 4.65% |
| 12 Å / 12 Å | 0.55940 | 0.52365 | 0.44319 | 13.28387 | 5.63% |

Hessian MAE is in eV/Å². The eigenvalue column uses the branch’s mass-weighted Eckart-projected eigenvalue metric (`eckart_eigval_mae_ev_a2`). Negative-count agreement means that prediction and reference have the same number of negative projected eigenvalues; it is not a measure of energy or force accuracy.

## Outlier rule and limitations

The rebuttal rule flags a model/sample when the absolute modified z-score of its Hessian MAE exceeds 10 relative to the median and median absolute deviation for that atom count. Each size group has only 10 molecules. The 51 unique samples flagged by any of the three models are removed from every model for the paired outlier-excluded comparison (659 remain). There are 34 flags for 5 Å/all, 14 for 5 Å/5 Å, and 38 for 12 Å/12 Å; a molecule can be flagged by multiple models. Flagging alone does not establish that a reference is wrong. Because the rule uses absolute scores, two 12 Å/12 Å flags have unusually low error relative to their size peers, rather than unusually high error.

All 51 flagged reference Hessians contain finite values. Eight flagged structures have a closest atom separation below 0.8 Å. PubChem_5280489 has a 0.074 Å H–H separation and reference Hessian maximum magnitude about 46,771 eV/Å², warranting inspection of its geometry and reference calculation. Many other flags are model-specific and do not have comparably short separations.

## Complete outlier list

`Max |z|` is the largest absolute modified z-score among the models that flagged the sample. Hessian errors are eV/Å²; minimum separation is Å.

| Sample | Atoms | Flagged models | Max abs(z) | Minimum separation | MAE 5/all | MAE 5/5 | MAE 12/12 |
|---|---:|---|---:|---:|---:|---:|---:|
| PubChem_5281673 | 53 | 12A_12A; 5A_5A; 5A_all | 33706.4 | 0.380 | 16.8255 | 16.8398 | 16.8369 |
| PubChem_5280489 | 96 | 12A_12A; 5A_5A; 5A_all | 28551.4 | 0.074 | 21.8578 | 21.4045 | 21.7369 |
| PubChem_5281794 | 44 | 12A_12A; 5A_5A; 5A_all | 15328.4 | 0.620 | 6.9953 | 6.9308 | 7.2286 |
| PubChem_5281675 | 52 | 12A_12A; 5A_5A; 5A_all | 11245.8 | 0.587 | 6.7142 | 6.7366 | 6.7251 |
| PubChem_5280349 | 53 | 5A_5A; 5A_all | 10508.2 | 0.620 | 5.4203 | 5.2532 | 5.6869 |
| PubChem_5280805 | 73 | 12A_12A; 5A_5A; 5A_all | 7305.0 | 0.587 | 7.2104 | 7.1701 | 7.3594 |
| PubChem_5280344 | 51 | 12A_12A; 5A_5A; 5A_all | 5990.5 | 0.620 | 6.5815 | 6.5617 | 6.6951 |
| PubChem_5281792 | 42 | 12A_12A; 5A_5A; 5A_all | 5641.3 | 0.380 | 9.6297 | 9.5684 | 10.0292 |
| chno_xyz__n033_pubchem_02_4e9d6319a4 | 33 | 12A_12A; 5A_all | 1321.8 | 1.080 | 0.1145 | 0.0059 | 0.9504 |
| chno_xyz__n035_pubchem_07_8b3afd7d12 | 35 | 12A_12A; 5A_all | 1133.1 | 1.016 | 0.0775 | 0.0073 | 0.9594 |
| chno_xyz__n035_pubchem_05_46d2d4caeb | 35 | 12A_12A; 5A_all | 955.4 | 1.094 | 0.0683 | 0.0068 | 0.8103 |
| chno_xyz__n036_pubchem_01_9372688d2e | 36 | 12A_12A; 5A_all | 842.2 | 1.080 | 0.1292 | 0.0055 | 0.8482 |
| chno_xyz__n036_pubchem_09_be2e4d5977 | 36 | 12A_12A; 5A_all | 806.9 | 1.012 | 0.1323 | 0.0055 | 0.8131 |
| chno_xyz__n036_pubchem_04_8a06f2e81b | 36 | 12A_12A; 5A_all | 660.3 | 1.012 | 0.1285 | 0.0056 | 0.6671 |
| chno_xyz__n036_pubchem_03_9753251700 | 36 | 12A_12A; 5A_all | 625.1 | 1.012 | 0.1306 | 0.0052 | 0.6320 |
| chno_xyz__n030_pubchem_04_2c581595d2 | 30 | 12A_12A; 5A_all | 443.3 | 1.080 | 0.0809 | 0.0066 | 0.4525 |
| chno_xyz__n039_pubchem_02_7a8f6a9c70 | 39 | 12A_12A; 5A_all | 334.1 | 1.081 | 0.1165 | 0.0058 | 0.9423 |
| chno_xyz__n039_pubchem_01_0a1f9a6ff4 | 39 | 12A_12A; 5A_all | 314.0 | 1.080 | 0.1439 | 0.0049 | 0.8864 |
| chno_xyz__n034_pubchem_05_97e54e66fa | 34 | 12A_12A; 5A_all | 278.9 | 1.084 | 0.0770 | 0.0065 | 0.5670 |
| chno_xyz__n031_pubchem_06_e15c77b082 | 31 | 12A_12A; 5A_all | 268.5 | 1.094 | 0.1158 | 0.0064 | 0.4142 |
| chno_xyz__n037_pubchem_09_bf1544d8ab | 37 | 12A_12A; 5A_all | 263.7 | 1.084 | 0.1027 | 0.0059 | 0.6684 |
| chno_xyz__n034_pubchem_03_6fbd989337 | 34 | 12A_12A; 5A_all | 199.7 | 1.085 | 0.0657 | 0.0077 | 0.4087 |
| chno_xyz__n032_pubchem_06_8b2b26c9ac | 32 | 12A_12A; 5A_all | 183.3 | 1.094 | 0.0630 | 0.0067 | 0.5637 |
| chno_xyz__n043_pubchem_09_be1552cadb | 43 | 12A_12A | 146.1 | 1.089 | 0.0543 | 0.0058 | 0.7299 |
| chno_xyz__n043_pubchem_06_854e1985bc | 43 | 12A_12A; 5A_all | 135.0 | 1.084 | 0.1554 | 0.0049 | 0.6754 |
| chno_xyz__n043_pubchem_05_c52799fa6c | 43 | 12A_12A | 103.6 | 1.092 | 0.0447 | 0.0068 | 0.5212 |
| chno_xyz__n046_pubchem_02_ff8f0e37ae | 46 | 12A_12A | 66.3 | 1.087 | 0.1100 | 0.0049 | 0.8668 |
| chno_xyz__n046_pubchem_03_ace899fa3a | 46 | 12A_12A; 5A_all | 63.8 | 1.086 | 0.1744 | 0.0046 | 0.8348 |
| chno_xyz__n046_pubchem_07_234fc5db1d | 46 | 12A_12A | 56.4 | 1.084 | 0.0963 | 0.0056 | 0.7396 |
| chno_xyz__n031_pubchem_07_35deea2bdf | 31 | 12A_12A; 5A_all | 50.4 | 1.085 | 0.0339 | 0.0085 | 0.0854 |
| chno_xyz__n046_pubchem_04_f3878c6165 | 46 | 12A_12A | 48.2 | 1.089 | 0.0493 | 0.0059 | 0.6352 |
| chno_xyz__n031_pubchem_08_8ad1d3a56a | 31 | 12A_12A; 5A_all | 48.2 | 1.084 | 0.0400 | 0.0069 | 0.0821 |
| chno_xyz__n035_pubchem_10_da4c68144e | 35 | 12A_12A; 5A_all | 40.5 | 1.093 | 0.0263 | 0.0066 | 0.0429 |
| chno_xyz__n038_pubchem_09_46d090e834 | 38 | 12A_12A | 21.7 | 0.972 | 0.0182 | 0.0071 | 0.0630 |
| chno_xyz__n100_pubchem_01_2067bbedd4 | 100 | 5A_5A | 17.7 | 1.019 | 0.0428 | 0.0143 | 0.4557 |
| chno_xyz__n058_pubchem_08_cf06f97a88 | 58 | 5A_all | 16.0 | 1.080 | 0.2683 | 0.0034 | 0.7220 |
| chno_xyz__n058_pubchem_03_b4a120b4f9 | 58 | 5A_all | 16.0 | 1.080 | 0.2683 | 0.0034 | 0.7220 |
| chno_xyz__n058_pubchem_09_644b4dc9f4 | 58 | 5A_all | 16.0 | 1.080 | 0.2680 | 0.0036 | 0.6970 |
| chno_xyz__n086_pubchem_07_da3121ab0c | 86 | 12A_12A | 14.5 | 1.016 | 0.0254 | 0.0051 | 0.2829 |
| chno_xyz__n061_pubchem_10_c54bccf0d9 | 61 | 12A_12A | 14.2 | 0.973 | 0.0159 | 0.0074 | 0.0890 |
| chno_xyz__n037_pubchem_05_de8c8b427f | 37 | 12A_12A | 13.9 | 1.090 | 0.0264 | 0.0070 | 0.0445 |
| chno_xyz__n097_pubchem_04_b350a4372e | 97 | 5A_5A | 13.7 | 1.019 | 0.0334 | 0.0146 | 0.3270 |
| chno_xyz__n041_pubchem_08_5eac95a579 | 41 | 5A_5A | 11.9 | 0.972 | 0.0135 | 0.0146 | 0.0161 |
| chno_xyz__n094_pubchem_03_d0da68bf22 | 94 | 5A_5A | 11.9 | 0.972 | 0.0318 | 0.0067 | 0.3680 |
| chno_xyz__n094_pubchem_04_8f1a85008a | 94 | 5A_all | 11.8 | 1.019 | 0.3262 | 0.0032 | 0.4428 |
| chno_xyz__n091_pubchem_09_e77d16fae6 | 91 | 5A_5A | 11.6 | 1.073 | 0.0229 | 0.0088 | 0.3349 |
| chno_xyz__n076_pubchem_03_8cff5d42ef | 76 | 5A_5A | 11.1 | 1.035 | 0.0245 | 0.0107 | 0.3019 |
| chno_xyz__n075_pubchem_04_5ffe965e89 | 75 | 5A_all | 10.8 | 1.092 | 0.2438 | 0.0040 | 0.6403 |
| chno_xyz__n045_pubchem_08_c6e5c36f1c | 45 | 12A_12A | 10.7 | 1.012 | 0.1696 | 0.0044 | 0.9501 |
| chno_xyz__n083_pubchem_03_1b8fa1d8e7 | 83 | 5A_all | 10.4 | 1.089 | 0.2803 | 0.0040 | 0.5422 |
| chno_xyz__n033_pubchem_08_7bdc4e5df3 | 33 | 12A_12A | 10.3 | 1.020 | 0.0124 | 0.0074 | 0.0163 |

Complete per-model per-molecule metrics are in `raw/0/metrics.csv`, `raw/1/metrics.csv`, and `raw/2/metrics.csv`. `outlier_samples.csv` has one row per flagged molecule; `outliers.csv` has one row per model-specific flag; `per-size.csv` contains all-sample and shared-outlier-excluded means. The checkpoint and reference SHA-256 manifests are included alongside these files.

## Mass-weighted, Eckart-projected metrics

Definition: full-space projector P = QᵀQ (Q = rebuttal reduced Eckart basis), applied to M^-1/2 (H_pred − H_ref) M^-1/2. Block RMSE = mean over all N² atom-pair 3×3 blocks of each block's elementwise RMSE. Eigenvalue MAE and |cos v1| use the shared reduced basis Q. Units: eV/(Å²·amu); cosine dimensionless. 710 molecules per model, all succeeded.

Mean (median) values. "Clean" removes the same 51 molecules flagged by the original rule (n = 659).

| Model | Subset | MAE | Block RMSE | Eigval MAE | abs cos v1 |
|---|---|---|---|---|---|
| 5 Å / all | all | 0.0813 (0.0216) | 0.1227 (0.0274) | 5.03 (0.396) | 0.245 (0.079) |
| 5 Å / all | clean | 0.0306 (0.0213) | 0.0394 (0.0266) | 0.696 (0.398) | 0.246 (0.080) |
| 5 Å / 5 Å | all | 0.0524 (0.00156) | 0.0853 (0.00195) | 4.37 (0.0156) | 0.758 (0.946) |
| 5 Å / 5 Å | clean | 0.00163 (0.00155) | 0.00204 (0.00195) | 0.0233 (0.0159) | 0.763 (0.946) |
| 12 Å / 12 Å | all | 0.181 (0.159) | 0.257 (0.212) | 13.3 (10.2) | 0.174 (0.020) |
| 12 Å / 12 Å | clean | 0.131 (0.159) | 0.175 (0.212) | 9.25 (10.6) | 0.175 (0.020) |

Applying the same outlier rule to the projected MAE flags 45 molecules (43 shared with the original 51); excluding those instead changes no mean by more than ~1%. Files: `projected-analysis/{summary.json,per-molecule.csv,per-size.csv,projected-outliers.csv,projected-size-generalization.png}`.
