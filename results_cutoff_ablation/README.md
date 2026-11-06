# hip_v2 vs hip_v3: graph-cutoff bug and cutoff ablation

**Question.** On the PubChem 30–100-atom set, why does the released `hip_v2.ckpt` generalise better
than `hip_v3.ckpt`?

**Answer so far.**
1. Both released checkpoints were trained with a **5 Å backbone graph** (`max_radius=5.0`,
   `max_neighbors=32`), not the 12 Å graph of the paper model `hesspred_v1`.
2. Until 2026-05-03, the training code also built the **Hessian graph** from the backbone cutoff and
   neighbour cap. The configured `cutoff_hessian=100` only set the range of the Hessian-edge distance
   expansion, so hip_v2 was effectively trained as **5 Å backbone / 5 Å Hessian**.
3. In a controlled ablation (identical settings except the cutoffs), **5 Å / 5 Å generalises to
   large molecules by far the best**. Its Hessian MAE is about 19× lower than 5 Å / fully connected
   and about 80× lower than 12 Å / 12 Å, although all three match on T1x validation.
4. The released checkpoints were also evaluated under different conditions: hip_v2 with
   `max_neighbors=1000`, hip_v3 with `max_neighbors=32`. With a 5 Å radius, a 32-neighbour cap
   truncates the graph of 26% of atoms (550/710 molecules) in the PubChem set, but never applies to
   T1x training molecules (< 32 atoms). A controlled re-evaluation is in
   [Released checkpoints](#released-checkpoints-same-evaluator).

## 1. Evidence for the bug

### Released checkpoints (`hyper_parameters` read from the Hugging Face files)

| checkpoint (HF commit) | uploaded | backbone `max_radius` | `max_neighbors` | `cutoff_hessian` | train set (HORM Kaggle) | epochs |
|---|---|---|---|---|---|---|
| `hesspred_v1.ckpt` ([3a0aef9d](https://huggingface.co/andreasburger/hip/tree/3a0aef9d/ckpt)) | 2025-09 | 12.0 (precomputed graph, `otf_graph=False`) | 20 | 100 | v5 `ts1x_hess_train_big-hesspred` + RGD1 | 262 |
| `hip_v2.ckpt` original ([3a0aef9d](https://huggingface.co/andreasburger/hip/commit/3a0aef9d)) | 2025-10-20 | **5.0** | 32 | 100 | v5 `ts1x_hess_train_big` | 1144 |
| `hip_v2.ckpt` re-upload ([d3abd854](https://huggingface.co/andreasburger/hip/commit/d3abd854)) | 2026-06-12 | 5.0 | **1000** | 100 | same; weights bit-identical | 1144 |
| `hip_v3.ckpt` ([49095693](https://huggingface.co/andreasburger/hip/commit/49095693)) | 2026-05-06 | **5.0** | 32 | 100 | v6 `ts1x_hess_train` (byte-identical to v5 `train_big`) | 1108 |

The graph settings of hip_v2 and hip_v3 are identical. The backbone RBF buffer
`distance_expansion.offset` in the weights spans 0–5 Å in both (0–12 Å in `hesspred_v1`). The buffer
is fixed at construction and restored on every resume, so both models used 5 Å from the first step.

The training data are also identical. HORM v6 `ts1x_hess_train.lmdb` has the same SHA-256 as v5
`ts1x_hess_train_big.lmdb` (`a404b9eb…`); v6 only renamed it and dropped the `.h5`/`.json`
side files. `ts1x-val.lmdb` is byte-identical too (`aab37592…`). The energy target is the LMDB `ae`
field in every code version, with no E0 or atom-reference transform.

The remaining saved differences are optimiser, schedule and length:
- v2: lr 3e-4, betas (0.8, 0.95), Muon lr 0.02, weight decay 1e-5 (Muon 0.01), cosine T_max 800
  with 40 warm-up epochs, 1144 epochs.
- v3: lr 4.4e-4, betas (0.965, 0.965), Muon lr 1e-3, weight decay 2e-5 (Muon 1e-3), T_max 300 with
  10 warm-up epochs, 1108 epochs.

### Code at the time hip_v2 was uploaded (commit [744423b](https://github.com/BurgerAndreas/hip/tree/744423bec207ed3092462eaf18c8e39116a31b20), 2025-10-20)

- The 5 Å / 32 override was added to `configs/train.yaml` on 2025-09-23 in
  [81bf018](https://github.com/BurgerAndreas/hip/commit/81bf018c3229a250f50f2b930bae6fd32ed44e08)
  ("preprocess datasets") and was still present when hip_v2 was uploaded
  ([train.yaml L20-22](https://github.com/BurgerAndreas/hip/blob/744423bec207ed3092462eaf18c8e39116a31b20/configs/train.yaml#L20-L22)).
  It overrides the model default of 12 Å / 20 neighbours in `configs/equiformer_v2.yaml`.
- Backbone graph: `generate_graph_nopbc` calls `radius_graph(pos, r=self.cutoff)`
  ([L662-683](https://github.com/BurgerAndreas/hip/blob/744423bec207ed3092462eaf18c8e39116a31b20/nets/equiformer_v2/equiformer_v2_oc20.py#L662-L683)).
- Hessian graph: with `otfgraph_in_model=True`, training calls `forward(otf_graph=True)`
  ([training_module.py L565-569](https://github.com/BurgerAndreas/hip/blob/744423bec207ed3092462eaf18c8e39116a31b20/hip/training_module.py#L565-L569)).
  The forward pass then calls `add_graph_batch(data, cutoff=self.cutoff, max_neighbors=self.max_neighbors)`
  ([L731-738](https://github.com/BurgerAndreas/hip/blob/744423bec207ed3092462eaf18c8e39116a31b20/nets/equiformer_v2/equiformer_v2_oc20.py#L731-L738)),
  which builds the Hessian graph with the backbone cutoff
  ([hessian_pred_utils.py L380-421](https://github.com/BurgerAndreas/hip/blob/744423bec207ed3092462eaf18c8e39116a31b20/nets/equiformer_v2/hessian_pred_utils.py#L380-L421)).
  `cutoff_hessian` is only used for the Hessian-edge RBF range.
- [f0c441f](https://github.com/BurgerAndreas/hip/commit/f0c441fb784182e10c5c8fa304376880524ea7a5)
  (2025-11-14) renamed `max_radius` to `cutoff` and kept 5 Å.
  [a8c44d9](https://github.com/BurgerAndreas/hip/commit/a8c44d99c96d352a6f66436805201e5baab6de13)
  (2026-05-03, three days before hip_v3 was uploaded) introduced the fully connected Hessian graph,
  `fully_connected_hessian=True`, as the default. Current evaluation code therefore runs both
  released models with a fully connected Hessian graph that neither was trained on.
- Current code caps the backbone at `max_num_neighbors=self.max_neighbors`
  ([equiformer_v2_oc20.py L690](https://github.com/BurgerAndreas/hip/blob/natcompsci-rebuttal/nets/equiformer_v2/equiformer_v2_oc20.py#L690)),
  so `hip_v3.ckpt`/`hip_v3.yaml` as shipped (32) are truncated on molecules with more than 32 atoms.

Neighbour counts on the PubChem set: at 5 Å, a mean of 27.4 and a maximum of 75 neighbours, with
25.9% of atoms above 32. At 7 Å, a mean of 43.4 and 75.5% above 32. At 12 Å, a mean of 64.8 and a
maximum of 99.

### Existing released-model results (separate runs on the NVIDIA cluster, raw PubChem metrics, eV/Å²)

From `results_eval_largehessians_orca_hip_v{2,3}` and `..._hf_horm_eqv2_autograd` on this branch.

| model | eval `max_neighbors` | Hessian MAE mean (median) | Hessian RMSE mean (median) | Eckart eigval MAE mean (median) |
|---|---|---|---|---|
| hip_v2 | 1000 | 0.388 (0.293) | 2.70 (0.99) | 4.79 (0.44) |
| hip_v3 | 32 | 0.350 (0.227) | 3.61 (2.30) | 6.14 (1.71) |
| HORM EqV2 (autograd) | – | 0.564 (0.448) | 4.36 (2.59) | 10.8 (7.12) |

## 2. Controlled cutoff ablation (killarney, L40S)

All variants share seed 20261102, the hip_v3 optimiser, HORM v5 `ts1x_hess_train_big`, batch size 64,
`limit_train_batches=0.1`, `max_neighbors=1000` (uncapped) and 100 epochs (269,500 steps). Training
code: `hip-neighbors-retrain-20261102` at
[40cc830](https://github.com/BurgerAndreas/hip/commit/40cc830ff243a655b55841c8cf47261243c0534a) (branch `neighbors`). Scripts are in `scripts/`
(`killarney_train_24h_l40s_bz64.sbatch`, `seeded_train.py`, `short100_train.py`,
`train_short100.sbatch`, `train_7A.sbatch`). Checkpoints:
<https://huggingface.co/andreasburger/hip/tree/main/ablation>. One run per configuration, so there is
no estimate of seed-to-seed variance.

### T1x validation (in distribution; `final-validation.json`)

| variant | MAE E (eV) | MAE F (eV/Å) | MAE Hessian (eV/Å²) | MAE eigvals Eckart | abs cos v1 | neg eigvals match |
|---|---|---|---|---|---|---|
| 5 Å / fully connected | 0.0307 | 0.0090 | 0.0196 | 0.0140 | 0.977 | 0.901 |
| 5 Å / 5 Å | 0.0327 | 0.0090 | 0.0202 | 0.0145 | 0.975 | 0.831 |
| 12 Å / 12 Å | 0.0280 | 0.0091 | 0.0201 | 0.0166 | 0.978 | 0.861 |

### PubChem 30–100 atoms, 710 molecules (out of distribution)

Mass-weighted, Eckart-projected metrics in eV/(Å²·amu). Each cell gives the mean with the median in
parentheses. "Clean" excludes the 51 molecules flagged by the rebuttal outlier rule (n = 659).

Definitions:
- Full-space projector P = QᵀQ, applied to M^-1/2 (H_pred − H_ref) M^-1/2.
- Block RMSE is the mean over all N² 3×3 atom-pair blocks of the elementwise RMSE.
- Eigenvalues and |cos v1| use the reduced Eckart basis Q.

| variant | subset | Hessian MAE | block RMSE | eigval MAE | abs cos v1 |
|---|---|---|---|---|---|
| 5 Å / fully connected | all | 0.0813 (0.0216) | 0.123 (0.0274) | 5.03 (0.396) | 0.245 (0.079) |
| 5 Å / fully connected | clean | 0.0306 (0.0213) | 0.0394 (0.0266) | 0.696 (0.398) | 0.246 (0.080) |
| 5 Å / 5 Å | all | 0.0524 (0.00156) | 0.0853 (0.00195) | 4.37 (0.0156) | 0.758 (0.946) |
| 5 Å / 5 Å | clean | **0.00163** (0.00155) | **0.00204** (0.00195) | **0.0233** (0.0159) | **0.763** (0.946) |
| 12 Å / 12 Å | all | 0.181 (0.159) | 0.257 (0.212) | 13.3 (10.2) | 0.174 (0.020) |
| 12 Å / 12 Å | clean | 0.131 (0.159) | 0.175 (0.212) | 9.25 (10.6) | 0.175 (0.020) |
| 7 Å / 7 Å | – | training (jobs 7320441→7320443), evaluation queued (7320698) | | | |

Raw (unprojected) results, the outlier analysis and plots are in `pubchem/`:
- `report.md` and `summary.json`
- `outliers.csv`, `outlier-reference-diagnostics.csv`
- `size-generalization*.png`
- projected metrics: `projected-analysis/`

Some flagged references have unphysical geometries: PubChem_5280489 has an H–H distance of 0.074 Å,
and PubChem_5281673 and PubChem_5281792 have contacts of 0.38 Å.

Validation of the evaluator: on six smoke-test molecules, the adapted evaluator agrees with the
unmodified `natcompsci-rebuttal` evaluator to within 5.6e-7. The full-space projector matches the
rebuttal projector numerically.

## Released checkpoints, same evaluator

Job 7320536 evaluates hip_v2 and hip_v3 (HF files above) with `max_neighbors` set to 32 and 1000,
using the same projected evaluator (`scripts/eval_released.sbatch`). Results will be added here.

## Interpretation and open points

- On T1x, the cutoff hardly matters. The large differences appear only under size extrapolation, and
  they favour a short, local Hessian graph. A local Hessian graph predicts zero for distant atom-pair
  blocks, which matches the near-zero far-field couplings in the reference. A fully connected or 12 Å
  Hessian graph must learn those zeros from molecules of fewer than 32 atoms, and its predictions
  degrade above about 40 atoms.
- So the 5 Å setting in hip_v2, Hessian graph included, was an accident, but it is what made hip_v2
  generalise.
- **Correction (Hessian graph history).** On `main` the Hessian graph went through three phases:
  - until 2026-01-02 it reused the backbone settings (5 Å, 32 neighbours);
  - [195218c](https://github.com/BurgerAndreas/hip/commit/195218c) ("hessian graph cutoff bug",
    2026-01-02) switched it to `cutoff_hessian=100` with a 1,000,000-neighbour cap, i.e. every atom
    pair in these molecules;
  - a8c44d9 (2026-05-03) replaced that with the explicit `fully_connected_hessian` flag.
- hip_v2 (uploaded 2025-10-20) therefore trained as **5 Å / 5 Å**. hip_v3's hyperparameters (HORM v6
  path, Lightning 2.6.1, both from 2026-03-28) date its last launch to between 2026-03-28 and
  2026-05-03. So it most likely trained as **5 Å / fully connected**, the configuration that is about
  19× worse on PubChem in the ablation. Earlier training segments may have used other code, because
  the hyperparameters record only the last launch.
- Optimiser: the saved optimiser state shows that hip_v2 was trained with **AdamW** (one parameter
  group, Adam moments; `lr_muon` unused), while hip_v3 used Muon plus auxiliary Adam. The eigenvalue
  loss was off for both (`alpha=None`). The ablations all used the hip_v3 optimiser, so the
  optimiser effect is untested.
- Pending: a behavioural probe (`scripts/probe_graph.py`, job 7321535) evaluates each checkpoint on
  T1x-val and PubChem with the backbone capped at 32 or 1000 neighbours, and the Hessian graph fully
  connected or at the backbone cutoff. The in-distribution optimum identifies the training graph.
