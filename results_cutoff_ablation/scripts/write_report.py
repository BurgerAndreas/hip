"""Generate a human-readable report from the saved evaluation artifacts."""
from pathlib import Path
import csv,json,collections
p=Path(__file__).resolve().parent/'results-20261106'
d=json.loads((p/'summary.json').read_text())
with (p/'outlier_samples.csv').open(newline='') as f: outliers=list(csv.DictReader(f))
with (p/'outliers.csv').open(newline='') as f: flags=list(csv.DictReader(f))
counts=collections.Counter(x['model'] for x in flags)
labels={'5A_all':'5 Å / all-to-all','5A_5A':'5 Å / 5 Å','12A_12A':'12 Å / 12 Å'}
lines=['# PubChem size-generalization results','',
'All three epoch-100 checkpoint continuations were evaluated against 710 ORCA ωB97X/6-31G(d) CHNO reference Hessians: 10 molecules for each atom count from 30 to 100. All 2,130 model/sample evaluations succeeded. The evaluation follows `natcompsci-rebuttal` commit `666463211257c5ae3d345fae6882ae22f718b505` and reference dataset revision `f869cabd92ebcc66223891d0b3e815766c9bba0e`.','',
'The model is the exact retraining code used for the checkpoints; the reference evaluator and frequency analysis follow the rebuttal branch. Checkpoint weights and graph cutoffs were checked, and the unmodified evaluator agreed on six smoke-test evaluations to within 5.6e-7 in the reported error metrics.','',
'## Overall metrics','',
'| Backbone / Hessian | Hessian MAE mean, all | Hessian MAE median, all | Hessian MAE mean, outliers removed | Eckart eigenvalue MAE mean, all | Negative-count agreement, all |',
'|---|---:|---:|---:|---:|---:|']
for key,label in labels.items():
 a=d['models'][key]['all'];c=d['models'][key]['without_shared_outliers'];
 lines.append(f"| {label} | {a['mean']['hessian_mae_ev_a2']:.5f} | {a['median']['hessian_mae_ev_a2']:.5f} | {c['mean']['hessian_mae_ev_a2']:.5f} | {a['mean']['eckart_eigval_mae_ev_a2']:.5f} | {a['mean']['neg_num_agree']*100:.2f}% |")
lines += ['', 'Hessian MAE is in eV/Å². The eigenvalue column uses the branch’s mass-weighted Eckart-projected eigenvalue metric (`eckart_eigval_mae_ev_a2`). Negative-count agreement means that prediction and reference have the same number of negative projected eigenvalues; it is not a measure of energy or force accuracy.', '',
'## Outlier rule and limitations','',
'The rebuttal rule flags a model/sample when the absolute modified z-score of its Hessian MAE exceeds 10 relative to the median and median absolute deviation for that atom count. Each size group has only 10 molecules. The 51 unique samples flagged by any of the three models are removed from every model for the paired outlier-excluded comparison (659 remain). There are 34 flags for 5 Å/all, 14 for 5 Å/5 Å, and 38 for 12 Å/12 Å; a molecule can be flagged by multiple models. Flagging alone does not establish that a reference is wrong. Because the rule uses absolute scores, two 12 Å/12 Å flags have unusually low error relative to their size peers, rather than unusually high error.','',
'All 51 flagged reference Hessians contain finite values. Eight flagged structures have a closest atom separation below 0.8 Å. PubChem_5280489 has a 0.074 Å H–H separation and reference Hessian maximum magnitude about 46,771 eV/Å², warranting inspection of its geometry and reference calculation. Many other flags are model-specific and do not have comparably short separations.','',
'## Complete outlier list','',
'`Max |z|` is the largest absolute modified z-score among the models that flagged the sample. Hessian errors are eV/Å²; minimum separation is Å.','',
'| Sample | Atoms | Flagged models | Max abs(z) | Minimum separation | MAE 5/all | MAE 5/5 | MAE 12/12 |',
'|---|---:|---|---:|---:|---:|---:|---:|']
for x in outliers:
 lines.append('| {sample} | {n} | {models} | {z:.1f} | {dist:.3f} | {a:.4f} | {b:.4f} | {c:.4f} |'.format(sample=x['sample_name'],n=x['natoms'],models=x['flagged_models'],z=float(x['max_abs_modified_z']),dist=float(x['reference_min_distance_angstrom']),a=float(x['5A_all_hessian_mae_ev_a2']),b=float(x['5A_5A_hessian_mae_ev_a2']),c=float(x['12A_12A_hessian_mae_ev_a2'])))
lines += ['', 'Complete per-model per-molecule metrics are in `raw/0/metrics.csv`, `raw/1/metrics.csv`, and `raw/2/metrics.csv`. `outlier_samples.csv` has one row per flagged molecule; `outliers.csv` has one row per model-specific flag; `per-size.csv` contains all-sample and shared-outlier-excluded means. The checkpoint and reference SHA-256 manifests are included alongside these files.']
(p/'report.md').write_text('\n'.join(lines)+'\n')
print('wrote report with',len(outliers),'outliers')
