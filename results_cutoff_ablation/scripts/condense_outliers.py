"""Produce one row per flagged sample, retaining errors for all three models."""
from pathlib import Path
import csv, math
r=Path(__file__).resolve().parent/'results-20261106'
models=('5A_all','5A_5A','12A_12A')
with (r/'outliers.csv').open(newline='') as f: flags=list(csv.DictReader(f))
with (r/'outlier_samples.csv').open('w',newline='') as f:
 fields=['sample_name','natoms','flagged_models','max_abs_modified_z','reference_min_distance_angstrom','reference_closest_atomic_numbers','reference_hessian_mean_abs_ev_a2','reference_negative_eigenvalues']+[f'{m}_hessian_mae_ev_a2' for m in models]+[f'{m}_modified_z' for m in models]
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 data={}
 for x in flags:
  d=data.setdefault(x['sample_name'],{'sample_name':x['sample_name'],'natoms':int(x['natoms']),'flagged_models':[],'max_abs_modified_z':0.0})
  d['flagged_models'].append(x['model']);d[x['model']+'_modified_z']=x['modified_z_score'];d['max_abs_modified_z']=max(d['max_abs_modified_z'],abs(float(x['modified_z_score'])))
 for i,m in enumerate(models):
  with (r/'raw'/str(i)/'metrics.csv').open(newline='') as src:
   for row in csv.DictReader(src):
    if row['sample_name'] in data:
     data[row['sample_name']][m+'_hessian_mae_ev_a2']=row['hessian_mae_ev_a2']
     data[row['sample_name']]['reference_negative_eigenvalues']=row['true_neg_num']
 with (r/'outlier-reference-diagnostics.csv').open(newline='') as src:
  for row in csv.DictReader(src):
   d=data[row['sample_name']]
   d['reference_min_distance_angstrom']=row['min_distance_angstrom']
   d['reference_closest_atomic_numbers']=row['closest_atomic_numbers']
   d['reference_hessian_mean_abs_ev_a2']=row['reference_hessian_mean_abs_ev_a2']
 for d in sorted(data.values(),key=lambda x:-x['max_abs_modified_z']):
  d['flagged_models']='; '.join(d['flagged_models']);w.writerow(d)
print('unique outliers',len(data))
