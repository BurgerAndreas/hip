"""Summarize reference Hessians and geometric proximity for flagged samples."""
import csv, math
from pathlib import Path
import h5py
import numpy as np
r=Path('/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106')
ref=r.parent/'pubchem-reference/orca_wb97x_631gd_chno_30_100/h5'
with (r/'analysis/outliers.csv').open(newline='') as f: flags=list(csv.DictReader(f))
selected={row['sample_name'] for row in flags}
rows=[]
for name in sorted(selected):
 p=ref/(name+'.h5')
 with h5py.File(p,'r') as h:
  pos=np.asarray(h['coordinates_bohr'],dtype=float)*0.529177210903
  H=np.asarray(h['hessian_hartree_per_bohr2'],dtype=float)*27.211386245988/(0.529177210903**2)
  z=np.asarray(h['atomic_numbers'],dtype=int)
 dist=np.linalg.norm(pos[:,None,:]-pos[None,:,:],axis=2)
 np.fill_diagonal(dist,np.inf)
 pair=np.unravel_index(dist.argmin(),dist.shape)
 rows.append({'sample_name':name,'natoms':len(z),'min_distance_angstrom':float(dist[pair]),'closest_atomic_numbers':f'{z[pair[0]]}-{z[pair[1]]}','reference_hessian_mean_abs_ev_a2':float(np.abs(H).mean()),'reference_hessian_max_abs_ev_a2':float(np.abs(H).max()),'reference_hessian_finite':bool(np.isfinite(H).all())})
with (r/'analysis/outlier-reference-diagnostics.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('diagnosed',len(rows),'outliers; closest distance',min(x['min_distance_angstrom'] for x in rows),'angstrom; finite',sum(x['reference_hessian_finite'] for x in rows),flush=True)
