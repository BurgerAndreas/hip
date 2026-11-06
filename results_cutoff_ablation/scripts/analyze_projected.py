"""Summarize full-space projected Hessians and reduced-space vibrational metrics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Check that the 3N-space projector matches the branch's explicit full form.
import sys
root='/project/aip-aspuru/aburger/hip-pubchem-eval-20261106'
sys.path.insert(0,root)
sys.path.insert(0,root+'/scripts')
from rebuttal_frequency_analysis import get_trans_rot_projector
rng=np.random.default_rng(893)
for n in (4,12,30):
    coords=rng.normal(size=3*n)
    masses=rng.uniform(1,16,size=n)
    q=get_trans_rot_projector(coords,masses,full=False)
    direct=get_trans_rot_projector(coords,masses,full=True)
    np.testing.assert_allclose(q.T @ q,direct,rtol=0,atol=1e-12)
print('PASS: reduced-basis and explicit full-space Eckart projectors agree',flush=True)

r=Path('/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106')
out=r/'projected-analysis';out.mkdir(exist_ok=True)
labels=('5A_all','5A_5A','12A_12A')
fields=['projected_hessian_mae_ev_a2_amu','projected_block_rmse_mean_ev_a2_amu','projected_hessian_rmse_ev_a2_amu','eckart_eigval_mae_ev_a2','eckart_abs_cos_v1']
original=pd.read_csv(r/'analysis/outliers.csv')
original_excluded=set(original['sample_name'])
frames={};flags=[]
for i,label in enumerate(labels):
 df=pd.read_csv(r/'projected'/str(i)/'metrics.csv')
 old=pd.read_csv(r/'results'/str(i)/'metrics.csv')
 if len(df)!=710 or df['sample_name'].duplicated().any() or (df['status']!='ok').any():raise ValueError(f'{label}: missing, duplicate or failed sample')
 for c in fields+['natoms','hessian_mae_ev_a2']:
  df[c]=pd.to_numeric(df[c],errors='raise')
  if not np.isfinite(df[c]).all():raise ValueError(f'{label}: non-finite {c}')
 if (df['eckart_abs_cos_v1']>1+1e-10).any() or (df['eckart_abs_cos_v1']<0).any():raise ValueError('invalid cosine')
 if (df['projected_hessian_mae_ev_a2_amu']>df['projected_block_rmse_mean_ev_a2_amu']+1e-10).any():raise ValueError('block RMSE below MAE')
 if (df['projected_block_rmse_mean_ev_a2_amu']>df['projected_hessian_rmse_ev_a2_amu']+1e-10).any():raise ValueError('block mean RMSE above global RMSE')
 z=df.set_index('sample_name').sort_index();o=old.set_index('sample_name').sort_index()
 if not z.index.equals(o.index):raise ValueError('sample names differ from original evaluation')
 for key in ('hessian_mae_ev_a2','eckart_eigval_mae_ev_a2'):
  if not np.allclose(z[key],o[key],rtol=2e-5,atol=1e-6):raise ValueError(f'{label} changed prior metric {key}')
 values=df['projected_hessian_mae_ev_a2_amu'];med=values.groupby(df['natoms']).transform('median');mad=(values-med).abs().groupby(df['natoms']).transform('median')
 with np.errstate(divide='ignore',invalid='ignore'):score=.6745*(values-med)/mad
 f=df.loc[score.abs()>10].copy();f.insert(0,'model',label);f['modified_z_score']=score[score.abs()>10];flags.append(f)
 frames[label]=df
names=[set(x['sample_name']) for x in frames.values()]
if any(x!=names[0] for x in names[1:]):raise ValueError('model sample mismatch')
flagged=pd.concat(flags,ignore_index=True);flagged.to_csv(out/'projected-outliers.csv',index=False)
projected_excluded=set(flagged['sample_name'])
summary={'samples_per_model':710,'original_rule_outliers':len(original_excluded),'projected_rule_outliers':len(projected_excluded),'projected_outlier_samples':sorted(projected_excluded),'definition':'Full-space P = Q.T @ Q where Q is the rebuttal reduced Eckart projector. P (M^-1/2 H_pred M^-1/2 - M^-1/2 H_ref M^-1/2) P. Block RMSE is the mean over N² atom-pair 3x3 blocks of each block’s elementwise RMSE. Eigenvalues and |cos v1| use the shared reduced basis Q.','units':'Hessian and eigenvalue quantities eV / (angstrom^2 amu); cosine dimensionless','models':{}}
combined=[];size_rows=[]
for label,df in frames.items():
 df=df.copy();df.insert(0,'model',label);df['original_outlier']=df['sample_name'].isin(original_excluded);df['projected_outlier']=df['sample_name'].isin(projected_excluded);combined.append(df)
 summaries={}
 for name,subset in [('all',df),('without_original_outliers',df[~df['original_outlier']]),('without_projected_outliers',df[~df['projected_outlier']])]:
  summaries[name]={'n':len(subset),'mean':{c:float(subset[c].mean()) for c in fields},'median':{c:float(subset[c].median()) for c in fields}}
  for n,g in subset.groupby('natoms'):
   item={'model':label,'subset':name,'natoms':int(n),'count':len(g)};item.update({c:float(g[c].mean()) for c in fields});size_rows.append(item)
 summary['models'][label]=summaries
pd.concat(combined,ignore_index=True).to_csv(out/'per-molecule.csv',index=False)
size=pd.DataFrame(size_rows);size.to_csv(out/'per-size.csv',index=False)
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
colors={'5A_all':'#b34532','5A_5A':'#306ca1','12A_12A':'#39845a'}
names={'5A_all':'5 Å / all','5A_5A':'5 Å / 5 Å','12A_12A':'12 Å / 12 Å'}
panels=[(fields[0],'Projected Hessian MAE',True),(fields[1],'Mean atom-pair block RMSE',True),(fields[3],'Projected eigenvalue MAE',True),(fields[4],'|cos(v1)|',False)]
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
for ax,(metric,title,log) in zip(axes.flat,panels):
 for label in labels:
  g=size[(size['subset']=='without_original_outliers')&(size['model']==label)]
  ax.plot(g['natoms'],g[metric],label=names[label],color=colors[label],linewidth=1.3)
 ax.set_title(title);ax.set_xlabel('Number of atoms');ax.grid(alpha=.2,which='both')
 if log:ax.set_yscale('log');ax.set_ylabel('eV / (Å² amu)')
 else:ax.set_ylim(0,1.02);ax.set_ylabel('Absolute cosine')
axes[0,0].legend(frameon=False)
fig.suptitle(f'PubChem projected metrics · excluding {len(original_excluded)} original-rule outliers');fig.tight_layout()
fig.savefig(out/'projected-size-generalization.png',dpi=180);fig.savefig(out/'projected-size-generalization.pdf');plt.close(fig)
print(json.dumps({'samples_per_model':710,'original_outliers':len(original_excluded),'projected_outliers':len(projected_excluded),'projected_outlier_rows':len(flagged)}),flush=True)
