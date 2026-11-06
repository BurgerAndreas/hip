"""Summarize paired PubChem size extrapolation with rebuttal outlier rules."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106')
out=root/'analysis'; out.mkdir(exist_ok=True)
labels=('5A_all','5A_5A','12A_12A')
frames={}
for i,label in enumerate(labels):
    path=root/'results'/str(i)/'metrics.csv'
    df=pd.read_csv(path)
    if len(df)!=710 or df['sample_name'].duplicated().any() or (df['status']!='ok').any():
        raise ValueError(f'{label}: expected 710 unique successful samples, got {len(df)}, failures {(df["status"]!="ok").sum()}')
    for col in ('natoms','hessian_mae_ev_a2','hessian_rel_mae','eckart_eigval_mae_ev_a2','eckart_lowest_eigval_mae_ev_a2','neg_num_agree'):
        df[col]=pd.to_numeric(df[col],errors='raise')
        if not np.isfinite(df[col]).all(): raise ValueError(f'{label}: non-finite {col}')
    frames[label]=df
names=[set(df['sample_name']) for df in frames.values()]
if any(s != names[0] for s in names[1:]):raise ValueError('Model samples differ; paired comparisons are invalid')
sizes=[df.set_index('sample_name')['natoms'].sort_index() for df in frames.values()]
if any(not sizes[0].equals(s) for s in sizes[1:]):raise ValueError('Sample atom counts differ across models')

flags=[]
for label,df in frames.items():
    values=df['hessian_mae_ev_a2']
    med=values.groupby(df['natoms']).transform('median')
    dev=(values-med).abs()
    mad=dev.groupby(df['natoms']).transform('median')
    with np.errstate(divide='ignore',invalid='ignore'):
        score=.6745*(values-med)/mad
    flagged=df.loc[score.abs()>10].copy()
    flagged.insert(0,'model',label)
    flagged['size_median_hessian_mae']=med[score.abs()>10]
    flagged['size_mad_hessian_mae']=mad[score.abs()>10]
    flagged['modified_z_score']=score[score.abs()>10]
    flags.append(flagged)
flagged=pd.concat(flags,ignore_index=True)
if not flagged.empty:flagged=flagged.sort_values(['natoms','sample_name','model'])
flagged.to_csv(out/'outliers.csv',index=False)
removed=set(flagged['sample_name'])

fields=['hessian_mae_ev_a2','hessian_rel_mae','eckart_eigval_mae_ev_a2','eckart_lowest_eigval_mae_ev_a2','neg_num_agree']
summary={'dataset':'andreasburger/hip:orca_wb97x_631gd_chno_30_100/h5','dataset_revision':'f869cabd92ebcc66223891d0b3e815766c9bba0e','rebuttal_commit':'666463211257c5ae3d345fae6882ae22f718b505','outlier_rule':'absolute modified z-score > 10 within each atom count, on Hessian MAE; remove the union of flagged sample names from all models','outlier_sample_count':len(removed),'outlier_samples':sorted(removed),'models':{}}
size_rows=[]
for label,df in frames.items():
    model_summary={}
    for tag,subset in [('all',df),('without_shared_outliers',df[~df['sample_name'].isin(removed)])]:
        model_summary[tag]={'n':len(subset),'mean':{k:float(subset[k].mean()) for k in fields},'median':{k:float(subset[k].median()) for k in fields}}
        for n,g in subset.groupby('natoms'):
            row={'model':label,'subset':tag,'natoms':int(n),'count':len(g)}
            row.update({k:float(g[k].mean()) for k in fields});size_rows.append(row)
    summary['models'][label]=model_summary
(out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
size=pd.DataFrame(size_rows);size.to_csv(out/'per-size.csv',index=False)

colors={'5A_all':'#b34532','5A_5A':'#306ca1','12A_12A':'#39845a'}
fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharex=True)
for ax,(tag,title) in zip(axes,[('all','All samples'),('without_shared_outliers',f'Excluding {len(removed)} shared outlier samples')]):
    for label in labels:
        d=size[(size['subset']==tag)&(size['model']==label)]
        ax.plot(d['natoms'],d['hessian_mae_ev_a2'],label=label,color=colors[label],linewidth=1.4)
    ax.set_title(title);ax.set_xlabel('Number of atoms');ax.set_ylabel('Mean Hessian MAE (eV / Å²)');ax.grid(alpha=.2)
axes[1].legend(frameon=False)
fig.suptitle('PubChem ORCA size generalization · epoch 100');fig.tight_layout()
fig.savefig(out/'size-generalization.png',dpi=180);fig.savefig(out/'size-generalization.pdf');plt.close(fig)
print(json.dumps({'samples_per_model':710,'outlier_samples':len(removed),'outlier_rows':len(flagged),'summary':str(out/'summary.json')}),flush=True)
