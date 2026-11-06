"""Draw size trends with a logarithmic scale so low-error curves remain visible."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/pubchem-eval-20261106/analysis')
d=pd.read_csv(p/'per-size.csv')
colors={'5A_all':'#b34532','5A_5A':'#306ca1','12A_12A':'#39845a'}
labels={'5A_all':'5 Å / all','5A_5A':'5 Å / 5 Å','12A_12A':'12 Å / 12 Å'}
fig,ax=plt.subplots(1,2,figsize=(12,4.5),sharex=True,sharey=True)
for a,(s,t) in zip(ax,[('all','All 710 molecules'),('without_shared_outliers','Excluding 51 flagged molecules')]):
 for key in colors:
  z=d[(d['subset']==s)&(d['model']==key)]
  a.plot(z['natoms'],z['hessian_mae_ev_a2'],label=labels[key],color=colors[key],linewidth=1.3)
 a.set_yscale('log');a.set_xlabel('Number of atoms');a.set_title(t);a.grid(alpha=.2,which='both')
ax[0].set_ylabel('Mean Hessian MAE (eV / Å²; log scale)');ax[1].legend(frameon=False)
fig.suptitle('PubChem size generalization at epoch 100');fig.tight_layout()
fig.savefig(p/'size-generalization-log.png',dpi=180);fig.savefig(p/'size-generalization-log.pdf');plt.close(fig)
print('PASS: generated log-scale figure',flush=True)
