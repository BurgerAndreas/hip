#!/usr/bin/env python3
"""Infer the graph a checkpoint was trained with from how it behaves.

Evaluates each checkpoint on a T1x-val subset (in distribution) and a PubChem subset under
several runtime graph settings. The setting that reproduces in-distribution accuracy is the
training setting. Prints one summary table and writes per-molecule errors to
$PROBE_OUT/probe-graph-per-molecule.csv (default <run>/pubchem-eval-20261106/probe-out).
"""
import glob, sys, json, csv, os
from pathlib import Path
import numpy as np, torch, h5py

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_projected_pubchem import load_module_for_inference, make_batch, load_hf_orca_hessian
from hip.ff_lmdb import LmdbDataset

P = '/project/aip-aspuru/aburger/hip-neighbors-runs-20261102'
CKPTS = {
    'hip_v2': f'{P}/pubchem-eval-20261106/released-ckpt/hip_v2.ckpt',
    'hip_v3': f'{P}/pubchem-eval-20261106/released-ckpt/hip_v3.ckpt',
    'abl_5_all': f'{P}/short100-20261105-12nG1k/final/0/final-epoch-100.ckpt',
    'abl_5_5': f'{P}/short100-20261105-12nG1k/final/1/final-epoch-100.ckpt',
    'abl_12_12': f'{P}/short100-20261105-12nG1k/final/2/final-epoch-100.ckpt',
}
VAL = '/project/aip-aspuru/aburger/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/ts1x-val.lmdb'
H5 = sorted(glob.glob(f'{P}/pubchem-reference/orca_wb97x_631gd_chno_30_100/h5/*.h5'))
HB2EVA2 = 27.211386245988 / 0.529177210903**2
dev = torch.device('cuda')

val = LmdbDataset(VAL)
rng = np.random.default_rng(0)
val_idx = rng.choice(len(val), 300, replace=False)
pub = [H5[i] for i in np.linspace(0, len(H5) - 1, 60).astype(int)]

def nb_stats(pos, r):
    d = torch.cdist(pos, pos); d.fill_diagonal_(1e9)
    return (d < r).sum(1).float()

# neighbour statistics (T1x vs PubChem)
stats = {}
for name, items in (('t1x', [val[int(i)].pos for i in val_idx]),
                    ('pubchem', [torch.tensor(load_hf_orca_hessian(Path(p))[1] * 0.529177210903, dtype=torch.float32) for p in pub])):
    for r in (5.0, 7.0, 12.0):
        n = torch.cat([nb_stats(x, r) for x in items])
        stats[f'{name}_r{r:g}'] = (float(n.mean()), float(n.max()))
    stats[f'{name}_natoms'] = (float(np.mean([len(x) for x in items])), float(max(len(x) for x in items)))
for k, v in stats.items(): print(f'NB {k}: mean {v[0]:.1f} max {v[1]:.0f}', flush=True)

# runtime graph settings: (label, max_neighbors, fully_connected_hessian, hessian_graph_cutoff or None=own backbone cutoff)
SETTINGS = [('nb32_Hfull', 32, True, None), ('nb1000_Hfull', 1000, True, None),
            ('nb32_Hown', 32, False, None), ('nb1000_Hown', 1000, False, None)]

def run(model, batch):
    with torch.no_grad():
        e, f, out = model.forward(batch, otf_graph=True, hessian=True)
    n = batch.pos.numel()
    return e, f, out['hessian'].reshape(n, n)

OUT = Path(os.environ.get('PROBE_OUT', f'{P}/pubchem-eval-20261106/probe-out')); OUT.mkdir(parents=True, exist_ok=True)
PM = OUT / 'probe-graph-per-molecule.csv'
FIELDS = ['ckpt', 'setting', 'dataset', 'sample', 'natoms', 'hessian_mae_ev_a2', 'forces_mae_ev_a', 'energy_ae_ev']
with PM.open('w', newline='') as fh: csv.DictWriter(fh, FIELDS).writeheader()
rows = []
for cname, path in CKPTS.items():
    module = load_module_for_inference(Path(path), dev)
    model = module.potential.to(dev).eval()
    own = float(model.cutoff)
    for label, nb, fc, hc in SETTINGS:
        model.max_neighbors = nb
        model.fully_connected_hessian = fc
        model.hessian_graph_cutoff = own if hc is None else hc
        t_h, t_f, t_e, per = [], [], [], []
        for i in val_idx:
            d = val[int(i)]
            b = make_batch(d.z.numpy(), d.pos.numpy(), dev)
            e, f, h = run(model, b)
            n = 3 * len(d.z)
            t_h.append(float((h.cpu() - d.hessian.reshape(n, n)).abs().mean()))
            t_f.append(float((f.cpu() - d.forces).abs().mean()))
            t_e.append(abs(float(e.squeeze()) - float(d.ae)))
            per.append(dict(ckpt=cname, setting=label, dataset='t1x_val', sample=int(i), natoms=len(d.z),
                            hessian_mae_ev_a2=t_h[-1], forces_mae_ev_a=t_f[-1], energy_ae_ev=t_e[-1]))
        p_h = []
        for p in pub:
            z, xb, _, hb = load_hf_orca_hessian(Path(p))
            b = make_batch(z, xb * 0.529177210903, dev)
            _, _, h = run(model, b)
            p_h.append(float((h.cpu().double() - torch.as_tensor(hb * HB2EVA2)).abs().mean()))
            per.append(dict(ckpt=cname, setting=label, dataset='pubchem', sample=Path(p).stem, natoms=len(z),
                            hessian_mae_ev_a2=p_h[-1], forces_mae_ev_a='', energy_ae_ev=''))
        rows.append(dict(ckpt=cname, setting=label, backbone=own, t1x_H=np.mean(t_h), t1x_F=np.mean(t_f),
                         t1x_E=np.mean(t_e), pub_H_mean=np.mean(p_h), pub_H_median=np.median(p_h)))
        print('ROW', json.dumps(rows[-1]), flush=True)
        with PM.open('a', newline='') as fh: csv.DictWriter(fh, FIELDS).writerows(per)
    del model, module; torch.cuda.empty_cache()

print('\n| ckpt | setting | T1x H MAE | T1x F MAE | T1x E MAE | PubChem H MAE mean | median |')
for r in rows:
    print(f"| {r['ckpt']} | {r['setting']} | {r['t1x_H']:.4f} | {r['t1x_F']:.4f} | {r['t1x_E']:.4f} | {r['pub_H_mean']:.4f} | {r['pub_H_median']:.4f} |")
