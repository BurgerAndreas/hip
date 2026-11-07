#!/usr/bin/env python3
"""Inference-time cross test: each checkpoint with its own backbone, Hessian graph radius swept.

Separates the backbone and Hessian-graph contributions to PubChem size extrapolation. 60 PubChem
molecules (same subset as probe_graph.py), raw Hessian MAE in eV/A^2, and MAE restricted to atom
pairs closer / farther than 5 A. Prints one markdown table.
"""
import sys, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import probe_graph_common as C

rows = []
for cname, path in C.CKPTS.items():
    module = C.load_module_for_inference(Path(path), C.dev)
    model = module.potential.to(C.dev).eval(); model.max_neighbors = 1000
    for hg in (4.0, 5.0, 6.0, 7.0, 9.0, 12.0, 16.0, 32.0, 'full'):
        model.fully_connected_hessian = hg == 'full'
        model.hessian_graph_cutoff = 100.0 if hg == 'full' else hg
        mae, near, far = [], [], []
        for p in C.pub:
            z, xb, _, hb = C.load_hf_orca_hessian(Path(p))
            pos = xb * 0.529177210903
            b = C.make_batch(z, pos, C.dev)
            with torch.no_grad():
                _, _, out = model.forward(b, otf_graph=True, hessian=True)
            n = len(z); d = np.abs(out['hessian'].reshape(3*n, 3*n).double().cpu().numpy() - hb * C.HB2EVA2)
            blk = d.reshape(n, 3, n, 3).mean(axis=(1, 3))
            r = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
            mae.append(d.mean()); near.append(blk[r < 5].mean()); far.append(blk[r >= 5].mean())
        rows.append(dict(ckpt=cname, backbone=float(model.cutoff), hessian_graph=hg,
                         med=np.median(mae), mean=np.mean(mae), near=np.median(near), far=np.median(far)))
        print('ROW', json.dumps(rows[-1]), flush=True)
    del model, module; torch.cuda.empty_cache()
print('\n| ckpt | backbone | Hessian graph | PubChem H MAE median | mean | median MAE pairs < 5 A | pairs >= 5 A |')
for r in rows:
    print(f"| {r['ckpt']} | {r['backbone']:g} | {r['hessian_graph']} | {r['med']:.4f} | {r['mean']:.4f} | {r['near']:.4f} | {r['far']:.5f} |")
