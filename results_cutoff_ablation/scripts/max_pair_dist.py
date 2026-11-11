import sys, numpy as np, torch
from multiprocessing import Pool
from hip.ff_lmdb import LmdbDataset
PATH = sys.argv[1]; W = 16
def work(rng):
    torch.set_num_threads(1); ds = LmdbDataset(PATH); out = []
    for i in range(*rng):
        p = ds[i].pos.double(); out.append((len(p), torch.pdist(p).max().item() if len(p) > 1 else 0.0))
    return out
if __name__ == '__main__':
    n = len(LmdbDataset(PATH)); print('samples', n, flush=True)
    edges = np.linspace(0, n, W * 8 + 1).astype(int)
    with Pool(W) as pool: res = pool.map(work, list(zip(edges[:-1], edges[1:])))
    a = np.array([x for r in res for x in r]); na, mx = a[:, 0].astype(int), a[:, 1]
    j = int(mx.argmax())
    print(f'MAX pair distance {mx[j]:.3f} A at index {j} (natoms {na[j]})')
    print('per-molecule max distance percentiles 50/90/99/99.9:', np.round(np.percentile(mx, [50, 90, 99, 99.9]), 3).tolist())
    print('natoms max', int(na.max()), 'mean', round(float(na.mean()), 2))
    for t in (5, 7.5, 10, 12, 14.36): print(f'fraction of molecules with any pair > {t} A: {(mx > t).mean():.5f}')
