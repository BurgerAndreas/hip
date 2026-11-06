import sys, os, torch, numpy as np
code, mode = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
torch.manual_seed(0)
from torch_geometric.data import Data, Batch
from hip.training_module import PotentialModule
CK = '/project/aip-aspuru/aburger/hip-neighbors-runs-20261102/short100-20261105-12nG1k/final/1/final-epoch-100.ckpt'
PotentialModule.fix_paths = lambda self, tc: dict(tc)
m = PotentialModule.load_from_checkpoint(CK, strict=False, map_location='cpu').potential.eval()
if mode == 'env':
    m.edge_envelope = True; m.envelope_exponent = 5
# chain molecule: methanol-like cluster + one probe atom moved along x across 5 A from atom 0
base = torch.tensor([[0,0,0],[1.4,0,0],[-0.5,0.9,0],[-0.5,-0.9,0],[1.8,0.9,0.3],[2.6,-0.6,0.2]],dtype=torch.float32)
z = torch.tensor([6,8,1,1,1,6])
def run(xp):
    pos = torch.cat([base, torch.tensor([[-xp,0.0,0.0]])]); zz = torch.cat([z, torch.tensor([1])])
    d = Data(pos=pos, z=zz, charges=zz, natoms=torch.tensor([7]), cell=None, pbc=torch.tensor(False))
    b = Batch.from_data_list([d])
    with torch.no_grad(): e, f, o = m.forward(b, otf_graph=True, hessian=True)
    H = o['hessian'].reshape(21, 21)
    return float(e.sum()), H
xs = [4.90, 4.97, 4.99, 4.999, 5.001, 5.01, 5.05]
out = [run(x) for x in xs]
for x,(e,H) in zip(xs,out):
    blk = H[0:3, 18:21].abs().max().item()   # block between atom 0 and probe atom 6 (distance = x)
    print(f'{mode} x={x:.3f} E={e:.6f} |H_0,probe|max={blk:.5f} H_sum={H.abs().sum().item():.4f}')
np.save(f'/tmp/env_test_{mode}_{os.path.basename(code)}.npy', np.array([o[1].numpy() for o in out]))
