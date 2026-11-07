import glob, sys, json
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

pub = [H5[i] for i in np.linspace(0, len(H5) - 1, 60).astype(int)]
