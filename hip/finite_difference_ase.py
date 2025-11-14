# from
# https://github.com/jmusiel/gibby/blob/main/gibby/utils/ase_utils.py

import numpy as np
import shutil
import os
import sys
from ase.vibrations import Vibrations
from ase import Atoms


def get_fmax(forces: np.ndarray):
    return np.sqrt((forces**2).sum(axis=1).max())


class suppress_stdout:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_finite_difference_hessian(atoms, run_dir=None, hessian_delta=0.01, tags=None):
    if run_dir is not None:
        cwd = os.getcwd()
        os.chdir(run_dir)
    shutil.rmtree("vib", ignore_errors=True)
    # if tags is not None:
    #     indices = [i for i in range(len(atoms)) if tags[i] == 2]
    # elif len(atoms.constraints) > 0:
    #     indices = [i for i in range(len(atoms)) if i not in atoms.constraints[0].index]
    # else:
    #     indices = None
    vib = Vibrations(
        atoms,
        # indices=indices,
        delta=hessian_delta,
    )
    with suppress_stdout():
        vib.run()
        vib.summary(log=sys.stdout)
    shutil.rmtree("vib", ignore_errors=True)
    hessian = vib.H
    if run_dir is not None:
        os.chdir(cwd)
    return hessian


if __name__ == "__main__":
    from hip.equiformer_ase_calculator import EquiformerASECalculator
    from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
    from hip.path_config import DATA_PATH_HORM_SAMPLE

    dataset = LmdbDataset(DATA_PATH_HORM_SAMPLE)

    sample = dataset[0]

    calc = EquiformerASECalculator(
        # checkpoint_path=checkpoint_path,
        # hessian_method="predict",
    )

    atoms = Atoms(
        symbols=[Z_TO_ATOM_SYMBOL[int(z)] for z in sample["z"].tolist()],
        positions=sample["pos"],
    )
    atoms.calc = calc

    hessian = get_finite_difference_hessian(atoms)
    print(hessian.shape)
