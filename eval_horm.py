import torch
import argparse
import numpy as np
from tqdm import tqdm
import wandb
import pandas as pd
import os
# from torch_geometric.loader import DataLoader as TGDataLoader

from pathlib import Path

import bisect
import pickle

import lmdb
from torch.utils.data import Dataset
import scipy.constants as spc
from ase import Atoms

try:
    from fairchem.core.datasets import data_list_collater
    from fairchem.core.datasets.atomic_data import AtomicData
except ImportError as e:
    print(f"Error importing fairchem: {e}")
    data_list_collater = None
    AtomicData = None


#################################################################################
# Adapted from
# dependencies/pysisyphus/pysisyphus/Geometry.py
#################################################################################
# from pysisyphus.constants import AU2J, BOHR2ANG, C, R, AU2KJPERMOL, NA
# Bohr radius in m
BOHR2M = spc.value("Bohr radius")
# Bohr -> Å conversion factor
BOHR2ANG = BOHR2M * 1e10
# Å -> Bohr conversion factor
ANG2BOHR = 1 / BOHR2ANG
# Hartree to J
AU2J = spc.value("Hartree energy")
# Speed of light in m/s
C = spc.c
NA = spc.Avogadro
HARTREE2EV = spc.physical_constants["Hartree energy in eV"][0]

# Taken from periodictable-1.5.0
MASS_DICT = {
    "x": 1.0,  # dummy atom
    "n": 14.0067,
    "h": 1.00794,
    "he": 4.002602,
    "li": 6.941,
    "be": 9.012182,
    "b": 10.811,
    "c": 12.0107,
    "o": 15.9994,
    "f": 18.9984032,
    "ne": 20.1797,
    "na": 22.98977,
    "mg": 24.305,
    "al": 26.981538,
    "si": 28.0855,
    "p": 30.973761,
    "s": 32.065,
    "cl": 35.453,
    "ar": 39.948,
    "k": 39.0983,
    "ca": 40.078,
    "sc": 44.95591,
    "ti": 47.867,
    "v": 50.9415,
    "cr": 51.9961,
    "mn": 54.938049,
    "fe": 55.845,
    "co": 58.9332,
    "ni": 58.6934,
    "cu": 63.546,
    "zn": 65.409,
    "ga": 69.723,
    "ge": 72.64,
    "as": 74.9216,
    "se": 78.96,
    "br": 79.904,
    "kr": 83.798,
    "rb": 85.4678,
}
KNOWN_ATOMS = tuple(MASS_DICT.keys())


#################################################################################
# Frequency analysis functions
#################################################################################


def _to_torch_double(array_like, device=None):
    if isinstance(array_like, torch.Tensor):
        return array_like.to(dtype=torch.float64, device=device)
    return torch.as_tensor(array_like, dtype=torch.float64, device=device)


def inertia_tensor_torch(coords3d, masses):
    """Inertia tensor using torch."""
    coords3d_t = _to_torch_double(coords3d)
    masses_t = _to_torch_double(masses)
    x, y, z = coords3d_t.T
    squares = torch.sum(coords3d_t**2 * masses_t[:, None], dim=0)
    I_xx = squares[1] + squares[2]
    I_yy = squares[0] + squares[2]
    I_zz = squares[0] + squares[1]
    I_xy = -torch.sum(masses_t * x * y)
    I_xz = -torch.sum(masses_t * x * z)
    I_yz = -torch.sum(masses_t * y * z)
    return torch.stack(
        [
            torch.stack([I_xx, I_xy, I_xz]),
            torch.stack([I_xy, I_yy, I_yz]),
            torch.stack([I_xz, I_yz, I_zz]),
        ]
    )


def get_trans_rot_vectors_torch(cart_coords, masses, rot_thresh=1e-6):
    """Torch version of get_trans_rot_vectors."""
    cart_coords_t = _to_torch_double(cart_coords)
    masses_t = _to_torch_double(masses)

    coords3d = cart_coords_t.reshape(-1, 3)
    total_mass = torch.sum(masses_t)
    com = (coords3d * masses_t[:, None]).sum(dim=0) / total_mass
    coords3d_centered = coords3d - com[None, :]

    _, Iv = torch.linalg.eigh(inertia_tensor_torch(coords3d, masses_t))
    Iv = Iv.T  # rows are eigenvectors

    masses_rep = masses_t.repeat_interleave(3)
    sqrt_masses = torch.sqrt(masses_rep)
    num = masses_t.numel()

    # Translation vectors (mass-weighted unit vectors along axes)
    trans_vecs = []  # (3, 3N)
    device = cart_coords_t.device
    for vec in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        tiled = _to_torch_double(vec, device=device).repeat(num)
        v = sqrt_masses * tiled
        trans_vecs.append(v / torch.linalg.norm(v))  # (3N,)

    # Rotation vectors
    rot_vecs = torch.zeros(
        (3, cart_coords_t.numel()), dtype=torch.float64, device=device
    )
    for i in range(masses_t.size(0)):
        p_vec = Iv @ coords3d_centered[i]
        for ix in range(3):
            rot_vecs[0, 3 * i + ix] = Iv[2, ix] * p_vec[1] - Iv[1, ix] * p_vec[2]
            rot_vecs[1, 3 * i + ix] = Iv[2, ix] * p_vec[0] - Iv[0, ix] * p_vec[2]
            rot_vecs[2, 3 * i + ix] = Iv[0, ix] * p_vec[1] - Iv[1, ix] * p_vec[0]
    rot_vecs = rot_vecs * sqrt_masses[None, :]  # (3, 3N)

    # Drop vectors with vanishing norms
    norms = torch.linalg.norm(rot_vecs, dim=1)  # (3)
    keep = norms > rot_thresh
    rot_vecs = rot_vecs[keep]  # (3, 3N)

    trans_vecs = torch.stack(trans_vecs)  # (3, 3N)
    tr_vecs = torch.cat([trans_vecs, rot_vecs], dim=0)  # (6, 3N)
    Q, _ = torch.linalg.qr(tr_vecs.T)
    return Q.T  # (6, 3N)


def get_trans_rot_projector_torch(cart_coords, masses, full=False):
    tr_vecs = get_trans_rot_vectors_torch(cart_coords, masses=masses)
    if full:
        n = tr_vecs.size(1)
        P = torch.eye(n, dtype=tr_vecs.dtype, device=tr_vecs.device)
        for tr_vec in tr_vecs:
            P = P - torch.outer(tr_vec, tr_vec)
        return P
    else:
        U, S, _ = torch.linalg.svd(tr_vecs.T, full_matrices=True)
        P = U[:, S.numel() :].T
        return P


def massweigh_hessian_torch(hessian, masses3d):
    """mass-weighted hessian M^(-1/2) H M^(-1/2) using torch."""
    h_t = _to_torch_double(hessian, device=hessian.device)
    m_t = _to_torch_double(masses3d, device=hessian.device)
    mm_sqrt_inv = torch.diag(
        1.0 / torch.sqrt(m_t),
    )
    return mm_sqrt_inv @ h_t @ mm_sqrt_inv


def unweight_mw_hessian_torch(mw_hessian, masses3d):
    h_t = _to_torch_double(mw_hessian, device=mw_hessian.device)
    m_t = _to_torch_double(masses3d, device=mw_hessian.device)
    mm_sqrt = torch.diag(
        torch.sqrt(m_t),
    )
    return mm_sqrt @ h_t @ mm_sqrt


def massweigh_and_eckartprojection_torch(
    hessian: torch.Tensor,
    cart_coords: torch.Tensor,
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    """Eckart projection starting from not-mass-weighted Hessian (torch).

    hessian: torch.Tensor (N*3, N*3)
    cart_coords: torch.Tensor (N*3)
    atomsymbols: list[str] (N)
    """
    masses_t = torch.tensor(
        [MASS_DICT[atom.lower()] for atom in atomsymbols],
        dtype=torch.float64,
        device=hessian.device,
    )
    masses3d_t = masses_t.repeat_interleave(3)

    mw_hessian_t = massweigh_hessian_torch(hessian, masses3d_t)
    P_t = get_trans_rot_projector_torch(cart_coords, masses=masses_t, full=False)
    proj_hessian_t = P_t @ mw_hessian_t @ P_t.T
    proj_hessian_t = (proj_hessian_t + proj_hessian_t.T) / 2.0
    return proj_hessian_t


def analyze_frequencies_torch(
    hessian: torch.Tensor,  # eV/Angstrom^2
    cart_coords: torch.Tensor,  # Angstrom
    atomsymbols: list[str],
    ev_thresh: float = -1e-6,
):
    cart_coords = cart_coords.reshape(-1, 3).to(hessian.device)
    hessian = hessian.reshape(cart_coords.numel(), cart_coords.numel())

    if isinstance(atomsymbols[0], torch.Tensor):
        atomsymbols = atomsymbols.tolist()
    if not isinstance(atomsymbols[0], str):
        # atomic numbers were passed instead of symbols
        atomsymbols = [Z_TO_ATOM_SYMBOL[z] for z in atomsymbols]

    proj_hessian = massweigh_and_eckartprojection_torch(
        hessian, cart_coords, atomsymbols
    )
    eigvals, eigvecs = torch.linalg.eigh(proj_hessian)

    neg_inds = eigvals < ev_thresh
    neg_eigvals = eigvals[neg_inds]
    neg_num = sum(neg_inds)
    # # eigval_str = np.array2string(eigvals[:10], precision=4)
    # if neg_num > 0:
    #     wavenumbers = eigval_to_wavenumber(neg_eigvals)
    #     # wavenum_str = np.array2string(wavenumbers, precision=2)
    # else:
    #     wavenumbers = None
    return {
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        # "wavenumbers": wavenumbers,
        "neg_eigvals": neg_eigvals,
        "neg_num": neg_num,
        "natoms": len(atomsymbols),
    }


#################################################################################
# HORM dataset
#################################################################################

GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8, 9])
GLOBAL_ATOM_SYMBOLS = np.array(["H", "C", "N", "O", "F"])
Z_TO_ATOM_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
}


def onehot_convert(atomic_numbers, device):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],  # H
        6: [0, 1, 0, 0, 0],  # C
        7: [0, 0, 1, 0, 0],  # N
        8: [0, 0, 0, 1, 0],  # O
        9: [0, 0, 0, 0, 1],  # F
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return torch.tensor(onehot, dtype=torch.int64, device=device)


class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Args:
            config (dict): Dataset configuration
            transform (callable, optional): Data transform function.
                    (default: :obj:`None`)
    """

    def __init__(self, src, transform=None, **kwargs):
        super(LmdbDataset, self).__init__()

        self.path = Path(src)
        if not self.path.is_file():
            db_paths = sorted(self.path.glob("*.lmdb"))
            assert len(db_paths) > 0, f"No LMDBs found in '{self.path}'"

            self.metadata_path = self.path / "metadata.npz"

            self._keys, self.envs = [], []
            for db_path in db_paths:
                self.envs.append(self.connect_db(db_path))
                length = pickle.loads(
                    self.envs[-1].begin().get("length".encode("ascii"))
                )
                self._keys.append(list(range(length)))

            keylens = [len(k) for k in self._keys]
            self._keylen_cumulative = np.cumsum(keylens).tolist()
            self.num_samples = sum(keylens)
        else:
            self.metadata_path = self.path.parent / "metadata.npz"
            self.env = self.connect_db(self.path)
            try:
                # Try to get the stored length value first
                self.num_samples = pickle.loads(
                    self.env.begin().get("length".encode("ascii"))
                )
            except (TypeError, KeyError):
                # Fallback to entries count if length key doesn't exist
                self.num_samples = self.env.stat()["entries"]

            self._keys = [f"{j}".encode("ascii") for j in range(self.num_samples)]

        self.transform = transform

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_samples} samples"
            )

        if not self.path.is_file():
            # Figure out which db this should be indexed from.
            db_idx = bisect.bisect(self._keylen_cumulative, idx)
            # Extract index of element within that db.
            el_idx = idx
            if db_idx != 0:
                el_idx = idx - self._keylen_cumulative[db_idx - 1]
            assert el_idx >= 0

            # Return features.
            datapoint_pickled = (
                self.envs[db_idx]
                .begin()
                .get(f"{self._keys[db_idx][el_idx]}".encode("ascii"))
            )
            data_object = pickle.loads(datapoint_pickled)
            data_object.id = f"{db_idx}_{el_idx}"
        else:
            datapoint_pickled = self.env.begin().get(self._keys[idx])
            if datapoint_pickled is None:
                raise KeyError(f"No data found for index {idx}")
            data_object = pickle.loads(datapoint_pickled)

        if self.transform is not None:
            data_object = self.transform(data_object)

        data_object.dataset_idx = torch.tensor(idx)

        indices = data_object.one_hot.long().argmax(dim=1)
        data_object.z = GLOBAL_ATOM_NUMBERS.to(data_object.pos.device)[
            indices.to(data_object.pos.device)
        ]

        return data_object

    def connect_db(self, lmdb_path=None):
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=1099511627776 * 2,
        )
        return env

    def close_db(self):
        if not self.path.is_file():
            for env in self.envs:
                env.close()
        else:
            self.env.close()


# Example usage:
# dataset_dir = os.path.expanduser(
#     "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
# )
# dataset_files = [
#     "ts1x-val.lmdb",
#     "ts1x_hess_train_big.lmdb",
#     "RGD1.lmdb",
# ]
# lmdb_path = os.path.join(dataset_dir, dataset_files[0])
# lmdb_dataset = LmdbDataset(lmdb_path)


def _get_derivatives(x, y, retain_graph=None, create_graph=False):
    """Helper function to compute derivatives"""
    grad = torch.autograd.grad(
        [y.sum()], [x], retain_graph=retain_graph, create_graph=create_graph
    )[0]
    return grad


def compute_hessian(coords, energy, forces=None):
    """Compute Hessian matrix using autograd."""
    # Compute forces if not given
    if forces is None:
        forces = -_get_derivatives(coords, energy, create_graph=True)

    # Get number of components (n_atoms * 3)
    n_comp = forces.reshape(-1).shape[0]

    # Initialize hessian
    hess = []
    for f in forces.reshape(-1):
        # Compute second-order derivative for each element
        hess_row = _get_derivatives(coords, -f, retain_graph=True)
        hess.append(hess_row)

    # Stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp, -1)


# def conditional_grad(dec):
#     "Decorator to enable/disable grad depending on whether force/energy predictions are being made"

#     # Adapted from https://stackoverflow.com/questions/60907323/accessing-class-property-as-decorator-argument
#     def decorator(func):
#         @wraps(func)
#         def cls_method(self, *args, **kwargs):
#             f = func
#             if self.regress_forces and not getattr(self, "direct_forces", 0):
#                 f = dec(func)
#             return f(self, *args, **kwargs)

#         return cls_method

#     return decorator


def _fairchem_ase_get_hessian(_atoms: Atoms, _calc, vmap: bool = False) -> np.ndarray:
    """
    Get the Hessian matrix for the given atomic structure.
    Args:
        atoms (Atoms): The atomic structure to calculate the Hessian for.
        vmap (bool): Whether to use vectorized mapping for Hessian calculation. Defaults to True.
    Returns:
        np.ndarray: The Hessian matrix.
    """
    # Turn on create_graph for the first derivative
    if "energyandforcehead" in _calc.predictor.model.module.output_heads:
        if hasattr(
            _calc.predictor.model.module.output_heads["energyandforcehead"], "head"
        ):
            _calc.predictor.model.module.output_heads[
                "energyandforcehead"
            ].head.training = True
        _calc.predictor.model.module.output_heads["energyandforcehead"].training = True
    else:
        _calc.predictor.model.module.output_heads["energy"].training = True
        _calc.predictor.model.module.output_heads["forces"].training = True

    with torch.enable_grad():
        # Convert using the current a2g object
        data_object = _calc.a2g(_atoms)
        data_object.pos.requires_grad = True

        # Batch and predict
        batch = data_list_collater([data_object], otf_graph=True)
        pred = _calc.predictor.predict(batch)

    # Get the forces and positions
    energy = pred["energy"]
    positions = batch["pos"]
    forces = pred["forces"].flatten()

    # Calculate the Hessian using autograd
    if vmap:
        hessian = (
            torch.vmap(
                lambda vec: torch.autograd.grad(
                    -forces,
                    positions,
                    grad_outputs=vec,
                    retain_graph=True,
                )[0],
            )(torch.eye(forces.numel(), device=forces.device)).detach()
            # .cpu()
            # .numpy()
        )
    else:
        hessian = torch.zeros((len(forces), len(forces)), device=forces.device)
        for i in range(len(forces)):
            hessian[:, i] = (
                torch.autograd.grad(
                    -forces[i],
                    positions,
                    retain_graph=True,
                )[0]
                .flatten()
                .detach()
                # .cpu()
                # .numpy()
            )

    # Turn off create_graph for the first derivative
    if "energyandforcehead" in _calc.predictor.model.module.output_heads:
        if hasattr(
            _calc.predictor.model.module.output_heads["energyandforcehead"], "head"
        ):
            _calc.predictor.model.module.output_heads[
                "energyandforcehead"
            ].head.training = False
        _calc.predictor.model.module.output_heads["energyandforcehead"].training = False
    else:
        _calc.predictor.model.module.output_heads["energy"].training = False
        _calc.predictor.model.module.output_heads["forces"].training = False

    return energy, forces, hessian.reshape(len(_atoms) * 3, len(_atoms) * 3)


def get_numerical_hessian(_atoms: Atoms, _calc, eps: float = 1e-4) -> np.ndarray:
    """
    Get the Hessian matrix for the given atomic structure.
    Args:
        atoms (Atoms): The atomic structure to calculate the Hessian for.
        eps (float): The finite difference step size. Defaults to 1e-4.
    Returns:
        np.ndarray: The Hessian matrix.
    """
    # Create displaced atoms in batch
    data_list = []
    for i in range(len(_atoms)):
        for j in range(3):
            displaced_plus = _atoms.copy()
            displaced_minus = _atoms.copy()

            displaced_plus.positions[i, j] += eps
            displaced_minus.positions[i, j] -= eps

            data_plus = _calc.a2g(displaced_plus)
            data_minus = _calc.a2g(displaced_minus)

            data_list.append(data_plus)
            data_list.append(data_minus)

    # Batch and predict
    batch = data_list_collater(data_list, otf_graph=True)
    pred = _calc.predictor.predict(batch)
    energy = pred["energy"]
    # Get the forces
    forces = pred["forces"].reshape(-1, len(_atoms), 3)

    # Calculate the Hessian using finite differences
    hessian = torch.zeros((len(_atoms) * 3, len(_atoms) * 3), device=energy.device)
    for i in range(len(_atoms)):
        for j in range(3):
            idx = i * 3 + j
            force_plus = forces[2 * idx].flatten().detach()
            force_minus = forces[2 * idx + 1].flatten().detach()
            hessian[:, idx] = (force_minus - force_plus) / (2 * eps)

    return energy, forces, hessian.reshape(len(_atoms) * 3, len(_atoms) * 3)


# Latest Mace models
# | Model Name           | Elements Covered | Training Dataset | Level of Theory     | Target System     | Model Size                                                                                                                                                                                                                                                                                                                                                                        | GitHub Release | Notes                                                              | License |
# | -------------------- | ---------------- | ---------------- | ------------------- | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------- | ------------------------------------------------------------------ | ------- |
# | MACE-MP-0a           | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [small](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-10-mace-128-L0_energy_epoch-249.model), [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model), [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0/2024-01-07-mace-128-L2_epoch-199.model) | >=v0.3.6       | Initial release of foundation model.                               | MIT     |
# | MACE-MP-0b3          | 89               | MPTrj            | DFT (PBE+U)         | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mp_0b3/mace-mp-0b3-medium.model)                                                                                                                                                                                                                                                                      | >=v0.3.10      | Improved high pressure stability and reference energies.           | MIT     |
# | MACE-MPA-0           | 89               | MPTrj + sAlex    | DFT (PBE+U)         | Materials         | [medium-mpa-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_mpa_0/mace-mpa-0-medium.model)                                                                                                                                                                                                                                                                  | >=v0.3.10      | Improved accuracy for materials, improved high pressure stability. | MIT     |
# | MACE-OMAT-0          | 89               | OMAT             | DFT (PBE+U) VASP 54 | Materials         | [medium-omat-0](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omat_0/mace-omat-0-medium.model)                                                                                                                                                                                                                                                               | >=v0.3.10      |                                                                    | ASL     |
# | MACE-OFF23           | 10               | SPICE v1         | DFT (wB97M+D3)      | Organic Chemistry | [small](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_small.model), [medium](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_medium.model), [large](https://github.com/ACEsuit/mace-off/blob/main/mace_off23/MACE-OFF23_large.model)                                                                                                    | >=v0.3.6       | Initial release covering neutral organic chemistry.                | ASL     |
# | MACE-MATPES-PBE-0    | 89               | MATPES-PBE       | DFT (PBE)           | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-pbe-omat-ft.model)                                                                                                                                                                                                                                                               | >=v0.3.10      | No +U correction.                                                  | ASL     |
# | MACE-MATPES-r2SCAN-0 | 89               | MATPES-r2SCAN    | DFT (r2SCAN)        | Materials         | [medium](https://github.com/ACEsuit/mace-foundations/releases/download/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model)                                                                                                                                                                                                                                                            | >=v0.3.10      | Better functional for materials.                                   | ASL     |
# | MACE-OMOL-0          | 89               | OMOL             | DFT (wB97M-VV10)    | Molecules/Transition metals/Cations         | [large](https://github.com/ACEsuit/mace-foundations/releases/download/mace_omol_0/MACE-omol-0-extra-large-1024.model)                                                                                                                                                                                                                                                           | >=v0.3.14      | Charge/Spin embedding, very good molecular accuracy.                                   | ASL     |

# latest fairchem models:
# fairchem/src/fairchem/core/calculate/pretrained_models.json
# uma-s-1p1
# uma-m-1p1
# eSEN-sm-direct	All	esen_sm_direct_all.pt
# eSEN-sm-conserving	All	esen_sm_conserving_all.pt
# eSEN-md-direct	All	esen_md_direct_all.pt


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # build a human readable name for the run
    _name = f"{args.model}_{args.ckpt}_{args.dataset}_{args.max_samples}"

    if args.model == "mace_omol":
        from mace.calculators import mace_omol

        calc = mace_omol(model=args.ckpt, device=device, default_dtype="float32")
        print(
            f"Number of parameters in the model: {sum(p.numel() for p in calc.models[0].parameters())}"
        )

        def get_energy_force_hessian(_atoms, _calc):
            batch = _calc._atoms_to_batch(_atoms)
            model = _calc.models[0]
            out = model(
                _calc._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=_calc.use_compile,
            )
            return out["energy"], out["forces"], out["hessian"]

    elif args.model == "mace_off":
        from mace.calculators import mace_off

        calc = mace_off(args.ckpt, device=device, default_dtype="float32")
        print(
            f"Number of parameters in the model: {sum(p.numel() for p in calc.models[0].parameters())}"
        )

        def get_energy_force_hessian(_atoms, _calc):
            batch = _calc._atoms_to_batch(_atoms)
            model = _calc.models[0]
            out = model(
                _calc._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=_calc.use_compile,
            )
            return out["energy"], out["forces"], out["hessian"]

    elif args.model == "uma":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        # uma-s-1p1, uma-m-1p1
        predictor = pretrained_mlip.get_predict_unit(
            f"uma-{args.ckpt}-1p1", device=device
        )
        calc = FAIRChemCalculator(predictor, task_name="omol")
        print(calc.predictor.model.__class__.__name__)
        print(calc.predictor.model)
        print(
            f"Number of parameters in the model: {sum(p.numel() for p in calc.predictor.model.parameters())}"
        )

        def get_energy_force_hessian(_atoms, _calc):
            return _fairchem_ase_get_hessian(_atoms, _calc)

    elif args.model == "esen":
        from fairchem.core import pretrained_mlip, FAIRChemCalculator

        # esen-md-direct-all-omol, esen-sm-conserving-all-omol, esen-sm-direct-all-omol
        predictor = pretrained_mlip.get_predict_unit(
            f"esen-{args.ckpt}-all-omol", device=device
        )
        calc = FAIRChemCalculator(predictor, task_name="omol")
        print(calc.predictor.model.__class__.__name__)
        print(calc.predictor.model)
        print(
            f"Number of parameters in the model: {sum(p.numel() for p in calc.predictor.model.parameters())}"
        )

        def get_energy_force_hessian(_atoms, _calc):
            return _fairchem_ase_get_hessian(_atoms, _calc)
    else:
        raise ValueError(f"Invalid model: {args.model}")

    # test call
    print("\nTesting model forward pass...")
    atoms = Atoms(
        numbers=[1, 6, 7, 8, 9],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    )
    atoms.calc = calc
    energy_model, forces_model, hessian_model = get_energy_force_hessian(atoms, calc)
    print(energy_model.shape, forces_model.shape, hessian_model.shape)
    print()

    wandb.init(
        project="horm",
        name=_name,
        config={**args.__dict__},
    )

    # Create results file path
    results_dir = "results_evalhorm"
    os.makedirs(results_dir, exist_ok=True)
    results_file = f"{results_dir}/{_name}_metrics.csv"

    n_total_samples = None

    # Check if results already exist and redo is False
    if os.path.exists(results_file) and not args.redo:
        print(f"Loading existing results from {results_file}")
        df_results = pd.read_csv(results_file)

    else:
        torch.manual_seed(42)
        np.random.seed(42)

        dataset_dir = os.path.expanduser(
            "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
        )
        dataset_path = os.path.join(dataset_dir, args.dataset)
        dataset = LmdbDataset(dataset_path)
        # dataloader = TGDataLoader(dataset, batch_size=1, shuffle=True)

        # Initialize metrics collection for per-sample DataFrame
        sample_metrics = []
        n_samples = 0

        if args.max_samples is not None:
            n_total_samples = min(args.max_samples, len(dataset))
            np.random.seed(42)
            rnd_idx = np.random.randint(0, len(dataset), n_total_samples)
        else:
            n_total_samples = len(dataset)
            rnd_idx = np.arange(n_total_samples)

        for batch_idx in tqdm(rnd_idx, desc="Eval", total=n_total_samples):
            batch = dataset[batch_idx]
            batch = batch.to(device)

            n_atoms = batch.pos.shape[0]

            # Collect per-sample metrics
            sample_data = {
                "sample_idx": n_samples,
                "natoms": n_atoms,
            }

            # Get ase atoms from batch
            positions = batch.pos.cpu().numpy()
            atomic_numbers = batch.z.cpu().tolist()
            atoms = Atoms(numbers=atomic_numbers, positions=positions)
            atoms.calc = calc

            torch.cuda.reset_peak_memory_stats()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            # Call calculator energy, force, hessian
            energy_model, forces_model, hessian_model = get_energy_force_hessian(
                atoms, calc
            )

            end_event.record()
            torch.cuda.synchronize()

            time_taken = start_event.elapsed_time(end_event)  # ms
            memory_usage = torch.cuda.max_memory_allocated() / 1e6  # Convert to MB
            sample_data["time"] = time_taken  # ms
            sample_data["memory"] = memory_usage

            forces_model = forces_model.reshape(batch.forces.shape)
            hessian_model = hessian_model.reshape(n_atoms * 3, n_atoms * 3)
            hessian_true = batch.hessian.reshape(n_atoms * 3, n_atoms * 3)

            # Compute hessian eigenspectra
            eigvals_model, eigvecs_model = torch.linalg.eigh(hessian_model)

            # Compute errors
            if "energy" in batch:  # RGD1 dataset
                energy_true = batch.energy
            else:  # T1x, QM9 dataset
                energy_true = batch.ae
            e_mae = torch.mean(
                torch.abs(energy_model.squeeze() - energy_true.squeeze())
            )
            e_mae_per_atom = e_mae / n_atoms
            sample_data["energy_mae"] = e_mae.item()
            sample_data["energy_mae_per_atom"] = e_mae_per_atom.item()
            f_mae = torch.mean(torch.abs(forces_model - batch.forces))
            sample_data["forces_mae"] = f_mae.item()

            # Reshape true hessian
            n_atoms = batch.pos.shape[0]
            h_mae = torch.mean(torch.abs(hessian_model - hessian_true))
            sample_data["hessian_mae"] = h_mae.item()

            # Eigenvalue error
            eigvals_true, eigvecs_true = torch.linalg.eigh(hessian_true)

            # Asymmetry error
            asymmetry_mae = torch.mean(torch.abs(hessian_model - hessian_model.T))
            true_asymmetry_mae = torch.mean(torch.abs(hessian_true - hessian_true.T))
            sample_data["asymmetry_mae"] = asymmetry_mae.item()
            sample_data["true_asymmetry_mae"] = true_asymmetry_mae.item()

            # Additional metrics
            eigval_mae = torch.mean(
                torch.abs(eigvals_model - eigvals_true)
            )  # eV/Angstrom^2
            sample_data["eigval_mae"] = eigval_mae.item()

            ########################
            # Mass weighted + Eckart projection
            ########################

            true_freqs = analyze_frequencies_torch(
                hessian=hessian_true,
                cart_coords=batch.pos,
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            true_neg_num = true_freqs["neg_num"].item()
            true_eigvecs_eckart = true_freqs["eigvecs"]
            true_eigvals_eckart = true_freqs["eigvals"]

            freqs_model = analyze_frequencies_torch(
                hessian=hessian_model,
                cart_coords=batch.pos,
                atomsymbols=[Z_TO_ATOM_SYMBOL[z.item()] for z in batch.z],
            )
            freqs_model_neg_num = freqs_model["neg_num"].item()
            eigvecs_model_eckart = freqs_model["eigvecs"]
            eigvals_model_eckart = freqs_model["eigvals"]

            sample_data["true_neg_num"] = true_neg_num
            sample_data["true_is_minima"] = 1 if true_neg_num == 0 else 0
            sample_data["true_is_ts"] = 1 if true_neg_num == 1 else 0
            sample_data["true_is_ts_order2"] = 1 if true_neg_num == 2 else 0
            sample_data["true_is_higher_order"] = 1 if true_neg_num > 2 else 0
            sample_data["model_neg_num"] = freqs_model_neg_num
            sample_data["model_is_ts"] = 1 if freqs_model_neg_num == 1 else 0
            sample_data["model_is_minima"] = 1 if freqs_model_neg_num == 0 else 0
            sample_data["model_is_ts_order2"] = 1 if freqs_model_neg_num == 2 else 0
            sample_data["model_is_higher_order"] = 1 if freqs_model_neg_num > 2 else 0
            sample_data["neg_num_agree"] = (
                1 if (true_neg_num == freqs_model_neg_num) else 0
            )

            sample_data["eigval_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart - true_eigvals_eckart)
            ).item()
            sample_data["eigval1_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[0] - true_eigvals_eckart[0])
            ).item()
            sample_data["eigval2_mae_eckart"] = torch.mean(
                torch.abs(eigvals_model_eckart[1] - true_eigvals_eckart[1])
            ).item()
            sample_data["eigvec1_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 0], true_eigvecs_eckart[:, 0])
            ).item()
            sample_data["eigvec2_cos_eckart"] = torch.abs(
                torch.dot(eigvecs_model_eckart[:, 1], true_eigvecs_eckart[:, 1])
            ).item()

            sample_metrics.append(sample_data)
            n_samples += 1

            # Memory management
            torch.cuda.empty_cache()

            if args.max_samples is not None and n_samples >= args.max_samples:
                break

        # Create DataFrame from collected metrics
        df_results = pd.DataFrame(sample_metrics)

        # Save DataFrame
        df_results.to_csv(results_file, index=False)
        print(f"Saved results to {results_file}")

    # Compute aggregated results by looping over all numeric columns
    aggregated_results = {}
    for col in df_results.columns:
        if pd.api.types.is_numeric_dtype(df_results[col]):
            aggregated_results[col] = df_results[col].mean()
        else:
            print(
                f"Skipping column {col} because it is not numeric: {df_results[col].dtype}"
            )
            continue

    wandb.log(aggregated_results)

    wandb.finish()

    return df_results, aggregated_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model on dataset")
    parser.add_argument(
        "--ckpt",
        "-c",
        type=str,
        default="medium",
        help="Path to checkpoint file or ckpt name the calculator will recognize.",
    )
    parser.add_argument(
        "--model",
        "-ml",
        type=str,
        default="mace_omol",
        choices=["mace_omol", "mace_off", "uma", "esen"],
        help="Model name",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="ts1x-val.lmdb",
        help="Dataset file name (e.g., ts1x-val.lmdb, ts1x_hess_train_big.lmdb, RGD1.lmdb).",
    )
    parser.add_argument(
        "--max_samples",
        "-m",
        type=int,
        default=1000,
        help="Maximum number of samples to evaluate (default: 1000)",
    )
    parser.add_argument(
        "--redo",
        "-r",
        type=bool,
        default=False,
        help="Run eval from scratch even if results already exist",
    )

    args = parser.parse_args()

    torch.manual_seed(42)

    df_results, aggregated_results = run_evaluation(
        args=args,
    )
