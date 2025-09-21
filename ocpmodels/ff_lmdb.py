"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import bisect
import pickle
from pathlib import Path

import lmdb
import numpy as np
from torch.utils.data import Dataset

# from torch_geometric.data import Batch


class LmdbDataset(Dataset):
    r"""Dataset class to load from LMDB files containing relaxation
    trajectories or single point computations.

    Useful for Structure to Energy & Force (S2EF), Initial State to
    Relaxed State (IS2RS), and Initial State to Relaxed Energy (IS2RE) tasks.

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


def remove_hessian_transform(data):
    # Remove 'hessian' if present as attribute
    if hasattr(data, "hessian"):
        delattr(data, "hessian")
    # Remove 'hessian' if present as key (for dict-like)
    # if isinstance(data, dict) and 'hessian' in data:
    #     del data['hessian']
    return data


def fix_hessian_eigen_transform(data):
    """Fixes errors in old versions of the code.
    Only needed for legacy compatibility.
    You can probably remove this function.

    Sizes of tensors must match except in dimension 0.

    Hessian eigen information is stored as:
    hessian_eigenvalues: torch.Size([2])
    hessian_eigenvectors: torch.Size([2, N*3])

    Instead save as:
    hessian_eigenvalue_1: torch.Size([1])
    hessian_eigenvalue_2: torch.Size([1])
    hessian_eigenvector_1: torch.Size([N, 3])
    hessian_eigenvector_2: torch.Size([N, 3])
    """
    # Check if hessian eigenvalue/eigenvector data exists
    if hasattr(data, "hessian_eigenvalues") and hasattr(data, "hessian_eigenvectors"):
        eigenvalues = data.hessian_eigenvalues
        eigenvectors = data.hessian_eigenvectors

        # Split eigenvalues into separate attributes
        data.hessian_eigenvalue_1 = eigenvalues[0:1]  # Keep as [1] tensor
        data.hessian_eigenvalue_2 = eigenvalues[1:2]  # Keep as [1] tensor

        # Reshape and split eigenvectors from [2, N*3] to [N, 3] format
        n_atoms = len(data.pos)  # Get number of atoms from positions
        eigenvector_1 = eigenvectors[0].reshape(n_atoms, 3)
        eigenvector_2 = eigenvectors[1].reshape(n_atoms, 3)

        data.hessian_eigenvector_1 = eigenvector_1
        data.hessian_eigenvector_2 = eigenvector_2

        # Remove original attributes
        delattr(data, "hessian_eigenvalues")
        delattr(data, "hessian_eigenvectors")

    # Remove batch artifacts manually
    for key in ["batch", "ptr"]:
        if hasattr(data, key):
            delattr(data, key)

    return data


if __name__ == "__main__":
    import os

    dataset_dir = os.path.expanduser(
        "~/.cache/kagglehub/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm/versions/5/"
    )
    dataset_files = [
        "ts1x-val.lmdb",
        "ts1x_hess_train_big.lmdb",
        "RGD1.lmdb",
    ]
    lmdb_path = os.path.join(dataset_dir, dataset_files[0])
    lmdb_dataset = LmdbDataset(lmdb_path)
    print("length of lmdb_dataset:", len(lmdb_dataset))
    print("first element of lmdb_dataset:", lmdb_dataset[0])
    print("first element of lmdb_dataset.pos:", lmdb_dataset[0].pos)
    print("first element of lmdb_dataset.ae:", lmdb_dataset[0].ae)
    first_elem = lmdb_dataset[0]
    print("")
    print("hasattr(first_elem, 'hessian'):", hasattr(first_elem, "hessian"))
    print("'hessian' in first_elem:", "hessian" in first_elem)

    # Test with transform that removes hessian
    lmdb_dataset_no_hessian = LmdbDataset(lmdb_path, transform=remove_hessian_transform)
    first_elem = lmdb_dataset_no_hessian[0]
    print("")
    print("hasattr(first_elem, 'hessian'):", hasattr(first_elem, "hessian"))
    print("'hessian' in first_elem:", "hessian" in first_elem)

    for fname in dataset_files:
        path = os.path.join(dataset_dir, fname)
        ds = LmdbDataset(path)
        print(f"Size of {fname}: {len(ds)}")
