"""
Dataset adapter for QM9 Hessian dataset from HuggingFace.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

from datasets import load_from_disk, DatasetDict
from datasets import Dataset as HuggingFaceDataset
from hip.ff_lmdb import GLOBAL_ATOM_NUMBERS, onehot_convert


class QM9HessianDataset(TorchDataset):
    """
    Dataset adapter for QM9 Hessian dataset stored in HuggingFace format.

    Converts HuggingFace dataset entries to PyTorch Geometric Data objects
    compatible with the training module.

    Args:
        dataset_path: Path to the HuggingFace dataset directory or dataset name
        split: Dataset split name (None for all data, "train" for train split, "test" for test split)
        transform: Optional transform to apply to data objects
    """

    def __init__(
        self, dataset_path: str, split: Optional[str] = None, transform=None, keep_fluorine: bool = True, **kwargs
    ):
        super(QM9HessianDataset, self).__init__()

        assert split in [None, "train", "test"], f"Invalid split: {split}"

        # Handle path with split specification: "path:solvent"
        if ":" in dataset_path:
            dataset_path, solvent = dataset_path.rsplit(":", 1)
        else:
            solvent = "vacuum"

        # Load the dataset
        dataset: DatasetDict = load_from_disk(dataset_path)
        dataset: HuggingFaceDataset = dataset[solvent]

        # Filter out samples containing Fluorine (z=9)
        if keep_fluorine is False:
            dataset = dataset.filter(
                lambda x: 9 not in x["atomic_numbers"],
                num_proc=8, # use 8 processes to filter the dataset
                # batched=True,
                # batch_size=1000,
                # keep_in_memory=True,
            )

        if split is not None:
            split_ds = dataset.train_test_split(test_size=0.1, seed=42)
            dataset = split_ds[split]
        self.dataset: HuggingFaceDataset = dataset

        self.transform = transform
        self.num_samples = len(self.dataset)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx >= self.num_samples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.num_samples} samples"
            )

        # Get sample from HuggingFace dataset
        sample = self.dataset[idx]

        # Extract data
        positions = np.array(sample["positions"], dtype=np.float32)
        atomic_numbers = np.array(sample["atomic_numbers"], dtype=np.int64)
        energy = float(sample["energy"])
        forces = np.array(sample["forces"], dtype=np.float32)
        hessian = np.array(sample["hessian"], dtype=np.float32)

        # Convert to tensors
        pos = torch.from_numpy(positions)  # [n_atoms, 3]
        z = torch.from_numpy(atomic_numbers)
        forces = torch.from_numpy(forces)  # [n_atoms, 3]
        hessian = torch.from_numpy(hessian).reshape(-1)  # [n_atoms * 3 * n_atoms * 3]

        # Create one-hot encoding
        atomic_numbers_list = atomic_numbers.tolist()
        one_hot = onehot_convert(atomic_numbers_list, device=torch.device("cpu"))

        # Number of atoms
        natoms = torch.tensor(len(atomic_numbers), dtype=torch.int64)

        # Atomization energy (using total energy for now)
        # For QM9, atomization energy would be: E_total - sum(E_atomic)
        # But we'll use total energy as ae for compatibility
        ae = torch.tensor(energy, dtype=torch.float32)

        # Create PyTorch Geometric Data object
        data = Data(
            pos=pos,
            z=z,
            one_hot=one_hot,
            ae=ae,  # atomization energy
            forces=forces,
            hessian=hessian,
            natoms=natoms,
            dataset_idx=torch.tensor(idx, dtype=torch.int64),
        )

        # Apply transform if provided
        if self.transform is not None:
            data = self.transform(data)

        return data


if __name__ == "__main__":
    # pip install datasets (from HuggingFace)
    from datasets import load_from_disk, DatasetDict
    from hip.path_config import DATASET_DIR_QM9HESSIAN, fix_dataset_path

    # https://figshare.com/articles/dataset/_b_Hessian_QM9_Dataset_b_/26363959?file=49271011
    dataset = load_from_disk("/ssd/Code/Datastore/qm9hessian/hessian_qm9_DatasetDict")
    print("Original dataset:")
    print(dataset)

    # Split each dataset into train and test
    split_dataset = DatasetDict()
    for split_name, ds in dataset.items():
        split_ds = ds.train_test_split(test_size=0.1, seed=42)
        split_dataset[f"{split_name}_train"] = split_ds["train"]
        split_dataset[f"{split_name}_test"] = split_ds["test"]
        print(f"\n{split_name}:")
        print(f"  Train: {len(split_ds['train'])} samples")
        print(f"  Test: {len(split_ds['test'])} samples")

    print("\nSplit dataset structure:")
    print(split_dataset)
    # DatasetDict({
    #     vacuum: Dataset({
    #         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
    #         num_rows: 41645
    #     })
    #     thf: Dataset({
    #         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
    #         num_rows: 41645
    #     })
    #     toluene: Dataset({
    #         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
    #         num_rows: 41645
    #     })
    #     water: Dataset({
    #         features: ['energy', 'positions', 'atomic_numbers', 'forces', 'frequencies', 'normal_modes', 'hessian', 'label'],
    #         num_rows: 41645
    #     })
    # })

    #########################################################################################################
    # Example usage:
    dataset = QM9HessianDataset(
        # Available splits: ['vacuum', 'thf', 'toluene', 'water']
        dataset_path="/ssd/Code/Datastore/qm9hessian/hessian_qm9_DatasetDict:vacuum",
        split="train",
    )
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]

    for k, v in sample.items():
        print(k, v.shape)
