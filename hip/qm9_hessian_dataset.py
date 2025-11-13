"""
Dataset adapter for QM9 Hessian dataset from HuggingFace.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from hip.ff_lmdb import GLOBAL_ATOM_NUMBERS, onehot_convert


class QM9HessianDataset(Dataset):
    """
    Dataset adapter for QM9 Hessian dataset stored in HuggingFace format.
    
    Converts HuggingFace dataset entries to PyTorch Geometric Data objects
    compatible with the training module.
    
    Args:
        dataset_path: Path to the HuggingFace dataset directory or dataset name
        split: Dataset split name (e.g., 'vacuum_train', 'thf_test')
        transform: Optional transform to apply to data objects
    """
    
    def __init__(
        self,
        dataset_path: str,
        split: Optional[str] = None,
        transform=None,
        **kwargs
    ):
        super(QM9HessianDataset, self).__init__()
        
        from datasets import load_from_disk, DatasetDict
        
        # Handle path with split specification: "path:split_name"
        if ":" in dataset_path:
            dataset_path, split = dataset_path.rsplit(":", 1)
        
        # Load the dataset
        dataset = load_from_disk(dataset_path)
        
        # Handle DatasetDict vs Dataset
        if isinstance(dataset, DatasetDict):
            if split is None:
                # If no split specified, try to use the first available split
                available_splits = list(dataset.keys())
                if len(available_splits) == 1:
                    split = available_splits[0]
                    print(f"No split specified, using: {split}")
                else:
                    raise ValueError(
                        "split must be specified when loading a DatasetDict. "
                        f"Available splits: {available_splits}"
                    )
            if split not in dataset:
                raise ValueError(
                    f"Split '{split}' not found in dataset. "
                    f"Available splits: {list(dataset.keys())}"
                )
            self.dataset = dataset[split]
        else:
            # If it's already a Dataset, use it directly
            self.dataset = dataset
        
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
        pos = torch.from_numpy(positions)
        z = torch.from_numpy(atomic_numbers)
        forces = torch.from_numpy(forces)
        hessian = torch.from_numpy(hessian)
        
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
            ae=ae,
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
    # Example usage:
    # dataset = QM9HessianDataset(
    #     dataset_path="/ssd/Code/Datastore/qm9hessian/hessian_qm9_DatasetDict:vacuum_train"
    # )
    # print(f"Dataset length: {len(dataset)}")
    # sample = dataset[0]
    # print(f"Sample keys: {sample.keys}")
    # print(f"Positions shape: {sample.pos.shape}")
    # print(f"Atomic numbers: {sample.z}")
    # print(f"Hessian shape: {sample.hessian.shape}")
    pass

