"""Small Transition1x wrapper for scan scripts.

This keeps the HIP repo independent of GAD+ while preserving the convenient
sample attributes used by the glycine proton-transfer scan.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transition1x import Dataloader as T1xDataloader


class Transition1xDataset(Dataset):
    """Load final Transition1x reactant/TS/product geometries."""

    def __init__(
        self,
        h5_path: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        transform=None,
    ):
        self.transform = transform
        self.samples: list[Data] = []
        loader = T1xDataloader(h5_path, datasplit=split, only_final=True)

        for idx, mol in enumerate(loader):
            if max_samples is not None and len(self.samples) >= max_samples:
                break
            try:
                ts = mol["transition_state"]
                reactant = mol["reactant"]

                if len(ts["atomic_numbers"]) != len(reactant["atomic_numbers"]):
                    continue

                product = mol.get("product")
                has_product = (
                    product is not None
                    and len(product.get("atomic_numbers", [])) == len(ts["atomic_numbers"])
                )
                if has_product:
                    pos_product = torch.tensor(product["positions"], dtype=torch.float)
                else:
                    pos_product = torch.zeros_like(
                        torch.tensor(ts["positions"], dtype=torch.float)
                    )

                self.samples.append(
                    Data(
                        z=torch.tensor(ts["atomic_numbers"], dtype=torch.long),
                        pos_transition=torch.tensor(ts["positions"], dtype=torch.float),
                        pos_reactant=torch.tensor(reactant["positions"], dtype=torch.float),
                        pos_product=pos_product,
                        has_product=torch.tensor([has_product], dtype=torch.bool),
                        energy=torch.tensor(ts["wB97x_6-31G(d).energy"], dtype=torch.float),
                        forces=torch.tensor(ts["wB97x_6-31G(d).forces"], dtype=torch.float),
                        rxn=ts["rxn"],
                        formula=ts["formula"],
                    )
                )
            except Exception as exc:
                print(f"[WARN] Skipping idx={idx}: {exc}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        data = self.samples[idx]
        if self.transform is not None:
            data = self.transform(data)
        return data
