from typing import Dict, List, Optional, Tuple
from omegaconf import ListConfig
import yaml
import os
import torch
import warnings
from pathlib import Path
import json

from torch_geometric.data import Batch
from torch_geometric.data import Data as TGData
from torch_geometric.loader import DataLoader as TGDataLoader

from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20
from nets.prediction_utils import compute_extra_props
from ocpmodels.common.relaxation.ase_utils import (
    batch_to_atoms,
    ase_atoms_to_torch_geometric,
)
from ocpmodels.preprocessing import AtomsToGraphs

from ase.calculators.calculator import Calculator
from ase import Atoms

from hip.training_module import PotentialModule, compute_extra_props
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from ocpmodels.hessian_graph_transform import HessianGraphTransform, FOLLOW_BATCH
from hip.training_module import SchemaUniformDataset


def get_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
    ).potential.to(device)
    model.eval()
    model.name = model_name
    return model


def get_dataloader(dataset_name, model, batch_size=1, shuffle=False):
    # Prepare dataset and dataloader
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    dataset = LmdbDataset(fix_dataset_path(dataset_name), transform=transform)
    dataset = SchemaUniformDataset(dataset)
    loader = TGDataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, follow_batch=FOLLOW_BATCH
    )
    return loader



