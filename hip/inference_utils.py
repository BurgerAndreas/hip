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


# TODO: deprecated
def get_model(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    # model_config["otf_graph"] = False
    print("model_config", model_config)
    return EquiformerV2_OC20(**model_config), model_config


# TODO: deprecated
def get_model_and_dataloader_for_hessian_prediction(
    batch_size,
    shuffle,
    device,
    dataset_path=None,
    config_path=None,
    checkpoint_path=None,
    dataloader_kwargs={},
):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    # Model
    if config_path is None:
        config_path = os.path.join(project_root, "configs/equiformer_v2.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    model_config = config["model"]
    model_config["do_hessian"] = True
    model_config["otf_graph"] = False
    model = EquiformerV2_OC20(**model_config)
    # Checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(project_root, "ckpt/eqv2.ckpt")
    state_dict = torch.load(checkpoint_path, weights_only=True)["state_dict"]
    state_dict = {k.replace("potential.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.train()
    model.to(device)
    # Dataset
    transform = HessianGraphTransform(
        cutoff=model.cutoff,
        cutoff_hessian=model.cutoff_hessian,
        max_neighbors=model.max_neighbors,
        use_pbc=model.use_pbc,
    )
    if dataset_path is None:
        dataset_path = os.path.join(project_root, "data/sample_100.lmdb")
    dataset = LmdbDataset(dataset_path, transform=transform)
    # Dataloader
    follow_batch = ["diag_ij", "edge_index", "message_idx_ij"]
    dataloader = TGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=follow_batch,
        **dataloader_kwargs,
    )
    return model, dataloader
