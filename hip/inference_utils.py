import torch
from torch_geometric.loader import DataLoader as TGDataLoader

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path
from hip.training_module import SchemaUniformDataset


def get_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
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
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    dataset = SchemaUniformDataset(dataset)
    loader = TGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
