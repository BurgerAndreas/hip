from pathlib import Path
import shutil

import torch
from torch_geometric.loader import DataLoader as TGDataLoader

from hip.training_module import PotentialModule
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path


HIP_CHECKPOINT_REPO_ID = "andreasburger/hip"
HIP_AUTO_DOWNLOAD_CHECKPOINTS = {
    "hip_v2.ckpt": "ckpt/hip_v2.ckpt",
    "hip_v3.ckpt": "ckpt/hip_v3.ckpt",
}


def resolve_checkpoint_path(checkpoint_path):
    if checkpoint_path is None:
        return checkpoint_path

    path = Path(checkpoint_path).expanduser()
    if path.exists() or path.name not in HIP_AUTO_DOWNLOAD_CHECKPOINTS:
        return str(path)

    from huggingface_hub import hf_hub_download

    path.parent.mkdir(parents=True, exist_ok=True)
    cached_path = hf_hub_download(
        repo_id=HIP_CHECKPOINT_REPO_ID,
        filename=HIP_AUTO_DOWNLOAD_CHECKPOINTS[path.name],
    )
    shutil.copyfile(cached_path, path)
    return str(path)


def get_model_from_checkpoint(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    model_name = ckpt["hyper_parameters"]["model_config"]["name"]
    model = PotentialModule.load_from_checkpoint(
        checkpoint_path,
        strict=False,
        map_location=device,
        weights_only=False,
    ).potential
    model.eval()
    model.name = model_name
    return model


def get_dataloader(dataset_name, model, batch_size=1, shuffle=False):
    # Prepare dataset and dataloader
    dataset = LmdbDataset(fix_dataset_path(dataset_name))
    loader = TGDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader
