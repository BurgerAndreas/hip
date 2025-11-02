from pathlib import Path

import pytest
import torch
import hydra

from torch_geometric.data import Batch as TGBatch

from scripts.train import setup_training
from hip.ff_lmdb import LmdbDataset
from hip.training_module import SchemaUniformDataset
from hip.path_config import fix_dataset_path


def _compose_cfgs():
    cfg_dir = "/ssd/Code/hip/configs"
    # Compose both configs within a single Hydra context for cleanliness
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg_small = hydra.compose(
            config_name="train",
            overrides=["experiment=debug", "model.max_radius=3.0", "run_name=pytest"],
        )
        cfg_large = hydra.compose(
            config_name="train",
            overrides=["experiment=debug", "model.max_radius=100.0", "run_name=pytest"],
        )
    # Avoid Hydra interpolation fields that require a Hydra runtime
    for _cfg in (cfg_small, cfg_large):
        _cfg.job_name = "results"
        _cfg.config_name = "train"
        _cfg.override_dirname = ""
    return cfg_small, cfg_large


def _get_first_sample_batch(cfg):
    trn_path = fix_dataset_path(cfg.training.trn_path)
    dataset = SchemaUniformDataset(LmdbDataset(Path(trn_path)))
    data0 = dataset[0]
    batch = TGBatch.from_data_list([data0])
    return batch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for model init")
def test_hessian_graph_shrinks_with_smaller_radius():
    cfg_small, cfg_large = _compose_cfgs()

    # Build modules/models as in training
    _, pm_small = setup_training(cfg_small)
    _, pm_large = setup_training(cfg_large)

    # Sanity: cutoff follows max_radius
    assert pm_small.potential.cutoff == pytest.approx(3.0)
    assert pm_large.potential.cutoff == pytest.approx(100.0)

    # Use the exact same first sample for both runs
    batch_small = _get_first_sample_batch(cfg_small)
    batch_large = _get_first_sample_batch(cfg_large)

    device = "cuda"
    # Ensure model and batch are on the same device
    pm_small.potential = pm_small.potential.to(device)
    pm_large.potential = pm_large.potential.to(device)

    # Forward pass (builds graphs on the fly)
    _ = pm_small.potential.forward(batch_small.to(device), hessian=True, otf_graph=True)
    _ = pm_large.potential.forward(batch_large.to(device), hessian=True, otf_graph=True)

    # Compare Hessian graph sizes for the first (and only) sample
    nedges_small = int(batch_small.nedges_hessian[0].item())
    nedges_large = int(batch_large.nedges_hessian[0].item())

    assert nedges_small <= nedges_large
    assert (nedges_large - nedges_small) > 0


