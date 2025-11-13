from pathlib import Path

import hydra
import pytest
import torch

from torch_geometric.data import Batch as TGBatch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path


def _compose_cfg():
    cfg_dir = "/ssd/Code/hip/configs"
    with hydra.initialize_config_dir(config_dir=cfg_dir, version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=["experiment=debug", "run_name=pytest"],
        )
    # Avoid Hydra interpolation fields that require a Hydra runtime
    cfg.job_name = "results"
    cfg.config_name = "train"
    cfg.override_dirname = ""
    return cfg


def _first_val_batch(cfg):
    val_path = fix_dataset_path(cfg.training.val_path)
    dataset = LmdbDataset(Path(val_path))
    data0 = dataset[0]
    batch = TGBatch.from_data_list([data0])
    return batch


def _checkpoint_path():
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "ckpt/hip_v2.ckpt"
    return ckpt_path


def _mae(pred, target):
    return torch.mean(torch.abs(pred.reshape(-1) - target.reshape(-1)))


@pytest.mark.parametrize("device", ["cpu"])  # GPU covered in a separate test
def test_torch_calculator_cpu_energy_forces_mae(device):
    cfg = _compose_cfg()

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    calc = EquiformerTorchCalculator(
        checkpoint_path=str(ckpt_path),
        hessian_method="predict",
        device=device,
    )

    batch = _first_val_batch(cfg)
    results = calc.predict(batch=batch)

    # Bring tensors to common device for comparison
    energy_pred = results["energy"].squeeze()
    forces_pred = results["forces"]

    # Targets from batch (CPU) -> move to prediction device
    energy_true = batch.ae.to(energy_pred.device).squeeze()
    forces_true = batch.forces.to(forces_pred.device)

    e_mae = _mae(energy_pred, energy_true).item()
    f_mae = _mae(forces_pred, forces_true).item()

    print(f"Torch CPU: Energy MAE: {e_mae:.2e}, Forces MAE: {f_mae:.2e}")
    assert e_mae < 0.5, f"Energy MAE: {e_mae:.2e}"
    assert f_mae < 0.1, f"Forces MAE: {f_mae:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU check")
def test_torch_calculator_gpu_energy_forces_mae():
    cfg = _compose_cfg()

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    device = "cuda"
    calc = EquiformerTorchCalculator(
        checkpoint_path=str(ckpt_path),
        hessian_method="predict",
        device=device,
    )

    batch = _first_val_batch(cfg)
    results = calc.predict(batch=batch)

    energy_pred = results["energy"].squeeze()
    forces_pred = results["forces"]

    energy_true = batch.ae.to(energy_pred.device).squeeze()
    forces_true = batch.forces.to(forces_pred.device)

    e_mae = _mae(energy_pred, energy_true).item()
    f_mae = _mae(forces_pred, forces_true).item()

    print(f"Torch GPU: Energy MAE: {e_mae:.2e}, Forces MAE: {f_mae:.2e}")
    assert e_mae < 0.5, f"Energy MAE: {e_mae:.2e}"
    assert f_mae < 0.1, f"Forces MAE: {f_mae:.2e}"
