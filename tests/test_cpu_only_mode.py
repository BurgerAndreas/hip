from pathlib import Path

import hydra
import pytest

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


def test_torch_calculator_loads_without_gpu(monkeypatch):
    # Hide all GPUs from the process (best-effort; not asserting visibility)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    cfg = _compose_cfg()
    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    # Force CPU explicitly
    calc = EquiformerTorchCalculator(
        checkpoint_path=str(ckpt_path),
        hessian_method="predict",
        device="cpu",
    )

    # Ensure calculator chose CPU
    assert str(calc.device) == "cpu", f"Device should be CPU, got {calc.device}"

    # Predict on the first validation sample
    batch = _first_val_batch(cfg)
    results = calc.predict(batch=batch)

    # Sanity: energy and forces are returned
    assert "energy" in results and "forces" in results
    assert results["forces"].ndim == 2 and results["forces"].shape[1] == 3
