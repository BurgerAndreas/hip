from pathlib import Path

import hydra
import numpy as np
import pytest
import torch

from ase import Atoms

from torch_geometric.data import Batch as TGBatch

from hip.equiformer_ase_calculator import EquiformerASECalculator
from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.inference_utils import get_model_from_checkpoint
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


def _to_atoms(batch: TGBatch) -> Atoms:
    pos = batch.pos.detach().cpu().numpy()
    if hasattr(batch, "z"):
        symbols = [Z_TO_ATOM_SYMBOL[int(z)] for z in batch.z.tolist()]
    else:
        raise RuntimeError("Batch missing atomic numbers 'z'")
    atoms = Atoms(symbols=symbols, positions=pos)
    return atoms


def _checkpoint_path():
    project_root = Path(__file__).resolve().parents[1]
    ckpt_path = project_root / "ckpt/hip_v2.ckpt"
    return ckpt_path


def _mae_torch(pred_t, true_t):
    return torch.mean(torch.abs(pred_t - true_t))


def _conservative_model(ckpt_path, device):
    model = get_model_from_checkpoint(str(ckpt_path), device)
    model.direct_forces = False
    model.grad_forces = True
    model.eval()
    return model


@pytest.mark.parametrize("device", ["cpu"])  # GPU covered in a separate test
def test_ase_calculator_cpu_energy_forces_mae(device):
    cfg = _compose_cfg()

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    batch = _first_val_batch(cfg)
    atoms = _to_atoms(batch)

    ase_calc = EquiformerASECalculator(
        checkpoint_path=str(ckpt_path),
        hessian_method="predict",
        device=device,
    )
    atoms.calc = ase_calc

    ase_calc.calculate(atoms)
    results = ase_calc.results

    energy_pred = torch.tensor(results["energy"], dtype=batch.ae.dtype)
    forces_pred = torch.tensor(results["forces"], dtype=batch.forces.dtype)

    energy_true = batch.ae.detach().cpu().squeeze()
    forces_true = batch.forces.detach().cpu()

    e_mae = _mae_torch(energy_pred, energy_true).item()
    f_mae = _mae_torch(forces_pred, forces_true).item()

    print(f"ASE CPU: Energy MAE: {e_mae:.2e}, Forces MAE: {f_mae:.2e}")
    assert e_mae < 0.5, f"Energy MAE: {e_mae:.2e}"
    assert f_mae < 0.1, f"Forces MAE: {f_mae:.2e}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU check")
def test_ase_calculator_gpu_energy_forces_mae():
    cfg = _compose_cfg()

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    batch = _first_val_batch(cfg)
    atoms = _to_atoms(batch)

    ase_calc = EquiformerASECalculator(
        checkpoint_path=str(ckpt_path),
        hessian_method="predict",
        device="cuda",
    )
    atoms.calc = ase_calc

    ase_calc.calculate(atoms)
    results = ase_calc.results

    energy_pred = torch.tensor(results["energy"], dtype=batch.ae.dtype)
    forces_pred = torch.tensor(results["forces"], dtype=batch.forces.dtype)

    energy_true = batch.ae.detach().cpu().squeeze()
    forces_true = batch.forces.detach().cpu()

    e_mae = _mae_torch(energy_pred, energy_true).item()
    f_mae = _mae_torch(forces_pred, forces_true).item()

    print(f"ASE GPU: Energy MAE: {e_mae:.2e}, Forces MAE: {f_mae:.2e}")
    assert e_mae < 0.5, f"Energy MAE: {e_mae:.2e}"
    assert f_mae < 0.1, f"Forces MAE: {f_mae:.2e}"


@pytest.mark.parametrize("device", ["cpu"])
def test_ase_calculator_conservative_forces_work_under_no_grad(device):
    cfg = _compose_cfg()

    ckpt_path = _checkpoint_path()
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    batch = _first_val_batch(cfg)
    atoms = _to_atoms(batch)

    ase_calc = EquiformerASECalculator(
        model=_conservative_model(ckpt_path, device),
        hessian_method="predict",
        device=device,
    )
    atoms.calc = ase_calc

    ase_calc.calculate(atoms)
    reference = {
        "energy": ase_calc.results["energy"],
        "forces": np.array(ase_calc.results["forces"], copy=True),
    }

    with torch.no_grad():
        ase_calc.calculate(atoms)
    under_no_grad = ase_calc.results

    assert np.allclose(reference["energy"], under_no_grad["energy"])
    assert np.allclose(reference["forces"], under_no_grad["forces"])
