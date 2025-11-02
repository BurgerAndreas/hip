import hydra
import pytest
import torch
from pathlib import Path

from torch_geometric.data import Batch as TGBatch

from hip.training_module import PotentialModule, SchemaUniformDataset
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


def _get_pm_and_batch(cfg):
    # Initialize model/module similarly to training
    pm = PotentialModule(dict(cfg.model), dict(cfg.optimizer), dict(cfg.training))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pm.potential = pm.potential.to(device)

    # First training sample batch
    trn_path = fix_dataset_path(cfg.training.trn_path)
    dataset = SchemaUniformDataset(LmdbDataset(Path(trn_path)))
    data0 = dataset[0]
    batch = TGBatch.from_data_list([data0]).to(device)
    return pm, batch


def _random_rotation(device, dtype):
    # Generate a random rotation matrix with det=+1
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    # Ensure right-handed (det=+1)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for model init"
)
def test_equivariance_energy_forces_hessian():
    cfg = _compose_cfg()
    pm, batch_base = _get_pm_and_batch(cfg)

    # Base forward
    e1, f1, out1 = pm.potential.forward(batch_base, hessian=True, otf_graph=True)
    N = int(batch_base.natoms.sum().item())
    H1 = out1["hessian"].reshape(N * 3, N * 3)

    # Rotated forward
    R = _random_rotation(device=batch_base.pos.device, dtype=batch_base.pos.dtype)
    batch_rot = batch_base.clone()
    batch_rot.pos = batch_rot.pos @ R
    e2, f2, out2 = pm.potential.forward(batch_rot, hessian=True, otf_graph=True)
    H2 = out2["hessian"].reshape(N * 3, N * 3)

    # Energy invariance
    e_err_abs = torch.abs(e1 - e2)
    e_err_rel = e_err_abs / (torch.abs(e1) + 1e-12)
    print(
        f"Energy invariance: abs_err={e_err_abs.item():.2e}, rel_err={e_err_rel.item():.2e}"
    )
    assert torch.allclose(e1, e2, rtol=1e-4, atol=1e-4)

    # Force equivariance: f1 == f2 @ R.T
    f2_rot = f2 @ R.T
    f_err_abs = torch.norm(f1 - f2_rot)
    f_err_rel = f_err_abs / (torch.norm(f1) + 1e-12)
    print(
        f"Force equivariance: abs_err={f_err_abs.item():.2e}, rel_err={f_err_rel.item():.2e}"
    )
    assert torch.allclose(f1, f2 @ R.T, rtol=1e-3, atol=1e-3)

    # Hessian equivariance: H1 == (I kron R) @ H2 @ (I kron R).T
    eye_N = torch.eye(N, device=R.device, dtype=R.dtype)
    # R_big = torch.kron(eye_N, R)
    # Build block rotation without torch.kron to avoid stride/view issues on some builds
    R_big = torch.einsum("ij,ab->iajb", eye_N, R).reshape(N * 3, N * 3)
    H2_transformed = R_big @ H2 @ R_big.T
    H_err_abs = torch.norm(H1 - H2_transformed)
    H_err_rel = H_err_abs / (torch.norm(H1) + 1e-12)
    print(
        f"Hessian equivariance: abs_err={H_err_abs.item():.2e}, rel_err={H_err_rel.item():.2e}"
    )
    assert torch.allclose(H1, R_big @ H2 @ R_big.T, rtol=2e-3, atol=2e-3)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for model init"
)
def test_hessian_symmetry():
    cfg = _compose_cfg()
    pm, batch = _get_pm_and_batch(cfg)
    e, f, out = pm.potential.forward(batch, hessian=True, otf_graph=True)
    N = int(batch.natoms.sum().item())
    H = out["hessian"].reshape(N * 3, N * 3)
    # Symmetry: H == H.T
    H_sym_err_abs = torch.norm(H - H.T)
    H_sym_err_rel = H_sym_err_abs / (torch.norm(H) + 1e-12)
    print(
        f"Hessian symmetry: abs_err={H_sym_err_abs.item():.2e}, rel_err={H_sym_err_rel.item():.2e}"
    )
    assert torch.allclose(H, H.T, rtol=1e-5, atol=1e-5)
