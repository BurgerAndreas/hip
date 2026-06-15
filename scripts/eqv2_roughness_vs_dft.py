#!/usr/bin/env python
"""Does eqv2 force-field roughness predict where autograd Hessians fail vs DFT?

For each HORM sample we compute two independent quantities for the *eqv2* baseline:

ROUGHNESS / self-inconsistency (does the autograd Hessian represent the force field?)
  - fd_plateau:  min over a small grid of step sizes h of
                 ||H_fd(h) - H_autograd||_F / ||H_autograd||_F,
                 where H_fd is the central finite-difference force Jacobian and
                 H_autograd = -dF/dx. For a smooth field this -> 0 as h->0; the plateau
                 value measures roughness of F(x).
  - asym:        ||H_ag - H_ag^T|| / ||H_ag||  (non-conservativeness of the direct forces).

ACCURACY vs DFT (HORM ships the wB97x/6-31G(d) Hessian per sample)
  - hess_rel_err:    ||sym(H_ag) - H_dft||_F / ||H_dft||_F
  - eigval_mae:      MAE of raw-Cartesian eigenvalues (sym(H_ag) vs H_dft)
  - vib_eigval_mae:  MAE of mass-weighted, Eckart-projected vibrational eigenvalues
  - neg_match:       does the autograd Hessian reproduce the DFT negative-mode count?
  - force_rel_err:   ||F - F_dft|| / ||F_dft||   (accuracy control)

Hypothesis (causal): rougher samples (larger fd_plateau / asym) have larger autograd-Hessian
error vs DFT. Output is a per-sample table; plot with plotting/plot_eqv2_roughness_vs_dft.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.ff_lmdb import LmdbDataset, Z_TO_ATOM_SYMBOL
from hip.frequency_analysis import analyze_frequencies_np
from hip.path_config import fix_dataset_path

# reuse the validated force-field / FD / autograd helpers (scripts/ is on sys.path[0] when
# this file is run as `python scripts/eqv2_roughness_vs_dft.py`).
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from force_smoothness_scan import (  # noqa: E402
    ForceField,
    autograd_hessian,
    fd_full_jacobian,
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel_fro(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc = EquiformerTorchCalculator(
        checkpoint_path=str(args.checkpoint), hessian_method="autograd", device=device
    )
    potential = calc.potential

    dataset = LmdbDataset(fix_dataset_path(str(args.dataset)))
    n = len(dataset) if args.max_samples is None else min(args.max_samples, len(dataset))
    h_grid = [float(x) for x in args.h_grid.split(",")]

    rows = []
    for i in range(n):
        data = dataset[i]
        z = data.z.to(torch.int64)
        n_atoms = z.numel()
        n3 = 3 * n_atoms
        base = data.pos.reshape(-1).to(torch.float64).clone()
        symbols = [Z_TO_ATOM_SYMBOL[int(zz)] for zz in z]

        field = ForceField(potential, z, device, dtype)
        e0, f0 = field(base)
        fmax = float(f0.abs().max())

        # --- autograd Hessian (force Jacobian) ---
        H_ag = autograd_hessian(potential, base, z, device, dtype).numpy()  # (n3, n3), CPU
        H_ag_sym = 0.5 * (H_ag + H_ag.T)
        asym = float(np.linalg.norm(H_ag - H_ag.T) / (np.linalg.norm(H_ag) + 1e-30))

        # --- roughness: FD self-inconsistency plateau ---
        fd_errs = []
        for h in h_grid:
            H_fd = fd_full_jacobian(field, base, h).numpy()
            fd_errs.append(_rel_fro(H_fd, H_ag))
        fd_errs = np.array(fd_errs)
        j = int(fd_errs.argmin())
        fd_plateau = float(fd_errs[j])
        fd_plateau_h = float(h_grid[j])

        # --- DFT reference Hessian (eV/Ang^2) ---
        H_dft = data.hessian.reshape(n3, n3).to(torch.float64).numpy()
        hess_rel_err = _rel_fro(H_ag_sym, H_dft)
        hess_mae = float(np.mean(np.abs(H_ag_sym - H_dft)))

        evals_ag = np.linalg.eigvalsh(H_ag_sym)
        evals_dft = np.linalg.eigvalsh(H_dft)
        eigval_mae = float(np.mean(np.abs(np.sort(evals_ag) - np.sort(evals_dft))))

        # --- force accuracy control ---
        F_dft = data.forces.reshape(-1).to(torch.float64).numpy()
        F_model = f0.numpy()
        force_rel_err = _rel_fro(F_model, F_dft)

        # --- vibrational (mass-weighted + Eckart) ---
        coords_np = base.reshape(-1, 3).numpy()
        vib_model = analyze_frequencies_np(H_ag_sym, coords_np, symbols)
        vib_true = analyze_frequencies_np(H_dft, coords_np, symbols)
        neg_model = int(vib_model["neg_num"])
        neg_true = int(vib_true["neg_num"])
        vib_eigval_mae = float(
            np.mean(np.abs(np.sort(vib_model["eigvals"]) - np.sort(vib_true["eigvals"])))
        )

        rows.append(
            dict(
                idx=i,
                natoms=n_atoms,
                fmax=fmax,
                asym=asym,
                fd_plateau=fd_plateau,
                fd_plateau_h=fd_plateau_h,
                hess_rel_err=hess_rel_err,
                hess_mae=hess_mae,
                eigval_mae=eigval_mae,
                vib_eigval_mae=vib_eigval_mae,
                force_rel_err=force_rel_err,
                neg_model=neg_model,
                neg_true=neg_true,
                neg_match=int(neg_model == neg_true),
            )
        )
        print(
            f"[{i + 1}/{n}] N={n_atoms:2d} fd_plateau={fd_plateau:.3f} asym={asym:.3f} "
            f"hess_rel_err={hess_rel_err:.3f} neg {neg_model}/{neg_true}",
            flush=True,
        )

    df = pd.DataFrame(rows)
    csv = out_dir / f"{args.tag}.csv"
    df.to_csv(csv, index=False)
    df.to_parquet(out_dir / f"{args.tag}.parquet", index=False)

    # quick correlation print (Spearman is robust to the heavy tails)
    def _spearman(a, b):
        ra = pd.Series(a).rank()
        rb = pd.Series(b).rank()
        return float(np.corrcoef(ra, rb)[0, 1])

    print("\n=== summary over", len(df), "samples ===")
    for rough in ["fd_plateau", "asym"]:
        for acc in ["hess_rel_err", "eigval_mae", "vib_eigval_mae"]:
            print(f"  Spearman({rough}, {acc}) = {_spearman(df[rough], df[acc]):+.3f}")
    print(f"  neg-mode agreement: {df['neg_match'].mean() * 100:.1f}%")
    print(f"\nWrote {csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=_project_root() / "ckpt" / "eqv2.ckpt")
    p.add_argument("--dataset", type=str, default="data/sample_100.lmdb")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--h-grid", type=str, default="1e-4,3e-4,1e-3,3e-3")
    p.add_argument("--output-dir", type=Path, default=_project_root() / "runs" / "eqv2_roughness_vs_dft")
    p.add_argument("--tag", default="eqv2_sample100")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
