#!/usr/bin/env python
"""Probe the smoothness of MLIP forces and the reliability of autograd Hessians.

Motivation
----------
HIP's MLIP (e.g. ``hip_v3``) predicts *direct*, non-conservative forces ``F(x)``.
The autograd Hessian used as a baseline is computed by differentiating those
forces, so (see ``hip/hessian_utils.compute_hessian``)::

    H_autograd[i, j] = - dF_i / dx_j

i.e. the autograd Hessian is (minus) the Jacobian of the force field. Its quality
is therefore entirely controlled by how *smooth* ``F(x)`` is. If the learned force
field is rough/wiggly at small length scales, its Jacobian (the autograd Hessian)
is noisy even though forces themselves look fine.

This script runs two analyses on a single geometry taken from Transition1x
(the same geometries as HORM T1x, just unscrambled):

1. 1D line scan + spectrum
   Displace the geometry along a direction ``d``: ``x(λ) = x0 + λ d``, evaluate
   ``F(x(λ))`` on a fine grid, and look at:
     - the directional (generalized) force ``g(λ) = d · F``
     - the energy ``E(λ)`` and the non-conservativeness check ``-dE/dλ`` vs ``g``
     - the power spectrum of ``g(λ)`` (FFT) -> high-frequency content == roughness.

2. Finite-difference convergence sweep
   Compare the autograd Hessian (force Jacobian) against central finite-difference
   force derivatives over a range of step sizes ``h``:
     - directional:   FD ``H·d ≈ -(F(x+h d) - F(x-h d)) / (2h)``   vs   ``H_autograd · d``
     - full Jacobian: ``H_fd[:, j] = -(F(x+h e_j) - F(x-h e_j)) / (2h)``
   For a smooth field the error decreases as O(h^2) until a numerical noise floor
   (~ eps * |F| / h) is hit -> classic U-shape. The location/height of the upturn
   measures the roughness/noise length scale of the force field.

Outputs: PNG figures, an .npz with all raw arrays, and a printed summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Data as TGData

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.hessian_utils import compute_hessian
from hip.transition1x_dataset import Transition1xDataset
from nets.equiformer_v2.equiformer_v2_oc20 import center_batch_positions


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_batch(coords: torch.Tensor, z: torch.Tensor, dtype: torch.dtype):
    """Single-molecule torch_geometric batch with a chosen float dtype."""
    data = TGData(
        pos=coords.reshape(-1, 3).to(dtype),
        z=z.reshape(-1).to(torch.int64),
        charges=z.reshape(-1).to(torch.int64),
        natoms=torch.tensor([z.numel()], dtype=torch.int64),
        cell=None,
        pbc=torch.tensor(False, dtype=torch.bool),
    )
    return TGBatch.from_data_list([data])


class ForceField:
    """Thin wrapper that returns (energy, flat-forces) for a geometry."""

    def __init__(self, potential, z, device, dtype):
        self.potential = potential
        self.z = z.to(device)
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(self, coords_flat: torch.Tensor):
        coords = coords_flat.reshape(-1, 3).to(self.device, self.dtype)
        batch = _build_batch(coords, self.z, self.dtype).to(self.device)
        batch = center_batch_positions(batch)
        energy, forces, _ = self.potential.forward(
            batch, otf_graph=True, hessian=False
        )
        return (
            float(energy.reshape(-1)[0].item()),
            forces.detach().reshape(-1).to(torch.float64).cpu(),
        )


def autograd_hessian(potential, coords_flat, z, device, dtype) -> torch.Tensor:
    """H[i,j] = -dF_i/dx_j via autograd (the calculator's `autograd` baseline)."""
    coords = coords_flat.reshape(-1, 3).to(device, dtype).clone()
    batch = _build_batch(coords, z.to(device), dtype).to(device)
    batch = center_batch_positions(batch)
    with torch.enable_grad():
        batch.pos.requires_grad_(True)
        energy, forces, _ = potential.forward(batch, otf_graph=True, hessian=False)
        hess = compute_hessian(coords=batch.pos, energy=energy, forces=forces)
    # return on CPU so all orchestration tensors (base, directions, H_ag) share a device;
    # force evaluations are moved back onto `device` inside ForceField.__call__.
    return hess.detach().to(torch.float64).cpu()


def fd_directional(field: ForceField, base: torch.Tensor, d: torch.Tensor, h: float):
    """Central FD estimate of H·d = -(F(x+h d) - F(x-h d)) / (2h)."""
    _, fp = field(base + h * d)
    _, fm = field(base - h * d)
    return -(fp - fm) / (2.0 * h)


def fd_full_jacobian(field: ForceField, base: torch.Tensor, h: float) -> torch.Tensor:
    """Full force-Jacobian Hessian via central differences (n = 3N columns)."""
    n = base.numel()
    H = torch.zeros((n, n), dtype=torch.float64)
    for j in range(n):
        ep = base.clone()
        ep[j] += h
        em = base.clone()
        em[j] -= h
        _, fp = field(ep)
        _, fm = field(em)
        H[:, j] = -(fp - fm) / (2.0 * h)
    return H


def pick_direction(kind, base, z, sample, calc, device, dtype, seed):
    """Return a normalized 3N direction vector (float64)."""
    n3 = base.numel()
    if kind == "random":
        g = torch.Generator().manual_seed(seed)
        d = torch.randn(n3, generator=g, dtype=torch.float64)
    elif kind == "reactant_product":
        rp = (sample.pos_product - sample.pos_reactant).reshape(-1).to(torch.float64)
        if rp.norm() < 1e-8:
            raise ValueError("reactant_product direction unavailable (no product).")
        d = rp
    elif kind in {"lowest_hess", "softest"}:
        H = autograd_hessian(calc.potential, base, z, device, dtype)
        Hsym = 0.5 * (H + H.T)
        evals, evecs = torch.linalg.eigh(Hsym)
        d = evecs[:, 0].to(torch.float64)
    else:
        raise ValueError(f"unknown direction kind: {kind}")
    return (d / d.norm()).to(torch.float64)


def power_spectrum(signal: np.ndarray, dlam: float, detrend_deg: int = 3):
    """Return (freqs cycles/Å, magnitude) of the detrended signal."""
    x = np.arange(signal.size)
    coeffs = np.polyfit(x, signal, detrend_deg)
    resid = signal - np.polyval(coeffs, x)
    window = np.hanning(resid.size)
    spec = np.fft.rfft(resid * window)
    freqs = np.fft.rfftfreq(resid.size, d=dlam)
    return freqs, np.abs(spec)


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc = EquiformerTorchCalculator(
        checkpoint_path=str(args.checkpoint),
        hessian_method="autograd",
        device=device,
    )
    if dtype == torch.float64:
        calc.potential = calc.potential.double()

    dataset = Transition1xDataset(
        str(args.h5), split=args.split, max_samples=args.sample_id + 1
    )
    sample = dataset[args.sample_id]
    z = sample.z.to(torch.int64)
    geom_attr = {
        "transition_state": "pos_transition",
        "reactant": "pos_reactant",
        "product": "pos_product",
    }[args.geometry]
    base = getattr(sample, geom_attr).reshape(-1).to(torch.float64).clone()
    n_atoms = z.numel()
    field = ForceField(calc.potential, z, device, dtype)

    eps = torch.finfo(dtype).eps
    e0, f0 = field(base)
    fscale = float(f0.abs().max())
    print(
        f"[geom] split={args.split} sample_id={args.sample_id} geometry={args.geometry} "
        f"rxn={sample.rxn} formula={sample.formula} N={n_atoms} dtype={args.dtype}"
    )
    print(f"[geom] E0={e0:.6f} eV  max|F|={fscale:.4e} eV/Å  float-eps={eps:.2e}")

    # ---- autograd Hessian reference (force Jacobian) ----
    H_ag = autograd_hessian(calc.potential, base, z, device, dtype)
    asym = float((H_ag - H_ag.T).norm() / (H_ag.norm() + 1e-30))
    print(f"[autograd H] ||H-H^T||/||H|| = {asym:.4e}  (non-conservativeness of forces)")

    results = {
        "split": args.split,
        "sample_id": args.sample_id,
        "geometry": args.geometry,
        "rxn": str(sample.rxn),
        "formula": str(sample.formula),
        "n_atoms": int(n_atoms),
        "dtype": args.dtype,
        "E0": e0,
        "fmax": fscale,
        "H_asym_rel": asym,
        "atomic_numbers": z.cpu().numpy(),
        "coords0": base.reshape(-1, 3).cpu().numpy(),
    }

    # =========================================================================
    # 1. Line scan + spectrum
    # =========================================================================
    directions = args.directions
    lam = np.linspace(-args.amplitude, args.amplitude, args.n_scan)
    dlam = float(lam[1] - lam[0])

    fig_scan, axes = plt.subplots(
        3, len(directions), figsize=(5.2 * len(directions), 11), squeeze=False
    )
    fig_spec, sax = plt.subplots(1, len(directions), figsize=(5.6 * len(directions), 4.2), squeeze=False)

    scan_store = {}
    for di, kind in enumerate(directions):
        d = pick_direction(kind, base, z, sample, calc, device, dtype, args.seed)
        g = np.zeros_like(lam)  # directional force d·F
        Evals = np.zeros_like(lam)
        Fnorm = np.zeros_like(lam)
        for k, l in enumerate(lam):
            e, f = field(base + float(l) * d)
            g[k] = float(torch.dot(f, d))
            Evals[k] = e
            Fnorm[k] = float(f.norm())

        # non-conservativeness check: for a conservative field -dE/dλ == g
        dEdl = -np.gradient(Evals, dlam)

        freqs, mag = power_spectrum(g, dlam)
        # roughness metric: fraction of spectral power above cutoff
        cutoff = args.spectrum_cutoff
        hi = freqs >= cutoff
        power = mag**2
        hf_frac = float(power[hi].sum() / (power.sum() + 1e-30))

        scan_store[kind] = dict(lam=lam, g=g, E=Evals, dEdl=dEdl, Fnorm=Fnorm,
                                freqs=freqs, mag=mag, hf_frac=hf_frac, d=d.cpu().numpy())
        print(f"[scan:{kind}] HF power fraction (>{cutoff:g} cyc/Å) = {hf_frac:.3e}")

        ax0, ax1, ax2 = axes[0, di], axes[1, di], axes[2, di]
        ax0.plot(lam, g, lw=1.0, color="C0")
        ax0.set_title(f"direction = {kind}")
        ax0.set_ylabel("directional force  d·F  [eV/Å]")
        ax0.grid(alpha=0.3)

        ax1.plot(lam, g, lw=1.0, color="C0", label="d·F (force)")
        ax1.plot(lam, dEdl, lw=1.0, ls="--", color="C3", label="-dE/dλ (FD of energy)")
        ax1.set_ylabel("conservative check [eV/Å]")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        ax2.plot(lam, Evals - Evals.min(), lw=1.0, color="C2")
        ax2.set_ylabel("E - min(E) [eV]")
        ax2.set_xlabel("displacement λ [Å]")
        ax2.grid(alpha=0.3)

        sx = sax[0, di]
        sx.semilogy(freqs, mag + 1e-30, lw=0.9, color="C0")
        sx.axvline(cutoff, color="grey", ls=":", lw=1)
        sx.set_title(f"{kind}: HF frac={hf_frac:.2e}")
        sx.set_xlabel("spatial frequency [cycles/Å]")
        sx.set_ylabel("|FFT(d·F)|")
        sx.grid(alpha=0.3)

    fig_scan.suptitle(
        f"Force line scan — {sample.formula} ({args.geometry}), {args.checkpoint.name}",
        fontsize=11,
    )
    fig_scan.tight_layout()
    scan_png = out_dir / f"{args.tag}_line_scan.png"
    fig_scan.savefig(scan_png, dpi=150)
    spec_png = out_dir / f"{args.tag}_force_spectrum.png"
    fig_spec.tight_layout()
    fig_spec.savefig(spec_png, dpi=150)
    plt.close(fig_scan)
    plt.close(fig_spec)

    # =========================================================================
    # 2. Finite-difference convergence sweep
    # =========================================================================
    h_values = np.logspace(np.log10(args.h_min), np.log10(args.h_max), args.n_h)
    # use the first requested direction for the directional convergence curve
    d_conv = torch.as_tensor(scan_store[directions[0]]["d"], dtype=torch.float64)
    Hd_ref = (H_ag @ d_conv).to(torch.float64)

    dir_err = np.zeros_like(h_values)
    for i, h in enumerate(h_values):
        Hd_fd = fd_directional(field, base, d_conv, float(h))
        dir_err[i] = float((Hd_fd - Hd_ref).norm() / (Hd_ref.norm() + 1e-30))

    do_full = args.fd_full and (3 * n_atoms <= args.fd_full_max_dim)
    full_err = None
    h_full = None
    if do_full:
        h_full = np.logspace(np.log10(args.h_min), np.log10(args.h_max), args.n_h_full)
        full_err = np.zeros_like(h_full)
        for i, h in enumerate(h_full):
            H_fd = fd_full_jacobian(field, base, float(h))
            full_err[i] = float((H_fd - H_ag).norm() / (H_ag.norm() + 1e-30))
    elif args.fd_full:
        print(f"[fd] skipping full Jacobian (3N={3 * n_atoms} > {args.fd_full_max_dim})")

    # reference numerical noise floor ~ eps * |F| / h (relative to ||Hd_ref||)
    noise_floor = eps * fscale / h_values / (float(Hd_ref.norm()) / max(1, 3 * n_atoms) + 1e-30)

    fig_fd, fax = plt.subplots(figsize=(7, 5))
    fax.loglog(h_values, dir_err, "o-", color="C0", label="directional  ||H_fd·d - H·d|| / ||H·d||")
    if full_err is not None:
        fax.loglog(h_full, full_err, "s-", color="C1", label="full Jacobian  ||H_fd - H|| / ||H||")
    fax.loglog(h_values, noise_floor, ":", color="grey", label="~ float-noise floor (eps·|F|/h)")
    # O(h^2) guide
    ref_h = h_values
    guide = dir_err[np.argmin(dir_err)] * (ref_h / ref_h[np.argmin(dir_err)]) ** 2
    fax.loglog(ref_h, guide, "--", color="k", alpha=0.4, label="O(h²) guide")
    fax.set_xlabel("FD step size  h  [Å]")
    fax.set_ylabel("relative error vs autograd Hessian")
    fax.set_title(f"FD convergence — autograd H = force Jacobian ({args.checkpoint.name})")
    fax.legend(fontsize=8)
    fax.grid(alpha=0.3, which="both")
    fig_fd.tight_layout()
    fd_png = out_dir / f"{args.tag}_fd_convergence.png"
    fig_fd.savefig(fd_png, dpi=150)
    plt.close(fig_fd)

    best_i = int(np.argmin(dir_err))
    print(
        f"[fd] best directional agreement: err={dir_err[best_i]:.3e} at h={h_values[best_i]:.2e} Å"
    )
    if full_err is not None:
        bj = int(np.argmin(full_err))
        print(f"[fd] best full-Jacobian agreement: err={full_err[bj]:.3e} at h={h_full[bj]:.2e} Å")

    # ---- save arrays ----
    npz_path = out_dir / f"{args.tag}_arrays.npz"
    save = dict(
        lam=lam,
        h_values=h_values,
        dir_err=dir_err,
        noise_floor=noise_floor,
        H_autograd=H_ag.cpu().numpy(),
    )
    if full_err is not None:
        save["h_full"] = h_full
        save["full_err"] = full_err
    for kind, s in scan_store.items():
        save[f"scan_{kind}_lam"] = s["lam"]
        save[f"scan_{kind}_g"] = s["g"]
        save[f"scan_{kind}_E"] = s["E"]
        save[f"scan_{kind}_freqs"] = s["freqs"]
        save[f"scan_{kind}_mag"] = s["mag"]
        save[f"scan_{kind}_d"] = s["d"]
    np.savez_compressed(npz_path, **save)

    results["hf_power_fraction"] = {k: v["hf_frac"] for k, v in scan_store.items()}
    results["fd_best_dir_err"] = float(dir_err[best_i])
    results["fd_best_dir_h"] = float(h_values[best_i])
    (out_dir / f"{args.tag}_summary.json").write_text(json.dumps(results, indent=2, default=str))

    print("\nWrote:")
    for p in [scan_png, spec_png, fd_png, npz_path, out_dir / f"{args.tag}_summary.json"]:
        print(f"  {p}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, default=_project_root() / "ckpt" / "hip_v3.ckpt")
    p.add_argument("--h5", type=Path, default=_project_root() / "data" / "transition1x.h5")
    p.add_argument("--split", default="test")
    p.add_argument("--sample-id", type=int, default=0)
    p.add_argument(
        "--geometry",
        choices=["transition_state", "reactant", "product"],
        default="transition_state",
    )
    p.add_argument(
        "--directions",
        nargs="+",
        default=["lowest_hess", "random"],
        choices=["lowest_hess", "softest", "random", "reactant_product"],
        help="scan directions to probe",
    )
    p.add_argument("--amplitude", type=float, default=0.2, help="line scan half-range [Å]")
    p.add_argument("--n-scan", type=int, default=2001, help="points in the line scan")
    p.add_argument("--spectrum-cutoff", type=float, default=20.0, help="HF metric cutoff [cycles/Å]")
    p.add_argument("--h-min", type=float, default=1e-4, help="smallest FD step [Å]")
    p.add_argument("--h-max", type=float, default=2e-1, help="largest FD step [Å]")
    p.add_argument("--n-h", type=int, default=31, help="directional FD steps")
    p.add_argument("--n-h-full", type=int, default=13, help="full-Jacobian FD steps")
    p.add_argument("--fd-full", action="store_true", default=True)
    p.add_argument("--no-fd-full", dest="fd_full", action="store_false")
    p.add_argument("--fd-full-max-dim", type=int, default=90, help="skip full Jacobian above this 3N")
    p.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=_project_root() / "runs" / "force_smoothness")
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    if args.tag is None:
        args.tag = f"{args.checkpoint.stem}_{args.split}_s{args.sample_id}_{args.geometry}"
    run(args)


if __name__ == "__main__":
    main()
