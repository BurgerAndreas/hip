#!/usr/bin/env python
"""Dense 1D glycine proton-transfer path scan for the autograd-vs-direct Hessian story.

Motivation
----------
The autograd ("AD") Hessian of an MLIP is the Jacobian of its *direct* force field,
``H_auto = -dF/dx`` (see ``hip/hessian_utils.compute_hessian``). Differentiation is a
high-pass filter: a small high-frequency wiggle in ``F(x)`` is negligible in the force
error but is amplified by ``~1/lambda`` in the Hessian. DFT forces are converged (no
wiggle) and HIP predicts the Hessian *directly* (never differentiates forces), so
neither pays this penalty.

This script samples the physical proton-transfer coordinate densely (the antisymmetric
proton slide ``xi = q_NH - q_OH``) and records, for EQV2 (AD) and HIP (direct):

- energy ``E(lambda)`` and path-projected force ``g(lambda) = t . F``
- directional curvature ``kappa(lambda) = t^T H t`` (autograd for EQV2, predicted for HIP)
- the finite-difference curvature of the *sampled* EQV2 force, ``-dg/dlambda``
- the power spectrum of ``g(lambda)`` and its ``|k|``-weighted (differentiated) spectrum

It also writes ORCA inputs for a (coarser) DFT subset on the *same* path so the existing
ORCA array / ``collect_glycine_pt_orca.py`` / ``cache_glycine_pt_orca_vibrations.py``
tooling produces a smooth DFT reference curve.

The companion figure is ``plotting/plot_glycine_pt_path_mechanism.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import torch

from hip.equiformer_torch_calculator import EquiformerTorchCalculator
from hip.frequency_analysis import analyze_frequencies_torch
from hip.transition1x_dataset import Transition1xDataset


SPLIT = "test"
SAMPLE_ID = 5
N_ATOM = 4
O_ATOM = 3
H_ATOM = 9
Z_TO_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
DEFAULT_ORCA_ROUTE = "! wB97X-D3 6-31G(d) TightSCF Grid5 FinalGrid6 EnGrad Freq"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def symbols_from_z(atomic_nums: np.ndarray) -> list[str]:
    return [Z_TO_SYMBOL[int(z)] for z in atomic_nums.tolist()]


def perpendicular_reference(coords: np.ndarray) -> np.ndarray:
    """Unit vector perpendicular to the N->O axis, in the plane of the TS H position."""
    n_pos, o_pos, h_pos = coords[N_ATOM], coords[O_ATOM], coords[H_ATOM]
    axis = (o_pos - n_pos) / np.linalg.norm(o_pos - n_pos)
    h_rel = h_pos - n_pos
    perp = h_rel - np.dot(h_rel, axis) * axis
    norm = float(np.linalg.norm(perp))
    if norm > 1e-8:
        return perp / norm
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, axis)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    perp = trial - np.dot(trial, axis) * axis
    return perp / np.linalg.norm(perp)


def path_frame(ts_coords: np.ndarray) -> dict[str, object]:
    """Geometry of the proton-transfer line: heavy-atom frame fixed at the TS."""
    n_pos, o_pos, h_pos = ts_coords[N_ATOM], ts_coords[O_ATOM], ts_coords[H_ATOM]
    axis_vec = o_pos - n_pos
    r_no = float(np.linalg.norm(axis_vec))
    axis = axis_vec / r_no
    perp_unit = perpendicular_reference(ts_coords)
    h_perp = float(np.linalg.norm((h_pos - n_pos) - np.dot(h_pos - n_pos, axis) * axis))
    return {"axis": axis, "perp_unit": perp_unit, "r_no": r_no, "h_perp": h_perp}


def place_proton(ts_coords: np.ndarray, x_along: float, frame: dict[str, object]) -> np.ndarray:
    """Slide H9 to N + x_along*axis + h_perp*perp, keeping heavy atoms fixed."""
    coords = np.array(ts_coords, dtype=float, copy=True)
    n_pos = coords[N_ATOM]
    coords[H_ATOM] = n_pos + float(x_along) * frame["axis"] + frame["h_perp"] * frame["perp_unit"]
    return coords


def path_tangent(frame: dict[str, object], n_atoms: int) -> np.ndarray:
    """Unit 3N Cartesian tangent: only H9 moves, along the N->O axis."""
    tangent = np.zeros((n_atoms, 3), dtype=float)
    tangent[H_ATOM] = frame["axis"]
    flat = tangent.reshape(-1)
    return flat / np.linalg.norm(flat)


def bond_lengths(coords: np.ndarray) -> tuple[float, float]:
    q_nh = float(np.linalg.norm(coords[N_ATOM] - coords[H_ATOM]))
    q_oh = float(np.linalg.norm(coords[O_ATOM] - coords[H_ATOM]))
    return q_nh, q_oh


def x_along_range(frame: dict[str, object], min_bond: float) -> tuple[float, float]:
    h_perp, r_no = frame["h_perp"], frame["r_no"]
    x0 = float(np.sqrt(max(min_bond**2 - h_perp**2, 0.0)))
    x_min, x_max = x0, r_no - x0
    if x_min >= x_max:
        x_min, x_max = 0.2 * r_no, 0.8 * r_no
    return x_min, x_max


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n{comment}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def write_orca_input(
    path: Path,
    symbols: list[str],
    coords: np.ndarray,
    route: str,
    charge: int,
    multiplicity: int,
) -> None:
    with path.open("w") as handle:
        handle.write(f"{route}\n\n%pal nprocs 16 end\n%maxcore 4000\n\n")
        handle.write(f"* xyz {charge} {multiplicity}\n")
        for symbol, xyz in zip(symbols, coords, strict=True):
            handle.write(f"  {symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")
        handle.write("*\n")


class PathModel:
    """Thin wrapper returning (energy, forces, hessian) for one geometry."""

    def __init__(self, checkpoint: Path, hessian_method: str, device: str):
        self.device = device
        self.hessian_method = hessian_method
        self.calc = EquiformerTorchCalculator(
            checkpoint_path=str(checkpoint),
            hessian_method=hessian_method,
            device=device,
        )

    def evaluate(self, coords: np.ndarray, atomic_nums: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """Return (energy, forces, raw Hessian). The Hessian is NOT symmetrized here so
        callers can measure the autograd non-conservativeness ||H - H^T|| / ||H||."""
        n_atoms = int(atomic_nums.size)
        out = self.calc.predict(
            coords=torch.tensor(coords, dtype=torch.float32, device=self.device),
            atomic_nums=torch.tensor(atomic_nums, dtype=torch.long, device=self.device),
            hessian_method=self.hessian_method,
            do_hessian=True,
        )
        energy = float(out["energy"].detach().cpu().reshape(-1)[0].item())
        forces = out["forces"].reshape(n_atoms, 3).detach().cpu().numpy().astype(np.float64)
        hessian = out["hessian"].reshape(3 * n_atoms, 3 * n_atoms).detach().cpu().numpy().astype(np.float64)
        return energy, forces, hessian


def asymmetry(hessian_raw: np.ndarray) -> float:
    num = float(np.linalg.norm(hessian_raw - hessian_raw.T))
    den = float(np.linalg.norm(hessian_raw)) + 1e-30
    return num / den


def vibrational_spectrum(
    hessian_sym: np.ndarray, coords: np.ndarray, atomic_nums: np.ndarray
) -> tuple[np.ndarray, int]:
    freq = analyze_frequencies_torch(
        torch.tensor(hessian_sym, dtype=torch.float64),
        torch.tensor(coords, dtype=torch.float64),
        [int(z) for z in atomic_nums],
    )
    evals = freq["eigvals"].detach().cpu().numpy().astype(np.float64)
    return evals, int(freq["neg_num"])


def power_spectrum(signal: np.ndarray, dlam: float, detrend_deg: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(signal.size)
    coeffs = np.polyfit(x, signal, min(detrend_deg, signal.size - 1))
    resid = signal - np.polyval(coeffs, x)
    window = np.hanning(resid.size)
    spec = np.abs(np.fft.rfft(resid * window))
    freqs = np.fft.rfftfreq(resid.size, d=dlam)
    return freqs, spec


def hf_fraction(freqs: np.ndarray, mag: np.ndarray, cutoff: float, weight_k: bool) -> float:
    power = mag**2
    if weight_k:
        power = power * (2.0 * np.pi * freqs) ** 2  # differentiation transfer function |k|^2
    total = float(power[1:].sum())
    if total <= 0.0:
        return 0.0
    return float(power[freqs >= cutoff].sum() / total)


def run_dense_path(
    ts_coords: np.ndarray,
    atomic_nums: np.ndarray,
    frame: dict[str, object],
    x_values: np.ndarray,
    eqv2: PathModel,
    hip: PathModel,
) -> dict[str, np.ndarray]:
    n_atoms = int(atomic_nums.size)
    tangent = path_tangent(frame, n_atoms)
    rows = {key: [] for key in (
        "x_along", "xi", "q_nh", "q_oh",
        "eqv2_energy", "hip_energy", "eqv2_g", "hip_g",
        "eqv2_kappa_auto", "hip_kappa_pred", "eqv2_fmax", "hip_fmax",
        "eqv2_asym", "hip_asym", "eqv2_n_negative", "hip_n_negative",
        "h_diff_frob_rel",
    )}
    eqv2_evals_rows: list[np.ndarray] = []
    hip_evals_rows: list[np.ndarray] = []
    eqv2_hess_rows: list[np.ndarray] = []
    coords_rows: list[np.ndarray] = []
    for i, x in enumerate(x_values):
        coords = place_proton(ts_coords, float(x), frame)
        q_nh, q_oh = bond_lengths(coords)
        e_eqv2, f_eqv2, h_eqv2_raw = eqv2.evaluate(coords, atomic_nums)
        e_hip, f_hip, h_hip_raw = hip.evaluate(coords, atomic_nums)
        h_eqv2, h_hip = symmetrize(h_eqv2_raw), symmetrize(h_hip_raw)
        eqv2_hess_rows.append(h_eqv2_raw)
        coords_rows.append(coords)

        eqv2_evals, eqv2_neg = vibrational_spectrum(h_eqv2, coords, atomic_nums)
        hip_evals, hip_neg = vibrational_spectrum(h_hip, coords, atomic_nums)
        eqv2_evals_rows.append(eqv2_evals)
        hip_evals_rows.append(hip_evals)

        rows["x_along"].append(float(x))
        rows["xi"].append(q_nh - q_oh)
        rows["q_nh"].append(q_nh)
        rows["q_oh"].append(q_oh)
        rows["eqv2_energy"].append(e_eqv2)
        rows["hip_energy"].append(e_hip)
        rows["eqv2_g"].append(float(f_eqv2.reshape(-1) @ tangent))
        rows["hip_g"].append(float(f_hip.reshape(-1) @ tangent))
        rows["eqv2_kappa_auto"].append(float(tangent @ h_eqv2 @ tangent))
        rows["hip_kappa_pred"].append(float(tangent @ h_hip @ tangent))
        rows["eqv2_fmax"].append(float(np.abs(f_eqv2).max()))
        rows["hip_fmax"].append(float(np.abs(f_hip).max()))
        rows["eqv2_asym"].append(asymmetry(h_eqv2_raw))
        rows["hip_asym"].append(asymmetry(h_hip_raw))
        rows["eqv2_n_negative"].append(eqv2_neg)
        rows["hip_n_negative"].append(hip_neg)
        rows["h_diff_frob_rel"].append(
            float(np.linalg.norm(h_eqv2 - h_hip) / (np.linalg.norm(h_hip) + 1e-30))
        )
        if (i + 1) % 50 == 0 or i + 1 == x_values.size:
            print(f"  dense path {i + 1}/{x_values.size}", flush=True)
    arrays = {key: np.asarray(value, dtype=float) for key, value in rows.items()}
    arrays["tangent"] = tangent
    arrays["eqv2_evals"] = np.stack(eqv2_evals_rows)
    arrays["hip_evals"] = np.stack(hip_evals_rows)
    arrays["eqv2_hessian_cartesian"] = np.stack(eqv2_hess_rows)  # RAW (unsymmetrized)
    arrays["coords_angstrom"] = np.stack(coords_rows)
    return arrays


def write_dft_inputs(
    out_dir: Path,
    ts_coords: np.ndarray,
    atomic_nums: np.ndarray,
    frame: dict[str, object],
    x_values: np.ndarray,
    route: str,
    charge: int,
    multiplicity: int,
) -> int:
    symbols = symbols_from_z(atomic_nums)
    xyz_dir = out_dir / "xyz"
    orca_dir = out_dir / "orca_inputs"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    orca_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "scan_manifest.csv"
    list_path = out_dir / "orca_input_list.txt"

    fieldnames = ["grid_id", "i_nh", "i_oh", "q_nh", "q_oh", "x_along", "xyz_path", "orca_input_path"]
    with manifest_path.open("w", newline="") as manifest_handle, list_path.open("w") as list_handle:
        writer = csv.DictWriter(manifest_handle, fieldnames=fieldnames)
        writer.writeheader()
        for grid_id, x in enumerate(x_values):
            coords = place_proton(ts_coords, float(x), frame)
            q_nh, q_oh = bond_lengths(coords)
            name = f"path_{grid_id:04d}_x_{x:.4f}_qNH_{q_nh:.3f}_qOH_{q_oh:.3f}"
            xyz_path = xyz_dir / f"{name}.xyz"
            inp_path = orca_dir / f"{name}.inp"
            comment = f"split={SPLIT} sample_id={SAMPLE_ID} x_along={x:.6f} q_nh={q_nh:.6f} q_oh={q_oh:.6f}"
            write_xyz(xyz_path, symbols, coords, comment)
            write_orca_input(inp_path, symbols, coords, route, charge, multiplicity)
            writer.writerow({
                "grid_id": grid_id,
                "i_nh": grid_id,
                "i_oh": grid_id,
                "q_nh": float(q_nh),
                "q_oh": float(q_oh),
                "x_along": float(x),
                "xyz_path": str(xyz_path.resolve()),
                "orca_input_path": str(inp_path.resolve()),
            })
            list_handle.write(f"{inp_path.resolve()}\n")
    return int(x_values.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", type=Path, default=project_root() / "data" / "transition1x.h5")
    parser.add_argument("--hip-checkpoint", type=Path, default=project_root() / "ckpt" / "hip_v2.ckpt")
    parser.add_argument("--eqv2-checkpoint", type=Path, default=project_root() / "ckpt" / "eqv2.ckpt")
    parser.add_argument("--output-dir", type=Path, default=project_root() / "runs" / "glycine_pt_path")
    parser.add_argument("--n-dense", type=int, default=1000, help="MLIP samples along the path.")
    parser.add_argument(
        "--n-dft",
        type=int,
        default=1000,
        help="ORCA samples along the same path. Equal to --n-dense puts DFT on the same grid.",
    )
    parser.add_argument("--min-bond", type=float, default=0.95, help="Shortest N-H/O-H bond at the path ends [A].")
    parser.add_argument("--spectrum-cutoff", type=float, default=8.0, help="HF metric cutoff [cycles/A].")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--multiplicity", type=int, default=1)
    parser.add_argument("--orca-route", default=DEFAULT_ORCA_ROUTE)
    parser.add_argument("--skip-dft-inputs", action="store_true", help="Only run the MLIP dense path.")
    parser.add_argument("--skip-mlip", action="store_true", help="Only write the DFT ORCA inputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = Transition1xDataset(str(args.h5), split=SPLIT, max_samples=SAMPLE_ID + 1)
    sample = dataset[SAMPLE_ID]
    atomic_nums = sample.z.detach().cpu().numpy().astype(int)
    ts_coords = sample.pos_transition.detach().cpu().numpy().reshape(-1, 3)
    frame = path_frame(ts_coords)
    x_min, x_max = x_along_range(frame, args.min_bond)
    x_dense = np.linspace(x_min, x_max, args.n_dense)
    x_dft = np.linspace(x_min, x_max, args.n_dft)

    print(
        f"[path] r(N-O)={frame['r_no']:.4f} A  h_perp={frame['h_perp']:.4f} A  "
        f"x_along in [{x_min:.4f}, {x_max:.4f}] A",
        flush=True,
    )

    metadata = {
        "split": SPLIT,
        "sample_id": SAMPLE_ID,
        "formula": str(sample.formula),
        "rxn": str(sample.rxn),
        "n_atom": N_ATOM,
        "o_atom": O_ATOM,
        "h_atom": H_ATOM,
        "r_no": frame["r_no"],
        "h_perp": frame["h_perp"],
        "x_min": float(x_min),
        "x_max": float(x_max),
        "n_dense": int(args.n_dense),
        "n_dft": int(args.n_dft),
        "min_bond": float(args.min_bond),
        "spectrum_cutoff": float(args.spectrum_cutoff),
        "hip_checkpoint": str(args.hip_checkpoint),
        "eqv2_checkpoint": str(args.eqv2_checkpoint),
        "orca_route": args.orca_route,
    }

    if not args.skip_mlip:
        print(f"[mlip] loading HIP={args.hip_checkpoint.name} EQV2={args.eqv2_checkpoint.name}", flush=True)
        hip = PathModel(args.hip_checkpoint, "predict", args.device)
        eqv2 = PathModel(args.eqv2_checkpoint, "autograd", args.device)
        arrays = run_dense_path(ts_coords, atomic_nums, frame, x_dense, eqv2, hip)

        dlam = float(x_dense[1] - x_dense[0])
        arrays["eqv2_kappa_fd"] = -np.gradient(arrays["eqv2_g"], dlam)
        arrays["hip_kappa_fd"] = -np.gradient(arrays["hip_g"], dlam)

        spectra = {}
        for label in ("eqv2_g", "hip_g"):
            freqs, mag = power_spectrum(arrays[label], dlam)
            spectra[f"{label}_freqs"] = freqs
            spectra[f"{label}_mag"] = mag
        arrays.update(spectra)
        arrays["atomic_numbers"] = atomic_nums.astype(float)

        metadata["hf_fraction_force_eqv2"] = hf_fraction(
            spectra["eqv2_g_freqs"], spectra["eqv2_g_mag"], args.spectrum_cutoff, weight_k=False
        )
        metadata["hf_fraction_curvature_eqv2"] = hf_fraction(
            spectra["eqv2_g_freqs"], spectra["eqv2_g_mag"], args.spectrum_cutoff, weight_k=True
        )
        metadata["hf_fraction_force_hip"] = hf_fraction(
            spectra["hip_g_freqs"], spectra["hip_g_mag"], args.spectrum_cutoff, weight_k=False
        )
        metadata["hf_fraction_curvature_hip"] = hf_fraction(
            spectra["hip_g_freqs"], spectra["hip_g_mag"], args.spectrum_cutoff, weight_k=True
        )

        npz_path = out_dir / "path_arrays.npz"
        np.savez_compressed(npz_path, **arrays)
        print(f"[mlip] wrote {npz_path}", flush=True)
        print(
            f"[mlip] EQV2 HF power fraction (>{args.spectrum_cutoff:g} cyc/A): "
            f"force={metadata['hf_fraction_force_eqv2']:.3e}  "
            f"curvature(|k|^2 weighted)={metadata['hf_fraction_curvature_eqv2']:.3e}",
            flush=True,
        )

    if not args.skip_dft_inputs:
        n_written = write_dft_inputs(
            out_dir, ts_coords, atomic_nums, frame, x_dft,
            args.orca_route, args.charge, args.multiplicity,
        )
        print(f"[dft] wrote {n_written} ORCA inputs to {out_dir / 'orca_inputs'}", flush=True)
        print(f"[dft] manifest: {out_dir / 'scan_manifest.csv'}", flush=True)
        print(f"[dft] input list: {out_dir / 'orca_input_list.txt'}", flush=True)

    # Merge into any existing metadata so a later DFT-only pass keeps MLIP fields (and vice versa).
    metadata_path = out_dir / "path_metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text())
        existing.update(metadata)
        metadata = existing
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[done] metadata: {metadata_path}", flush=True)


if __name__ == "__main__":
    main()
