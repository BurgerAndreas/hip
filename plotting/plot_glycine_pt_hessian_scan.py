#!/usr/bin/env python
"""Plot glycine proton-transfer PES and Hessian-comparison diagnostics.

Defaults match the current Transition1x test sample 5 scan:
q_nh = d(N4, H9), q_oh = d(O3, H9). ORCA wB97X/6-31G(d) is treated as the
reference. HIP/MLIP Hessians are assumed to be Cartesian Hessians in eV/A^2 on
the same grid.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import DFT_COLOR, LINE_WIDTH, MARKER_SIZE, THIN_LINE_WIDTH, finish_axis, model_color, model_palette


HARTREE_TO_EV = 27.211386245988
EV_TO_KCALMOL = 23.060548867
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM = HARTREE_TO_EV / BOHR_TO_ANGSTROM
HARTREE_PER_BOHR2_TO_EV_PER_ANG2 = HARTREE_TO_EV / (BOHR_TO_ANGSTROM**2)
DFT_LABEL = "DFT"
HIP_LABEL = "HIP"
AD_LABEL = "AD"
O_ATOM = 3
N_ATOM = 4
H_ATOM = 9
MASS_BY_Z = {
    1: 1.008,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998,
    15: 30.974,
    16: 32.065,
    17: 35.453,
}


@dataclass
class ModelData:
    label: str
    hessians_ev_ang2: np.ndarray
    energies_ev: np.ndarray | None = None
    forces_ev_ang: np.ndarray | None = None


@dataclass
class VibDiagnostics:
    evals: np.ndarray
    modes: np.ndarray
    n_negative: np.ndarray


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower())
    return safe.strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("runs/glycine_pt_scan"),
        help="Directory containing HIP scan outputs.",
    )
    parser.add_argument(
        "--orca-dir",
        type=Path,
        default=Path("orca_wb97x_631gd_glycine_pt_nh_oh_scan_80"),
        help="Directory containing ORCA HDF5 and metadata.",
    )
    parser.add_argument(
        "--orca-vib-cache",
        type=Path,
        default=None,
        help=(
            "Dense-grid ORCA NPZ cache from scripts/cache_glycine_pt_orca_vibrations.py. "
            "Defaults to scan-dir/orca_vib_cache.npz when present."
        ),
    )
    parser.add_argument(
        "--hip-arrays",
        type=Path,
        default=None,
        help="HIP NPZ with hessians_cartesian. Defaults to scan-dir/hip_v2_arrays.npz.",
    )
    parser.add_argument(
        "--hip-predictions",
        type=Path,
        default=None,
        help="HIP predictions CSV/parquet. Defaults to scan-dir/hip_v2_predictions.csv.",
    )
    parser.add_argument(
        "--mlip-arrays",
        type=Path,
        default=None,
        help=(
            "Optional MLIP/autograd NPZ on the same grid. Defaults to the current "
            "AD glycine output when present."
        ),
    )
    parser.add_argument("--mlip-label", default=AD_LABEL)
    parser.add_argument(
        "--model",
        action="append",
        nargs=2,
        metavar=("LABEL", "NPZ"),
        default=[],
        help="Additional model arrays to include. May be repeated.",
    )
    parser.add_argument("--hessian-key", default="hessians_cartesian")
    parser.add_argument("--energy-key", default="energies")
    parser.add_argument("--force-key", default="forces")
    parser.add_argument("--n-eigs", type=int, default=8)
    parser.add_argument(
        "--vib-cache",
        type=Path,
        default=None,
        help="Reusable vibrational diagnostics cache. Defaults to scan-dir/glycine_pt_vib_cache.npz.",
    )
    parser.add_argument(
        "--redo-vib-cache",
        action="store_true",
        help="Recompute vibrational diagnostics even when the cache exists.",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Compute/update the vibrational diagnostics cache and exit before plotting.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_orca(orca_dir: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    metadata = pd.read_csv(orca_dir / "metadata.csv")
    h5_path = orca_dir / "h5" / "glycine_pt_scan.h5"
    with h5py.File(h5_path, "r") as h5:
        arrays = {
            "grid_id": np.asarray(h5["grid_id"][:], dtype=int),
            "atomic_numbers": np.asarray(h5["atomic_numbers"][:], dtype=int),
            "coords_angstrom": np.asarray(h5["coordinates_bohr"][:], dtype=float)
            * BOHR_TO_ANGSTROM,
            "energy_hartree": np.asarray(h5["energy_hartree"][:], dtype=float),
            "forces_ev_ang": np.asarray(h5["forces_hartree_per_bohr"][:], dtype=float)
            * HARTREE_PER_BOHR_TO_EV_PER_ANGSTROM,
            "hessian_ev_ang2": np.asarray(h5["hessian_hartree_per_bohr2"][:], dtype=float)
            * HARTREE_PER_BOHR2_TO_EV_PER_ANG2,
            "q_nh": np.asarray(h5["q_nh_angstrom"][:], dtype=float),
            "q_oh": np.asarray(h5["q_oh_angstrom"][:], dtype=float),
        }
    return metadata, arrays


def load_orca_vib_cache(scan_dir: Path, cache_path: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    with np.load(cache_path) as cache:
        grid_id = np.asarray(cache["grid_id"], dtype=int)
        atomic_numbers = np.asarray(cache["atomic_numbers"], dtype=int)
        coords_angstrom = np.asarray(cache["coords_angstrom"], dtype=float)
        hessians = np.asarray(cache["hessian_ev_ang2"], dtype=float)
        q_nh = np.asarray(cache["q_nh"], dtype=float)
        q_oh = np.asarray(cache["q_oh"], dtype=float)
        vib_evals = np.asarray(cache["vib_evals_ev_ang2_amu"], dtype=float)
        vib_modes = np.asarray(cache["vib_modes_mw"], dtype=float)
        n_negative = np.asarray(cache["n_negative"], dtype=int)
        forces_ev_ang = (
            np.asarray(cache["forces_ev_ang"], dtype=float)
            if "forces_ev_ang" in cache.files
            else None
        )

    energies_path = scan_dir / "orca_energies.csv"
    if energies_path.exists():
        metadata = read_table(energies_path).sort_values("grid_id").reset_index(drop=True)
        energy_hartree = metadata["energy_hartree"].to_numpy(dtype=float)
    else:
        metadata = pd.DataFrame({"grid_id": grid_id, "q_nh": q_nh, "q_oh": q_oh})
        energy_hartree = np.full(grid_id.shape, np.nan, dtype=float)

    if len(metadata) != len(grid_id):
        raise ValueError(f"{energies_path} row count does not match {cache_path}")

    arrays = {
        "grid_id": grid_id,
        "atomic_numbers": np.repeat(atomic_numbers[None, :], len(grid_id), axis=0),
        "coords_angstrom": coords_angstrom,
        "energy_hartree": energy_hartree,
        "hessian_ev_ang2": hessians,
        "q_nh": q_nh,
        "q_oh": q_oh,
        "vib_evals": vib_evals,
        "vib_modes": vib_modes,
        "n_negative": n_negative,
    }
    if forces_ev_ang is not None:
        arrays["forces_ev_ang"] = forces_ev_ang
    return metadata, arrays


def load_npz_model(
    label: str,
    npz_path: Path,
    hessian_key: str,
    energy_key: str,
    force_key: str,
    n_grid: int,
) -> ModelData:
    data = np.load(npz_path)
    if hessian_key not in data:
        raise KeyError(f"{npz_path} does not contain {hessian_key!r}; keys={data.files}")
    hessians = np.asarray(data[hessian_key], dtype=float)
    if hessians.shape[0] != n_grid:
        raise ValueError(f"{label} has {hessians.shape[0]} Hessians, expected {n_grid}")
    energies = np.asarray(data[energy_key], dtype=float) if energy_key in data else None
    forces = np.asarray(data[force_key], dtype=float) if force_key in data else None
    return ModelData(label=label, hessians_ev_ang2=hessians, energies_ev=energies, forces_ev_ang=forces)


def load_hip_model(
    scan_dir: Path,
    arrays_path: Path,
    predictions_path: Path,
    hessian_key: str,
) -> ModelData:
    predictions = read_table(predictions_path).sort_values("grid_id")
    model = load_npz_model(
        label=HIP_LABEL,
        npz_path=arrays_path,
        hessian_key=hessian_key,
        energy_key="energies",
        force_key="forces",
        n_grid=len(predictions),
    )
    model.energies_ev = predictions["hip_v2_energy"].to_numpy(dtype=float)
    return model


def load_default_models(args: argparse.Namespace, scan_dir: Path, n_grid: int) -> list[ModelData]:
    hip_arrays = args.hip_arrays or scan_dir / "hip_v2_arrays.npz"
    hip_predictions = args.hip_predictions or scan_dir / "hip_v2_predictions.csv"
    models = [load_hip_model(scan_dir, hip_arrays, hip_predictions, args.hessian_key)]

    default_eqv2_dirs = [
        scan_dir.parent / scan_dir.name.replace("glycine_pt_scan", "glycine_pt_eqv2_autograd", 1),
        scan_dir.parent / "glycine_pt_eqv2_autograd",
    ]
    default_eqv2_arrays = next(
        (
            path / "eqv2_autograd_arrays.npz"
            for path in default_eqv2_dirs
            if (path / "eqv2_autograd_arrays.npz").exists()
        ),
        None,
    )
    mlip_arrays = args.mlip_arrays or default_eqv2_arrays
    if mlip_arrays is not None:
        models.append(
            load_npz_model(
                label=args.mlip_label,
                npz_path=mlip_arrays,
                hessian_key=args.hessian_key,
                energy_key=args.energy_key,
                force_key=args.force_key,
                n_grid=n_grid,
            )
        )

    for label, npz_path in args.model:
        models.append(
            load_npz_model(
                label=label,
                npz_path=Path(npz_path),
                hessian_key=args.hessian_key,
                energy_key=args.energy_key,
                force_key=args.force_key,
                n_grid=n_grid,
            )
        )
    return models


def symmetrize(hessians: np.ndarray) -> np.ndarray:
    return 0.5 * (hessians + np.swapaxes(hessians, -1, -2))


def frob_relative_error(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    denom = np.linalg.norm(ref_h.reshape(ref_h.shape[0], -1), axis=1)
    numer = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def reaction_center_error(
    model_h: np.ndarray,
    ref_h: np.ndarray,
    atoms: tuple[int, ...] = (3, 4, 9),
) -> np.ndarray:
    idx = np.array([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    model_block = symmetrize(model_h)[:, idx[:, None], idx]
    ref_block = symmetrize(ref_h)[:, idx[:, None], idx]
    denom = np.linalg.norm(ref_block.reshape(ref_block.shape[0], -1), axis=1)
    numer = np.linalg.norm((model_block - ref_block).reshape(model_block.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def distance_gradient(coords: np.ndarray, atom_a: int, atom_b: int) -> np.ndarray:
    grad = np.zeros_like(coords, dtype=float)
    vec = coords[atom_a] - coords[atom_b]
    dist = max(float(np.linalg.norm(vec)), 1e-12)
    unit = vec / dist
    grad[atom_a] = unit
    grad[atom_b] = -unit
    return grad


def normalized_projection(forces: np.ndarray, directions: np.ndarray) -> np.ndarray:
    flat_forces = np.asarray(forces, dtype=float).reshape(forces.shape[0], -1)
    flat_directions = np.asarray(directions, dtype=float).reshape(directions.shape[0], -1)
    norms = np.linalg.norm(flat_directions, axis=1)
    return np.einsum("ij,ij->i", flat_forces, flat_directions) / np.maximum(norms, 1e-12)


def force_projections(
    forces_ev_ang: np.ndarray,
    coords_angstrom: np.ndarray,
) -> dict[str, np.ndarray]:
    q_nh_dirs = []
    q_oh_dirs = []
    pt_dirs = []
    h_no_components = []
    for coords, forces in zip(coords_angstrom, forces_ev_ang, strict=True):
        q_nh_grad = distance_gradient(coords, N_ATOM, H_ATOM)
        q_oh_grad = distance_gradient(coords, O_ATOM, H_ATOM)
        q_nh_dirs.append(q_nh_grad)
        q_oh_dirs.append(q_oh_grad)
        # Positive proton-transfer CV points from N-bound toward O-bound configurations.
        pt_dirs.append(q_nh_grad - q_oh_grad)

        n_to_o = coords[O_ATOM] - coords[N_ATOM]
        n_to_o = n_to_o / max(float(np.linalg.norm(n_to_o)), 1e-12)
        h_no_components.append(float(np.dot(forces[H_ATOM], n_to_o)))

    q_nh_dirs_arr = np.stack(q_nh_dirs)
    q_oh_dirs_arr = np.stack(q_oh_dirs)
    pt_dirs_arr = np.stack(pt_dirs)
    return {
        "q_nh": normalized_projection(forces_ev_ang, q_nh_dirs_arr),
        "q_oh": normalized_projection(forces_ev_ang, q_oh_dirs_arr),
        "pt": normalized_projection(forces_ev_ang, pt_dirs_arr),
        "h_no": np.asarray(h_no_components, dtype=float),
    }


def force_metric_frame(
    df: pd.DataFrame,
    model: ModelData,
    ref_forces_ev_ang: np.ndarray,
    coords_angstrom: np.ndarray,
) -> pd.DataFrame | None:
    if model.forces_ev_ang is None:
        return None
    prefix = safe_label(model.label)
    model_forces = np.asarray(model.forces_ev_ang, dtype=float)
    ref_forces = np.asarray(ref_forces_ev_ang, dtype=float)
    ref_proj = force_projections(ref_forces, coords_angstrom)
    model_proj = force_projections(model_forces, coords_angstrom)

    metrics = df[["grid_id", "q_nh", "q_oh"]].copy()
    metrics[f"{prefix}_force_cartesian_mae"] = np.mean(np.abs(model_forces - ref_forces), axis=(1, 2))
    for key in ("pt", "q_nh", "q_oh", "h_no"):
        metrics[f"{prefix}_force_{key}"] = model_proj[key]
        metrics[f"{prefix}_force_{key}_error"] = model_proj[key] - ref_proj[key]
        metrics[f"orca_force_{key}"] = ref_proj[key]
    return metrics


def compute_vib_diagnostics(
    hessians_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    n_eigs: int,
) -> VibDiagnostics:
    n_grid = hessians_ev_ang2.shape[0]
    eval_rows = []
    mode_rows = []
    n_negative = []
    for idx in range(n_grid):
        evals_np, modes_np = vibrational_eigh(
            hessian_ev_ang2=hessians_ev_ang2[idx],
            coords_angstrom=coords_angstrom[idx],
            atomic_numbers=atomic_numbers[idx],
        )
        eval_rows.append(evals_np[:n_eigs])
        mode_rows.append(modes_np[:, :n_eigs])
        n_negative.append(int((evals_np < -1e-6).sum()))
    return VibDiagnostics(
        evals=np.stack(eval_rows),
        modes=np.stack(mode_rows),
        n_negative=np.asarray(n_negative, dtype=int),
    )


def _cache_keys(prefix: str) -> tuple[str, str, str]:
    return f"{prefix}_evals", f"{prefix}_modes", f"{prefix}_n_negative"


def _diag_to_cache(prefix: str, diag: VibDiagnostics) -> dict[str, np.ndarray]:
    eval_key, mode_key, nneg_key = _cache_keys(prefix)
    return {
        eval_key: diag.evals,
        mode_key: diag.modes,
        nneg_key: diag.n_negative,
    }


def _diag_from_cache(cache: np.lib.npyio.NpzFile, prefix: str) -> VibDiagnostics:
    eval_key, mode_key, nneg_key = _cache_keys(prefix)
    return VibDiagnostics(
        evals=np.asarray(cache[eval_key], dtype=float),
        modes=np.asarray(cache[mode_key], dtype=float),
        n_negative=np.asarray(cache[nneg_key], dtype=int),
    )


def _cache_has_diagnostics(
    cache: np.lib.npyio.NpzFile,
    model_prefixes: list[str],
    grid_ids: np.ndarray,
    n_eigs: int,
) -> bool:
    required = set(_cache_keys("orca"))
    for prefix in model_prefixes:
        required.update(_cache_keys(prefix))
    if not required.issubset(cache.files):
        return False
    if "grid_id" not in cache.files or "n_eigs" not in cache.files:
        return False
    if int(np.asarray(cache["n_eigs"]).reshape(-1)[0]) != n_eigs:
        return False
    return np.array_equal(np.asarray(cache["grid_id"], dtype=int), grid_ids)


def load_or_compute_vib_diagnostics(
    cache_path: Path,
    redo_cache: bool,
    ref_hessians: np.ndarray,
    models: list[ModelData],
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
    grid_ids: np.ndarray,
    n_eigs: int,
    ref_diag_override: VibDiagnostics | None = None,
) -> tuple[VibDiagnostics, dict[str, VibDiagnostics]]:
    model_prefixes = [safe_label(model.label) for model in models]
    if cache_path.exists() and not redo_cache:
        with np.load(cache_path) as cache:
            if _cache_has_diagnostics(cache, model_prefixes, grid_ids, n_eigs):
                print(f"Loading vibrational diagnostics cache: {cache_path}", flush=True)
                return (
                    _diag_from_cache(cache, "orca"),
                    {
                        model.label: _diag_from_cache(cache, safe_label(model.label))
                        for model in models
                    },
                )
            print(f"Ignoring stale vibrational diagnostics cache: {cache_path}", flush=True)

    if ref_diag_override is None:
        print("Computing ORCA vibrational diagnostics...", flush=True)
        ref_diag = compute_vib_diagnostics(
            ref_hessians, coords_angstrom, atomic_numbers, n_eigs=n_eigs
        )
    else:
        print("Using precomputed ORCA vibrational diagnostics.", flush=True)
        ref_diag = ref_diag_override
    model_diags: dict[str, VibDiagnostics] = {}
    for model in models:
        print(f"Computing {model.label} vibrational diagnostics...", flush=True)
        model_diags[model.label] = compute_vib_diagnostics(
            model.hessians_ev_ang2, coords_angstrom, atomic_numbers, n_eigs=n_eigs
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "grid_id": np.asarray(grid_ids, dtype=int),
        "n_eigs": np.asarray([n_eigs], dtype=int),
        "labels": np.asarray([model.label for model in models], dtype="S64"),
        "prefixes": np.asarray(model_prefixes, dtype="S64"),
    }
    payload.update(_diag_to_cache("orca", ref_diag))
    for model in models:
        payload.update(_diag_to_cache(safe_label(model.label), model_diags[model.label]))
    np.savez_compressed(cache_path, **payload)
    print(f"Wrote vibrational diagnostics cache: {cache_path}", flush=True)
    return ref_diag, model_diags


def mode_overlap(model_modes: np.ndarray, ref_modes: np.ndarray, mode_index: int = 0) -> np.ndarray:
    model = model_modes[:, :, mode_index]
    ref = ref_modes[:, :, mode_index]
    dots = np.einsum("ij,ij->i", model, ref)
    model_norm = np.linalg.norm(model, axis=1)
    ref_norm = np.linalg.norm(ref, axis=1)
    return np.abs(dots) / np.maximum(model_norm * ref_norm, 1e-12)


def eckart_generators(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    masses = np.asarray(masses, dtype=float).reshape(-1)
    n_atoms = xyz.shape[0]
    sqrt_m = np.sqrt(masses)
    sqrt_m3 = np.repeat(sqrt_m, 3)

    com = (xyz * masses[:, None]).sum(axis=0) / masses.sum()
    rel = xyz - com[None, :]

    cols = []
    for axis in np.eye(3):
        col = sqrt_m3 * np.tile(axis, n_atoms)
        cols.append(col / max(np.linalg.norm(col), 1e-12))

    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    rotations = (
        np.stack([np.zeros_like(rx), -rz, ry], axis=1),
        np.stack([rz, np.zeros_like(ry), -rx], axis=1),
        np.stack([-ry, rx, np.zeros_like(rz)], axis=1),
    )
    for rot in rotations:
        col = (rot * sqrt_m[:, None]).reshape(-1)
        norm = np.linalg.norm(col)
        if norm > 1e-12:
            cols.append(col / norm)
    return np.stack(cols, axis=1)


def vibrational_basis(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    generators = eckart_generators(coords, masses)
    q, r = np.linalg.qr(generators, mode="reduced")
    diag = np.abs(np.diag(r))
    rank = max(int((diag > 1e-6).sum()), 1)
    u, _, _ = np.linalg.svd(q[:, :rank], full_matrices=True)
    return u[:, rank:]


def vibrational_eigh(
    hessian_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    atomic_numbers = np.asarray(atomic_numbers, dtype=int).reshape(-1)
    masses = np.array([MASS_BY_Z[int(z)] for z in atomic_numbers], dtype=float)
    n_atoms = atomic_numbers.size
    hessian = np.asarray(hessian_ev_ang2, dtype=float).reshape(3 * n_atoms, 3 * n_atoms)
    hessian = 0.5 * (hessian + hessian.T)
    m3 = np.repeat(masses, 3)
    hessian_mw = hessian / np.sqrt(np.outer(m3, m3))
    q_vib = vibrational_basis(coords_angstrom, masses)
    hessian_red = q_vib.T @ hessian_mw @ q_vib
    hessian_red = 0.5 * (hessian_red + hessian_red.T)
    evals, evecs_red = np.linalg.eigh(hessian_red)
    return evals, q_vib @ evecs_red


def to_grid(df: pd.DataFrame, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tmp = df[["q_nh", "q_oh"]].copy()
    tmp["value"] = values
    pivot = tmp.pivot(index="q_oh", columns="q_nh", values="value").sort_index()
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    return x, y, z


def heatmap(
    ax: plt.Axes,
    df: pd.DataFrame,
    values: np.ndarray,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    levels: int = 15,
    contour_values: np.ndarray | None = None,
) -> None:
    x, y, z = to_grid(df, values)
    mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap)
    if contour_values is not None:
        _, _, contour_z = to_grid(df, contour_values)
        ax.contour(x, y, contour_z, levels=levels, colors="k", linewidths=0.45, alpha=0.45)
    ax.set_title(title)
    ax.set_xlabel(r"$q_\mathrm{NH}$ = d(N4,H9) [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}$ = d(O3,H9) [$\AA$]")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label(cbar_label)


def save_energy_figure(
    df: pd.DataFrame,
    orca_energy_kcal: np.ndarray,
    models: list[ModelData],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 1 + len(models), figsize=(5.2 * (1 + len(models)), 4.4))
    axes = np.atleast_1d(axes)
    heatmap(
        axes[0],
        df,
        orca_energy_kcal,
        f"{DFT_LABEL} PES",
        r"relative energy [kcal mol$^{-1}$]",
        cmap="magma",
    )
    for ax, model in zip(axes[1:], models, strict=False):
        if model.energies_ev is None:
            ax.axis("off")
            ax.set_title(f"{model.label}: no energies")
            continue
        model_rel_kcal = (model.energies_ev - np.nanmin(model.energies_ev)) * EV_TO_KCALMOL
        heatmap(
            ax,
            df,
            model_rel_kcal,
            f"{model.label} PES",
            r"relative energy [kcal mol$^{-1}$]",
            cmap="magma",
            contour_values=orca_energy_kcal,
        )
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_pes_surfaces.png", dpi=dpi)
    plt.close(fig)


def save_hessian_metric_figure(
    df: pd.DataFrame,
    orca_energy_kcal: np.ndarray,
    ref_diag: VibDiagnostics,
    model: ModelData,
    model_diag: VibDiagnostics,
    ref_hessians: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    rel_frob = frob_relative_error(model.hessians_ev_ang2, ref_hessians)
    rc_frob = reaction_center_error(model.hessians_ev_ang2, ref_hessians)
    eig0_error = model_diag.evals[:, 0] - ref_diag.evals[:, 0]
    overlap0 = mode_overlap(model_diag.modes, ref_diag.modes, mode_index=0)
    nneg_delta = model_diag.n_negative - ref_diag.n_negative

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.7))
    heatmap(
        axes[0, 0],
        df,
        rel_frob,
        f"{model.label}: full Hessian error",
        r"$||H-H_\mathrm{DFT}||_F / ||H_\mathrm{DFT}||_F$",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[0, 1],
        df,
        rc_frob,
        f"{model.label}: O-N-H error",
        r"relative Frobenius error",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[0, 2],
        df,
        eig0_error,
        f"{model.label}: lowest eigenvalue error",
        r"$\lambda_0 - \lambda_{0,\mathrm{DFT}}$ [eV A$^{-2}$ amu$^{-1}$]",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 0],
        df,
        overlap0,
        f"{model.label}: unstable-mode overlap",
        r"$|\langle v_0, v_{0,\mathrm{DFT}}\rangle|$",
        cmap="magma",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 1],
        df,
        ref_diag.n_negative,
        f"{DFT_LABEL} number of negative modes",
        "count",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 2],
        df,
        nneg_delta,
        f"{model.label}: negative-mode count error",
        r"$n_\mathrm{neg} - n_{\mathrm{neg,DFT}}$",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    fig.tight_layout(pad=0.01)
    safe_label = model.label.lower().replace(" ", "_").replace("/", "_")
    fig.savefig(output_dir / f"glycine_pt_hessian_metrics_{safe_label}.png", dpi=dpi)
    plt.close(fig)

    metrics = df[["grid_id", "q_nh", "q_oh"]].copy()
    metrics[f"{safe_label}_relative_hessian_error"] = rel_frob
    metrics[f"{safe_label}_reaction_center_error"] = rc_frob
    metrics[f"{safe_label}_eig0_error"] = eig0_error
    metrics[f"{safe_label}_mode0_overlap"] = overlap0
    metrics[f"{safe_label}_nneg_delta"] = nneg_delta
    return metrics


def save_method_summary_figure(
    metrics_by_label: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> None:
    labels = list(metrics_by_label)
    if not labels:
        return

    metric_specs = [
        ("relative_hessian_error", "full Hessian rel. error", False),
        ("reaction_center_error", "O-N-H rel. error", False),
        ("eig0_error", r"$|\Delta\lambda_0|$", True),
        ("mode0_overlap", "unstable-mode overlap", False),
        ("nneg_delta", "negative-mode agreement", False),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.1 * len(metric_specs), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (suffix, title, use_abs) in zip(axes, metric_specs, strict=True):
        values = []
        tick_labels = []
        for label in labels:
            prefix = safe_label(label)
            column = f"{prefix}_{suffix}"
            data = metrics_by_label[label][column].to_numpy(dtype=float)
            if suffix == "nneg_delta":
                data = (data == 0).astype(float)
            elif use_abs:
                data = np.abs(data)
            values.append(data)
            tick_labels.append(label)

        if suffix == "nneg_delta":
            means = [float(np.nanmean(v)) for v in values]
            sns.barplot(x=labels, y=means, ax=ax, palette=model_palette(labels), hue=labels, legend=False, alpha=0.8)
            ax.set_ylim(0.0, 1.05)
            ax.set_ylabel("fraction of grid points")
            for idx, mean in enumerate(means):
                ax.text(idx, mean + 0.025, f"{mean:.2f}", ha="center", va="bottom", fontsize=8)
        else:
            plot_df = pd.DataFrame(
                {
                    "label": np.concatenate([np.repeat(label, len(data)) for label, data in zip(tick_labels, values, strict=True)]),
                    "value": np.concatenate(values),
                }
            )
            sns.boxplot(data=plot_df, x="label", y="value", ax=ax, palette=model_palette(tick_labels), hue="label", legend=False, showfliers=False)
            if suffix in {"relative_hessian_error", "reaction_center_error"}:
                ax.set_yscale("log")
            ax.set_ylabel(title)

        ax.set_title(title)
        ax.tick_params(axis="x", labelrotation=25)
        finish_axis(ax)

    fig.suptitle(f"Glycine Hessian Method Summary vs {DFT_LABEL}")
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_hessian_method_summary.png", dpi=dpi)
    plt.close(fig)


def save_force_projection_figure(
    df: pd.DataFrame,
    force_metrics_by_label: dict[str, pd.DataFrame],
    orca_energy_kcal: np.ndarray,
    output_dir: Path,
    dpi: int,
) -> None:
    if not force_metrics_by_label:
        return

    labels = list(force_metrics_by_label)
    nrows = len(labels)
    fig, axes = plt.subplots(nrows, 3, figsize=(14.4, 4.2 * nrows), squeeze=False)
    ref_values = next(iter(force_metrics_by_label.values()))["orca_force_pt"].to_numpy(dtype=float)
    force_lim = max(float(np.nanmax(np.abs(ref_values))), 1.0)

    for row_idx, label in enumerate(labels):
        prefix = safe_label(label)
        metrics = force_metrics_by_label[label]
        model_values = metrics[f"{prefix}_force_pt"].to_numpy(dtype=float)
        error_values = metrics[f"{prefix}_force_pt_error"].to_numpy(dtype=float)
        error_lim = max(float(np.nanmax(np.abs(error_values))), 1.0)

        heatmap(
            axes[row_idx, 0],
            df,
            ref_values,
            f"{DFT_LABEL} force along q_NH - q_OH",
            "projected force [eV/A]",
            cmap="coolwarm",
            contour_values=orca_energy_kcal,
        )
        axes[row_idx, 0].collections[0].set_clim(-force_lim, force_lim)
        heatmap(
            axes[row_idx, 1],
            df,
            model_values,
            f"{label}: force along q_NH - q_OH",
            "projected force [eV/A]",
            cmap="coolwarm",
            contour_values=orca_energy_kcal,
        )
        axes[row_idx, 1].collections[0].set_clim(-force_lim, force_lim)
        heatmap(
            axes[row_idx, 2],
            df,
            error_values,
            f"{label} - {DFT_LABEL} force projection",
            "force error [eV/A]",
            cmap="coolwarm",
            contour_values=orca_energy_kcal,
        )
        axes[row_idx, 2].collections[0].set_clim(-error_lim, error_lim)

    fig.suptitle("Projected Forces Along Glycine Proton-Transfer CV")
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_force_projection_pt.png", dpi=dpi)
    plt.close(fig)


def save_force_summary_figure(
    force_metrics_by_label: dict[str, pd.DataFrame],
    output_dir: Path,
    dpi: int,
) -> None:
    labels = list(force_metrics_by_label)
    if not labels:
        return

    metric_specs = [
        ("force_cartesian_mae", "Cartesian force MAE"),
        ("force_pt_error", r"$|F_{q_\mathrm{NH}-q_\mathrm{OH}}|$ error"),
        ("force_q_nh_error", r"$|F_{q_\mathrm{NH}}|$ error"),
        ("force_q_oh_error", r"$|F_{q_\mathrm{OH}}|$ error"),
        ("force_h_no_error", "H force along N->O error"),
    ]
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(4.1 * len(metric_specs), 4.2))
    axes = np.atleast_1d(axes)

    for ax, (suffix, title) in zip(axes, metric_specs, strict=True):
        values = []
        for label in labels:
            prefix = safe_label(label)
            column = f"{prefix}_{suffix}"
            data = force_metrics_by_label[label][column].to_numpy(dtype=float)
            if suffix != "force_cartesian_mae":
                data = np.abs(data)
            values.append(data)
        plot_df = pd.DataFrame(
            {
                "label": np.concatenate([np.repeat(label, len(data)) for label, data in zip(labels, values, strict=True)]),
                "value": np.concatenate(values),
            }
        )
        sns.boxplot(data=plot_df, x="label", y="value", ax=ax, palette=model_palette(labels), hue="label", legend=False, showfliers=False)
        ax.set_title(title)
        ax.set_ylabel("eV/A")
        ax.tick_params(axis="x", labelrotation=25)
        finish_axis(ax)

    fig.suptitle(f"Glycine Force Summary vs {DFT_LABEL}")
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_force_method_summary.png", dpi=dpi)
    plt.close(fig)


def save_low_mode_figure(
    df: pd.DataFrame,
    ref_diag: VibDiagnostics,
    model_diags: dict[str, VibDiagnostics],
    output_dir: Path,
    dpi: int,
) -> int:
    # Pick the DFT point closest to an index-1 saddle with the most negative lowest mode.
    index1 = np.where(ref_diag.n_negative == 1)[0]
    if len(index1) > 0:
        selected = int(index1[np.argmin(ref_diag.evals[index1, 0])])
    else:
        selected = int(np.argmin(ref_diag.evals[:, 0]))

    row = df.iloc[selected]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    mode_ids = np.arange(ref_diag.evals.shape[1])
    sns.lineplot(x=mode_ids, y=ref_diag.evals[selected], ax=ax, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=DFT_LABEL, color=DFT_COLOR)
    for label, diag in model_diags.items():
        sns.lineplot(x=mode_ids, y=diag.evals[selected], ax=ax, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, linestyle="--", label=label, color=model_color(label))
    ax.axhline(0.0, color="k", linewidth=THIN_LINE_WIDTH)
    ax.set_xlabel("vibrational mode index")
    ax.set_ylabel(r"projected Hessian eigenvalue [eV A$^{-2}$ amu$^{-1}$]")
    ax.set_title(
        f"Low-mode spectrum at grid {int(row.grid_id)} "
        f"($q_{{NH}}$={row.q_nh:.3f} A, $q_{{OH}}$={row.q_oh:.3f} A)"
    )
    ax.legend()
    finish_axis(ax)
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_low_mode_spectrum.png", dpi=dpi)
    plt.close(fig)
    return selected


def save_reaction_center_blocks(
    selected: int,
    df: pd.DataFrame,
    ref_hessians: np.ndarray,
    models: list[ModelData],
    output_dir: Path,
    dpi: int,
    atoms: tuple[int, ...] = (3, 4, 9),
) -> None:
    idx = np.array([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    labels = [f"{atom}{axis}" for atom in atoms for axis in ("x", "y", "z")]
    row = df.iloc[selected]

    ncols = 1 + len(models)
    fig, axes = plt.subplots(1, ncols, figsize=(4.8 * ncols, 4.2), squeeze=False)
    ref_block = symmetrize(ref_hessians)[selected][idx[:, None], idx]
    vmax = np.nanmax(np.abs(ref_block))
    im = axes[0, 0].imshow(ref_block, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0, 0].set_title(f"{DFT_LABEL} O-N-H Hessian")
    for ax in axes[0]:
        ax.set_xticks(range(len(labels)), labels, rotation=90)
        ax.set_yticks(range(len(labels)), labels)
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)

    for ax, model in zip(axes[0, 1:], models, strict=False):
        block = symmetrize(model.hessians_ev_ang2)[selected][idx[:, None], idx] - ref_block
        vmax_diff = np.nanmax(np.abs(block))
        im = ax.imshow(block, cmap="coolwarm", vmin=-vmax_diff, vmax=vmax_diff)
        ax.set_title(f"{model.label} - {DFT_LABEL}")
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(
        f"O-N-H block at grid {int(row.grid_id)} "
        f"($q_{{NH}}$={row.q_nh:.3f} A, $q_{{OH}}$={row.q_oh:.3f} A)"
    )
    fig.tight_layout(pad=0.01)
    fig.savefig(output_dir / "glycine_pt_reaction_center_hessian_blocks.png", dpi=dpi)
    plt.close(fig)


def save_method_comparison_figure(
    df: pd.DataFrame,
    orca_energy_kcal: np.ndarray,
    left: ModelData,
    right: ModelData,
    left_diag: VibDiagnostics,
    right_diag: VibDiagnostics,
    output_dir: Path,
    dpi: int,
) -> pd.DataFrame:
    rel_hessian_diff = frob_relative_error(left.hessians_ev_ang2, right.hessians_ev_ang2)
    eig0_delta = left_diag.evals[:, 0] - right_diag.evals[:, 0]
    overlap0 = mode_overlap(left_diag.modes, right_diag.modes, mode_index=0)
    nneg_delta = left_diag.n_negative - right_diag.n_negative

    fig, axes = plt.subplots(2, 2, figsize=(10.7, 8.7))
    heatmap(
        axes[0, 0],
        df,
        rel_hessian_diff,
        f"{left.label} vs {right.label}: Hessian difference",
        r"$||H_1-H_2||_F / ||H_2||_F$",
        cmap="viridis",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[0, 1],
        df,
        eig0_delta,
        f"{left.label} - {right.label}: lowest eigenvalue",
        r"$\Delta\lambda_0$ [eV A$^{-2}$ amu$^{-1}$]",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 0],
        df,
        overlap0,
        f"{left.label} vs {right.label}: unstable-mode overlap",
        r"$|\langle v_{0,1}, v_{0,2}\rangle|$",
        cmap="magma",
        contour_values=orca_energy_kcal,
    )
    heatmap(
        axes[1, 1],
        df,
        nneg_delta,
        f"{left.label} - {right.label}: negative modes",
        "count difference",
        cmap="coolwarm",
        contour_values=orca_energy_kcal,
    )
    fig.tight_layout(pad=0.01)
    left_key = safe_label(left.label)
    right_key = safe_label(right.label)
    fig.savefig(output_dir / f"glycine_pt_hessian_compare_{left_key}_vs_{right_key}.png", dpi=dpi)
    plt.close(fig)

    metrics = df[["grid_id", "q_nh", "q_oh"]].copy()
    prefix = f"{left_key}_vs_{right_key}"
    metrics[f"{prefix}_relative_hessian_difference"] = rel_hessian_diff
    metrics[f"{prefix}_eig0_delta"] = eig0_delta
    metrics[f"{prefix}_mode0_overlap"] = overlap0
    metrics[f"{prefix}_nneg_delta"] = nneg_delta
    return metrics


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    output_dir = args.output_dir or scan_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    vib_cache = args.vib_cache or scan_dir / "glycine_pt_vib_cache.npz"
    orca_vib_cache = args.orca_vib_cache or (
        scan_dir / "orca_vib_cache.npz" if (scan_dir / "orca_vib_cache.npz").exists() else None
    )

    hip_predictions = args.hip_predictions or scan_dir / "hip_v2_predictions.csv"

    if orca_vib_cache is not None:
        _, orca = load_orca_vib_cache(scan_dir, orca_vib_cache)
    else:
        _, orca = load_orca(args.orca_dir)
    df = read_table(hip_predictions).sort_values("grid_id").reset_index(drop=True)
    grid_ids = df["grid_id"].to_numpy(dtype=int)
    order = np.argsort(orca["grid_id"])
    for key in (
        "grid_id",
        "atomic_numbers",
        "coords_angstrom",
        "energy_hartree",
        "hessian_ev_ang2",
        "forces_ev_ang",
        "vib_evals",
        "vib_modes",
        "n_negative",
    ):
        if key in orca:
            orca[key] = orca[key][order]
    if not np.array_equal(grid_ids, orca["grid_id"]):
        raise ValueError("HIP predictions and ORCA HDF5 grid_id ordering do not match")

    orca_rel_kcal = (orca["energy_hartree"] - np.nanmin(orca["energy_hartree"])) * HARTREE_TO_EV
    orca_rel_kcal = orca_rel_kcal * EV_TO_KCALMOL
    ref_hessians = orca["hessian_ev_ang2"]
    ref_forces = orca.get("forces_ev_ang")
    coords_angstrom = orca["coords_angstrom"]
    atomic_numbers = orca["atomic_numbers"]
    ref_diag_override = None
    if {"vib_evals", "vib_modes", "n_negative"}.issubset(orca):
        ref_diag_override = VibDiagnostics(
            evals=orca["vib_evals"][:, : args.n_eigs],
            modes=orca["vib_modes"][:, :, : args.n_eigs],
            n_negative=orca["n_negative"],
        )

    models = load_default_models(args, scan_dir, n_grid=len(df))
    ref_diag, model_diags = load_or_compute_vib_diagnostics(
        cache_path=vib_cache,
        redo_cache=args.redo_vib_cache,
        ref_hessians=ref_hessians,
        models=models,
        coords_angstrom=coords_angstrom,
        atomic_numbers=atomic_numbers,
        grid_ids=grid_ids,
        n_eigs=args.n_eigs,
        ref_diag_override=ref_diag_override,
    )
    if args.cache_only:
        print("Cache-only mode complete.", flush=True)
        return

    save_energy_figure(df, orca_rel_kcal, models, output_dir, args.dpi)

    metrics_frames = [df[["grid_id", "q_nh", "q_oh"]].copy()]
    metrics_frames[0]["orca_energy_relative_kcalmol"] = orca_rel_kcal
    metrics_frames[0]["orca_eig0"] = ref_diag.evals[:, 0]
    metrics_frames[0]["orca_n_negative"] = ref_diag.n_negative
    metrics_by_label: dict[str, pd.DataFrame] = {}
    for model in models:
        metrics = save_hessian_metric_figure(
            df=df,
            orca_energy_kcal=orca_rel_kcal,
            ref_diag=ref_diag,
            model=model,
            model_diag=model_diags[model.label],
            ref_hessians=ref_hessians,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        metrics_by_label[model.label] = metrics
        metrics_frames.append(metrics.drop(columns=["q_nh", "q_oh"]))

    save_method_summary_figure(metrics_by_label, output_dir, args.dpi)

    force_metrics_by_label: dict[str, pd.DataFrame] = {}
    if ref_forces is None:
        print("Skipping force plots: ORCA reference forces are not available.", flush=True)
    else:
        force_ref_columns_added = False
        for model in models:
            force_metrics = force_metric_frame(
                df=df,
                model=model,
                ref_forces_ev_ang=ref_forces,
                coords_angstrom=coords_angstrom,
            )
            if force_metrics is None:
                continue
            force_metrics_by_label[model.label] = force_metrics
            frame = force_metrics.drop(columns=["q_nh", "q_oh"])
            if force_ref_columns_added:
                frame = frame.drop(columns=[col for col in frame.columns if col.startswith("orca_force_")])
            force_ref_columns_added = True
            metrics_frames.append(frame)

        save_force_projection_figure(
            df=df,
            force_metrics_by_label=force_metrics_by_label,
            orca_energy_kcal=orca_rel_kcal,
            output_dir=output_dir,
            dpi=args.dpi,
        )
        save_force_summary_figure(force_metrics_by_label, output_dir, args.dpi)

    if len(models) >= 2:
        method_metrics = save_method_comparison_figure(
            df=df,
            orca_energy_kcal=orca_rel_kcal,
            left=models[0],
            right=models[1],
            left_diag=model_diags[models[0].label],
            right_diag=model_diags[models[1].label],
            output_dir=output_dir,
            dpi=args.dpi,
        )
        metrics_frames.append(method_metrics.drop(columns=["q_nh", "q_oh"]))

    selected = save_low_mode_figure(df, ref_diag, model_diags, output_dir, args.dpi)
    save_reaction_center_blocks(selected, df, ref_hessians, models, output_dir, args.dpi)

    metrics_df = metrics_frames[0]
    for frame in metrics_frames[1:]:
        metrics_df = metrics_df.merge(frame, on="grid_id", how="left")
    metrics_path = output_dir / "glycine_pt_hessian_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    print(f"Wrote plots to {output_dir}", flush=True)
    print(f"Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
