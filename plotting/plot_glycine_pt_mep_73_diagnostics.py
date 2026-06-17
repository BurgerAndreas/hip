#!/usr/bin/env python
"""Plot DFT/HIP/EQV2 diagnostics on the 73-frame glycine proton-transfer MEP."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import DFT_COLOR, LINE_WIDTH, MARKER_SIZE, SMALL_MARKER_SIZE, THIN_LINE_WIDTH, finish_axis, model_color


HARTREE_TO_EV = 27.211386245988
EV_TO_KCALMOL = 23.060548867

DFT_LABEL = "DFT"
HIP_LABEL = "HIP"
EQV2_LABEL = "AD"
O_ATOM = 3
N_ATOM = 4
H_TRANSFER_ATOM = 9
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
METHOD_COLORS = {
    DFT_LABEL: DFT_COLOR,
    HIP_LABEL: model_color(HIP_LABEL),
    EQV2_LABEL: model_color(EQV2_LABEL),
}
METHOD_MARKERS = {
    DFT_LABEL: "D",
    HIP_LABEL: "s",
    EQV2_LABEL: "o",
}
METHOD_LINESTYLES = {
    DFT_LABEL: "-",
    HIP_LABEL: "-",
    EQV2_LABEL: "-",
}
NEGATIVE_MODE_DODGE = {
    DFT_LABEL: -0.06,
    HIP_LABEL: 0.0,
    EQV2_LABEL: 0.06,
}


@dataclass
class MethodData:
    label: str
    energies_ev: np.ndarray
    forces_ev_ang: np.ndarray
    hessians_ev_ang2: np.ndarray
    coords_angstrom: np.ndarray
    atomic_numbers: np.ndarray
    q_nh: np.ndarray
    q_oh: np.ndarray


@dataclass
class VibDiagnostics:
    evals: np.ndarray
    modes_mw: np.ndarray
    n_negative: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mep-dir", type=Path, default=Path("runs/glycine_pt_mep_73"))
    parser.add_argument("--orca-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--x-axis", choices=["xi", "frame"], default="xi")
    parser.add_argument("--n-eigs", type=int, default=6)
    parser.add_argument("--negative-threshold", type=float, default=-1e-6)
    parser.add_argument("--reaction-center-atoms", default="3,4,9")
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower())
    return safe.strip("_") or "method"


def load_orca(path: Path) -> MethodData:
    with np.load(path) as data:
        coords = np.asarray(data["coords_angstrom"], dtype=float)
        atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)
        q_nh, q_oh = load_or_compute_cvs(data, coords)
        energy_key = "energy_hartree_engrad" if "energy_hartree_engrad" in data.files else "energy_hartree"
        return MethodData(
            label=DFT_LABEL,
            energies_ev=np.asarray(data[energy_key], dtype=float) * HARTREE_TO_EV,
            forces_ev_ang=np.asarray(data["forces_ev_ang"], dtype=float),
            hessians_ev_ang2=np.asarray(data["hessian_ev_ang2"], dtype=float),
            coords_angstrom=coords,
            atomic_numbers=atomic_numbers,
            q_nh=q_nh,
            q_oh=q_oh,
        )


def load_model(label: str, path: Path) -> MethodData:
    with np.load(path) as data:
        coords = np.asarray(data["coords_angstrom"], dtype=float)
        atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)
        q_nh, q_oh = load_or_compute_cvs(data, coords)
        return MethodData(
            label=label,
            energies_ev=np.asarray(data["energies"], dtype=float),
            forces_ev_ang=np.asarray(data["forces"], dtype=float),
            hessians_ev_ang2=np.asarray(data["hessians_cartesian"], dtype=float),
            coords_angstrom=coords,
            atomic_numbers=atomic_numbers,
            q_nh=q_nh,
            q_oh=q_oh,
        )


def load_or_compute_cvs(data: np.lib.npyio.NpzFile, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if "q_nh" in data.files and "q_oh" in data.files:
        return np.asarray(data["q_nh"], dtype=float), np.asarray(data["q_oh"], dtype=float)
    q_nh = np.linalg.norm(coords[:, N_ATOM] - coords[:, H_TRANSFER_ATOM], axis=1)
    q_oh = np.linalg.norm(coords[:, O_ATOM] - coords[:, H_TRANSFER_ATOM], axis=1)
    return q_nh, q_oh


def validate_same_grid(ref: MethodData, methods: list[MethodData]) -> None:
    n_frames = ref.coords_angstrom.shape[0]
    for method in methods:
        if method.coords_angstrom.shape != ref.coords_angstrom.shape:
            raise ValueError(f"{method.label} coordinates do not match {DFT_LABEL}")
        if method.forces_ev_ang.shape != ref.forces_ev_ang.shape:
            raise ValueError(f"{method.label} forces do not match {DFT_LABEL}")
        if method.hessians_ev_ang2.shape != ref.hessians_ev_ang2.shape:
            raise ValueError(f"{method.label} Hessians do not match {DFT_LABEL}")
        if method.energies_ev.shape[0] != n_frames:
            raise ValueError(f"{method.label} has {method.energies_ev.shape[0]} energies, expected {n_frames}")


def finite_difference_tangent(coords: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(coords, dtype=float)
    tangent[0] = coords[1] - coords[0]
    tangent[-1] = coords[-1] - coords[-2]
    tangent[1:-1] = coords[2:] - coords[:-2]
    flat = tangent.reshape(tangent.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    return tangent / np.maximum(norms[:, None, None], 1e-12)


def distance_gradient(coords: np.ndarray, atom_a: int, atom_b: int) -> np.ndarray:
    grad = np.zeros_like(coords, dtype=float)
    vec = coords[atom_a] - coords[atom_b]
    dist = max(float(np.linalg.norm(vec)), 1e-12)
    unit = vec / dist
    grad[atom_a] = unit
    grad[atom_b] = -unit
    return grad


def cv_directions(coords: np.ndarray) -> dict[str, np.ndarray]:
    q_nh_dirs = []
    q_oh_dirs = []
    xi_dirs = []
    for frame in coords:
        q_nh = distance_gradient(frame, N_ATOM, H_TRANSFER_ATOM)
        q_oh = distance_gradient(frame, O_ATOM, H_TRANSFER_ATOM)
        q_nh_dirs.append(q_nh)
        q_oh_dirs.append(q_oh)
        xi_dirs.append(q_nh - q_oh)
    return {
        "xi": np.stack(xi_dirs),
        "q_nh": np.stack(q_nh_dirs),
        "q_oh": np.stack(q_oh_dirs),
    }


def normalized_projection(vectors: np.ndarray, directions: np.ndarray) -> np.ndarray:
    flat_vectors = np.asarray(vectors, dtype=float).reshape(vectors.shape[0], -1)
    flat_dirs = np.asarray(directions, dtype=float).reshape(directions.shape[0], -1)
    norms = np.linalg.norm(flat_dirs, axis=1)
    return np.einsum("ij,ij->i", flat_vectors, flat_dirs) / np.maximum(norms, 1e-12)


def symmetrize(hessians: np.ndarray) -> np.ndarray:
    return 0.5 * (hessians + np.swapaxes(hessians, -1, -2))


def frobenius_relative_error(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    numer = np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)
    denom = np.linalg.norm(symmetrize(ref_h).reshape(ref_h.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def frobenius_absolute_error(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    return np.linalg.norm(diff.reshape(diff.shape[0], -1), axis=1)


def hessian_element_mae(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    return np.mean(np.abs(diff.reshape(diff.shape[0], -1)), axis=1)


def reaction_center_error(model_h: np.ndarray, ref_h: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    idx = np.asarray([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    model_block = symmetrize(model_h)[:, idx[:, None], idx]
    ref_block = symmetrize(ref_h)[:, idx[:, None], idx]
    numer = np.linalg.norm((model_block - ref_block).reshape(model_block.shape[0], -1), axis=1)
    denom = np.linalg.norm(ref_block.reshape(ref_block.shape[0], -1), axis=1)
    return numer / np.maximum(denom, 1e-12)


def reaction_center_absolute_error(model_h: np.ndarray, ref_h: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    idx = np.asarray([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    model_block = symmetrize(model_h)[:, idx[:, None], idx]
    ref_block = symmetrize(ref_h)[:, idx[:, None], idx]
    return np.linalg.norm((model_block - ref_block).reshape(model_block.shape[0], -1), axis=1)


def reaction_center_element_mae(model_h: np.ndarray, ref_h: np.ndarray, atoms: tuple[int, ...]) -> np.ndarray:
    idx = np.asarray([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    model_block = symmetrize(model_h)[:, idx[:, None], idx]
    ref_block = symmetrize(ref_h)[:, idx[:, None], idx]
    return np.mean(np.abs((model_block - ref_block).reshape(model_block.shape[0], -1)), axis=1)


def masses_from_z(atomic_numbers: np.ndarray) -> np.ndarray:
    return np.asarray([MASS_BY_Z[int(z)] for z in atomic_numbers], dtype=float)


def eckart_generators(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    xyz = np.asarray(coords, dtype=float).reshape(-1, 3)
    n_atoms = xyz.shape[0]
    sqrt_m = np.sqrt(masses)
    sqrt_m3 = np.repeat(sqrt_m, 3)
    center_of_mass = (xyz * masses[:, None]).sum(axis=0) / masses.sum()
    rel = xyz - center_of_mass[None, :]

    cols = []
    for axis in np.eye(3):
        col = sqrt_m3 * np.tile(axis, n_atoms)
        cols.append(col / max(float(np.linalg.norm(col)), 1e-12))

    rx, ry, rz = rel[:, 0], rel[:, 1], rel[:, 2]
    rotations = (
        np.stack([np.zeros_like(rx), -rz, ry], axis=1),
        np.stack([rz, np.zeros_like(ry), -rx], axis=1),
        np.stack([-ry, rx, np.zeros_like(rz)], axis=1),
    )
    for rot in rotations:
        col = (rot * sqrt_m[:, None]).reshape(-1)
        norm = float(np.linalg.norm(col))
        if norm > 1e-12:
            cols.append(col / norm)
    return np.stack(cols, axis=1)


def vibrational_basis(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    generators = eckart_generators(coords, masses)
    q, r = np.linalg.qr(generators, mode="reduced")
    rank = max(int((np.abs(np.diag(r)) > 1e-6).sum()), 1)
    u, _, _ = np.linalg.svd(q[:, :rank], full_matrices=True)
    return u[:, rank:]


def vibrational_eigh(
    hessian_ev_ang2: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masses = masses_from_z(atomic_numbers)
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


def compute_vib_diagnostics(method: MethodData, threshold: float) -> VibDiagnostics:
    eval_rows = []
    mode_rows = []
    for hessian, coords in zip(method.hessians_ev_ang2, method.coords_angstrom, strict=True):
        evals, modes = vibrational_eigh(hessian, coords, method.atomic_numbers)
        eval_rows.append(evals)
        mode_rows.append(modes)
    evals_arr = np.stack(eval_rows)
    return VibDiagnostics(
        evals=evals_arr,
        modes_mw=np.stack(mode_rows),
        n_negative=(evals_arr < threshold).sum(axis=1).astype(int),
    )


def load_orca_vib_or_compute(orca_cache: Path, method: MethodData, threshold: float) -> VibDiagnostics:
    with np.load(orca_cache) as data:
        if {"vib_evals_ev_ang2_amu", "vib_modes_mw"}.issubset(data.files):
            evals = np.asarray(data["vib_evals_ev_ang2_amu"], dtype=float)
            modes = np.asarray(data["vib_modes_mw"], dtype=float)
            return VibDiagnostics(
                evals=evals,
                modes_mw=modes,
                n_negative=(evals < threshold).sum(axis=1).astype(int),
            )
    return compute_vib_diagnostics(method, threshold)


def mass_weighted_direction_alignment(
    modes_mw: np.ndarray,
    directions_cart: np.ndarray,
    coords_angstrom: np.ndarray,
    atomic_numbers: np.ndarray,
) -> np.ndarray:
    masses = masses_from_z(atomic_numbers)
    sqrt_m3 = np.repeat(np.sqrt(masses), 3)
    alignments = []
    for mode, direction, coords in zip(modes_mw[:, :, 0], directions_cart, coords_angstrom, strict=True):
        direction_mw = direction.reshape(-1) * sqrt_m3
        q_vib = vibrational_basis(coords, masses)
        direction_mw = q_vib @ (q_vib.T @ direction_mw)
        denom = max(float(np.linalg.norm(mode) * np.linalg.norm(direction_mw)), 1e-12)
        alignments.append(abs(float(np.dot(mode, direction_mw))) / denom)
    return np.asarray(alignments, dtype=float)


def parse_atoms(text: str) -> tuple[int, ...]:
    atoms = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not atoms:
        raise ValueError("--reaction-center-atoms must contain at least one atom index")
    return atoms


def setup_axis(ax: plt.Axes, x_label: str) -> None:
    ax.set_xlabel(x_label)
    finish_axis(ax)


def save_png(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.savefig(path, dpi=dpi)
    print(path, flush=True)


def save_energy_plot(x: np.ndarray, x_label: str, methods: list[MethodData], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for method in methods:
        rel_kcal = (method.energies_ev - np.nanmin(method.energies_ev)) * EV_TO_KCALMOL
        sns.lineplot(x=x, y=rel_kcal, ax=ax, marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH, label=method.label, color=METHOD_COLORS.get(method.label))
    ax.set_ylabel(r"relative energy [kcal mol$^{-1}$]")
    setup_axis(ax, x_label)
    ax.legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_energy.png", dpi)
    plt.close(fig)


def save_force_projection_plot(
    x: np.ndarray,
    x_label: str,
    force_projections: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    specs = [
        ("tangent", r"MEP tangent, $F\cdot\hat t$"),
        ("xi", r"$q_\mathrm{NH}-q_\mathrm{OH}$"),
        ("q_nh", r"$q_\mathrm{NH}$"),
        ("q_oh", r"$q_\mathrm{OH}$"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.3), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), specs, strict=True):
        for label, projections in force_projections.items():
            sns.lineplot(
                x=x,
                y=projections[key],
                ax=ax,
                marker=METHOD_MARKERS.get(label, "o"),
                markersize=SMALL_MARKER_SIZE,
                lw=THIN_LINE_WIDTH,
                label=label,
                color=METHOD_COLORS.get(label),
            )
        ax.axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
        ax.set_title(title)
        ax.set_ylabel(r"projected force [eV $\AA^{-1}$]")
        setup_axis(ax, x_label)
    axes[0, 0].legend(fontsize=8, frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_projected_forces.png", dpi)
    plt.close(fig)


def save_lowest_eigenvalue_plot(
    x: np.ndarray,
    x_label: str,
    vib: dict[str, VibDiagnostics],
    n_eigs: int,
    output_dir: Path,
    dpi: int,
) -> None:
    n_eigs = min(n_eigs, min(diag.evals.shape[1] for diag in vib.values()))
    ncols = min(3, n_eigs)
    nrows = int(np.ceil(n_eigs / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.4 * nrows), sharex=True, squeeze=False)
    plot_labels = [DFT_LABEL] + [label for label in vib if label != DFT_LABEL]
    for idx, ax in enumerate(axes.ravel()):
        if idx >= n_eigs:
            ax.axis("off")
            continue
        ax.axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
        for label in plot_labels:
            diag = vib[label]
            sns.lineplot(
                x=x,
                y=diag.evals[:, idx],
                ax=ax,
                marker=METHOD_MARKERS.get(label, "o"),
                markersize=MARKER_SIZE,
                lw=THIN_LINE_WIDTH,
                linestyle=METHOD_LINESTYLES.get(label, "-"),
                label=label,
                color=METHOD_COLORS.get(label),
                zorder=2 if label == DFT_LABEL else 3,
            )
        ax.set_title(f"Mode {idx + 1}")
        ax.set_ylabel(r"$\lambda$ [eV $\AA^{-2}$ amu$^{-1}$]")
        setup_axis(ax, x_label)
    for idx, ax in enumerate(axes.ravel()):
        legend = ax.get_legend()
        if idx == 0:
            ax.legend(fontsize=16, markerscale=1.6, frameon=True, edgecolor="none")
        elif legend is not None:
            legend.remove()
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_lowest_hessian_eigenvalues.png", dpi)
    plt.close(fig)


def save_negative_mode_plot(
    x: np.ndarray,
    x_label: str,
    vib: dict[str, VibDiagnostics],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for label, diag in vib.items():
        y_values = diag.n_negative + NEGATIVE_MODE_DODGE.get(label, 0.0)
        ax.step(
            x,
            y_values,
            where="mid",
            lw=LINE_WIDTH,
            label=label,
            color=METHOD_COLORS.get(label),
        )
        sns.scatterplot(
            x=x,
            y=y_values,
            ax=ax,
            s=36,
            marker=METHOD_MARKERS.get(label, "o"),
            color=METHOD_COLORS.get(label),
            legend=False,
        )
    ax.set_ylabel("negative mode count")
    ax.yaxis.get_major_locator().set_params(integer=True)
    setup_axis(ax, x_label)
    ax.legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_negative_modes.png", dpi)
    plt.close(fig)


def save_hessian_error_plot(
    x: np.ndarray,
    x_label: str,
    errors: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    for ax, key, title in (
        (axes[0], "full", "Full Hessian Error vs DFT"),
        (axes[1], "reaction_center", "O-N-H Block Error vs DFT"),
    ):
        for label, values in errors.items():
            sns.lineplot(x=x, y=values[key], ax=ax, marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH, label=label, color=METHOD_COLORS.get(label))
        ax.set_title(title)
        ax.set_ylabel(r"$||H-H_\mathrm{DFT}||_F \ / \ ||H_\mathrm{DFT}||_F$")
        ax.set_yscale("log")
        setup_axis(ax, x_label)
    axes[0].legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_hessian_error_vs_dft.png", dpi)
    plt.close(fig)


def save_hessian_absolute_error_plot(
    x: np.ndarray,
    x_label: str,
    errors: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    for ax, key, title in (
        (axes[0], "full", "Full Hessian Absolute Error vs DFT"),
        (axes[1], "reaction_center", "O-N-H Block Absolute Error vs DFT"),
    ):
        for label, values in errors.items():
            sns.lineplot(x=x, y=values[key], ax=ax, marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH, label=label, color=METHOD_COLORS.get(label))
        ax.set_title(title)
        ax.set_ylabel(r"$||H-H_\mathrm{DFT}||_F$ [eV $\AA^{-2}$]")
        ax.set_yscale("log")
        setup_axis(ax, x_label)
    axes[0].legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_hessian_absolute_error_vs_dft.png", dpi)
    plt.close(fig)


def save_hessian_element_mae_plot(
    x: np.ndarray,
    x_label: str,
    errors: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True)
    for ax, key, title in (
        (axes[0], "full", "Full Hessian Element MAE vs DFT"),
        (axes[1], "reaction_center", "O-N-H Block Element MAE vs DFT"),
    ):
        for label, values in errors.items():
            sns.lineplot(x=x, y=values[key], ax=ax, marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH, label=label, color=METHOD_COLORS.get(label))
        ax.set_title(title)
        ax.set_ylabel(r"mean $|H-H_\mathrm{DFT}|$ [eV $\AA^{-2}$]")
        ax.set_yscale("log")
        setup_axis(ax, x_label)
    axes[0].legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_hessian_element_mae_vs_dft.png", dpi)
    plt.close(fig)


def save_mode_alignment_plot(
    x: np.ndarray,
    x_label: str,
    alignments: dict[str, dict[str, np.ndarray]],
    output_dir: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), sharex=True, sharey=True)
    for ax, key, title in (
        (axes[0], "tangent", "Lowest Mode Alignment With MEP Tangent"),
        (axes[1], "xi", r"Lowest Mode Alignment With $q_\mathrm{NH}-q_\mathrm{OH}$"),
    ):
        for label, values in alignments.items():
            sns.lineplot(x=x, y=values[key], ax=ax, marker="o", markersize=MARKER_SIZE, lw=LINE_WIDTH, label=label, color=METHOD_COLORS.get(label))
        ax.set_title(title)
        ax.set_ylabel(r"$|\cos\theta|$")
        ax.set_ylim(-0.02, 1.02)
        setup_axis(ax, x_label)
    axes[0].legend(frameon=True, edgecolor="none")
    fig.tight_layout(pad=0.01)
    save_png(fig, output_dir / "mep_mode_alignment.png", dpi)
    plt.close(fig)


def build_metrics_frame(
    x: np.ndarray,
    x_name: str,
    methods: list[MethodData],
    vib: dict[str, VibDiagnostics],
    force_projections: dict[str, dict[str, np.ndarray]],
    hessian_errors: dict[str, dict[str, np.ndarray]],
    hessian_absolute_errors: dict[str, dict[str, np.ndarray]],
    hessian_element_maes: dict[str, dict[str, np.ndarray]],
    alignments: dict[str, dict[str, np.ndarray]],
    n_eigs: int,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "frame": np.arange(x.size),
            x_name: x,
            "q_nh": methods[0].q_nh,
            "q_oh": methods[0].q_oh,
            "xi": methods[0].q_nh - methods[0].q_oh,
        }
    )
    for method in methods:
        prefix = safe_label(method.label)
        frame[f"{prefix}_relative_energy_kcalmol"] = (
            method.energies_ev - np.nanmin(method.energies_ev)
        ) * EV_TO_KCALMOL
        for key, values in force_projections[method.label].items():
            frame[f"{prefix}_force_{key}_ev_ang"] = values
        diag = vib[method.label]
        frame[f"{prefix}_n_negative"] = diag.n_negative
        for idx in range(min(n_eigs, diag.evals.shape[1])):
            frame[f"{prefix}_eig{idx}_ev_ang2_amu"] = diag.evals[:, idx]
        for key, values in alignments[method.label].items():
            frame[f"{prefix}_mode0_alignment_{key}"] = values
    for label, values in hessian_errors.items():
        prefix = safe_label(label)
        frame[f"{prefix}_hessian_relative_error"] = values["full"]
        frame[f"{prefix}_reaction_center_relative_error"] = values["reaction_center"]
    for label, values in hessian_absolute_errors.items():
        prefix = safe_label(label)
        frame[f"{prefix}_hessian_absolute_error_ev_ang2"] = values["full"]
        frame[f"{prefix}_reaction_center_absolute_error_ev_ang2"] = values["reaction_center"]
    for label, values in hessian_element_maes.items():
        prefix = safe_label(label)
        frame[f"{prefix}_hessian_element_mae_ev_ang2"] = values["full"]
        frame[f"{prefix}_reaction_center_element_mae_ev_ang2"] = values["reaction_center"]
    return frame


def main() -> None:
    args = parse_args()
    mep_dir = args.mep_dir
    output_dir = args.output_dir or Path("plots") / mep_dir.name / "mep_diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    orca_path = args.orca_cache or mep_dir / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or mep_dir / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or mep_dir / "eqv2_autograd_arrays.npz"

    dft = load_orca(orca_path)
    hip = load_model(HIP_LABEL, hip_path)
    eqv2 = load_model(EQV2_LABEL, eqv2_path)
    methods = [dft, hip, eqv2]
    validate_same_grid(dft, [hip, eqv2])

    frame = np.arange(dft.coords_angstrom.shape[0])
    xi = dft.q_nh - dft.q_oh
    x = xi if args.x_axis == "xi" else frame
    x_name = "xi" if args.x_axis == "xi" else "frame"
    x_label = (
        r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"
        if args.x_axis == "xi"
        else "frame index"
    )

    tangent = finite_difference_tangent(dft.coords_angstrom)
    dirs = cv_directions(dft.coords_angstrom)
    projection_dirs = {"tangent": tangent, **dirs}
    force_projections = {
        method.label: {
            key: normalized_projection(method.forces_ev_ang, direction)
            for key, direction in projection_dirs.items()
        }
        for method in methods
    }

    print("Computing vibrational diagnostics for MLIP Hessians...", flush=True)
    vib = {
        DFT_LABEL: load_orca_vib_or_compute(orca_path, dft, args.negative_threshold),
        HIP_LABEL: compute_vib_diagnostics(hip, args.negative_threshold),
        EQV2_LABEL: compute_vib_diagnostics(eqv2, args.negative_threshold),
    }

    reaction_center_atoms = parse_atoms(args.reaction_center_atoms)
    hessian_errors = {
        HIP_LABEL: {
            "full": frobenius_relative_error(hip.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_error(
                hip.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
        EQV2_LABEL: {
            "full": frobenius_relative_error(eqv2.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_error(
                eqv2.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
    }
    hessian_absolute_errors = {
        HIP_LABEL: {
            "full": frobenius_absolute_error(hip.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_absolute_error(
                hip.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
        EQV2_LABEL: {
            "full": frobenius_absolute_error(eqv2.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_absolute_error(
                eqv2.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
    }
    hessian_element_maes = {
        HIP_LABEL: {
            "full": hessian_element_mae(hip.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_element_mae(
                hip.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
        EQV2_LABEL: {
            "full": hessian_element_mae(eqv2.hessians_ev_ang2, dft.hessians_ev_ang2),
            "reaction_center": reaction_center_element_mae(
                eqv2.hessians_ev_ang2, dft.hessians_ev_ang2, reaction_center_atoms
            ),
        },
    }

    alignments = {
        label: {
            "tangent": mass_weighted_direction_alignment(
                diag.modes_mw, tangent, dft.coords_angstrom, dft.atomic_numbers
            ),
            "xi": mass_weighted_direction_alignment(
                diag.modes_mw, dirs["xi"], dft.coords_angstrom, dft.atomic_numbers
            ),
        }
        for label, diag in vib.items()
    }

    save_energy_plot(x, x_label, methods, output_dir, args.dpi)
    save_force_projection_plot(x, x_label, force_projections, output_dir, args.dpi)
    save_lowest_eigenvalue_plot(x, x_label, vib, args.n_eigs, output_dir, args.dpi)
    save_negative_mode_plot(x, x_label, vib, output_dir, args.dpi)
    save_hessian_error_plot(x, x_label, hessian_errors, output_dir, args.dpi)
    save_hessian_absolute_error_plot(x, x_label, hessian_absolute_errors, output_dir, args.dpi)
    save_hessian_element_mae_plot(x, x_label, hessian_element_maes, output_dir, args.dpi)
    save_mode_alignment_plot(x, x_label, alignments, output_dir, args.dpi)

    metrics = build_metrics_frame(
        x=x,
        x_name=x_name,
        methods=methods,
        vib=vib,
        force_projections=force_projections,
        hessian_errors=hessian_errors,
        hessian_absolute_errors=hessian_absolute_errors,
        hessian_element_maes=hessian_element_maes,
        alignments=alignments,
        n_eigs=args.n_eigs,
    )
    metrics_path = output_dir / "mep_diagnostics_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print(f"Wrote plots to {output_dir}", flush=True)
    print(f"Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
