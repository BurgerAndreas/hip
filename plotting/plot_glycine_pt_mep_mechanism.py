#!/usr/bin/env python
"""DFT-referenced AD/HIP mechanism figures for the glycine proton transfer.

Unlike ``plot_glycine_pt_path_mechanism.py`` (which runs on the synthetic rigid
proton-slide path and has no DFT reference, so it estimates the "smooth" force by
polynomial detrending), this script consumes a path where ORCA forces and
Hessians exist. By default that is the 150-point rigid proton-slide line in
``runs/glycine_pt_path_n150``. That lets us use
DFT directly as the baseline instead of a polynomial trend:

    residual(model) = (F_model . t_hat) - (F_DFT . t_hat)

Figures (x-axis is the reaction coordinate xi = q_NH - q_OH):

- ``mep_ad_energy_force.png``       : energy and projected force g = F.t_hat for DFT/AD/HIP.
- ``mep_force_residual_vs_dft.png`` : g overlaid + DFT-referenced residual g_model - g_DFT.
- ``mep_curvature_vs_dft.png``      : directional curvature kappa = t_hat^T H t_hat for the
                                      AD autograd Hessian, HIP Hessian, the finite-difference
                                      derivative of the AD force (-dg/ds along arc length s),
                                      and the DFT Hessian; plus the model-vs-DFT and
                                      AD-autograd-vs-AD-FD gaps.

Note on the derivative variable: kappa = t_hat^T H t_hat is curvature per unit
*Cartesian* displacement, so the consistent finite-difference comparison is
-dg/ds with s the Cartesian arc length along the path (not d/dxi).
"""
from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

from plot_style import (  # noqa: E402
    ACCENT_COLOR,
    AD_COLOR,
    DFT_COLOR,
    HIP_COLOR,
    LEFTNET_CF_FORCE_COLOR,
    LINE_WIDTH,
    MARKER_SIZE,
    PLOTLY_FONT_COLOR,
    THIN_LINE_WIDTH,
    apply_invisible_ticks,
    finish_axis,
)

HARTREE_TO_EV = 27.211386245988

DEFAULT_MEP_DIR = Path("runs/glycine_pt_mep_145")

DFT_LABEL = "DFT"
HIP_LABEL = "HIP"
AD_LABEL = "EqV2"
AD_ORIG_LABEL = "EqV2 (no H)"
LEFTNET_CF_LABEL = "LeftNet-CF"
LEFTNET_CF_ORIG_LABEL = "LeftNet-CF (no H)"

METHOD_COLORS = {
    DFT_LABEL: DFT_COLOR,
    HIP_LABEL: HIP_COLOR,
    AD_LABEL: AD_COLOR,
    AD_ORIG_LABEL: AD_COLOR,
    LEFTNET_CF_LABEL: LEFTNET_CF_FORCE_COLOR,
    LEFTNET_CF_ORIG_LABEL: LEFTNET_CF_FORCE_COLOR,
}
ORIG_COMPARE_LABELS = {AD_ORIG_LABEL, LEFTNET_CF_ORIG_LABEL}
XLABEL = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"
FORCE_PROJ = r"g = F\cdot\hat t"
FORCE_PROJ_XI = r"F\cdot\widehat{\xi}"
O_ATOM = 3
N_ATOM = 4
H_TRANSFER_ATOM = 9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mep-dir", type=Path, default=DEFAULT_MEP_DIR)
    parser.add_argument("--orca-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--leftnet-cf-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-orig-arrays", type=Path, default=None)
    parser.add_argument("--leftnet-cf-orig-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument(
        "--print-denser-run",
        action="store_true",
        help="Print one-line commands to prepare a denser MEP, run ORCA/MLIPs, cache ORCA, and plot.",
    )
    parser.add_argument(
        "--dense-frames-per-leg",
        type=int,
        default=16,
        help="Frames per geodesic leg for --print-denser-run. Default doubles the 73-frame path to 145 frames.",
    )
    parser.add_argument(
        "--dense-mep-initial-images",
        type=int,
        default=10,
        help="Initial full NEB image count used to infer the number of geodesic legs.",
    )
    parser.add_argument(
        "--dense-run-name",
        default=None,
        help="Run stem for --print-denser-run. Defaults to glycine_pt_mep_<estimated frame count>.",
    )
    return parser.parse_args()


def qcmd(parts: list[str | Path]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def print_denser_run_commands(args: argparse.Namespace) -> None:
    n_legs = args.dense_mep_initial_images - 1
    if n_legs <= 0:
        raise ValueError("--dense-mep-initial-images must be at least 2")
    n_frames = args.dense_frames_per_leg * n_legs + 1
    run_name = args.dense_run_name or f"glycine_pt_mep_{n_frames}"
    render_dir = Path("runs") / f"{run_name}_xyzrender"
    mep_dir = Path("runs") / run_name
    trajectory_xyz = render_dir / "xyz" / "reaction_path.xyz"
    array_range = f"0-{n_frames - 1}%20"
    hip_export = ",".join(
        [
            "ALL",
            f"OUTPUT_DIR={mep_dir}",
            f"TRAJECTORY_XYZ={trajectory_xyz}",
            "CHECKPOINT=ckpt/hip_v2.ckpt",
            "HESSIAN_METHOD=predict",
            "MODEL_LABEL=hip_v2",
            "OUTPUT_PREFIX=hip_v2",
        ]
    )
    eqv2_export = ",".join(
        [
            "ALL",
            f"OUTPUT_DIR={mep_dir}",
            f"TRAJECTORY_XYZ={trajectory_xyz}",
            "CHECKPOINT=ckpt/eqv2.ckpt",
            "HESSIAN_METHOD=autograd",
            "MODEL_LABEL=eqv2_autograd",
            "OUTPUT_PREFIX=eqv2_autograd",
        ]
    )

    commands = [
        qcmd(
            [
                "uv",
                "run",
                "python",
                "plotting/visualize_glycine_pt_xyzrender.py",
                "--frames-per-leg",
                str(args.dense_frames_per_leg),
                "--output-dir",
                render_dir,
            ]
        ),
        qcmd(
            [
                "uv",
                "run",
                "python",
                "scripts/glycine_pt_mep_hessian_scan.py",
                "--trajectory-xyz",
                trajectory_xyz,
                "--output-dir",
                mep_dir,
                "--skip-model",
            ]
        ),
        qcmd(
            [
                "sbatch",
                f"--array={array_range}",
                f"--export=ALL,SCAN_DIR={mep_dir}",
                "scripts/run_glycine_pt_orca_array.sbatch",
            ]
        ),
        qcmd(
            [
                "sbatch",
                f"--export={hip_export}",
                "scripts/run_glycine_pt_mep_mlip.sbatch",
            ]
        ),
        qcmd(
            [
                "sbatch",
                f"--export={eqv2_export}",
                "scripts/run_glycine_pt_mep_mlip.sbatch",
            ]
        ),
        qcmd(
            [
                "uv",
                "run",
                "python",
                "scripts/cache_glycine_pt_orca_vibrations.py",
                "--scan-dir",
                mep_dir,
            ]
        ),
        qcmd(["uv", "run", "python", "plotting/plot_glycine_pt_mep_mechanism.py", "--mep-dir", mep_dir]),
        qcmd(["uv", "run", "python", "plotting/plot_glycine_pt_mep_73_diagnostics.py", "--mep-dir", mep_dir]),
    ]

    print(f"# Estimated frames: {n_frames}")
    print(f"# Denser MEP dir: {mep_dir}")
    for command in commands:
        print(command)


def style_axis(ax: plt.Axes) -> None:
    finish_axis(ax)
    apply_invisible_ticks(ax)


def savefig(
    fig: plt.Figure,
    path: Path,
    dpi: int,
    *,
    bottom: float | None = None,
    tight_pad: float = 0.4,
    wspace: float | None = None,
    bbox_inches: str | None = None,
    pad_inches: float = 0.02,
) -> None:
    fig.tight_layout(pad=tight_pad)
    if wspace is not None:
        fig.subplots_adjust(wspace=wspace)
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    save_kwargs: dict[str, object] = {}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
        save_kwargs["pad_inches"] = pad_inches
    fig.savefig(path, dpi=dpi, **save_kwargs)
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def finite_difference_tangent(coords: np.ndarray) -> np.ndarray:
    """Unit Cartesian tangent (n_frames, n_atoms, 3) via central differences."""
    tangent = np.empty_like(coords, dtype=float)
    tangent[0] = coords[1] - coords[0]
    tangent[-1] = coords[-1] - coords[-2]
    tangent[1:-1] = coords[2:] - coords[:-2]
    flat = tangent.reshape(tangent.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1)
    return tangent / np.maximum(norms[:, None, None], 1e-12)


def arc_length(coords: np.ndarray) -> np.ndarray:
    """Cumulative Cartesian arc length [Angstrom] along the ordered path."""
    flat = coords.reshape(coords.shape[0], -1)
    steps = np.linalg.norm(np.diff(flat, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def projected_force(forces: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    flat_f = forces.reshape(forces.shape[0], -1)
    flat_t = tangent.reshape(tangent.shape[0], -1)
    return np.einsum("ij,ij->i", flat_f, flat_t)


def distance_gradient(coords: np.ndarray, atom_a: int, atom_b: int) -> np.ndarray:
    grad = np.zeros_like(coords, dtype=float)
    vec = coords[atom_a] - coords[atom_b]
    dist = max(float(np.linalg.norm(vec)), 1e-12)
    unit = vec / dist
    grad[atom_a] = unit
    grad[atom_b] = -unit
    return grad


def xi_unit_direction(coords: np.ndarray) -> np.ndarray:
    """Unit Cartesian direction of the proton-transfer CV gradient, frame-wise."""
    directions = np.empty_like(coords, dtype=float)
    for i, frame in enumerate(coords):
        xi_dir = distance_gradient(frame, N_ATOM, H_TRANSFER_ATOM) - distance_gradient(
            frame, O_ATOM, H_TRANSFER_ATOM
        )
        flat = xi_dir.reshape(-1)
        directions[i] = xi_dir / max(float(np.linalg.norm(flat)), 1e-12)
    return directions


def directional_curvature(hessians: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """kappa_i = t_hat_i^T H_i t_hat_i with H symmetrized [eV/Angstrom^2]."""
    flat_t = tangent.reshape(tangent.shape[0], -1)
    sym = 0.5 * (hessians + np.swapaxes(hessians, -1, -2))
    return np.einsum("ij,ijk,ik->i", flat_t, sym, flat_t)


class MethodArrays:
    def __init__(self, label: str, energies_ev: np.ndarray, forces: np.ndarray, hessians: np.ndarray) -> None:
        self.label = label
        self.energies_ev = energies_ev
        self.forces = forces
        self.hessians = hessians


def load_methods(
    orca_path: Path,
    hip_path: Path,
    eqv2_path: Path,
    leftnet_cf_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, MethodArrays]]:
    with np.load(orca_path) as data:
        coords = np.asarray(data["coords_angstrom"], dtype=float)
        q_nh = np.asarray(data["q_nh"], dtype=float)
        q_oh = np.asarray(data["q_oh"], dtype=float)
        energy_key = "energy_hartree_engrad" if "energy_hartree_engrad" in data.files else "energy_hartree"
        dft = MethodArrays(
            DFT_LABEL,
            np.asarray(data[energy_key], dtype=float) * HARTREE_TO_EV,
            np.asarray(data["forces_ev_ang"], dtype=float),
            np.asarray(data["hessian_ev_ang2"], dtype=float),
        )
    with np.load(hip_path) as data:
        hip = MethodArrays(
            HIP_LABEL,
            np.asarray(data["energies"], dtype=float),
            np.asarray(data["forces"], dtype=float),
            np.asarray(data["hessians_cartesian"], dtype=float),
        )
    with np.load(eqv2_path) as data:
        ad = MethodArrays(
            AD_LABEL,
            np.asarray(data["energies"], dtype=float),
            np.asarray(data["forces"], dtype=float),
            np.asarray(data["hessians_cartesian"], dtype=float),
        )
    leftnet_cf = None
    if leftnet_cf_path is not None and leftnet_cf_path.exists():
        with np.load(leftnet_cf_path) as data:
            leftnet_cf = MethodArrays(
                LEFTNET_CF_LABEL,
                np.asarray(data["energies"], dtype=float),
                np.asarray(data["forces"], dtype=float),
                np.asarray(data["hessians_cartesian"], dtype=float),
            )
    xi = q_nh - q_oh
    methods = {DFT_LABEL: dft, HIP_LABEL: hip, AD_LABEL: ad}
    if leftnet_cf is not None:
        methods[LEFTNET_CF_LABEL] = leftnet_cf
    return xi, coords, methods


def hessian_element_mae(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    sym_model = 0.5 * (model_h + np.swapaxes(model_h, -1, -2))
    sym_ref = 0.5 * (ref_h + np.swapaxes(ref_h, -1, -2))
    diff = sym_model - sym_ref
    return np.mean(np.abs(diff.reshape(diff.shape[0], -1)), axis=1)


def load_optional_method(path: Path | None, label: str) -> MethodArrays | None:
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        forces = np.asarray(data["forces"], dtype=float)
        n_frames = forces.shape[0]
        energies = (
            np.asarray(data["energies"], dtype=float)
            if "energies" in data.files
            else np.zeros(n_frames, dtype=float)
        )
        hessians = (
            np.asarray(data["hessians_cartesian"], dtype=float)
            if "hessians_cartesian" in data.files
            else np.zeros((n_frames, 1, 1), dtype=float)
        )
        return MethodArrays(label, energies, forces, hessians)


def hessian_mae_for_labels(
    compare_labels: tuple[str, ...],
    methods: dict[str, MethodArrays],
    optional_models: dict[str, MethodArrays],
) -> dict[str, np.ndarray]:
    dft_h = methods[DFT_LABEL].hessians
    mae: dict[str, np.ndarray] = {}
    for label in compare_labels:
        method = methods.get(label) or optional_models.get(label)
        if method is None or method.hessians.ndim < 3 or method.hessians.shape[-1] == 1:
            continue
        mae[label] = hessian_element_mae(method.hessians, dft_h)
    return mae


def reorder_method(method: MethodArrays, order: np.ndarray) -> None:
    method.energies_ev = method.energies_ev[order]
    method.forces = method.forces[order]
    method.hessians = method.hessians[order]


def plot_energy_force(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    methods: dict[str, MethodArrays],
    g: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))
    for label in methods:
        e = methods[label].energies_ev - np.nanmin(methods[label].energies_ev)
        sns.lineplot(x=xi, y=e, ax=axes[0], color=METHOD_COLORS[label], lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=label)
    axes[0].set_ylabel(r"$E - \min E$ [eV]")
    axes[0].set_xlabel(XLABEL)
    axes[0].legend(fontsize=10, frameon=True, edgecolor="none")

    for label in methods:
        sns.lineplot(x=xi, y=g[label], ax=axes[1], color=METHOD_COLORS[label], lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=label)
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_ylabel(rf"${FORCE_PROJ}$ [eV/$\AA$]")
    axes[1].set_xlabel(XLABEL)
    axes[1].legend(fontsize=10, frameon=True, edgecolor="none")

    for ax in axes:
        style_axis(ax)
    savefig(fig, output_dir / "mep_ad_energy_force.png", dpi)


def plot_force_residual(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    g: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    for label in g:
        sns.lineplot(x=xi, y=g[label], ax=axes[0], color=METHOD_COLORS[label], lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=label)
    axes[0].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[0].set_ylabel(rf"${FORCE_PROJ}$ [eV/$\AA$]")
    axes[0].set_xlabel(XLABEL)
    axes[0].legend(fontsize=10, frameon=True, edgecolor="none")

    for label in (AD_LABEL, HIP_LABEL, LEFTNET_CF_LABEL):
        if label not in g:
            continue
        residual = g[label] - g[DFT_LABEL]
        sns.lineplot(x=xi, y=residual, ax=axes[1], color=METHOD_COLORS[label], lw=THIN_LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=f"{label} $-$ DFT")
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_ylabel(rf"$g_\mathrm{{model}} - g_\mathrm{{DFT}}$ [eV/$\AA$]")
    axes[1].set_xlabel(XLABEL)
    axes[1].legend(fontsize=10, frameon=True, edgecolor="none")

    for ax in axes:
        style_axis(ax)
    savefig(fig, output_dir / "mep_force_residual_vs_dft.png", dpi)


AD_EMPHASIS_LINE_WIDTH = LINE_WIDTH + 0.4
NO_HIP_LEGEND_FONTSIZE = 14
FOUR_MODEL_LEGEND_FONTSIZE = 12
NO_HIP_X_PAD_FRACTION = 0.015
COMPACT_TIGHT_PAD = 0.02
COMPACT_PAD_INCHES = 0.01
COMPACT_WSPACE_2PANEL = 0.18
COMPACT_WSPACE_3PANEL = 0.20


def style_no_hip_legend(
    ax: plt.Axes,
    *,
    fontsize: int,
    labelcolor: str | None = None,
    loc: str = "best",
    bbox_to_anchor: tuple[float, float] | None = None,
) -> None:
    legend_kwargs = {"fontsize": fontsize, "frameon": True, "edgecolor": "none", "loc": loc}
    if bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = bbox_to_anchor
    legend = ax.legend(**legend_kwargs)
    if labelcolor is not None:
        for text in legend.get_texts():
            text.set_color(labelcolor)


def tight_xlim(x: np.ndarray, pad_fraction: float = NO_HIP_X_PAD_FRACTION) -> tuple[float, float]:
    x_span = float(np.max(x) - np.min(x))
    pad = max(pad_fraction * x_span, 0.01)
    return float(np.min(x) - pad), float(np.max(x) + pad)


def plot_force_residual_no_hip(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    g: dict[str, np.ndarray],
    *,
    compare_labels: tuple[str, ...] = (AD_LABEL,),
    linestyles: dict[str, str] | None = None,
    hessian_mae: dict[str, np.ndarray] | None = None,
    force_label: str = FORCE_PROJ,
    residual_ylabel: str = r"g_\mathrm{model} - g_\mathrm{DFT}",
    filename: str = "mep_force_residual_vs_dft_no_hip.png",
    legend_fontsize: int = NO_HIP_LEGEND_FONTSIZE,
    legend_labelcolor: str | None = None,
    legend_loc: str = "best",
    legend_bbox_to_anchor: tuple[float, float] | None = None,
    method_colors: dict[str, str] | None = None,
    tight_pad: float = 0.4,
    wspace: float | None = None,
    bbox_inches: str | None = None,
    pad_inches: float = 0.02,
) -> None:
    styles = linestyles or {}
    colors = {**METHOD_COLORS, **(method_colors or {})}
    ncols = 3 if hessian_mae else 2
    fig, axes = plt.subplots(1, ncols, figsize=(16.5 if ncols == 3 else 11.5, 4.0))
    if ncols == 2:
        axes = np.asarray(axes)

    sns.lineplot(
        x=xi,
        y=g[DFT_LABEL],
        ax=axes[0],
        color=colors[DFT_LABEL],
        lw=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        label=DFT_LABEL,
    )
    for label in compare_labels:
        if label not in g:
            continue
        lw = AD_EMPHASIS_LINE_WIDTH if label == AD_LABEL else LINE_WIDTH
        sns.lineplot(
            x=xi,
            y=g[label],
            ax=axes[0],
            color=colors[label],
            lw=lw,
            linestyle=styles.get(label, "-"),
            label=label,
        )
    axes[0].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[0].set_ylabel(rf"${force_label}$ [eV/$\AA$]")
    axes[0].set_xlabel(XLABEL)
    style_no_hip_legend(
        axes[0],
        fontsize=legend_fontsize,
        labelcolor=legend_labelcolor,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor,
    )

    for label in compare_labels:
        if label not in g:
            continue
        lw = AD_EMPHASIS_LINE_WIDTH if label == AD_LABEL else LINE_WIDTH
        sns.lineplot(
            x=xi,
            y=g[label] - g[DFT_LABEL],
            ax=axes[1],
            color=colors[label],
            lw=lw,
            linestyle=styles.get(label, "-"),
            label="_nolegend_",
        )
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_ylabel(rf"${residual_ylabel}$ [eV/$\AA$]")
    axes[1].set_xlabel(XLABEL)

    if hessian_mae:
        for label in compare_labels:
            if label not in hessian_mae:
                continue
            lw = AD_EMPHASIS_LINE_WIDTH if label == AD_LABEL else LINE_WIDTH
            sns.lineplot(
                x=xi,
                y=hessian_mae[label],
                ax=axes[2],
                color=colors[label],
                lw=lw,
                linestyle=styles.get(label, "-"),
                label="_nolegend_",
            )
        axes[2].set_ylabel(r"mean $|H-H_\mathrm{DFT}|$ [eV $\AA^{-2}$]")
        axes[2].set_xlabel(XLABEL)
        axes[2].set_yscale("log")

    shared_xlim = tight_xlim(xi)
    for ax in axes:
        ax.set_xlim(*shared_xlim)
        style_axis(ax)
    savefig(
        fig,
        output_dir / filename,
        dpi,
        tight_pad=tight_pad,
        wspace=wspace,
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )


def plot_curvature(
    output_dir: Path,
    dpi: int,
    xi: np.ndarray,
    kappa_ad_auto: np.ndarray,
    kappa_hip_auto: np.ndarray,
    kappa_ad_fd: np.ndarray,
    kappa_dft: np.ndarray,
    kappa_leftnet_cf: np.ndarray | None = None,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    sns.lineplot(x=xi, y=kappa_dft, ax=axes[0], color=DFT_COLOR, lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"DFT $\hat t^\top H\hat t$")
    sns.lineplot(x=xi, y=kappa_ad_auto, ax=axes[0], color=AD_COLOR, lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"EqV2 $\hat t^\top H\hat t$")
    sns.lineplot(x=xi, y=kappa_hip_auto, ax=axes[0], color=HIP_COLOR, lw=LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"HIP $\hat t^\top H\hat t$")
    if kappa_leftnet_cf is not None:
        sns.lineplot(
            x=xi,
            y=kappa_leftnet_cf,
            ax=axes[0],
            color=LEFTNET_CF_FORCE_COLOR,
            lw=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=r"LeftNet-CF $\hat t^\top H\hat t$",
        )
    sns.lineplot(x=xi, y=kappa_ad_fd, ax=axes[0], color=ACCENT_COLOR, lw=THIN_LINE_WIDTH, marker="o", markersize=MARKER_SIZE, alpha=0.8, label=r"EqV2 force FD $-dg/ds$")
    axes[0].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[0].set_ylabel(r"$\kappa$ [eV/$\AA^2$]")
    axes[0].set_xlabel(XLABEL)
    axes[0].legend(fontsize=9, frameon=True, edgecolor="none")

    sns.lineplot(x=xi, y=kappa_ad_auto - kappa_dft, ax=axes[1], color=AD_COLOR, lw=THIN_LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"EqV2 autograd $-$ DFT")
    sns.lineplot(x=xi, y=kappa_hip_auto - kappa_dft, ax=axes[1], color=HIP_COLOR, lw=THIN_LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"HIP $-$ DFT")
    if kappa_leftnet_cf is not None:
        sns.lineplot(
            x=xi,
            y=kappa_leftnet_cf - kappa_dft,
            ax=axes[1],
            color=LEFTNET_CF_FORCE_COLOR,
            lw=THIN_LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            label=r"LeftNet-CF $-$ DFT",
        )
    sns.lineplot(x=xi, y=kappa_ad_auto - kappa_ad_fd, ax=axes[1], color=ACCENT_COLOR, lw=THIN_LINE_WIDTH, marker="o", markersize=MARKER_SIZE, label=r"EqV2 autograd $-$ EqV2 FD")
    axes[1].axhline(0.0, color="grey", lw=THIN_LINE_WIDTH)
    axes[1].set_ylabel(r"$\Delta\kappa$ [eV/$\AA^2$]")
    axes[1].set_xlabel(XLABEL)
    axes[1].legend(fontsize=9, frameon=True, edgecolor="none")

    for ax in axes:
        style_axis(ax)
    fig.text(
        0.5,
        0.005,
        r"FD curvature uses arc length $s$; the EqV2$-$EqV2 FD gap also includes path curvature $d\hat t/ds$",
        fontsize=7.5,
        va="bottom",
        ha="center",
        color="grey",
    )
    savefig(fig, output_dir / "mep_curvature_vs_dft.png", dpi, bottom=0.16)


def main() -> None:
    args = parse_args()
    if args.print_denser_run:
        print_denser_run_commands(args)
        return

    mep_dir = args.mep_dir
    output_dir = args.output_dir or Path("plots") / mep_dir.name / "mep_mechanism"
    output_dir.mkdir(parents=True, exist_ok=True)

    orca_path = args.orca_cache or mep_dir / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or mep_dir / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or mep_dir / "eqv2_autograd_arrays.npz"
    leftnet_cf_path = args.leftnet_cf_arrays
    if leftnet_cf_path is None and (mep_dir / "leftnet_cf_arrays.npz").exists():
        leftnet_cf_path = mep_dir / "leftnet_cf_arrays.npz"
    eqv2_orig_path = args.eqv2_orig_arrays
    if eqv2_orig_path is None and (mep_dir / "eqv2_orig_autograd_arrays.npz").exists():
        eqv2_orig_path = mep_dir / "eqv2_orig_autograd_arrays.npz"
    leftnet_cf_orig_path = args.leftnet_cf_orig_arrays
    if leftnet_cf_orig_path is None and (mep_dir / "leftnet_cf_orig_arrays.npz").exists():
        leftnet_cf_orig_path = mep_dir / "leftnet_cf_orig_arrays.npz"

    xi, coords, methods = load_methods(orca_path, hip_path, eqv2_path, leftnet_cf_path)

    order = np.argsort(xi)
    xi = xi[order]
    coords = coords[order]
    for method in methods.values():
        reorder_method(method, order)

    optional_models: dict[str, MethodArrays] = {}
    for label, path in (
        (AD_ORIG_LABEL, eqv2_orig_path),
        (LEFTNET_CF_ORIG_LABEL, leftnet_cf_orig_path),
    ):
        method = load_optional_method(path, label)
        if method is not None:
            reorder_method(method, order)
            optional_models[label] = method

    tangent = finite_difference_tangent(coords)
    xi_dir = xi_unit_direction(coords)
    s = arc_length(coords)

    g = {label: projected_force(method.forces, tangent) for label, method in methods.items()}
    g_xi = {label: projected_force(method.forces, xi_dir) for label, method in methods.items()}
    for label, method in optional_models.items():
        g_xi[label] = projected_force(method.forces, xi_dir)

    kappa_ad_auto = directional_curvature(methods[AD_LABEL].hessians, tangent)
    kappa_dft = directional_curvature(methods[DFT_LABEL].hessians, tangent)
    kappa_hip = directional_curvature(methods[HIP_LABEL].hessians, tangent)
    kappa_leftnet_cf = (
        directional_curvature(methods[LEFTNET_CF_LABEL].hessians, tangent) if LEFTNET_CF_LABEL in methods else None
    )
    kappa_ad_fd = -np.gradient(g[AD_LABEL], s)
    kappa_leftnet_cf_fd = -np.gradient(g[LEFTNET_CF_LABEL], s) if LEFTNET_CF_LABEL in g else None

    plot_energy_force(output_dir, args.dpi, xi, methods, g)
    plot_force_residual(output_dir, args.dpi, xi, g)
    plot_force_residual_no_hip(output_dir, args.dpi, xi, g)
    plot_force_residual_no_hip(
        output_dir,
        args.dpi,
        xi,
        g_xi,
        force_label=FORCE_PROJ_XI,
        residual_ylabel=r"(F\cdot\widehat{\xi})_\mathrm{model} - (F\cdot\widehat{\xi})_\mathrm{DFT}",
        filename="mep_force_xi_residual_vs_dft_no_hip.png",
    )
    if LEFTNET_CF_LABEL in g_xi:
        leftnet_cf_compare = (AD_LABEL, LEFTNET_CF_LABEL)
        plot_force_residual_no_hip(
            output_dir,
            args.dpi,
            xi,
            g_xi,
            compare_labels=leftnet_cf_compare,
            force_label=FORCE_PROJ_XI,
            residual_ylabel=r"(F\cdot\widehat{\xi})_\mathrm{model} - (F\cdot\widehat{\xi})_\mathrm{DFT}",
            filename="mep_force_xi_residual_vs_dft_no_hip_leftnet_cf.png",
            tight_pad=COMPACT_TIGHT_PAD,
            wspace=COMPACT_WSPACE_2PANEL,
            bbox_inches="tight",
            pad_inches=COMPACT_PAD_INCHES,
        )
        plot_force_residual_no_hip(
            output_dir,
            args.dpi,
            xi,
            g_xi,
            compare_labels=leftnet_cf_compare,
            hessian_mae=hessian_mae_for_labels(leftnet_cf_compare, methods, optional_models),
            force_label=FORCE_PROJ_XI,
            residual_ylabel=r"(F\cdot\widehat{\xi})_\mathrm{model} - (F\cdot\widehat{\xi})_\mathrm{DFT}",
            filename="mep_force_xi_residual_vs_dft_no_hip_leftnet_cf_hessian_mae.png",
            tight_pad=COMPACT_TIGHT_PAD,
            wspace=COMPACT_WSPACE_3PANEL,
            bbox_inches="tight",
            pad_inches=COMPACT_PAD_INCHES,
        )
        orig_compare = tuple(
            label
            for label in (AD_LABEL, AD_ORIG_LABEL, LEFTNET_CF_LABEL, LEFTNET_CF_ORIG_LABEL)
            if label in g_xi
        )
        if any(label in ORIG_COMPARE_LABELS for label in orig_compare):
            orig_linestyles = {label: ":" for label in orig_compare if label in ORIG_COMPARE_LABELS}
            plot_force_residual_no_hip(
                output_dir,
                args.dpi,
                xi,
                g_xi,
                compare_labels=orig_compare,
                linestyles=orig_linestyles,
                force_label=FORCE_PROJ_XI,
                residual_ylabel=r"(F\cdot\widehat{\xi})_\mathrm{model} - (F\cdot\widehat{\xi})_\mathrm{DFT}",
                filename="mep_force_xi_residual_vs_dft_no_hip_leftnet_cf_orig.png",
                legend_fontsize=FOUR_MODEL_LEGEND_FONTSIZE,
                legend_labelcolor=PLOTLY_FONT_COLOR,
                legend_loc="upper left",
                legend_bbox_to_anchor=(0.0, 1.01),
                tight_pad=COMPACT_TIGHT_PAD,
                wspace=COMPACT_WSPACE_2PANEL,
                bbox_inches="tight",
                pad_inches=COMPACT_PAD_INCHES,
            )
            plot_force_residual_no_hip(
                output_dir,
                args.dpi,
                xi,
                g_xi,
                compare_labels=orig_compare,
                linestyles=orig_linestyles,
                hessian_mae=hessian_mae_for_labels(orig_compare, methods, optional_models),
                force_label=FORCE_PROJ_XI,
                residual_ylabel=r"(F\cdot\widehat{\xi})_\mathrm{model} - (F\cdot\widehat{\xi})_\mathrm{DFT}",
                filename="mep_force_xi_residual_vs_dft_no_hip_leftnet_cf_orig_hessian_mae.png",
                legend_fontsize=FOUR_MODEL_LEGEND_FONTSIZE,
                legend_labelcolor=PLOTLY_FONT_COLOR,
                legend_loc="upper left",
                legend_bbox_to_anchor=(0.0, 1.01),
                tight_pad=COMPACT_TIGHT_PAD,
                wspace=COMPACT_WSPACE_3PANEL,
                bbox_inches="tight",
                pad_inches=COMPACT_PAD_INCHES,
            )
    plot_curvature(output_dir, args.dpi, xi, kappa_ad_auto, kappa_hip, kappa_ad_fd, kappa_dft, kappa_leftnet_cf)

    metrics = pd.DataFrame(
        {
            "xi": xi,
            "arc_length_ang": s,
            "dft_g_ev_ang": g[DFT_LABEL],
            "ad_g_ev_ang": g[AD_LABEL],
            "hip_g_ev_ang": g[HIP_LABEL],
            "ad_minus_dft_g_ev_ang": g[AD_LABEL] - g[DFT_LABEL],
            "hip_minus_dft_g_ev_ang": g[HIP_LABEL] - g[DFT_LABEL],
            "dft_kappa_ev_ang2": kappa_dft,
            "ad_kappa_auto_ev_ang2": kappa_ad_auto,
            "hip_kappa_auto_ev_ang2": kappa_hip,
            "ad_kappa_fd_ev_ang2": kappa_ad_fd,
            "ad_kappa_auto_minus_dft": kappa_ad_auto - kappa_dft,
            "ad_kappa_auto_minus_fd": kappa_ad_auto - kappa_ad_fd,
        }
    )
    if LEFTNET_CF_LABEL in methods and kappa_leftnet_cf is not None:
        metrics["leftnet_cf_g_ev_ang"] = g[LEFTNET_CF_LABEL]
        metrics["leftnet_cf_minus_dft_g_ev_ang"] = g[LEFTNET_CF_LABEL] - g[DFT_LABEL]
        metrics["leftnet_cf_kappa_auto_ev_ang2"] = kappa_leftnet_cf
        metrics["leftnet_cf_kappa_auto_minus_dft"] = kappa_leftnet_cf - kappa_dft
        if kappa_leftnet_cf_fd is not None:
            metrics["leftnet_cf_kappa_fd_ev_ang2"] = kappa_leftnet_cf_fd
            metrics["leftnet_cf_kappa_auto_minus_fd"] = kappa_leftnet_cf - kappa_leftnet_cf_fd
    for label in (AD_ORIG_LABEL, LEFTNET_CF_ORIG_LABEL):
        if label in g_xi:
            key = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
            metrics[f"{key}_g_xi_ev_ang"] = g_xi[label]
            metrics[f"{key}_minus_dft_g_xi_ev_ang"] = g_xi[label] - g_xi[DFT_LABEL]
    for label, values in hessian_mae_for_labels(
        tuple(label for label in (AD_LABEL, AD_ORIG_LABEL, LEFTNET_CF_LABEL, LEFTNET_CF_ORIG_LABEL) if label in g_xi),
        methods,
        optional_models,
    ).items():
        key = label.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        metrics[f"{key}_hessian_mae_ev_ang2"] = values
    metrics_path = output_dir / "mep_mechanism_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    ad_resid = g[AD_LABEL] - g[DFT_LABEL]
    hip_resid = g[HIP_LABEL] - g[DFT_LABEL]
    print(
        f"EqV2-DFT force residual: std={ad_resid.std():.4e}  max|.|={np.abs(ad_resid).max():.4e} eV/A",
        flush=True,
    )
    print(
        f"HIP-DFT force residual: std={hip_resid.std():.4e}  max|.|={np.abs(hip_resid).max():.4e} eV/A",
        flush=True,
    )
    if LEFTNET_CF_LABEL in g:
        leftnet_cf_resid = g[LEFTNET_CF_LABEL] - g[DFT_LABEL]
        print(
            f"LeftNet-CF-DFT force residual: std={leftnet_cf_resid.std():.4e}  "
            f"max|.|={np.abs(leftnet_cf_resid).max():.4e} eV/A",
            flush=True,
        )
    print(
        f"EqV2 curvature gap: median|autograd-DFT|={np.median(np.abs(kappa_ad_auto - kappa_dft)):.4f}  "
        f"median|autograd-FD|={np.median(np.abs(kappa_ad_auto - kappa_ad_fd)):.4f} eV/A^2",
        flush=True,
    )
    print(f"Wrote plots to {output_dir}", flush=True)
    print(f"Wrote metrics to {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
