#!/usr/bin/env python
"""Plot DFT-only force and Hessian diagnostics on the glycine 2D CV grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.cache_glycine_pt_orca_vibrations import mode_alignment, normalized_curvature, vibrational_eigh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/glycine_pt_scan_n36"))
    parser.add_argument("--vib-cache", type=Path, default=None)
    parser.add_argument("--orca-energies", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def as_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    pivot = df.pivot(index="q_oh", columns="q_nh", values=value_col).sort_index(axis=0).sort_index(axis=1)
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    return x, y, z


def contour_levels(values: np.ndarray, step: float = 10.0) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return np.array([0.0, step])
    lo = np.floor(finite.min() / step) * step
    hi = np.ceil(finite.max() / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def add_energy_contours(ax: plt.Axes, df: pd.DataFrame) -> None:
    x, y, z = as_grid(df, "orca_relative_kcalmol")
    levels = contour_levels(z.compressed(), step=10.0)
    ax.contour(x, y, z, levels=levels, colors="k", linewidths=0.5, alpha=0.55)


def heatmap(
    df: pd.DataFrame,
    value_col: str,
    output_path: Path,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
    dpi: int = 300,
    discrete: bool = False,
) -> None:
    x, y, z = as_grid(df, value_col)
    fig, ax = plt.subplots(figsize=(6.4, 5.3), constrained_layout=True)
    if discrete:
        finite = z.compressed()
        vmin = int(np.nanmin(finite))
        vmax = int(np.nanmax(finite))
        n_levels = vmax - vmin + 1
        discrete_cmap = plt.get_cmap(cmap, n_levels)
        boundaries = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
        norm = matplotlib.colors.BoundaryNorm(boundaries, discrete_cmap.N)
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=discrete_cmap, norm=norm)
        cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(vmin, vmax + 1), spacing="proportional")
    else:
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap)
        cbar = fig.colorbar(mesh, ax=ax)
    add_energy_contours(ax, df)
    ax.set_title(title)
    ax.set_xlabel(r"$q_\mathrm{NH}=d(\mathrm{N4,H9})$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}=d(\mathrm{O3,H9})$ [$\AA$]")
    ax.set_aspect("equal", adjustable="box")
    cbar.set_label(cbar_label)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def finite_difference_force(df: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        df.pivot(index="q_oh", columns="q_nh", values="orca_relative_kcalmol")
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    q_nh = pivot.columns.to_numpy(dtype=float)
    q_oh = pivot.index.to_numpy(dtype=float)
    energy = pivot.to_numpy(dtype=float)
    d_e_d_nh = np.full_like(energy, np.nan, dtype=float)
    d_e_d_oh = np.full_like(energy, np.nan, dtype=float)

    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            if not np.isfinite(energy[i, j]):
                continue
            left = j - 1 if j > 0 and np.isfinite(energy[i, j - 1]) else None
            right = j + 1 if j + 1 < energy.shape[1] and np.isfinite(energy[i, j + 1]) else None
            if left is not None and right is not None:
                d_e_d_nh[i, j] = (energy[i, right] - energy[i, left]) / (q_nh[right] - q_nh[left])
            elif right is not None:
                d_e_d_nh[i, j] = (energy[i, right] - energy[i, j]) / (q_nh[right] - q_nh[j])
            elif left is not None:
                d_e_d_nh[i, j] = (energy[i, j] - energy[i, left]) / (q_nh[j] - q_nh[left])

            down = i - 1 if i > 0 and np.isfinite(energy[i - 1, j]) else None
            up = i + 1 if i + 1 < energy.shape[0] and np.isfinite(energy[i + 1, j]) else None
            if down is not None and up is not None:
                d_e_d_oh[i, j] = (energy[up, j] - energy[down, j]) / (q_oh[up] - q_oh[down])
            elif up is not None:
                d_e_d_oh[i, j] = (energy[up, j] - energy[i, j]) / (q_oh[up] - q_oh[i])
            elif down is not None:
                d_e_d_oh[i, j] = (energy[i, j] - energy[down, j]) / (q_oh[i] - q_oh[down])

    rows = []
    for i, y_val in enumerate(q_oh):
        for j, x_val in enumerate(q_nh):
            if np.isfinite(energy[i, j]):
                rows.append(
                    {
                        "q_nh": x_val,
                        "q_oh": y_val,
                        "force_q_nh": -d_e_d_nh[i, j],
                        "force_q_oh": -d_e_d_oh[i, j],
                    }
                )
    return pd.DataFrame(rows)


def plot_force_field(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    force_df = finite_difference_force(df)
    x, y, z = as_grid(df, "orca_relative_kcalmol")
    fig, ax = plt.subplots(figsize=(6.7, 5.5), constrained_layout=True)
    levels = contour_levels(z.compressed(), step=10.0)
    mesh = ax.contourf(x, y, z, levels=levels, cmap="turbo", extend="max")
    ax.contour(x, y, z, levels=levels, colors="k", linewidths=0.45, alpha=0.55)

    # Subsample for legibility on the 36x36 grid and omit the steep lower-left outliers.
    arrows = force_df.iloc[::2].dropna(subset=["force_q_nh", "force_q_oh"]).copy()
    arrow_magnitude = np.hypot(arrows["force_q_nh"], arrows["force_q_oh"])
    arrows = arrows[arrow_magnitude <= 500.0]
    ax.quiver(
        arrows["q_nh"],
        arrows["q_oh"],
        arrows["force_q_nh"],
        arrows["force_q_oh"],
        color="white",
        edgecolor="black",
        linewidth=0.25,
        width=0.004,
        scale=3600,
    )
    ax.set_title(r"DFT CV force field $(-\nabla E)$ over ORCA energy contours")
    ax.set_xlabel(r"$q_\mathrm{NH}=d(\mathrm{N4,H9})$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}=d(\mathrm{O3,H9})$ [$\AA$]")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"ORCA relative energy [kcal mol$^{-1}$]")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_curvatures(df: pd.DataFrame, output_path: Path, dpi: int) -> None:
    specs = [
        ("curvature_q_nh_ev_ang2", r"$q_\mathrm{NH}$ curvature"),
        ("curvature_q_oh_ev_ang2", r"$q_\mathrm{OH}$ curvature"),
        ("curvature_pt_ev_ang2", r"$q_\mathrm{NH}-q_\mathrm{OH}$ curvature"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.9), constrained_layout=True)
    for ax, (col, title) in zip(axes, specs, strict=True):
        x, y, z = as_grid(df, col)
        finite = z.compressed()
        lim = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))))
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap="coolwarm", vmin=-lim, vmax=lim)
        add_energy_contours(ax, df)
        ax.set_title(title)
        ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
        ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]")
        ax.set_aspect("equal", adjustable="box")
        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label(r"projected curvature [eV $\AA^{-2}$]")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_alignment(df: pd.DataFrame, output_path: Path, dpi: int, zero_threshold: float = 0.05) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.3), constrained_layout=True)
    x, y, z = as_grid(df, "unstable_mode_pt_abs_alignment")
    finite = z.compressed()
    vmax = float(np.ceil(finite.max() * 20.0) / 20.0) if finite.size else 1.0
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("0.8")
    cmap.set_bad("white")
    mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, vmin=zero_threshold, vmax=vmax)
    add_energy_contours(ax, df)
    ax.set_title(r"$|\cos\theta(v_1, pt = q_{NH} - q_{OH})|$")
    ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]")
    ax.set_aspect("equal", adjustable="box")
    cbar = fig.colorbar(mesh, ax=ax, extend="min")
    cbar.set_label(rf"mode alignment ($|\cos\theta|<{zero_threshold:g}$ in gray)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def reaction_center_hessian_norm(hessians_ev_ang2: np.ndarray, atoms: tuple[int, ...] = (3, 4, 9)) -> np.ndarray:
    idx = np.array([3 * atom + comp for atom in atoms for comp in range(3)], dtype=int)
    hessians = 0.5 * (hessians_ev_ang2 + np.swapaxes(hessians_ev_ang2, -1, -2))
    blocks = hessians[:, idx[:, None], idx]
    return np.linalg.norm(blocks.reshape(blocks.shape[0], -1), axis=1)


def add_model_alignment(
    df: pd.DataFrame,
    arrays_path: Path,
    label: str,
    column: str,
    coords_angstrom: np.ndarray,
    masses_amu: np.ndarray,
    q_nh_direction: np.ndarray,
    q_oh_direction: np.ndarray,
    pt_direction: np.ndarray,
) -> pd.DataFrame:
    if not arrays_path.exists():
        print(f"Skipping {label} alignment; missing {arrays_path}", flush=True)
        return df
    arrays = np.load(arrays_path)
    if "hessians_cartesian" not in arrays:
        print(f"Skipping {label} alignment; hessians_cartesian missing from {arrays_path}", flush=True)
        return df
    hessians = arrays["hessians_cartesian"]
    if hessians.shape[0] != len(df):
        raise ValueError(f"{label} has {hessians.shape[0]} Hessians, expected {len(df)}")

    eval0 = []
    eval1 = []
    n_negative = []
    alignments = []
    for hessian, coords, pt in zip(hessians, coords_angstrom, pt_direction, strict=True):
        evals, modes = vibrational_eigh(hessian, coords, masses_amu)
        eval0.append(float(evals[0]))
        eval1.append(float(evals[1]))
        n_negative.append(int((evals < -1e-6).sum()))
        alignments.append(abs(mode_alignment(modes[:, 0], pt, masses_amu)))

    prefix = column.removesuffix("_unstable_mode_pt_abs_alignment")
    eval0_arr = np.asarray(eval0, dtype=float)
    eval1_arr = np.asarray(eval1, dtype=float)
    df[f"{prefix}_lowest_vib_eval_ev_ang2_amu"] = eval0_arr
    df[f"{prefix}_lowest_two_vib_eval_product_ev2_ang4_amu2"] = eval0_arr * eval1_arr
    df[f"{prefix}_n_negative"] = np.asarray(n_negative, dtype=int)
    df[column] = np.asarray(alignments, dtype=float)
    df[f"{prefix}_reaction_center_hessian_frobenius_ev_ang2"] = reaction_center_hessian_norm(hessians)
    df[f"{prefix}_curvature_q_nh_ev_ang2"] = np.asarray(
        [normalized_curvature(hessian, direction) for hessian, direction in zip(hessians, q_nh_direction, strict=True)],
        dtype=float,
    )
    df[f"{prefix}_curvature_q_oh_ev_ang2"] = np.asarray(
        [normalized_curvature(hessian, direction) for hessian, direction in zip(hessians, q_oh_direction, strict=True)],
        dtype=float,
    )
    df[f"{prefix}_curvature_pt_ev_ang2"] = np.asarray(
        [normalized_curvature(hessian, direction) for hessian, direction in zip(hessians, pt_direction, strict=True)],
        dtype=float,
    )
    return df


def plot_method_alignment(df: pd.DataFrame, output_path: Path, dpi: int, zero_threshold: float = 0.05) -> None:
    specs = [
        ("unstable_mode_pt_abs_alignment", "DFT"),
        ("hip_unstable_mode_pt_abs_alignment", "HIP"),
        ("ad_unstable_mode_pt_abs_alignment", "AD"),
    ]
    specs = [(col, label) for col, label in specs if col in df.columns]
    if len(specs) < 2:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(3.35 * len(specs), 3.55), constrained_layout=False)
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("0.8")
    cmap.set_bad("white")
    mesh = None
    for idx, (ax, (col, label)) in enumerate(zip(axes, specs, strict=True)):
        x, y, z = as_grid(df, col)
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, vmin=zero_threshold, vmax=1.0)
        add_energy_contours(ax, df)
        ax.set_title(label)
        ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
        ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]" if idx == 0 else "")
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Unstable-mode alignment with glycine proton-transfer CV")
    fig.subplots_adjust(left=0.065, right=0.885, bottom=0.15, top=0.84, wspace=0.08)
    assert mesh is not None
    cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), extend="min", pad=0.01)
    cbar.set_label(rf"$|\cos\theta(v_1, q_\mathrm{{NH}} - q_\mathrm{{OH}})|$ ($<{zero_threshold:g}$ in gray)")
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_method_heatmaps(
    df: pd.DataFrame,
    specs: list[tuple[str, str]],
    output_path: Path,
    title: str,
    cbar_label: str,
    dpi: int,
    cmap: str = "viridis",
    symmetric: bool = False,
    discrete: bool = False,
) -> None:
    specs = [(col, label) for col, label in specs if col in df.columns]
    if len(specs) < 2:
        return

    grids = [as_grid(df, col) for col, _ in specs]
    finite_parts = [grid[2].compressed() for grid in grids if grid[2].compressed().size]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)

    fig, axes = plt.subplots(1, len(specs), figsize=(3.15 * len(specs), 3.45), constrained_layout=False)
    axes = np.atleast_1d(axes)
    mesh = None
    if discrete:
        vmin = int(np.nanmin(finite))
        vmax = int(np.nanmax(finite))
        n_levels = vmax - vmin + 1
        plot_cmap = plt.get_cmap(cmap, n_levels)
        boundaries = np.arange(vmin - 0.5, vmax + 1.5, 1.0)
        norm = matplotlib.colors.BoundaryNorm(boundaries, plot_cmap.N)
    else:
        if symmetric:
            lim = max(abs(float(np.nanmin(finite))), abs(float(np.nanmax(finite))))
            vmin = -lim
            vmax = lim
        else:
            vmin = float(np.nanmin(finite))
            vmax = float(np.nanmax(finite))
        plot_cmap = cmap
        norm = None

    for idx, (ax, (grid, (_, label))) in enumerate(zip(axes, zip(grids, specs, strict=True), strict=True)):
        x, y, z = grid
        if discrete:
            mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=plot_cmap, norm=norm)
        else:
            mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=plot_cmap, vmin=vmin, vmax=vmax)
        add_energy_contours(ax, df)
        ax.set_title(label)
        ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
        ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]" if idx == 0 else "")
        ax.set_aspect("equal", adjustable="box")
        if idx > 0:
            ax.yaxis.set_major_formatter(matplotlib.ticker.NullFormatter())

    assert mesh is not None
    fig.suptitle(title)
    fig.subplots_adjust(left=0.07, right=0.89, bottom=0.15, top=0.84, wspace=0.015)
    if discrete:
        cbar = fig.colorbar(
            mesh,
            ax=axes.ravel().tolist(),
            ticks=np.arange(vmin, vmax + 1),
            spacing="proportional",
            pad=0.01,
        )
    else:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.01)
    cbar.set_label(cbar_label)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    scan_dir = args.scan_dir
    vib_cache = args.vib_cache or scan_dir / "orca_vib_cache.npz"
    orca_energies = args.orca_energies or scan_dir / "orca_energies.csv"
    output_dir = args.output_dir or scan_dir / "plots_dft_only"
    output_dir.mkdir(parents=True, exist_ok=True)

    energies = pd.read_csv(orca_energies)
    if "orca_energy_relative_kcalmol" in energies:
        energy_col = "orca_energy_relative_kcalmol"
    else:
        energy_col = "energy_relative_kcalmol"
    energy_df = energies[["grid_id", "q_nh", "q_oh", energy_col]].rename(
        columns={energy_col: "orca_relative_kcalmol"}
    )

    with np.load(vib_cache) as cache:
        coords_angstrom = cache["coords_angstrom"]
        masses_amu = cache["masses_amu"]
        q_nh_direction = cache["q_nh_direction"]
        q_oh_direction = cache["q_oh_direction"]
        pt_direction = cache["pt_direction"]
        vib_df = pd.DataFrame(
            {
                "grid_id": cache["grid_id"].astype(int),
                "lowest_vib_eval_ev_ang2_amu": cache["vib_evals_ev_ang2_amu"][:, 0],
                "lowest_two_vib_eval_product_ev2_ang4_amu2": (
                    cache["vib_evals_ev_ang2_amu"][:, 0]
                    * cache["vib_evals_ev_ang2_amu"][:, 1]
                ),
                "n_negative": cache["n_negative"].astype(int),
                "unstable_mode_pt_alignment": cache["unstable_mode_pt_alignment"],
                "unstable_mode_pt_abs_alignment": cache["unstable_mode_pt_abs_alignment"],
                "reaction_center_hessian_frobenius_ev_ang2": cache[
                    "reaction_center_hessian_frobenius_ev_ang2"
                ],
                "curvature_q_nh_ev_ang2": cache["curvature_q_nh_ev_ang2"],
                "curvature_q_oh_ev_ang2": cache["curvature_q_oh_ev_ang2"],
                "curvature_pt_ev_ang2": cache["curvature_pt_ev_ang2"],
            }
        )
    df = energy_df.merge(vib_df, on="grid_id", validate="one_to_one").sort_values("grid_id")
    suffix = scan_dir.name.removeprefix("glycine_pt_scan") if scan_dir.name.startswith("glycine_pt_scan") else ""
    hip_arrays = args.hip_arrays or scan_dir / "hip_v2_arrays.npz"
    eqv2_arrays = args.eqv2_arrays or scan_dir.parent / f"glycine_pt_eqv2_autograd{suffix}" / "eqv2_autograd_arrays.npz"
    df = add_model_alignment(
        df,
        hip_arrays,
        "HIP",
        "hip_unstable_mode_pt_abs_alignment",
        coords_angstrom,
        masses_amu,
        q_nh_direction,
        q_oh_direction,
        pt_direction,
    )
    df = add_model_alignment(
        df,
        eqv2_arrays,
        "AD",
        "ad_unstable_mode_pt_abs_alignment",
        coords_angstrom,
        masses_amu,
        q_nh_direction,
        q_oh_direction,
        pt_direction,
    )

    plot_force_field(df, output_dir / "glycine_pt_dft_cv_force_field.png", args.dpi)
    heatmap(
        df,
        "lowest_vib_eval_ev_ang2_amu",
        output_dir / "glycine_pt_dft_lowest_hessian_eigenvalue.png",
        "DFT lowest vibrational Hessian eigenvalue",
        r"lowest eigenvalue [eV $\AA^{-2}$ amu$^{-1}$]",
        cmap="coolwarm",
        dpi=args.dpi,
    )
    plot_method_heatmaps(
        df,
        [
            ("lowest_vib_eval_ev_ang2_amu", "DFT"),
            ("hip_lowest_vib_eval_ev_ang2_amu", "HIP"),
            ("ad_lowest_vib_eval_ev_ang2_amu", "AD"),
        ],
        output_dir / "glycine_pt_lowest_hessian_eigenvalue_methods.png",
        "Lowest vibrational Hessian eigenvalue",
        r"lowest eigenvalue [eV $\AA^{-2}$ amu$^{-1}$]",
        args.dpi,
        cmap="coolwarm",
        symmetric=True,
    )
    heatmap(
        df,
        "lowest_two_vib_eval_product_ev2_ang4_amu2",
        output_dir / "glycine_pt_dft_lowest_two_hessian_eigenvalue_product.png",
        "DFT product of lowest two vibrational Hessian eigenvalues",
        r"eigenvalue product [eV$^2$ $\AA^{-4}$ amu$^{-2}$]",
        cmap="coolwarm",
        dpi=args.dpi,
    )
    plot_method_heatmaps(
        df,
        [
            ("lowest_two_vib_eval_product_ev2_ang4_amu2", "DFT"),
            ("hip_lowest_two_vib_eval_product_ev2_ang4_amu2", "HIP"),
            ("ad_lowest_two_vib_eval_product_ev2_ang4_amu2", "AD"),
        ],
        output_dir / "glycine_pt_lowest_two_hessian_eigenvalue_product_methods.png",
        "Product of lowest two vibrational Hessian eigenvalues",
        r"eigenvalue product [eV$^2$ $\AA^{-4}$ amu$^{-2}$]",
        args.dpi,
        cmap="coolwarm",
        symmetric=True,
    )
    heatmap(
        df,
        "n_negative",
        output_dir / "glycine_pt_dft_n_negative_modes.png",
        "DFT number of negative vibrational modes",
        "negative mode count",
        cmap="viridis",
        dpi=args.dpi,
        discrete=True,
    )
    plot_method_heatmaps(
        df,
        [
            ("n_negative", "DFT"),
            ("hip_n_negative", "HIP"),
            ("ad_n_negative", "AD"),
        ],
        output_dir / "glycine_pt_n_negative_modes_methods.png",
        "Number of negative vibrational modes",
        "negative mode count",
        args.dpi,
        cmap="viridis",
        discrete=True,
    )
    plot_alignment(df, output_dir / "glycine_pt_dft_unstable_mode_alignment.png", args.dpi)
    plot_method_alignment(df, output_dir / "glycine_pt_unstable_mode_alignment_methods.png", args.dpi)
    heatmap(
        df,
        "reaction_center_hessian_frobenius_ev_ang2",
        output_dir / "glycine_pt_dft_reaction_center_hessian_norm.png",
        "DFT O-N-H Hessian block norm",
        r"Frobenius norm [eV $\AA^{-2}$]",
        cmap="magma",
        dpi=args.dpi,
    )
    plot_method_heatmaps(
        df,
        [
            ("reaction_center_hessian_frobenius_ev_ang2", "DFT"),
            ("hip_reaction_center_hessian_frobenius_ev_ang2", "HIP"),
            ("ad_reaction_center_hessian_frobenius_ev_ang2", "AD"),
        ],
        output_dir / "glycine_pt_reaction_center_hessian_norm_methods.png",
        "O-N-H Hessian block norm",
        r"Frobenius norm [eV $\AA^{-2}$]",
        args.dpi,
        cmap="magma",
    )
    plot_curvatures(df, output_dir / "glycine_pt_dft_cv_curvatures.png", args.dpi)
    for col_suffix, label in [
        ("curvature_q_nh_ev_ang2", r"$q_\mathrm{NH}$ curvature"),
        ("curvature_q_oh_ev_ang2", r"$q_\mathrm{OH}$ curvature"),
        ("curvature_pt_ev_ang2", r"$q_\mathrm{NH}-q_\mathrm{OH}$ curvature"),
    ]:
        plot_method_heatmaps(
            df,
            [
                (col_suffix, "DFT"),
                (f"hip_{col_suffix}", "HIP"),
                (f"ad_{col_suffix}", "AD"),
            ],
            output_dir / f"glycine_pt_{col_suffix.removesuffix('_ev_ang2')}_methods.png",
            label,
            r"projected curvature [eV $\AA^{-2}$]",
            args.dpi,
            cmap="coolwarm",
            symmetric=True,
        )
    df.to_csv(output_dir / "glycine_pt_dft_cv_diagnostics.csv", index=False)
    print(f"Wrote DFT CV diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
