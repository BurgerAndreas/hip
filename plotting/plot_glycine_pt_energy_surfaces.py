#!/usr/bin/env python
"""Plot ORCA and model 2D energy surfaces for the glycine proton-transfer scan."""
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

from plot_style import DFT_COLOR, LINE_WIDTH, MARKER_SIZE, THIN_LINE_WIDTH, finish_axis, model_color


EV_TO_KCALMOL = 23.060548867
DFT_LABEL = "DFT"
HIP_LABEL = "HIP"
AD_LABEL = "AD"


@dataclass
class EnergyModel:
    label: str
    path: Path
    energy_col: str
    key: str


@dataclass
class DerivativeModel:
    label: str
    path: Path
    key: str


def safe_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", label.strip().lower())
    return safe.strip("_") or "model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=Path("runs/glycine_pt_scan"),
        help="Directory containing HIP glycine proton-transfer scan outputs.",
    )
    parser.add_argument(
        "--orca-dir",
        type=Path,
        default=Path("orca_wb97x_631gd_glycine_pt_nh_oh_scan_80"),
        help="Directory containing ORCA package outputs and metadata.csv.",
    )
    parser.add_argument(
        "--hip-energies",
        type=Path,
        default=None,
        help="Backwards-compatible HIP energies table. Defaults to scan-dir/hip_v2_energies.csv.",
    )
    parser.add_argument(
        "--orca-energies",
        type=Path,
        default=None,
        help="ORCA energies table. Defaults to orca-dir/metadata.csv.",
    )
    parser.add_argument(
        "--model-energy",
        action="append",
        nargs=3,
        metavar=("LABEL", "PATH", "ENERGY_COL"),
        default=[],
        help="Model energy table to plot. May be repeated.",
    )
    parser.add_argument(
        "--orca-vib-cache",
        type=Path,
        default=None,
        help="ORCA vibrational cache with DFT Hessians, forces, and PT directions.",
    )
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--energy-contour-step",
        type=float,
        default=10.0,
        help="Contour spacing for relative energies in kcal/mol.",
    )
    parser.add_argument(
        "--energy-vmax",
        type=float,
        default=200.0,
        help="Upper color limit for relative energy plots in kcal/mol.",
    )
    parser.add_argument(
        "--linecut-q-oh",
        type=float,
        nargs="*",
        default=(1.15, 1.75, 2.15),
        help="q_oh values to use for q_nh line cuts.",
    )
    parser.add_argument(
        "--linecut-q-nh",
        type=float,
        nargs="*",
        default=(1.0, 1.65, 2.3),
        help="q_nh values to use for q_oh line cuts.",
    )
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def require_columns(df: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def load_orca(path: Path) -> pd.DataFrame:
    df = read_table(path)
    if {"q_nh_angstrom", "q_oh_angstrom"}.issubset(df.columns):
        q_nh_col = "q_nh_angstrom"
        q_oh_col = "q_oh_angstrom"
    else:
        q_nh_col = "q_nh"
        q_oh_col = "q_oh"

    grid_col = "grid_id" if "grid_id" in df.columns else "job_id"
    require_columns(df, {grid_col, q_nh_col, q_oh_col, "energy_relative_kcalmol"}, str(path))
    out = df[[grid_col, q_nh_col, q_oh_col, "energy_relative_kcalmol"]].copy()
    out = out.rename(
        columns={
            grid_col: "grid_id",
            q_nh_col: "q_nh",
            q_oh_col: "q_oh",
            "energy_relative_kcalmol": "orca_relative_kcalmol",
        }
    )
    out["grid_id"] = out["grid_id"].astype(str).str.replace("grid_", "", regex=False).astype(int)
    return out.sort_values("grid_id").reset_index(drop=True)


def load_model_energy(model: EnergyModel) -> pd.DataFrame:
    df = read_table(model.path)
    require_columns(df, {"grid_id", "q_nh", "q_oh", model.energy_col}, str(model.path))
    out = df[["grid_id", "q_nh", "q_oh", model.energy_col]].copy()
    out["grid_id"] = out["grid_id"].astype(int)
    energy_ev = out[model.energy_col].astype(float)
    out[f"{model.key}_energy_ev"] = energy_ev

    relative_col = f"{model.energy_col}_relative"
    if relative_col in df.columns:
        out[f"{model.key}_relative_kcalmol"] = df[relative_col].astype(float) * EV_TO_KCALMOL
    else:
        out[f"{model.key}_relative_kcalmol"] = (energy_ev - energy_ev.min()) * EV_TO_KCALMOL

    return out.drop(columns=[model.energy_col]).sort_values("grid_id").reset_index(drop=True)


def default_models(args: argparse.Namespace) -> list[EnergyModel]:
    models: list[EnergyModel] = []
    hip_path = args.hip_energies or args.scan_dir / "hip_v2_energies.csv"
    models.append(EnergyModel(HIP_LABEL, hip_path, "hip_v2_energy", "hip_direct"))

    suffix = ""
    prefix = "glycine_pt_scan"
    if args.scan_dir.name.startswith(prefix):
        suffix = args.scan_dir.name.removeprefix(prefix)
    eqv2_path = args.scan_dir.parent / f"glycine_pt_eqv2_autograd{suffix}" / "eqv2_autograd_energies.csv"
    if eqv2_path.exists():
        models.append(EnergyModel(AD_LABEL, eqv2_path, "eqv2_autograd_energy", "eqv2_autograd"))

    for label, path, energy_col in args.model_energy:
        models.append(EnergyModel(label, Path(path), energy_col, safe_label(label)))
    return models


def scan_suffix(scan_dir: Path) -> str:
    prefix = "glycine_pt_scan"
    if scan_dir.name.startswith(prefix):
        return scan_dir.name.removeprefix(prefix)
    return ""


def default_derivative_models(args: argparse.Namespace) -> list[DerivativeModel]:
    suffix = scan_suffix(args.scan_dir)
    models = [
        DerivativeModel(HIP_LABEL, args.hip_arrays or args.scan_dir / "hip_v2_arrays.npz", "hip_direct"),
    ]
    eqv2_path = args.eqv2_arrays or args.scan_dir.parent / f"glycine_pt_eqv2_autograd{suffix}" / "eqv2_autograd_arrays.npz"
    if eqv2_path.exists():
        models.append(DerivativeModel(AD_LABEL, eqv2_path, "eqv2_autograd"))
    return models


def merge_surfaces(orca: pd.DataFrame, models: list[EnergyModel]) -> pd.DataFrame:
    merged = orca.copy()
    for model in models:
        model_df = load_model_energy(model)
        tmp = merged.merge(
            model_df,
            on="grid_id",
            suffixes=("", f"_{model.key}"),
            validate="one_to_one",
        )
        if not np.allclose(tmp["q_nh"], tmp[f"q_nh_{model.key}"]):
            raise ValueError(f"{model.label} and ORCA q_nh values do not match by grid_id")
        if not np.allclose(tmp["q_oh"], tmp[f"q_oh_{model.key}"]):
            raise ValueError(f"{model.label} and ORCA q_oh values do not match by grid_id")
        tmp = tmp.drop(columns=[f"q_nh_{model.key}", f"q_oh_{model.key}"])
        tmp[f"{model.key}_minus_orca_kcalmol"] = (
            tmp[f"{model.key}_relative_kcalmol"] - tmp["orca_relative_kcalmol"]
        )
        merged = tmp
    return merged.sort_values("grid_id")


def as_grid(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    pivot = (
        df.pivot(index="q_oh", columns="q_nh", values=value_col)
        .sort_index(axis=0)
        .sort_index(axis=1)
    )
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = np.ma.masked_invalid(pivot.to_numpy(dtype=float))
    return x, y, z


def contour_levels(values: np.ndarray, step: float, vmax: float | None = None) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return np.array([0.0, step])
    lo = np.floor(finite.min() / step) * step
    hi_raw = finite.max() if vmax is None else float(vmax)
    hi = np.ceil(hi_raw / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def add_min_marker(ax: plt.Axes, df: pd.DataFrame, value_col: str, label: str) -> None:
    row = df.loc[df[value_col].idxmin()]
    ax.plot(row["q_nh"], row["q_oh"], marker="*", color="white", markeredgecolor="black", ms=12)
    ax.text(
        row["q_nh"] + 0.025,
        row["q_oh"] + 0.025,
        f"{label} min\n{int(row['grid_id'])}",
        color="white",
        fontsize=8,
        weight="bold",
    )


def plot_surface(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    title: str,
    cbar_label: str,
    cmap: str,
    contour_step: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    contour_source: str | None = None,
    contour_color: str = "k",
    cbar_extend: str = "neither",
) -> None:
    x, y, z = as_grid(df, value_col)
    if contour_step is None:
        mesh = ax.pcolormesh(x, y, z, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        levels = contour_levels(z.compressed(), contour_step, vmax=vmax)
        mesh = ax.contourf(x, y, z, levels=levels, cmap=cmap, extend="max")

    if contour_source is not None:
        _, _, contour_z = as_grid(df, contour_source)
        levels = contour_levels(contour_z.compressed(), contour_step or 10.0)
        ax.contour(x, y, contour_z, levels=levels, colors=contour_color, linewidths=0.65, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel(r"$q_\mathrm{NH}=d(\mathrm{N4,H9})$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}=d(\mathrm{O3,H9})$ [$\AA$]")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(mesh, ax=ax, extend=cbar_extend)
    cbar.set_label(cbar_label)


def flat_projection(values: np.ndarray, directions: np.ndarray) -> np.ndarray:
    flat_values = np.asarray(values, dtype=float).reshape(values.shape[0], -1)
    flat_dirs = np.asarray(directions, dtype=float).reshape(directions.shape[0], -1)
    norms = np.linalg.norm(flat_dirs, axis=1)
    return np.einsum("ij,ij->i", flat_values, flat_dirs) / np.maximum(norms, 1e-12)


def hessian_curvature(hessians: np.ndarray, directions: np.ndarray) -> np.ndarray:
    hess = np.asarray(hessians, dtype=float)
    flat_dirs = np.asarray(directions, dtype=float).reshape(directions.shape[0], -1)
    denom = np.einsum("ij,ij->i", flat_dirs, flat_dirs)
    numer = np.einsum("bi,bij,bj->b", flat_dirs, hess, flat_dirs)
    return numer / np.maximum(denom, 1e-12)


def robust_abs_limit(values: list[np.ndarray], floor: float = 1.0, percentile: float = 99.0) -> float:
    finite = np.concatenate([np.asarray(value, dtype=float).ravel() for value in values])
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return floor
    return max(float(np.nanpercentile(np.abs(finite), percentile)), floor)


def save_derivative_surface_comparison(
    df: pd.DataFrame,
    models: list[DerivativeModel],
    output_dir: Path,
    dpi: int,
    *,
    ref_col: str,
    model_col_suffix: str,
    error_col_suffix: str,
    title_prefix: str,
    cbar_label: str,
    error_cbar_label: str,
    output_name: str,
    cmap: str = "coolwarm",
    error_limit: float | None = None,
) -> None:
    if not models:
        return

    n_panels = 1 + 2 * len(models)
    nrows = 2
    ncols = int(np.ceil(n_panels / nrows))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), constrained_layout=False)
    axes = np.atleast_1d(axes_grid).ravel()

    top_values = [df[ref_col].to_numpy(dtype=float)]
    top_values.extend(df[f"{model.key}_{model_col_suffix}"].to_numpy(dtype=float) for model in models)
    value_lim = robust_abs_limit(top_values)
    error_values = [df[f"{model.key}_{error_col_suffix}"].to_numpy(dtype=float) for model in models]
    error_lim = robust_abs_limit(error_values) if error_limit is None else float(error_limit)

    plot_surface(
        axes[0],
        df,
        ref_col,
        DFT_LABEL,
        cbar_label,
        cmap,
        contour_step=None,
        vmin=-value_lim,
        vmax=value_lim,
        contour_source="orca_relative_kcalmol",
    )

    for idx, model in enumerate(models, start=1):
        plot_surface(
            axes[idx],
            df,
            f"{model.key}_{model_col_suffix}",
            model.label,
            cbar_label,
            cmap,
            contour_step=None,
            vmin=-value_lim,
            vmax=value_lim,
            contour_source="orca_relative_kcalmol",
        )

    for offset, model in enumerate(models, start=1 + len(models)):
        plot_surface(
            axes[offset],
            df,
            f"{model.key}_{error_col_suffix}",
            f"{model.label} - {DFT_LABEL}",
            error_cbar_label,
            "coolwarm",
            contour_step=None,
            vmin=-error_lim,
            vmax=error_lim,
            contour_source="orca_relative_kcalmol",
            cbar_extend="both",
        )

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle(title_prefix)
    fig.subplots_adjust(left=0.055, right=0.97, bottom=0.075, top=0.92, wspace=0.25, hspace=0.28)
    fig.savefig(output_dir / output_name, dpi=dpi)
    plt.close(fig)


def save_surface_comparison(
    df: pd.DataFrame,
    models: list[EnergyModel],
    output_dir: Path,
    dpi: int,
    step: float,
    vmax: float | None,
) -> None:
    n_panels = 1 + 2 * len(models)
    nrows = 2
    ncols = int(np.ceil(n_panels / nrows))
    fig, axes_grid = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), constrained_layout=False)
    axes = np.atleast_1d(axes_grid).ravel()
    plot_surface(
        axes[0],
        df,
        "orca_relative_kcalmol",
        DFT_LABEL,
        r"relative energy [kcal mol$^{-1}$]",
        "turbo",
        contour_step=step,
        vmax=vmax,
    )

    for idx, model in enumerate(models, start=1):
        plot_surface(
            axes[idx],
            df,
            f"{model.key}_relative_kcalmol",
            model.label,
            r"relative energy [kcal mol$^{-1}$]",
            "turbo",
            contour_step=step,
            vmax=vmax,
            contour_source="orca_relative_kcalmol",
        )

    for offset, model in enumerate(models, start=1 + len(models)):
        err_col = f"{model.key}_minus_orca_kcalmol"
        plot_surface(
            axes[offset],
            df,
            err_col,
            f"{model.label} - {DFT_LABEL}",
            r"relative energy error [kcal mol$^{-1}$]",
            "coolwarm",
            contour_step=None,
            vmin=-110.0,
            vmax=110.0,
            contour_source="orca_relative_kcalmol",
        )

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    fig.suptitle("Glycine intramolecular proton-transfer scan, Transition1x test sample 5")
    fig.subplots_adjust(left=0.055, right=0.97, bottom=0.075, top=0.92, wspace=0.25, hspace=0.28)
    fig.savefig(output_dir / "glycine_pt_energy_surfaces.png", dpi=dpi)
    plt.close(fig)


def save_overlay(df: pd.DataFrame, models: list[EnergyModel], output_dir: Path, dpi: int, step: float, vmax: float | None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.3), constrained_layout=True)
    plot_surface(
        ax,
        df,
        "orca_relative_kcalmol",
        f"{DFT_LABEL} surface with model contours",
        rf"{DFT_LABEL} relative energy [kcal mol$^{{-1}}$]",
        "turbo",
        contour_step=step,
        vmax=vmax,
    )
    colors = ["white", "0.2", "magenta", "yellow"]
    for model, color in zip(models, colors, strict=False):
        x, y, z_model = as_grid(df, f"{model.key}_relative_kcalmol")
        levels = contour_levels(z_model.compressed(), step, vmax=vmax)
        ax.contour(x, y, z_model, levels=levels, colors=color, linewidths=0.85, linestyles="--")
    fig.savefig(output_dir / "glycine_pt_orca_with_model_contours.png", dpi=dpi)
    plt.close(fig)


def save_parity(df: pd.DataFrame, models: list[EnergyModel], output_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.3), constrained_layout=True)
    x = df["orca_relative_kcalmol"].to_numpy(dtype=float)
    max_y = float(np.nanmax(x))
    for model in models:
        y = df[f"{model.key}_relative_kcalmol"].to_numpy(dtype=float)
        max_y = max(max_y, float(np.nanmax(y)))
        sns.scatterplot(x=x, y=y, ax=ax, s=42, edgecolor="k", linewidth=0.3, color=model_color(model.label), label=model.label)
    lim = [0.0, max_y * 1.03]
    sns.lineplot(x=lim, y=lim, ax=ax, color=DFT_COLOR, linestyle="--", linewidth=THIN_LINE_WIDTH)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(rf"{DFT_LABEL} relative energy [kcal mol$^{{-1}}$]")
    ax.set_ylabel(r"model relative energy [kcal mol$^{-1}$]")
    ax.set_title(f"Model vs {DFT_LABEL} energy parity")
    ax.legend(fontsize=8)
    finish_axis(ax)
    fig.savefig(output_dir / "glycine_pt_model_orca_parity.png", dpi=dpi)
    plt.close(fig)


def add_derivative_metrics(
    df: pd.DataFrame,
    vib_cache_path: Path,
    models: list[DerivativeModel],
) -> pd.DataFrame:
    if not vib_cache_path.exists():
        print(f"Skipping force/Hessian surfaces; missing ORCA vib cache: {vib_cache_path}", flush=True)
        return df

    cache = np.load(vib_cache_path)
    required = {
        "grid_id",
        "q_nh",
        "q_oh",
        "forces_ev_ang",
        "hessian_ev_ang2",
        "pt_direction",
        "curvature_pt_ev_ang2",
    }
    missing = sorted(required - set(cache.files))
    if missing:
        print(f"Skipping force/Hessian surfaces; {vib_cache_path} is missing {missing}", flush=True)
        return df

    cache_df = pd.DataFrame(
        {
            "grid_id": cache["grid_id"].astype(int),
            "q_nh_cache": cache["q_nh"].astype(float),
            "q_oh_cache": cache["q_oh"].astype(float),
            "dft_force_pt": flat_projection(cache["forces_ev_ang"], cache["pt_direction"]),
            "dft_curvature_pt": cache["curvature_pt_ev_ang2"].astype(float),
        }
    )
    merged = df.merge(cache_df, on="grid_id", validate="one_to_one")
    if not np.allclose(merged["q_nh"], merged["q_nh_cache"]):
        raise ValueError(f"{vib_cache_path} q_nh values do not match ORCA energies by grid_id")
    if not np.allclose(merged["q_oh"], merged["q_oh_cache"]):
        raise ValueError(f"{vib_cache_path} q_oh values do not match ORCA energies by grid_id")
    merged = merged.drop(columns=["q_nh_cache", "q_oh_cache"])

    for model in models:
        if not model.path.exists():
            print(f"Skipping derivative metrics for {model.label}; missing arrays: {model.path}", flush=True)
            continue
        arrays = np.load(model.path)
        missing_arrays = sorted({"forces", "hessians_cartesian"} - set(arrays.files))
        if missing_arrays:
            print(f"Skipping derivative metrics for {model.label}; {model.path} is missing {missing_arrays}", flush=True)
            continue
        if arrays["forces"].shape[0] != len(merged) or arrays["hessians_cartesian"].shape[0] != len(merged):
            raise ValueError(f"{model.label} derivative arrays do not have {len(merged)} grid points")

        force_pt = flat_projection(arrays["forces"], cache["pt_direction"])
        curvature_pt = hessian_curvature(arrays["hessians_cartesian"], cache["pt_direction"])
        merged[f"{model.key}_force_pt"] = force_pt
        merged[f"{model.key}_force_pt_error"] = force_pt - merged["dft_force_pt"].to_numpy(dtype=float)
        merged[f"{model.key}_curvature_pt"] = curvature_pt
        merged[f"{model.key}_curvature_pt_error"] = curvature_pt - merged["dft_curvature_pt"].to_numpy(dtype=float)

    return merged


def nearest_values(available: np.ndarray, requested: tuple[float, ...]) -> list[float]:
    out = []
    for value in requested:
        nearest = float(available[np.argmin(np.abs(available - value))])
        if nearest not in out:
            out.append(nearest)
    return out


def min_profile_along_pt_cv(df: pd.DataFrame, value_col: str, n_bins: int = 80) -> pd.DataFrame:
    work = df[["q_nh", "q_oh", value_col]].copy()
    work["pt_cv"] = work["q_nh"] - work["q_oh"]
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["pt_cv", value_col])
    if work.empty:
        return work

    bins = np.linspace(work["pt_cv"].min(), work["pt_cv"].max(), n_bins + 1)
    work["pt_cv_bin"] = np.clip(np.digitize(work["pt_cv"], bins) - 1, 0, n_bins - 1)
    idx = work.groupby("pt_cv_bin", sort=True)[value_col].idxmin()
    return work.loc[idx].sort_values("pt_cv")


def save_linecuts(
    df: pd.DataFrame,
    models: list[EnergyModel],
    output_dir: Path,
    dpi: int,
    requested_q_oh: tuple[float, ...],
    requested_q_nh: tuple[float, ...],
) -> None:
    del requested_q_oh, requested_q_nh

    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    profile = min_profile_along_pt_cv(df, "orca_relative_kcalmol")
    sns.lineplot(x=profile["pt_cv"], y=profile["orca_relative_kcalmol"], ax=ax, marker="o", markersize=MARKER_SIZE, linewidth=LINE_WIDTH, label=DFT_LABEL, color=DFT_COLOR)

    for model in models:
        value_col = f"{model.key}_relative_kcalmol"
        profile = min_profile_along_pt_cv(df, value_col)
        sns.lineplot(x=profile["pt_cv"], y=profile[value_col], ax=ax, marker="o", markersize=MARKER_SIZE, linestyle="--", linewidth=LINE_WIDTH, label=model.label, color=model_color(model.label))

    ax.set_xlabel(r"proton-transfer CV $q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]")
    ax.set_ylabel(r"minimum relative energy [kcal mol$^{-1}$]")
    ax.set_title("Minimum-energy profile along glycine proton-transfer CV")
    finish_axis(ax)
    ax.legend(fontsize=8)

    fig.savefig(output_dir / "glycine_pt_energy_linecuts.png", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    orca_path = args.orca_energies or args.orca_dir / "metadata.csv"
    output_dir = args.output_dir or args.scan_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = default_models(args)
    df = merge_surfaces(load_orca(orca_path), models)
    derivative_models = default_derivative_models(args)
    vib_cache_path = args.orca_vib_cache or args.scan_dir / "orca_vib_cache.npz"
    df = add_derivative_metrics(df, vib_cache_path, derivative_models)
    df.to_csv(output_dir / "glycine_pt_energy_surface_data.csv", index=False)

    save_surface_comparison(
        df=df,
        models=models,
        output_dir=output_dir,
        dpi=args.dpi,
        step=args.energy_contour_step,
        vmax=args.energy_vmax,
    )
    save_overlay(df, models, output_dir, args.dpi, args.energy_contour_step, args.energy_vmax)
    save_parity(df, models, output_dir, args.dpi)
    plotted_derivative_models = [
        model
        for model in derivative_models
        if f"{model.key}_force_pt" in df.columns and f"{model.key}_curvature_pt" in df.columns
    ]
    if plotted_derivative_models and "dft_force_pt" in df.columns:
        save_derivative_surface_comparison(
            df,
            plotted_derivative_models,
            output_dir,
            args.dpi,
            ref_col="dft_force_pt",
            model_col_suffix="force_pt",
            error_col_suffix="force_pt_error",
            title_prefix="Projected force along glycine proton-transfer CV",
            cbar_label=r"projected force [eV $\AA^{-1}$]",
            error_cbar_label=r"force error [eV $\AA^{-1}$]",
            output_name="glycine_pt_force_pt_surfaces.png",
            error_limit=0.5,
        )
    if plotted_derivative_models and "dft_curvature_pt" in df.columns:
        save_derivative_surface_comparison(
            df,
            plotted_derivative_models,
            output_dir,
            args.dpi,
            ref_col="dft_curvature_pt",
            model_col_suffix="curvature_pt",
            error_col_suffix="curvature_pt_error",
            title_prefix="Hessian curvature along glycine proton-transfer CV",
            cbar_label=r"PT curvature [eV $\AA^{-2}$]",
            error_cbar_label=r"curvature error [eV $\AA^{-2}$]",
            output_name="glycine_pt_hessian_pt_curvature_surfaces.png",
            error_limit=5.0,
        )
    save_linecuts(
        df,
        models,
        output_dir,
        args.dpi,
        requested_q_oh=tuple(args.linecut_q_oh),
        requested_q_nh=tuple(args.linecut_q_nh),
    )

    print(f"Wrote plots and merged data to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
