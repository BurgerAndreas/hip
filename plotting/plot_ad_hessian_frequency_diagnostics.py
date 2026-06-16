#!/usr/bin/env python
"""Scatter roughness/frequency diagnostics against AD Hessian/eigenvector metrics.

This recomputes diagnostics from the raw line-scan NPZs and plots every line
point, not binned summaries. For each model, there are two line points per
center Hessian sample; the center target metric is repeated for both directions.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from hip.frequency_analysis import analyze_frequencies_np  # noqa: E402
from plot_style import (  # noqa: E402
    EQV2_FORCE_COLOR,
    EQV2_NO_H_FORCE_COLOR,
    LEFTNET_CF_FORCE_COLOR,
    LEFTNET_DF_FORCE_COLOR,
    finish_axis,
)

Z_TO_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    15: "P",
    16: "S",
    17: "Cl",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


DEFAULT_SCAN_DIR = project_root() / "runs" / "t1x_val_force_spectra_100x2x51"
DEFAULT_HORM_DIR = project_root().parent / "HORM" / "runs" / "t1x_val_center_hessians_leftnet"


@dataclass(frozen=True)
class ModelSpec:
    label: str
    spectra_csv: Path
    color: str
    hessian_summary_csv: Path | None = None


DIAGNOSTICS: list[tuple[str, str, str]] = [
    ("force_resid_norm_detrended_rms", "Residual norm detrended RMS", "log"),
    ("force_resid_norm_tv_detrended", "Residual norm detrended TV", "log"),
    ("force_resid_norm_slope_tv", "Residual norm slope TV", "log"),
    ("force_resid_norm_second_deriv_rms", "Residual norm second-deriv RMS", "log"),
    ("proj_resid_detrended_rms", "Projected residual detrended RMS", "log"),
    ("proj_resid_slope_tv", "Projected residual slope TV", "log"),
    ("proj_power_l1_detrended", "Projected spectral-shape L1", "linear"),
]


def spearman(a: pd.Series, b: pd.Series) -> float:
    ra = pd.Series(np.asarray(a, dtype=float)).rank()
    rb = pd.Series(np.asarray(b, dtype=float)).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


def normalized_scan_direction(coords_ang: np.ndarray) -> np.ndarray:
    direction = coords_ang[-1] - coords_ang[0]
    norm = float(np.linalg.norm(direction))
    if norm <= 0.0:
        raise ValueError("Line-scan endpoints have zero displacement.")
    return direction / norm


def detrend_linear(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(x, y, deg=1)
    return y - np.polyval(coeffs, x)


def scalar_roughness_metrics(y: np.ndarray, lam: np.ndarray) -> dict[str, float]:
    """Return roughness metrics for a scalar force/residual line signal."""
    z = detrend_linear(y, lam)
    dlam = float(np.mean(np.diff(lam)))
    slopes = np.diff(y) / dlam
    second = np.diff(y, n=2) / (dlam * dlam)
    return {
        "detrended_rms": float(np.sqrt(np.mean(z * z))),
        "tv_detrended": float(np.sum(np.abs(np.diff(z)))),
        "slope_tv": float(np.sum(np.abs(np.diff(slopes)))),
        "second_deriv_rms": float(np.sqrt(np.mean(second * second))),
    }


def line_diagnostics(row: pd.Series) -> dict[str, float]:
    model_npz = np.load(row.eqv2_line_npz_path)
    dft_npz = np.load(row.dft_line_npz_path)

    lam = model_npz["lambda_ang"].astype(float)
    direction = normalized_scan_direction(model_npz["coords_ang"])
    force_model = model_npz["forces_ev_ang"].astype(float)
    force_dft = dft_npz["forces_ev_ang"].astype(float)
    force_resid = force_model - force_dft

    projected_model = np.einsum("pij,ij->p", force_model, direction)
    projected_dft = np.einsum("pij,ij->p", force_dft, direction)
    projected_resid = projected_model - projected_dft
    resid_norm = np.sqrt(np.mean(force_resid.reshape(force_resid.shape[0], -1) ** 2, axis=1))

    out: dict[str, float] = {}
    for key, value in scalar_roughness_metrics(resid_norm, lam).items():
        out[f"force_resid_norm_{key}"] = value
    for key, value in scalar_roughness_metrics(projected_resid, lam).items():
        out[f"proj_resid_{key}"] = value

    model_signal = detrend_linear(projected_model, lam)
    dft_signal = detrend_linear(projected_dft, lam)
    model_power = np.abs(np.fft.rfft(model_signal)) ** 2
    dft_power = np.abs(np.fft.rfft(dft_signal)) ** 2
    model_power[0] = 0.0
    dft_power[0] = 0.0
    model_power = model_power / (float(model_power.sum()) + 1e-30)
    dft_power = dft_power / (float(dft_power.sum()) + 1e-30)
    out["proj_power_l1_detrended"] = float(np.sum(np.abs(model_power - dft_power)))
    return out


def hessian_mae(model_hessian: np.ndarray, dft_hessian: np.ndarray) -> float:
    sym = 0.5 * (model_hessian + model_hessian.T)
    return float(np.mean(np.abs(sym - dft_hessian)))


def atomic_symbols(npz_file: np.lib.npyio.NpzFile) -> list[str]:
    if "symbols" in npz_file.files:
        return [str(symbol) for symbol in npz_file["symbols"]]
    return [Z_TO_SYMBOL[int(z)] for z in npz_file["atomic_numbers"]]


def eigvec1_cos_eckart(
    model_hessian: np.ndarray,
    dft_hessian: np.ndarray,
    coords_ang: np.ndarray,
    symbols: list[str],
) -> float:
    model_freqs = analyze_frequencies_np(
        hessian=0.5 * (model_hessian + model_hessian.T),
        cart_coords=coords_ang,
        atomsymbols=symbols,
    )
    dft_freqs = analyze_frequencies_np(
        hessian=0.5 * (dft_hessian + dft_hessian.T),
        cart_coords=coords_ang,
        atomsymbols=symbols,
    )
    model_vec = model_freqs["eigvecs"][:, 0]
    dft_vec = dft_freqs["eigvecs"][:, 0]
    return float(abs(np.dot(model_vec, dft_vec)))


def eigval_mae_eckart(
    model_hessian: np.ndarray,
    dft_hessian: np.ndarray,
    coords_ang: np.ndarray,
    symbols: list[str],
) -> float:
    model_freqs = analyze_frequencies_np(
        hessian=0.5 * (model_hessian + model_hessian.T),
        cart_coords=coords_ang,
        atomsymbols=symbols,
    )
    dft_freqs = analyze_frequencies_np(
        hessian=0.5 * (dft_hessian + dft_hessian.T),
        cart_coords=coords_ang,
        atomsymbols=symbols,
    )
    return float(np.mean(np.abs(model_freqs["eigvals"] - dft_freqs["eigvals"])))


def load_eqv2_orig_targets(scan_dir: Path) -> pd.DataFrame:
    csv_path = scan_dir / "ad_hessians" / "eqv2_orig" / "eqv2_orig_selected_ad_hessian_metrics.csv"
    return pd.read_csv(csv_path)


def load_dft_hessians(eqv2_orig_targets: pd.DataFrame) -> dict[tuple[int, int], np.ndarray]:
    dft_hessians: dict[tuple[int, int], np.ndarray] = {}
    for row in eqv2_orig_targets.itertuples(index=False):
        key = (int(row.geom_rank), int(row.dataset_idx))
        dft_hessians[key] = np.load(row.sample_npz_path)["hessian_dft_ev_ang2"].astype(float)
    return dft_hessians


def load_eqv2_npz_targets(eqv2_targets: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for row in eqv2_targets.itertuples(index=False):
        npz = np.load(row.sample_npz_path)
        model_hessian = npz["hessian_ad_ev_ang2"].astype(float)
        dft_hessian = npz["hessian_dft_ev_ang2"].astype(float)
        coords_ang = npz["coords_ang"].astype(float)
        symbols = atomic_symbols(npz)
        if target_metric == "eigvec1_cos_eckart":
            target_value = eigvec1_cos_eckart(model_hessian, dft_hessian, coords_ang, symbols)
        elif target_metric == "eigval_mae_eckart":
            target_value = eigval_mae_eckart(model_hessian, dft_hessian, coords_ang, symbols)
        else:
            raise ValueError(f"Unsupported Eckart target metric: {target_metric}")
        rows.append(
            {
                "geom_rank": int(row.geom_rank),
                "dataset_idx": int(row.dataset_idx),
                target_metric: target_value,
            }
        )
    return pd.DataFrame(rows)


def load_eqv2_raw_targets(scan_dir: Path, tag: str, target_metric: str) -> pd.DataFrame | None:
    csv_path = scan_dir / "ad_hessians" / tag / f"{tag}_selected_ad_hessian_metrics.csv"
    if not csv_path.exists():
        return None
    targets = pd.read_csv(csv_path)
    if target_metric == "hessian_error":
        return targets[["geom_rank", "dataset_idx", "hessian_error"]].copy()
    return load_eqv2_npz_targets(targets, target_metric)


def load_leftnet_targets(
    spec: ModelSpec,
    dft_hessians: dict[tuple[int, int], np.ndarray],
    target_metric: str,
) -> pd.DataFrame:
    if spec.hessian_summary_csv is None:
        raise ValueError(f"{spec.label} does not have a LeftNet Hessian summary.")
    summary = pd.read_csv(spec.hessian_summary_csv)
    rows: list[dict[str, float | int]] = []
    for row in summary[summary.status.eq("ok")].itertuples(index=False):
        key = (int(row.geom_rank), int(row.dataset_idx))
        dft_hessian = dft_hessians.get(key)
        if dft_hessian is None:
            continue
        npz = np.load(row.hessian_npz_path)
        model_hessian = npz["hessian_ev_ang2"].astype(float)
        target_value = hessian_mae(model_hessian, dft_hessian)
        if target_metric == "eigvec1_cos_eckart":
            target_value = eigvec1_cos_eckart(
                model_hessian,
                dft_hessian,
                npz["coords_ang"].astype(float),
                atomic_symbols(npz),
            )
        elif target_metric == "eigval_mae_eckart":
            target_value = eigval_mae_eckart(
                model_hessian,
                dft_hessian,
                npz["coords_ang"].astype(float),
                atomic_symbols(npz),
            )
        rows.append({"geom_rank": key[0], "dataset_idx": key[1], target_metric: target_value})
    return pd.DataFrame(rows)


def compute_line_table(spec: ModelSpec, targets: pd.DataFrame, target_metric: str) -> pd.DataFrame:
    spectra = pd.read_csv(spec.spectra_csv)
    diag_rows: list[dict[str, float | int]] = []
    for row in spectra.itertuples(index=False):
        row_series = pd.Series(row._asdict())
        diag = line_diagnostics(row_series)
        diag_rows.append(
            {
                "geom_rank": int(row.geom_rank),
                "dataset_idx": int(row.dataset_idx),
                "direction_id": int(row.direction_id),
                **diag,
            }
        )
    diagnostics = pd.DataFrame(diag_rows)
    merged = spectra.merge(diagnostics, on=["geom_rank", "dataset_idx", "direction_id"])
    merged = merged.merge(
        targets[["geom_rank", "dataset_idx", target_metric]],
        on=["geom_rank", "dataset_idx"],
        suffixes=("", "_target"),
    )
    target_column = f"{target_metric}_target"
    if target_column in merged.columns:
        merged[target_metric] = merged[target_column]
        merged = merged.drop(columns=[target_column])
    merged["model"] = spec.label
    return merged


def default_model_specs(scan_dir: Path, horm_dir: Path) -> list[ModelSpec]:
    return [
        ModelSpec(
            "eqv2.ckpt",
            scan_dir / "force_spectra_analysis" / "force_spectra_summary.csv",
            EQV2_FORCE_COLOR,
        ),
        ModelSpec(
            "eqv2_orig",
            scan_dir / "force_spectra_analysis_eqv2_orig" / "force_spectra_summary.csv",
            EQV2_NO_H_FORCE_COLOR,
        ),
        ModelSpec(
            "leftnet-cf",
            scan_dir / "t1x_val_force_spectra_leftnet" / "force_spectra_analysis" / "leftnet-cf" / "force_spectra_summary.csv",
            LEFTNET_CF_FORCE_COLOR,
            horm_dir / "leftnet-cf" / "hessian_summary.csv",
        ),
        ModelSpec(
            "leftnet-cf-orig",
            scan_dir / "t1x_val_force_spectra_leftnet" / "force_spectra_analysis" / "leftnet-cf-orig" / "force_spectra_summary.csv",
            LEFTNET_CF_FORCE_COLOR,
            horm_dir / "leftnet-cf-orig" / "hessian_summary.csv",
        ),
        ModelSpec(
            "leftnet-df",
            scan_dir / "t1x_val_force_spectra_leftnet" / "force_spectra_analysis" / "leftnet-df" / "force_spectra_summary.csv",
            LEFTNET_DF_FORCE_COLOR,
            horm_dir / "leftnet-df" / "hessian_summary.csv",
        ),
        ModelSpec(
            "leftnet-df-orig",
            scan_dir / "t1x_val_force_spectra_leftnet" / "force_spectra_analysis" / "leftnet-df-orig" / "force_spectra_summary.csv",
            LEFTNET_DF_FORCE_COLOR,
            horm_dir / "leftnet-df-orig" / "hessian_summary.csv",
        ),
    ]


def load_all_data(scan_dir: Path, horm_dir: Path, target_metric: str) -> pd.DataFrame:
    eqv2_orig_targets = load_eqv2_orig_targets(scan_dir)
    dft_hessians = load_dft_hessians(eqv2_orig_targets)
    tables: list[pd.DataFrame] = []
    for spec in default_model_specs(scan_dir, horm_dir):
        if spec.label == "eqv2.ckpt" and target_metric == "hessian_error":
            targets = load_eqv2_raw_targets(scan_dir, "eqv2", target_metric)
            if targets is None:
                targets = pd.read_csv(spec.spectra_csv).groupby(["geom_rank", "dataset_idx"], as_index=False)["hessian_error"].mean()
        elif spec.label == "eqv2_orig":
            if target_metric == "hessian_error":
                targets = eqv2_orig_targets[["geom_rank", "dataset_idx", "hessian_error"]].copy()
            else:
                targets = load_eqv2_npz_targets(eqv2_orig_targets, target_metric)
        elif spec.label == "eqv2.ckpt":
            targets = load_eqv2_raw_targets(scan_dir, "eqv2", target_metric)
            if targets is None:
                print(f"Skipping {spec.label:16s}: raw Hessian NPZs are not available for {target_metric}.")
                continue
        else:
            targets = load_leftnet_targets(spec, dft_hessians, target_metric)
        table = compute_line_table(spec, targets, target_metric)
        tables.append(table)
        print(f"Loaded {spec.label:16s}: {len(table):4d} line points")
    return pd.concat(tables, ignore_index=True)


def target_label(target_metric: str) -> str:
    if target_metric == "eigvec1_cos_eckart":
        return r"$|\cos(v_{1,\mathrm{model}}^{Eckart}, v_{1,\mathrm{DFT}}^{Eckart})|$"
    if target_metric == "eigval_mae_eckart":
        return r"Eckart eigenvalue MAE [eV/$\AA^2$]"
    return "Hessian MAE [eV/$\\AA^2$]"


def target_short_name(target_metric: str) -> str:
    if target_metric == "eigvec1_cos_eckart":
        return "Eckart first-eigenvector cosine"
    if target_metric == "eigval_mae_eckart":
        return "Eckart eigenvalue MAE"
    return "Hessian MAE"


def seaborn_model_palette(models: pd.Series) -> dict[str, tuple[float, float, float]]:
    labels = list(dict.fromkeys(models.tolist()))
    colors = sns.color_palette("deep", n_colors=len(labels))
    return dict(zip(labels, colors, strict=True))


def save_combined_scatter(df: pd.DataFrame, out_dir: Path, target_metric: str) -> Path:
    palette = seaborn_model_palette(df["model"])
    fig, axes = plt.subplots(3, 3, figsize=(17, 14), sharey=False)
    axes_flat = axes.ravel()

    for ax, (column, label, scale) in zip(axes_flat, DIAGNOSTICS):
        for model, group in df.groupby("model", sort=False):
            ax.scatter(
                group[column],
                group[target_metric],
                s=22,
                alpha=0.32,
                color=palette.get(model),
                edgecolors="none",
                label=model,
            )
        rho = spearman(df[column], df[target_metric])
        ax.text(
            0.04,
            0.96,
            rf"all models $\rho_s={rho:+.2f}$",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
        )
        ax.set_xlabel(label)
        ax.set_ylabel(target_label(target_metric))
        if scale == "log":
            ax.set_xscale("log")
        if target_metric in {"hessian_error", "eigval_mae_eckart"}:
            ax.set_yscale("log")
        else:
            ax.set_ylim(-0.03, 1.03)
        finish_axis(ax)

    for ax in axes_flat[len(DIAGNOSTICS) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=True, edgecolor="none")
    fig.tight_layout(rect=(0, 0.045, 1, 1), pad=0.35)
    path = out_dir / f"ad_hessian_frequency_diagnostic_scatters_vs_{target_metric}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_per_model_scatter(df: pd.DataFrame, out_dir: Path, target_metric: str) -> list[Path]:
    paths: list[Path] = []
    palette = seaborn_model_palette(df["model"])
    for model, group in df.groupby("model", sort=False):
        fig, axes = plt.subplots(3, 3, figsize=(17, 14), sharey=True)
        axes_flat = axes.ravel()
        color = palette[model]
        for ax, (column, label, scale) in zip(axes_flat, DIAGNOSTICS):
            ax.scatter(group[column], group[target_metric], s=24, alpha=0.42, color=color, edgecolors="none")
            rho = spearman(group[column], group[target_metric])
            ax.text(
                0.04,
                0.96,
                rf"$\rho_s={rho:+.2f}$",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.82),
            )
            ax.set_xlabel(label)
            ax.set_ylabel(target_label(target_metric))
            if scale == "log":
                ax.set_xscale("log")
            if target_metric in {"hessian_error", "eigval_mae_eckart"}:
                ax.set_yscale("log")
            else:
                ax.set_ylim(-0.03, 1.03)
            finish_axis(ax)
        for ax in axes_flat[len(DIAGNOSTICS) :]:
            ax.axis("off")
        fig.tight_layout(pad=0.35)
        path = out_dir / f"{model.replace('.', '_').replace('-', '_')}_frequency_diagnostic_scatters_vs_{target_metric}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def save_correlations(df: pd.DataFrame, out_dir: Path, target_metric: str) -> Path:
    rows: list[dict[str, float | str | int]] = []
    for model, group in df.groupby("model", sort=False):
        for column, label, _scale in DIAGNOSTICS:
            rows.append(
                {
                    "model": model,
                    "diagnostic": column,
                    "label": label,
                    "n": int(len(group)),
                    f"spearman_{target_metric}": spearman(group[column], group[target_metric]),
                }
            )
    for column, label, _scale in DIAGNOSTICS:
        rows.append(
            {
                "model": "all",
                "diagnostic": column,
                "label": label,
                "n": int(len(df)),
                f"spearman_{target_metric}": spearman(df[column], df[target_metric]),
            }
        )
    out_csv = out_dir / f"ad_hessian_frequency_diagnostic_correlations_vs_{target_metric}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--horm-dir", type=Path, default=DEFAULT_HORM_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=project_root() / "plots" / DEFAULT_SCAN_DIR.name / "ad_hessians" / "frequency_diagnostic_plots",
    )
    parser.add_argument(
        "--target-metric",
        choices=("hessian_error", "eigvec1_cos_eckart", "eigval_mae_eckart"),
        default="hessian_error",
        help="Metric to plot on the y-axis.",
    )
    parser.add_argument("--write-per-model", action="store_true", help="Also write one multi-panel scatter figure per model.")
    parser.add_argument(
        "--reuse-line-points",
        action="store_true",
        help="Load the cached line-points CSV instead of recomputing diagnostics from the raw NPZs.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_csv = args.out_dir / f"ad_hessian_frequency_diagnostic_line_points_vs_{args.target_metric}.csv"
    if args.reuse_line_points and data_csv.exists():
        print(f"Reusing cached line points: {data_csv}")
        df = pd.read_csv(data_csv)
    else:
        df = load_all_data(args.scan_dir, args.horm_dir, args.target_metric)
        df.to_csv(data_csv, index=False)
    corr_csv = save_correlations(df, args.out_dir, args.target_metric)
    combined_png = save_combined_scatter(df, args.out_dir, args.target_metric)
    written = [data_csv, corr_csv, combined_png]
    if args.write_per_model:
        written.extend(save_per_model_scatter(df, args.out_dir, args.target_metric))

    print("Wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
