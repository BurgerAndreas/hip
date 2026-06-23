#!/usr/bin/env python
"""Method-comparison diagnostics on the DFT-relaxed glycine PT surface.

Styling matches ``plot_glycine_pt_dft_cv_diagnostics.py`` (Seaborn theme, viridis
family colormaps, shared axis/colorbar formatting).  Layout uses ``(s, sigma)``
coordinates instead of ``(q_nh, q_oh)``.  Includes a DFT force-field panel with
finite-difference gradients on ``(s, sigma)``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


for extra in (str(ROOT), str(ROOT / "plotting")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from plot_style import PLOTLY_FONT_COLOR, finish_axis  # noqa: E402
from scripts.cache_glycine_pt_orca_vibrations import (  # noqa: E402
    NEG_EIGVAL_THRESHOLD,
    mode_alignment,
    vibrational_eigh,
)

HARTREE_TO_KCAL = 627.5094740631
ALIGNMENT_ZERO_THRESHOLD = 0.4
LAM_MIN_VMIN = -14.3
LAM_MIN_VMAX = 0.0
SIGMA_YTICKS = (2.5, 3.0, 3.5, 4.0)


def axes_label_size() -> float:
    return float(matplotlib.rcParams["axes.labelsize"])


def tick_label_size() -> float:
    return float(matplotlib.rcParams["xtick.labelsize"])


def title_size() -> float:
    return float(matplotlib.rcParams["axes.titlesize"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=ROOT / "runs" / "glycine_pt_scan_relaxed_dft_eval")
    parser.add_argument("--vib-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--stationary-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_stationary_json(args: argparse.Namespace, scan: Path) -> Path:
    if args.stationary_json is not None:
        return args.stationary_json
    local = scan / "stationary_points.json"
    if local.exists():
        return local
    return ROOT / "runs" / "glycine_pt_scan_relaxed" / "stationary_points.json"


STATIONARY_STYLE = {
    "reactant": dict(marker="o", label="Reactant", facecolor="white", edgecolor="black"),
    "product": dict(marker="^", label="Product", facecolor="white", edgecolor="black"),
    "ts": dict(marker="*", label="TS", facecolor="white", edgecolor="black"),
}


def make_lam_min_cmap() -> matplotlib.colors.LinearSegmentedColormap:
    """Viridis truncated so λ=0 is green; yellow (viridis top) reserved for λ>0."""
    viridis = matplotlib.colormaps["viridis"]
    green_fraction = 1.0 - 5.0 / 22.0
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "viridis_to_green", viridis(np.linspace(0.0, green_fraction, 256))
    )
    cmap.set_over(viridis(1.0))
    return cmap


def make_alignment_cmap() -> matplotlib.colors.Colormap:
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_under("0.8")
    cmap.set_bad("white")
    return cmap


def contour_levels(values: np.ndarray, step: float = 10.0) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return np.array([0.0, step])
    lo = np.floor(finite.min() / step) * step
    hi = np.ceil(finite.max() / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def style_colorbar(
    cbar,
    label: str,
    *,
    extend: str = "neither",
    label_fontsize: float | None = None,
    tick_fontsize: float | None = None,
) -> None:
    cbar.set_label(label, fontsize=label_fontsize or axes_label_size())
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(
        which="both",
        axis="both",
        length=0,
        labelsize=tick_fontsize or tick_label_size(),
    )
    cbar.outline.set_visible(False)
    if extend == "min":
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def finish_panel(ax, *, idx: int = 0, hide_nonfirst_y_ticks: bool = False) -> None:
    finish_axis(ax)
    ax.set_yticks(SIGMA_YTICKS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(
        axis="both",
        which="major",
        length=3.0,
        width=0.8,
        bottom=True,
        left=True,
        color="#2F4565",
        labelsize=tick_label_size(),
    )
    if hide_nonfirst_y_ticks and idx > 0:
        ax.tick_params(axis="y", left=False, labelleft=False)


def overlay_stationary(ax, pts) -> None:
    if not pts:
        return
    for key, style in STATIONARY_STYLE.items():
        if key not in pts:
            continue
        ax.scatter(
            [pts[key]["s"]], [pts[key]["sigma"]],
            marker=style["marker"], s=170 if style["marker"] == "*" else 90,
            c=style["facecolor"], edgecolors=style["edgecolor"],
            linewidths=1.1, zorder=6, label=style["label"],
        )


def add_row_legend(
    fig,
    axes,
    *,
    between_titles: bool = False,
    anchor_axis: int = -1,
    anchor_offset: float = -0.06,
) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    legend_kw = dict(
        fontsize=tick_label_size() * 0.65,
        frameon=True,
        edgecolor="none",
        markerscale=0.85,
    )
    if between_titles and len(axes) > 1:
        fig.draw_without_rendering()
        pos_anchor = axes[anchor_axis].get_position()
        x = pos_anchor.x0 + anchor_offset
        trans = blended_transform_factory(fig.transFigure, axes[0].transAxes)
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            bbox_to_anchor=(x, 1.03),
            bbox_transform=trans,
            ncol=3,
            borderaxespad=0,
            handletextpad=0.35,
            columnspacing=0.9,
            **legend_kw,
        )
    else:
        legend = axes[0].legend(loc="upper center", ncol=3, **legend_kw)
    for text in legend.get_texts():
        text.set_color(PLOTLY_FONT_COLOR)


def add_energy_contours(ax, xs, ys, energy_grid) -> None:
    levels = contour_levels(energy_grid.compressed(), step=10.0)
    ax.contour(xs, ys, energy_grid, levels=levels, colors="k", linewidths=0.5, alpha=0.55)


def make_grid(s: np.ndarray, sigma: np.ndarray, values: np.ndarray):
    """Pivot node values onto the regular (s, sigma) grid; gaps become masked."""
    s_r = np.round(np.asarray(s, dtype=float), 6)
    sig_r = np.round(np.asarray(sigma, dtype=float), 6)
    xs = np.unique(s_r)
    ys = np.unique(sig_r)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    for sv, gv, val in zip(s_r, sig_r, np.asarray(values, dtype=float)):
        grid[yi[gv], xi[sv]] = val
    return xs, ys, np.ma.masked_invalid(grid)


def finite_difference_force_ss(
    s: np.ndarray, sigma: np.ndarray, energy_kcal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, z = make_grid(s, sigma, energy_kcal)
    energy = z.filled(np.nan)
    d_e_d_s = np.full_like(energy, np.nan)
    d_e_d_sigma = np.full_like(energy, np.nan)
    for i in range(energy.shape[0]):
        for j in range(energy.shape[1]):
            if not np.isfinite(energy[i, j]):
                continue
            left = j - 1 if j > 0 and np.isfinite(energy[i, j - 1]) else None
            right = j + 1 if j + 1 < energy.shape[1] and np.isfinite(energy[i, j + 1]) else None
            if left is not None and right is not None:
                d_e_d_s[i, j] = (energy[i, right] - energy[i, left]) / (xs[right] - xs[left])
            elif right is not None:
                d_e_d_s[i, j] = (energy[i, right] - energy[i, j]) / (xs[right] - xs[j])
            elif left is not None:
                d_e_d_s[i, j] = (energy[i, j] - energy[i, left]) / (xs[j] - xs[left])

            down = i - 1 if i > 0 and np.isfinite(energy[i - 1, j]) else None
            up = i + 1 if i + 1 < energy.shape[0] and np.isfinite(energy[i + 1, j]) else None
            if down is not None and up is not None:
                d_e_d_sigma[i, j] = (energy[up, j] - energy[down, j]) / (ys[up] - ys[down])
            elif up is not None:
                d_e_d_sigma[i, j] = (energy[up, j] - energy[i, j]) / (ys[up] - ys[i])
            elif down is not None:
                d_e_d_sigma[i, j] = (energy[i, j] - energy[down, j]) / (ys[i] - ys[down])

    s_pts, sig_pts, f_s, f_sig = [], [], [], []
    for i, sig_val in enumerate(ys):
        for j, s_val in enumerate(xs):
            if np.isfinite(energy[i, j]) and np.isfinite(d_e_d_s[i, j]) and np.isfinite(d_e_d_sigma[i, j]):
                s_pts.append(float(s_val))
                sig_pts.append(float(sig_val))
                f_s.append(-float(d_e_d_s[i, j]))
                f_sig.append(-float(d_e_d_sigma[i, j]))
    return xs, ys, z, np.asarray(s_pts), np.asarray(sig_pts), np.asarray(f_s), np.asarray(f_sig)


def subsample_force_arrows(
    s_pts: np.ndarray,
    sig_pts: np.ndarray,
    f_s: np.ndarray,
    f_sig: np.ndarray,
    *,
    stride: int = 2,
    mag_max: float = 500.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mag = np.hypot(f_s, f_sig)
    keep = mag <= mag_max
    s_pts, sig_pts, f_s, f_sig = s_pts[keep], sig_pts[keep], f_s[keep], f_sig[keep]
    return s_pts[::stride], sig_pts[::stride], f_s[::stride], f_sig[::stride]


def draw_panel(
    ax,
    xs,
    ys,
    grid,
    *,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    title=None,
    idx: int = 0,
    energy_grid=None,
    hide_nonfirst_y_ticks: bool = False,
) -> matplotlib.collections.QuadMesh:
    mesh = ax.pcolormesh(xs, ys, grid, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="nearest")
    if energy_grid is not None:
        add_energy_contours(ax, xs, ys, energy_grid)
    ax.set_title(title, fontsize=title_size() * 0.85 if title else None)
    ax.set_xlabel(r"$s = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]")
    ax.set_ylabel(r"$\sigma = q_\mathrm{NH} + q_\mathrm{OH}$ [$\AA$]" if idx == 0 else "")
    ax.axvline(0.0, color="0.35", lw=0.6, ls="--", alpha=0.7)
    finish_panel(ax, idx=idx, hide_nonfirst_y_ticks=hide_nonfirst_y_ticks)
    return mesh


def vib_metrics(hessians_ev, coords, masses, pt_dirs):
    """lambda_min, n_negative, |softest-mode alignment with PT| per node."""
    n = hessians_ev.shape[0]
    lam = np.empty(n)
    nneg = np.empty(n, dtype=int)
    align = np.empty(n)
    for i in range(n):
        evals, modes = vibrational_eigh(hessians_ev[i], coords[i], masses)
        lam[i] = evals[0]
        nneg[i] = int((evals < -NEG_EIGVAL_THRESHOLD).sum())
        align[i] = abs(mode_alignment(modes[:, 0], pt_dirs[i], masses))
    return lam, nneg, align


def v1_cos_vs_dft(hessians_ev, v1_dft, coords, masses):
    """|cos(v1_method, v1_DFT)| per node (Eckart-projected softest modes)."""
    n = hessians_ev.shape[0]
    out = np.empty(n)
    for i in range(n):
        _, modes = vibrational_eigh(hessians_ev[i], coords[i], masses)
        out[i] = abs(np.dot(modes[:, 0], v1_dft[i]))
    return out


def main() -> None:
    args = parse_args()
    scan = args.scan_dir
    vib_path = args.vib_cache or scan / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or scan / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or scan / "eqv2_autograd_arrays.npz"
    out_dir = args.output_dir or ROOT / "runs" / "glycine_pt_scan_relaxed" / "plots_relaxed_dft_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    stat_path = resolve_stationary_json(args, scan)
    stationary = json.loads(stat_path.read_text()) if stat_path.exists() else {}

    dft = np.load(vib_path)
    hip = np.load(hip_path)
    eqv2 = np.load(eqv2_path)

    gid = dft["grid_id"]
    for other, name in ((hip, "hip"), (eqv2, "eqv2")):
        if not np.array_equal(gid, other["grid_id"]):
            raise ValueError(f"grid_id ordering mismatch between DFT and {name}")

    s = dft["s"]
    sigma = dft["sigma"]
    masses = dft["masses_amu"]
    coords = dft["coords_angstrom"]
    pt_dirs = dft["pt_direction"]

    lam_dft = dft["vib_evals_ev_ang2_amu"][:, 0]
    nneg_dft = dft["n_negative"]
    align_dft = dft["unstable_mode_pt_abs_alignment"]
    H_dft = dft["hessian_ev_ang2"]
    F_dft = dft["forces_ev_ang"]
    e_rel = (dft["energy_hartree_engrad"] - dft["energy_hartree_engrad"].min()) * HARTREE_TO_KCAL

    lam_hip, nneg_hip, align_hip = vib_metrics(hip["hessians_cartesian"], coords, masses, pt_dirs)
    lam_eqv2, nneg_eqv2, align_eqv2 = vib_metrics(eqv2["hessians_cartesian"], coords, masses, pt_dirs)

    hmae_hip = np.abs(hip["hessians_cartesian"] - H_dft).mean(axis=(1, 2))
    hmae_eqv2 = np.abs(eqv2["hessians_cartesian"] - H_dft).mean(axis=(1, 2))
    fmae_hip = np.abs(hip["forces"] - F_dft).mean(axis=(1, 2))
    fmae_eqv2 = np.abs(eqv2["forces"] - F_dft).mean(axis=(1, 2))

    v1_dft = dft["vib_modes_mw"][:, :, 0]
    cos_hip = v1_cos_vs_dft(hip["hessians_cartesian"], v1_dft, coords, masses)
    cos_eqv2 = v1_cos_vs_dft(eqv2["hessians_cartesian"], v1_dft, coords, masses)

    lam_min_cmap = make_lam_min_cmap()
    alignment_cmap = make_alignment_cmap()

    def grids(values):
        return make_grid(s, sigma, values)

    energy_grid = grids(e_rel)[2]

    # === Figure 1: relaxed DFT energy surface ===
    xs, ys, z = grids(e_rel)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = draw_panel(ax, xs, ys, z, cmap="viridis", idx=0)
    levels = contour_levels(z.compressed(), step=10.0)
    ax.contour(xs, ys, z, levels=levels, colors="k", linewidths=0.45, alpha=0.55)
    overlay_stationary(ax, stationary)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=tick_label_size() * 0.75,
        frameon=True,
        edgecolor="none",
        borderaxespad=0,
    )
    fig.draw_without_rendering()
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(mesh, cax=cax, extend="max")
    style_colorbar(
        cbar,
        r"DFT relative energy [kcal mol$^{-1}$]",
        label_fontsize=axes_label_size() * 0.85,
        tick_fontsize=tick_label_size() * 0.85,
    )
    fname = out_dir / "relaxed_energy_surface.png"
    fig.savefig(fname, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote {rel_to_repo(fname)}")
    plt.close(fig)

    # === Figure 1b: DFT force field on (s, sigma) ===
    xs_ff, ys_ff, z_ff, s_pts, sig_pts, f_s, f_sig = finite_difference_force_ss(s, sigma, e_rel)
    s_pts, sig_pts, f_s, f_sig = subsample_force_arrows(s_pts, sig_pts, f_s, f_sig)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = ax.pcolormesh(xs_ff, ys_ff, z_ff, cmap="viridis", shading="nearest")
    levels = contour_levels(z_ff.compressed(), step=10.0)
    ax.contour(xs_ff, ys_ff, z_ff, levels=levels, colors="k", linewidths=0.45, alpha=0.55)
    ax.quiver(
        s_pts, sig_pts, f_s, f_sig,
        color="white", edgecolor="black", linewidth=0.25, width=0.004, scale=2500, zorder=5,
    )
    ax.set_xlabel(r"$s = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]")
    ax.set_ylabel(r"$\sigma = q_\mathrm{NH} + q_\mathrm{OH}$ [$\AA$]")
    ax.axvline(0.0, color="0.35", lw=0.6, ls="--", alpha=0.7)
    overlay_stationary(ax, stationary)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        fontsize=tick_label_size() * 0.75,
        frameon=True,
        edgecolor="none",
        borderaxespad=0,
    )
    finish_panel(ax, idx=0)
    fig.draw_without_rendering()
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(mesh, cax=cax, extend="max")
    style_colorbar(
        cbar,
        r"DFT relative energy [kcal mol$^{-1}$]",
        label_fontsize=axes_label_size() * 0.85,
        tick_fontsize=tick_label_size() * 0.85,
        extend="max",
    )
    fname = out_dir / "relaxed_dft_force_field.png"
    fig.savefig(fname, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote {rel_to_repo(fname)}")
    plt.close(fig)

    def panel_row(
        metric_by_method,
        labels,
        fname,
        *,
        cbar_label,
        cmap,
        discrete: bool = False,
        vmin=None,
        vmax=None,
        mark_stationary: bool = False,
        legend_between_titles: bool = False,
        legend_anchor_axis: int = -1,
        legend_anchor_offset: float = -0.06,
        extend: str = "neither",
        cbar_ticks: np.ndarray | None = None,
    ):
        gridded = [grids(m) for m in metric_by_method]
        finite = np.concatenate([g[2].compressed() for g in gridded])
        if vmin is None:
            vmin = float(finite.min())
        if vmax is None:
            vmax = float(finite.max())

        norm = None
        plot_cmap = cmap
        cbar_vmin, cbar_vmax = vmin, vmax
        if discrete:
            cbar_vmin = int(np.floor(vmin))
            cbar_vmax = int(np.ceil(vmax))
            n_levels = cbar_vmax - cbar_vmin + 1
            plot_cmap = plt.get_cmap(cmap, n_levels)
            boundaries = np.arange(cbar_vmin - 0.5, cbar_vmax + 1.5, 1.0)
            norm = BoundaryNorm(boundaries, plot_cmap.N)
            vmin = vmax = None

        n = len(metric_by_method)
        fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.0), constrained_layout=True)
        axes = np.atleast_1d(axes)
        mesh = None
        for idx, (ax, (xs_, ys_, z_), label) in enumerate(zip(axes, gridded, labels, strict=True)):
            mesh = draw_panel(
                ax, xs_, ys_, z_,
                cmap=plot_cmap, norm=norm, vmin=vmin, vmax=vmax,
                title=label, idx=idx, energy_grid=energy_grid,
                hide_nonfirst_y_ticks=True,
            )
            if mark_stationary:
                overlay_stationary(ax, stationary)
        if mark_stationary:
            add_row_legend(
                fig,
                axes,
                between_titles=legend_between_titles,
                anchor_axis=legend_anchor_axis,
                anchor_offset=legend_anchor_offset,
            )
        fig.draw_without_rendering()
        cax = axes[-1].inset_axes([1.05, 0.0, 0.05, 1.0])
        if discrete:
            cbar = fig.colorbar(
                mesh,
                cax=cax,
                ticks=np.arange(cbar_vmin, cbar_vmax + 1),
                spacing="proportional",
            )
        elif cbar_ticks is not None:
            cbar = fig.colorbar(mesh, cax=cax, extend=extend, ticks=cbar_ticks)
        else:
            cbar = fig.colorbar(mesh, cax=cax, extend=extend)
        style_colorbar(cbar, cbar_label, extend=extend)
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"Wrote {rel_to_repo(out_path)}")
        plt.close(fig)

    panel_row(
        [lam_dft, lam_hip, lam_eqv2], ["DFT", "HIP", "AD"],
        "relaxed_lambda_min.png",
        cbar_label=r"$\lambda_\mathrm{min}$ [eV $\AA^{-2}$ amu$^{-1}$]",
        cmap=lam_min_cmap,
        vmin=LAM_MIN_VMIN, vmax=LAM_MIN_VMAX,
        extend="max",
        mark_stationary=True,
        legend_between_titles=True,
        legend_anchor_axis=1,
        legend_anchor_offset=-0.03,
    )

    nmax = int(max(nneg_dft.max(), nneg_hip.max(), nneg_eqv2.max()))
    panel_row(
        [nneg_dft, nneg_hip, nneg_eqv2], ["DFT", "HIP", "AD"],
        "relaxed_n_negative.png",
        cbar_label="negative mode count",
        cmap="viridis",
        discrete=True,
        vmin=0,
        vmax=max(1, nmax),
        mark_stationary=True,
        legend_between_titles=True,
        legend_anchor_axis=1,
        legend_anchor_offset=-0.03,
    )

    panel_row(
        [align_dft, align_hip, align_eqv2], ["DFT", "HIP", "AD"],
        "relaxed_mode_alignment.png",
        cbar_label=rf"$|\cos\theta(v_1, q_\mathrm{{NH}} - q_\mathrm{{OH}})|$",
        cmap=alignment_cmap,
        vmin=ALIGNMENT_ZERO_THRESHOLD,
        vmax=1.0,
        extend="min",
        cbar_ticks=np.linspace(ALIGNMENT_ZERO_THRESHOLD, 1.0, 4),
        mark_stationary=True,
    )

    panel_row(
        [hmae_hip, hmae_eqv2], ["HIP", "AD"],
        "relaxed_hessian_mae.png",
        cbar_label=r"Hessian MAE [eV $\AA^{-2}$]",
        cmap="viridis",
        vmin=0.0,
        vmax=0.4,
        extend="max",
        mark_stationary=True,
        legend_between_titles=True,
    )

    panel_row(
        [cos_hip, cos_eqv2], ["HIP", "AD"],
        "relaxed_eckart_v1_cos_vs_dft.png",
        cbar_label=rf"$|\cos(v_1^\mathrm{{Eckart}})|$ vs DFT",
        cmap="viridis_r",
        vmin=0.5,
        vmax=1.0,
        mark_stationary=True,
        legend_between_titles=True,
    )

    panel_row(
        [fmae_hip, fmae_eqv2], ["HIP", "AD"],
        "relaxed_force_mae.png",
        cbar_label=r"Force MAE [eV $\AA^{-1}$]",
        cmap="viridis",
        vmin=0.0,
        vmax=0.10,
        extend="max",
        mark_stationary=True,
        legend_between_titles=True,
    )

    def med(x):
        return float(np.median(x))

    print("Wrote plots to", rel_to_repo(out_dir))
    print(f"{'metric':<28}{'HIP':>14}{'eqv2(AD)':>14}")
    print(f"{'Hessian MAE vs DFT':<28}{med(hmae_hip):>14.4f}{med(hmae_eqv2):>14.4f}")
    print(f"{'|cos v1| vs DFT':<28}{med(cos_hip):>14.4f}{med(cos_eqv2):>14.4f}")
    print(f"{'Force MAE vs DFT':<28}{med(fmae_hip):>14.4f}{med(fmae_eqv2):>14.4f}")
    print(f"{'mean # neg eigs':<28}{nneg_hip.mean():>14.3f}{nneg_eqv2.mean():>14.3f}")
    print(f"{'(DFT mean # neg eigs)':<28}{nneg_dft.mean():>14.3f}")
    print(f"{'nodes with >1 neg eig':<28}{int((nneg_hip>1).sum()):>14d}{int((nneg_eqv2>1).sum()):>14d}")
    print(f"{'(DFT nodes >1 neg eig)':<28}{int((nneg_dft>1).sum()):>14d}")


if __name__ == "__main__":
    main()
