#!/usr/bin/env python
"""DFT-relaxed glycine PT diagnostics plotted in (q_NH, q_OH) coordinates.

This is the q-coordinate companion to
``plot_glycine_pt_dft_relaxed_diagnostics_colour.py``.  It uses the same
DFT-relaxed data and metrics, but renders the curvilinear ``(s, sigma)`` grid as
``q_NH=(sigma+s)/2`` and ``q_OH=(sigma-s)/2``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402

from plot_glycine_pt_dft_relaxed_diagnostics_colour import (  # noqa: E402
    ALIGNMENT_ZERO_THRESHOLD,
    HARTREE_TO_KCAL,
    LAM_MIN_VMAX,
    LAM_MIN_VMIN,
    PLOTLY_FONT_COLOR,
    ROOT,
    add_row_legend,
    contour_levels,
    finite_difference_force_ss,
    make_alignment_cmap,
    make_grid,
    make_lam_min_cmap,
    rel_to_repo,
    resolve_stationary_json,
    style_colorbar,
    subsample_force_arrows,
    tick_label_size,
    title_size,
    vib_metrics,
    v1_cos_vs_dft,
)
from plot_style import editorial_sequential_cmap, finish_axis  # noqa: E402

Q_LIM = (0.85, 2.78)


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


def to_q_mesh(s_values: np.ndarray, sigma_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s_mesh, sigma_mesh = np.meshgrid(s_values, sigma_values)
    return 0.5 * (sigma_mesh + s_mesh), 0.5 * (sigma_mesh - s_mesh)


def make_q_grid(s: np.ndarray, sigma: np.ndarray, values: np.ndarray):
    s_values, sigma_values, z = make_grid(s, sigma, values)
    q_nh, q_oh = to_q_mesh(s_values, sigma_values)
    return q_nh, q_oh, z


def finish_q_panel(ax, *, idx: int = 0, hide_nonfirst_y_ticks: bool = False) -> None:
    finish_axis(ax)
    ticks = np.arange(1.0, 2.76, 0.5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(*Q_LIM)
    ax.set_ylim(*Q_LIM)
    ax.tick_params(
        axis="both",
        which="major",
        length=0.0,
        width=0.0,
        bottom=True,
        left=True,
        labelsize=tick_label_size(),
    )
    ax.set_aspect("equal", adjustable="box")
    if hide_nonfirst_y_ticks and idx > 0:
        ax.tick_params(axis="y", left=False, labelleft=False)


def overlay_stationary_q(ax, pts) -> None:
    if not pts:
        return
    styles = {
        "reactant": dict(marker="o", label="Reactant", facecolor="white", edgecolor="black"),
        "product": dict(marker="^", label="Product", facecolor="white", edgecolor="black"),
        "ts": dict(marker="*", label="TS", facecolor="white", edgecolor="black"),
    }
    for key, style in styles.items():
        if key not in pts:
            continue
        ax.scatter(
            [pts[key]["q_nh"]],
            [pts[key]["q_oh"]],
            marker=style["marker"],
            s=170 if style["marker"] == "*" else 90,
            c=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=1.1,
            zorder=6,
            label=style["label"],
        )


def add_energy_contours_q(ax, q_nh, q_oh, energy_grid) -> None:
    levels = contour_levels(energy_grid.compressed(), step=10.0)
    ax.contour(q_nh, q_oh, energy_grid, levels=levels, colors="k", linewidths=0.5, alpha=0.55)


def draw_panel_q(
    ax,
    q_nh,
    q_oh,
    grid,
    *,
    cmap,
    norm=None,
    vmin=None,
    vmax=None,
    title=None,
    idx: int = 0,
    energy_grid=None,
    energy_q_nh=None,
    energy_q_oh=None,
    hide_nonfirst_y_ticks: bool = False,
    show_xlabel: bool = True,
) -> matplotlib.collections.QuadMesh:
    mesh = ax.pcolormesh(q_nh, q_oh, grid, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, shading="nearest")
    if energy_grid is not None:
        add_energy_contours_q(ax, energy_q_nh, energy_q_oh, energy_grid)
    ax.set_title(title, fontsize=title_size() * 0.85 if title else None)
    ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]" if show_xlabel else "")
    ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]" if idx == 0 else "")
    finish_q_panel(ax, idx=idx, hide_nonfirst_y_ticks=hide_nonfirst_y_ticks)
    return mesh


def main() -> None:
    args = parse_args()
    scan = args.scan_dir
    vib_path = args.vib_cache or scan / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or scan / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or scan / "eqv2_autograd_arrays.npz"
    out_dir = args.output_dir or ROOT / "runs" / "glycine_pt_scan_relaxed" / "plots_relaxed_dft_q_c"
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
        return make_q_grid(s, sigma, values)

    energy_q_nh, energy_q_oh, energy_grid = grids(e_rel)

    # === Figure 1: relaxed DFT energy surface ===
    q_nh, q_oh, z = grids(e_rel)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = draw_panel_q(ax, q_nh, q_oh, z, cmap="viridis", idx=0)
    levels = contour_levels(z.compressed(), step=10.0)
    ax.contour(q_nh, q_oh, z, levels=levels, colors="k", linewidths=0.45, alpha=0.55)
    overlay_stationary_q(ax, stationary)
    fig.draw_without_rendering()
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(mesh, cax=cax, extend="max")
    style_colorbar(
        cbar,
        r"DFT relative energy [kcal mol$^{-1}$]",
        tick_fontsize=tick_label_size() * 0.85,
    )
    fname = out_dir / "relaxed_energy_surface.png"
    fig.savefig(fname, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
    print(f"Wrote {rel_to_repo(fname)}")
    plt.close(fig)

    # === Figure 1b: DFT force field, transformed from (s, sigma) to (q_NH, q_OH) ===
    xs_ff, ys_ff, z_ff, s_pts, sig_pts, f_s, f_sig = finite_difference_force_ss(s, sigma, e_rel)
    q_nh_ff, q_oh_ff = to_q_mesh(xs_ff, ys_ff)
    q_nh_pts = 0.5 * (sig_pts + s_pts)
    q_oh_pts = 0.5 * (sig_pts - s_pts)
    f_q_nh = f_s + f_sig
    f_q_oh = -f_s + f_sig
    q_nh_pts, q_oh_pts, f_q_nh, f_q_oh = subsample_force_arrows(q_nh_pts, q_oh_pts, f_q_nh, f_q_oh)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    mesh = ax.pcolormesh(q_nh_ff, q_oh_ff, z_ff, cmap="viridis", shading="nearest")
    levels = contour_levels(z_ff.compressed(), step=10.0)
    ax.contour(q_nh_ff, q_oh_ff, z_ff, levels=levels, colors="k", linewidths=0.45, alpha=0.55)
    ax.quiver(
        q_nh_pts,
        q_oh_pts,
        f_q_nh,
        f_q_oh,
        color="white",
        edgecolor="black",
        linewidth=0.25,
        width=0.004,
        scale=2500,
        zorder=5,
    )
    ax.set_xlabel(r"$q_\mathrm{NH}$ [$\AA$]")
    ax.set_ylabel(r"$q_\mathrm{OH}$ [$\AA$]")
    overlay_stationary_q(ax, stationary)
    finish_q_panel(ax, idx=0)
    fig.draw_without_rendering()
    cax = ax.inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(mesh, cax=cax, extend="max")
    style_colorbar(
        cbar,
        r"DFT rel. energy [kcal mol$^{-1}$]",
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
        wspace: float = 0.0,
        xlabel_only_idx: int | None = None,
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
        fig, axes = plt.subplots(
            1,
            n,
            figsize=(4.0 * n, 4.0),
            constrained_layout=False,
            gridspec_kw={"wspace": wspace},
        )
        axes = np.atleast_1d(axes)
        mesh = None
        for idx, (ax, (q_nh_, q_oh_, z_), label) in enumerate(zip(axes, gridded, labels, strict=True)):
            mesh = draw_panel_q(
                ax,
                q_nh_,
                q_oh_,
                z_,
                cmap=plot_cmap,
                norm=norm,
                vmin=vmin,
                vmax=vmax,
                title=label,
                idx=idx,
                energy_grid=energy_grid,
                energy_q_nh=energy_q_nh,
                energy_q_oh=energy_q_oh,
                hide_nonfirst_y_ticks=True,
                show_xlabel=xlabel_only_idx is None or idx == xlabel_only_idx,
            )
            if mark_stationary:
                overlay_stationary_q(ax, stationary)
        if False and mark_stationary:
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
        [lam_dft, lam_hip, lam_eqv2],
        ["DFT", "HIP", "AD"],
        "relaxed_lambda_min.png",
        cbar_label=r"$\lambda_\mathrm{min}$ [eV $\AA^{-2}$ amu$^{-1}$]",
        cmap=lam_min_cmap,
        vmin=LAM_MIN_VMIN,
        vmax=LAM_MIN_VMAX,
        extend="max",
        mark_stationary=True,
        legend_between_titles=True,
        legend_anchor_axis=1,
        legend_anchor_offset=-0.03,
    )

    nmax = int(max(nneg_dft.max(), nneg_hip.max(), nneg_eqv2.max()))
    panel_row(
        [nneg_dft, nneg_hip, nneg_eqv2],
        ["DFT", "HIP", "AD"],
        "relaxed_n_negative.png",
        cbar_label="Negative Mode Count",
        cmap=editorial_sequential_cmap(),
        discrete=True,
        vmin=0,
        vmax=max(1, nmax),
        mark_stationary=True,
        legend_between_titles=True,
        legend_anchor_axis=1,
        legend_anchor_offset=-0.03,
        wspace=0.12,
        xlabel_only_idx=1,
    )

    panel_row(
        [align_dft, align_hip, align_eqv2],
        ["DFT", "HIP", "AD"],
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
        [hmae_hip, hmae_eqv2],
        ["HIP", "AD"],
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
        [cos_hip, cos_eqv2],
        ["HIP", "AD"],
        "relaxed_eckart_v1_cos_vs_dft.png",
        cbar_label=rf"$|\cos(v_1^\mathrm{{Eckart}})|$ vs DFT",
        cmap="viridis_r",
        vmin=0.5,
        vmax=1.0,
        mark_stationary=True,
        legend_between_titles=True,
    )

    panel_row(
        [fmae_hip, fmae_eqv2],
        ["HIP", "AD"],
        "relaxed_force_mae.png",
        cbar_label=r"Force MAE [eV $\AA^{-1}$]",
        cmap="viridis",
        vmin=0.0,
        vmax=0.10,
        extend="max",
        mark_stationary=True,
        legend_between_titles=True,
    )

    print("Wrote q-coordinate plots to", rel_to_repo(out_dir))


if __name__ == "__main__":
    main()
