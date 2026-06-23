#!/usr/bin/env python
"""Stack glycine PT panels into one figure: relaxed TS negative modes (top row
sourced from a pre-rendered DFT/HIP/AD negative-mode image), MEP Hessian
eigenvalues, and xi-projected force residuals with Hessian MAE."""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import BoundaryNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory, offset_copy  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "plotting")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from plot_style import (  # noqa: E402
    DFT_COLOR,
    HIP_COLOR,
    LEFTNET_CF_FORCE_COLOR,
    LINE_WIDTH,
    MARKER_SIZE,
    PLOTLY_FONT_COLOR,
    apply_invisible_ticks,
    apply_plot_style,
    finish_axis,
    model_color,
)
from scripts.cache_glycine_pt_orca_vibrations import (  # noqa: E402
    NEG_EIGVAL_THRESHOLD,
    vibrational_eigh,
)

HARTREE_TO_KCAL = 627.5094740631

HARTREE_TO_EV = 27.211386245988

DEFAULT_SCAN_DIR = ROOT / "runs" / "glycine_pt_scan_relaxed_dft_eval"
DEFAULT_MEP_DIR = ROOT / "runs" / "glycine_pt_mep_145"
DEFAULT_TS_IMAGE = (
    ROOT / "plots" / "glycine_pt_xyzrender" / "png" / "transition_state_cropped_rotated_left_marked.png"
)
DEFAULT_ROW1_IMAGE = (
    ROOT / "runs" / "glycine_pt_scan_relaxed" / "plots_relaxed_dft_q_c" / "relaxed_n_negative.png"
)

DFT_LABEL = "DFT"
HIP_LABEL = "HIP"
EQV2_LABEL = "EqV2 AD"
EQV2_MECH_LABEL = "EqV2 AD"
LEFTNET_CF_LABEL = "LeftNet-CF AD"

SIGMA_YTICKS = (2.5, 3.0, 3.5, 4.0)
XI_XTICKS = (-1.0, 0.0, 1.0)
STATIONARY_STYLE = {
    "reactant": dict(marker="o", label="Reactant", facecolor="white", edgecolor="black"),
    "product": dict(marker="^", label="Product", facecolor="white", edgecolor="black"),
    "ts": dict(marker="*", label="TS", facecolor="white", edgecolor="black"),
}
STATIONARY_LEGEND_ORDER = ("reactant", "ts", "product")
STATIONARY_LEGEND_FONTSIZE_SCALE = 1.18
PANEL_LABEL_SIZE = 18
PANEL_LABEL_COLOR = "#2F4565"
ROW_LABEL_Y_DOTS = 28
ROW_LABEL_Y_DOTS_BC_EXTRA = 48
ROW_LABEL_X_DOTS = -72

METHOD_COLORS = {
    DFT_LABEL: DFT_COLOR,
    HIP_LABEL: HIP_COLOR,
    EQV2_LABEL: model_color(EQV2_LABEL),
    EQV2_MECH_LABEL: model_color(EQV2_LABEL),
    LEFTNET_CF_LABEL: LEFTNET_CF_FORCE_COLOR,
}
METHOD_MARKERS = {DFT_LABEL: "D", HIP_LABEL: "s", EQV2_LABEL: "o"}
METHOD_LINESTYLES = {DFT_LABEL: "-", HIP_LABEL: "-", EQV2_LABEL: "-"}
PLOT_METHOD_ORDER = (DFT_LABEL, EQV2_LABEL, HIP_LABEL)
DFT_EIGENVALUE_COLOR = "#5A5A5A"
NEGATIVE_MODE_DODGE = {DFT_LABEL: -0.06, HIP_LABEL: 0.0, EQV2_LABEL: 0.06}
METHOD_ZORDER = {DFT_LABEL: 2, EQV2_LABEL: 3, HIP_LABEL: 4}

XLABEL = r"$\xi = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]"
X_LABEL_PAD = 1.5
FORCE_PROJ_XI = r"F\cdot\widehat{\xi}"
AD_EMPHASIS_LINE_WIDTH = LINE_WIDTH + 0.4

O_ATOM = 3
N_ATOM = 4
H_TRANSFER_ATOM = 9
MASS_BY_Z = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 15: 30.974, 16: 32.065, 17: 35.453}


def rel_to_repo(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def apply_xi_xticks(ax: plt.Axes) -> None:
    ax.set_xticks(XI_XTICKS)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))


def axes_label_size() -> float:
    return float(matplotlib.rcParams["axes.labelsize"])


def tick_label_size() -> float:
    return float(matplotlib.rcParams["xtick.labelsize"])


def title_size() -> float:
    return float(matplotlib.rcParams["axes.titlesize"])


def apply_axis_typography(
    ax: plt.Axes,
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    title_fontsize: float | None = None,
) -> None:
    """Apply uniform tick, label, and title sizes across all plot rows."""
    if x_label is not None:
        ax.set_xlabel(x_label, fontsize=axes_label_size(), labelpad=X_LABEL_PAD)
    if y_label is not None:
        ax.set_ylabel(y_label, fontsize=axes_label_size())
    if title is not None:
        ax.set_title(title, fontsize=title_fontsize if title_fontsize is not None else title_size())
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size())


def ordered_method_labels(labels) -> list[str]:
    label_set = set(labels)
    return [label for label in PLOT_METHOD_ORDER if label in label_set]


def method_plot_marker(label: str, *, both_mlip_markered: bool) -> str:
    if both_mlip_markered and label == HIP_LABEL:
        return "D"
    return METHOD_MARKERS.get(label, "o")


def method_zorder(label: str) -> int:
    return METHOD_ZORDER.get(label, 3)


def dft_dot_lineplot_kwargs() -> dict:
    """Black dots only for DFT, matching the middle eigenvalue panels."""
    return {
        "marker": ".",
        "markersize": MARKER_SIZE * 1.8,
        "markeredgecolor": DFT_EIGENVALUE_COLOR,
        "markeredgewidth": 0.0,
        "lw": 0,
        "linestyle": "none",
        "color": DFT_EIGENVALUE_COLOR,
        "zorder": method_zorder(DFT_LABEL),
    }


def setup_axis(
    ax: plt.Axes,
    x_label: str,
    *,
    y_label: str | None = None,
    title: str | None = None,
    title_fontsize: float | None = None,
) -> None:
    apply_axis_typography(
        ax,
        x_label=x_label,
        y_label=y_label,
        title=title,
        title_fontsize=title_fontsize,
    )
    finish_axis(ax)
    apply_invisible_ticks(ax)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size())


def style_axis(ax: plt.Axes, *, x_label: str | None = None, y_label: str | None = None) -> None:
    apply_axis_typography(ax, x_label=x_label, y_label=y_label)
    finish_axis(ax)
    apply_invisible_ticks(ax)
    ax.tick_params(axis="both", which="major", labelsize=tick_label_size())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=DEFAULT_SCAN_DIR)
    parser.add_argument("--mep-dir", type=Path, default=DEFAULT_MEP_DIR)
    parser.add_argument("--vib-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--stationary-json", type=Path, default=None)
    parser.add_argument("--ts-image", type=Path, default=DEFAULT_TS_IMAGE)
    parser.add_argument("--row1-image", type=Path, default=DEFAULT_ROW1_IMAGE)
    parser.add_argument("--orca-cache", type=Path, default=None)
    parser.add_argument("--mep-hip-arrays", type=Path, default=None)
    parser.add_argument("--mep-eqv2-arrays", type=Path, default=None)
    parser.add_argument("--leftnet-cf-arrays", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--x-axis", choices=["xi", "frame"], default="xi")
    parser.add_argument("--n-eigs", type=int, default=6)
    parser.add_argument("--negative-threshold", type=float, default=-1e-6)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def resolve_stationary_json(args: argparse.Namespace, scan: Path) -> Path:
    if args.stationary_json is not None:
        return args.stationary_json
    local = scan / "stationary_points.json"
    if local.exists():
        return local
    return ROOT / "runs" / "glycine_pt_scan_relaxed" / "stationary_points.json"


def contour_levels(values: np.ndarray, step: float = 10.0) -> np.ndarray:
    finite = np.asarray(values[np.isfinite(values)], dtype=float)
    if finite.size == 0:
        return np.array([0.0, step])
    lo = np.floor(finite.min() / step) * step
    hi = np.ceil(finite.max() / step) * step
    if hi <= lo:
        hi = lo + step
    return np.arange(lo, hi + 0.5 * step, step)


def trim_white(image: Image.Image, threshold: int = 252, padding: int = 10) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = Image.new("RGB", rgba.size, "white")
    rgb.paste(rgba, mask=rgba.getchannel("A"))
    pix = rgb.load()
    width, height = rgb.size
    min_x, min_y = width, height
    max_x = max_y = -1
    for y in range(height):
        for x in range(width):
            r, g, b = pix[x, y]
            if min(r, g, b) < threshold:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    if max_x < 0:
        return rgba
    return rgba.crop(
        (
            max(min_x - padding, 0),
            max(min_y - padding, 0),
            min(max_x + padding + 1, width),
            min(max_y + padding + 1, height),
        )
    )


def make_grid(s: np.ndarray, sigma: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    s_r = np.round(np.asarray(s, dtype=float), 6)
    sig_r = np.round(np.asarray(sigma, dtype=float), 6)
    xs = np.unique(s_r)
    ys = np.unique(sig_r)
    xi = {v: i for i, v in enumerate(xs)}
    yi = {v: i for i, v in enumerate(ys)}
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)
    for sv, gv, val in zip(s_r, sig_r, np.asarray(values, dtype=float), strict=True):
        grid[yi[gv], xi[sv]] = val
    return xs, ys, np.ma.masked_invalid(grid)


def n_negative_from_hessians(hessians_ev: np.ndarray, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    counts = np.empty(hessians_ev.shape[0], dtype=int)
    for i in range(hessians_ev.shape[0]):
        evals, _ = vibrational_eigh(hessians_ev[i], coords[i], masses)
        counts[i] = int((evals < -NEG_EIGVAL_THRESHOLD).sum())
    return counts


def overlay_stationary(ax: plt.Axes, stationary: dict) -> None:
    if not stationary:
        return
    for key, style in STATIONARY_STYLE.items():
        if key not in stationary:
            continue
        ax.scatter(
            [stationary[key]["s"]],
            [stationary[key]["sigma"]],
            marker=style["marker"],
            s=170 if style["marker"] == "*" else 90,
            c=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=1.1,
            zorder=6,
            label=style["label"],
        )


def overlay_stationary_on_curve(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    frames: dict[str, int],
    *,
    base_size: float = 55.0,
    star_size: float = 130.0,
) -> None:
    """Mark reactant/product/TS frames on a line plot with the top-row glyphs."""
    for key, style in STATIONARY_STYLE.items():
        frame = frames[key]
        ax.scatter(
            [x[frame]],
            [y[frame]],
            marker=style["marker"],
            s=star_size if style["marker"] == "*" else base_size,
            facecolors=style["facecolor"],
            edgecolors=style["edgecolor"],
            linewidths=1.0,
            zorder=12,
            clip_on=False,
        )


def align_row_left(fig: plt.Figure, axes: list[plt.Axes], target_left: float) -> None:
    """Shift/stretch a row of axes so its leftmost content reaches ``target_left``
    while keeping the right edge fixed."""
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    cur_left = min(ax.get_tightbbox(renderer).transformed(inv).x0 for ax in axes)
    shift = cur_left - target_left
    if shift <= 1e-4:
        return
    positions = [ax.get_position() for ax in axes]
    x0 = min(p.x0 for p in positions)
    x1 = max(p.x1 for p in positions)
    new_x0 = x0 - shift
    scale = (x1 - new_x0) / (x1 - x0)
    for ax, p in zip(axes, positions, strict=True):
        ax.set_position([new_x0 + (p.x0 - x0) * scale, p.y0, p.width * scale, p.height])


def stationary_legend_handles() -> list[Line2D]:
    handles = []
    for key in STATIONARY_LEGEND_ORDER:
        style = STATIONARY_STYLE[key]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="none",
                markersize=9.5,
                markerfacecolor=style["facecolor"],
                markeredgecolor=style["edgecolor"],
                markeredgewidth=1.1,
                label=style["label"],
            )
        )
    return handles


def row_label_x_figure(fig: plt.Figure, ref_ax: plt.Axes) -> float:
    """Shared in-axes x for row labels, measured from the reference row's top-left."""
    transform = offset_copy(
        ref_ax.transAxes,
        fig=fig,
        x=ROW_LABEL_X_DOTS,
        y=ROW_LABEL_Y_DOTS,
        units="dots",
    )
    return float(fig.transFigure.inverted().transform(transform.transform((0, 1)))[0])


def add_row_labels(fig: plt.Figure, anchor_axes: list[plt.Axes]) -> None:
    """Add bold a/b/c row labels aligned on one vertical line inside the figure."""
    fig.draw_without_rendering()
    x_label = row_label_x_figure(fig, anchor_axes[1])
    for panel_label, ax in zip("abc", anchor_axes, strict=True):
        y_dots = ROW_LABEL_Y_DOTS + (ROW_LABEL_Y_DOTS_BC_EXTRA if panel_label in "bc" else 0)
        base = blended_transform_factory(fig.transFigure, ax.transAxes)
        transform = offset_copy(base, fig=fig, x=0, y=y_dots, units="dots")
        ax.text(
            x_label,
            1.0,
            panel_label,
            transform=transform,
            fontsize=PANEL_LABEL_SIZE,
            fontfamily="DejaVu Sans",
            fontweight="bold",
            color=PANEL_LABEL_COLOR,
            va="top",
            ha="left",
            clip_on=False,
            zorder=20,
        )


def add_stationary_legend(fig: plt.Figure, gs_row) -> None:
    """Place Reactant / TS / Product legend centered above the top row."""
    fig.draw_without_rendering()
    y_top = gs_row.get_position(fig).y1
    legend = fig.legend(
        handles=stationary_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, y_top + 0.008),
        bbox_transform=fig.transFigure,
        ncol=3,
        fontsize=tick_label_size() * STATIONARY_LEGEND_FONTSIZE_SCALE,
        frameon=True,
        edgecolor="none",
        markerscale=1.05,
        borderaxespad=0,
        handletextpad=0.4,
        columnspacing=1.1,
    )
    for text in legend.get_texts():
        text.set_color(PLOTLY_FONT_COLOR)


def finish_surface_axis(ax: plt.Axes, *, idx: int) -> None:
    finish_axis(ax)
    apply_xi_xticks(ax)
    ax.set_yticks(SIGMA_YTICKS)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.tick_params(
        axis="both",
        which="major",
        length=0,
        width=0.8,
        bottom=True,
        left=True,
        color=PLOTLY_FONT_COLOR,
        labelsize=tick_label_size(),
    )
    if idx > 0:
        ax.tick_params(axis="y", left=False, labelleft=False)


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


def vibrational_eigh_path(
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


def compute_vib_diagnostics(method_label, hessians, coords, atomic_numbers, threshold: float):
    eval_rows = []
    mode_rows = []
    for hessian, frame_coords in zip(hessians, coords, strict=True):
        evals, modes = vibrational_eigh_path(hessian, frame_coords, atomic_numbers)
        eval_rows.append(evals)
        mode_rows.append(modes)
    evals_arr = np.stack(eval_rows)
    return {
        "label": method_label,
        "evals": evals_arr,
        "modes_mw": np.stack(mode_rows),
        "n_negative": (evals_arr < threshold).sum(axis=1).astype(int),
    }


def load_orca_vib_or_compute(orca_cache: Path, hessians, coords, atomic_numbers, threshold: float):
    with np.load(orca_cache) as data:
        if {"vib_evals_ev_ang2_amu", "vib_modes_mw"}.issubset(data.files):
            evals = np.asarray(data["vib_evals_ev_ang2_amu"], dtype=float)
            modes = np.asarray(data["vib_modes_mw"], dtype=float)
            return {
                "label": DFT_LABEL,
                "evals": evals,
                "modes_mw": modes,
                "n_negative": (evals < threshold).sum(axis=1).astype(int),
            }
    return compute_vib_diagnostics(DFT_LABEL, hessians, coords, atomic_numbers, threshold)


def distance_gradient(coords: np.ndarray, atom_a: int, atom_b: int) -> np.ndarray:
    grad = np.zeros_like(coords, dtype=float)
    vec = coords[atom_a] - coords[atom_b]
    dist = max(float(np.linalg.norm(vec)), 1e-12)
    unit = vec / dist
    grad[atom_a] = unit
    grad[atom_b] = -unit
    return grad


def xi_unit_direction(coords: np.ndarray) -> np.ndarray:
    directions = np.empty_like(coords, dtype=float)
    for i, frame in enumerate(coords):
        xi_dir = distance_gradient(frame, N_ATOM, H_TRANSFER_ATOM) - distance_gradient(
            frame, O_ATOM, H_TRANSFER_ATOM
        )
        flat = xi_dir.reshape(-1)
        directions[i] = xi_dir / max(float(np.linalg.norm(flat)), 1e-12)
    return directions


def projected_force(forces: np.ndarray, direction: np.ndarray) -> np.ndarray:
    flat_f = forces.reshape(forces.shape[0], -1)
    flat_t = direction.reshape(direction.shape[0], -1)
    return np.einsum("ij,ij->i", flat_f, flat_t)


def symmetrize(hessians: np.ndarray) -> np.ndarray:
    return 0.5 * (hessians + np.swapaxes(hessians, -1, -2))


def hessian_element_mae(model_h: np.ndarray, ref_h: np.ndarray) -> np.ndarray:
    diff = symmetrize(model_h) - symmetrize(ref_h)
    return np.mean(np.abs(diff.reshape(diff.shape[0], -1)), axis=1)


def draw_ts_negative_panel(
    fig: plt.Figure,
    gs_row,
    *,
    row1_image: Path,
    ts_image: Path,
    dpi: int,
) -> Image.Image:
    """Build the top-row composite: TS render plus a pre-rendered DFT/HIP/AD
    negative-mode image (the three surfaces with colorbar)."""
    right = trim_white(Image.open(row1_image), padding=4).convert("RGB")
    ts_img = trim_white(Image.open(ts_image), padding=4).convert("RGB")
    target_height = round(right.height * 0.88)
    scale = target_height / ts_img.height
    left = ts_img.resize((round(ts_img.width * scale), target_height), Image.Resampling.LANCZOS)

    gap_px = 34
    panel = Image.new("RGB", (left.width + gap_px + right.width, right.height), "white")
    panel.paste(left, (0, round((right.height - left.height) / 2)))
    panel.paste(right, (left.width + gap_px, 0))
    return panel


def place_top_panel(fig: plt.Figure, gs_row, panel: Image.Image, lower_axes: list[plt.Axes]) -> plt.Axes:
    """Place the composite top-row image so it spans the full content width of the
    rows below (removing the left-label inset) and sits flush at the top."""
    aspect = panel.width / panel.height
    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x0s, x1s = [], []
    for ax in lower_axes:
        bbox = ax.get_tightbbox(renderer)
        if bbox is None:
            continue
        bbox_fig = bbox.transformed(inv)
        x0s.append(bbox_fig.x0)
        x1s.append(bbox_fig.x1)
    left = min(x0s)
    right = max(x1s)
    width_frac = right - left
    fig_w, fig_h = fig.get_size_inches()
    height_frac = width_frac * fig_w / (aspect * fig_h)
    y_top = gs_row.get_position(fig).y1
    ax = fig.add_axes([left, y_top - height_frac, width_frac, height_frac])
    ax.imshow(panel, aspect="auto")
    ax.axis("off")
    return ax


def draw_lowest_eigenvalues(
    fig: plt.Figure,
    gs_row,
    x: np.ndarray,
    x_label: str,
    vib: dict[str, dict],
    n_eigs: int,
    stationary_frames: dict[str, int],
    *,
    use_xi_xticks: bool = True,
) -> None:
    n_eigs = min(n_eigs, min(diag["evals"].shape[1] for diag in vib.values()))
    ncols = min(3, n_eigs)
    nrows = int(np.ceil(n_eigs / ncols))
    sub = gs_row.subgridspec(nrows, ncols, hspace=0.12, wspace=0.34)
    axes = [fig.add_subplot(sub[i // ncols, i % ncols]) for i in range(nrows * ncols)]

    plot_labels = ordered_method_labels(vib)
    x_span = float(np.max(x) - np.min(x))
    shared_xlim = (float(np.min(x) - 0.015 * x_span), float(np.max(x) + 0.015 * x_span))

    for idx, ax in enumerate(axes):
        if idx >= n_eigs:
            ax.axis("off")
            continue
        for label in plot_labels:
            diag = vib[label]
            is_dft = label == DFT_LABEL
            plot_color = DFT_EIGENVALUE_COLOR if is_dft else METHOD_COLORS.get(label)
            line_kwargs = dft_dot_lineplot_kwargs() if is_dft else {
                "marker": None,
                "markersize": 0,
                "markeredgecolor": None,
                "markeredgewidth": None,
                "lw": LINE_WIDTH,
                "linestyle": METHOD_LINESTYLES.get(label, "-"),
                "color": plot_color,
                "zorder": method_zorder(label),
            }
            sns.lineplot(
                x=x,
                y=diag["evals"][:, idx],
                ax=ax,
                label=label,
                **line_kwargs,
            )
        overlay_stationary_on_curve(ax, x, vib[DFT_LABEL]["evals"][:, idx], stationary_frames)
        y_label = rf"$\lambda_{{{idx + 1}}}$ [eV $\AA^{{-2}}$ amu$^{{-1}}$]"
        y_values = np.concatenate([vib[label]["evals"][:, idx] for label in plot_labels])
        y_span = float(np.max(y_values) - np.min(y_values))
        y_pad = max(0.02 * y_span, 0.01)
        ax.set_ylim(float(np.min(y_values) - y_pad), float(np.max(y_values) + y_pad))
        if idx == 0:
            ax.set_ylim(-17, 1)
            ax.set_yticks([0, -5, -10, -15])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%d"))
        ax.set_xlim(*shared_xlim)
        if use_xi_xticks:
            apply_xi_xticks(ax)
        setup_axis(
            ax,
            "",
            y_label=y_label,
        )
        legend = ax.get_legend()
        if idx == 0:
            new_legend = ax.legend(
                fontsize=tick_label_size(),
                markerscale=1.4,
                frameon=True,
                edgecolor="none",
                loc="center left",
                bbox_to_anchor=(-0.02, 0.20),
            )
            for text in new_legend.get_texts():
                text.set_color(PLOTLY_FONT_COLOR)
        elif legend is not None:
            legend.remove()


def draw_force_xi_residual(
    fig: plt.Figure,
    gs_row,
    xi: np.ndarray,
    g_xi: dict[str, np.ndarray],
    hessian_mae: dict[str, np.ndarray],
    compare_labels: tuple[str, ...],
    stationary_frames: dict[str, int],
) -> None:
    sub = gs_row.subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(sub[0, i]) for i in range(3)]

    sns.lineplot(
        x=xi,
        y=g_xi[DFT_LABEL],
        ax=axes[0],
        label=DFT_LABEL,
        **dft_dot_lineplot_kwargs(),
    )
    for label in compare_labels:
        if label not in g_xi:
            continue
        lw = AD_EMPHASIS_LINE_WIDTH if label == EQV2_MECH_LABEL else LINE_WIDTH
        sns.lineplot(
            x=xi,
            y=g_xi[label],
            ax=axes[0],
            color=METHOD_COLORS[label],
            lw=lw,
            label=label,
        )
    overlay_stationary_on_curve(axes[0], xi, g_xi[DFT_LABEL], stationary_frames)

    for label in compare_labels:
        if label not in g_xi:
            continue
        lw = AD_EMPHASIS_LINE_WIDTH if label == EQV2_MECH_LABEL else LINE_WIDTH
        sns.lineplot(
            x=xi,
            y=g_xi[label] - g_xi[DFT_LABEL],
            ax=axes[1],
            color=METHOD_COLORS[label],
            lw=lw,
            label="_nolegend_",
        )
    overlay_stationary_on_curve(axes[1], xi, np.zeros_like(xi), stationary_frames)

    for label in compare_labels:
        if label not in hessian_mae:
            continue
        lw = AD_EMPHASIS_LINE_WIDTH if label == EQV2_MECH_LABEL else LINE_WIDTH
        sns.lineplot(
            x=xi,
            y=hessian_mae[label],
            ax=axes[2],
            color=METHOD_COLORS[label],
            lw=lw,
            label="_nolegend_",
        )

    x_span = float(np.max(xi) - np.min(xi))
    shared_xlim = (float(np.min(xi) - 0.015 * x_span), float(np.max(xi) + 0.015 * x_span))
    style_axis(
        axes[0],
        x_label="",
        y_label=rf"${FORCE_PROJ_XI}$ [eV/$\AA$]",
    )
    style_axis(
        axes[1],
        x_label=XLABEL,
        y_label=r"$(F\cdot\widehat{\xi}) - (F\cdot\widehat{\xi})_\mathrm{DFT}$ [eV/$\AA$]",
    )
    style_axis(
        axes[2],
        x_label="",
        y_label=r"mean $|H-H_\mathrm{DFT}|$ [eV $\AA^{-2}$]",
    )
    force_legend = axes[0].legend(
        fontsize=tick_label_size(),
        frameon=True,
        edgecolor="none",
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.05),
    )
    for text in force_legend.get_texts():
        text.set_color(PLOTLY_FONT_COLOR)
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlim(*shared_xlim)
        apply_xi_xticks(ax)
    y_bottom, y_top = axes[2].get_ylim()
    pad_frac = 0.03
    y_marker = 10 ** (np.log10(y_bottom) + pad_frac * (np.log10(y_top) - np.log10(y_bottom)))
    overlay_stationary_on_curve(axes[2], xi, np.full(xi.shape, y_marker), stationary_frames)
    axes[2].set_ylim(y_bottom, y_top)


def load_mep_data(args: argparse.Namespace):
    mep_dir = args.mep_dir
    orca_path = args.orca_cache or mep_dir / "orca_vib_cache.npz"
    hip_path = args.mep_hip_arrays or mep_dir / "hip_v2_arrays.npz"
    eqv2_path = args.mep_eqv2_arrays or mep_dir / "eqv2_autograd_arrays.npz"
    leftnet_cf_path = args.leftnet_cf_arrays
    if leftnet_cf_path is None and (mep_dir / "leftnet_cf_arrays.npz").exists():
        leftnet_cf_path = mep_dir / "leftnet_cf_arrays.npz"

    with np.load(orca_path) as data:
        coords = np.asarray(data["coords_angstrom"], dtype=float)
        atomic_numbers = np.asarray(data["atomic_numbers"], dtype=int)
        q_nh = np.asarray(data["q_nh"], dtype=float)
        q_oh = np.asarray(data["q_oh"], dtype=float)
        dft_forces = np.asarray(data["forces_ev_ang"], dtype=float)
        dft_hessians = np.asarray(data["hessian_ev_ang2"], dtype=float)
        dft_energy = np.asarray(data["energy_hartree_engrad"], dtype=float)

    with np.load(hip_path) as data:
        hip_forces = np.asarray(data["forces"], dtype=float)
        hip_hessians = np.asarray(data["hessians_cartesian"], dtype=float)

    with np.load(eqv2_path) as data:
        eqv2_forces = np.asarray(data["forces"], dtype=float)
        eqv2_hessians = np.asarray(data["hessians_cartesian"], dtype=float)

    leftnet_cf_forces = None
    leftnet_cf_hessians = None
    if leftnet_cf_path is not None and leftnet_cf_path.exists():
        with np.load(leftnet_cf_path) as data:
            leftnet_cf_forces = np.asarray(data["forces"], dtype=float)
            leftnet_cf_hessians = np.asarray(data["hessians_cartesian"], dtype=float)

    xi = q_nh - q_oh
    order = np.argsort(xi)
    xi = xi[order]
    coords = coords[order]
    dft_forces = dft_forces[order]
    dft_hessians = dft_hessians[order]
    dft_energy = dft_energy[order]
    hip_forces = hip_forces[order]
    hip_hessians = hip_hessians[order]
    eqv2_forces = eqv2_forces[order]
    eqv2_hessians = eqv2_hessians[order]
    if leftnet_cf_forces is not None:
        leftnet_cf_forces = leftnet_cf_forces[order]
        leftnet_cf_hessians = leftnet_cf_hessians[order]

    return {
        "xi": xi,
        "coords": coords,
        "atomic_numbers": atomic_numbers,
        "dft_hessians": dft_hessians,
        "hip_hessians": hip_hessians,
        "eqv2_hessians": eqv2_hessians,
        "leftnet_cf_hessians": leftnet_cf_hessians,
        "dft_forces": dft_forces,
        "hip_forces": hip_forces,
        "eqv2_forces": eqv2_forces,
        "leftnet_cf_forces": leftnet_cf_forces,
        "dft_energy": dft_energy,
        "orca_path": orca_path,
    }


def plot_combined(args: argparse.Namespace) -> Path:
    apply_plot_style()

    mep_dir = args.mep_dir
    output = args.output or Path("plots") / mep_dir.name / "glycine_pt_combined_panel_relaxed_q_c.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    mep = load_mep_data(args)
    xi = mep["xi"]
    x = xi if args.x_axis == "xi" else np.arange(xi.size)
    x_label = XLABEL if args.x_axis == "xi" else "frame index"

    print("Computing vibrational diagnostics for MEP Hessians...", flush=True)
    vib = {
        DFT_LABEL: load_orca_vib_or_compute(
            mep["orca_path"],
            mep["dft_hessians"],
            mep["coords"],
            mep["atomic_numbers"],
            args.negative_threshold,
        ),
        HIP_LABEL: compute_vib_diagnostics(
            HIP_LABEL,
            mep["hip_hessians"],
            mep["coords"],
            mep["atomic_numbers"],
            args.negative_threshold,
        ),
        EQV2_LABEL: compute_vib_diagnostics(
            EQV2_LABEL,
            mep["eqv2_hessians"],
            mep["coords"],
            mep["atomic_numbers"],
            args.negative_threshold,
        ),
    }

    xi_dir = xi_unit_direction(mep["coords"])
    g_xi = {
        DFT_LABEL: projected_force(mep["dft_forces"], xi_dir),
        EQV2_MECH_LABEL: projected_force(mep["eqv2_forces"], xi_dir),
    }
    compare_labels = (EQV2_MECH_LABEL,)
    hessian_mae = {
        EQV2_MECH_LABEL: hessian_element_mae(mep["eqv2_hessians"], mep["dft_hessians"]),
    }
    if mep["leftnet_cf_forces"] is not None:
        g_xi[LEFTNET_CF_LABEL] = projected_force(mep["leftnet_cf_forces"], xi_dir)
        compare_labels = (EQV2_MECH_LABEL, LEFTNET_CF_LABEL)
        hessian_mae[LEFTNET_CF_LABEL] = hessian_element_mae(
            mep["leftnet_cf_hessians"], mep["dft_hessians"]
        )

    stationary_frames = {
        "reactant": 0,
        "product": int(xi.size - 1),
        "ts": int(np.argmax(mep["dft_energy"])),
    }

    fig = plt.figure(figsize=(16.5, 15.9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.22, 1.55, 0.95], hspace=0.22)

    draw_lowest_eigenvalues(
        fig, gs[1], x, x_label, vib, args.n_eigs, stationary_frames, use_xi_xticks=args.x_axis == "xi"
    )
    row2_axes = list(fig.axes)
    draw_force_xi_residual(fig, gs[2], xi, g_xi, hessian_mae, compare_labels, stationary_frames)
    row3_axes = [ax for ax in fig.axes if ax not in row2_axes]

    fig.draw_without_rendering()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    target_left = min(ax.get_tightbbox(renderer).transformed(inv).x0 for ax in row2_axes)
    align_row_left(fig, row3_axes, target_left)

    lower_axes = list(fig.axes)

    panel = draw_ts_negative_panel(
        fig,
        gs[0],
        row1_image=args.row1_image,
        ts_image=args.ts_image,
        dpi=args.dpi,
    )
    top_ax = place_top_panel(fig, gs[0], panel, lower_axes)
    add_stationary_legend(fig, gs[0])
    add_row_labels(fig, [top_ax, row2_axes[0], row3_axes[0]])

    fig.savefig(output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return output


def main() -> None:
    out = plot_combined(parse_args())
    print(f"Wrote {rel_to_repo(out)}")


if __name__ == "__main__":
    main()
