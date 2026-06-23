#!/usr/bin/env python
"""Plot relaxed glycine PT TS render next to negative-mode count panels."""
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

from plot_style import PLOTLY_FONT_COLOR, apply_plot_style, finish_axis  # noqa: E402
from scripts.cache_glycine_pt_orca_vibrations import (  # noqa: E402
    NEG_EIGVAL_THRESHOLD,
    vibrational_eigh,
)

HARTREE_TO_KCAL = 627.5094740631
SIGMA_YTICKS = (2.5, 3.0, 3.5, 4.0)
STATIONARY_STYLE = {
    "reactant": dict(marker="o", label="Reactant", facecolor="white", edgecolor="black"),
    "product": dict(marker="^", label="Product", facecolor="white", edgecolor="black"),
    "ts": dict(marker="*", label="TS", facecolor="white", edgecolor="black"),
}


def axes_label_size() -> float:
    return float(matplotlib.rcParams["axes.labelsize"])


def tick_label_size() -> float:
    return float(matplotlib.rcParams["xtick.labelsize"])


def title_size() -> float:
    return float(matplotlib.rcParams["axes.titlesize"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-dir", type=Path, default=ROOT / "runs" / "glycine_pt_scan_relaxed")
    parser.add_argument("--vib-cache", type=Path, default=None)
    parser.add_argument("--hip-arrays", type=Path, default=None)
    parser.add_argument("--eqv2-arrays", type=Path, default=None)
    parser.add_argument("--stationary-json", type=Path, default=None)
    parser.add_argument(
        "--ts-image",
        type=Path,
        default=ROOT / "plots" / "glycine_pt_xyzrender" / "png" / "transition_state_cropped_rotated_left_marked.png",
        help="Annotated transition-state render to place at the left.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
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
    """Pivot node values onto the regular relaxed (s, sigma) grid."""
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


def add_stationary_legend(fig: plt.Figure, axes: list[plt.Axes]) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        return
    fig.draw_without_rendering()
    pos_anchor = axes[1].get_position()
    trans = blended_transform_factory(fig.transFigure, axes[0].transAxes)
    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(pos_anchor.x0 - 0.03, 1.01),
        bbox_transform=trans,
        ncol=3,
        fontsize=tick_label_size() * 0.72,
        frameon=True,
        edgecolor="none",
        markerscale=0.95,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    for text in legend.get_texts():
        text.set_color(PLOTLY_FONT_COLOR)


def finish_surface_axis(ax: plt.Axes, *, idx: int) -> None:
    finish_axis(ax)
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


def build_ts_negative_panel_image(
    *,
    vib_path: Path,
    hip_path: Path,
    eqv2_path: Path,
    stat_path: Path,
    ts_image: Path,
    dpi: int,
) -> Image.Image:
    """Render TS render plus DFT/HIP/AD negative-mode surfaces as one RGB image."""
    apply_plot_style()

    stationary = json.loads(stat_path.read_text()) if stat_path.exists() else {}
    dft = np.load(vib_path)
    hip = np.load(hip_path)
    eqv2 = np.load(eqv2_path)

    grid_id = dft["grid_id"]
    for other, name in ((hip, "HIP"), (eqv2, "AD")):
        if not np.array_equal(grid_id, other["grid_id"]):
            raise ValueError(f"grid_id ordering mismatch between DFT and {name}")

    s = dft["s"]
    sigma = dft["sigma"]
    coords = dft["coords_angstrom"]
    masses = dft["masses_amu"]
    energy = (dft["energy_hartree_engrad"] - dft["energy_hartree_engrad"].min()) * HARTREE_TO_KCAL
    energy_xs, energy_ys, energy_grid = make_grid(s, sigma, energy)

    panels = [
        ("DFT", dft["n_negative"].astype(int)),
        ("HIP", n_negative_from_hessians(hip["hessians_cartesian"], coords, masses)),
        ("AD", n_negative_from_hessians(eqv2["hessians_cartesian"], coords, masses)),
    ]
    gridded = [(label, *make_grid(s, sigma, values)) for label, values in panels]

    finite = np.concatenate([z.compressed() for _, _, _, z in gridded])
    vmin = int(np.nanmin(finite))
    vmax = int(np.nanmax(finite))
    cmap = plt.get_cmap("viridis", vmax - vmin + 1)
    norm = BoundaryNorm(np.arange(vmin - 0.5, vmax + 1.5, 1.0), cmap.N)

    fig, axes = plt.subplots(1, 3, figsize=(4.4 * 3, 4.0), constrained_layout=True)
    axes = list(np.atleast_1d(axes))
    mesh = None
    levels = contour_levels(energy_grid.compressed(), step=10.0)
    for idx, (ax, (label, xs, ys, z)) in enumerate(zip(axes, gridded, strict=True)):
        mesh = ax.pcolormesh(xs, ys, z, shading="nearest", cmap=cmap, norm=norm)
        ax.contour(energy_xs, energy_ys, energy_grid, levels=levels, colors="k", linewidths=0.5, alpha=0.55)
        overlay_stationary(ax, stationary)
        ax.set_title(label, fontsize=title_size() * 0.85)
        ax.set_xlabel(r"$s = q_\mathrm{NH} - q_\mathrm{OH}$ [$\AA$]")
        ax.set_ylabel(r"$\sigma = q_\mathrm{NH} + q_\mathrm{OH}$ [$\AA$]" if idx == 0 else "")
        ax.axvline(0.0, color="0.35", lw=0.6, ls="--", alpha=0.7)
        finish_surface_axis(ax, idx=idx)

    assert mesh is not None
    add_stationary_legend(fig, axes)
    fig.draw_without_rendering()
    cax = axes[-1].inset_axes([1.05, 0.0, 0.05, 1.0])
    cbar = fig.colorbar(mesh, cax=cax, ticks=np.arange(vmin, vmax + 1), spacing="proportional")
    cbar.set_label("negative mode count", fontsize=axes_label_size())
    cbar.ax.minorticks_off()
    cbar.ax.tick_params(which="both", axis="both", length=0, labelsize=tick_label_size())
    cbar.outline.set_visible(False)

    heatmap_buf = io.BytesIO()
    fig.savefig(heatmap_buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    heatmap_buf.seek(0)

    right = Image.open(heatmap_buf).convert("RGB")
    ts_img = trim_white(Image.open(ts_image), padding=4).convert("RGB")
    target_height = round(right.height * 0.88)
    scale = target_height / ts_img.height
    left = ts_img.resize((round(ts_img.width * scale), target_height), Image.Resampling.LANCZOS)

    gap_px = 34
    panel = Image.new("RGB", (left.width + gap_px + right.width, right.height), "white")
    panel.paste(left, (0, round((right.height - left.height) / 2)))
    panel.paste(right, (left.width + gap_px, 0))
    return panel


def plot_panel(args: argparse.Namespace) -> Path:
    scan = args.scan_dir
    vib_path = args.vib_cache or scan / "orca_vib_cache.npz"
    hip_path = args.hip_arrays or scan / "hip_v2_arrays.npz"
    eqv2_path = args.eqv2_arrays or scan / "eqv2_autograd_arrays.npz"
    stat_path = resolve_stationary_json(args, scan)
    output = args.output or scan / "plots_relaxed_c" / "relaxed_transition_state_n_negative_modes.png"

    panel = build_ts_negative_panel_image(
        vib_path=vib_path,
        hip_path=hip_path,
        eqv2_path=eqv2_path,
        stat_path=stat_path,
        ts_image=args.ts_image,
        dpi=args.dpi,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output)
    return output


def main() -> None:
    out = plot_panel(parse_args())
    print(f"Wrote {rel_to_repo(out)}")


if __name__ == "__main__":
    main()
