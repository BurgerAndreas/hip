#!/usr/bin/env python
"""Prepare and combine glycine proton-transfer TS renders with CV diagnostic panels."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

from plot_style import PLOTLY_FONT_COLOR

matplotlib.use("Agg")

DEFAULT_TS_RENDER = Path("plots/glycine_pt_xyzrender/png/transition_state.png")
DEFAULT_CROPPED_TS = Path("plots/glycine_pt_xyzrender/png/transition_state_cropped_rotated_left.png")
DEFAULT_MARKED_TS = Path("plots/glycine_pt_xyzrender/png/transition_state_cropped_rotated_left_marked.png")
DEFAULT_DIAGNOSTIC = Path(
    "plots/glycine_pt_scan_n36/dft_cv_diagnostics/glycine_pt_n_negative_modes_methods_qoh_crop2p3.png"
)


def trim_white(image: Image.Image, threshold: int = 252, padding: int = 28) -> Image.Image:
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


def crop_rotate_ts(
    ts_render: Path,
    output_path: Path,
    rotation_deg: float = 90.0,
    trim_threshold: int = 252,
    trim_padding: int = 28,
    post_rotate_padding: int = 18,
) -> Image.Image:
    transition = Image.open(ts_render)
    trimmed = trim_white(transition, threshold=trim_threshold, padding=trim_padding)
    rotated = trimmed.rotate(
        rotation_deg,
        expand=True,
        fillcolor=(255, 255, 255, 0),
        resample=Image.Resampling.BICUBIC,
    )
    rotated = trim_white(rotated, threshold=trim_threshold, padding=post_rotate_padding)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rotated.convert("RGB").save(output_path)
    return rotated


def annotate_ts_labels(
    cropped_ts: Path,
    output_path: Path,
    q_nh_xy: tuple[float, float] = (890.0, 105.0),
    q_oh_xy: tuple[float, float] = (1505.0, 475.0),
    fontsize: float = 34.0,
    dpi: int = 300,
) -> Image.Image:
    img = Image.open(cropped_ts).convert("RGB")
    width, height = img.size

    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    ax.text(
        q_nh_xy[0],
        q_nh_xy[1],
        r"$q_\mathrm{NH}$",
        color=PLOTLY_FONT_COLOR,
        fontsize=fontsize,
        ha="center",
        va="center",
    )
    ax.text(
        q_oh_xy[0],
        q_oh_xy[1],
        r"$q_\mathrm{OH}$",
        color=PLOTLY_FONT_COLOR,
        fontsize=fontsize,
        ha="center",
        va="center",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return Image.open(output_path).convert("RGB")


def default_combined_path(diagnostic_path: Path) -> Path:
    stem = diagnostic_path.stem
    if stem.startswith("glycine_pt_"):
        suffix = stem.removeprefix("glycine_pt_")
        return diagnostic_path.with_name(f"glycine_pt_transition_state_{suffix}{diagnostic_path.suffix}")
    return diagnostic_path.with_name(f"transition_state_{stem}{diagnostic_path.suffix}")


def combine_panel(
    marked_ts: Path,
    diagnostic: Path,
    output_path: Path,
    gap_px: int = 12,
) -> Image.Image:
    left = Image.open(marked_ts).convert("RGB")
    right = Image.open(diagnostic).convert("RGB")
    target_height = right.height
    scale = target_height / left.height
    left_scaled = left.resize(
        (round(left.width * scale), target_height),
        Image.Resampling.LANCZOS,
    )

    panel = Image.new(
        "RGB",
        (left_scaled.width + gap_px + right.width, target_height),
        "white",
    )
    panel.paste(left_scaled, (0, 0))
    panel.paste(right, (left_scaled.width + gap_px, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    return panel


def add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ts-render", type=Path, default=DEFAULT_TS_RENDER)
    parser.add_argument("--cropped-ts", type=Path, default=DEFAULT_CROPPED_TS)
    parser.add_argument("--marked-ts", type=Path, default=DEFAULT_MARKED_TS)
    parser.add_argument("--rotation-deg", type=float, default=90.0)
    parser.add_argument("--q-nh-x", type=float, default=890.0)
    parser.add_argument("--q-nh-y", type=float, default=105.0)
    parser.add_argument("--q-oh-x", type=float, default=1505.0)
    parser.add_argument("--q-oh-y", type=float, default=475.0)
    parser.add_argument("--label-fontsize", type=float, default=34.0)
    parser.add_argument("--dpi", type=int, default=300)


def add_combine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--marked-ts", type=Path, default=DEFAULT_MARKED_TS)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gap-px", type=int, default=12)


def add_all_args(parser: argparse.ArgumentParser) -> None:
    add_prepare_args(parser)
    parser.add_argument("--diagnostic", type=Path, default=DEFAULT_DIAGNOSTIC)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--gap-px", type=int, default=12)


def run_prepare(args: argparse.Namespace) -> tuple[Path, Path]:
    crop_rotate_ts(args.ts_render, args.cropped_ts, rotation_deg=args.rotation_deg)
    annotate_ts_labels(
        args.cropped_ts,
        args.marked_ts,
        q_nh_xy=(args.q_nh_x, args.q_nh_y),
        q_oh_xy=(args.q_oh_x, args.q_oh_y),
        fontsize=args.label_fontsize,
        dpi=args.dpi,
    )
    print(f"Wrote cropped TS render: {args.cropped_ts}")
    print(f"Wrote annotated TS render: {args.marked_ts}")
    return args.cropped_ts, args.marked_ts


def run_combine(args: argparse.Namespace) -> Path:
    output_path = args.output or default_combined_path(args.diagnostic)
    combine_panel(args.marked_ts, args.diagnostic, output_path, gap_px=args.gap_px)
    print(f"Wrote combined panel: {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare",
        help="Crop, rotate, and annotate the xyzrender transition-state PNG.",
    )
    add_prepare_args(prepare_parser)
    prepare_parser.set_defaults(func=run_prepare)

    combine_parser = subparsers.add_parser(
        "combine",
        help="Place an annotated TS render next to a diagnostic heatmap panel.",
    )
    add_combine_args(combine_parser)
    combine_parser.set_defaults(func=run_combine)

    all_parser = subparsers.add_parser(
        "all",
        help="Run prepare and combine in one step.",
    )
    add_all_args(all_parser)

    def run_all(args: argparse.Namespace) -> Path:
        run_prepare(args)
        return run_combine(args)

    all_parser.set_defaults(func=run_all)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
