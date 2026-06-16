#!/usr/bin/env python
"""Render the Transition1x glycine proton-transfer example with xyzrender."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hip.transition1x_dataset import Transition1xDataset


SPLIT = "test"
SAMPLE_ID = 5


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_transition1x_h5() -> Path:
    return project_root() / "data" / "transition1x.h5"


def symbols_from_atomic_numbers(atomic_numbers: np.ndarray) -> list[str]:
    z_to_symbol = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl"}
    return [z_to_symbol[int(z)] for z in atomic_numbers.tolist()]


def write_xyz(path: Path, symbols: list[str], coords: np.ndarray, comment: str) -> None:
    with path.open("w") as handle:
        handle.write(f"{len(symbols)}\n")
        handle.write(f"{comment}\n")
        for symbol, xyz in zip(symbols, coords, strict=False):
            handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def write_trajectory_xyz(
    path: Path,
    symbols: list[str],
    frames: list[np.ndarray],
    sample_label: str,
) -> None:
    with path.open("w") as handle:
        for frame_idx, coords in enumerate(frames):
            handle.write(f"{len(symbols)}\n")
            handle.write(f"{sample_label} frame={frame_idx:03d}\n")
            for symbol, xyz in zip(symbols, coords, strict=False):
                handle.write(f"{symbol:2s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


def read_trajectory_xyz(path: Path, n_atoms: int) -> list[np.ndarray]:
    lines = path.read_text().splitlines()
    frames: list[np.ndarray] = []
    idx = 0
    while idx < len(lines):
        nat = int(lines[idx].strip())
        if nat != n_atoms:
            raise ValueError(f"{path} has {nat} atoms, expected {n_atoms}")
        atom_lines = lines[idx + 2 : idx + 2 + n_atoms]
        coords = np.asarray([[float(x) for x in line.split()[1:4]] for line in atom_lines], dtype=float)
        frames.append(coords)
        idx += n_atoms + 2
    return frames


def write_individual_xyz_frames(
    frame_dir: Path,
    symbols: list[str],
    frames: list[np.ndarray],
    sample_label: str,
) -> None:
    frame_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frame_dir.glob("frame_*.xyz"):
        old_frame.unlink()
    for frame_idx, coords in enumerate(frames):
        write_xyz(frame_dir / f"frame_{frame_idx:03d}.xyz", symbols, coords, f"{sample_label} geodesic frame={frame_idx:03d}")


def run_command(command: str, *args: str) -> None:
    subprocess.run([*shlex.split(command), *args], check=True)


def run_geodesic_leg(
    command: str,
    input_path: Path,
    output_path: Path,
    nimages: int,
    tol: float,
    maxiter: int,
) -> None:
    run_command(
        command,
        str(input_path),
        "--output",
        str(output_path),
        "--nimages",
        str(nimages),
        "--tol",
        str(tol),
        "--maxiter",
        str(maxiter),
    )


def run_geodesic_path_through_anchors(
    work_dir: Path,
    symbols: list[str],
    anchors: np.ndarray,
    frames_per_leg: int,
    command: str,
    tol: float,
    maxiter: int,
    label: str,
) -> list[np.ndarray]:
    leg_nimages = frames_per_leg + 1
    frames: list[np.ndarray] = []

    for leg_idx, (start, stop) in enumerate(zip(anchors[:-1], anchors[1:], strict=True)):
        leg_label = f"{label}_leg_{leg_idx:02d}"
        leg_input = work_dir / f"geodesic_input_{leg_idx:02d}.xyz"
        leg_output = work_dir / f"geodesic_output_{leg_idx:02d}.xyz"

        write_trajectory_xyz(leg_input, symbols, [start, stop], leg_label)
        run_geodesic_leg(command, leg_input, leg_output, leg_nimages, tol, maxiter)

        leg_frames = read_trajectory_xyz(leg_output, len(symbols))
        leg_frames[0] = np.array(start, dtype=float, copy=True)
        leg_frames[-1] = np.array(stop, dtype=float, copy=True)
        frames.extend(leg_frames if leg_idx == 0 else leg_frames[1:])

    return frames


def load_panel_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default(size=size)


def load_sample(h5_path: Path):
    dataset = Transition1xDataset(str(h5_path), split=SPLIT, max_samples=SAMPLE_ID + 1)
    sample = dataset[SAMPLE_ID]
    symbols = symbols_from_atomic_numbers(sample.z.detach().cpu().numpy().astype(int))
    reactant = sample.pos_reactant.detach().cpu().numpy().reshape(-1, 3)
    transition_state = sample.pos_transition.detach().cpu().numpy().reshape(-1, 3)
    product = sample.pos_product.detach().cpu().numpy().reshape(-1, 3)
    return sample, symbols, reactant, transition_state, product


def likely_final_mep_images(
    h5_path: Path,
    split: str,
    formula: str,
    rxn: str,
    n_initial: int,
) -> tuple[np.ndarray, list[int | str]]:
    with h5py.File(h5_path, "r") as handle:
        group = handle[split][formula][rxn]
        positions = np.asarray(group["positions"], dtype=float)
        n_internal = n_initial - 2
        if n_internal <= 0:
            raise ValueError(f"n_initial must be at least 3, got {n_initial}")
        if (positions.shape[0] - n_initial) % n_internal != 0:
            raise ValueError(
                "Cannot infer final internal-image block: "
                f"n_frames={positions.shape[0]}, n_initial={n_initial}, n_internal={n_internal}"
            )

        start = positions.shape[0] - n_internal
        reactant = np.asarray(group["reactant"]["positions"], dtype=float)[0]
        product = np.asarray(group["product"]["positions"], dtype=float)[0]
        internal = positions[start:]

    images = np.concatenate([reactant[None], internal, product[None]], axis=0)
    indices: list[int | str] = ["reactant", *range(start, positions.shape[0]), "product"]
    return images, indices


def assemble_panel(image_paths: list[Path], labels: list[str], output_path: Path) -> None:
    images = [Image.open(path).convert("RGBA") for path in image_paths]
    width = max(image.width for image in images)
    height = max(image.height for image in images)
    title_height = 128
    gap = 18
    margin = 24
    panel = Image.new(
        "RGBA",
        (3 * width + 2 * gap + 2 * margin, height + title_height + 2 * margin),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    font = load_panel_font(72)

    for idx, (image, label) in enumerate(zip(images, labels, strict=True)):
        x = margin + idx * (width + gap) + (width - image.width) // 2
        y = margin + title_height
        panel.alpha_composite(image, (x, y))
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        draw.text(
            (margin + idx * (width + gap) + (width - text_width) / 2, margin),
            label,
            fill="black",
            font=font,
        )

    panel.convert("RGB").save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=Path, default=default_transition1x_h5())
    parser.add_argument("--output-dir", type=Path, default=Path("plots/glycine_pt_xyzrender"))
    parser.add_argument("--frames-per-leg", type=int, default=8)
    parser.add_argument(
        "--mep-initial-images",
        type=int,
        default=10,
        help="Number of images in the initial full NEB path used to infer the final internal-image block.",
    )
    parser.add_argument("--geodesic-tol", type=float, default=0.002)
    parser.add_argument("--geodesic-maxiter", type=int, default=15)
    parser.add_argument(
        "--geodesic-cmd",
        default="geodesic_interpolate",
        help="Command used to run geodesic-interpolate.",
    )
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument(
        "--xyzrender-cmd",
        default="xyzrender",
        help="Command used to run xyzrender.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    xyz_dir = out_dir / "xyz"
    png_dir = out_dir / "png"
    geodesic_dir = xyz_dir / "geodesic"
    frame_dir = xyz_dir / "geodesic_frames"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    png_dir.mkdir(parents=True, exist_ok=True)
    geodesic_dir.mkdir(parents=True, exist_ok=True)

    sample, symbols, reactant, transition_state, product = load_sample(args.h5)
    sample_label = f"Transition1x {SPLIT} sample_id={SAMPLE_ID} rxn={sample.rxn}"

    reactant_xyz = xyz_dir / "reactant.xyz"
    ts_xyz = xyz_dir / "transition_state.xyz"
    product_xyz = xyz_dir / "product.xyz"
    mep_xyz = xyz_dir / "mep_images.xyz"
    trajectory_xyz = xyz_dir / "reaction_path.xyz"
    write_xyz(reactant_xyz, symbols, reactant, f"{sample_label} reactant")
    write_xyz(ts_xyz, symbols, transition_state, f"{sample_label} transition_state")
    write_xyz(product_xyz, symbols, product, f"{sample_label} product")

    mep_images, mep_indices = likely_final_mep_images(
        args.h5,
        SPLIT,
        str(sample.formula),
        str(sample.rxn),
        args.mep_initial_images,
    )
    write_trajectory_xyz(mep_xyz, symbols, list(mep_images), f"{sample_label} likely_final_mep")

    frames = run_geodesic_path_through_anchors(
        geodesic_dir,
        symbols,
        mep_images,
        args.frames_per_leg,
        args.geodesic_cmd,
        args.geodesic_tol,
        args.geodesic_maxiter,
        "likely_final_mep",
    )
    write_trajectory_xyz(trajectory_xyz, symbols, frames, sample_label)
    write_individual_xyz_frames(frame_dir, symbols, frames, sample_label)

    orientation_ref = out_dir / "orientation_ref.xyz"
    render_common = [
        "--config",
        "pmol",
        "--hy",
        "--canvas-size",
        "900",
        "--ref",
        str(orientation_ref),
    ]
    render_specs = [
        (reactant_xyz, png_dir / "reactant.png", []),
        (ts_xyz, png_dir / "transition_state.png", ["--ts-bond", "5-10", "--ts-bond", "4-10"]),
        (product_xyz, png_dir / "product.png", []),
    ]
    for input_path, output_path, extra_args in render_specs:
        run_command(
            args.xyzrender_cmd,
            str(input_path),
            *render_common,
            *extra_args,
            "-o",
            str(output_path),
        )

    panel_path = out_dir / "glycine_pt_reactant_ts_product.png"
    assemble_panel(
        [png_dir / "reactant.png", png_dir / "transition_state.png", png_dir / "product.png"],
        ["Reactant", "Transition State", "Product"],
        panel_path,
    )

    gif_path = out_dir / "glycine_pt_reaction.gif"
    run_command(
        args.xyzrender_cmd,
        str(trajectory_xyz),
        "--config",
        "pmol",
        "--hy",
        "--gif-trj",
        "--trj-bonds",
        "--gif-fps",
        str(args.gif_fps),
        "-go",
        str(gif_path),
    )

    print(f"sample: split={SPLIT} sample_id={SAMPLE_ID} rxn={sample.rxn} formula={sample.formula}")
    print(f"likely MEP anchors: {mep_indices}")
    print(f"geodesic frames: {len(frames)}")
    print(f"panel: {panel_path}")
    print(f"gif: {gif_path}")
    print(f"trajectory_xyz: {trajectory_xyz}")
    print(f"individual frame xyz: {frame_dir}")


if __name__ == "__main__":
    main()
