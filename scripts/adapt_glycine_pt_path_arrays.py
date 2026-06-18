#!/usr/bin/env python
"""Repackage glycine proton-slide path arrays into MEP-style MLIP files.

The MEP diagnostics plotters consume one NPZ per model with generic keys:
``energies``, ``forces``, ``hessians_cartesian``, ``coords_angstrom``,
``atomic_numbers``, ``q_nh``, ``q_oh``, and ``xi``. The rigid proton-slide
pipeline stores the same quantities in one ``path_arrays.npz`` with model
prefixes, so this adapter writes:

    runs/glycine_pt_path_n150/eqv2_autograd_arrays.npz
    runs/glycine_pt_path_n150/hip_v2_arrays.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_PATH_DIR = Path("runs/glycine_pt_path_n150")

COMMON_KEYS = ("coords_angstrom", "atomic_numbers", "q_nh", "q_oh", "xi")
MODEL_KEY_MAP = {
    "eqv2_autograd": {
        "output": "eqv2_autograd_arrays.npz",
        "energies": "eqv2_energy",
        "forces": "eqv2_forces",
        "hessians_cartesian": ("eqv2_hessians_cartesian", "eqv2_hessian_cartesian"),
    },
    "hip_v2": {
        "output": "hip_v2_arrays.npz",
        "energies": "hip_energy",
        "forces": "hip_forces",
        "hessians_cartesian": ("hip_hessians_cartesian", "hip_hessian_cartesian"),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-dir", type=Path, default=DEFAULT_PATH_DIR)
    parser.add_argument("--path-arrays", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise KeyError(key)
    return np.asarray(data[key])


def require_first_key(data: np.lib.npyio.NpzFile, keys: str | tuple[str, ...]) -> np.ndarray:
    if isinstance(keys, str):
        return require_key(data, keys)
    for key in keys:
        if key in data.files:
            return np.asarray(data[key])
    raise KeyError(" or ".join(keys))


def validate_frame_count(arrays: dict[str, np.ndarray], n_frames: int, label: str) -> None:
    for key, value in arrays.items():
        if key == "atomic_numbers":
            continue
        if value.shape[0] != n_frames:
            raise ValueError(f"{label} {key} has {value.shape[0]} frames, expected {n_frames}")


def build_model_arrays(data: np.lib.npyio.NpzFile, label: str, spec: dict[str, object]) -> dict[str, np.ndarray]:
    missing: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    for key in COMMON_KEYS:
        try:
            arrays[key] = require_key(data, key)
        except KeyError:
            missing.append(key)
    for output_key in ("energies", "forces", "hessians_cartesian"):
        try:
            arrays[output_key] = require_first_key(data, spec[output_key])  # type: ignore[index]
        except KeyError as exc:
            missing.append(str(exc).strip("'"))
    if missing:
        missing_text = ", ".join(missing)
        raise KeyError(
            f"{label} cannot be adapted because {missing_text} are missing from path_arrays.npz. "
            "Regenerate it with scripts/glycine_pt_path_scan.py so full Cartesian forces and Hessians are saved."
        )

    arrays["atomic_numbers"] = arrays["atomic_numbers"].astype(int)
    arrays["frame_id"] = np.arange(arrays["coords_angstrom"].shape[0], dtype=int)
    if "x_along" in data.files:
        arrays["x_along"] = np.asarray(data["x_along"], dtype=float)
    validate_frame_count(arrays, arrays["coords_angstrom"].shape[0], label)
    return arrays


def main() -> None:
    args = parse_args()
    path_arrays = args.path_arrays or args.path_dir / "path_arrays.npz"
    output_dir = args.output_dir or args.path_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_arrays: dict[str, tuple[Path, dict[str, np.ndarray]]] = {}
    errors: list[str] = []
    with np.load(path_arrays) as data:
        for label, spec in MODEL_KEY_MAP.items():
            try:
                arrays = build_model_arrays(data, label, spec)
            except (KeyError, ValueError) as exc:
                message = exc.args[0] if exc.args else str(exc)
                errors.append(str(message))
                continue
            output_path = output_dir / str(spec["output"])
            model_arrays[label] = (output_path, arrays)
    if errors:
        detail = "\n".join(f"- {message}" for message in errors)
        raise SystemExit(f"ERROR: cannot adapt {path_arrays}\n{detail}") from None
    for output_path, arrays in model_arrays.values():
        np.savez_compressed(output_path, **arrays)
        print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
