#!/usr/bin/env python
"""Plot DFT-relaxed glycine PT TS render next to negative-mode count panels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (str(ROOT), str(ROOT / "plotting")):
    if extra not in sys.path:
        sys.path.insert(0, extra)

from plot_glycine_pt_relaxed_ts_n_negative_panel import (  # noqa: E402
    plot_panel,
    rel_to_repo,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        type=Path,
        default=ROOT / "runs" / "glycine_pt_scan_relaxed_dft_eval",
    )
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


def main() -> None:
    args = parse_args()
    if args.output is None:
        args.output = (
            ROOT
            / "runs"
            / "glycine_pt_scan_relaxed"
            / "plots_relaxed_dft_c"
            / "relaxed_transition_state_n_negative_modes.png"
        )
    out = plot_panel(args)
    print(f"Wrote {rel_to_repo(out)}")


if __name__ == "__main__":
    main()
