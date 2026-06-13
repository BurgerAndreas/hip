#!/usr/bin/env python
"""Collect ORCA energies for the glycine proton-transfer scan.

Run this after ORCA outputs are copied back beside the generated input files.
It writes:
    - orca_energies.{csv,parquet}
    - energy_comparison_hip_v2_orca.{csv,parquet}, when HIP energies exist
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


HARTREE_TO_KCAL_MOL = 627.509474
FINAL_ENERGY_RE = re.compile(r"FINAL SINGLE POINT ENERGY\s+(-?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)")


def parse_orca_output(path: Path) -> dict[str, object]:
    energy_hartree = None
    n_final_energy_lines = 0
    terminated_normally = False
    scf_converged = False

    with path.open(errors="replace") as handle:
        for line in handle:
            match = FINAL_ENERGY_RE.search(line)
            if match:
                energy_hartree = float(match.group(1))
                n_final_energy_lines += 1
            if "ORCA TERMINATED NORMALLY" in line:
                terminated_normally = True
            if "SCF CONVERGED AFTER" in line:
                scf_converged = True

    return {
        "orca_output_path": str(path),
        "orca_output_exists": path.exists(),
        "orca_terminated_normally": terminated_normally,
        "orca_scf_converged": scf_converged,
        "orca_n_final_energy_lines": n_final_energy_lines,
        "orca_energy_hartree": energy_hartree,
    }


def output_path_from_input(input_path: str, output_dir: Path | None) -> Path:
    inp = Path(input_path)
    if output_dir is not None:
        return output_dir / f"{inp.stem}.out"
    return inp.with_suffix(".out")


def write_orca_tables(scan_dir: Path, output_dir: Path | None) -> tuple[Path, Path, Path | None]:
    manifest = pd.read_csv(scan_dir / "scan_manifest.csv")
    rows: list[dict[str, object]] = []

    for row in manifest.to_dict(orient="records"):
        out_path = output_path_from_input(str(row["orca_input_path"]), output_dir)
        parsed = parse_orca_output(out_path) if out_path.exists() else {
            "orca_output_path": str(out_path),
            "orca_output_exists": False,
            "orca_terminated_normally": False,
            "orca_scf_converged": False,
            "orca_n_final_energy_lines": 0,
            "orca_energy_hartree": None,
        }
        rows.append(row | parsed)

    orca_df = pd.DataFrame(rows)
    energies = orca_df["orca_energy_hartree"].dropna()
    if len(energies):
        e_min = float(energies.min())
        orca_df["orca_energy_relative_hartree"] = orca_df["orca_energy_hartree"] - e_min
        orca_df["orca_energy_relative_kcalmol"] = (
            orca_df["orca_energy_relative_hartree"] * HARTREE_TO_KCAL_MOL
        )
    else:
        orca_df["orca_energy_relative_hartree"] = None
        orca_df["orca_energy_relative_kcalmol"] = None

    orca_parquet = scan_dir / "orca_energies.parquet"
    orca_csv = scan_dir / "orca_energies.csv"
    orca_df.to_parquet(orca_parquet, index=False)
    orca_df.to_csv(orca_csv, index=False)

    comparison_path = None
    hip_path = scan_dir / "hip_v2_energies.parquet"
    if hip_path.exists():
        hip_df = pd.read_parquet(hip_path)
        comparison = hip_df.merge(
            orca_df[
                [
                    "grid_id",
                    "orca_output_path",
                    "orca_output_exists",
                    "orca_terminated_normally",
                    "orca_scf_converged",
                    "orca_energy_hartree",
                    "orca_energy_relative_hartree",
                    "orca_energy_relative_kcalmol",
                ]
            ],
            on="grid_id",
            how="left",
        )
        comparison_path = scan_dir / "energy_comparison_hip_v2_orca.parquet"
        comparison.to_parquet(comparison_path, index=False)
        comparison.to_csv(scan_dir / "energy_comparison_hip_v2_orca.csv", index=False)

    return orca_parquet, orca_csv, comparison_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-dir", type=Path, default=Path("runs/glycine_pt_scan"))
    parser.add_argument(
        "--orca-output-dir",
        type=Path,
        default=None,
        help="Directory containing ORCA .out files. Defaults to next to each .inp file.",
    )
    args = parser.parse_args()

    orca_parquet, orca_csv, comparison_path = write_orca_tables(
        args.scan_dir,
        args.orca_output_dir,
    )
    print(f"ORCA energies: {orca_parquet}")
    print(f"ORCA energies CSV: {orca_csv}")
    if comparison_path is not None:
        print(f"HIP/ORCA comparison: {comparison_path}")


if __name__ == "__main__":
    main()
