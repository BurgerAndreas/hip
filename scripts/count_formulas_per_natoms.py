#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def count_distinct_formulas_per_natoms(csv_path: Path) -> dict[int, int]:
    natoms_to_formulas: dict[int, set[str]] = defaultdict(set)

    with csv_path.open(mode="r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)

        natoms_index = header.index("natoms")
        formula_index = header.index("formula")

        for row in reader:
            natoms_value = int(row[natoms_index])
            formula_value = row[formula_index]
            natoms_to_formulas[natoms_value].add(formula_value)

    return {natoms: len(formulas) for natoms, formulas in natoms_to_formulas.items()}


def write_output(results: dict[int, int], output_path: Path | None) -> None:
    sorted_items = sorted(results.items(), key=lambda kv: kv[0])

    print(f"Atoms: Distinct Formulas")
    for natoms, distinct_count in sorted_items:
        print(f"{natoms}: {distinct_count}")

    if output_path is not None:
        with output_path.open(mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["natoms", "distinct_formulas"])
            for natoms, distinct_count in sorted_items:
                writer.writerow([natoms, distinct_count])


"""
uv run scripts/count_formulas_per_natoms.py /ssd/Code/hip/metadata/dataset_metadata_ts1x-val.csv --out /ssd/Code/hip/metadata/dataset_metadata_ts1x-val_formulas_per_natoms.csv
uv run scripts/count_formulas_per_natoms.py /ssd/Code/hip/metadata/dataset_metadata_ts1x_hess_train_big.csv --out /ssd/Code/hip/metadata/dataset_metadata_ts1x_hess_train_big_formulas_per_natoms.csv
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Count distinct chemical formulas per natoms from a CSV with columns: index,natoms,formula,..."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to input CSV (e.g., /ssd/Code/hip/metadata/dataset_metadata_ts1x-val.csv)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional path to write results CSV (columns: natoms,distinct_formulas)",
    )

    args = parser.parse_args()

    results = count_distinct_formulas_per_natoms(args.csv_path)
    write_output(results, args.out)


if __name__ == "__main__":
    main()
