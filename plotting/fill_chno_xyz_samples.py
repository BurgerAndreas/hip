#!/usr/bin/env python3
"""Fill data/large with CHNO-only XYZ samples by exact atom count.

The script preserves existing entries in data/large/molecules.txt and adds
enough geometries to reach a target number of samples for each atom count.
It prefers local LMDB datasets selected through their metadata parquet files,
then falls back to deterministic RDKit conformer generation if needed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ase.io
import pandas as pd
import requests
from ase import Atoms

from hip.ff_lmdb import LmdbDataset
from hip.path_config import fix_dataset_path

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


ALLOWED_ATOMIC_NUMBERS = {1, 6, 7, 8}
ALLOWED_SYMBOLS = {"H", "C", "N", "O"}


@dataclass(frozen=True)
class MoleculeEntry:
    name: str
    atoms: int
    fmt: str
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill CHNO-only XYZ samples for exact atom-count buckets."
    )
    parser.add_argument("--data-large-dir", default="data/large")
    parser.add_argument("--out-subdir", default="chno_xyz")
    parser.add_argument("--min-atoms", type=int, default=30)
    parser.add_argument("--max-atoms", type=int, default=100)
    parser.add_argument("--samples-per-count", type=int, default=5)
    parser.add_argument(
        "--metadata",
        nargs="*",
        default=[
            "dataset_metadata_ts1x-val.parquet",
            "dataset_metadata_RGD1.parquet",
            "dataset_metadata_ts1x_hess_train_big.parquet",
        ],
        help="Metadata parquet files to use for candidate selection.",
    )
    parser.add_argument(
        "--scratch-dir",
        default=None,
        help="Optional scratch directory for temporary files. The current workflow streams samples and normally does not need it.",
    )
    parser.add_argument(
        "--no-rdkit-fallback",
        action="store_true",
        help="Fail instead of generating deterministic RDKit conformers for uncovered buckets.",
    )
    parser.add_argument(
        "--pubchem-records-per-formula",
        type=int,
        default=12,
        help="Maximum PubChem CIDs to inspect per formula.",
    )
    parser.add_argument(
        "--pubchem-formulas-per-count",
        type=int,
        default=30,
        help="Maximum formula candidates to query per atom count.",
    )
    return parser.parse_args()


def molecule_path(data_large_dir: Path, entry: MoleculeEntry) -> Path:
    return data_large_dir / f"{entry.name}.{entry.fmt.lower()}"


def read_molecules_file(molecules_file: Path) -> list[MoleculeEntry]:
    if not molecules_file.exists():
        return []

    entries: list[MoleculeEntry] = []
    with molecules_file.open() as handle:
        for line in handle.readlines()[1:]:
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            entries.append(
                MoleculeEntry(
                    name=parts[0],
                    atoms=int(parts[1]),
                    fmt=parts[2],
                    source=parts[3],
                )
            )
    return entries


def write_molecules_file(molecules_file: Path, entries: list[MoleculeEntry]) -> None:
    deduped = {entry.name: entry for entry in entries}
    ordered = sorted(deduped.values(), key=lambda item: (item.atoms, item.name))

    with molecules_file.open("w") as handle:
        handle.write("Name\tAtoms\tFormat\tSource\n")
        for entry in ordered:
            handle.write(f"{entry.name}\t{entry.atoms}\t{entry.fmt}\t{entry.source}\n")


def is_chno_atoms(atoms: Atoms) -> bool:
    return set(atoms.get_atomic_numbers()).issubset(ALLOWED_ATOMIC_NUMBERS)


def count_existing_chno(
    data_large_dir: Path,
    entries: list[MoleculeEntry],
    min_atoms: int,
    max_atoms: int,
) -> dict[int, list[str]]:
    by_count: dict[int, list[str]] = defaultdict(list)

    for entry in entries:
        if not min_atoms <= entry.atoms <= max_atoms:
            continue
        path = molecule_path(data_large_dir, entry)
        if not path.exists():
            continue
        try:
            atoms = ase.io.read(path)
        except Exception as exc:
            print(f"Skipping unreadable existing geometry {path}: {exc}")
            continue
        if len(atoms) == entry.atoms and is_chno_atoms(atoms):
            by_count[entry.atoms].append(entry.name)

    return by_count


def dataset_name_from_metadata(metadata_path: Path) -> str:
    name = metadata_path.name
    if name.startswith("dataset_metadata_") and name.endswith(".parquet"):
        return name.removeprefix("dataset_metadata_").removesuffix(".parquet") + ".lmdb"
    raise ValueError(f"Cannot infer dataset name from metadata path: {metadata_path}")


def formula_is_chno(formula: str) -> bool:
    elements = re.findall(r"([A-Z][a-z]?)(?:\d*)", formula)
    return bool(elements) and set(elements).issubset(ALLOWED_SYMBOLS)


def load_candidate_indices(
    metadata_paths: list[Path],
    needed_counts: set[int],
) -> dict[str, dict[int, list[int]]]:
    candidates: dict[str, dict[int, list[int]]] = {}

    for metadata_path in metadata_paths:
        if not metadata_path.exists():
            print(f"Metadata file not found, skipping: {metadata_path}")
            continue

        dataset_name = dataset_name_from_metadata(metadata_path)
        df = pd.read_parquet(metadata_path)
        natoms_column = "n_atoms" if "n_atoms" in df.columns else "natoms"
        if natoms_column not in df.columns or "index" not in df.columns:
            print(f"Metadata lacks natoms/index columns, skipping: {metadata_path}")
            continue

        mask = df[natoms_column].isin(sorted(needed_counts))
        if "formula" in df.columns:
            mask &= df["formula"].astype(str).map(formula_is_chno)

        grouped: dict[int, list[int]] = defaultdict(list)
        for atom_count, dataset_idx in df.loc[
            mask, [natoms_column, "index"]
        ].itertuples(index=False, name=None):
            grouped[int(atom_count)].append(int(dataset_idx))

        if grouped:
            candidates[dataset_name] = grouped
            print(
                f"Loaded candidates from {metadata_path}: "
                f"{sum(len(v) for v in grouped.values())} samples across {len(grouped)} counts"
            )

    return candidates


def write_xyz(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(path, atoms, format="xyz")


def cid_to_atoms_from_sdf(cid: int) -> Atoms | None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    try:
        response = requests.get(url, params={"record_type": "3d"}, timeout=30)
        if response.status_code != 200:
            return None
        from io import StringIO

        atoms = ase.io.read(
            StringIO(response.content.decode("utf-8", errors="ignore")),
            format="sdf",
        )
    except Exception:
        return None
    return atoms


def cid_to_isomeric_smiles(cid: int) -> str | None:
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
        f"{cid}/property/IsomericSMILES/JSON"
    )
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        props = response.json()["PropertyTable"]["Properties"][0]
        return props.get("IsomericSMILES")
    except Exception:
        return None


def pubchem_charge(cid: int) -> int | None:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/Charge/JSON"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        props = response.json()["PropertyTable"]["Properties"][0]
        return int(props["Charge"])
    except Exception:
        return None


def atoms_from_smiles(smiles: str, seed: int) -> Atoms | None:
    if not RDKIT_AVAILABLE:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass

    conformer = mol.GetConformer()
    positions = [list(conformer.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    return Atoms(numbers=numbers, positions=positions)


def pubchem_cids_for_formula(formula: str, max_records: int) -> list[int]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastformula/{formula}/cids/JSON"
    try:
        response = requests.get(url, params={"MaxRecords": max_records}, timeout=30)
        if response.status_code != 200:
            return []
        return [int(cid) for cid in response.json()["IdentifierList"]["CID"]]
    except Exception:
        return []


def formula_string(c_count: int, h_count: int, n_count: int, o_count: int) -> str:
    parts = [f"C{c_count}", f"H{h_count}"]
    if n_count:
        parts.append("N" if n_count == 1 else f"N{n_count}")
    if o_count:
        parts.append("O" if o_count == 1 else f"O{o_count}")
    return "".join(parts)


def formula_candidates_for_atom_count(atom_count: int) -> list[str]:
    candidates: list[tuple[tuple[int, int, int, int], str]] = []

    for c_count in range(2, min(45, atom_count) + 1):
        for n_count in range(0, 7):
            for o_count in range(0, 9):
                h_count = atom_count - c_count - n_count - o_count
                if h_count < 0:
                    continue
                heavy_count = c_count + n_count + o_count
                if heavy_count < 4:
                    continue
                # Degree of unsaturation for CHNO formulas. Keep plausible neutral organics.
                dbe_twice = 2 * c_count + 2 + n_count - h_count
                if dbe_twice < 0 or dbe_twice % 2 != 0:
                    continue
                dbe = dbe_twice // 2
                if dbe > 12:
                    continue
                if h_count > 2 * c_count + n_count + 2:
                    continue
                # Avoid formula lists dominated by heteroatom chains.
                if c_count < max(3, heavy_count // 2):
                    continue
                hetero_count = n_count + o_count
                score = (
                    abs(hetero_count - 2),
                    abs(dbe - 3),
                    abs(heavy_count - atom_count // 3),
                    c_count,
                )
                formula = formula_string(c_count, h_count, n_count, o_count)
                candidates.append((score, formula))

    ordered = []
    seen = set()
    for _, formula in sorted(candidates):
        if formula not in seen:
            ordered.append(formula)
            seen.add(formula)
    return ordered


def add_pubchem_samples(
    data_large_dir: Path,
    out_subdir: str,
    entries: list[MoleculeEntry],
    existing_by_count: dict[int, list[str]],
    min_atoms: int,
    max_atoms: int,
    samples_per_count: int,
    records_per_formula: int,
    formulas_per_count: int,
) -> None:
    existing_names = {entry.name for entry in entries}
    used_cids = {
        int(entry.source.split(":", 1)[1].split(":", 1)[0])
        for entry in entries
        if entry.source.startswith("pubchem3d:")
        or entry.source.startswith("pubchem_smiles:")
    }

    for atom_count in range(min_atoms, max_atoms + 1):
        if len(existing_by_count[atom_count]) >= samples_per_count:
            continue

        for formula in formula_candidates_for_atom_count(atom_count)[:formulas_per_count]:
            if len(existing_by_count[atom_count]) >= samples_per_count:
                break

            cids = pubchem_cids_for_formula(formula, records_per_formula)
            if cids:
                print(f"N={atom_count}: {formula} -> {len(cids)} PubChem CIDs")
            time.sleep(0.15)

            for cid in cids:
                if len(existing_by_count[atom_count]) >= samples_per_count:
                    break
                if cid in used_cids:
                    continue
                charge = pubchem_charge(cid)
                if charge != 0:
                    print(f"Skipping CID {cid}: PubChem charge is {charge}")
                    time.sleep(0.15)
                    continue

                atoms = cid_to_atoms_from_sdf(cid)
                source = f"pubchem3d:{cid}:{formula}"
                if atoms is None:
                    smiles = cid_to_isomeric_smiles(cid)
                    if not smiles:
                        continue
                    atoms = atoms_from_smiles(smiles, seed=cid)
                    source = f"pubchem_smiles:{cid}:{smiles}"

                if atoms is None or len(atoms) != atom_count or not is_chno_atoms(atoms):
                    continue

                digest = hashlib.sha1(str(cid).encode("utf-8")).hexdigest()[:10]
                sample_idx = len(existing_by_count[atom_count]) + 1
                stem = f"{out_subdir}/n{atom_count:03d}_pubchem_{sample_idx:02d}_{digest}"
                if stem in existing_names:
                    continue

                write_xyz(data_large_dir / f"{stem}.xyz", atoms)
                entries.append(MoleculeEntry(stem, atom_count, "xyz", source))
                existing_names.add(stem)
                used_cids.add(cid)
                existing_by_count[atom_count].append(stem)
                print(f"Added {stem} ({atom_count} atoms) from {source}")
                time.sleep(0.15)


def sample_to_atoms(sample) -> Atoms:
    atomic_numbers = sample.z.detach().cpu().numpy().astype(int)
    positions = sample.pos.detach().cpu().numpy()
    return Atoms(numbers=atomic_numbers, positions=positions)


def add_lmdb_samples(
    data_large_dir: Path,
    out_subdir: str,
    entries: list[MoleculeEntry],
    existing_by_count: dict[int, list[str]],
    candidates_by_dataset: dict[str, dict[int, list[int]]],
    samples_per_count: int,
) -> None:
    existing_names = {entry.name for entry in entries}

    for dataset_name, by_count in candidates_by_dataset.items():
        missing_counts = [
            atom_count
            for atom_count, names in existing_by_count.items()
            if len(names) < samples_per_count and atom_count in by_count
        ]
        if not missing_counts:
            continue

        try:
            dataset = LmdbDataset(fix_dataset_path(dataset_name))
        except Exception as exc:
            print(f"Could not open {dataset_name}, skipping: {exc}")
            continue

        dataset_tag = dataset_name.removesuffix(".lmdb").replace("-", "_")
        for atom_count in missing_counts:
            for dataset_idx in by_count[atom_count]:
                if len(existing_by_count[atom_count]) >= samples_per_count:
                    break

                try:
                    atoms = sample_to_atoms(dataset[dataset_idx])
                except Exception as exc:
                    print(f"Could not read {dataset_name}[{dataset_idx}]: {exc}")
                    continue

                if len(atoms) != atom_count or not is_chno_atoms(atoms):
                    continue

                digest = hashlib.sha1(
                    f"{dataset_name}:{dataset_idx}".encode("utf-8")
                ).hexdigest()[:10]
                stem = f"{out_subdir}/n{atom_count:03d}_{dataset_tag}_{digest}"
                if stem in existing_names:
                    continue

                write_xyz(data_large_dir / f"{stem}.xyz", atoms)
                source = f"{dataset_name}:{dataset_idx}"
                entries.append(MoleculeEntry(stem, atom_count, "xyz", source))
                existing_names.add(stem)
                existing_by_count[atom_count].append(stem)
                print(f"Added {stem} ({atom_count} atoms) from {source}")

        if hasattr(dataset, "close_db"):
            dataset.close_db()


def branched_carbon_chain(main_carbons: int, branch_positions: tuple[int, ...]) -> str:
    tokens = []
    for idx in range(main_carbons):
        token = "C"
        if idx in branch_positions:
            token += "(C)"
        tokens.append(token)
    return "".join(tokens)


def alkane_smiles(c_count: int) -> list[str]:
    """Return simple alkane isomers with exactly c_count carbons."""
    candidates = ["C" * c_count]

    if c_count >= 5:
        main = c_count - 1
        for pos in range(1, min(main - 1, 6)):
            candidates.append(branched_carbon_chain(main, (pos,)))

    if c_count >= 7:
        main = c_count - 2
        branch_pairs = [(1, 2), (1, 3), (2, 4), (2, 5), (3, 5)]
        for pair in branch_pairs:
            if max(pair) < main - 1:
                candidates.append(branched_carbon_chain(main, pair))

    return candidates


def hetero_chain_smiles(c_count: int, hetero: str) -> list[str]:
    """Return alcohol/ether or amine positional isomers with one heteroatom."""
    candidates: list[str] = []

    # Terminal functional group plus internal heteroatom placements.
    candidates.append("C" * c_count + hetero)
    for pos in range(1, c_count):
        candidates.append("C" * pos + hetero + "C" * (c_count - pos))

    # Add a few methyl-branched variants while preserving the exact atom count.
    if c_count >= 7:
        main = c_count - 1
        for pos in range(1, min(main - 1, 5)):
            chain = branched_carbon_chain(main, (pos,))
            candidates.append(chain + hetero)

    return candidates


def candidate_smiles_for_total_atoms(atom_count: int) -> list[str]:
    """Construct carbon-rich neutral CHNO candidates for an exact total atom count.

    Saturated acyclic molecules have total atoms:
      alkane: C_n H_(2n+2) => 3n + 2
      one alcohol/ether oxygen: C_n H_(2n+2) O => 3n + 3
      one amine nitrogen: C_n H_(2n+3) N => 3n + 4
    Those three families cover every total atom count.
    """
    remainder = (atom_count - 2) % 3
    if remainder == 0:
        c_count = (atom_count - 2) // 3
        raw_candidates = alkane_smiles(c_count)
    elif remainder == 1:
        c_count = (atom_count - 3) // 3
        raw_candidates = hetero_chain_smiles(c_count, "O")
    else:
        c_count = (atom_count - 4) // 3
        raw_candidates = hetero_chain_smiles(c_count, "N")

    canonical: list[str] = []
    seen: set[str] = set()
    for smiles in raw_candidates:
        mol = Chem.MolFromSmiles(smiles) if RDKIT_AVAILABLE else None
        if mol is None:
            continue
        key = Chem.MolToSmiles(mol)
        if key not in seen:
            canonical.append(key)
            seen.add(key)
    return canonical


def generate_rdkit_atoms(atom_count: int, variant: int) -> tuple[Atoms, str] | None:
    if not RDKIT_AVAILABLE:
        return None

    candidates = candidate_smiles_for_total_atoms(atom_count)
    if not candidates:
        return None

    for attempt in range(len(candidates)):
        smiles = candidates[(variant + attempt) % len(candidates)]
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canonical = Chem.MolToSmiles(mol)
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() != atom_count:
            continue

        params = AllChem.ETKDGv3()
        params.randomSeed = 1729 + 1009 * atom_count + variant + attempt
        if AllChem.EmbedMolecule(mol, params) != 0:
            continue
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol):
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            else:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass

        conformer = mol.GetConformer()
        positions = [
            list(conformer.GetAtomPosition(i))
            for i in range(mol.GetNumAtoms())
        ]
        numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        atoms = Atoms(numbers=numbers, positions=positions)
        return atoms, canonical

    return None


def add_rdkit_fallback_samples(
    data_large_dir: Path,
    out_subdir: str,
    entries: list[MoleculeEntry],
    existing_by_count: dict[int, list[str]],
    min_atoms: int,
    max_atoms: int,
    samples_per_count: int,
) -> None:
    existing_names = {entry.name for entry in entries}
    used_smiles: set[str] = set()

    for atom_count in range(min_atoms, max_atoms + 1):
        variant = 0
        while len(existing_by_count[atom_count]) < samples_per_count:
            result = generate_rdkit_atoms(atom_count, variant)
            variant += 1
            if result is None:
                raise RuntimeError(f"Could not generate fallback molecule for N={atom_count}")

            atoms, smiles = result
            smiles_key = f"{atom_count}:{smiles}"
            if smiles_key in used_smiles:
                continue
            used_smiles.add(smiles_key)

            sample_idx = len(existing_by_count[atom_count]) + 1
            stem = f"{out_subdir}/n{atom_count:03d}_rdkit_{sample_idx:02d}"
            if stem in existing_names:
                continue

            write_xyz(data_large_dir / f"{stem}.xyz", atoms)
            entries.append(MoleculeEntry(stem, atom_count, "xyz", f"rdkit:{smiles}"))
            existing_names.add(stem)
            existing_by_count[atom_count].append(stem)
            print(f"Added {stem} ({atom_count} atoms) from RDKit SMILES {smiles}")


def main() -> None:
    args = parse_args()
    data_large_dir = Path(args.data_large_dir)
    data_large_dir.mkdir(parents=True, exist_ok=True)

    if args.scratch_dir:
        Path(args.scratch_dir).mkdir(parents=True, exist_ok=True)

    molecules_file = data_large_dir / "molecules.txt"
    entries = read_molecules_file(molecules_file)
    existing_by_count = count_existing_chno(
        data_large_dir,
        entries,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
    )
    for atom_count in range(args.min_atoms, args.max_atoms + 1):
        existing_by_count.setdefault(atom_count, [])

    needed_counts = {
        atom_count
        for atom_count, names in existing_by_count.items()
        if len(names) < args.samples_per_count
    }
    print(
        f"Starting with {sum(len(v) for v in existing_by_count.values())} "
        f"CHNO geometries in {args.min_atoms}-{args.max_atoms}; "
        f"{len(needed_counts)} atom counts need more samples."
    )

    add_pubchem_samples(
        data_large_dir=data_large_dir,
        out_subdir=args.out_subdir,
        entries=entries,
        existing_by_count=existing_by_count,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        samples_per_count=args.samples_per_count,
        records_per_formula=args.pubchem_records_per_formula,
        formulas_per_count=args.pubchem_formulas_per_count,
    )

    metadata_paths = [Path(path) for path in args.metadata]
    needed_counts = {
        atom_count
        for atom_count, names in existing_by_count.items()
        if len(names) < args.samples_per_count
    }
    candidates = load_candidate_indices(metadata_paths, needed_counts)
    add_lmdb_samples(
        data_large_dir=data_large_dir,
        out_subdir=args.out_subdir,
        entries=entries,
        existing_by_count=existing_by_count,
        candidates_by_dataset=candidates,
        samples_per_count=args.samples_per_count,
    )

    remaining = {
        atom_count: args.samples_per_count - len(names)
        for atom_count, names in existing_by_count.items()
        if len(names) < args.samples_per_count
    }
    if remaining and args.no_rdkit_fallback:
        raise RuntimeError(f"Missing samples after LMDB extraction: {remaining}")
    if remaining:
        print(f"Filling remaining counts with RDKit conformers: {remaining}")
        add_rdkit_fallback_samples(
            data_large_dir=data_large_dir,
            out_subdir=args.out_subdir,
            entries=entries,
            existing_by_count=existing_by_count,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            samples_per_count=args.samples_per_count,
        )

    write_molecules_file(molecules_file, entries)

    missing = {
        atom_count: len(names)
        for atom_count, names in existing_by_count.items()
        if len(names) < args.samples_per_count
    }
    if missing:
        raise RuntimeError(f"Still missing required CHNO geometries: {missing}")

    total_target = (args.max_atoms - args.min_atoms + 1) * args.samples_per_count
    print(
        f"Done. Have at least {args.samples_per_count} CHNO geometries for each "
        f"atom count {args.min_atoms}-{args.max_atoms} ({total_target} target slots)."
    )


if __name__ == "__main__":
    main()
