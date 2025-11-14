#!/usr/bin/env python3
"""
Script to find and download molecules from PubChem and ZINC databases
with atom counts between 100 and 240 atoms.
"""

import requests
import time
from pathlib import Path
import ase.io
from typing import List, Tuple, Optional


def get_atom_count_from_sdf(sdf_content: bytes) -> Optional[int]:
    """Count atoms from SDF file content."""
    content = sdf_content.decode("utf-8", errors="ignore")
    lines = content.split("\n")

    # SDF format: line 4 has counts line: "aaabbblllfffcccsss..."
    # First 3 digits are number of atoms
    if len(lines) >= 4:
        counts_line = lines[3]
        if len(counts_line) >= 3:
            atom_count_str = counts_line[:3].strip()
            if atom_count_str.isdigit():
                return int(atom_count_str)
    return None


def download_pubchem_sdf(cid: int, output_path: Path) -> Tuple[bool, Optional[int]]:
    """Download SDF file from PubChem by CID. Returns (success, atom_count)."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"

    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        atom_count = get_atom_count_from_sdf(response.content)
        with open(output_path, "wb") as f:
            f.write(response.content)
        return True, atom_count
    return False, None


def get_pubchem_compound_info(cid: int) -> Optional[dict]:
    """Get compound information from PubChem API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/MolecularWeight,HeavyAtomCount/JSON"

    response = requests.get(url, timeout=30)
    if response.status_code == 200:
        data = response.json()
        if "PropertyTable" in data and "Properties" in data["PropertyTable"]:
            props = data["PropertyTable"]["Properties"][0]
            mw = props.get("MolecularWeight")
            heavy_atoms = props.get("HeavyAtomCount")
            # Estimate total atoms (heavy atoms + hydrogens)
            # This is approximate - we'll verify by downloading SDF
            estimated_atoms = heavy_atoms * 2 if heavy_atoms else None  # Rough estimate
            return {
                "molecular_weight": float(mw) if mw else None,
                "heavy_atom_count": heavy_atoms,
                "estimated_atom_count": estimated_atoms,
            }
    return None


def search_pubchem_by_molecular_weight(
    min_mw: float, max_mw: float, max_results: int = 100
) -> List[int]:
    """
    Search PubChem by molecular weight range.
    Note: This uses a workaround since PubChem doesn't directly support MW range queries.
    We'll search for specific CIDs and check their properties.
    """
    # Strategy: Search for compounds with molecular weight in range
    # We'll use a property search query
    # Note: PubChem PUG REST doesn't directly support range queries easily
    # So we'll try a different approach: search by formula or use known CIDs

    # Alternative: Search by molecular formula patterns that might give us molecules in this range
    # Or we can try searching for specific compound types

    # For now, let's try searching some known compound IDs that might be in this range
    # We'll start from some reasonable CID ranges
    found_cids = []

    # Try searching in a CID range that might contain molecules of interest
    # PubChem has millions of compounds, so we'll sample strategically
    start_cid = 1000000  # Start from a reasonable range
    end_cid = start_cid + 10000  # Check 10000 compounds

    print(f"Searching PubChem CIDs {start_cid} to {end_cid}...")

    for cid in range(start_cid, end_cid):
        if len(found_cids) >= max_results:
            break

        info = get_pubchem_compound_info(cid)
        if info and info.get("atom_count"):
            atom_count = info["atom_count"]
            mw = info.get("molecular_weight", 0)

            if min_mw <= mw <= max_mw and 100 <= atom_count <= 240:
                found_cids.append(cid)
                print(f"Found CID {cid}: {atom_count} atoms, MW={mw:.2f}")

        # Rate limiting
        if cid % 100 == 0:
            time.sleep(0.1)

    return found_cids


def search_pubchem_by_atom_count(
    min_atoms: int, max_atoms: int, max_results: int = 10
) -> List[Tuple[int, int]]:
    """
    Search PubChem for compounds with atom count in range.
    Returns list of (CID, atom_count) tuples.

    Strategy: Sample from CID ranges known to contain drug-like molecules.
    We'll check compounds in ranges where medium-sized organic molecules are common.
    """
    found_compounds = []

    # Try some known CIDs first that might be in range
    known_cids_to_try = [
        14009497,  # Hectane (C100H202) - might be too large, but worth checking
        123591,  # Buckminsterfullerene (C60) - 60 atoms, too small but nearby
    ]

    print("Checking known CIDs first...")
    for cid in known_cids_to_try:
        if len(found_compounds) >= max_results:
            break
        info = get_pubchem_compound_info(cid)
        if info:
            heavy_atoms = info.get("heavy_atom_count")
            estimated = info.get("estimated_atom_count")
            mw = info.get("molecular_weight", 0)
            # Use estimated count for filtering, but we'll verify with actual download
            if (
                estimated and min_atoms <= estimated <= max_atoms * 1.5
            ):  # Wider range for estimation
                found_compounds.append((cid, estimated))
                print(
                    f"  ✓ Found CID {cid}: ~{estimated} atoms (est), {heavy_atoms} heavy, MW={mw:.2f}"
                )
        time.sleep(0.1)

    if len(found_compounds) >= max_results:
        return found_compounds

    # Known CID ranges that often contain medium-sized organic molecules
    # Sample more densely - try every 10th instead of every 100th
    search_ranges = [
        (100000, 101000, 10),  # Early range, sample every 10th
        (500000, 501000, 10),
        (1000000, 1001000, 10),  # Sample more densely
        (2000000, 2001000, 10),
        (5000000, 5001000, 10),
        (10000000, 10001000, 10),
        (14000000, 14001000, 10),  # Near Hectane range
    ]

    for start_cid, end_cid, step in search_ranges:
        if len(found_compounds) >= max_results:
            break

        print(f"Searching PubChem CIDs {start_cid} to {end_cid} (step={step})...")

        checked = 0
        for cid in range(start_cid, end_cid, step):
            if len(found_compounds) >= max_results:
                break

            checked += 1
            if checked % 10 == 0:
                print(
                    f"  Checked {checked} compounds, found {len(found_compounds)} so far..."
                )

            info = get_pubchem_compound_info(cid)
            if info and info.get("atom_count"):
                atom_count = info["atom_count"]
                if min_atoms <= atom_count <= max_atoms:
                    mw = info.get("molecular_weight", 0)
                    found_compounds.append((cid, atom_count))
                    print(f"  ✓ Found CID {cid}: {atom_count} atoms, MW={mw:.2f}")

            time.sleep(0.1)  # Rate limiting

    return found_compounds


def verify_molecule_atom_count(file_path: Path) -> Optional[int]:
    """Load molecule and verify actual atom count."""
    atoms = ase.io.read(str(file_path))
    return len(atoms)


def add_molecule_to_list(
    molecules_file: Path, name: str, atom_count: int, file_format: str, source: str
):
    """Add molecule entry to molecules.txt file."""
    # Read existing entries
    existing_names = set()
    lines = []

    if molecules_file.exists():
        with open(molecules_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                if line.strip():
                    parts = line.strip().split("\t")
                    if parts:
                        existing_names.add(parts[0])

    # Check if already exists
    if name in existing_names:
        print(f"Molecule {name} already in molecules.txt, skipping")
        return

    # Add new entry
    new_line = f"{name}\t{atom_count}\t{file_format}\t{source}\n"

    # Write back with new entry (maintain sorted order by atom count)
    entries = []
    for line in lines[1:]:
        if line.strip():
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                entries.append((int(parts[1]), line))

    # Add new entry
    entries.append((atom_count, new_line))

    # Sort by atom count
    entries.sort(key=lambda x: x[0])

    # Write back
    with open(molecules_file, "w") as f:
        f.write("Name\tAtoms\tFormat\tSource\n")
        for _, line in entries:
            f.write(line)


def download_from_cid_list(
    cids: List[int],
    data_large_dir: Path,
    molecules_file: Path,
    min_atoms: int,
    max_atoms: int,
) -> int:
    """Download molecules from a list of PubChem CIDs."""
    downloaded = 0

    for cid in cids:
        name = f"pubchem_{cid}"
        sdf_path = data_large_dir / f"{name}.sdf"

        if sdf_path.exists():
            print(f"Skipping {name} - file already exists")
            # Check if it's in molecules.txt
            actual_count = verify_molecule_atom_count(sdf_path)
            if actual_count and min_atoms <= actual_count <= max_atoms:
                add_molecule_to_list(
                    molecules_file, name, actual_count, "sdf", "pubchem_mols"
                )
                downloaded += 1
            continue

        print(f"\nDownloading CID {cid}...")
        success, atom_count_from_sdf = download_pubchem_sdf(cid, sdf_path)

        if not success:
            print(f"✗ Failed to download CID {cid}")
            continue

        # Verify atom count from downloaded file
        actual_count = verify_molecule_atom_count(sdf_path)
        if not actual_count:
            print(f"✗ Could not verify atom count for CID {cid}")
            sdf_path.unlink()
            continue

        if min_atoms <= actual_count <= max_atoms:
            print(f"✓ Downloaded {name}: {actual_count} atoms")
            add_molecule_to_list(
                molecules_file, name, actual_count, "sdf", "pubchem_mols"
            )
            downloaded += 1
        else:
            print(
                f"✗ CID {cid} has {actual_count} atoms (outside range {min_atoms}-{max_atoms})"
            )
            sdf_path.unlink()  # Remove file if wrong size

        time.sleep(0.5)  # Rate limiting

    return downloaded


def main():
    """Main function to find and download molecules."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Find and download molecules with 100-240 atoms from PubChem/ZINC"
    )
    parser.add_argument(
        "--cids", type=str, help="Comma-separated list of PubChem CIDs to download"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2,
        help="Target number of molecules to find (default: 2)",
    )
    parser.add_argument(
        "--min-atoms", type=int, default=100, help="Minimum atom count (default: 100)"
    )
    parser.add_argument(
        "--max-atoms", type=int, default=240, help="Maximum atom count (default: 240)"
    )
    parser.add_argument(
        "--search", action="store_true", help="Search PubChem automatically (slower)"
    )

    args = parser.parse_args()

    data_large_dir = Path("data/large")
    data_large_dir.mkdir(parents=True, exist_ok=True)
    molecules_file = data_large_dir / "molecules.txt"

    min_atoms = args.min_atoms
    max_atoms = args.max_atoms
    target_count = args.target

    print(f"Searching for molecules with {min_atoms}-{max_atoms} atoms...")
    print(f"Target: {target_count} molecules\n")

    downloaded = 0

    # If CIDs provided, use those
    if args.cids:
        print("=" * 60)
        print("Downloading from provided CIDs...")
        print("=" * 60)
        cid_list = [int(cid.strip()) for cid in args.cids.split(",")]
        downloaded = download_from_cid_list(
            cid_list, data_large_dir, molecules_file, min_atoms, max_atoms
        )

    # Otherwise, search PubChem
    elif args.search:
        print("=" * 60)
        print("Searching PubChem...")
        print("=" * 60)
        print(
            "Note: This may take a while. Consider using --cids with specific CIDs instead."
        )

        found_compounds = search_pubchem_by_atom_count(
            min_atoms, max_atoms, max_results=target_count * 5
        )

        downloaded = download_from_cid_list(
            [cid for cid, _ in found_compounds],
            data_large_dir,
            molecules_file,
            min_atoms,
            max_atoms,
        )
    else:
        print("=" * 60)
        print("No search method specified.")
        print("=" * 60)
        print("Options:")
        print("  1. Use --cids to provide specific PubChem CIDs")
        print("     Example: python find_molecules_100_240.py --cids 123456,789012")
        print("  2. Use --search to automatically search PubChem (slower)")
        print("     Example: python find_molecules_100_240.py --search")
        print("\nTo find CIDs manually:")
        print("  - Visit https://pubchem.ncbi.nlm.nih.gov/")
        print("  - Use Advanced Search with atom count filters")
        print("  - Export the CIDs and use with --cids option")
        return

    if downloaded < target_count:
        print(f"\nOnly found {downloaded} molecules (target: {target_count}).")
        print("\nTips:")
        print("  - Try searching PubChem manually at https://pubchem.ncbi.nlm.nih.gov/")
        print("  - Use Advanced Search with atom count: 100-240")
        print("  - Export CIDs and use: --cids <list_of_cids>")
        print("  - ZINC database: https://zinc.docking.org/")

    print(f"\n✓ Done! Found and added {downloaded} molecules.")


if __name__ == "__main__":
    main()
