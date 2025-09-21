import os
import zipfile
import urllib.request
import logging
import h5py
import numpy as np
from tqdm import tqdm

import lmdb
import pickle

import torch
from torch_geometric.data import Data as TGDData

from hip.ff_lmdb import LmdbDataset
from hip.align_ordered_mols import find_rigid_alignment

"""
Preprocessing RGD1 dataset to to save as torch_geometric data format in an LMDB file.
"""

RGD1_URL = "https://figshare.com/ndownloader/articles/21066901/versions/9"

# Convert number to symbol
NUM2ELEMENT_RGD1 = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
max_element_num = max(NUM2ELEMENT_RGD1.keys())

data_dir = "data/rgd1"
zip_filename = "rgd1_raw.zip"


def center_coordinates(coords):
    """Center coordinates using translation and SVD rotation (same as t1x.py)"""
    coords -= coords.mean(0)
    U, _, _ = np.linalg.svd(coords.T)
    if np.linalg.det(U) < 0:
        U[:, -1] *= -1
    coords = coords @ U
    return coords


def download_url(url: str, filename: str) -> str:
    """Download if file does not exist already. Returns path to file."""
    try:
        if os.path.exists(filename):
            print(f"Using downloaded file: {filename}")
            return filename

        print(f"Downloading {url} to {filename}")
        urllib.request.urlretrieve(url, filename)
        print(f"Download completed: {filename}")
        return filename

    except urllib.error.URLError as e:
        if os.path.exists(filename):
            print(f"No internet connection! Using existing file: {filename}")
            return filename
        raise ValueError(f"Could not download {url}: {e}")


def extract_zip_and_list_contents(zip_path: str, extract_dir: str = "rgd1_raw"):
    """Extract zip and list all contents to understand the data structure."""
    # if file RGD1_CHNO.h5 already exist, skip extraction
    if os.path.exists(f"{extract_dir}/RGD1_CHNO.h5"):
        print(f"RGD1_CHNO.h5 already exists in {extract_dir}, skipping extraction")
        return extract_dir

    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)

    print(f"Extracting {zip_path} to {extract_dir}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        # List all files first
        file_list = zip_ref.namelist()
        print(f"Found {len(file_list)} files in archive:")

        for i, filename in enumerate(file_list[:20]):  # Show first 20 files
            print(f"  {i + 1}: {filename}")

        if len(file_list) > 20:
            print(f"  ... and {len(file_list) - 20} more files")

        # Extract all files
        zip_ref.extractall(extract_dir)
        print(f"Extraction completed to {extract_dir}")

    return extract_dir


def load_rgd1_data(raw_data_dir="rgd1_raw", method="all_ts", val_frac=0.1):
    """Load RGD1 data and return first transition state per reaction"""
    print("Loading RGD1 data")

    # Load h5 files
    RXN_ind2geometry = h5py.File(f"{raw_data_dir}/RGD1_CHNO.h5", "r")

    print(f"Loaded {len(RXN_ind2geometry)} reactions")

    # Group reactions by their base ID to identify single vs multiple TS
    # Create list for single transition state reactions
    print("\nFiltering reactions:")
    # idea:
    # The reaction IDs follow a pattern like "MR_XXXXXX_Y" where Y is the transition state index.

    # First, group reactions by their base ID (everything before the last underscore)
    reaction_groups = {}
    for Rind in RXN_ind2geometry.keys():
        base_id = "_".join(
            Rind.split("_")[:-1]
        )  # Remove the last part after underscore
        if base_id not in reaction_groups:
            reaction_groups[base_id] = []
        reaction_groups[base_id].append(Rind)
    print(f"Found {len(reaction_groups)} reaction groups (reactant-product pairs)")

    # Identify reactions and their number of transition states
    single_ts_reaction_ids = []
    multi_ts_reaction_ids = []
    first_ts_per_reaction_ids = []
    all_ts_reaction_ids = []
    num_groups_visited = 0
    val_start_idx = None
    for base_id, reaction_list in reaction_groups.items():
        if len(reaction_list) == 1:
            single_ts_reaction_ids.extend(reaction_list)
        else:
            multi_ts_reaction_ids.extend(reaction_list)
        # Sort and take the first (lowest index)
        first_ts_per_reaction_ids.append(sorted(reaction_list)[0])
        all_ts_reaction_ids.extend(reaction_list)
        if num_groups_visited == round(len(reaction_groups) * (1 - val_frac)):
            val_start_idx = {
                "unique_ts": len(single_ts_reaction_ids),
                "first_ts": len(first_ts_per_reaction_ids),
                "all_ts": len(all_ts_reaction_ids),
            }
        num_groups_visited += 1

    print(
        f"Found {len(single_ts_reaction_ids)} reactions with single transition states"
    )
    print(
        f"Found {len(multi_ts_reaction_ids)} total transition states from reactions with multiple TSs"
    )
    print(
        f"Found {len([base_id for base_id, reaction_list in reaction_groups.items() if len(reaction_list) > 1])} reactions with multiple transition states"
    )

    # unique_ts: only include reactions with unique transition states
    # first_ts: take the first transition state per reaction
    # all_ts: take all transition states per reaction
    # reaction := pair of reactant and product
    if method == "unique_ts":
        reaction_ids_to_process = single_ts_reaction_ids
    elif method == "first_ts":
        reaction_ids_to_process = first_ts_per_reaction_ids
    elif method == "all_ts":
        reaction_ids_to_process = all_ts_reaction_ids
    else:
        raise ValueError(f"Invalid method: {method}.")

    val_start_idx = val_start_idx[method]
    print(
        f"Validation start index: {val_start_idx} out of {len(reaction_ids_to_process)} ({val_start_idx / len(reaction_ids_to_process):.2%})"
    )

    return RXN_ind2geometry, reaction_ids_to_process, val_start_idx


def raw_reaction_data_to_torch_geometric_lmdb(
    RXN_ind2geometry, reaction_ids_to_process, val_start_idx
):
    """Process RGD1 reactions and return processed data.
    Only keep geometries, elements, and smiles.
    """

    # ---- Prepare output LMDB ----
    for split in ["train", "val"]:
        if split == "train":
            ids_this_split = reaction_ids_to_process[:val_start_idx]
        else:
            ids_this_split = reaction_ids_to_process[val_start_idx:]

        output_lmdb_path = f"{data_dir}/rgd1_minimal_{split}.lmdb"
        # cleanup previous files
        if os.path.exists(output_lmdb_path):
            os.remove(output_lmdb_path)
        if os.path.exists(output_lmdb_path.replace(".lmdb", ".lmdb-lock")):
            os.remove(output_lmdb_path.replace(".lmdb", ".lmdb-lock"))
        os.makedirs(data_dir, exist_ok=True)

        smallest_sample_idx = 0
        smallest_sample_natoms = 1000000000

        # map size in megabytes
        # Maximum size database may grow to
        # used to size the memory mapping.
        # If database grows larger than map_size, an exception will be raised and the user must close and reopen Environment.
        # On 64-bit there is no penalty for making this huge (say 1TB)
        map_size = 10 * 1024 * 1024 * 1024  # 10 GB
        out_env = lmdb.open(output_lmdb_path, map_size=map_size, subdir=False)

        print(f"\nProcessing RGD1 reactions for {split} split")

        num_samples_written = 0
        with out_env.begin(write=True) as txn:
            for sample_idx, Rind in tqdm(
                enumerate(ids_this_split),
                desc="Writing LMDB",
                total=len(ids_this_split),
            ):
                try:
                    Rxn = RXN_ind2geometry[Rind]

                    # Get geometries
                    reactant = np.array(Rxn.get("RG"))
                    ts = np.array(Rxn.get("TSG"))
                    product = np.array(Rxn.get("PG"))

                    # parse smiles
                    # multiple molecules will be separated by '.'
                    Rsmiles = Rxn.get("Rsmiles")[()].decode("utf-8")
                    Psmiles = Rxn.get("Psmiles")[()].decode("utf-8")

                    # Get elements and convert to atomic numbers
                    # e.g. [6 6 6 6 6 6 7 7 8 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
                    elements_nums = np.array(Rxn.get("elements"))

                    natoms = len(elements_nums)
                    if natoms < smallest_sample_natoms:
                        smallest_sample_idx = sample_idx
                        smallest_sample_natoms = natoms

                    # count the number of each element
                    # e.g. [ 0 16  0  0  0  0  6  2  1  0]
                    element_counts_vector = np.bincount(
                        elements_nums, minlength=max_element_num + 1
                    )
                    # also represent as a string (so we can use set later)
                    # e.g. H16-C6-N2-O1-F0
                    element_counts_string = ""
                    for _i in NUM2ELEMENT_RGD1.keys():
                        element_counts_string += (
                            f"-{NUM2ELEMENT_RGD1[_i]}{element_counts_vector[_i]}"
                        )
                    element_counts_string = element_counts_string[1:]

                    # Align TS to reactant
                    R_TS, t_TS = find_rigid_alignment(ts, reactant)  # Align TS to R
                    ts = (R_TS.dot(ts.T)).T + t_TS

                    # reactant and product are already aligned in the dataset
                    # R_P, t_P = find_rigid_alignment(product, reactant)  # Align P to R
                    # product = (R_P.dot(product.T)).T + t_P

                    # Center coordinates of TS, apply the same translation to reactant and product
                    translate = ts.mean(0)
                    ts = ts - translate
                    reactant = reactant - translate
                    product = product - translate

                    # Rotate coordinates using SVD
                    U, _, _ = np.linalg.svd(ts.T)
                    if np.linalg.det(U) < 0:
                        U[:, -1] *= -1
                    ts = ts @ U
                    reactant = reactant @ U
                    product = product @ U

                    data = TGDData(
                        z=torch.tensor(elements_nums, dtype=torch.long),
                        natoms=torch.tensor(len(elements_nums), dtype=torch.long),
                        pos_transition=torch.tensor(ts, dtype=torch.float),  # (N, 3)
                        pos_reactant=torch.tensor(
                            reactant, dtype=torch.float
                        ),  # (N, 3)
                        pos_product=torch.tensor(product, dtype=torch.float),  # (N, 3)
                        smiles_reactant=Rsmiles,
                        smiles_product=Psmiles,
                        element_counts_string=element_counts_string,
                        element_counts_vector=torch.tensor(
                            element_counts_vector, dtype=torch.long
                        ),
                    )

                    txn.put(
                        f"{sample_idx}".encode("ascii"),
                        pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL),
                    )
                    num_samples_written += 1

                except Exception as e:
                    logging.warning(f"Error processing reaction {Rind}: {e}")
                    continue

            # end of loop
            txn.put(
                "length".encode("ascii"),
                pickle.dumps(num_samples_written, protocol=pickle.HIGHEST_PROTOCOL),
            )
        out_env.close()
        print(f"Done. {num_samples_written} reactions written to {output_lmdb_path}")
        print(
            f"Smallest sample: {smallest_sample_idx} with {smallest_sample_natoms} atoms"
        )


if __name__ == "__main__":
    # Create data directory
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    # Download RGD1 dataset
    if not os.path.exists(zip_filename):
        print("Downloading RGD1 dataset")
        download_url(RGD1_URL, zip_filename)

    # Extract the dataset
    extract_dir = extract_zip_and_list_contents(zip_filename)

    # Load RGD1 data
    RXN_ind2geometry, reaction_ids_to_process, val_start_idx = load_rgd1_data(
        extract_dir
    )

    # Process and save reactions
    raw_reaction_data_to_torch_geometric_lmdb(
        RXN_ind2geometry, reaction_ids_to_process, val_start_idx
    )

    print("RGD1 data processing completed successfully!")
