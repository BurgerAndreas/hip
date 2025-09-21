import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def collect_raw_reaction_data(
    raw_data_dir="rgd1_raw", method="first_ts", print_info=False
):
    assert method in ["unique_ts", "first_ts", "all_ts"], f"Invalid method: {method}."
    # unique_ts: only include reactions with unique transition states
    # first_ts: take the first transition state per reaction
    # all_ts: take all transition states per reaction
    # reaction := pair of reactant and product

    # convert number to symbol
    NUM2ELEMENT_RGD1 = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

    # load in h5 files
    # Geometries of R, P, TS
    RXN_ind2geometry = h5py.File(f"{raw_data_dir}/RGD1_CHNO.h5", "r")
    # Geometries of R, P
    RP_molidx2geometry = h5py.File(f"{raw_data_dir}/RGD1_RPs.h5", "r")

    #####################################################################################

    # load CSV file with atom-mapped SMILES and energies
    csv_data = pd.read_csv(f"{raw_data_dir}/RGD1CHNO_AMsmiles.csv")
    # Create mapping from reaction ID to CSV data for faster lookup
    csv_data_dict = csv_data.set_index("reaction").to_dict("index")

    if print_info:
        print(f"Created CSV lookup dictionary with {len(csv_data_dict)} entries")

        # load in molecule dictionary
        lines = open(
            f"{raw_data_dir}/RandP_smiles.txt", "r", encoding="utf-8"
        ).readlines()
        RP_smiles2molidx = dict()
        for lc, line in enumerate(lines):
            if lc == 0:
                continue
            RP_smiles2molidx[line.split()[0]] = line.split()[1]

        # Print what we loaded
        print(f"\nLoaded CSV data with {len(csv_data)} reactions")
        print(f"CSV columns: {list(csv_data.columns)}")

        print(f"\nLoaded {len(RXN_ind2geometry)} reactions (RXN_ind2geometry)")
        print("First 3 RXN_ind2geometry:")
        for i, (k, v) in enumerate(RXN_ind2geometry.items()):
            print(k, v)
            if i > 2:
                break
        print(f"keys of first sample: {list(RXN_ind2geometry['MR_100001_2'].keys())}")

        print(
            f"\nLoaded {len(RP_molidx2geometry)} reactants and products geometries (RP_molidx2geometry)"
        )
        print("First 3 RP_molidx2geometry:")
        for i, (k, v) in enumerate(RP_molidx2geometry.items()):
            print(k, v)
            if i > 2:
                break
        print(f"keys of first sample: {list(RP_molidx2geometry['mol2'].keys())}")

        print(
            f"\nLoaded {len(RP_smiles2molidx)} reactants and products smiles (RP_smiles2molidx)"
        )
        print("First 3 RP_smiles2molidx:")
        for i, (k, v) in enumerate(RP_smiles2molidx.items()):
            print(k, v)
            if i > 2:
                break

    #####################################################################################

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
    for base_id, reaction_list in reaction_groups.items():
        if len(reaction_list) == 1:
            single_ts_reaction_ids.extend(reaction_list)
        else:
            multi_ts_reaction_ids.extend(reaction_list)
        # Sort and take the first (lowest index)
        first_ts_per_reaction_ids.append(sorted(reaction_list)[0])
        all_ts_reaction_ids.extend(reaction_list)

    print(
        f"Found {len(single_ts_reaction_ids)} reactions with single transition states"
    )
    print(
        f"Found {len(multi_ts_reaction_ids)} total transition states from reactions with multiple TSs"
    )
    print(
        f"Found {len([base_id for base_id, reaction_list in reaction_groups.items() if len(reaction_list) > 1])} reactions with multiple transition states"
    )

    if method == "unique_ts":
        reaction_ids_to_process = single_ts_reaction_ids
    elif method == "first_ts":
        reaction_ids_to_process = first_ts_per_reaction_ids
    elif method == "all_ts":
        reaction_ids_to_process = all_ts_reaction_ids
    else:
        raise ValueError(f"Invalid method: {method}.")

    processed_reactions = []
    failed_reactions = []
    print(f"Processing {len(reaction_ids_to_process)} reactions")
    for Rind in tqdm(reaction_ids_to_process, desc=f"Processing {method} TS reactions"):
        Rxn = RXN_ind2geometry[Rind]

        try:
            # Create reaction data dictionary
            reaction_data = {
                "reaction_id": Rind,
                # Energies
                "R_E": np.array(Rxn.get("R_E")),
                "P_E": np.array(Rxn.get("P_E")),
                "TS_E": np.array(Rxn.get("TS_E")),
                # Enthalpies
                "R_H": np.array(Rxn.get("R_H")),
                "P_H": np.array(Rxn.get("P_H")),
                "TS_H": np.array(Rxn.get("TS_H")),
                # Gibbs free energies
                "R_F": np.array(Rxn.get("R_F")),
                "P_F": np.array(Rxn.get("P_F")),
                "TS_F": np.array(Rxn.get("TS_F")),
                # SMILES
                "Rsmiles": Rxn.get("Rsmiles")[()].decode("utf-8"),
                "Psmiles": Rxn.get("Psmiles")[()].decode("utf-8"),
                # Elements and geometries
                "elements": [
                    NUM2ELEMENT_RGD1[Ei] for Ei in np.array(Rxn.get("elements"))
                ],
                "TS_geometry": np.array(Rxn.get("TSG")),
                "R_geometry": np.array(Rxn.get("RG")),
                "P_geometry": np.array(Rxn.get("PG")),
                # Individual molecule data
                "reactant_molecules": [],
                "product_molecules": [],
            }

            # Add CSV data if available
            if Rind in csv_data_dict:
                csv_row = csv_data_dict[Rind]
                reaction_data.update(
                    {
                        "atom_mapped_reactant_smiles": csv_row["reactant"],
                        "atom_mapped_product_smiles": csv_row["product"],
                        "activation_energy_forward": csv_row["DE_F"],
                        "activation_energy_backward": csv_row["DE_B"],
                        "gibbs_energy_forward": csv_row["DG_F"],
                        "gibbs_energy_backward": csv_row["DG_B"],
                        "enthalpy_change": csv_row["DH"],
                    }
                )

            # Load individual reactant/product molecule data
            Rsmiles = reaction_data["Rsmiles"]
            Psmiles = reaction_data["Psmiles"]

            # Process reactant molecules
            for r_smiles in Rsmiles.split("."):
                if r_smiles in RP_smiles2molidx:
                    mol_id = RP_smiles2molidx[r_smiles]
                    if mol_id in RP_molidx2geometry:
                        molecule = RP_molidx2geometry[mol_id]
                        mol_data = {
                            "smiles": r_smiles,
                            "mol_id": mol_id,
                            "DFT_geometry": np.array(molecule.get("DFTG")),
                            "DFT_SPE": np.array(molecule.get("DFT_SPE")),
                            "xTB_geometry": np.array(molecule.get("xTBG")),
                            "xTB_SPE": np.array(molecule.get("xTB_SPE")),
                            "elements": np.array(molecule.get("elements")),
                        }
                        reaction_data["reactant_molecules"].append(mol_data)
                    else:
                        # some molecular data here is missing
                        # insert dummy data?
                        pass

            # Process product molecules
            for p_smiles in Psmiles.split("."):
                if p_smiles in RP_smiles2molidx:
                    mol_id = RP_smiles2molidx[p_smiles]
                    if mol_id in RP_molidx2geometry:
                        molecule = RP_molidx2geometry[mol_id]
                        mol_data = {
                            "smiles": p_smiles,
                            "mol_id": mol_id,
                            "DFT_geometry": np.array(molecule.get("DFTG")),
                            "DFT_SPE": np.array(molecule.get("DFT_SPE")),
                            "xTB_geometry": np.array(molecule.get("xTBG")),
                            "xTB_SPE": np.array(molecule.get("xTB_SPE")),
                            "elements": np.array(molecule.get("elements")),
                        }
                        reaction_data["product_molecules"].append(mol_data)
                    else:
                        # some molecular data here is missing
                        # insert dummy data?
                        pass

            processed_reactions.append(reaction_data)

        except Exception as e:
            # print(f"Error processing reaction {Rind}: {e}")
            failed_reactions.append(Rind)
            continue

    print(
        f"\nSuccessfully processed {len(processed_reactions)} single transition state reactions"
    )
    print(
        f"Failed to process {len(failed_reactions)} / {len(reaction_ids_to_process)} reactions"
    )
    return processed_reactions


if __name__ == "__main__":
    processed_reactions = collect_raw_reaction_data(method="all_ts")

    # Display sample data structure
    if processed_reactions:
        sample = processed_reactions[0]
        print(f"\nSample reaction data structure for {sample['reaction_id']}:")
        print(f"  - Reactant SMILES: {sample['Rsmiles']}")
        print(f"  - Product SMILES: {sample['Psmiles']}")

        # Show CSV data if available
        if "atom_mapped_reactant_smiles" in sample:
            print(
                f"  - Atom-mapped reactant SMILES: {sample['atom_mapped_reactant_smiles'][:80]}"
            )
            print(
                f"  - Atom-mapped product SMILES: {sample['atom_mapped_product_smiles'][:80]}"
            )
            print(
                f"  - Activation energy forward: {sample['activation_energy_forward']:.4f}"
            )
            print(
                f"  - Activation energy backward: {sample['activation_energy_backward']:.4f}"
            )
            print(f"  - Enthalpy change: {sample['enthalpy_change']:.4f}")

        print(f"  - Number of atoms: {len(sample['elements'])}")
        print(
            f"  - Activation energy (TS-R): {float(sample['TS_E'] - sample['R_E']):.4f}"
        )
        print(f"  - Reaction energy (P-R): {float(sample['P_E'] - sample['R_E']):.4f}")
        print(f"  - Number of reactant molecules: {len(sample['reactant_molecules'])}")
        print(f"  - Number of product molecules: {len(sample['product_molecules'])}")
