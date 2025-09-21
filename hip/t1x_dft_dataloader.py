# pylint: disable=stop-iteration-return

import h5py
import os

REFERENCE_ENERGIES = {
    1: -13.62222753701504,
    6: -1029.4130839658328,
    7: -1484.8710358098756,
    8: -2041.8396277138045,
    9: -2712.8213146878606,
}


def get_molecular_reference_energy(atomic_numbers):
    molecular_reference_energy = 0
    for atomic_number in atomic_numbers:
        molecular_reference_energy += REFERENCE_ENERGIES[atomic_number]

    return molecular_reference_energy


def generator(formula, rxn, grp):
    """Iterates through a h5 group"""

    energies = grp["wB97x_6-31G(d).energy"]
    forces = grp["wB97x_6-31G(d).forces"]
    atomic_numbers = list(grp["atomic_numbers"])
    positions = grp["positions"]
    molecular_reference_energy = get_molecular_reference_energy(atomic_numbers)

    for energy, force, positions in zip(energies, forces, positions):
        d = {
            "rxn": rxn,
            "wB97x_6-31G(d).energy": energy.__float__(),
            "wB97x_6-31G(d).atomization_energy": energy
            - molecular_reference_energy.__float__(),
            "wB97x_6-31G(d).forces": force.tolist(),
            "positions": positions,
            "formula": formula,
            "atomic_numbers": atomic_numbers,
        }

        # Optionally include Hessian if present (eV/Å^2)
        if "wB97x_6-31G(d).hessian" in grp:
            d["wB97x_6-31G(d).hessian"] = grp["wB97x_6-31G(d).hessian"][:]
        if "noiserms" in grp:
            d["noiserms"] = float(grp["noiserms"][()])
        if "positions_noised" in grp:
            d["positions_noised"] = grp["positions_noised"][:]
        # Optional original val index
        if "idx" in grp:
            d["idx"] = int(grp["idx"][()])

        yield d


class Dataloader:
    """
    Can iterate through h5 data set for Transition1x.

    hdf5_file: path to data
    only_final: if True, the iterator will only loop through reactant, product and transition
    state instead of all configurations for each reaction and return them in dictionaries.
    """

    def __init__(self, hdf5_file, datasplit="data", only_final=False):
        self.hdf5_file = hdf5_file
        self.only_final = only_final

        if not os.path.exists(hdf5_file):
            raise FileNotFoundError(f"File {hdf5_file} not found")
        else:
            print(f"{__name__} Dataloader: {hdf5_file} found")

        self.datasplit = datasplit
        if datasplit:
            assert datasplit in [
                "data",
                "train",
                "val",
                "test",
            ], "datasplit must be one of 'all', 'train', 'val' or 'test'"

    def __iter__(self):
        with h5py.File(self.hdf5_file, "r") as f:
            split = f[self.datasplit]

            for formula, grp in split.items():
                for rxn, subgrp in grp.items():
                    reactant = next(generator(formula, rxn, subgrp["reactant"]))
                    product = next(generator(formula, rxn, subgrp["product"]))

                    if self.only_final:
                        transition_state = next(
                            generator(formula, rxn, subgrp["transition_state"])
                        )
                        yield {
                            "rxn": rxn,
                            "reactant": reactant,
                            "product": product,
                            "transition_state": transition_state,
                        }
                    else:
                        yield reactant
                        yield product
                        for molecule in generator(formula, rxn, subgrp):
                            yield molecule


if __name__ == "__main__":
    # Minimal usage example: iterate a few samples from the validation split

    # default_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "transition1x.h5")
    default_path = os.path.abspath(
        os.path.join("data", "t1x_val_reactant_hessian_100.h5")
    )
    if not os.path.exists(default_path):
        default_path = "Transition1x/data/transition1x.h5"

    dl = Dataloader(default_path, datasplit="val", only_final=True)
    for i, sample in enumerate(dl):
        if i >= 3:
            break
        reactant = sample["reactant"]
        coords = reactant["positions"]
        dft_hessian = reactant["wB97x_6-31G(d).hessian"]
        idx = reactant.get("idx", None)
