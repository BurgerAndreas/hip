import h5py
import numpy as np
import os

path_to_h5file = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "relaxations",
    "ani1x",
    "ani1x.h5",
)
print("path_to_h5file:", path_to_h5file)


def download_ani1x(dest_path=path_to_h5file):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    # wget https://springernature.figshare.com/ndownloader/files/18112775 -O data/relaxations/ani1x/ani1x.h5
    os.system(
        f"curl -L -O 'https://springernature.figshare.com/ndownloader/files/18112775' -H 'User-Agent: Mozilla/5.0' -H 'Referer: https://springernature.figshare.com/' -o {dest_path}"
    )
    # curl -L -O 'https://api.figshare.com/v2/file/18112775/download'


def iter_data_buckets(h5filename, keys=["wb97x_dz.energy"]):
    """Iterate over buckets of data in ANI HDF5 file.
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard("atomic_numbers")
    keys.discard("coordinates")
    with h5py.File(h5filename, "r") as f:
        for grp in f.values():
            Nc = grp["coordinates"].shape[0]
            mask = np.ones(Nc, dtype=bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d["atomic_numbers"] = grp["atomic_numbers"][()]
            d["coordinates"] = grp["coordinates"][()][mask]
            yield d


if __name__ == "__main__":
    # List of keys to point to requested data
    data_keys = [
        "wb97x_dz.energy",
        "wb97x_dz.forces",
    ]  # Original ANI-1x data (https://doi.org/10.1063/1.5023802)
    # data_keys = ['wb97x_tz.energy','wb97x_tz.forces'] # CHNO portion of the data set used in AIM-Net (https://doi.org/10.1126/sciadv.aav6490)
    # data_keys = ['ccsd(t)_cbs.energy'] # The coupled cluster ANI-1ccx data set (https://doi.org/10.1038/s41467-019-10827-4)
    # data_keys = ['wb97x_dz.dipoles'] # A subset of this data was used for training the ACA charge model (https://doi.org/10.1021/acs.jpclett.8b01939)

    # if not os.path.exists(path_to_h5file):
    #     download_ani1x()
    # else:
    #     print("Using existing ANI-1x data file:", path_to_h5file)

    # Example for extracting DFT/DZ energies and forces
    for idx, data in enumerate(iter_data_buckets(path_to_h5file, keys=data_keys)):
        X = data["coordinates"]
        Z = data["atomic_numbers"]
        E = data["wb97x_dz.energy"][idx]
        F = data["wb97x_dz.forces"][idx]
        print(E, Z)
        break
