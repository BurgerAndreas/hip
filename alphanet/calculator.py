from typing import Optional
import torch
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from alphanet.alphanet_module import PotentialModule
from torch_geometric.data import Data


def onehot_convert(atomic_numbers):
    """
    Convert a list of atomic numbers into an one-hot matrix
    """
    encoder = {
        1: [1, 0, 0, 0, 0],
        6: [0, 1, 0, 0, 0],
        7: [0, 0, 1, 0, 0],
        8: [0, 0, 0, 1, 0],
    }
    onehot = [encoder[i] for i in atomic_numbers]
    return np.array(onehot)


def mols_to_batch(molecules):
    """
    Function used to transfer a list of ase mols into leftnet input entry format
    """
    natoms, batch, charge = [], [], []
    for count, mol in enumerate(molecules):
        atomic_numbers = mol.get_atomic_numbers()
        coordinates = mol.get_positions()
        natoms.append(len(atomic_numbers))
        batch += [count for i in atomic_numbers]
        charge += [i for i in atomic_numbers]
        if count == 0:
            pos = coordinates
            one_hot = onehot_convert(atomic_numbers)
        else:
            pos = np.vstack([pos, coordinates])
            one_hot = np.vstack([one_hot, onehot_convert(atomic_numbers)])
    # compile as data
    data = Data(
        natoms=torch.tensor(np.array(natoms), dtype=torch.int64),
        pos=torch.tensor(pos, dtype=torch.float32).requires_grad_(True),
        one_hot=torch.tensor(one_hot, dtype=torch.int64),
        z=torch.tensor(np.array(charge), dtype=torch.int32),
        batch=torch.tensor(np.array(batch), dtype=torch.int64),
        ae=torch.tensor(np.array(natoms), dtype=torch.int64),
    )
    return data


class AlphaNetCalculator(Calculator):
    """LeftNet ASE Calculator.
    args:
        model: torch.nn.Module finetuned alphanet model
    """

    def __init__(
        self,
        weight: str = "/root/.local/mlff/alphanet/ts1x-tuned_epoch1169.ckpt",
        device: Optional[torch.device] = None,
        **kwargs,
    ):
        """Initializes the calculator.

        Args:
            model (GraphRegressor): The finetuned model to use for predictions.
            brute_force_knn: whether to use a 'brute force' k-nearest neighbors method for graph construction.
                Defaults to None, in which case brute_force is used if a GPU is available (2-6x faster),
                but not on CPU (1.5x faster - 4x slower). For very large systems (>10k atoms),
                brute_force may OOM on GPU, so it is recommended to set to False in that case.
            system_config (SystemConfig): The config defining how an atomic system is featurized.
            device (Optional[torch.device], optional): The device to use for the model.
            **kwargs: Additional keyword arguments for parent Calculator class.
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}  # type: ignore
        # load model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        pm = PotentialModule.load_from_checkpoint(weight, map_location=self.device, strict=False)
        pm.eval()
        self.model = pm.potential

        properties = ["energy", "forces"]
        self.implemented_properties = properties

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        batch = mols_to_batch([atoms])
        batch = batch.to(self.device)  # type: ignore
        self.model = self.model.to(self.device)  # type: ignore
        self.atoms = atoms
        self.results = {}
        # predict energy and force (by autograd)
        out = self.model.forward(batch)
        self.energy = out[0][0]
        if "energy" in self.implemented_properties:
            self.results["energy"] = float(out[0][0].detach().cpu().item())

        if "forces" in self.implemented_properties:
            self.results["forces"] = out[1].detach().cpu().numpy()
