import torch
import numpy as np
from nets.scatter_utils import scatter_mean

GLOBAL_ATOM_NUMBERS = torch.tensor([1, 6, 7, 8])
GLOBAL_ATOM_SYMBOLS = np.array(["H", "C", "N", "O"])
Z_TO_ATOM_SYMBOL = {
    1: "H",
    6: "C",
    7: "N",
    8: "O",
}


def onehot_convert(atomic_numbers, device):
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
    return torch.tensor(onehot, dtype=torch.int64, device=device)


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


# def compute_extra_props(batch, pos_require_grad=True):
#     """Adds device, z, and removes mean batch"""
#     device = batch.pos.device
#     if hasattr(batch, "one_hot"):
#         # this is only for the HORM dataset
#         # it uses a weird convention
#         # atom types are encoded as one-hot vectors of shape (N, 5)
#         # where the fifth is unused, likely a padding or None class
#         # corresponds to H, C, N, O, None
#         indices = batch.one_hot.long().argmax(dim=1)
#         batch.z = GLOBAL_ATOM_NUMBERS.to(device)[indices.to(device)]
#     elif hasattr(batch, "z"):
#         batch.z = batch.z.to(device)
#         batch.one_hot = onehot_convert(batch.z.tolist(), device)
#     else:
#         raise ValueError("batch has no one_hot or z attribute")
#     batch.pos = remove_mean_batch(batch.pos, batch.batch)
#     # # atomization energy. shape used by equiformerv2
#     # if not hasattr(batch, "ae"):
#     #     if hasattr(batch, "energy"):
#     #         batch.ae = batch.energy.clone()
#     if pos_require_grad:
#         batch.pos.requires_grad_(True)
#     return batch
