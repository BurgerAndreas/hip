import logging
import time
import math
import numpy as np
import torch

try:
    from e3nn import o3
except ImportError:
    pass

import einops
import plotly.express as px


def check_symmetry(hessian, N, nsamples=100):
    hessian = hessian.view(N * 3, N * 3)
    # test symmetry
    errors_abs = []
    errors_rel = []
    for _ in range(nsamples):
        i = torch.randint(0, hessian.shape[0], (1,)).item()
        j = torch.randint(0, hessian.shape[1], (1,)).item()
        hij = hessian[i, j]
        hji = hessian[j, i]
        abs_error = torch.abs(hij - hji).item()
        rel_error = abs_error / (torch.abs(hij).item() + 1e-8)
        errors_abs.append(abs_error)
        errors_rel.append(rel_error)
    print(
        f"Hessian symmetry check - Abs error: mean={sum(errors_abs) / len(errors_abs):.2e}, max={max(errors_abs):.2e}"
    )
    print(
        f"Hessian symmetry check - Rel error: mean={sum(errors_rel) / len(errors_rel):.2e}, max={max(errors_rel):.2e}"
    )


def add_extra_props_for_hessian(data, offset_indices=False):
    # add extra props for convience
    nedges = data.nedges_hessian
    B = data.batch.max().item() + 1
    # vectorized pointer build
    _nedges = nedges.to(device=data.batch.device, dtype=torch.long)
    _sizes = (_nedges * 3) ** 2
    # indices are computed for each sample individually
    # so we need to offset the indices by the number of entries in the previous samples in the batch
    if offset_indices:
        if hasattr(data, "offsetdone") and (data.offsetdone is True):
            return data
        data.offsetdone = True
        # Precompute exclusive cumulative offsets once (O(B))
        natoms = data.natoms.to(device=data.batch.device, dtype=torch.long)
        hess_entries_per_sample = (natoms * 3) ** 2
        node_entries_per_sample = natoms * 9
        cumsum_hess = torch.cumsum(hess_entries_per_sample, dim=0)
        cumsum_node = torch.cumsum(node_entries_per_sample, dim=0)
        hess_offsets = torch.zeros_like(cumsum_hess)
        node_offsets = torch.zeros_like(cumsum_node)
        if B > 1:
            data.ptr_1d_hessian = torch.empty(
                B + 1, device=data.batch.device, dtype=torch.long
            )
            data.ptr_1d_hessian[0] = 0
            if B > 0:
                data.ptr_1d_hessian[1:] = torch.cumsum(_sizes, dim=0)
            hess_offsets[1:] = cumsum_hess[:-1]
            node_offsets[1:] = cumsum_node[:-1]
        # Parallelize offsets across all elements using repeat_interleave per-sample lengths
        edge_counts = (_nedges * 9).to(dtype=torch.long)
        node_counts = (natoms * 9).to(dtype=torch.long)
        # Build full-length offset vectors
        if edge_counts.sum().item() > 0:
            full_edge_hess_offsets = torch.repeat_interleave(hess_offsets, edge_counts)
            data.message_idx_ij += full_edge_hess_offsets
            data.message_idx_ji += full_edge_hess_offsets
        if node_counts.sum().item() > 0:
            full_node_hess_offsets = torch.repeat_interleave(hess_offsets, node_counts)
            full_node_node_offsets = torch.repeat_interleave(node_offsets, node_counts)
            data.diag_ij += full_node_hess_offsets
            data.diag_ji += full_node_hess_offsets
            data.node_transpose_idx += full_node_node_offsets

    return data


##############################################################################################################
# The following functions are all the same, but with different implementations
# They all build the Hessian matrix from the edge features


def _blockdiagonal_N_3_N_3_loop(N, edge_index, messages):
    device = messages.device
    dtype = messages.dtype
    hessian = torch.zeros((N, 3, N, 3), device=device, dtype=dtype)
    for ij in range(edge_index.shape[1]):
        i, j = edge_index[0, ij], edge_index[1, ij]
        hessian[i, :, j, :] += messages[ij]
        hessian[j, :, i, :] += messages[ij].T
    return hessian



# support function that can be moved to dataloader
def _get_flat_indexadd_message_indices(N, edge_index):
    # Vectorized construction of 1D indices for i->j and j->i contributions
    # edge_index: (2, E)
    device = edge_index.device
    E = edge_index.shape[1]
    i = edge_index[0].to(dtype=torch.long)
    j = edge_index[1].to(dtype=torch.long)
    # Prepare coordinate offsets (3x3 per edge)
    ci = torch.arange(3, device=device, dtype=torch.long).view(1, 3, 1)
    cj = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
    i = i.view(E, 1, 1)
    j = j.view(E, 1, 1)
    N3 = N * 3
    # i -> j block indices
    idx_ij = ((i * 3 + ci) * N3 + (j * 3 + cj)).reshape(-1)
    # j -> i block indices (transpose)
    idx_ji = ((j * 3 + ci) * N3 + (i * 3 + cj)).reshape(-1)
    return idx_ij, idx_ji


def _flat_indexadd(edge_index, messages, data):
    # do the same thing in 1d, but indexing messageflat without storing it in values
    device = messages.device
    dtype = messages.dtype
    E = edge_index.shape[1]
    messageflat = messages.reshape(-1)
    total_entries = 0
    for _N in data.natoms:
        total_entries += _N * 3 * _N * 3
    hessian1d = torch.zeros(total_entries, device=device, dtype=dtype)
    indices_ij = data.message_idx_ij  # (E*3*3) -> (N*3*N*3)
    indices_ji = data.message_idx_ji  # (E*3*3) -> (N*3*N*3)
    # Reshape messageflat to (E, 3, 3) and transpose each 3x3 matrix
    messages_3x3 = messageflat.view(E, 3, 3)
    messages_3x3_T = messages_3x3.transpose(-2, -1)  # Transpose last two dimensions
    messageflat_transposed = messages_3x3_T.reshape(-1)  # Flatten back
    # Add both contributions
    hessian1d.index_add_(0, indices_ij, messageflat)  # i->j direct
    hessian1d.index_add_(0, indices_ji, messageflat_transposed)  # j->i transposed
    return hessian1d


##############################################################################################################
# The following functions are all the same, but with different implementations
# They all add the node features to the diagonal


def _add_node_diagonal_2d_loop(hessian, l012_node_features, N):
    """Add node embeddings to diagonal using 2D indexing with loops"""
    # hessian: (N*3,N*3)
    # l012_node_features: (N,3,3)
    for ii in range(N):
        hessian[ii * 3 : (ii + 1) * 3, ii * 3 : (ii + 1) * 3] += l012_node_features[ii]
        # Add transpose for symmetry
        hessian[ii * 3 : (ii + 1) * 3, ii * 3 : (ii + 1) * 3] += l012_node_features[
            ii
        ].T
    return hessian

# support function that can be moved to dataloader
def _get_node_diagonal_1d_indexadd_indices(N, device):
    # Vectorized build of diagonal indices for direct and transpose contributions
    # Shapes: (N, 3, 3) -> flatten to (N*9)
    ii = torch.arange(N, device=device, dtype=torch.long)
    ci = torch.arange(3, device=device, dtype=torch.long)
    cj = torch.arange(3, device=device, dtype=torch.long)
    Ii, Ci, Cj = torch.meshgrid(ii, ci, cj, indexing="ij")
    # 1D index for diagonal element (ii*3 + coord_i, ii*3 + coord_j)
    diag_idx = (Ii * 3 + Ci) * (N * 3) + (Ii * 3 + Cj)
    diag_idx = diag_idx.reshape(-1)
    # Transpose indices for node features: swap coord_i and coord_j
    node_transpose_idx = Ii * 9 + Cj * 3 + Ci
    node_transpose_idx = node_transpose_idx.reshape(-1)
    # Both diag arrays are identical by construction
    return diag_idx, diag_idx.clone(), node_transpose_idx


def _add_node_diagonal_1d_indexadd(hessianflat, l012_node_features, data):
    """Add node embeddings to diagonal using 1D indexing with index_add"""
    # diag_ij, diag_ji, node_transpose_idx = _get_node_diagonal_1d_indexadd_indices(N, device)
    diag_ij = data.diag_ij  # (N*3*3) -> (N*3*N*3)
    diag_ji = data.diag_ji  # (N*3*3) -> (N*3*N*3)
    node_transpose_idx = data.node_transpose_idx  # (N*3*3) -> (N*3*3)
    # Flatten node features for direct indexing
    l012_node_features_flat = l012_node_features.reshape(-1)  # (N*3*3)
    # Use two index_add calls: one for direct, one for transpose
    hessianflat.index_add_(0, diag_ij, l012_node_features_flat)
    hessianflat.index_add_(0, diag_ji, l012_node_features_flat[node_transpose_idx])
    return hessianflat

##############################################################################################################


def predict_hessian_1d_fast(edge_index, data, l012_edge_features, l012_node_features):
    """
    Predict the Hessian matrix in a 1D format.
    Fast because it uses index_add.
    Total entries: B*N*3*N*3
    Return shape: (B*N*3*N*3)
    """
    # fast
    hessian = _flat_indexadd(edge_index, l012_edge_features, data)
    hessian = _add_node_diagonal_1d_indexadd(hessian, l012_node_features, data)
    return hessian


def predict_hessian_blockdiagonal_robust(
    edge_index, data, l012_edge_features, l012_node_features
):
    """
    Predict the Hessian matrix in a block diagonal format.
    Robust because it uses an explicit loop over messages and features.
    Total entries: B*N*3*B*N*3 (instead of B*N*3*N*3)
    Return shape: (B*N*3, B*N*3)
    """
    # trusworthy
    N = data.natoms.sum().item()
    hessian = _blockdiagonal_N_3_N_3_loop(N, edge_index, l012_edge_features)
    hessian = hessian.reshape(N * 3, N * 3)
    hessian = _add_node_diagonal_2d_loop(hessian, l012_node_features, N)
    return hessian