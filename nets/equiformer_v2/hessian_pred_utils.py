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


def add_extra_props_for_hessian_slow(data, offset_indices=False):
    """Fix indices for batched Hessian prediction.

    If you encounter the following error:
    AttributeError: 'GlobalStorage' object has no attribute 'edge_index_ptr'. Did you mean: 'edge_index'
    You need to add follow_batch=['diag_ij', 'edge_index', 'message_idx_ij'] to the dataloader.
    """
    # add extra props for convience
    nedges = data.nedges_hessian
    B = data.batch.max().item() + 1
    ptr_1d_hessian = [0]
    for b in range(B):
        ptr_1d_hessian.append(ptr_1d_hessian[-1] + (nedges[b].item() * 3) ** 2)
    data.ptr_1d_hessian = torch.tensor(
        ptr_1d_hessian, device=data.batch.device, dtype=torch.long
    )
    # indices are computed for each sample individually
    # so we need to offset the indices by the number of entries in the previous samples in the batch
    if offset_indices:
        if hasattr(data, "offsetdone") and (data.offsetdone is True):
            return data
        data.offsetdone = True
        for b in range(B):
            if b > 0:
                nodes_last_sample = data.natoms[b - 1].item()
                hessian_entries_last_sample = (nodes_last_sample * 3) ** 2
                # (E*3*3) -> (N*3*N*3)
                e_start = data.edge_index_ptr[b].item() * 9
                # assert e_start == data.message_idx_ij_ptr[b], \
                #     f"e_start={e_start} != data.message_idx_ij_ptr[b]={data.message_idx_ij_ptr[b]} * 9"
                # e_dist = nedges[b].item() * 9
                # shift message_index by number of hessian entries in previous samples
                data.message_idx_ij[e_start:] = (
                    data.message_idx_ij[e_start:] + hessian_entries_last_sample
                )
                data.message_idx_ji[e_start:] = (
                    data.message_idx_ji[e_start:] + hessian_entries_last_sample
                )

                # (N*3*3) -> (N*3*N*3)
                _start = data.ptr[b] * 9  # == data.diag_ij_ptr[b]
                # _dist = data.natoms[b].item() * 9
                # shift diag_ij by (N*3*N*3)
                data.diag_ij[_start:] = (
                    data.diag_ij[_start:] + hessian_entries_last_sample
                )
                data.diag_ji[_start:] = (
                    data.diag_ji[_start:] + hessian_entries_last_sample
                )
                # (N*3*3) -> (N*3*3)
                # shift node_transpose_idx by (N_last_sample*3*3)
                _node_entries_last_sample = nodes_last_sample * 9
                data.node_transpose_idx[_start:] = (
                    data.node_transpose_idx[_start:] + _node_entries_last_sample
                )

            # # make sure our arithmetic is correct
            # if b == B-1:
            #     # last sample in batch
            #     e_start = data.edge_index_ptr[b].item() * 9
            #     e_dist = nedges[b].item() * 9
            #     assert e_start + e_dist == data.message_idx_ij.numel(), f"{e_start + e_dist} != {data.message_idx_ij.shape}"
            #     _start = data.ptr[b] * 9
            #     _dist = data.natoms[b].item() * 9
            #     assert _start + _dist == data.diag_ij.numel(), f"{_start + _dist} != {data.diag_ij.shape}"
            #     assert _start + _dist == data.node_transpose_idx.numel(), f"{_start + _dist} != {data.node_transpose_idx.shape}"

    return data


# slightly faster than add_extra_props_for_hessian, gives the same result. does not matter though
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


##############################################################################################################
# The following functions are all the same, but with different implementations
# They all build the Hessian matrix from the edge features


def _blockdiagonal_N_3_N_3_loop(N, edge_index, sym_message):
    device = sym_message.device
    dtype = sym_message.dtype
    hessian = torch.zeros((N, 3, N, 3), device=device, dtype=dtype)
    for ij in range(edge_index.shape[1]):
        i, j = edge_index[0, ij], edge_index[1, ij]
        hessian[i, :, j, :] += sym_message[ij]
        hessian[j, :, i, :] += sym_message[ij].T
    return hessian


def _blockdiagonal_N3_N3_loop(N, edge_index, sym_message):
    device = sym_message.device
    dtype = sym_message.dtype
    hessian2 = torch.zeros(N * 3, N * 3, device=device, dtype=dtype)
    for ij in range(edge_index.shape[1]):
        i, j = edge_index[0, ij], edge_index[1, ij]
        hessian2[i * 3 : (i + 1) * 3, j * 3 : (j + 1) * 3] += sym_message[ij]
        hessian2[j * 3 : (j + 1) * 3, i * 3 : (i + 1) * 3] += sym_message[ij].T
    return hessian2


def _blockdiagonal_N3_N3_loop_explicit(N, edge_index, sym_message):
    # again but maximally explicit
    device = sym_message.device
    dtype = sym_message.dtype
    hessian3 = torch.zeros(N * 3, N * 3, device=device, dtype=dtype)
    for ij in range(edge_index.shape[1]):
        # atom indices
        i, j = edge_index[0, ij], edge_index[1, ij]
        # atom indices * 3 = coordinate indices
        _i = i * 3
        _j = j * 3
        # x
        hessian3[_i, _j] += sym_message[ij][0, 0]
        hessian3[_i, _j + 1] += sym_message[ij][0, 1]
        hessian3[_i, _j + 2] += sym_message[ij][0, 2]
        # y
        hessian3[_i + 1, _j] += sym_message[ij][1, 0]
        hessian3[_i + 1, _j + 1] += sym_message[ij][1, 1]
        hessian3[_i + 1, _j + 2] += sym_message[ij][1, 2]
        # z
        hessian3[_i + 2, _j] += sym_message[ij][2, 0]
        hessian3[_i + 2, _j + 1] += sym_message[ij][2, 1]
        hessian3[_i + 2, _j + 2] += sym_message[ij][2, 2]

        hessian3[_j, _i] += sym_message[ij][0, 0]
        hessian3[_j, _i + 1] += sym_message[ij][1, 0]
        hessian3[_j, _i + 2] += sym_message[ij][2, 0]
        # y
        hessian3[_j + 1, _i] += sym_message[ij][0, 1]
        hessian3[_j + 1, _i + 1] += sym_message[ij][1, 1]
        hessian3[_j + 1, _i + 2] += sym_message[ij][2, 1]
        # z
        hessian3[_j + 2, _i] += sym_message[ij][0, 2]
        hessian3[_j + 2, _i + 1] += sym_message[ij][1, 2]
        hessian3[_j + 2, _i + 2] += sym_message[ij][2, 2]
    return hessian3


# only works for single sample B=1
def _flat_indexadd_explicit(edge_index, sym_message, data):
    if hasattr(data, "nedges_hessian"):
        nedges = data.nedges_hessian
    else:
        nedges = data.nedges
    # do the same thing but in 1d
    device = sym_message.device
    dtype = sym_message.dtype
    total_entries = 0
    for _N in data.natoms:
        total_entries += _N * 3 * _N * 3
    hessianflat = torch.zeros(total_entries, device=device, dtype=dtype)
    messageflat = sym_message.reshape(-1)
    # Build indices for both i->j and j->i contributions
    indices = []
    values = []
    # keep track of sample sizes across batch
    n_edges_prev = 0
    n_node_entries_prev = 0
    n_entries_prev = 0
    for _b in range(data.batch.max().item() + 1):
        _edge_index = edge_index[:, n_edges_prev : n_edges_prev + nedges[_b]]
        E = _edge_index.shape[1]
        N = data.natoms[_b].item()
        for ij in range(E):  # loop over messages
            # message from node i to node j
            i, j = _edge_index[0, ij], _edge_index[1, ij]  # already includes prev edges
            # Add i->j contribution
            for coord_i in range(3):
                for coord_j in range(3):
                    # 1D index in hessian: (i*3 + coord_i) * (N*3) + (j*3 + coord_j)
                    idx_hessian = (i * 3 + coord_i) * (N * 3) + (j * 3 + coord_j)
                    idx_hessian += n_entries_prev
                    # 1D index in message: ij * 9 + coord_i * 3 + coord_j
                    idx_message = ij * 9 + coord_i * 3 + coord_j
                    idx_message += n_node_entries_prev
                    indices.append(idx_hessian)
                    values.append(messageflat[idx_message])
            # Add j->i contribution (transpose)
            for coord_i in range(3):
                for coord_j in range(3):
                    # 1D index in hessian: (j*3 + coord_i) * (N*3) + (i*3 + coord_j)
                    idx_hessian = (j * 3 + coord_i) * (N * 3) + (i * 3 + coord_j)
                    idx_hessian += n_entries_prev
                    # 1D index in message: ij * 9 + coord_j * 3 + coord_i (transpose)
                    idx_message = ij * 9 + coord_j * 3 + coord_i
                    idx_message += n_node_entries_prev
                    indices.append(idx_hessian)
                    values.append(messageflat[idx_message])
        n_entries_prev += (data.natoms[_b].item() * 3) ** 2
        n_node_entries_prev += data.natoms[_b].item() * 3 * 3
        n_edges_prev += nedges[_b].item()
    # Convert to tensors
    indices = torch.tensor(indices, device=device, dtype=torch.long)
    values = torch.tensor(values, device=device, dtype=dtype)
    # Use index_add to efficiently add all values
    assert indices.max().item() < hessianflat.shape[0]
    hessianflat.index_add_(0, indices, values)
    return hessianflat


# support function that can be moved to dataloader
# TODO: speedup
def _get_flat_indexadd_message_indices_slow(N, edge_index):
    E = edge_index.shape[1]
    device = edge_index.device
    # We need 2 * E * 3 * 3 indices (for both i->j and j->i contributions)
    # indices_ij = torch.zeros(E*3*3, device=device, dtype=torch.long)
    # indices_ji = torch.zeros(E*3*3, device=device, dtype=torch.long)
    indices_ij = torch.zeros(E * 3 * 3, dtype=torch.long)
    indices_ji = torch.zeros(E * 3 * 3, dtype=torch.long)
    # Build indices for i->j contributions
    idx = 0
    for ij in range(E):
        i, j = edge_index[0, ij], edge_index[1, ij]
        for coord_i in range(3):
            for coord_j in range(3):
                # 1D index in hessian: (i*3 + coord_i) * (N*3) + (j*3 + coord_j)
                hess_idx = (i * 3 + coord_i) * (N * 3) + (j * 3 + coord_j)
                indices_ij[idx] = hess_idx
                idx += 1
    # Build indices for j->i contributions (transpose)
    idx = 0
    for ij in range(E):
        i, j = edge_index[0, ij], edge_index[1, ij]
        for coord_i in range(3):
            for coord_j in range(3):
                # 1D index in hessian: (j*3 + coord_i) * (N*3) + (i*3 + coord_j)
                hess_idx = (j * 3 + coord_i) * (N * 3) + (i * 3 + coord_j)
                indices_ji[idx] = hess_idx
                idx += 1
    # assert indices_ij.max().item() < N * 3 * N * 3, \
    #     f"indices_ij.max()={indices_ij.max().item()} < N*3*N*3={N*3*N*3}"
    # assert indices_ji.max().item() < N * 3 * N * 3, \
    #     f"indices_ji.max()={indices_ji.max().item()} < N*3*N*3={N*3*N*3}"
    indices_ij = indices_ij.to(device)
    indices_ji = indices_ji.to(device)
    return indices_ij, indices_ji


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


def _flat_indexadd(edge_index, sym_message, data):
    # do the same thing in 1d, but indexing messageflat without storing it in values
    device = sym_message.device
    dtype = sym_message.dtype
    E = edge_index.shape[1]
    messageflat = sym_message.reshape(-1)
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
    assert indices_ij.max().item() < hessian1d.shape[0], (
        f"indices_ij.max()={indices_ij.max().item()} < hessian1d={hessian1d.shape[0]}"
    )
    assert indices_ji.max().item() < hessian1d.shape[0], (
        f"indices_ji.max()={indices_ji.max().item()} < hessian1d={hessian1d.shape[0]}"
    )
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


# probably only works for single sample B=1
def _add_node_diagonal_1d_loop(hessianflat, l012_node_features, data):
    """Add node embeddings to diagonal using 1D indexing with loops"""
    l012_node_features_flat = l012_node_features.reshape(-1)  # (N*3*3,)
    # loop over batches
    n_node_entries_prev = 0
    n_entries_prev = 0
    for b in range(data.batch.max().item() + 1):
        # get the number of atoms in this batch
        N = data.natoms[b].item()
        # Add diagonal elements: for each atom ii, add its 3x3 matrix to diagonal
        for ii in range(N):
            for coord_i in range(3):
                for coord_j in range(3):
                    # 1D index for diagonal element (ii*3 + coord_i, ii*3 + coord_j)
                    diag_idx = (ii * 3 + coord_i) * (N * 3) + (ii * 3 + coord_j)
                    diag_idx += n_entries_prev
                    # 1D index in node features: ii * 9 + coord_i * 3 + coord_j
                    node_idx = ii * 9 + coord_i * 3 + coord_j
                    node_idx += n_node_entries_prev
                    hessianflat[diag_idx] += l012_node_features_flat[node_idx]
                    # Add transpose for symmetry
                    node_idx_T = ii * 9 + coord_j * 3 + coord_i
                    node_idx_T += n_node_entries_prev
                    hessianflat[diag_idx] += l012_node_features_flat[node_idx_T]
        n_node_entries_prev += N * 9
        n_entries_prev += N * 3 * N * 3
    return hessianflat


def _get_node_diagonal_1d_indexadd_indices_slow(N, device):
    # Build diagonal indices for direct and transpose contributions
    diag_indices_direct = torch.zeros(N * 3 * 3, device=device, dtype=torch.long)
    diag_indices_transpose = torch.zeros(N * 3 * 3, device=device, dtype=torch.long)
    idx = 0
    for ii in range(N):
        for coord_i in range(3):
            for coord_j in range(3):
                # 1D index for diagonal element (ii*3 + coord_i, ii*3 + coord_j)
                diag_idx = (ii * 3 + coord_i) * (N * 3) + (ii * 3 + coord_j)
                diag_indices_direct[idx] = diag_idx
                diag_indices_transpose[idx] = diag_idx
                idx += 1
    # Create transpose indices for node features
    node_transpose_idx = torch.zeros(N * 3 * 3, device=device, dtype=torch.long)
    idx = 0
    for ii in range(N):
        for coord_i in range(3):
            for coord_j in range(3):
                # Transpose: swap coord_i and coord_j
                node_idx_T = ii * 9 + coord_j * 3 + coord_i
                node_transpose_idx[idx] = node_idx_T
                idx += 1
    return diag_indices_direct, diag_indices_transpose, node_transpose_idx


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
# Tests


def test_hessian_methods_for_single_sample(
    edge_index, sym_message, l012_node_features, data
):
    """Wasteful but simple B*N*3*B*N*3 Hessian.
    Ends up being block diagonal (only B*N*3*N*3 non-zero entries).
    sym_message: edge features (E, 3, 3)
    l012_node_features: node features (N, 9)
    """
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    E = edge_index.shape[1]
    device = sym_message.device
    dtype = sym_message.dtype
    assert B == 1, "This function is only for single sample"

    print(f"Timing Hessian methods for N={N}, E={E}")

    # Time method 1: _blockdiagonal_N_3_N_3_loop
    torch.cuda.synchronize()
    start_time = time.time()
    hessian = _blockdiagonal_N_3_N_3_loop(N, edge_index, sym_message)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_blockdiagonal_N_3_N_3_loop: {(end_time - start_time) * 1000:.2f}ms")

    # Time method 2: _blockdiagonal_N3_N3_loop
    torch.cuda.synchronize()
    start_time = time.time()
    hessian2 = _blockdiagonal_N3_N3_loop(N, edge_index, sym_message)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_blockdiagonal_N3_N3_loop: {(end_time - start_time) * 1000:.2f}ms")
    assert torch.allclose(hessian.reshape(N * 3, N * 3), hessian2)

    # Time method 3: _blockdiagonal_N3_N3_loop_explicit
    torch.cuda.synchronize()
    start_time = time.time()
    hessian3 = _blockdiagonal_N3_N3_loop_explicit(N, edge_index, sym_message)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_blockdiagonal_N3_N3_loop_explicit: {(end_time - start_time) * 1000:.2f}ms")
    assert torch.allclose(hessian.reshape(N * 3, N * 3), hessian3)

    # # Time method 4: _flat_indexadd_explicit
    # torch.cuda.synchronize()
    # start_time = time.time()
    # hessianflat = _flat_indexadd_explicit(edge_index, sym_message, data)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # print(f"_flat_indexadd_explicit: {(end_time - start_time)*1000:.2f}ms")
    # assert torch.allclose(hessian.reshape(N * 3 * N * 3), hessianflat)

    # Time method 5: _flat_indexadd
    torch.cuda.synchronize()
    start_time = time.time()
    hessianflat2 = _flat_indexadd(edge_index, sym_message, data)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_flat_indexadd: {(end_time - start_time) * 1000:.2f}ms")
    assert torch.allclose(hessian.reshape(N * 3 * N * 3), hessianflat2)
    # also test this
    indices_ij = data.message_idx_ij
    indices_ji = data.message_idx_ji
    if data.batch.max().item() == 0:  # single sample
        _ij, _ji = _get_flat_indexadd_message_indices(data.natoms.item(), edge_index)
        assert torch.allclose(data.edge_index, edge_index)
        assert torch.allclose(indices_ij, _ij)
        assert torch.allclose(indices_ji, _ji)

    # Test all three ways to add node embeddings to diagonal
    hessian = hessian.reshape(N * 3, N * 3)

    # Method 1: 2D indexing with loops
    torch.cuda.synchronize()
    start_time = time.time()
    hessian_with_diag_2d = _add_node_diagonal_2d_loop(hessian, l012_node_features, N)
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_add_node_diagonal_2d_loop: {(end_time - start_time) * 1000:.2f}ms")

    # Method 2: 1D indexing with loops
    torch.cuda.synchronize()
    start_time = time.time()
    hessianflat2_with_diag_1d = _add_node_diagonal_1d_loop(
        hessianflat2, l012_node_features, data
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_add_node_diagonal_1d_loop: {(end_time - start_time) * 1000:.2f}ms")
    assert torch.allclose(
        hessian_with_diag_2d.reshape(N * 3 * N * 3), hessianflat2_with_diag_1d
    )

    # Method 3: 1D indexing with index_add
    torch.cuda.synchronize()
    start_time = time.time()
    hessianflat2_with_diag_indexadd = _add_node_diagonal_1d_indexadd(
        hessianflat2, l012_node_features, data
    )
    torch.cuda.synchronize()
    end_time = time.time()
    print(f"_add_node_diagonal_1d_indexadd: {(end_time - start_time) * 1000:.2f}ms")
    assert torch.allclose(hessianflat2_with_diag_1d, hessianflat2_with_diag_indexadd)
    if data.batch.max().item() == 0:  # single sample
        _diag_ij, _diag_ji, _node_transpose_idx = (
            _get_node_diagonal_1d_indexadd_indices(N, device)
        )
        assert torch.allclose(data.diag_ij, _diag_ij)
        assert torch.allclose(data.diag_ji, _diag_ji)
        assert torch.allclose(data.node_transpose_idx, _node_transpose_idx)

    # Use the 2D version as the final result
    hessian = hessian_with_diag_2d

    return hessian


def test_blockdiagonal_hessian_methods_for_batching(
    edge_index, sym_message, l012_node_features, data, _blockdiagfct
):
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    E = edge_index.shape[1]
    device = sym_message.device
    dtype = sym_message.dtype

    print(f"\nTiming {_blockdiagfct.__name__} for batching: B={B}, N={N}, E={E}")

    # compute the hessian for each sample in the batch separately
    torch.cuda.synchronize()
    start_time = time.time()
    individual_hessians = []
    for b in range(B):
        # Get number of atoms in this batch
        n_atoms_batch = data.natoms[b].item()
        # Find atoms belonging to this batch
        batch_atom_mask = data.batch == b
        batch_atom_indices = torch.where(batch_atom_mask)[0]
        # Find edges within this batch
        batch_edge_mask = (
            batch_atom_mask[edge_index[0]] & batch_atom_mask[edge_index[1]]
        )
        batch_edge_index = edge_index[:, batch_edge_mask]
        batch_sym_message = sym_message[batch_edge_mask]
        # Convert global atom indices to local indices
        min_atom_idx = batch_atom_indices.min()
        batch_edge_index_local = batch_edge_index - min_atom_idx
        # Compute hessian for this batch
        batch_hessian = _blockdiagfct(
            n_atoms_batch, batch_edge_index_local, batch_sym_message
        )
        batch_hessian = batch_hessian.reshape(n_atoms_batch, 3, n_atoms_batch, 3)
        individual_hessians.append(batch_hessian)
    torch.cuda.synchronize()
    end_time = time.time()
    print(
        f"{_blockdiagfct.__name__} (individual batches): {(end_time - start_time) * 1000:.2f}ms"
    )

    # Time the computation of the Hessian on the whole batch
    torch.cuda.synchronize()
    start_time = time.time()
    hessian = _blockdiagfct(N, edge_index, sym_message)
    hessian = hessian.reshape(N, 3, N, 3)
    torch.cuda.synchronize()
    end_time = time.time()
    print(
        f"{_blockdiagfct.__name__} (full batch):         {(end_time - start_time) * 1000:.2f}ms"
    )

    # compare by extracting blocks from full hessian
    atom_offset = 0
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert atom_offset == data.ptr[b].item(), (
            f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        )
        # Extract the block from the full hessian
        block_start = atom_offset
        block_end = atom_offset + n_atoms_batch
        extracted_block = hessian[block_start:block_end, :, block_start:block_end, :]
        individual_hessian = individual_hessians[b]
        # Compare the blocks
        is_close = torch.allclose(extracted_block, individual_hessian, atol=1e-6)
        max_diff = torch.max(torch.abs(extracted_block - individual_hessian)).item()
        assert is_close, (
            f"Batch {b} hessian doesn't match! Max difference: {max_diff:.2e}"
        )
        atom_offset += n_atoms_batch

    return hessian


def test_flat_hessian_methods_same_as_loop_batching(
    edge_index, sym_message, l012_node_features, data, _flat_indexaddfct
):
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    total_entries = 0
    for _N in data.natoms:
        total_entries += _N * 3 * _N * 3
    E = edge_index.shape[1]
    device = sym_message.device
    dtype = sym_message.dtype

    print(f"\nTiming {_flat_indexaddfct.__name__} for batching: B={B}, N={N}, E={E}")

    # Time the computation of the Hessian on the whole batch
    torch.cuda.synchronize()
    start_time = time.time()
    hessianflat = _flat_indexaddfct(edge_index, sym_message, data)
    torch.cuda.synchronize()
    end_time = time.time()
    print(
        f"{_flat_indexaddfct.__name__} (full batch): {(end_time - start_time) * 1000:.2f}ms"
    )
    assert hessianflat.numel() == total_entries, (
        f"Total entries {total_entries} does not match hessianflat.numel() {hessianflat.numel()}"
    )

    # Test batching consistency by comparing individual batch results to block diagonal hessian
    hessian = _blockdiagonal_N_3_N_3_loop(N, edge_index, sym_message)
    # hessian = hessian.reshape(N, 3, N, 3)

    # Compare by extracting blocks from full hessian
    atom_offset = 0
    past_entries = 0
    test_mask = torch.zeros_like(hessian)
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert atom_offset == data.ptr[b].item(), (
            f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        )
        # Extract the block from the full hessian
        block_start = atom_offset
        block_end = atom_offset + n_atoms_batch
        _hessian_blockdiagonal = hessian[
            block_start:block_end, :, block_start:block_end, :
        ]
        n_entries_batch = n_atoms_batch * 3 * n_atoms_batch * 3
        _hessian_flat = hessianflat[past_entries : past_entries + n_entries_batch]
        # Compare the blocks
        # _hessian_blockdiagonal = _hessian_blockdiagonal.reshape(-1)
        _hessian_flat = _hessian_flat.reshape(n_atoms_batch, 3, n_atoms_batch, 3)
        is_close = torch.allclose(_hessian_blockdiagonal, _hessian_flat, atol=1e-6)
        max_diff = torch.max(torch.abs(_hessian_blockdiagonal - _hessian_flat)).item()
        assert is_close, (
            f"Batch {b} hessian doesn't match! {_flat_indexaddfct.__name__} Max difference: {max_diff:.2e}"
        )
        atom_offset += n_atoms_batch
        past_entries += n_entries_batch
    # invert the mask
    test_mask = 1 - test_mask
    # all other (non-visited) entries should be zero
    assert torch.allclose(hessian * test_mask, torch.zeros_like(hessian))
    return hessianflat


def test_blockdiagonal_nodediagonal_hessian_methods_for_batching(
    edge_index, sym_message, l012_node_features, data, _blockdiagfct, _nodediagfct
):
    """
    Test node diagonal methods for batching scenarios.
    Tests _add_node_diagonal_2d_loop and other node diagonal methods.
    """
    B = data.batch.max().item() + 1
    N = data.natoms.sum().item()
    # total_entries = 0
    # for _N in data.natoms:
    #     total_entries += _N * 3 * _N * 3

    E = edge_index.shape[1]
    device = sym_message.device
    dtype = sym_message.dtype

    print(f"\nTiming {_nodediagfct.__name__} for batching: B={B}, N={N}, E={E}")

    # First build the block diagonal hessian using the provided method
    hessian = _blockdiagfct(N, edge_index, sym_message)
    hessian = hessian.reshape(N * 3, N * 3)

    # Test the node diagonal method on the full batch
    torch.cuda.synchronize()
    start_time = time.time()
    hessian_with_diag_full = _nodediagfct(hessian, l012_node_features, N)
    torch.cuda.synchronize()
    end_time = time.time()
    print(
        f"{_nodediagfct.__name__} (full batch): {(end_time - start_time) * 1000:.2f}ms"
    )

    # Test the node diagonal method on individual batches
    torch.cuda.synchronize()
    start_time = time.time()
    individual_hessians_with_diag = []
    for b in range(B):
        # Get number of atoms in this batch
        n_atoms_batch = data.natoms[b].item()
        # Find atoms belonging to this batch
        batch_atom_mask = data.batch == b
        batch_atom_indices = torch.where(batch_atom_mask)[0]
        # Find edges within this batch
        batch_edge_mask = (
            batch_atom_mask[edge_index[0]] & batch_atom_mask[edge_index[1]]
        )
        batch_edge_index = edge_index[:, batch_edge_mask]
        batch_sym_message = sym_message[batch_edge_mask]
        # Convert global atom indices to local indices
        min_atom_idx = batch_atom_indices.min()
        batch_edge_index_local = batch_edge_index - min_atom_idx

        # Get node features for this batch
        batch_node_features = l012_node_features[batch_atom_indices]

        # Compute hessian for this batch
        batch_hessian = _blockdiagfct(
            n_atoms_batch, batch_edge_index_local, batch_sym_message
        )
        batch_hessian = batch_hessian.reshape(n_atoms_batch * 3, n_atoms_batch * 3)

        # Add node diagonal for this batch
        batch_hessian_with_diag = _nodediagfct(
            batch_hessian, batch_node_features, n_atoms_batch
        )
        individual_hessians_with_diag.append(batch_hessian_with_diag)
    torch.cuda.synchronize()
    end_time = time.time()
    print(
        f"{_nodediagfct.__name__} (individual batches): {(end_time - start_time) * 1000:.2f}ms"
    )

    # Compare by extracting blocks from full hessian
    atom_offset = 0
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert atom_offset == data.ptr[b].item(), (
            f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        )

        # Extract the block from the full hessian
        block_start = atom_offset * 3
        block_end = (atom_offset + n_atoms_batch) * 3
        extracted_block = hessian_with_diag_full[
            block_start:block_end, block_start:block_end
        ]
        individual_hessian_with_diag = individual_hessians_with_diag[b]

        # Compare the blocks
        is_close = torch.allclose(
            extracted_block, individual_hessian_with_diag, atol=1e-6
        )
        max_diff = torch.max(
            torch.abs(extracted_block - individual_hessian_with_diag)
        ).item()
        assert is_close, (
            f"Batch {b} hessian with diagonal doesn't match! Max difference: {max_diff:.2e}"
        )
        atom_offset += n_atoms_batch

    return hessian_with_diag_full


def test_fast_vs_trusworthy(
    num_atoms, edge_index, sym_message, l012_node_features, data
):
    B = data.batch.max().item() + 1

    # trusworthy B*N*3*B*N*3
    hessian_2d = _blockdiagonal_N_3_N_3_loop(num_atoms, edge_index, sym_message)
    hessian_2d = hessian_2d.reshape(num_atoms * 3, num_atoms * 3)
    # hessian_2d = _add_node_diagonal_2d_loop(hessian_2d, l012_node_features, num_atoms)
    # hessian_2d = hessian_2d.reshape(num_atoms * 3, num_atoms * 3)

    # fig = px.imshow(hessian_2d.detach().cpu().numpy())
    # fig.write_image("hessian.png")

    hessian_2d_2 = _blockdiagonal_N3_N3_loop(num_atoms, edge_index, sym_message)
    hessian_2d_2 = hessian_2d_2.reshape(num_atoms * 3, num_atoms * 3)
    assert torch.allclose(hessian_2d, hessian_2d_2), (
        f"hessian_2d and hessian_2d_2 don't match: {torch.max(torch.abs(hessian_2d - hessian_2d_2)):.2e}"
    )
    # fig = px.imshow(hessian_2d.detach().cpu().numpy())
    # fig.write_image("hessian_2d.png")

    # fast B*N*3*N*3
    hessian_1d = _flat_indexadd(edge_index, sym_message, data)
    # hessian_1d = _add_node_diagonal_1d_indexadd(hessian_1d, l012_node_features, data)
    # # Only works for B=1
    # fig = px.imshow(hessian_1d.reshape(num_atoms, 3, num_atoms, 3).detach().cpu().numpy())
    # fig.write_image("hessian_1d.png")

    # Compare by extracting blocks from full hessian
    atom_offset = 0
    n_entries_prev = 0
    test_mask = torch.zeros_like(hessian_1d)
    for b in range(B):
        n_atoms_batch = data.natoms[b].item()
        assert atom_offset == data.ptr[b].item(), (
            f"Atom offset {atom_offset} does not match batch {b} ptr {data.ptr[b].item()}"
        )

        # Extract the block from the full hessian
        block_start = atom_offset * 3
        block_end = (atom_offset + n_atoms_batch) * 3
        extracted_block_2d = hessian_2d[block_start:block_end, block_start:block_end]
        n_entries = n_atoms_batch * 3 * n_atoms_batch * 3
        extracted_block_1d = hessian_1d[n_entries_prev : n_entries_prev + n_entries]
        test_mask[n_entries_prev : n_entries_prev + n_entries] = 1
        # Compare the blocks
        extracted_block_1d = extracted_block_1d.reshape(
            n_atoms_batch * 3, n_atoms_batch * 3
        )
        is_close = torch.allclose(extracted_block_2d, extracted_block_1d, atol=1e-6)
        max_diff = torch.max(torch.abs(extracted_block_2d - extracted_block_1d)).item()
        assert is_close, (
            f"Batch {b} / {B - 1} hessian with diagonal doesn't match! Max difference: {max_diff:.2e}"
        )
        atom_offset += n_atoms_batch
        n_entries_prev += n_entries
    # invert the mask
    test_mask = 1 - test_mask
    # all other (non-visited) entries should be zero
    assert torch.allclose(hessian_1d * test_mask, torch.zeros_like(hessian_1d))
    print(f"test_fast_vs_trusworthy passed for B={B}")
    return hessian_2d, hessian_1d


def run_hessian_tests(edge_index, sym_message, l012_node_features, data):
    # run tests
    B = data.batch.max().item() + 1
    if B == 1:
        hessian = test_hessian_methods_for_single_sample(
            edge_index, sym_message, l012_node_features, data
        )
    else:
        hessian = test_blockdiagonal_hessian_methods_for_batching(
            edge_index,
            sym_message,
            l012_node_features,
            data,
            _blockdiagonal_N_3_N_3_loop,
        )
        hessian = test_blockdiagonal_hessian_methods_for_batching(
            edge_index,
            sym_message,
            l012_node_features,
            data,
            _blockdiagonal_N3_N3_loop,
        )
        hessian = test_blockdiagonal_hessian_methods_for_batching(
            edge_index,
            sym_message,
            l012_node_features,
            data,
            _blockdiagonal_N3_N3_loop_explicit,
        )
        # 1d

        # don't care
        # hessian = test_flat_hessian_methods_same_as_loop_batching(
        #     edge_index,
        #     sym_message,
        #     l012_node_features,
        #     data,
        #     _flat_indexadd_explicit,
        # )

        # # TODO: fix this
        # hessian = test_flat_hessian_methods_same_as_loop_batching(
        #     edge_index, sym_message, l012_node_features, data, _flat_indexadd
        # )

        # Test node diagonal methods for batching
        hessian = test_blockdiagonal_nodediagonal_hessian_methods_for_batching(
            edge_index,
            sym_message,
            l012_node_features,
            data,
            _blockdiagonal_N3_N3_loop,
            _add_node_diagonal_2d_loop,
        )

    # the one that really counts
    hessian_2d, hessian_1d = test_fast_vs_trusworthy(
        data.natoms.sum().item(), edge_index, sym_message, l012_node_features, data
    )
    return
