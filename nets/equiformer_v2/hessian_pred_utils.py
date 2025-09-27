import torch

from e3nn import o3

import einops

from torch_geometric.data import Batch as TGBatch

from ocpmodels.common.utils import (
    generate_graph,
    # generate_graph_nopbc,
)


# l0_features = x_message.embedding.narrow(dimension=1, start=0, length=1)
# l1_features = x_message.embedding.narrow(dimension=1, start=1, length=3)
# l2_features = x_message.embedding.narrow(dim=1, start=4, length=5) # length=2l+1
def irreps_to_cartesian_matrix(irreps: torch.Tensor) -> torch.Tensor:
    """Luca Thiede's creation.
    irreps: torch.Tensor [N, 9] or [E, 9]
    Returns:
        torch.Tensor [N, 3, 3] or [E, 3, 3]
    """
    # M = torch.zeros((irreps.shape[0], 3, 3), device=irreps.device, dtype=irreps.dtype)
    # for l3 in range(3):
    #     ClGo = o3.wigner_3j(1, 1, l3, dtype=irreps.dtype, device=irreps.device)
    #     if l3 == 0:
    #         features_l3 = irreps[..., :1]
    #     elif l3 == 1:
    #         features_l3 = irreps[..., 1:4]
    #     elif l3 == 2:
    #         features_l3 = irreps[..., 4:9]
    #     M += einops.einsum(ClGo, features_l3, "m1 m2 m3, b m3 -> b m1 m2")
    return (
        einops.einsum(
            o3.wigner_3j(1, 1, 0, dtype=irreps.dtype, device=irreps.device),
            irreps[..., :1],
            "m1 m2 m3, b m3 -> b m1 m2",
        )
        + einops.einsum(
            o3.wigner_3j(1, 1, 1, dtype=irreps.dtype, device=irreps.device),
            irreps[..., 1:4],
            "m1 m2 m3, b m3 -> b m1 m2",
        )
        + einops.einsum(
            o3.wigner_3j(1, 1, 2, dtype=irreps.dtype, device=irreps.device),
            irreps[..., 4:9],
            "m1 m2 m3, b m3 -> b m1 m2",
        )
    )


def add_extra_props_for_hessian(data: TGBatch, offset_indices: bool = True) -> TGBatch:
    """
    Optionally offset precomputed per-sample 1D indices to batched/global space
    and attach convenience pointers for flattened Hessians.

    Expected data attributes (before call):
        - batch: shape (sum_b N_b,)
        - natoms: shape (B,)
        - nedges_hessian: shape (B,)
        - message_idx_ij: shape (sum_b E_b*9,)
        - message_idx_ji: shape (sum_b E_b*9,)
        - diag_ij: shape (sum_b N_b*9,)
        - diag_ji: shape (sum_b N_b*9,)
        - node_transpose_idx: shape (sum_b N_b*9,)

    Does
    - message_idx_ij/message_idx_ji/diag_ij/diag_ji/node_transpose_idx are
        offset in-place to index into the batched flattened Hessian.
    - ptr_1d_hessian: of shape (B+1,) may be added (when B>1),
        acting as a pointer over per-sample flattened Hessian segments.

    Args:
        data: Object with attributes listed above.

    Returns:
        data: The same object, with fields updated/added as described above.
    """
    # add extra props for convience
    nedges = data.nedges_hessian
    B = data.batch.max().item() + 1
    # vectorized pointer build
    _nedges = nedges.to(device=data.batch.device, dtype=torch.long)
    _sizes = (_nedges * 3) ** 2
    # indices are computed for each sample individually
    # so we need to offset the indices by the number of entries in the previous samples in the batch
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


def _loop_offdiagonal_to_blockdiagonal_hessian(
    N: int, edge_index: torch.Tensor, messages: torch.Tensor
) -> torch.Tensor:
    """
    Assemble a block matrix H of shape (B*N,3,B*N,3) from edge messages.

    Args:
        N: int, number of atoms (total across batch for this call).
        edge_index: shape (2, E), directed edges i->j.
        messages: Tensor, shape (E, 3, 3), message blocks for each edge.

    Returns:
        hessian: Tensor, shape (B*N, 3, B*N, 3).
    """
    device = messages.device
    dtype = messages.dtype
    hessian = torch.zeros((N, 3, N, 3), device=device, dtype=dtype)
    for ij in range(edge_index.shape[1]):
        i, j = edge_index[0, ij], edge_index[1, ij]
        hessian[i, :, j, :] += messages[ij]
        hessian[j, :, i, :] += messages[ij].T
    return hessian


# support function that can be moved to dataloader
def _get_indexadd_offdiagonal_to_flat_hessian_message_indices(
    N: int, edge_index: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build flattened 1D indices to scatter (E,3,3) messages into a 1D Hessian.

    Args:
        N: int, number of atoms.
        edge_index: shape (2, E).

    Returns:
        idx_ij: shape (E*9,), indices for i->j blocks.
        idx_ji: shape (E*9,), indices for j->i blocks (transpose).
    """
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


def _indexadd_offdiagonal_to_flat_hessian(edge_index, messages, data):
    """
    Scatter edge message blocks into a flattened Hessian using 1D index_add.

    Args:
        edge_index: shape (2, E_total).
        messages: Tensor, shape (E_total, 3, 3).
        data: object with attributes
            - natoms: shape (B,)
            - message_idx_ij: shape (E_total*9,)
            - message_idx_ji: shape (E_total*9,)

    Returns:
        hessian1d: Tensor, shape (sum_b (N_b*3)^2,).
    """
    # do the same thing in 1d, but indexing messageflat without storing it in values
    total_entries = int(torch.sum((data.natoms * 3) ** 2).item())
    hessian1d = torch.zeros(total_entries, device=messages.device, dtype=messages.dtype)
    E = edge_index.shape[1]
    messageflat = messages.reshape(-1)
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


def _loop_diagonal_to_blockdiagonal_hessian(
    hessian: torch.Tensor, l012_node_features: torch.Tensor, N: int
) -> torch.Tensor:
    """
    Add per-node (3x3) features to the diagonal blocks of a 2D Hessian.

    Args:
        hessian: Tensor, shape (N*3, N*3).
        l012_node_features: Tensor, shape (N, 3, 3).
        N: int, number of atoms.

    Returns:
        hessian: Tensor, shape (N*3, N*3), updated in-place and returned.
    """
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
def _get_node_diagonal_1d_indexadd_indices(
    N: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build flattened indices for adding (N,3,3) node features into 1D Hessian.

    Args:
        N: int, number of atoms.
        device: torch.device for created index tensors.

    Returns:
        diag_ij: shape (N*9,), flattened indices for diagonal blocks.
        diag_ji: shape (N*9,), identical to diag_ij.
        node_transpose_idx: shape (N*9,), permutation indices that
            map flat(node_features) to flat(node_features.transpose(-2, -1)).
    """
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


def _indexadd_diagonal_to_flat_hessian(
    hessianflat: torch.Tensor, l012_node_features: torch.Tensor, data: torch.Tensor
) -> torch.Tensor:
    """
    Add node (3x3) features to the diagonal of a flattened Hessian using 1D
    index_add operations.

    Args:
        hessianflat: Tensor, shape (sum_b (N_b*3)^2,).
        l012_node_features: Tensor, shape (sum_b N_b, 3, 3).
        data: object with attributes
            - diag_ij: shape (sum_b N_b*9,)
            - diag_ji: shape (sum_b N_b*9,)
            - node_transpose_idx: shape (sum_b N_b*9,)

    Returns:
        hessianflat: Tensor, shape (sum_b (N_b*3)^2,), updated and returned.
    """
    node_transpose_idx = data.node_transpose_idx  # (N*3*3) -> (N*3*3)
    # Flatten node features for direct indexing
    l012_node_features_flat = l012_node_features.reshape(-1)  # (N*3*3)
    # Use two index_add calls: one for direct, one for transpose
    hessianflat.index_add_(0, data.diag_ij, l012_node_features_flat)
    hessianflat.index_add_(0, data.diag_ji, l012_node_features_flat[node_transpose_idx])
    # # V2
    # # Flatten node features and create transposed flatten via gather using precomputed permutation
    # l012_node_features_flat = l012_node_features.reshape(-1)#.contiguous()
    # l012_node_features_flat_T = l012_node_features_flat[data.node_transpose_idx]
    # # Fused index_add
    # fused_indices = torch.cat((data.diag_ij, data.diag_ji), dim=0)
    # fused_values = torch.cat((l012_node_features_flat, l012_node_features_flat_T), dim=0)
    # hessianflat.index_add_(0, fused_indices, fused_values)
    return hessianflat


##############################################################################################################


def l012_features_to_hessian(
    edge_index: torch.Tensor,
    data: TGBatch,
    l012_edge_features: torch.Tensor,
    l012_node_features: torch.Tensor,
) -> torch.Tensor:
    """
    Predict the Hessian in flattened 1D form using index_add-based assembly.

    Args:
        edge_index: shape (2, E_total).
        data: object with attributes required by `_indexadd_offdiagonal_to_flat_hessian` and
            `_indexadd_diagonal_to_flat_hessian` (see their docstrings), including
            `natoms`, `message_idx_ij`, `message_idx_ji`, `diag_ij`, `diag_ji`,
            and `node_transpose_idx`.
        l012_edge_features: Tensor, shape (E_total, 3, 3).
        l012_node_features: Tensor, shape (sum_b N_b, 3, 3).

    Returns:
        hessian: Tensor, shape (sum_b (N_b*3)^2,).
    """
    # fast
    hessian = _indexadd_offdiagonal_to_flat_hessian(
        edge_index, l012_edge_features, data
    )
    hessian = _indexadd_diagonal_to_flat_hessian(hessian, l012_node_features, data)
    return hessian


def l012_features_to_hessian_loops(
    edge_index: torch.Tensor,
    data: TGBatch,
    l012_edge_features: torch.Tensor,
    l012_node_features: torch.Tensor,
):
    """
    Predict the Hessian as a 2D matrix using explicit block assembly.

    Args:
        edge_index: shape (2, E_total).
        data: object with attribute `natoms: shape (B,)`.
        l012_edge_features: Tensor, shape (E_total, 3, 3).
        l012_node_features: Tensor, shape (sum_b N_b, 3, 3).

    Returns:
        hessian: Tensor, shape (B*N_total*3, B*N_total*3), where
            N_total = sum_b N_b.
    """
    # slow but trusworthy
    N = data.natoms.sum().item()
    hessian = _loop_offdiagonal_to_blockdiagonal_hessian(
        N, edge_index, l012_edge_features
    )
    hessian = hessian.reshape(N * 3, N * 3)
    hessian = _loop_diagonal_to_blockdiagonal_hessian(hessian, l012_node_features, N)
    return hessian


def add_graph_batch(
    data: TGBatch,
    cutoff: float = 100.0,
    max_neighbors: int = 32,
    use_pbc: bool = False,
) -> TGBatch:
    """
    Build Hessian graph and precompute globally-offset indices for a batched object.

    This combines the responsibilities of generating the Hessian graph (per-sample)
    and producing message/diagonal indices that directly index into the final
    flattened Hessian of length sum_b (N_b*3)^2.

    Expects `data` to contain at least:
      - pos: (sum_b N_b, 3)
      - batch: (sum_b N_b,)
      - natoms: (B,)

    Adds/overwrites on `data`:
      - edge_index_hessian, edge_distance_hessian, edge_distance_vec_hessian,
        cell_offsets_hessian, cell_offset_distances_hessian, neighbors_hessian
      - nedges_hessian: (B,)
      - message_idx_ij, message_idx_ji: (sum_b E_b * 9,)
      - diag_ij, diag_ji: (sum_b N_b * 9,)
      - node_transpose_idx: (sum_b N_b * 9,)
      - ptr_1d_hessian: (B+1,)
      - offsetdone: True
    """
    device = data.pos.device

    # 1) Generate batched Hessian graph
    (
        edge_index_hessian,
        edge_distance_hessian,
        edge_distance_vec_hessian,
        cell_offsets_hessian,
        cell_offset_distances_hessian,
        neighbors_hessian,
    ) = generate_graph(
        data,
        cutoff=cutoff,
        max_neighbors=max_neighbors,
        use_pbc=use_pbc,
    )

    data.edge_index_hessian = edge_index_hessian
    data.edge_distance_hessian = edge_distance_hessian
    data.edge_distance_vec_hessian = edge_distance_vec_hessian
    data.cell_offsets_hessian = cell_offsets_hessian
    data.cell_offset_distances_hessian = cell_offset_distances_hessian
    data.neighbors_hessian = neighbors_hessian

    # 2) Per-sample counts and offsets
    natoms = data.natoms.to(device=device, dtype=torch.long)  # (B,)
    B = natoms.shape[0]
    node_cumsum = torch.cumsum(natoms, dim=0)
    node_offsets = torch.zeros_like(node_cumsum)
    if B > 1:
        node_offsets[1:] = node_cumsum[:-1]

    # Hessian segment sizes per sample and their prefix sums
    hess_sizes_per_sample = (natoms * 3) ** 2  # (B,)
    hess_cumsum = torch.cumsum(hess_sizes_per_sample, dim=0)
    hess_offsets = torch.zeros_like(hess_cumsum)
    if B > 1:
        hess_offsets[1:] = hess_cumsum[:-1]

    # Pointer (B+1,)
    data.ptr_1d_hessian = torch.empty(B + 1, device=device, dtype=torch.long)
    data.ptr_1d_hessian[0] = 0
    data.ptr_1d_hessian[1:] = hess_cumsum

    # 3) Edge-wise indices (off-diagonal blocks)
    E_total = edge_index_hessian.shape[1]
    i_global = edge_index_hessian[0]
    j_global = edge_index_hessian[1]
    sample_by_edge = data.batch[i_global].to(dtype=torch.long)

    # Per-sample edge counts (B,)
    nedges_hessian = torch.bincount(sample_by_edge, minlength=B)
    data.nedges_hessian = nedges_hessian

    # Local (within-sample) node indices for each edge
    edge_node_offset = node_offsets[sample_by_edge]
    i_local = (i_global - edge_node_offset).to(dtype=torch.long)
    j_local = (j_global - edge_node_offset).to(dtype=torch.long)
    N3_by_edge = (natoms[sample_by_edge] * 3).to(dtype=torch.long)

    # Build 3x3 per-edge indices in a vectorized fashion
    ci = torch.arange(3, device=device, dtype=torch.long).view(1, 3, 1)
    cj = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
    i_local = i_local.view(E_total, 1, 1)
    j_local = j_local.view(E_total, 1, 1)
    N3_by_edge = N3_by_edge.view(E_total, 1, 1)

    idx_ij_in_sample = (i_local * 3 + ci) * N3_by_edge + (j_local * 3 + cj)
    idx_ji_in_sample = (j_local * 3 + ci) * N3_by_edge + (i_local * 3 + cj)

    # Add per-edge Hessian segment offsets to obtain global indices
    edge_hess_offset = hess_offsets[sample_by_edge].view(E_total, 1, 1)
    data.message_idx_ij = (idx_ij_in_sample + edge_hess_offset).reshape(-1)
    data.message_idx_ji = (idx_ji_in_sample + edge_hess_offset).reshape(-1)

    # 4) Node diagonal indices per sample, vectorized to global
    total_nodes = int(natoms.sum().item())
    if total_nodes > 0:
        # Per-node sample id and local indices
        sample_by_node = data.batch.to(dtype=torch.long, device=device)
        global_node_index = torch.arange(total_nodes, device=device, dtype=torch.long)
        ii_local = global_node_index - node_offsets[sample_by_node]
        N3_by_node = (natoms[sample_by_node] * 3).to(dtype=torch.long)
        hess_offset_by_node = hess_offsets[sample_by_node]

        # Build (N, 3, 3) broadcasted indices for rows/cols within each sample
        ci = torch.arange(3, device=device, dtype=torch.long).view(1, 3, 1)
        cj = torch.arange(3, device=device, dtype=torch.long).view(1, 1, 3)
        ii_local_b = ii_local.view(total_nodes, 1, 1)
        N3_b = N3_by_node.view(total_nodes, 1, 1)
        base_hess_b = hess_offset_by_node.view(total_nodes, 1, 1)

        row = ii_local_b * 3 + ci
        col = ii_local_b * 3 + cj
        local_diag = row * N3_b + col  # (N,3,3)
        global_diag = local_diag + base_hess_b
        data.diag_ij = global_diag.reshape(-1)
        data.diag_ji = data.diag_ij.clone()

        # Node transpose mapping indices into flat node-feature tensor
        # For each node: base = (node_offsets[s]*9 + ii_local*9), then add (Cj*3 + Ci)
        base_node_flat = (node_offsets[sample_by_node] * 9 + ii_local * 9).view(
            total_nodes, 1, 1
        )
        within9_T = cj * 3 + ci  # (1,3,3)
        node_transpose_idx = base_node_flat + within9_T
        data.node_transpose_idx = node_transpose_idx.reshape(-1)
    else:
        data.diag_ij = torch.empty(0, device=device, dtype=torch.long)
        data.diag_ji = torch.empty(0, device=device, dtype=torch.long)
        data.node_transpose_idx = torch.empty(0, device=device, dtype=torch.long)

    data.offsetdone = True
    return data


if __name__ == "__main__":
    import torch_geometric
    import copy
    import time

    def add_graph_single_sample(data):
        # Generate hessian graph
        (
            edge_index_hessian,
            edge_distance_hessian,
            edge_distance_vec_hessian,
            cell_offsets_hessian,
            cell_offset_distances_hessian,
            neighbors_hessian,
        ) = generate_graph(
            data,
            cutoff=100.0,
            max_neighbors=32,
            use_pbc=False,
        )

        # Store hessian graph information in data object
        data.edge_index_hessian = edge_index_hessian
        data.edge_distance_hessian = edge_distance_hessian
        data.edge_distance_vec_hessian = edge_distance_vec_hessian
        data.cell_offsets_hessian = cell_offsets_hessian
        data.cell_offset_distances_hessian = cell_offset_distances_hessian
        data.neighbors_hessian = neighbors_hessian
        # add number of edges, analagous to natoms
        data.nedges_hessian = torch.tensor(
            edge_index_hessian.shape[1], dtype=torch.long
        )

        # Precompute edge message indices for offdiagonal entries in the hessian
        # TODO: only works for a single sample, not for a batch
        N = data.natoms.sum().item()  # Number of atoms
        indices_ij, indices_ji = (
            _get_indexadd_offdiagonal_to_flat_hessian_message_indices(
                N=N, edge_index=edge_index_hessian
            )
        )
        # Store indices in data object
        data.message_idx_ij = indices_ij
        data.message_idx_ji = indices_ji

        # Precompute node message indices for diagonal entries in the hessian
        # TODO: only works for a single sample, not for a batch
        diag_ij, diag_ji, node_transpose_idx = _get_node_diagonal_1d_indexadd_indices(
            N=N, device=data.pos.device
        )
        # Store indices in data object
        data.diag_ij = diag_ij
        data.diag_ji = diag_ji
        data.node_transpose_idx = node_transpose_idx
        return data

    data1_base = torch_geometric.data.Data(
        pos=torch.randn(10, 3),
        z=torch.randint(1, 10, (10,)),
        natoms=torch.tensor([10]),
    )
    data2_base = torch_geometric.data.Data(
        pos=torch.randn(10, 3),
        z=torch.randint(1, 10, (10,)),
        natoms=torch.tensor([10]),
    )

    ##################################################################################
    # Test 1: the functions are equivalent
    ##################################################################################

    data1 = add_graph_single_sample(copy.deepcopy(data1_base))
    data2 = add_graph_single_sample(copy.deepcopy(data2_base))

    batch = torch_geometric.data.Batch.from_data_list([data1, data2])
    batch = add_extra_props_for_hessian(batch)

    edge_index = batch.edge_index_hessian
    rnd_messages = torch.randn(edge_index.shape[1], 3, 3)
    rnd_node_features = torch.randn(batch.natoms.sum().item(), 3, 3)
    hessian = l012_features_to_hessian(
        edge_index, batch, rnd_messages, rnd_node_features
    )
    hessian_loops = l012_features_to_hessian_loops(
        edge_index, batch, rnd_messages, rnd_node_features
    )
    print(hessian.shape)
    print(hessian_loops.shape)

    # first datapoint
    print(
        (
            hessian[: (batch.natoms[0] * 3) ** 2]
            - hessian_loops[: (batch.natoms[0] * 3), : (batch.natoms[0] * 3)].reshape(
                -1
            )
        )
        .abs()
        .max()
    )
    # second datapoint
    print(
        (
            hessian[(batch.natoms[0] * 3) ** 2 :]
            - hessian_loops[(batch.natoms[0] * 3) :, (batch.natoms[0] * 3) :].reshape(
                -1
            )
        )
        .abs()
        .max()
    )

    ##################################################################################
    # Test 2: add graph to batch (so we can do it on the fly during forward pass)
    ##################################################################################

    batch = torch_geometric.data.Batch.from_data_list(
        [copy.deepcopy(data1_base), copy.deepcopy(data2_base)]
    )
    batch = add_graph_batch(batch)

    edge_index = batch.edge_index_hessian
    # rnd_messages = torch.randn(edge_index.shape[1], 3, 3)
    # rnd_node_features = torch.randn(batch.natoms.sum().item(), 3, 3)
    hessian2 = l012_features_to_hessian(
        edge_index, batch, rnd_messages, rnd_node_features
    )
    hessian_loops2 = l012_features_to_hessian_loops(
        edge_index, batch, rnd_messages, rnd_node_features
    )

    print()
    # first datapoint
    print(
        (
            hessian2[: (batch.natoms[0] * 3) ** 2]
            - hessian_loops2[: (batch.natoms[0] * 3), : (batch.natoms[0] * 3)].reshape(
                -1
            )
        )
        .abs()
        .max()
    )
    # second datapoint
    print(
        (
            hessian2[(batch.natoms[0] * 3) ** 2 :]
            - hessian_loops2[(batch.natoms[0] * 3) :, (batch.natoms[0] * 3) :].reshape(
                -1
            )
        )
        .abs()
        .max()
    )

    # compare to before
    print()
    print((hessian2 - hessian).abs().max())
    print((hessian_loops2 - hessian_loops).abs().max())

    ##################################################################################
    # Test 3: timings for add_graph_single_sample, add_extra_props_for_hessian, add_graph_batch
    ##################################################################################
    print()

    def build_batch(num_atoms: int = 10, B: int = 2):
        samples = []
        for _ in range(B):
            samples.append(
                torch_geometric.data.Data(
                    pos=torch.randn(num_atoms, 3),
                    z=torch.randint(1, 10, (num_atoms,)),
                    natoms=torch.tensor([num_atoms]),
                )
            )
        return torch_geometric.data.Batch.from_data_list(samples)

    def time_call(fn, data, repeats: int = 5):
        # Warmup
        _ = fn(data)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = fn(data)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / repeats

    # Prepare base batch
    base_batch = build_batch(num_atoms=32, B=4)

    # Time add_graph_single_sample (on entire batch object as in Test 2)
    t_single = time_call(add_graph_single_sample, base_batch, repeats=100)
    print(f"add_graph_single_sample: {t_single:.3f} ms")

    # Time add_graph_batch (single call)
    t_batch = time_call(add_graph_batch, base_batch, repeats=100)
    print(f"add_graph_batch: {t_batch:.3f} ms")

    # Time l012_features_to_hessian (assembly only)
    batch_for_assembly = add_graph_batch(build_batch(num_atoms=32, B=4))
    edge_index_b = batch_for_assembly.edge_index_hessian
    rnd_messages_b = torch.randn(edge_index_b.shape[1], 3, 3)
    rnd_node_features_b = torch.randn(batch_for_assembly.natoms.sum().item(), 3, 3)

    def _call_assembly(_):
        return l012_features_to_hessian(
            edge_index_b, batch_for_assembly, rnd_messages_b, rnd_node_features_b
        )

    t_assemble = time_call(_call_assembly, None, repeats=100)
    print(f"l012_features_to_hessian: {t_assemble:.3f} ms")
