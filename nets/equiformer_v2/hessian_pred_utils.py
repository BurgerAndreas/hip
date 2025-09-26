import torch

from e3nn import o3

import einops

from torch_geometric.data import Batch as TGBatch


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


def _blockdiagonal_N_3_N_3_loop(
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
def _get_flat_indexadd_message_indices(
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
    print("get_flat_indexadd_message_indices edge_index: ", edge_index.shape)
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
    print("get_flat_indexadd_message_indices idx_ij: ", idx_ij.shape)
    print("get_flat_indexadd_message_indices idx_ji: ", idx_ji.shape)
    return idx_ij, idx_ji


def _flat_indexadd(edge_index, messages, data):
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
    print("flat_indexadd edge_index: ", edge_index.shape)
    print("flat_indexadd messages: ", messages.shape)
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
    print("flat_indexadd hessian1d: ", hessian1d.shape)
    return hessian1d


##############################################################################################################
# The following functions are all the same, but with different implementations
# They all add the node features to the diagonal


def _add_node_diagonal_2d_loop(
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
    print("add_node_diagonal_2d_loop hessian: ", hessian.shape)
    print("add_node_diagonal_2d_loop l012_node_features: ", l012_node_features.shape)
    # hessian: (N*3,N*3)
    # l012_node_features: (N,3,3)
    for ii in range(N):
        hessian[ii * 3 : (ii + 1) * 3, ii * 3 : (ii + 1) * 3] += l012_node_features[ii]
        # Add transpose for symmetry
        hessian[ii * 3 : (ii + 1) * 3, ii * 3 : (ii + 1) * 3] += l012_node_features[
            ii
        ].T
    print("add_node_diagonal_2d_loop hessian: ", hessian.shape)
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
    print("get_node_diagonal_1d_indexadd_indices N: ", N)
    print("get_node_diagonal_1d_indexadd_indices device: ", device)
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
    print("get_node_diagonal_1d_indexadd_indices diag_idx: ", diag_idx.shape)
    print(
        "get_node_diagonal_1d_indexadd_indices diag_idx.clone: ", diag_idx.clone().shape
    )
    print(
        "get_node_diagonal_1d_indexadd_indices node_transpose_idx: ",
        node_transpose_idx.shape,
    )
    return diag_idx, diag_idx.clone(), node_transpose_idx


def _add_node_diagonal_1d_indexadd(
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
    print("add_node_diagonal_1d_indexadd hessianflat: ", hessianflat.shape)
    print(
        "add_node_diagonal_1d_indexadd l012_node_features: ", l012_node_features.shape
    )
    print("add_node_diagonal_1d_indexadd data: ", data.diag_ij.shape)
    # diag_ij, diag_ji, node_transpose_idx = _get_node_diagonal_1d_indexadd_indices(N, device)
    diag_ij = data.diag_ij  # (N*3*3) -> (N*3*N*3)
    diag_ji = data.diag_ji  # (N*3*3) -> (N*3*N*3)
    node_transpose_idx = data.node_transpose_idx  # (N*3*3) -> (N*3*3)
    # Flatten node features for direct indexing
    l012_node_features_flat = l012_node_features.reshape(-1)  # (N*3*3)
    # Use two index_add calls: one for direct, one for transpose
    hessianflat.index_add_(0, diag_ij, l012_node_features_flat)
    hessianflat.index_add_(0, diag_ji, l012_node_features_flat[node_transpose_idx])
    print("add_node_diagonal_1d_indexadd hessianflat: ", hessianflat.shape)
    return hessianflat


##############################################################################################################


def predict_hessian_1d_fast(
    edge_index: torch.Tensor,
    data: TGBatch,
    l012_edge_features: torch.Tensor,
    l012_node_features: torch.Tensor,
) -> torch.Tensor:
    """
    Predict the Hessian in flattened 1D form using index_add-based assembly.

    Args:
        edge_index: shape (2, E_total).
        data: object with attributes required by `_flat_indexadd` and
            `_add_node_diagonal_1d_indexadd` (see their docstrings), including
            `natoms`, `message_idx_ij`, `message_idx_ji`, `diag_ij`, `diag_ji`,
            and `node_transpose_idx`.
        l012_edge_features: Tensor, shape (E_total, 3, 3).
        l012_node_features: Tensor, shape (sum_b N_b, 3, 3).

    Returns:
        hessian: Tensor, shape (sum_b (N_b*3)^2,).
    """
    # fast
    hessian = _flat_indexadd(edge_index, l012_edge_features, data)
    hessian = _add_node_diagonal_1d_indexadd(hessian, l012_node_features, data)
    return hessian


def predict_hessian_blockdiagonal_robust(
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
    # trusworthy
    N = data.natoms.sum().item()
    hessian = _blockdiagonal_N_3_N_3_loop(N, edge_index, l012_edge_features)
    hessian = hessian.reshape(N * 3, N * 3)
    hessian = _add_node_diagonal_2d_loop(hessian, l012_node_features, N)
    return hessian
