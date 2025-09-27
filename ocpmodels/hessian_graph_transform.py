import torch
from torch_geometric.transforms import BaseTransform
from nets.equiformer_v2.hessian_pred_utils import (
    _get_indexadd_offdiagonal_to_flat_hessian_message_indices,
    _get_node_diagonal_1d_indexadd_indices,
    # add_extra_props_for_hessian,
)
# from torch_geometric.data import Batch as TGBatch
# from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

from ocpmodels.common.utils import (
    generate_graph,
    generate_graph_nopbc,
)

# from torch_geometric.data import Data as TGData
# from torch_geometric.data import Batch as TGBatch
# from torch_geometric.data import Dataset as TGDataset

# from collections.abc import Mapping


# from torch.utils.data.dataloader import default_collate

FOLLOW_BATCH = ["diag_ij", "edge_index", "message_idx_ij"]


class HessianGraphTransform(BaseTransform):
    """
    Transform that precomputes graph and hessian message indices for efficient batching.
    Adds the following attributes to the data object:
        edge_index_t [E, 2]
        edge_dist [E]
        distance_vec [E, 3]
        cell_offsets (dummy) [1]
        cell_offset_distances (dummy) [1]
        neighbors (dummy) [1]

        nedges [B]
        message_idx_ij [E * 9]
        message_idx_ji [E * 9]

    where E is the number of edges in the graph.

    Extremely slow. For training it is highly recommended to preprocess the dataset and compute this once.
    """

    def __init__(self, cutoff=5.0, cutoff_hessian=100, max_neighbors=32, use_pbc=False):
        """
        Args:
            cutoff: cutoff radius for the graph
            max_neighbors: maximum number of neighbors for the graph. None means 32.
            use_pbc: whether to use periodic boundary conditions
        """
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_hessian = cutoff_hessian
        self.max_neighbors = max_neighbors
        self.use_pbc = use_pbc

    def __call__(self, data):
        """
        Apply the transform to precompute graph and hessian indices.

        Args:
            data: torch_geometric.data.Data object

        Returns:
            data: Modified data object with precomputed graph and indices
        """
        ########################################################################################
        # Generate graph for backbone and energy and forces
        (
            edge_index,
            edge_distance,
            edge_distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        ) = generate_graph(
            data,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            use_pbc=self.use_pbc,
        )

        # Store graph information in data object
        data.edge_index = edge_index  # transpose to match the batching order
        data.edge_distance = edge_distance
        data.edge_distance_vec = edge_distance_vec
        data.cell_offsets = cell_offsets
        data.cell_offset_distances = cell_offset_distances
        data.neighbors = neighbors
        # add number of edges, analagous to natoms
        data.nedges = torch.tensor(edge_index.shape[1], dtype=torch.long)

        ########################################################################################
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
            cutoff=self.cutoff_hessian,
            max_neighbors=self.max_neighbors,
            use_pbc=self.use_pbc,
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
        N = data.natoms.sum().item()  # Number of atoms
        indices_ij, indices_ji = _get_indexadd_offdiagonal_to_flat_hessian_message_indices(
            N=N, edge_index=edge_index_hessian
        )
        # Store indices in data object
        data.message_idx_ij = indices_ij
        data.message_idx_ji = indices_ji

        # Precompute node message indices for diagonal entries in the hessian
        diag_ij, diag_ji, node_transpose_idx = _get_node_diagonal_1d_indexadd_indices(
            N=N, device=data.pos.device
        )
        # Store indices in data object
        data.diag_ij = diag_ij
        data.diag_ji = diag_ji
        data.node_transpose_idx = node_transpose_idx

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"

