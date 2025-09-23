import torch
from torch_geometric.transforms import BaseTransform
from nets.equiformer_v2.hessian_pred_utils import (
    _get_flat_indexadd_message_indices,
    _get_node_diagonal_1d_indexadd_indices,
    add_extra_props_for_hessian,
)
from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20

from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from torch_geometric.nn import radius_graph

from torch_geometric.data import Data as TGData
from torch_geometric.data import Batch as TGBatch
from torch_geometric.data import Dataset as TGDataset

from collections.abc import Mapping
from typing import Any, List, Optional, Sequence, Union

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.loader.dataloader import Collater as TGCollater

from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame

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

        ########################################################################################
        # Precompute edge message indices for offdiagonal entries in the hessian
        N = data.natoms.sum().item()  # Number of atoms
        indices_ij, indices_ji = _get_flat_indexadd_message_indices(
            N=N, edge_index=edge_index_hessian
        )
        # Store indices in data object
        # Careful!
        # By default, PyG increments attributes by the number of nodes
        # whenever their attribute names contain the substring index (for historical reasons),
        # which comes in handy for attributes such as edge_index or node_index.
        # This will lead to unexpected behavior for attributes
        # whose names contain the substring index
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

        # add theoretical maximal number of edges
        data.max_nedges = torch.tensor(N * (N - 1), dtype=torch.long)

        return data

    def __repr__(self):
        return f"{self.__class__.__name__}()"


# Taken from ocpmodels/common/utils.py
def generate_graph(
    data,
    cutoff=None,
    max_neighbors=32,
    use_pbc=None,
):
    if use_pbc:
        edge_index, cell_offsets, neighbors = radius_graph_pbc(
            data, cutoff, max_neighbors
        )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        edge_index = radius_graph(
            data.pos,
            r=cutoff,
            batch=data.batch,
            max_num_neighbors=max_neighbors,
        )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)
        # neighbors = torch.tensor([0.0])

    return (
        edge_index,
        edge_dist,
        distance_vec,
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )


def generate_fullyconnected_graph_nopbc(data, cutoff, max_neighbors: int = 32):
    """Used by HORM.
    Maybe easier to differentiate through for autograd Hessian?
    """
    if max_neighbors is None:
        max_neighbors = 32
    pos = data.pos
    edge_index = radius_graph(
        pos, r=cutoff, batch=data.batch, max_num_neighbors=max_neighbors
    )
    j, i = edge_index
    posj = pos[j]
    posi = pos[i]
    vecs = posj - posi
    edge_distance_vec = vecs
    edge_distance = (vecs).norm(dim=-1)
    return (
        edge_index,
        edge_distance,
        edge_distance_vec,
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([0.0]),
    )


# HessianDataLoader
# Offsets indices due to batching.


class HessianBatchTransform:
    """
    Post-batching transform that applies batch-specific index offsetting.
    This should be used in a custom collate function after batching individual samples.

    This replaces the need to call add_extra_props_for_hessian(batch, offset_indices=True)
    during training, moving the computation to dataloader workers.
    """

    def __call__(self, batch):
        """
        Apply batch offsetting to precomputed indices.

        Args:
            batch: Batched torch_geometric.data.Batch object

        Returns:
            batch: Modified batch with offset indices
        """
        # This is equivalent to add_extra_props_for_hessian(batch, offset_indices=True)
        # but optimized for the dataloader worker context
        return add_extra_props_for_hessian(batch, offset_indices=True)


def _create_hessian_collate_fn(dataset, follow_batch, exclude_keys):
    """
    Create a custom collate function that applies batch offsetting after batching.
    Use this instead of the default PyG collate function when training with Hessian prediction.

    Returns:
        collate_fn: Function that can be passed to DataLoader
    """
    tg_collater = TGCollater(dataset, follow_batch, exclude_keys)
    hessian_batch_transform = HessianBatchTransform()

    def collate_fn(data_list):
        # First do the standard PyG batching
        batch = tg_collater(data_list)

        # Then apply batch-specific index offsetting
        batch = hessian_batch_transform(batch)

        return batch

    return collate_fn


class HessianDataLoader(torch.utils.data.DataLoader):
    r"""A copy of the torch_geometric dataloader, but adds our custom collate function after the torch_geometric collate function."""

    def __init__(
        self,
        dataset: Union[TGDataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        do_hessian_batch_offsetting: bool = True,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        self.do_hessian_batch_offsetting = do_hessian_batch_offsetting
        if self.do_hessian_batch_offsetting:
            collate_fn = _create_hessian_collate_fn(dataset, follow_batch, exclude_keys)
        else:
            collate_fn = TGCollater(dataset, follow_batch, exclude_keys)

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=collate_fn,
            **kwargs,
        )
