import math
from math import pi
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import Embedding
from torch_geometric.nn import radius_graph
from torch_geometric.nn.conv import MessagePassing

# from torch_scatter import scatter, segment_coo, segment_csr
from nets.scatter_utils import scatter, segment_coo, segment_csr


def get_max_neighbors_mask(natoms, index, atom_distance, max_num_neighbors_threshold):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_sort = distance_sort[:, :max_num_neighbors_threshold]
    index_sort = index_sort[:, :max_num_neighbors_threshold]

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors_threshold
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def check_and_reshape_cell(cell):
    if cell.dim() == 2 and cell.size(0) % 3 == 0 and cell.size(1) == 3:
        batch_size = cell.size(0) // 3
        cell = cell.reshape(batch_size, 3, 3)
    elif cell.dim() != 3 or cell.size(1) != 3 or cell.size(2) != 3:
        raise ValueError(
            "Invalid cell shape. Expected (batch_size, 3, 3), but got {}".format(
                cell.size()
            )
        )

    return cell


def radius_graph_pbc(data, radius, max_num_neighbors_threshold, pbc=[True, True, True]):
    device = data.pos.device
    batch_size = len(data.natoms)
    data.cell = check_and_reshape_cell(data.cell)
    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos
    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()
    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    # print(data.cell.shape)
    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [int(rep_a1.max()), int(rep_a2.max()), int(rep_a3.max())]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float32)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets=False,
    return_distance_vec=False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # correct for pbc
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    # redundancy: remove zero distances
    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]

    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]

    return out


## radial basis function to embed distances
## add comments: based on XXX code
class rbf_emb(nn.Module):
    """
    modified: delete cutoff with r
    """

    def __init__(self, num_rbf, rbound_upper, rbf_trainable=False):
        super().__init__()
        self.rbound_upper = rbound_upper
        self.rbound_lower = 0
        self.num_rbf = num_rbf
        self.rbf_trainable = rbf_trainable
        means, betas = self._initial_params()

        self.register_buffer("means", means)
        self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.rbound_upper))
        end_value = torch.exp(torch.scalar_tensor(-self.rbound_lower))
        means = torch.linspace(start_value, end_value, self.num_rbf)
        betas = torch.tensor(
            [(2 / self.num_rbf * (end_value - start_value)) ** -2] * self.num_rbf
        )
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        rbounds = 0.5 * (torch.cos(dist * pi / self.rbound_upper) + 1.0)
        rbounds = rbounds * (dist < self.rbound_upper).float()
        return rbounds * torch.exp(
            -self.betas * torch.square((torch.exp(-dist) - self.means))
        )


class NeighborEmb(MessagePassing):
    def __init__(self, hid_dim: int):
        super(NeighborEmb, self).__init__(aggr="add")
        self.embedding = nn.Embedding(95, hid_dim)
        self.hid_dim = hid_dim
        self.ln_emb = nn.LayerNorm(hid_dim, elementwise_affine=False)

    def forward(self, z, s, edge_index, embs):
        s_neighbors = self.ln_emb(self.embedding(z))
        s_neighbors = self.propagate(edge_index, x=s_neighbors, norm=embs)

        s = s + s_neighbors
        return s

    def message(self, x_j, norm):
        return norm.view(-1, self.hid_dim) * x_j


class S_vector(MessagePassing):
    def __init__(self, hid_dim: int):
        super(S_vector, self).__init__(aggr="add")
        self.hid_dim = hid_dim
        self.lin1 = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.LayerNorm(hid_dim, elementwise_affine=False),
            nn.SiLU(),
        )

    def forward(self, s, v, edge_index, emb):
        s = self.lin1(s)
        emb = emb.unsqueeze(1) * v

        v = self.propagate(edge_index, x=s, norm=emb)
        return v.view(-1, 3, self.hid_dim)

    def message(self, x_j, norm):
        x_j = x_j.unsqueeze(1)
        a = norm.view(-1, 3, self.hid_dim) * x_j
        return a.view(-1, 3 * self.hid_dim)


class EquiMessagePassing(MessagePassing):
    def __init__(
        self,
        hidden_channels,
        num_radial,
        hidden_channels_chi=96,
        head: int = 16,
        chi1: int = 32,
        chi2: int = 8,
        has_dropout_flag: bool = False,
        has_norm_before_flag=True,
        has_norm_after_flag=False,
        reduce_mode="sum",
        device=(
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        ),
    ):
        super(EquiMessagePassing, self).__init__(aggr="add", node_dim=0)

        self.device = device
        self.reduce_mode = reduce_mode
        self.chi1 = chi1
        self.chi2 = chi2
        self.head = head
        self.hidden_channels = hidden_channels
        self.hidden_channels_chi = hidden_channels_chi
        self.scale = nn.Linear(self.hidden_channels, self.hidden_channels_chi * 2)
        self.num_radial = num_radial
        self.dir_proj = nn.Sequential(
            nn.Linear(
                3 * self.hidden_channels + self.num_radial, self.hidden_channels * 3
            ),
            nn.SiLU(inplace=True),
            nn.Linear(self.hidden_channels * 3, self.hidden_channels * 3),
        )

        self.x_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )
        self.rbf_proj = nn.Linear(num_radial, hidden_channels * 3)
        self.x_layernorm = nn.LayerNorm(hidden_channels)
        self.diagonal = nn.Sequential(
            nn.Linear(hidden_channels * 3, hidden_channels_chi // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels_chi // 2, self.chi2),
        )
        self.has_dropout_flag = has_dropout_flag
        self.has_norm_before_flag = has_norm_before_flag
        self.has_norm_after_flag = has_norm_after_flag

        if self.has_norm_after_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1)
        if self.has_norm_before_flag:
            self.dx_layer_norm = nn.LayerNorm(self.chi1 + self.hidden_channels)

        self.dropout = nn.Dropout(p=0.5)
        self.diachi1 = torch.nn.Parameter(torch.randn((self.chi1), device=self.device))
        self.scale2 = nn.Sequential(
            nn.Linear(self.chi1, hidden_channels // 2),
        )

        self.kernel_real = torch.nn.Parameter(
            torch.randn(
                (self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2),
                device=self.device,
            )
        )
        self.kernel_imag = torch.nn.Parameter(
            torch.randn(
                (self.head + 1, (self.hidden_channels_chi) // self.head, self.chi2),
                device=self.device,
            )
        )
        self.fc_mps = nn.Linear(self.chi1, self.chi1)
        self.fc_dx = nn.Linear(self.chi1, hidden_channels)

        self.dia = nn.Linear(self.chi1, self.chi1)
        self.unitary = torch.nn.Parameter(
            torch.randn((self.chi1, self.chi1), device=self.device)
        )
        self.activation = nn.SiLU()

        self.inv_sqrt_3 = 1 / math.sqrt(3.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)
        self.x_layernorm = nn.LayerNorm(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_proj[0].weight)
        self.x_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.x_proj[2].weight)
        self.x_proj[2].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.rbf_proj.weight)
        self.rbf_proj.bias.data.fill_(0)
        self.x_layernorm.reset_parameters()
        nn.init.xavier_uniform_(self.dir_proj[0].weight)
        self.dir_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dir_proj[2].weight)
        self.dir_proj[2].bias.data.fill_(0)

    def forward(
        self, x, vec, edge_index, edge_rbf, weight, edge_vector, rope
    ):  # ,unitary):
        if rope != None:
            real, imag = torch.split(
                x, [self.hidden_channels // 2, self.hidden_channels // 2], dim=-1
            )
            dy_pre = torch.complex(real=real, imag=imag)
            dy_pre = dy_pre * rope
            x = torch.cat([dy_pre.real, dy_pre.imag], dim=-1)
        xh = self.x_proj(self.x_layernorm(x))

        rbfh = self.rbf_proj(edge_rbf)
        weight = self.dir_proj(weight)
        rbfh = rbfh * weight
        dx, dvec = self.propagate(
            edge_index, xh=xh, vec=vec, rbfh_ij=rbfh, r_ij=edge_vector, size=None
        )
        if self.has_norm_before_flag:
            dx = self.dx_layer_norm(dx)

        dx, dy = torch.split(dx, [self.chi1, self.hidden_channels], dim=-1)

        if self.has_norm_after_flag:
            dx = self.dx_layer_norm(dx)

        dx = self.scale2(dx)

        dx = torch.complex(torch.cos(dx), torch.sin(dx))

        return dx, dy, dvec

    def message(self, xh_j, vec_j, rbfh_ij, r_ij):
        x, xh2, xh3 = torch.split(xh_j * rbfh_ij, self.hidden_channels, dim=-1)
        xh2 = xh2 * self.inv_sqrt_3
        real, imagine = torch.split(self.scale(x), self.hidden_channels_chi, dim=-1)
        real = real.reshape(
            x.shape[0], self.head, (self.hidden_channels_chi) // self.head
        )
        imagine = imagine.reshape(
            x.shape[0], self.head, (self.hidden_channels_chi) // self.head
        )
        if self.has_dropout_flag:
            real = self.dropout(real)
            imagine = self.dropout(imagine)

        # complex invariant quantum state
        phi = torch.complex(real, imagine)
        q = phi
        a = torch.ones(
            q.shape[0],
            1,
            (self.hidden_channels_chi) // self.head,
            device=self.device,
            dtype=torch.complex64,
        )

        kernel = (
            torch.complex(self.kernel_real, self.kernel_imag)
            / math.sqrt((self.hidden_channels) // self.head)
        ).expand(q.shape[0], -1, -1, -1)
        equation = "ijl, ijlk->ik"
        conv = torch.einsum(
            equation, torch.cat([a, q], dim=1), kernel.to(torch.complex64)
        )

        a = 1.0 * self.activation(self.diagonal(rbfh_ij))
        b = a.unsqueeze(-1) * self.diachi1.unsqueeze(0).unsqueeze(0) + torch.ones(
            kernel.shape[0], self.chi2, self.chi1, device=self.device
        )
        dia = self.dia(b)
        equation = "ik,ikl->il"
        kernel = torch.einsum(equation, conv, dia.to(torch.complex64))
        kernel_real, kernel_imag = kernel.real, kernel.imag
        kernel_real, kernel_imag = self.fc_mps(kernel_real), self.fc_mps(kernel_imag)
        kernel = torch.angle(torch.complex(kernel_real, kernel_imag))
        agg = torch.cat([kernel, x], dim=-1)
        vec = vec_j * xh2.unsqueeze(1) + xh3.unsqueeze(1) * r_ij.unsqueeze(2)
        vec = vec * self.inv_sqrt_h

        return agg, vec

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(
            x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.reduce_mode
        )
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(
        self, inputs: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs


class FTE(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 2, bias=False)
        self.xvec_proj = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels * 3),
        )

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)
        self.inv_sqrt_h = 1 / math.sqrt(hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec_proj.weight)
        nn.init.xavier_uniform_(self.xvec_proj[0].weight)
        self.xvec_proj[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.xvec_proj[2].weight)
        self.xvec_proj[2].bias.data.fill_(0)

    def forward(self, x, vec, node_frame):
        vec = self.vec_proj(vec)
        vec1, vec2 = torch.split(vec, self.hidden_channels, dim=-1)
        scalar = torch.norm(vec1, dim=-2, p=1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = vec_dot * self.inv_sqrt_h

        x_vec_h = self.xvec_proj(torch.cat([x, scalar], dim=-1))
        xvec1, xvec2, xvec3 = torch.split(x_vec_h, self.hidden_channels, dim=-1)

        dx = xvec1 + xvec2 + vec_dot
        dx = dx * self.inv_sqrt_2

        dvec = xvec3.unsqueeze(1) * vec2

        return dx, dvec


class aggregate_pos(MessagePassing):
    def __init__(self, aggr="mean"):
        super(aggregate_pos, self).__init__(aggr=aggr)

    def forward(self, vector, edge_index):
        v = self.propagate(edge_index, x=vector)

        return v


class AlphaNet(nn.Module):
    def __init__(
        self,
        config,
        # device=torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
    ):
        super(AlphaNet, self).__init__()

        # self.device = device
        self.eps = config.eps
        self.num_layers = config.num_layers
        self.hidden_channels = config.hidden_channels
        self.a = nn.Parameter(torch.ones(108) * config.a)
        self.b = nn.Parameter(torch.ones(108) * config.b)
        self.cutoff = config.cutoff
        self.readout = config.readout
        self.chi1 = config.main_chi1
        self.pos_require_grad = config.pos_require_grad
        self.z_emb_ln = nn.LayerNorm(config.hidden_channels, elementwise_affine=False)
        self.z_emb = Embedding(95, config.hidden_channels)
        self.use_pbc = config.use_pbc
        # self.kernel1 = torch.nn.Parameter(torch.randn((config.hidden_channels, self.chi1 * 2), device=self.device))
        self.kernel1 = torch.nn.Parameter(
            torch.randn((config.hidden_channels, self.chi1 * 2))
        )
        self.kernels_real = []
        self.kernels_imag = []
        self.compute_forces = config.compute_forces
        self.compute_stress = config.compute_stress
        self.radial_emb = rbf_emb(config.num_radial, config.cutoff)
        self.radial_lin = nn.Sequential(
            nn.Linear(config.num_radial, config.hidden_channels),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels, config.hidden_channels),
        )

        self.neighbor_emb = NeighborEmb(config.hidden_channels)

        self.S_vector = S_vector(config.hidden_channels)
        self.lin = nn.Sequential(
            nn.Linear(3, config.hidden_channels // 4),
            nn.SiLU(inplace=True),
            nn.Linear(config.hidden_channels // 4, 1),
        )

        self.message_layers = nn.ModuleList()
        self.FTEs = nn.ModuleList()

        for _ in range(config.num_layers):
            self.message_layers.append(
                EquiMessagePassing(
                    hidden_channels=config.hidden_channels,
                    num_radial=config.num_radial,
                    head=config.head,
                    chi2=config.chi2,
                    chi1=config.mp_chi1,
                    has_dropout_flag=config.has_dropout_flag,
                    has_norm_before_flag=config.has_norm_before_flag,
                    has_norm_after_flag=config.has_norm_after_flag,
                    hidden_channels_chi=config.hidden_channels_chi,
                    #    device=device,
                    reduce_mode=config.reduce_mode,
                )
            )
            self.FTEs.append(FTE(config.hidden_channels))
            # kernel_real = torch.randn((config.hidden_channels, self.chi1, self.chi1), device=self.device)
            # kernel_imag = torch.randn((config.hidden_channels, self.chi1, self.chi1), device=self.device)
            kernel_real = torch.randn((config.hidden_channels, self.chi1, self.chi1))
            kernel_imag = torch.randn((config.hidden_channels, self.chi1, self.chi1))
            self.kernels_real.append(kernel_real)
            self.kernels_imag.append(kernel_imag)

        self.kernels_real = torch.nn.Parameter(torch.stack(self.kernels_real))
        self.kernels_imag = torch.nn.Parameter(torch.stack(self.kernels_imag))

        if config.output_dim != 0:
            self.num_targets = config.output_dim

        self.last_layer = nn.Linear(config.hidden_channels, self.num_targets)
        self.last_layer_quantum = nn.Linear(self.chi1 * 2, self.num_targets)

        self.mean_neighbor_pos = aggregate_pos(aggr="mean")

        self.inv_sqrt_2 = 1 / math.sqrt(2.0)

        self.reset_parameters()
        self.use_pbc = config.use_pbc
        self.use_sigmoid = config.use_sigmoid

    def reset_parameters(self):
        self.z_emb.reset_parameters()
        self.radial_emb.reset_parameters()
        for layer in self.message_layers:
            layer.reset_parameters()
        for layer in self.FTEs:
            layer.reset_parameters()
        self.last_layer.reset_parameters()

        for layer in self.radial_lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for layer in self.lin:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def _forward(self, data):
        pos = data.pos
        batch = data.batch
        z = data.z.long()

        # pos -= scatter(pos, batch, dim=0)[batch]  # centering before getting in the forward function for simplicity

        # if self.compute_forces:
        #     pos.requires_grad = True

        if self.compute_stress:
            pos, data.cell, displacement = self.get_symmetric_displacement(
                pos, data.cell, int(len(data.cell) / 3), data.batch
            )

        if self.use_pbc:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, self.cutoff, 50
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.neighbors = neighbors
            out = get_pbc_distances(
                pos,
                data.edge_index,
                data.cell,
                data.cell_offsets,
                data.neighbors,
                return_distance_vec=True,
            )
            edge_index = out["edge_index"]
            j, i = edge_index
            dist = out["distances"]
            vecs = out["distance_vec"]
        else:
            edge_index = radius_graph(
                pos, r=self.cutoff, batch=batch, max_num_neighbors=1000
            )
            j, i = edge_index
            vecs = pos[j] - pos[i]
            dist = vecs.norm(dim=-1)

        z_emb = self.z_emb_ln(self.z_emb(z))
        radial_emb = self.radial_emb(dist)
        radial_hidden = self.radial_lin(radial_emb)
        rbounds = 0.5 * (torch.cos(dist * pi / self.cutoff) + 1.0)
        radial_hidden = rbounds.unsqueeze(-1) * radial_hidden
        s = self.neighbor_emb(z, z_emb, edge_index, radial_hidden)
        vec = torch.zeros(s.size(0), 3, s.size(1), device=s.device)
        edge_diff = vecs
        edge_diff = edge_diff / (dist.unsqueeze(1) + self.eps)
        mean = scatter(pos[edge_index[0]], edge_index[1], reduce="mean", dim=0)

        # dim is chosen to be the first dimension of size 3
        edge_cross = torch.cross(pos[i] - mean[i], pos[j] - mean[i], dim=-1)
        edge_vertical = torch.cross(edge_diff, edge_cross, dim=-1)
        edge_frame = torch.cat(
            (
                edge_diff.unsqueeze(-1),
                edge_cross.unsqueeze(-1),
                edge_vertical.unsqueeze(-1),
            ),
            dim=-1,
        )

        node_frame = 0
        S_i_j = self.S_vector(s, edge_diff.unsqueeze(-1), edge_index, radial_hidden)
        scalrization1 = torch.sum(
            S_i_j[i].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1
        )
        scalrization2 = torch.sum(
            S_i_j[j].unsqueeze(2) * edge_frame.unsqueeze(-1), dim=1
        )
        scalrization1[:, 1, :] = torch.abs(scalrization1[:, 1, :].clone())
        scalrization2[:, 1, :] = torch.abs(scalrization2[:, 1, :].clone())

        scalar3 = (
            self.lin(torch.permute(scalrization1, (0, 2, 1)))
            + torch.permute(scalrization1, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1) / math.sqrt(self.hidden_channels)
        scalar4 = (
            self.lin(torch.permute(scalrization2, (0, 2, 1)))
            + torch.permute(scalrization2, (0, 2, 1))[:, :, 0].unsqueeze(2)
        ).squeeze(-1) / math.sqrt(self.hidden_channels)

        edge_weight = torch.cat((scalar3, scalar4), dim=-1) * rbounds.unsqueeze(-1)
        edge_weight = torch.cat((edge_weight, radial_hidden, radial_emb), dim=-1)
        equation = "ik,bi->bk"
        quantum = torch.einsum(equation, self.kernel1, z_emb)
        real, imagine = torch.split(quantum, self.chi1, dim=-1)
        quantum = torch.complex(real, imagine)

        for i in range(self.num_layers):
            rope = None
            if i > 0:
                rope, ds, dvec = self.message_layers[i](
                    s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope
                )
            else:
                rope, ds, dvec = self.message_layers[i](
                    s, vec, edge_index, radial_emb, edge_weight, edge_diff, rope=None
                )
            s = s + ds
            vec = vec + dvec

            equation = "ikl,bi,bl->bk"
            kerneli = torch.complex(self.kernels_real[i], self.kernels_imag[i])
            quantum = torch.einsum(equation, kerneli, s.to(torch.cfloat), quantum)
            quantum = quantum / quantum.abs().to(torch.cfloat)

            # FTE: frame transition encoding
            ds, dvec = self.FTEs[i](s, vec, node_frame)

            s = s + ds
            vec = vec + dvec

        s = (
            self.last_layer(s)
            + self.last_layer_quantum(torch.cat([quantum.real, quantum.imag], dim=-1))
            / self.chi1
        )

        z = data.z.long()

        if s.dim() == 2:
            s = self.a[z].unsqueeze(1) * s + self.b[z].unsqueeze(1)
        elif s.dim() == 1:
            s = (self.a[z] * s + self.b[z]).unsqueeze(1)
        else:
            raise ValueError(f"Unexpected shape of s: {s.shape}")

        s = scatter(s, batch, dim=0, reduce=self.readout)

        if self.use_sigmoid:
            s = torch.sigmoid((s - 0.5) * 5)

        if self.compute_forces and self.compute_stress:
            forces = self.cal_forces(s, pos)
            stress = self.cal_stress(s, displacement, data.cell)
            stress = stress.view(-1, 3)
            return s, forces, stress
        elif self.compute_forces:
            forces = self.cal_forces(s, pos)
            return s, forces
        return s

    def forward(self, data):
        return self._forward(data)

    def cal_forces(self, energy, positions):
        forces = -torch.autograd.grad(
            outputs=energy,
            inputs=positions,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        return forces

    def cal_stress(self, energy, displacement, cell):
        stress = torch.autograd.grad(
            energy,
            displacement,
            grad_outputs=torch.ones_like(energy),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]
        volume = torch.abs(torch.linalg.det(cell))
        stress = stress / volume.unsqueeze(-1).unsqueeze(-1)
        return stress

    def get_symmetric_displacement(
        self,
        positions: torch.Tensor,
        cell: Optional[torch.Tensor],
        num_graphs: int,
        batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if cell is None:
            cell = torch.zeros(
                num_graphs * 3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            )

        displacement = torch.zeros(
            (num_graphs, 3, 3),
            dtype=positions.dtype,
            device=positions.device,
        )
        displacement.requires_grad_(True)
        symmetric_displacement = 0.5 * (displacement + displacement.transpose(-1, -2))
        positions = positions + torch.einsum(
            "be,bec->bc", positions, symmetric_displacement[batch]
        )
        cell = cell.view(-1, 3, 3)
        cell = cell + torch.matmul(cell, symmetric_displacement)
        cell.view(-1, 3)
        return positions, cell, displacement

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
