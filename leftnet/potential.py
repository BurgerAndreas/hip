from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
from torch.autograd import grad

# from torch_scatter import scatter_sum, scatter_mean
from nets.scatter_utils import scatter_sum, scatter_mean
from torch_geometric.data import Data

from leftnet.model.core import GatedMLP
from leftnet.utils import (
    get_subgraph_mask,
    get_n_frag_switch,
    get_mask_for_frag,
    get_edges_index,
)
from leftnet.model import MLP, EGNN
# import pdb

FEATURE_MAPPING = ["pos", "one_hot", "charges"]


class BaseDynamics(nn.Module):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        update_pocket_coords: bool = True,
        condition_time: bool = True,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ) -> None:
        r"""Base dynamics class set up for denoising process.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        super().__init__()
        assert len(node_nfs) == len(fragment_names)
        for nf in node_nfs:
            assert nf > pos_dim
        if "act_fn" not in model_config:
            model_config["act_fn"] = "swish"
        if "in_node_nf" not in model_config:
            model_config["in_node_nf"] = model_config["in_hidden_channels"]
        self.model_config = model_config
        self.node_nfs = node_nfs
        self.edge_nf = edge_nf
        self.condition_nf = condition_nf
        self.fragment_names = fragment_names
        self.pos_dim = pos_dim
        self.update_pocket_coords = update_pocket_coords
        self.condition_time = condition_time
        self.edge_cutoff = edge_cutoff
        self.device = device

        if model is None:
            model = EGNN
        self.model = model(**model_config)
        print(f"leftnet.potential set model {model.__class__.__name__}")
        if source is not None and "model" in source:
            self.model.load_state_dict(source["model"])
        self.dist_dim = self.model.dist_dim if hasattr(self.model, "dist_dim") else 0

        self.embed_dim = model_config["in_node_nf"]
        self.edge_embed_dim = (
            model_config["in_edge_nf"] if "in_edge_nf" in model_config else 0
        )
        if condition_time:
            self.embed_dim -= 1
        if condition_nf > 0:
            self.embed_dim -= condition_nf
        assert self.embed_dim > 0

        self.build_encoders_decoders(enforce_same_encoding, source)
        del source

    def build_encoders_decoders(
        self,
        enfoce_name_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
    ):
        r"""Build encoders and decoders for nodes and edges."""
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for ii, name in enumerate(self.fragment_names):
            self.encoders.append(
                MLP(
                    in_dim=self.node_nfs[ii] - self.pos_dim,
                    out_dims=[2 * (self.node_nfs[ii] - self.pos_dim), self.embed_dim],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
            self.decoders.append(
                MLP(
                    in_dim=self.embed_dim,
                    out_dims=[
                        2 * (self.node_nfs[ii] - self.pos_dim),
                        self.node_nfs[ii] - self.pos_dim,
                    ],
                    activation=self.model_config["act_fn"],
                    last_layer_no_activation=True,
                )
            )
        if enfoce_name_encoding is not None:
            for ii in enfoce_name_encoding:
                self.encoders[ii] = self.encoders[0]
                self.decoders[ii] = self.decoders[0]
        if source is not None and "encoders" in source:
            self.encoders.load_state_dict(source["encoders"])
            self.decoders.load_state_dict(source["decoders"])

        if self.edge_embed_dim > 0:
            self.edge_encoder = MLP(
                in_dim=self.edge_nf,
                out_dims=[2 * self.edge_nf, self.edge_embed_dim],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
            self.edge_decoder = MLP(
                in_dim=self.edge_embed_dim + self.dist_dim,
                out_dims=[2 * self.edge_nf, self.edge_nf],
                activation=self.model_config["act_fn"],
                last_layer_no_activation=True,
            )
        else:
            self.edge_encoder, self.edge_decoder = None, None

    def forward(self):
        raise NotImplementedError


def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x


class Potential(BaseDynamics):
    def __init__(
        self,
        model_config: Dict,
        fragment_names: List[str],
        node_nfs: List[int],
        edge_nf: int,
        condition_nf: int = 0,
        pos_dim: int = 3,
        edge_cutoff: Optional[float] = None,
        model: nn.Module = EGNN,
        device: torch.device = torch.device("cuda"),
        enforce_same_encoding: Optional[List] = None,
        source: Optional[Dict] = None,
        timesteps: int = 5000,
        condition_time: bool = True,
        **kwargs,
    ) -> None:
        r"""Confindence score for generated samples.

        Args:
            model_config (Dict): config for the equivariant model.
            fragment_names (List[str]): list of names for fragments
            node_nfs (List[int]): list of number of input node attributues.
            edge_nf (int): number of input edge attributes.
            condition_nf (int): number of attributes for conditional generation.
            Defaults to 0.
            pos_dim (int): dimension for position vector. Defaults to 3.
            update_pocket_coords (bool): whether to update positions of everything.
                Defaults to True.
            condition_time (bool): whether to condition on time. Defaults to True.
            edge_cutoff (Optional[float]): cutoff for building intra-fragment edges.
                Defaults to None.
            model (Optional[nn.Module]): Module for equivariant model. Defaults to None.
        """
        model_config.update({"for_conf": False, "ff": True})
        update_pocket_coords = True
        super().__init__(
            model_config,
            fragment_names,
            node_nfs,
            edge_nf,
            condition_nf,
            pos_dim,
            update_pocket_coords,
            condition_time,
            edge_cutoff,
            model,
            device,
            enforce_same_encoding,
            source=source,
        )

        hidden_channels = model_config["hidden_channels"]
        self.readout = GatedMLP(
            in_dim=hidden_channels,
            out_dims=[hidden_channels, hidden_channels, 1],
            activation="swish",
            bias=True,
            last_layer_no_activation=True,
        )
        self.timesteps = timesteps

    def _forward(
        self,
        xh: List[Tensor],
        edge_index: Tensor,
        t: Tensor,
        conditions: Tensor,
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""predict confidence.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tensor: binary probability of confidence fo each graph.
        """
        pos = torch.concat(
            [_xh[:, : self.pos_dim] for _xh in xh],
            dim=0,
        )  # [N, 3]
        h = torch.concat(
            [
                self.encoders[ii](xh[ii][:, self.pos_dim :])
                for ii, name in enumerate(self.fragment_names)
            ],
            dim=0,
        )
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        condition_dim = 0
        if self.condition_time:
            if len(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[combined_mask]
            h = torch.cat([h, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.

        # [N, C], [N, 3]
        node_features, forces = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
        )  # (n_node, n_hidden)

        node_features = self.readout(node_features)
        ae = scatter_sum(
            node_features,
            index=combined_mask,
            dim=0,
        )  # (n_system, n_hidden)
        return ae.squeeze(), forces

    def forward_diffusion(self, batch, natoms, pos, one_hot, charges):
        masks = [batch]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [natoms]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        conditions = torch.zeros(natoms.size(0), 1, dtype=torch.long)
        conditions = conditions.to(batch.device)

        pos = remove_mean_batch(pos, batch)
        pos.requires_grad_(True)

        xh = [
            torch.cat(
                [one_hot, charges.view(-1, 1)],
                dim=1,
            ).float()
        ]

        t = torch.randint(0, self.timesteps, size=(1,)) / self.timesteps

        ae, forces = self._forward(
            xh=xh,
            edge_index=edge_index,
            t=torch.tensor([0.0]),
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        return ae, forces

    def forward(
        self,
        pyg_batch: Data,
        conditions: Optional[Tensor] = None,
        **kwargs
    ):
        masks = [pyg_batch.batch]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [pyg_batch.natoms]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        conditions = conditions or torch.zeros(
            pyg_batch.ae.size(0), 1, dtype=torch.long
        )
        conditions = conditions.to(pyg_batch.batch.device)

        pyg_batch.pos = remove_mean_batch(pyg_batch.pos, pyg_batch.batch)

        xh = [
            torch.cat(
                [pyg_batch.pos, pyg_batch.one_hot, pyg_batch.charges.view(-1, 1)],
                dim=1,
            )
        ]

        t = torch.randint(0, self.timesteps, size=(1,)) / self.timesteps

        ae, forces = self._forward(
            xh=xh,
            edge_index=edge_index,
            t=torch.tensor([0.0]),
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        return ae, forces

    def _forward_autograd(
        self,
        h: List[Tensor],
        pos: Tensor,
        edge_index: Tensor,
        t: Tensor,
        conditions: Tensor,
        n_frag_switch: Tensor,
        combined_mask: Tensor,
        edge_attr: Optional[Tensor] = None,
    ) -> Tensor:
        r"""predict confidence.

        Args:
            xh (List[Tensor]): list of concatenated tensors for pos and h
            edge_index (Tensor): [n_edge, 2]
            t (Tensor): time tensor. If dim is 1, same for all samples;
                otherwise different t for different samples
            conditions (Tensor): condition tensors
            n_frag_switch (Tensor): [n_nodes], fragment index for each nodes
            combined_mask (Tensor): [n_nodes], sample index for each node
            edge_attr (Optional[Tensor]): [n_edge, dim_edge_attribute]. Defaults to None.

        Raises:
            NotImplementedError: The fragement-position-fixed mode is not implement.

        Returns:
            Tensor: binary probability of confidence fo each graph.
        """
        h = torch.concat(
            [self.encoders[ii](h[ii]) for ii, name in enumerate(self.fragment_names)],
            dim=0,
        )
        if self.edge_encoder is not None:
            edge_attr = self.edge_encoder(edge_attr)

        condition_dim = 0
        if self.condition_time:
            if len(t.size()) == 1:
                # t is the same for all elements in batch.
                h_time = torch.empty_like(h[:, 0:1]).fill_(t.item())
            else:
                # t is different over the batch dimension.
                h_time = t[combined_mask]
            h = torch.cat([h, h_time], dim=1)
            condition_dim += 1

        if self.condition_nf > 0:
            h_condition = conditions[combined_mask]
            h = torch.cat([h, h_condition], dim=1)
            condition_dim += self.condition_nf

        subgraph_mask = get_subgraph_mask(edge_index, n_frag_switch)
        if self.update_pocket_coords:
            update_coords_mask = None
        else:
            raise NotImplementedError  # no need to mask pos for inpainting mode.

        node_features, forces = self.model(
            h,
            pos,
            edge_index,
            edge_attr,
            node_mask=None,
            edge_mask=None,
            update_coords_mask=update_coords_mask,
            subgraph_mask=subgraph_mask[:, None],
        )  # (n_node, n_hidden)

        node_features = self.readout(node_features)
        ae = scatter_sum(
            node_features,
            index=combined_mask,
            dim=0,
        )  # (n_system, n_hidden)
        return ae.squeeze(), forces

    @torch.enable_grad()
    def forward_autograd(
        self,
        pyg_batch: Data,
        conditions: Optional[Tensor] = None,
    ):
        masks = [pyg_batch.batch]
        combined_mask = torch.cat(masks)
        edge_index = get_edges_index(combined_mask, remove_self_edge=True)
        fragments_nodes = [pyg_batch.natoms]
        n_frag_switch = get_n_frag_switch(fragments_nodes)
        conditions = conditions or torch.zeros(
            pyg_batch.ae.size(0), 1, dtype=torch.long
        )
        conditions = conditions.to(pyg_batch.batch.device)

        pyg_batch.pos = remove_mean_batch(pyg_batch.pos, pyg_batch.batch)
        pyg_batch.pos.requires_grad_(True)

        h = [
            torch.cat(
                [pyg_batch.one_hot, pyg_batch.charges.view(-1, 1)],
                dim=1,
            ).float()
        ]

        t = torch.randint(0, self.timesteps, size=(1,)) / self.timesteps

        ae, forces = self._forward_autograd(
            h=h,
            pos=pyg_batch.pos,
            edge_index=edge_index,
            t=torch.tensor([0.0]),
            conditions=conditions,
            n_frag_switch=n_frag_switch,
            combined_mask=combined_mask,
            edge_attr=None,
        )
        forces = -grad(torch.sum(ae), pyg_batch.pos, create_graph=True)[0]
        return ae, forces
