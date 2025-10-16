import logging
import time
import math
import torch
import einops

from torch.autograd import grad
from ocpmodels.common.registry import registry
from ocpmodels.models.base import BaseModel
from ocpmodels.models.scn.smearing import (
    GaussianSmearing,
    # LinearSigmoidSmearing,
    # SigmoidSmearing,
    # SiLUSmearing,
)
from ocpmodels.common.utils import (
    compute_neighbors,
    conditional_grad,
    get_pbc_distances,
    radius_graph_pbc,
)
from nets.scatter_utils import scatter_mean
import torch_geometric
from torch_geometric.nn import radius_graph

from .gaussian_rbf import GaussianRadialBasisLayer
from .edge_rot_mat import init_edge_rot_mat
from .so3 import (
    CoefficientMappingModule,
    SO3_Embedding,
    SO3_Grid,
    SO3_Rotation,
    SO3_LinearV2,
)
from .module_list import ModuleListInfo

# from .so2_ops import SO2_Convolution
from .radial_function import RadialFunction
from .layer_norm import (
    EquivariantLayerNormArray,
    EquivariantLayerNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonics,
    EquivariantRMSNormArraySphericalHarmonicsV2,
    get_normalization_layer,
)
from .transformer_block import (
    SO2EquivariantGraphAttention,
    FeedForwardNetwork,
    TransBlockV2,
)
from .input_block import EdgeDegreeEmbedding
from .hessian_pred_utils import (
    add_extra_props_for_hessian,
    l012_features_to_hessian,
    irreps_to_cartesian_matrix,
    add_graph_batch,
    # _get_indexadd_offdiagonal_to_flat_hessian_message_indices,
    # _get_node_diagonal_1d_indexadd_indices,
)

# Statistics of IS2RE 100K
# _AVG_NUM_NODES = 77.81317
_AVG_DEGREE = 23.395238876342773  # IS2RE: 100k, max_radius = 5, max_neighbors = 100

# HORM T1x
# Number of edges: 176.7360948021343
# Number of edges (hessian): 186.7841774653667
# Number of atoms: 13.919938540433833
# _AVG_DEGREE_HESSIAN = 13.4184627987
# _AVG_DEGREE = 12.6966145927


# l0_features = x_message.embedding.narrow(dimension=1, start=0, length=1)
# l1_features = x_message.embedding.narrow(dimension=1, start=1, length=3)
# l2_features = x_message.embedding.narrow(dim=1, start=4, length=5) # length=2l+1
def get_scalar_from_embedding(embedding, data, avg_num_nodes=None):
    # get l=0 component
    embedding = embedding.embedding.narrow(1, 0, 1)
    scalars = torch.zeros(
        len(data.natoms), device=embedding.device, dtype=embedding.dtype
    )
    # sum over all nodes in the batch
    scalars.index_add_(0, data.batch, embedding.view(-1))
    # avg number of nodes=atoms in the batch
    if avg_num_nodes is None:
        avg_num_nodes = torch.sum(data.natoms) / len(data.natoms)
    return scalars / avg_num_nodes

def remove_mean_batch(x, indices):
    mean = scatter_mean(x, indices, dim=0)
    x = x - mean[indices]
    return x

@registry.register_model("equiformer_v2")
class EquiformerV2_OC20(BaseModel):
    """
    Equiformer with graph attention built upon SO(2) convolution and feedforward network built upon S2 activation

    Args:
        use_pbc (bool):         Use periodic boundary conditions
        regress_forces (bool):  Compute forces
        otf_graph (bool):       Compute graph On The Fly (OTF)
        max_neighbors (int):    Maximum number of neighbors per atom
        max_radius (float):     Maximum distance between nieghboring atoms in Angstroms
        max_num_elements (int): Maximum atomic number

        num_layers (int):             Number of layers in the GNN
        sphere_channels (int):        Number of spherical channels (one set per resolution)
        attn_hidden_channels (int): Number of hidden channels used during SO(2) graph attention
        num_heads (int):            Number of attention heads
        attn_alpha_head (int):      Number of channels for alpha vector in each attention head
        attn_value_head (int):      Number of channels for value vector in each attention head
        ffn_hidden_channels (int):  Number of hidden channels used during feedforward network
        norm_type (str):            Type of normalization layer (['layer_norm', 'layer_norm_sh', 'rms_norm_sh'])

        lmax_list (int):              List of maximum degree of the spherical harmonics (1 to 10)
        mmax_list (int):              List of maximum order of the spherical harmonics (0 to lmax)
        grid_resolution (int):        Resolution of SO3_Grid

        num_sphere_samples (int):     Number of samples used to approximate the integration of the sphere in the output blocks

        edge_channels (int):                Number of channels for the edge invariant features
        use_atom_edge_embedding (bool):     Whether to use atomic embedding along with relative distance for edge scalar features
        share_atom_edge_embedding (bool):   Whether to share `atom_edge_embedding` across all blocks
        use_m_share_rad (bool):             Whether all m components within a type-L vector of one channel share radial function weights
        distance_function ("gaussian", "sigmoid", "linearsigmoid", "silu"):  Basis function used for distances

        attn_activation (str):      Type of activation function for SO(2) graph attention
        use_s2_act_attn (bool):     Whether to use attention after S2 activation. Otherwise, use the same attention as Equiformer
        use_attn_renorm (bool):     Whether to re-normalize attention weights
        ffn_activation (str):       Type of activation function for feedforward network
        use_gate_act (bool):        If `True`, use gate activation. Otherwise, use S2 activation
        use_grid_mlp (bool):        If `True`, use projecting to grids and performing MLPs for FFNs.
        use_sep_s2_act (bool):      If `True`, use separable S2 activation when `use_gate_act` is False.

        alpha_drop (float):         Dropout rate for attention weights
        drop_path_rate (float):     Drop path rate
        proj_drop (float):          Dropout rate for outputs of attention and FFN in Transformer blocks

        weight_init (str):          ['normal', 'uniform'] initialization of weights of linear layers except those in radial functions
    """

    def __init__(
        self,
        # num_atoms,      # not used
        # bond_feat_dim,  # not used
        # num_targets,    # not used
        use_pbc=True,
        regress_forces=True,
        otf_graph=True,
        max_neighbors=500,
        max_radius=5.0,
        max_num_elements=90,
        num_layers=12,
        sphere_channels=128,
        attn_hidden_channels=128,
        num_heads=8,
        attn_alpha_channels=32,
        attn_value_channels=16,
        ffn_hidden_channels=512,
        norm_type="rms_norm_sh",
        lmax_list=[6],
        mmax_list=[2],
        grid_resolution=None,
        num_sphere_samples=128,
        edge_channels=128,
        use_atom_edge_embedding=True,
        share_atom_edge_embedding=False,
        use_m_share_rad=False,
        distance_function="gaussian",
        num_distance_basis=512,
        attn_activation="scaled_silu",
        use_s2_act_attn=False,
        use_attn_renorm=True,
        ffn_activation="scaled_silu",
        use_gate_act=False,
        use_grid_mlp=False,
        use_sep_s2_act=True,
        alpha_drop=0.1,
        drop_path_rate=0.05,
        proj_drop=0.0,
        weight_init="normal",
        avg_degree=_AVG_DEGREE,  # rescale factor for edge_degree_embedding
        # added for hessian prediction
        avg_degree_hessian=_AVG_DEGREE,
        avg_num_nodes=None,
        hessian_alpha_drop=0.0,
        num_layers_hessian=0,
        share_atom_edge_embedding_hessian=False,
        # if to also use atom type embedding or just relative distances for edge features
        # in edge_distance
        use_atom_edge_embedding_hessian=True,
        reuse_source_target_embedding_hessian=True,
        reinit_edge_degree_embedding_hessian=False,
        cutoff_hessian=100.0,
        hessian_no_attn_weights=False,  # messages without attention weights
        attn_wo_sigmoid=False,  # do not apply sigmoid to attention weights
        # not used, for compatibilit with old  with legacy ckpt
        name=None,
        num_targets=None,
        output_dim=None,
        readout=None,
        direct_forces=None,
        eps=None,
        hidden_channels=None,
        cutoff=None,
        pos_require_grad=None,
        num_radial=None,
        use_sigmoid=None,
        head=None,
        a=None,
        b=None,
        main_chi1=None,
        mp_chi1=None,
        chi2=None,
        hidden_channels_chi=None,
        has_dropout_flag=None,
        has_norm_before_flag=None,
        has_norm_after_flag=None,
        reduce_mode=None,
        compute_forces=None,
        compute_stress=None,
        do_hessian=None,
        num_gaussians_distance_hessian=None,
        hessian_build_method=None,
        hessian_drop_path_rate=0.05,
        hessian_proj_drop=0.0,
        device=None,
        **kwargs,
    ):
        super().__init__()

        self.use_pbc = use_pbc
        self.regress_forces = regress_forces
        self.otf_graph = otf_graph
        self.max_neighbors = max_neighbors
        self.max_radius = max_radius
        self.cutoff = max_radius
        if cutoff is not None:
            print(
                f"{self.__class__.__name__}: got cutoff {cutoff} and radius {max_radius}"
            )
        self.max_num_elements = max_num_elements
        self.avg_degree = avg_degree
        self.avg_num_nodes = avg_num_nodes

        self.num_layers = num_layers
        self.sphere_channels = sphere_channels
        self.attn_hidden_channels = attn_hidden_channels
        self.num_heads = num_heads
        self.attn_alpha_channels = attn_alpha_channels
        self.attn_value_channels = attn_value_channels
        self.ffn_hidden_channels = ffn_hidden_channels
        self.norm_type = norm_type

        self.lmax_list = lmax_list
        self.mmax_list = mmax_list
        self.grid_resolution = grid_resolution

        self.num_sphere_samples = num_sphere_samples

        self.edge_channels = edge_channels
        self.use_atom_edge_embedding = use_atom_edge_embedding
        self.share_atom_edge_embedding = share_atom_edge_embedding
        if self.share_atom_edge_embedding:
            assert self.use_atom_edge_embedding
            self.block_use_atom_edge_embedding = False
        else:
            # True in HORM
            # every transformer block will create their own atom number embedding
            self.block_use_atom_edge_embedding = self.use_atom_edge_embedding
        self.use_m_share_rad = use_m_share_rad
        self.distance_function = distance_function
        self.num_distance_basis = num_distance_basis

        self.attn_activation = attn_activation
        self.use_s2_act_attn = use_s2_act_attn
        self.use_attn_renorm = use_attn_renorm
        self.ffn_activation = ffn_activation
        self.use_gate_act = use_gate_act
        self.use_grid_mlp = use_grid_mlp
        self.use_sep_s2_act = use_sep_s2_act

        self.alpha_drop = alpha_drop
        self.drop_path_rate = drop_path_rate
        self.proj_drop = proj_drop
        self.hessian_alpha_drop = hessian_alpha_drop
        self.hessian_drop_path_rate = hessian_drop_path_rate
        self.hessian_proj_drop = hessian_proj_drop
        self.hessian_no_attn_weights = hessian_no_attn_weights
        self.attn_wo_sigmoid = attn_wo_sigmoid

        self.weight_init = weight_init
        assert self.weight_init in ["normal", "uniform"]

        self.device = torch.cuda.current_device()

        self.grad_forces = False
        self.num_resolutions = len(self.lmax_list)
        self.sphere_channels_all = self.num_resolutions * self.sphere_channels

        # Weights for message initialization
        self.sphere_embedding = torch.nn.Embedding(
            self.max_num_elements, self.sphere_channels_all
        )

        # Initialize the function used to measure the distances between atoms
        assert self.distance_function in [
            "gaussian",
        ]
        if self.distance_function == "gaussian":
            self.distance_expansion = GaussianSmearing(
                start=0.0,
                stop=self.cutoff,
                num_gaussians=600,
                basis_width_scalar=2.0,
            )
            # self.distance_expansion = GaussianRadialBasisLayer(num_basis=self.num_distance_basis, cutoff=self.max_radius)
        else:
            raise ValueError

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list = [int(self.distance_expansion.num_output)] + [
            self.edge_channels
        ] * 2

        # Initialize atom edge embedding
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            self.source_embedding = torch.nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.target_embedding = torch.nn.Embedding(
                self.max_num_elements, self.edge_channels_list[-1]
            )
            self.edge_channels_list[0] = (
                self.edge_channels_list[0] + 2 * self.edge_channels_list[-1]
            )
        else:
            self.source_embedding, self.target_embedding = None, None

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation = torch.nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation.append(SO3_Rotation(self.lmax_list[i]))

        # Initialize conversion between degree l and order m layouts
        self.mappingReduced = CoefficientMappingModule(self.lmax_list, self.mmax_list)

        # Initialize the transformations between spherical and grid representations
        self.SO3_grid = ModuleListInfo(
            f"({max(self.lmax_list)}, {max(self.lmax_list)})"
        )
        for l in range(max(self.lmax_list) + 1):
            SO3_m_grid = torch.nn.ModuleList()
            for m in range(max(self.lmax_list) + 1):
                SO3_m_grid.append(
                    SO3_Grid(
                        l, m, resolution=self.grid_resolution, normalization="component"
                    )
                )
            self.SO3_grid.append(SO3_m_grid)

        # Edge-degree embedding for initializing node features
        self.edge_degree_embedding = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list,
            self.block_use_atom_edge_embedding,
            rescale_factor=avg_degree,
        )

        # Initialize the blocks for each layer of EquiformerV2
        self.blocks = torch.nn.ModuleList()
        for i in range(self.num_layers):
            block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.alpha_drop,
                self.drop_path_rate,
                self.proj_drop,
            )
            self.blocks.append(block)

        # Output blocks for energy and forces
        self.norm = get_normalization_layer(
            self.norm_type, lmax=max(self.lmax_list), num_channels=self.sphere_channels
        )
        self.energy_block = FeedForwardNetwork(
            self.sphere_channels,
            self.ffn_hidden_channels,
            1,
            self.lmax_list,
            self.mmax_list,
            self.SO3_grid,
            self.ffn_activation,
            self.use_gate_act,
            self.use_grid_mlp,
            self.use_sep_s2_act,
        )
        if self.regress_forces:
            self.force_block = SO2EquivariantGraphAttention(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                1,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list,
                self.block_use_atom_edge_embedding,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.use_gate_act,
                self.use_sep_s2_act,
                alpha_drop=0.0,
            )

        ################################################################
        # Add extra head for Hessian prediction
        ################################################################

        self.cutoff_hessian = cutoff_hessian
        self.hessian_module_list = []
        # if to also use atom type embedding or just relative distances for edge features
        # in edge_distance
        self.use_atom_edge_embedding_hessian = use_atom_edge_embedding_hessian
        # if False, every transformer block has their own atom type embedding (default True)
        self.share_atom_edge_embedding_hessian = share_atom_edge_embedding_hessian
        if self.share_atom_edge_embedding_hessian:
            assert self.use_atom_edge_embedding_hessian
            self.block_use_atom_edge_embedding_hessian = False
        else:
            # True in HORM
            # every transformer block will create their own atom number embedding
            self.block_use_atom_edge_embedding_hessian = (
                self.use_atom_edge_embedding_hessian
            )

        # Initialize the module that compute WignerD matrices and other values for spherical harmonic calculations
        self.SO3_rotation_hessian = torch.nn.ModuleList()
        for i in range(self.num_resolutions):
            self.SO3_rotation_hessian.append(SO3_Rotation(self.lmax_list[i]))
        # self.hessian_module_list.append("SO3_rotation_hessian") # no trainable parameters

        self.distance_expansion_hessian = GaussianSmearing(
            start=0.0,
            stop=self.cutoff_hessian,
            num_gaussians=600,  # 600,
            basis_width_scalar=2.0,
        )
        # self.hessian_module_list.append("distance_expansion_hessian") # no trainable parameters

        # Initialize the sizes of radial functions (input channels and 2 hidden channels)
        self.edge_channels_list_hessian = [
            int(self.distance_expansion_hessian.num_output)
        ] + [self.edge_channels] * 2

        self.reuse_source_target_embedding_hessian = (
            reuse_source_target_embedding_hessian
        )
        if reuse_source_target_embedding_hessian:
            # if we are using the same embedding modules
            # make sure we use the same embedding settings as the backbone
            assert (
                self.share_atom_edge_embedding_hessian == self.share_atom_edge_embedding
            )
            assert self.use_atom_edge_embedding_hessian == self.use_atom_edge_embedding
            self.source_embedding_hessian = self.source_embedding
            self.target_embedding_hessian = self.target_embedding
        else:
            # Initialize atom edge embedding
            if (
                self.share_atom_edge_embedding_hessian
                and self.use_atom_edge_embedding_hessian
            ):
                self.source_embedding_hessian = torch.nn.Embedding(
                    self.max_num_elements, self.edge_channels_list[-1]
                )
                self.target_embedding_hessian = torch.nn.Embedding(
                    self.max_num_elements, self.edge_channels_list[-1]
                )
                self.edge_channels_list_hessian[0] = (
                    self.edge_channels_list_hessian[0]
                    + 2 * self.edge_channels_list_hessian[-1]
                )
                self.hessian_module_list.append("source_embedding_hessian")
                self.hessian_module_list.append("target_embedding_hessian")
            else:
                self.source_embedding_hessian, self.target_embedding_hessian = (
                    None,
                    None,
                )

        self.avg_degree_hessian = avg_degree_hessian
        self.reinit_edge_degree_embedding_hessian = reinit_edge_degree_embedding_hessian
        # only used if reinit_edge_degree_embedding_hessian is True
        # Edge-degree embedding for initializing node features
        self.edge_degree_embedding_hessian = EdgeDegreeEmbedding(
            self.sphere_channels,
            self.lmax_list,
            self.mmax_list,
            self.SO3_rotation_hessian,
            self.mappingReduced,
            self.max_num_elements,
            self.edge_channels_list_hessian,
            self.block_use_atom_edge_embedding_hessian,
            rescale_factor=avg_degree_hessian,
        )
        self.hessian_module_list.append("edge_degree_embedding_hessian")

        self.hessian_module_list += [
            "hessian_layers",
            "hessian_head",
            "hessian_edge_message_proj",
            "hessian_node_proj",
        ]
        # Initialize the blocks for each layer of EquiformerV2
        self.hessian_layers = torch.nn.ModuleList()
        self.num_layers_hessian = num_layers_hessian
        for i in range(self.num_layers_hessian):
            hessian_block = TransBlockV2(
                self.sphere_channels,
                self.attn_hidden_channels,
                self.num_heads,
                self.attn_alpha_channels,
                self.attn_value_channels,
                self.ffn_hidden_channels,
                self.sphere_channels,
                self.lmax_list,
                self.mmax_list,
                self.SO3_rotation_hessian,
                self.mappingReduced,
                self.SO3_grid,
                self.max_num_elements,
                self.edge_channels_list_hessian,
                self.block_use_atom_edge_embedding_hessian,
                self.use_m_share_rad,
                self.attn_activation,
                self.use_s2_act_attn,
                self.use_attn_renorm,
                self.ffn_activation,
                self.use_gate_act,
                self.use_grid_mlp,
                self.use_sep_s2_act,
                self.norm_type,
                self.hessian_alpha_drop,
                self.hessian_drop_path_rate,
                self.hessian_proj_drop,
            )
            self.hessian_layers.append(hessian_block)
        # copied from force prediction head
        self.hessian_head = SO2EquivariantGraphAttention(
            sphere_channels=self.sphere_channels,
            hidden_channels=self.attn_hidden_channels,
            num_heads=self.num_heads,
            attn_alpha_channels=self.attn_alpha_channels,
            attn_value_channels=self.attn_value_channels,
            # different output_channels affects the linear projection after aggregating the messages
            # not relevant for us
            output_channels=1,  # self.sphere_channels
            lmax_list=self.lmax_list,
            mmax_list=self.mmax_list,
            SO3_rotation=self.SO3_rotation_hessian,
            mappingReduced=self.mappingReduced,
            SO3_grid=self.SO3_grid,
            max_num_elements=self.max_num_elements,
            edge_channels_list=self.edge_channels_list_hessian,
            # Whether to use atomic embedding for edge scalar features
            # or just relative distance
            # different: use_atom_edge_embedding vs block_use_atom_edge_embedding
            use_atom_edge_embedding=self.block_use_atom_edge_embedding_hessian,
            use_m_share_rad=self.use_m_share_rad,
            activation=self.attn_activation,
            use_s2_act_attn=self.use_s2_act_attn,  # ?
            use_attn_renorm=self.use_attn_renorm,  # True
            use_gate_act=self.use_gate_act,  # False -> use S2 activation
            use_sep_s2_act=self.use_sep_s2_act,  # True -> use Separable S2 activation
            alpha_drop=self.hessian_alpha_drop,
        )
        self.hessian_edge_message_proj = SO3_LinearV2(
            in_features=self.num_heads * self.attn_value_channels,
            out_features=1,
            lmax=2,
        )
        self.hessian_node_proj = SO3_LinearV2(
            in_features=self.sphere_channels,
            out_features=1,
            lmax=2,
        )
        self._get_hessian_from_features = l012_features_to_hessian

        self.apply(self._init_weights)
        self.apply(self._uniform_init_rad_func_linear_weights)

    def grad_hess_ij(self, energy, posj, posi, create_graph=True):
        """Calculating the inter-atomic part of hessian matrices.Find the cross-derivative for the coordinates
        of atom i and atom j that interact on the interaction layer.
        require out_type='scalar' and grad_type='Hij' and sclar_outsize=1 and irreps_out=None
        """
        fj = -grad([torch.sum(energy)], [posj], create_graph=create_graph)[0]
        Hji = torch.zeros((fj.shape[0], 3, 3), device=fj.device)
        for i in range(3):
            gji = -grad(
                [fj[:, i].sum()], [posi], create_graph=create_graph, retain_graph=True
            )[0]
            Hji[:, i] = gji
        return Hji

    @conditional_grad(torch.enable_grad())
    def generate_graph(
        self,
        data: torch_geometric.data.Batch,
        cutoff=None,
        max_neighbors=None,
        use_pbc=None,
        otf_graph=None,
    ):
        cutoff = cutoff or self.cutoff
        max_neighbors = max_neighbors or self.max_neighbors
        use_pbc = use_pbc or self.use_pbc
        otf_graph = otf_graph or self.otf_graph

        if not otf_graph:
            try:
                edge_index = data.edge_index

                if use_pbc:
                    cell_offsets = data.cell_offsets
                    neighbors = data.neighbors

            except AttributeError:
                logging.warning(
                    "Turning otf_graph=True as required attributes not present in data object"
                )
                otf_graph = True

        if use_pbc:
            if otf_graph:
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
            if otf_graph:
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
            cell_offset_distances = torch.zeros_like(
                cell_offsets, device=data.pos.device
            )
            neighbors = compute_neighbors(data, edge_index)

        return (
            edge_index,
            edge_dist,
            distance_vec,
            cell_offsets,
            cell_offset_distances,
            neighbors,
        )

    def generate_graph_nopbc(self, data, otf_graph=None, cutoff=None):
        """Simplified graph generation without periodic boundary conditions.
        Used by HORM.
        Not sure why, maybe it is easier to differentiate through for autograd hessian?
        """
        otf_graph = otf_graph or self.otf_graph
        cutoff = cutoff or self.cutoff
        if otf_graph or not hasattr(data, 'edge_distance'):
            pos = data.pos
            edge_index = radius_graph(pos, r=cutoff, batch=data.batch)
            j, i = edge_index
            posj = pos[j]
            posi = pos[i]
            vecs = posj - posi
            edge_distance_vec = vecs
            edge_distance = (vecs).norm(dim=-1)
        else:
            edge_index = data.edge_index
            edge_distance = data.edge_distance
            edge_distance_vec = data.edge_distance_vec
            # cell_offsets = data.cell_offsets
            # cell_offset_distances = data.cell_offset_distances
            # neighbors = data.neighbors
        return edge_index, edge_distance, edge_distance_vec

    def forward(
        self,
        data: torch_geometric.data.Batch,
        hessian=True,
        return_l_features=False,
        otf_graph=None,  # will default to self.otf_graph
        return_sparse_hessian=False,
        **kwargs,
    ):
        """
        hessian=True means direct prediction of the Hessian.

        Only pass otf_graph=True if you want to do Hessian from autograd on forces (no hessian prediction)!

        Returns:
            energy: (N*B,)
            forces: (N*B, 3)
            outputs (Optional):
                hessian: (B*N*3*N*3)
        """
        data.pos = remove_mean_batch(data.pos, data.batch)
        
        self.batch_size = len(data.natoms)
        self.dtype = data.pos.dtype
        self.device = data.pos.device

        atomic_numbers = data.z.long()
        num_atoms = len(atomic_numbers)

        # # cell_offsets, cell_offset_distances, neighbors are not used in EquiformerV2
        # (
        #     edge_index,
        #     edge_distance,
        #     edge_distance_vec,
        #     _,  # cell_offsets,
        #     _,  # cell offset distances
        #     _,  # neighbors,
        # ) = self.generate_graph(data)

        (
            edge_index,  # [E, 2]
            edge_distance,  # [E]
            edge_distance_vec,  # [E, 3]
        ) = self.generate_graph_nopbc(data, otf_graph=otf_graph)

        if otf_graph or not hasattr(data, 'nedges_hessian'):
            # For Hessian prediction
            data = add_graph_batch(
                data,
                cutoff=self.cutoff,
                max_neighbors=self.max_neighbors,
                use_pbc=self.use_pbc,
            )
        else:
            data = add_extra_props_for_hessian(data)

        ###############################################################
        # Initialize data structures
        ###############################################################

        # Compute 3x3 rotation matrix per edge [E, 3, 3]
        assert edge_distance_vec.numel() > 0, (
            f"edge_distance_vec is empty. edge_index: {edge_index.shape}, edge_distance: {edge_distance}. "
            "Maybe you passed a data object instead of a batch. Use `from torch_geometric.data import Batch, Dataloader` to create a batch. "
            "Or the atoms are so far apart, that there are no edges."
        )
        edge_rot_mat = self._init_edge_rot_mat(data, edge_index, edge_distance_vec)

        # Initialize the WignerD matrices and other values for spherical harmonic calculations
        # used in edge_degree_embedding, TransBlockV2
        for i in range(self.num_resolutions):
            self.SO3_rotation[i].set_wigner(edge_rot_mat)

        ###############################################################
        # Initialize node embeddings
        ###############################################################

        # Init per node representations using an atomic number based embedding
        offset = 0
        x = SO3_Embedding(
            num_atoms,
            self.lmax_list,
            self.sphere_channels,
            self.device,
            self.dtype,
        )

        offset_res = 0
        offset = 0
        # Initialize the l = 0, m = 0 coefficients for each resolution
        for i in range(self.num_resolutions):
            if self.num_resolutions == 1:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)
            else:
                x.embedding[:, offset_res, :] = self.sphere_embedding(atomic_numbers)[
                    :, offset : offset + self.sphere_channels
                ]
            offset = offset + self.sphere_channels
            offset_res = offset_res + int((self.lmax_list[i] + 1) ** 2)

        # Edge encoding (distance and atom edge)
        edge_distance = self.distance_expansion(edge_distance)
        if self.share_atom_edge_embedding and self.use_atom_edge_embedding:
            source_element = atomic_numbers[edge_index[0]]  # Source atom atomic number
            target_element = atomic_numbers[edge_index[1]]  # Target atom atomic number
            source_embedding = self.source_embedding(source_element)
            target_embedding = self.target_embedding(target_element)
            edge_distance = torch.cat(
                (edge_distance, source_embedding, target_embedding), dim=1
            )
        else:
            # will be constructed in each transformer block
            pass

        # Edge-degree embedding for initializing node features
        edge_degree = self.edge_degree_embedding(
            atomic_numbers, edge_distance, edge_index
        )
        x.embedding = x.embedding + edge_degree.embedding

        ###############################################################
        # Update spherical node embeddings
        ###############################################################

        for i in range(self.num_layers):
            x = self.blocks[i](
                x,  # SO3_Embedding
                atomic_numbers,
                edge_distance,
                edge_index,
                batch=data.batch,  # for GraphDropPath
            )

        # Final layer norm
        x.embedding = self.norm(x.embedding)

        ###############################################################
        # Energy estimation
        ###############################################################
        node_energy = self.energy_block(x)
        energy = get_scalar_from_embedding(
            node_energy, data, avg_num_nodes=self.avg_num_nodes
        )

        # hessian_ij = self.grad_hess_ij(energy=energy, posj=posj, posi=posi)

        ###############################################################
        # Force estimation
        ###############################################################

        forces = self.force_block(x, atomic_numbers, edge_distance, edge_index)
        forces = forces.embedding.narrow(1, 1, 3)
        forces = forces.view(-1, 3)

        outputs = {}

        ###############################################################
        # Hessian estimation
        ###############################################################
        if hessian:
            # we are going to use a different graph here
            # with a bigger cutoff radius
            edge_index_hessian = data.edge_index_hessian
            edge_distance_hessian = data.edge_distance_hessian
            edge_distance_vec_hessian = data.edge_distance_vec_hessian

            # Compute 3x3 rotation matrix per edge
            edge_rot_mat_hessian = self._init_edge_rot_mat(
                data, edge_index_hessian, edge_distance_vec_hessian
            )

            # Initialize the WignerD matrices and other values for spherical harmonic calculations
            # used in edge_degree_embedding, TransBlockV2
            for i in range(self.num_resolutions):
                self.SO3_rotation_hessian[i].set_wigner(edge_rot_mat_hessian)

            # Edge encoding (distance and atom edge)
            edge_distance_hessian = self.distance_expansion_hessian(
                edge_distance_hessian
            )
            if (
                self.share_atom_edge_embedding_hessian
                and self.use_atom_edge_embedding_hessian
            ):
                source_element = atomic_numbers[
                    edge_index_hessian[0]
                ]  # Source atom atomic number
                target_element = atomic_numbers[
                    edge_index_hessian[1]
                ]  # Target atom atomic number
                source_embedding = self.source_embedding_hessian(source_element)
                target_embedding = self.target_embedding_hessian(target_element)
                edge_distance_hessian = torch.cat(
                    (edge_distance_hessian, source_embedding, target_embedding), dim=1
                )

            if self.reinit_edge_degree_embedding_hessian:
                # Edge-degree embedding for initializing node features
                edge_degree_hessian = self.edge_degree_embedding_hessian(
                    atomic_numbers, edge_distance_hessian, edge_index_hessian
                )
                x.embedding = x.embedding + edge_degree_hessian.embedding

            for i in range(self.num_layers_hessian):
                x = self.hessian_layers[i](
                    x,  # SO3_Embedding
                    atomic_numbers=atomic_numbers,
                    edge_distance=edge_distance_hessian,
                    edge_index=edge_index_hessian,
                    batch=data.batch,  # for GraphDropPath
                )

            # messages: SO3_Embedding (E, L, num_heads * attn_value_channels)
            x_message = self.hessian_head(
                x,
                atomic_numbers=atomic_numbers,
                edge_distance=edge_distance_hessian,
                edge_index=edge_index_hessian,
                return_raw_messages=self.hessian_no_attn_weights,  # messages without attention weights
                return_attn_messages=True,
                attn_wo_sigmoid=self.attn_wo_sigmoid,
            )
            # select l0, l1, l2 features
            l012_message_emb = x_message.embedding.narrow(
                dim=1, start=0, length=9
            )  # length=2l+1
            l012_edge_features = SO3_Embedding(
                length=l012_message_emb.shape[0],
                lmax_list=[2],
                num_channels=l012_message_emb.shape[2],
                device=self.device,
                dtype=self.dtype,
            )
            l012_edge_features.set_embedding(
                l012_message_emb
            )  # (E, 9, num_heads * attn_value_channels)
            # project channel dimension to 1
            l012_edge_features = self.hessian_edge_message_proj(l012_edge_features)
            l012_edge_features: torch.Tensor = l012_edge_features.embedding[:, :, 0]
            l012_edge_features_3x3 = irreps_to_cartesian_matrix(
                l012_edge_features
            )  # (E, 3, 3)
            # messages: torch.Tensor = l012_edge_features

            # combine message with node embeddings (self-connection)
            # node embeddings -> (N, 3, 3)
            l012_node_tensor = x.embedding.narrow(dim=1, start=0, length=9)
            l012_node_features = SO3_Embedding(
                length=l012_node_tensor.shape[0],
                lmax_list=[2],
                num_channels=l012_node_tensor.shape[2],
                device=self.device,
                dtype=self.dtype,
            )  # (N, 9, C)
            l012_node_features.set_embedding(l012_node_tensor)
            l012_node_features = self.hessian_node_proj(l012_node_features)
            l012_node_features: torch.Tensor = l012_node_features.embedding[:, :, 0]
            # (N, 3, 3)
            l012_node_features_3x3 = irreps_to_cartesian_matrix(l012_node_features)

            if return_sparse_hessian:
                # A sparse COO tensor can be constructed by providing the two tensors of indices and values, 
                # as well as the size of the sparse tensor (when it cannot be inferred from the indices and values tensors) to a function torch.sparse_coo_tensor()
                raise NotImplementedError("Sparse Hessian not implemented")
            else:
                hessian = self._get_hessian_from_features(
                    edge_index=edge_index_hessian,
                    data=data,
                    l012_edge_features=l012_edge_features_3x3,
                    l012_node_features=l012_node_features_3x3,
                )

            if return_l_features:
                outputs["l012_node_features"] = l012_node_features_3x3
                outputs["l012_edge_features"] = l012_edge_features_3x3
                outputs["l012_node_features_irreps"] = l012_node_features
                outputs["l012_edge_features_irreps"] = l012_edge_features

            outputs["hessian"] = hessian

        # return energy.reshape(data.ae.shape), forces, outputs
        return energy, forces, outputs

    # Initialize the edge rotation matrics
    def _init_edge_rot_mat(self, data, edge_index, edge_distance_vec):
        return init_edge_rot_mat(edge_distance_vec)

    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear) or isinstance(m, SO3_LinearV2):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            if self.weight_init == "normal":
                std = 1 / math.sqrt(m.in_features)
                torch.nn.init.normal_(m.weight, 0, std)

        elif isinstance(m, torch.nn.LayerNorm):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1.0)

    def _uniform_init_rad_func_linear_weights(self, m):
        if isinstance(m, RadialFunction):
            m.apply(self._uniform_init_linear_weights)

    def _uniform_init_linear_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
            std = 1 / math.sqrt(m.in_features)
            torch.nn.init.uniform_(m.weight, -std, std)

    @torch.jit.ignore
    def no_weight_decay(self):
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if (
                isinstance(module, torch.nn.Linear)
                or isinstance(module, SO3_LinearV2)
                or isinstance(module, torch.nn.LayerNorm)
                or isinstance(module, EquivariantLayerNormArray)
                or isinstance(module, EquivariantLayerNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonics)
                or isinstance(module, EquivariantRMSNormArraySphericalHarmonicsV2)
                or isinstance(module, GaussianRadialBasisLayer)
            ):
                for parameter_name, _ in module.named_parameters():
                    if isinstance(module, torch.nn.Linear) or isinstance(
                        module, SO3_LinearV2
                    ):
                        if "weight" in parameter_name:
                            continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)
        return set(no_wd_list)

    def get_muon_param_groups(
        self,
        **kwargs,
    ):
        """
        Build parameter groups for MuonWithAuxAdam with a strict scope:

        - Muon group (use_muon=True): ONLY parameters with ndim >= 2 inside
          `hessian_layers`.
        - Aux Adam group (use_muon=False): every other parameter in the model
          (embeddings, heads, biases/gains, blocks, etc.).

        Returns two param-group dicts.
        """

        muon_params = []
        adam_params = []

        for name, param in self.named_parameters():
            if name.startswith("hessian_layers.") and param.ndim >= 2:
                muon_params.append(param)
            else:
                adam_params.append(param)

        return muon_params, adam_params
