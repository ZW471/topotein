from copy import copy
from functools import partial

import torch_scatter
from beartype import beartype as typechecker
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding, GCPLayerNorm, GCP, GCPInteractions, \
    GCPMessagePassing, get_GCP_with_custom_cfg, GCPDropout
from typing import Any, Optional, Tuple, Union, Dict
from torch_geometric.data import Batch
from graphein.protein.tensor.data import ProteinBatch
import torch
from jaxtyping import Bool, Float, Int64, jaxtyped

from topotein.features.topotein_complex import TopoteinComplex
from topotein.models.graph_encoders.layers.location_attention import GeometryLocationAttention
from topotein.models.utils import centralize, lift_features_with_padding, map_to_cell_index, get_com, get_scalar_rep
from omegaconf import DictConfig, OmegaConf

from proteinworkshop.models.utils import localize, safe_norm, get_activations


class TCP(GCP):

    NODE_TYPE = "node"
    EDGE_TYPE = "edge"
    SSE_TYPE = "sse"
    PR_TYPE = "pr"

    def __init__(self, input_dims: ScalarVector, output_dims: ScalarVector, input_type: str,
                 nonlinearities: Tuple[Optional[str]] = ("silu", "silu"),
                 scalar_out_nonlinearity: Optional[str] = "silu", scalar_gate: int = 0, vector_gate: bool = True,
                 feedforward_out: bool = False, bottleneck: int = 1, scalarization_vectorization_output_dim: int = 3,
                 enable_e3_equivariance: bool = False, **kwargs):

        super().__init__(input_dims, output_dims, nonlinearities, scalar_out_nonlinearity, scalar_gate, vector_gate,
                         feedforward_out, bottleneck, scalarization_vectorization_output_dim, enable_e3_equivariance,
                         **kwargs)

        self.input_type = input_type
        if self.vector_input_dim:
            if self.input_type == self.SSE_TYPE:
                scalar_vector_frame_dim = scalarization_vectorization_output_dim * 6 + 3 * 4
                self.scalar_out = (
                    nn.Sequential(
                        nn.Linear(
                            self.hidden_dim
                            + self.scalar_input_dim
                            + scalar_vector_frame_dim,
                            self.scalar_output_dim,
                            ),
                        get_activations(scalar_out_nonlinearity),
                        nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                    )
                    if feedforward_out
                    else nn.Linear(
                        self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                        )
                )
            elif self.input_type == self.PR_TYPE:
                scalar_vector_frame_dim = (scalarization_vectorization_output_dim * 6 + 3 * 4) * 2
                self.scalar_out = (
                    nn.Sequential(
                        nn.Linear(
                            self.hidden_dim
                            + self.scalar_input_dim
                            + scalar_vector_frame_dim,
                            self.scalar_output_dim,
                        ),
                        get_activations(scalar_out_nonlinearity),
                        nn.Linear(self.scalar_output_dim, self.scalar_output_dim),
                    )
                    if feedforward_out
                    else nn.Linear(
                        self.hidden_dim + self.scalar_input_dim + scalar_vector_frame_dim,
                        self.scalar_output_dim,
                    )
                )
    @staticmethod
    def get_sse_edge_index_and_mask(edge_index, node_to_sse_mapping, must_between_sse=True):
        cell_edge_index = map_to_cell_index(edge_index, node_to_sse_mapping)

        if must_between_sse:
            mask = ((~(cell_edge_index == -1).any(dim=0)) & (cell_edge_index[0] != cell_edge_index[1]))
        else:
            mask = ((~(cell_edge_index[0] == -1)) & (cell_edge_index[0] != cell_edge_index[1]))

        return cell_edge_index[:, mask], mask

    @jaxtyped(typechecker=typechecker)
    def scalarize(
        self,
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        enable_e3_equivariance: bool,
        dim_size: int,
        node_pos: Optional[Float[torch.Tensor, "n_nodes 3"]] = None,
        node_to_sse_mapping: torch.Tensor = None,
        node_mask: Optional[Bool[torch.Tensor, "n_nodes"]] = None,
        cell_frames: Optional[Float[torch.Tensor, "batch_num_cells 3 3"]] = None,
        pr_frames: Optional[Float[torch.Tensor, "batch_num_prs 3 3"]] = None,
        **kwargs
    ) -> Float[torch.Tensor, "effective_batch_num_entities out_scalar_dim"]:
        row, col = edge_index[0], edge_index[1]
        input_type = self.input_type
        # gather source node features for each `entity` (i.e., node or edge)
        # note: edge inputs are already ordered according to source nodes
        if input_type == self.NODE_TYPE:
            vector_rep_i = vector_rep[row]
        elif input_type == self.EDGE_TYPE or input_type == self.SSE_TYPE or input_type == self.PR_TYPE:
            vector_rep_i = vector_rep
        else:
            raise ValueError(f"Invalid input type: {input_type}")

            # potentially enable E(3)-equivariance and, thereby, chirality-invariance
        if enable_e3_equivariance:
            frames[:, 1, :] = torch.abs(frames[:, 1, :])
            if cell_frames is not None:
                cell_frames[:, 1, :] = torch.abs(cell_frames[:, 1, :])

        # project equivariant values onto corresponding local frames

        if input_type == self.SSE_TYPE:
            edge_mask = None
            if node_mask is not None:
                edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
                edge_index = edge_index[:, edge_mask]

            cell_edge_index, mask = TCP.get_sse_edge_index_and_mask(edge_index, node_to_sse_mapping, must_between_sse=False)

            if edge_mask is not None:
                f_e_ij = frames[edge_mask][mask]
            else:
                f_e_ij = frames[mask]

            vector_rep_i = vector_rep_i[cell_edge_index[0]]
            sse_com = get_com(node_pos[node_to_sse_mapping.indices()[0]], node_to_sse_mapping.indices()[1], node_to_sse_mapping.size()[1])[cell_edge_index[0]]
            r_com_i = (node_pos[edge_index[0, mask]] - sse_com).unsqueeze(1)
            vector_rep_i = torch.concat([vector_rep_i, r_com_i, torch.cross(r_com_i, vector_rep_i, dim=-1), cell_frames[cell_edge_index[0]].transpose(-1, -2)], dim=1)
            if enable_e3_equivariance:
                raise NotImplementedError('E3 equivariance is not yet implemented for cell inputs.')
            local_scalar_rep_i = torch.bmm(vector_rep_i, f_e_ij)


            local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], -1)

        elif input_type == self.PR_TYPE:
            if pr_frames is None:
                raise ValueError(f"pr_frames must be provided for PR input type: {input_type}")
            batch = kwargs.get("batch", None)
            if batch is None:
                raise ValueError(f"batch must be provided to scalarization_kwargs for PR input type: {kwargs}")
            local_scalar_rep_i = torch.concat([get_scalar_rep(batch, vector_rep, 3, to_rank) for to_rank in [0, 2]], dim=1)

        elif input_type == self.NODE_TYPE or input_type == self.EDGE_TYPE:
            if node_mask is not None:
                edge_mask = node_mask[row] & node_mask[col]
                local_scalar_rep_i = torch.zeros(
                    (edge_index.shape[1], 3, 3), device=edge_index.device
                )
                local_scalar_rep_i[edge_mask] = torch.bmm(
                    vector_rep_i[edge_mask], frames[edge_mask]
                )
                local_scalar_rep_i = local_scalar_rep_i.transpose(-1, -2)
            else:
                local_scalar_rep_i = torch.bmm(vector_rep_i, frames)

            # reshape frame-derived geometric scalars
            local_scalar_rep_i = local_scalar_rep_i.reshape(vector_rep_i.shape[0], -1)
        else:
            raise ValueError(f"Invalid input type: {input_type}")

        if input_type == self.NODE_TYPE:
            # for node inputs, summarize all edge-wise geometric scalars using an average
            return torch_scatter.scatter(
                local_scalar_rep_i,
                # summarize according to source node indices due to the directional nature of GCP's equivariant frames
                row,
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )
        elif input_type == self.EDGE_TYPE or input_type == self.PR_TYPE:
            return local_scalar_rep_i
        elif input_type == self.SSE_TYPE:
            return torch_scatter.scatter(
                local_scalar_rep_i,
                cell_edge_index[0],
                dim=0,
                dim_size=dim_size,
                reduce="mean",
            )
        else:
            raise ValueError(f"Invalid input type: {input_type}")



    # noinspection PyMethodOverriding
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"],
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        ],
        edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        cell_frames: Optional[Float[torch.Tensor, "batch_num_cells 3 3"]] = None,
        pr_frames: Optional[Float[torch.Tensor, "batch_num_prs 3 3"]] = None,
        node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
        node_pos: Optional[Float[torch.Tensor, "n_nodes 3"]] = None,
        node_to_sse_mapping: Optional[Int64[torch.Tensor, "n_nodes batch_num_cells"]] = None,
        scalarization_kwargs: Optional[Dict] = None,
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"],
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            if scalarization_kwargs is None:
                scalarization_kwargs = {}
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                edge_index,
                frames,
                enable_e3_equivariance=self.enable_e3_equivariance,
                dim_size=vector_down_frames_hidden_rep.shape[0],
                node_mask=node_mask,
                cell_frames=cell_frames,
                pr_frames=pr_frames,
                node_pos=node_pos,
                node_to_sse_mapping=node_to_sse_mapping,
                **scalarization_kwargs,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


class TCPEmbedding(GCPEmbedding):
    def __init__(self, sse_input_dims, sse_hidden_dims, pr_input_dims, pr_hidden_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sse_input_dims = sse_input_dims
        self.sse_hidden_dims = sse_hidden_dims
        self.pr_input_dims = pr_input_dims
        self.pr_hidden_dims = pr_hidden_dims

        use_gcp_norm = kwargs.get("use_gcp_norm", True)
        nonlinearities = kwargs.get("nonlinearities", ("silu", "silu"))
        cfg = kwargs.get("cfg", None)

        self.sse_normalization = GCPLayerNorm(
            sse_input_dims if self.pre_norm else sse_hidden_dims,
            use_gcp_norm=use_gcp_norm
        )

        self.pr_normalization = GCPLayerNorm(
            pr_input_dims if self.pre_norm else pr_hidden_dims,
            use_gcp_norm=use_gcp_norm
        )

        self.sse_embedding = TCP(
            sse_input_dims,
            sse_hidden_dims,
            input_type=TCP.SSE_TYPE,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.pr_embedding = TCP(
            pr_input_dims,
            pr_hidden_dims,
            input_type=TCP.PR_TYPE,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

    @jaxtyped(typechecker=typechecker)
    def forward(
            self, batch: Union[Batch, ProteinBatch]
    ) -> Tuple[
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m chi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_nodes h_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x xi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_edges e_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_cells c_hidden_dim"],
                Float[torch.Tensor, "batch_num_cells c rho_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_cells c_hidden_dim"],
        ],
        Union[
            Tuple[
                Float[torch.Tensor, "batch_num_prs p_hidden_dim"],
                Float[torch.Tensor, "batch_num_prs p pi_hidden_dim"],
            ],
            Float[torch.Tensor, "batch_num_prs p_hidden_dim"],
        ],
    ]:
        batch.f_ij = batch.f_ij.transpose(-1, -2)
        node_rep, edge_rep = super().forward(batch)
        batch.f_ij = batch.f_ij.transpose(-1, -2)
        sse_rep = ScalarVector(batch.c, batch.rho)
        pr_rep = ScalarVector(batch.p, batch.pi)

        # TODO: calculate cell rep based on updated positions
        sse_rep = (
            sse_rep.scalar if not self.sse_embedding.vector_input_dim else sse_rep
        )
        pr_rep = (
            pr_rep.scalar if not self.pr_embedding.vector_input_dim else pr_rep
        )

        if self.pre_norm:
            sse_rep = self.sse_normalization(sse_rep)
            pr_rep = self.pr_normalization(pr_rep)

        node_to_sse_mapping = getattr(batch, "N0_2", batch.sse_cell_complex.calculator.eval("B0_2"))

        sse_rep = self.sse_embedding(
            sse_rep,
            batch.edge_index,
            frames=batch.f_ij,
            cell_frames=batch.f_ij_cell,
            node_mask=getattr(batch, "mask", None),
            node_pos=batch.pos,
            node_to_sse_mapping=node_to_sse_mapping
        )

        pr_rep = self.pr_embedding(
            pr_rep,
            batch.edge_index,
            frames=batch.f_ij,
            pr_frames=batch.pr_frames,
            node_mask=getattr(batch, "mask", None),
            scalarization_kwargs={"batch": batch}
        )

        if not self.pre_norm:
            sse_rep = self.sse_normalization(sse_rep)
            pr_rep = self.pr_normalization(pr_rep)


        return node_rep, edge_rep, sse_rep, pr_rep


class TCPMessagePassing(GCPMessagePassing):

    @jaxtyped(typechecker=typechecker)
    def message(
            self,
            node_rep: ScalarVector,
            edge_rep: ScalarVector,
            cell_rep: ScalarVector,
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            frames: Float[torch.Tensor, "batch_num_edges 3 3"],
            node_to_sse_mapping: torch.Tensor = None,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Float[torch.Tensor, "batch_num_edges message_dim"]:
        row, col = edge_index
        node_vector = node_rep.vector.reshape(
            node_rep.vector.shape[0],
            node_rep.vector.shape[1] * node_rep.vector.shape[2],
            )
        vector_reshaped = ScalarVector(node_rep.scalar, node_vector)

        node_s_row, node_v_row = vector_reshaped.idx(row)
        node_s_col, node_v_col = vector_reshaped.idx(col)

        node_v_row = node_v_row.reshape(node_v_row.shape[0], node_v_row.shape[1] // 3, 3)
        node_v_col = node_v_col.reshape(node_v_col.shape[0], node_v_col.shape[1] // 3, 3)

        # cell features for edge
        cell_vector = cell_rep.vector.reshape(
            cell_rep.vector.shape[0],
            cell_rep.vector.shape[1] * cell_rep.vector.shape[2],
            )
        cell_scalar = lift_features_with_padding(cell_rep.scalar, neighborhood=node_to_sse_mapping)
        cell_vector = lift_features_with_padding(cell_vector, neighborhood=node_to_sse_mapping)

        vector_reshaped = ScalarVector(cell_scalar, cell_vector)

        cell_s_row, cell_v_row = vector_reshaped.idx(row)
        cell_s_col, cell_v_col = vector_reshaped.idx(col)

        cell_v_row = cell_v_row.reshape(cell_v_row.shape[0], cell_v_row.shape[1] // 3, 3)
        cell_v_col = cell_v_col.reshape(cell_v_col.shape[0], cell_v_col.shape[1] // 3, 3)


        message = edge_rep.concat((
            ScalarVector(node_s_row, node_v_row),
            ScalarVector(node_s_col, node_v_col),
            ScalarVector(cell_s_row, cell_v_row),
            ScalarVector(cell_s_col, cell_v_col),
        ))

        message_residual = self.message_fusion[0](
            message, edge_index, frames, node_inputs=False, node_mask=node_mask
        )
        # edge message passing
        for module in self.message_fusion[1:]:
            # exchange geometric messages while maintaining residual connection to original message
            new_message = module(
                message_residual,
                edge_index,
                frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            message_residual = message_residual + new_message

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(
                message_residual.scalar
            )
            message_residual = ScalarVector(
                message_residual.scalar * message_residual_attn,
                message_residual.vector,
                )

        return message_residual.flatten()


    @jaxtyped(typechecker=typechecker)
    def forward(
            self,
            node_rep: ScalarVector,
            edge_rep: ScalarVector,
            cell_rep: ScalarVector,
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            frames: Float[torch.Tensor, "batch_num_edges 3 3"],
            node_to_sse_mapping: torch.Tensor = None,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Tuple[ScalarVector, ScalarVector]:
        message = self.message(
            node_rep, edge_rep, cell_rep, edge_index, frames, node_to_sse_mapping, node_mask
        )
        node_aggregate = self.aggregate(
            message, edge_index, dim_size=node_rep.scalar.shape[0]
        )

        cell_edge_index, mask = TCP.get_sse_edge_index_and_mask(edge_index, node_to_sse_mapping)

        cell_aggregate = self.aggregate(
            message[mask], cell_edge_index, dim_size=cell_rep.scalar.shape[0]
        )

        node_aggregate = ScalarVector.recover(node_aggregate, self.vector_output_dim)
        cell_aggregate = ScalarVector.recover(cell_aggregate, self.vector_output_dim)


        return node_aggregate, cell_aggregate





class TCPInteractions(GCPInteractions):
    def __init__(self, node_dims: ScalarVector, edge_dims: ScalarVector, sse_dims: ScalarVector, pr_dims: ScalarVector, cfg: DictConfig, layer_cfg: DictConfig,
                 dropout: float = 0.0, nonlinearities: Optional[Tuple[Any, Any]] = None, use_pr_pooling: bool = False):
        super().__init__(node_dims, edge_dims, cfg, layer_cfg, dropout, nonlinearities)
        self.interaction = TCPMessagePassing(
            ScalarVector(
                node_dims.scalar + sse_dims.scalar,
                node_dims.vector + sse_dims.vector,
            ),
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function="sum",
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )


        self.pr_pooling = use_pr_pooling
        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_GCP = partial(get_GCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleDict({
            "0": GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm),
            "2": GCPLayerNorm(sse_dims, use_gcp_norm=layer_cfg.use_gcp_norm),
            "3": GCPLayerNorm(pr_dims, use_gcp_norm=layer_cfg.use_gcp_norm),
        })
        self.gcp_dropout = nn.ModuleDict({
            "0": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
            "2": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
            "3": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
        })


        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_GCP(
                (
                    node_dims.scalar * 2 + sse_dims.scalar,
                    node_dims.vector * 2 + sse_dims.vector,
                ),
                hidden_dims,
                nonlinearities=("none", "none")
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        interaction_layers = [
            ff_GCP(
                hidden_dims,
                hidden_dims,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_GCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # build out feedforward (FF) network modules for cells
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_TCP = partial(get_TCP_with_custom_cfg, cfg=ff_cfg)
        hidden_dims = (
            (sse_dims.scalar, sse_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * sse_dims.scalar, 2 * sse_dims.vector)
        )
        ff_interaction_layers = [
            ff_TCP(
                (
                    sse_dims.scalar + node_dims.scalar * 2 + edge_dims.scalar,
                    sse_dims.vector + node_dims.vector * 2 + edge_dims.vector),
                hidden_dims,
                input_type=TCP.SSE_TYPE,
                nonlinearities=("none", "none")
                if layer_cfg.num_feedforward_layers == 1
                else cfg.nonlinearities,
                feedforward_out=layer_cfg.num_feedforward_layers == 1,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
        ]

        interaction_layers = [
            ff_TCP(
                hidden_dims,
                hidden_dims,
                input_type=TCP.SSE_TYPE,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_TCP(
                    hidden_dims,
                    sse_dims,
                    input_type=TCP.SSE_TYPE,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.cell_ff_network = nn.ModuleList(ff_interaction_layers)

        if self.pr_pooling:
            hidden_dims = (
                (pr_dims.scalar, pr_dims.vector)
                if layer_cfg.num_feedforward_layers == 1
                else (4 * pr_dims.scalar, 2 * pr_dims.vector)
            )
            ff_interaction_layers = [
                ff_TCP(
                    (
                        pr_dims.scalar + node_dims.scalar + sse_dims.scalar,
                        pr_dims.vector + node_dims.vector + sse_dims.vector),
                    hidden_dims,
                    input_type=TCP.PR_TYPE,
                    nonlinearities=("none", "none")
                    if layer_cfg.num_feedforward_layers == 1
                    else cfg.nonlinearities,
                    feedforward_out=layer_cfg.num_feedforward_layers == 1,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            ]

            interaction_layers = [
                ff_TCP(
                    hidden_dims,
                    hidden_dims,
                    input_type=TCP.PR_TYPE,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
                for _ in range(layer_cfg.num_feedforward_layers - 2)
            ]
            ff_interaction_layers.extend(interaction_layers)

            if layer_cfg.num_feedforward_layers > 1:
                ff_interaction_layers.append(
                    ff_TCP(
                        hidden_dims,
                        pr_dims,
                        input_type=TCP.PR_TYPE,
                        nonlinearities=("none", "none"),
                        feedforward_out=True,
                        enable_e3_equivariance=cfg.enable_e3_equivariance,
                    )
                )

            self.pr_ff_network = nn.ModuleList(ff_interaction_layers)

        self.attention_head_num = 1
        self.attention_hidden_dim = None
        self.disable_attention = getattr(cfg, "disable_attention", True)

        self.attentive_node2sse = GeometryLocationAttention(
            from_sv_dim=node_dims,
            to_sv_dim=node_dims,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='leaky_relu',
            concat=True,
            higher_to_lower=False,
            disable_attention=self.disable_attention,
            pool='sum'
        )

        self.attentive_sse2node = GeometryLocationAttention(
            from_sv_dim=sse_dims,
            to_sv_dim=node_dims,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='leaky_relu',
            concat=True,
            higher_to_lower=True,
            disable_attention=self.disable_attention,
        )

        if self.pr_pooling:
            self.attentive_node2pr = GeometryLocationAttention(
                from_sv_dim=node_dims,
                to_sv_dim=pr_dims,
                num_heads=self.attention_head_num * 4,
                hidden_dim=self.attention_hidden_dim,
                activation='leaky_relu',
                concat=True,
                higher_to_lower=False,
                disable_attention=False,
            )

            self.attentive_sse2pr = GeometryLocationAttention(
                from_sv_dim=sse_dims,
                to_sv_dim=pr_dims,
                num_heads=self.attention_head_num * 4,
                hidden_dim=self.attention_hidden_dim,
                activation='leaky_relu',
                concat=True,
                higher_to_lower=False,
                disable_attention=False,
            )

    @jaxtyped(typechecker=typechecker)
    def forward(
            self,
            batch,
            node_rep: Tuple[
                Float[torch.Tensor, "batch_num_nodes node_hidden_dim"],
                Float[torch.Tensor, "batch_num_nodes m 3"],
            ],
            edge_rep: Tuple[
                Float[torch.Tensor, "batch_num_edges edge_hidden_dim"],
                Float[torch.Tensor, "batch_num_edges x 3"],
            ],
            sse_rep: Tuple[
                Float[torch.Tensor, "batch_num_cells cell_hidden_dim"],
                Float[torch.Tensor, "batch_num_cells c 3"],
            ],
            pr_rep: Tuple[
                Float[torch.Tensor, "batch_num_pr pr_hidden_dim"],
                Float[torch.Tensor, "batch_num_pr p 3"],
            ],
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            frames: Float[torch.Tensor, "batch_num_edges 3 3"],
            pr_frames: Optional[Float[torch.Tensor, "batch_num_pr 3 3"]] = None,
            sse_frames: Optional[Float[torch.Tensor, "batch_num_cells 3 3"]] = None,
            node_frames: Optional[Float[torch.Tensor, "batch_num_nodes 3 3"]] = None,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
            node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
            ccc: TopoteinComplex = None
    ) -> Tuple[
        Tuple[
            Float[torch.Tensor, "batch_num_nodes hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes n 3"],
        ],
        Optional[Float[torch.Tensor, "batch_num_nodes 3"]],
        Tuple[
            Float[torch.Tensor, "batch_num_cells cell_hidden_dim"],
            Float[torch.Tensor, "batch_num_cells d 3"],
        ],
        Tuple[
            Float[torch.Tensor, "batch_num_pr pr_hidden_dim"],
            Float[torch.Tensor, "batch_num_pr p_out 3"],
        ],
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        node_to_sse_mapping = ccc.incidence_matrix(0, 2)
        sse_to_node_mapping = ccc.calculator.eval("B0_2.T")
        if self.pr_pooling:
            node_to_pr_mapping = ccc.incidence_matrix(0, 3)
            sse_to_pr_mapping = ccc.incidence_matrix(2, 3)

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)
            sse_rep = self.gcp_norm["2"](sse_rep)
            if self.pr_pooling:
                pr_rep = self.gcp_norm["3"](pr_rep)

        # forward propagate with interaction module
        hidden_residual, hidden_residual_cell = self.interaction(
            node_rep=node_rep,
            edge_rep=edge_rep,
            cell_rep=sse_rep,
            edge_index=edge_index,
            frames=frames.transpose(-1, -2),
            node_mask=node_mask,
            node_to_sse_mapping=node_to_sse_mapping,
        )

        # aggregate input and hidden features
        sse_com = ccc.get_com(2)
        node_rep_agg = self.attentive_node2sse(
            from_rank_sv=node_rep,
            to_rank_sv=hidden_residual_cell,
            incidence_matrix=node_to_sse_mapping,
            from_frame=node_frames,
            to_frame=sse_frames,
            from_pos=node_pos,
            to_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        cell_edge_index, mask = TCP.get_sse_edge_index_and_mask(edge_index, node_to_sse_mapping)
        edge_rep_agg = ScalarVector(*[torch_scatter.scatter(
            rep[mask],
            cell_edge_index[0],
            dim=0,
            dim_size=node_to_sse_mapping.shape[1],
            reduce="mean",
        ) for rep in edge_rep.vs()])
        sse_hidden_residual = ScalarVector(*hidden_residual_cell.concat((sse_rep, node_rep_agg, edge_rep_agg)))  # c_i || h_i || e_i || m_e
        # propagate with cell feedforward layers
        for module in self.cell_ff_network:
            sse_hidden_residual = module(
                sse_hidden_residual,
                edge_index,
                frames=frames,
                cell_frames=sse_frames,
                node_mask=node_mask,
                node_pos=node_pos,
                node_to_sse_mapping=node_to_sse_mapping
            )

        sse_rep_to_node = self.attentive_sse2node(
            from_rank_sv=sse_hidden_residual,
            to_rank_sv=hidden_residual,
            incidence_matrix=sse_to_node_mapping,
            from_frame=sse_frames,
            to_frame=node_frames,
            to_pos=node_pos,
            from_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        sse_rep_to_node = ScalarVector(*[lift_features_with_padding(res, neighborhood=node_to_sse_mapping) for res in sse_rep_to_node.vs()])
        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep, sse_rep_to_node)))  # h_i || m_e || m_c
        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                edge_index,
                frames.transpose(-1, -2),
                node_inputs=True,
                node_mask=node_mask,
            )
        if self.pr_pooling:
            pr_com = ccc.get_com(3)
            node_rep_to_pr = self.attentive_node2pr(
                from_rank_sv=hidden_residual,
                to_rank_sv=pr_rep,
                incidence_matrix=node_to_pr_mapping,
                from_frame=node_frames,
                to_frame=pr_frames,
                from_pos=node_pos,
                to_pos=pr_com[node_to_pr_mapping.indices()[1]]
            )

            sse_rep_to_pr = self.attentive_sse2pr(
                from_rank_sv=sse_hidden_residual,
                to_rank_sv=pr_rep,
                incidence_matrix=sse_to_pr_mapping,
                from_frame=sse_frames,
                to_frame=pr_frames,
                from_pos=sse_com,
                to_pos=pr_com[sse_to_pr_mapping.indices()[1]]
            )
            pr_hidden_residual = ScalarVector(*pr_rep.concat((node_rep_to_pr, sse_rep_to_pr)))
            for module in self.pr_ff_network:
                pr_hidden_residual = module(
                    pr_hidden_residual,
                    edge_index,
                    frames=frames,
                    pr_frames=pr_frames,
                    scalarization_kwargs={"batch": batch}
                )
            pr_rep = pr_rep + self.gcp_dropout["3"](pr_hidden_residual)

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout["0"](hidden_residual)
        sse_rep = sse_rep + self.gcp_dropout["2"](sse_hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)
            sse_rep = self.gcp_norm["2"](sse_rep)
            if self.pr_pooling:
                pr_rep = self.gcp_norm["3"](pr_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask)

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos, sse_rep, pr_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frames.transpose(-1, -2), node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        # TODO: also allow cell_rep update
        return node_rep, node_pos, sse_rep, pr_rep


@typechecker
def get_TCP_with_custom_cfg(
        input_dims: Any, output_dims: Any, input_type: str, cfg: DictConfig, **kwargs
):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return TCP(input_dims, output_dims, input_type, **cfg_dict)

#%%

if __name__ == "__main__":
    from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding
    from proteinworkshop.models.utils import localize, get_activations
    from omegaconf import OmegaConf, DictConfig
    from proteinworkshop.constants import PROJECT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch: ProteinBatch = torch.load(f'{PROJECT_PATH}/../test/data/sample_batch/sample_batch_for_tcp.pt', weights_only=False).to(device)
    batch.mask = torch.randn(batch.x.shape[0], device=device) > -.3
    print(batch)


    batch.f_ij = localize(batch.pos, batch.edge_index)
    batch.f_ij_cell = localize(batch.pos, batch.sse_cell_index_simple)
    batch.node_to_sse_mapping = batch.N2_0.T.coalesce()

    pos_centroid, X_c = centralize(batch.pos, batch.batch)
    ori_pos = batch.pos
    batch.pos = X_c
    batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = (
        batch.x,
        batch.x_vector_attr,
        batch.edge_attr,
        batch.edge_vector_attr,
        batch.sse_attr,
        batch.sse_vector_attr
    )

    cfg = OmegaConf.create({
        "r_max": 10.0,
        "num_rbf": 8,
        "scalar_gate": 0,
        "vector_gate": True,
        "enable_e3_equivariance": False,
        "nonlinearities": ["silu", "silu"],
        "pre_norm": False,
        "use_scalar_message_attention": True,
        "default_bottleneck": 4,
        "mp_cfg": {
            "edge_encoder": False,
            "edge_gate": False,
            "num_message_layers": 4,
            "message_residual": 0,
            "message_ff_multiplier": 1,
            "self_message": True,
        },
        "scalar_nonlinearity": "silu",
        "vector_nonlinearity": "silu",
        "use_gcp_norm": True,
        "use_gcp_dropout": True,
        "num_feedforward_layers": 2
    })


    tcp_embedding = TCPEmbedding(
        node_input_dims=[49, 2],
        edge_input_dims=[9, 1],
        sse_input_dims=[15, 8],
        node_hidden_dims=[128, 16],
        edge_hidden_dims=[32, 4],
        sse_hidden_dims=[128, 16],
        cfg=cfg
    )
    (h, chi), (e, xi), (c, rho) = tcp_embedding(batch)
    print(f'h: {h.shape}, chi: {chi.shape}')
    print(f'e: {e.shape}, xi: {xi.shape}')
    print(f'c: {c.shape}, rho: {rho.shape}')
    # msg_passing = TCPMessagePassing(
    #     ScalarVector(128 + 32, 16 + 4),
    #     ScalarVector(128, 16),
    #     ScalarVector(32, 4),
    #     cfg=cfg,
    #     mp_cfg=cfg.mp_cfg,
    #     reduce_function="sum",
    #     use_scalar_message_attention=cfg.use_scalar_message_attention,
    # )
    # print(msg_passing)
    #
    # msg = msg_passing(
    #     node_rep=ScalarVector(h, chi),
    #     edge_rep=ScalarVector(e, xi),
    #     cell_rep=ScalarVector(c, rho),
    #     edge_index=batch.edge_index,
    #     frames=batch.f_ij,
    #     cell_frames=batch.f_ij_cell,
    #     node_to_sse_mapping=batch.node_to_sse_mapping,
    #     node_mask = batch.mask,
    # )
    #
    # print(msg.scalar.shape)
    # print(msg.vector.shape)

    interactions = TCPInteractions(
        ScalarVector(128, 16),
        ScalarVector(32, 4),
        ScalarVector(128, 16),
        cfg=cfg,
        layer_cfg=cfg,
        dropout=0.0,
        nonlinearities=cfg.nonlinearities,
    )

    node_s_v, pos = interactions(
        node_rep=ScalarVector(h, chi),
        edge_rep=ScalarVector(e, xi),
        cell_rep=ScalarVector(c, rho),
        frames=batch.f_ij,
        cell_frames=batch.f_ij_cell,
        edge_index=batch.edge_index,
        node_mask = batch.mask,
        node_pos=batch.pos,
        node_to_sse_mapping=batch.node_to_sse_mapping,
    )
    print('>>result')
    print(node_s_v.scalar.shape)
    print(node_s_v.vector.shape)
    print(pos.shape)
