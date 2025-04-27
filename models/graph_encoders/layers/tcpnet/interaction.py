from copy import copy
from functools import partial
from typing import Optional, Tuple, Any

import torch
import torch_scatter
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Float, Int64, Bool
from omegaconf import DictConfig, OmegaConf
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPInteractions, GCPLayerNorm, GCPDropout
from topotein.models.graph_encoders.layers.location_attention import GeometryLocationAttention
from topotein.models.graph_encoders.layers.tcpnet.edge_msg_passing import TCPMessagePassing
from topotein.models.graph_encoders.layers.tcpnet.tcp import TCP
from topotein.models.utils import map_to_cell_index, lift_features_with_padding, get_com, sv_scatter, sv_aggregate, \
    sv_apply_proj


class TCPInteractions(GCPInteractions):
    def __init__(self, node_dims: ScalarVector, edge_dims: ScalarVector, sse_dims: ScalarVector, pr_dims: ScalarVector, cfg: DictConfig, layer_cfg: DictConfig,
                 dropout: float = 0.0, nonlinearities: Optional[Tuple[Any, Any]] = None):
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


        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_TCP = partial(get_TCP_with_custom_cfg, cfg=ff_cfg)

        self.gcp_norm = nn.ModuleDict({
            "0": GCPLayerNorm(node_dims, use_gcp_norm=layer_cfg.use_gcp_norm),
            "3": GCPLayerNorm(pr_dims, use_gcp_norm=layer_cfg.use_gcp_norm),
        })
        self.gcp_dropout = nn.ModuleDict({
            "0": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
            "3": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
        })

        # build out feedforward (FF) network modules
        hidden_dims = (
            (node_dims.scalar, node_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * node_dims.scalar, 2 * node_dims.vector)
        )
        ff_interaction_layers = [
            ff_TCP(
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
            ff_TCP(
                hidden_dims,
                hidden_dims,
                enable_e3_equivariance=cfg.enable_e3_equivariance,
            )
            for _ in range(layer_cfg.num_feedforward_layers - 2)
        ]
        ff_interaction_layers.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers.append(
                ff_TCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network = nn.ModuleList(ff_interaction_layers)

        # build out feedforward (FF) network modules for sses
        hidden_dims = (
            (sse_dims.scalar, sse_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * sse_dims.scalar, 2 * sse_dims.vector)
        )
        ff_interaction_layers = [
            ff_TCP(
                (
                    sse_dims.scalar + node_dims.scalar * 2 + edge_dims.scalar + pr_dims.scalar,
                    sse_dims.vector + node_dims.vector * 2 + edge_dims.vector + pr_dims.vector),
                hidden_dims,
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
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.sse_ff_network = nn.ModuleList(ff_interaction_layers)

        # build out feedforward (FF) network modules for prs
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
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.pr_ff_network = nn.ModuleList(ff_interaction_layers)

        self.attention_head_num = 8
        self.attention_hidden_dim = None

        self.attentive_node2pr = GeometryLocationAttention(
            from_vec_dim=node_dims.vector,
            to_vec_dim=pr_dims.vector,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=False,
        )
        self.W_d_node2pr_s = nn.Linear(node_dims.scalar * self.attention_head_num, node_dims.scalar, bias=False)
        self.W_d_node2pr_v = nn.Linear(node_dims.vector * self.attention_head_num, node_dims.vector, bias=False)

        self.attentive_pr2sse = GeometryLocationAttention(
            from_vec_dim=pr_dims.vector,
            to_vec_dim=sse_dims.vector,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=True,
        )
        self.W_d_pr2sse_s = nn.Linear(pr_dims.scalar * self.attention_head_num, pr_dims.scalar, bias=False)
        self.W_d_pr2sse_v = nn.Linear(pr_dims.vector * self.attention_head_num, pr_dims.vector, bias=False)

        self.attentive_sse2pr = GeometryLocationAttention(
            from_vec_dim=sse_dims.vector,
            to_vec_dim=pr_dims.vector,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=False,
        )
        self.W_d_sse2pr_s = nn.Linear(sse_dims.scalar * self.attention_head_num, sse_dims.scalar, bias=False)
        self.W_d_sse2pr_v = nn.Linear(sse_dims.vector * self.attention_head_num, sse_dims.vector, bias=False)

        self.attentive_node2sse = GeometryLocationAttention(
            from_vec_dim=node_dims.vector,
            to_vec_dim=sse_dims.vector,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=False,
        )
        self.W_d_node2sse_s = nn.Linear(node_dims.scalar * self.attention_head_num, node_dims.scalar, bias=False)
        self.W_d_node2sse_v = nn.Linear(node_dims.vector * self.attention_head_num, node_dims.vector, bias=False)

        self.attentive_sse2node = GeometryLocationAttention(
            from_vec_dim=sse_dims.vector,
            to_vec_dim=node_dims.vector,
            num_heads=self.attention_head_num,
            hidden_dim=5,
            activation='silu',
            concat=True,
            higher_to_lower=True,
        )
        self.W_d_sse2node_s = nn.Linear(sse_dims.scalar * self.attention_head_num, sse_dims.scalar, bias=False)
        self.W_d_sse2node_v = nn.Linear(sse_dims.vector * self.attention_head_num, sse_dims.vector, bias=False)

    @jaxtyped(typechecker=typechecker)
    def forward(
            self,
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
            frame_dict,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
            node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
            node_to_sse_mapping: torch.Tensor = None,
            sse_to_node_mapping: torch.Tensor = None,
            edge_to_sse_mapping: torch.Tensor = None,
            pr_to_sse_mapping: torch.Tensor = None,
            node_to_pr_mapping: torch.Tensor = None,
            sse_to_pr_mapping: torch.Tensor = None,
    ) -> Tuple[
        Tuple[
            Float[torch.Tensor, "batch_num_nodes hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes n 3"],
        ],
        Optional[Float[torch.Tensor, "batch_num_nodes 3"]],
        Tuple[
            Float[torch.Tensor, "batch_num_pr pr_hidden_dim"],
            Float[torch.Tensor, "batch_num_pr p_out 3"],
        ],
    ]:
        node_rep = ScalarVector(node_rep[0], node_rep[1])
        edge_rep = ScalarVector(edge_rep[0], edge_rep[1])

        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm[0](node_rep)

        # forward propagate with interaction module
        hidden_residual, hidden_residual_cell = self.interaction(
            node_rep=node_rep,
            edge_rep=edge_rep,
            sse_rep=sse_rep,
            edge_index=edge_index,
            edge_frames=frame_dict[1],
            node_mask=node_mask,
            node_to_sse_mapping=node_to_sse_mapping,
        )

        # aggregate input and hidden features
        # node_rep_agg = ScalarVector(*[torch_scatter.scatter(
        #     rep[node_to_sse_mapping.indices()[0]],
        #     node_to_sse_mapping.indices()[1],
        #     dim=0,
        #     dim_size=node_to_sse_mapping.shape[1],
        #     reduce="mean",
        # ) for rep in node_rep.vs()])
        sse_com = get_com(
            node_pos[node_to_sse_mapping.indices()[0]],
            node_to_sse_mapping.indices()[1],
            node_to_sse_mapping.size()[1]
        )
        pr_com = get_com(
            node_pos,
            node_to_pr_mapping.indices()[1],
            node_to_pr_mapping.size()[1]
        )
        node_rep_to_sse = self.attentive_node2sse(
            from_rank_sv=node_rep,
            to_rank_sv=sse_rep,
            incidence_matrix=node_to_sse_mapping,
            from_frame=frame_dict[0],
            to_frame=frame_dict[2],
            from_pos=node_pos,
            to_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        node_rep_agg = sv_aggregate(
            sv_apply_proj(node_rep_to_sse, self.W_d_node2sse_s, self.W_d_node2sse_v),
            node_to_sse_mapping,
            reduce="sum",
            indexed_input=True
        )
        edge_rep_agg = ScalarVector(*[torch_scatter.scatter(
            rep[edge_to_sse_mapping.indices()[0]],
            edge_to_sse_mapping.indices()[1],
            dim=0,
            dim_size=edge_to_sse_mapping.shape[1],
            reduce="sum",
        ) for rep in edge_rep.vs()])
        pr_rep_to_sse = self.attentive_pr2sse(
            from_rank_sv=pr_rep,
            to_rank_sv=sse_rep,
            incidence_matrix=pr_to_sse_mapping,
            from_frame=frame_dict[3],
            to_frame=frame_dict[2],
            from_pos=pr_com[pr_to_sse_mapping.indices()[0]],
            to_pos=sse_com,
        )
        pr_rep_to_sse = sv_apply_proj(pr_rep_to_sse, self.W_d_pr2sse_s, self.W_d_pr2sse_v)
        cell_hidden_residual = ScalarVector(*hidden_residual_cell.concat((sse_rep, node_rep_agg, edge_rep_agg, pr_rep_to_sse)))  # c_i || h_i || e_i || m_e
        # propagate with cell feedforward layers
        for module in self.sse_ff_network:
            cell_hidden_residual = module(
                cell_hidden_residual,
                frames=frame_dict[2],
            )
        sse_rep_to_node = self.attentive_sse2node(
            from_rank_sv=cell_hidden_residual,
            to_rank_sv=node_rep,
            incidence_matrix=sse_to_node_mapping,
            from_frame=frame_dict[2],
            to_frame=frame_dict[0],
            to_pos=node_pos,
            from_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        sse_rep_to_node = sv_apply_proj(sse_rep_to_node, self.W_d_sse2node_s, self.W_d_sse2node_v)
        cell_hidden_residual = ScalarVector(*[lift_features_with_padding(res, neighborhood=node_to_sse_mapping) for res in sse_rep_to_node.vs()])

        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep, cell_hidden_residual)))  # h_i || m_e || m_c
        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                frames=frame_dict[0]
            )


        node_rep_to_pr = self.attentive_node2pr(
            from_rank_sv=node_rep,
            to_rank_sv=pr_rep,
            incidence_matrix=node_to_pr_mapping,
            from_frame=frame_dict[0],
            to_frame=frame_dict[3],
            from_pos=node_pos,
            to_pos=pr_com[node_to_pr_mapping.indices()[1]]
        )
        node_rep_to_pr = sv_apply_proj(node_rep_to_pr, self.W_d_node2pr_s, self.W_d_node2pr_v)
        node_rep_to_pr = sv_aggregate(node_rep_to_pr, node_to_pr_mapping, reduce="sum", indexed_input=True)

        sse_rep_to_pr = self.attentive_sse2pr(
            from_rank_sv=sse_rep,
            to_rank_sv=pr_rep,
            incidence_matrix=sse_to_pr_mapping,
            from_frame=frame_dict[2],
            to_frame=frame_dict[3],
            from_pos=sse_com,
            to_pos=pr_com[sse_to_pr_mapping.indices()[1]]
        )
        sse_rep_to_pr = sv_apply_proj(sse_rep_to_pr, self.W_d_sse2pr_s, self.W_d_sse2pr_v)
        sse_rep_to_pr = sv_aggregate(sse_rep_to_pr, sse_to_pr_mapping, reduce="sum", indexed_input=True)

        pr_hidden_residual = ScalarVector(*node_rep_to_pr.concat((pr_rep, sse_rep_to_pr)))
        for module in self.pr_ff_network:
            pr_hidden_residual = module(
                pr_hidden_residual,
                frames=frame_dict[3]
            )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout["0"](hidden_residual)
        pr_rep = pr_rep + self.gcp_dropout["3"](pr_hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)
            pr_rep = self.gcp_norm["3"](pr_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask)

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos, pr_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frame_dict[1], node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        # TODO: also allow cell_rep update
        return node_rep, node_pos, pr_rep


@typechecker
def get_TCP_with_custom_cfg(
        input_dims: Any, output_dims: Any, cfg: DictConfig, **kwargs
):
    cfg_dict = copy(OmegaConf.to_container(cfg, throw_on_missing=True))
    cfg_dict["nonlinearities"] = cfg.nonlinearities
    del cfg_dict["scalar_nonlinearity"]
    del cfg_dict["vector_nonlinearity"]

    for key in kwargs:
        cfg_dict[key] = kwargs[key]

    return TCP(input_dims, output_dims, **cfg_dict)
