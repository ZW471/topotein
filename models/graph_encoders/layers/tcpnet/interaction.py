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
from proteinworkshop.models.graph_encoders.layers.gcp import GCPInteractions, GCPLayerNorm, GCPDropout, \
    GCPMessagePassing
from topotein.features.topotein_complex import TopoteinComplex
from topotein.models.graph_encoders.layers.location_attention import GeometryLocationAttention
from topotein.models.graph_encoders.layers.tcpnet.edge_msg_passing import TCPMessagePassing
from topotein.models.graph_encoders.layers.tcpnet.tcp import TCP
from topotein.models.utils import map_to_cell_index, lift_features_with_padding, get_com, sv_scatter, sv_aggregate, \
    sv_apply_proj


class TCPInteractions(GCPInteractions):
    def __init__(self, node_dims: ScalarVector, edge_dims: ScalarVector, sse_dims: ScalarVector, pr_dims: ScalarVector, cfg: DictConfig, layer_cfg: DictConfig,
                 dropout: float = 0.0, nonlinearities: Optional[Tuple[Any, Any]] = None):
        super().__init__(node_dims, edge_dims, cfg, layer_cfg, dropout, nonlinearities)
        self.interaction = GCPMessagePassing(
            node_dims,
            node_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function="sum",
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )
        self.interaction_sse = GCPMessagePassing(
            sse_dims,
            sse_dims,
            edge_dims,
            cfg=cfg,
            mp_cfg=layer_cfg.mp_cfg,
            reduce_function="sum",
            use_scalar_message_attention=layer_cfg.use_scalar_message_attention,
        )
        self.pr_update = cfg.pr_update
        self.sse_update = cfg.sse_update

        # config instantiations
        ff_cfg = copy(cfg)
        ff_cfg.nonlinearities = nonlinearities
        ff_TCP = partial(get_TCP_with_custom_cfg, cfg=ff_cfg)

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
            ff_TCP(
                (
                    node_dims.scalar * 2,
                    node_dims.vector * 2,
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

        self.aa_ff_norm = GCPLayerNorm(ScalarVector(
            node_dims.scalar + sse_dims.scalar,
            node_dims.vector + sse_dims.vector,
            ), use_gcp_norm=layer_cfg.use_gcp_norm)
        ff_interaction_layers_2 = [
            ff_TCP(
                (
                    node_dims.scalar + sse_dims.scalar,
                    node_dims.vector + sse_dims.vector,
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
        ff_interaction_layers_2.extend(interaction_layers)

        if layer_cfg.num_feedforward_layers > 1:
            ff_interaction_layers_2.append(
                ff_TCP(
                    hidden_dims,
                    node_dims,
                    nonlinearities=("none", "none"),
                    feedforward_out=True,
                    enable_e3_equivariance=cfg.enable_e3_equivariance,
                )
            )

        self.feedforward_network_2 = nn.ModuleList(ff_interaction_layers_2)

        # build out feedforward (FF) network modules for sses
        hidden_dims = (
            (sse_dims.scalar, sse_dims.vector)
            if layer_cfg.num_feedforward_layers == 1
            else (4 * sse_dims.scalar, 2 * sse_dims.vector)
        )
        self.sse_ff_norm = GCPLayerNorm(ScalarVector(
            sse_dims.scalar * 2 + node_dims.scalar + pr_dims.scalar,
            sse_dims.vector * 2 + node_dims.vector + pr_dims.vector
        ), use_gcp_norm=layer_cfg.use_gcp_norm)
        ff_interaction_layers = [
            ff_TCP(
                (
                    sse_dims.scalar * 2 + node_dims.scalar + pr_dims.scalar,
                    sse_dims.vector * 2 + node_dims.vector + pr_dims.vector),
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

        self.attention_head_num = 4
        self.attention_hidden_dim = None
        self.disable_attention = getattr(cfg, "disable_attention", False)

        self.attentive_node2sse = GeometryLocationAttention(
            from_sv_dim=node_dims,
            to_sv_dim=sse_dims,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=False,
            disable_attention=self.disable_attention,
        )

        self.attentive_sse2node = GeometryLocationAttention(
            from_sv_dim=sse_dims,
            to_sv_dim=node_dims,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=True,
            disable_attention=self.disable_attention,
        )

        # build out feedforward (FF) network modules for prs
        if self.pr_update:
            hidden_dims = (
                (pr_dims.scalar, pr_dims.vector)
                if layer_cfg.num_feedforward_layers == 1
                else (4 * pr_dims.scalar, 2 * pr_dims.vector)
            )
            self.pr_ff_norm = GCPLayerNorm(ScalarVector(
                pr_dims.scalar + node_dims.scalar + sse_dims.scalar,
                pr_dims.vector + node_dims.vector + sse_dims.vector
            ), use_gcp_norm=layer_cfg.use_gcp_norm)
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

            self.attentive_node2pr = GeometryLocationAttention(
                from_sv_dim=node_dims,
                to_sv_dim=pr_dims,
                num_heads=self.attention_head_num,
                hidden_dim=self.attention_hidden_dim,
                activation='silu',
                concat=True,
                higher_to_lower=False,
                disable_attention=self.disable_attention,
            )

            self.attentive_sse2pr = GeometryLocationAttention(
                from_sv_dim=sse_dims,
                to_sv_dim=pr_dims,
                num_heads=self.attention_head_num,
                hidden_dim=self.attention_hidden_dim,
                activation='silu',
                concat=True,
                higher_to_lower=False,
                disable_attention=self.disable_attention,
            )

        self.attentive_pr2sse = GeometryLocationAttention(
            from_sv_dim=pr_dims,
            to_sv_dim=sse_dims,
            num_heads=self.attention_head_num,
            hidden_dim=self.attention_hidden_dim,
            activation='silu',
            concat=True,
            higher_to_lower=True,
            disable_attention=self.disable_attention,
        )

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
            ccc: TopoteinComplex = None,
            node_to_sse_mapping: torch.Tensor = None,
            sse_to_node_mapping: torch.Tensor = None,
            edge_to_sse_mapping: torch.Tensor = None,
            edge_to_sse_outer_mapping: torch.Tensor = None,
            sse_to_edge_outer_mapping: torch.Tensor = None,
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
            Float[torch.Tensor, "batch_num_cells cell_hidden_dim"],
            Float[torch.Tensor, "batch_num_cells c 3"]
        ],
        Tuple[
            Float[torch.Tensor, "batch_num_pr pr_hidden_dim"],
            Float[torch.Tensor, "batch_num_pr p_out 3"],
        ],
    ]:
        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)
            sse_rep = self.gcp_norm["2"](sse_rep) if self.update_sse else sse_rep
            pr_rep = self.gcp_norm["3"](pr_rep) if self.pr_update else pr_rep

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep=node_rep,
            edge_rep=edge_rep,
            edge_index=edge_index,
            frames=frame_dict[1].transpose(-1, -2),  # gcp stores frames differently
            node_mask=node_mask
        )

        cell_edge_index = map_to_cell_index(edge_index, node_to_sse_mapping)
        mask = ((~(cell_edge_index == -1).any(dim=0)) & (cell_edge_index[0] != cell_edge_index[1]))
        sse_hidden_residual = self.interaction_sse(
            node_rep=sse_rep,
            edge_rep=edge_rep.idx(mask),
            edge_index=cell_edge_index[:, mask],
            frames=frame_dict[1].transpose(-1, -2)[mask],
        )

        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))  # h_i || m_e || m_c
        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                frames=frame_dict[0]
            )

        # return hidden_residual, node_pos, sse_rep, pr_rep
        hidden_residual_1 = hidden_residual

        # aggregate input and hidden features
        sse_com = ccc.get_com(2)
        pr_com = ccc.get_com(3)
        node_rep_agg = self.attentive_node2sse(
            from_rank_sv=hidden_residual_1,
            to_rank_sv=sse_hidden_residual,
            incidence_matrix=node_to_sse_mapping,
            from_frame=frame_dict[0],
            to_frame=frame_dict[2],
            from_pos=node_pos,
            to_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        pr_rep_to_sse = self.attentive_pr2sse(
            from_rank_sv=pr_rep,
            to_rank_sv=sse_hidden_residual,
            incidence_matrix=pr_to_sse_mapping,
            from_frame=frame_dict[3],
            to_frame=frame_dict[2],
            from_pos=pr_com[pr_to_sse_mapping.indices()[0]],
            to_pos=sse_com,
        )
        sse_hidden_residual = ScalarVector(*sse_hidden_residual.concat((sse_rep, node_rep_agg, pr_rep_to_sse)))  # c_i || h_i || e_i || m_e
        # propagate with sse feedforward layers
        sse_hidden_residual = self.sse_ff_norm(sse_hidden_residual)
        for module in self.sse_ff_network:
            sse_hidden_residual = module(
                sse_hidden_residual,
                frames=frame_dict[2],
            )
        sse_rep_to_node = self.attentive_sse2node(
            from_rank_sv=sse_hidden_residual,
            to_rank_sv=node_rep,
            incidence_matrix=sse_to_node_mapping,
            from_frame=frame_dict[2],
            to_frame=frame_dict[0],
            to_pos=node_pos,
            from_pos=sse_com[node_to_sse_mapping.indices()[1]]
        )
        sse_rep_to_node = ScalarVector(*[lift_features_with_padding(res, neighborhood=node_to_sse_mapping) for res in sse_rep_to_node.vs()])

        hidden_residual_2 = ScalarVector(*hidden_residual_1.concat((sse_rep_to_node, )))  # h_i || m_e || m_c
        hidden_residual_2 = self.aa_ff_norm(hidden_residual_2)
        # propagate with feedforward layers
        for module in self.feedforward_network_2:
            hidden_residual_2 = module(
                hidden_residual_2,
                frames=frame_dict[0]
            )



        if self.pr_update:
            node_rep_to_pr = self.attentive_node2pr(
                from_rank_sv=hidden_residual_1,
                to_rank_sv=pr_rep,
                incidence_matrix=node_to_pr_mapping,
                from_frame=frame_dict[0],
                to_frame=frame_dict[3],
                from_pos=node_pos,
                to_pos=pr_com[node_to_pr_mapping.indices()[1]]
            )

            sse_rep_to_pr = self.attentive_sse2pr(
                from_rank_sv=sse_hidden_residual,
                to_rank_sv=pr_rep,
                incidence_matrix=sse_to_pr_mapping,
                from_frame=frame_dict[2],
                to_frame=frame_dict[3],
                from_pos=sse_com,
                to_pos=pr_com[sse_to_pr_mapping.indices()[1]]
            )

            pr_hidden_residual = ScalarVector(*node_rep_to_pr.concat((pr_rep, sse_rep_to_pr)))
            pr_hidden_residual = self.pr_ff_norm(pr_hidden_residual)
            for module in self.pr_ff_network:
                pr_hidden_residual = module(
                    pr_hidden_residual,
                    frames=frame_dict[3]
                )

        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout["0"](hidden_residual_2)
        sse_rep = sse_rep + self.gcp_dropout["2"](sse_hidden_residual) if self.sse_update else sse_rep
        pr_rep = pr_rep + self.gcp_dropout["3"](pr_hidden_residual) if self.pr_update else pr_rep

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)
            sse_rep = self.gcp_norm["2"](sse_rep) if self.sse_update else sse_rep
            pr_rep = self.gcp_norm["3"](pr_rep) if self.pr_update else pr_rep

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask)

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos, sse_rep, pr_rep

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frame_dict[1], node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        # TODO: also allow cell_rep update
        return node_rep, node_pos, sse_rep, pr_rep


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
