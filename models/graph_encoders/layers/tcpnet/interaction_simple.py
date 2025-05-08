from copy import copy
from functools import partial
from typing import Optional, Tuple, Any

import torch
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Float, Int64, Bool
from omegaconf import DictConfig, OmegaConf
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPInteractions, GCPLayerNorm, GCPDropout, \
    GCPMessagePassing
from topotein.features.topotein_complex import TopoteinComplex
from topotein.models.graph_encoders.layers.tcpnet.tcp import TCP


class TCPInteractions(GCPInteractions):
    def __init__(self, node_dims: ScalarVector, edge_dims: ScalarVector, cfg: DictConfig, layer_cfg: DictConfig,
                 dropout: float = 0.0, nonlinearities: Optional[Tuple[Any, Any]] = None):
        super().__init__(node_dims, edge_dims, cfg, layer_cfg, dropout, nonlinearities)

        self.use_original_gcp = cfg.use_original_gcp
        self.interaction = GCPMessagePassing(
            node_dims,
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
        })
        self.gcp_dropout = nn.ModuleDict({
            "0": GCPDropout(dropout, use_gcp_dropout=layer_cfg.use_gcp_dropout),
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
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            frame_dict,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
            node_pos: Optional[Float[torch.Tensor, "batch_num_nodes 3"]] = None,
            ccc: TopoteinComplex = None,
    ) -> Tuple[
        Tuple[
            Float[torch.Tensor, "batch_num_nodes hidden_dim"],
            Float[torch.Tensor, "batch_num_nodes n 3"],
        ],
        Optional[Float[torch.Tensor, "batch_num_nodes 3"]],
    ]:
        # apply GCP normalization (1)
        if self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)

        # forward propagate with interaction module
        hidden_residual = self.interaction(
            node_rep=node_rep,
            edge_rep=edge_rep,
            edge_index=edge_index,
            frames=frame_dict[1].transpose(-1, -2),  # gcp stores frames differently
            node_mask=node_mask
        )

        hidden_residual = ScalarVector(*hidden_residual.concat((node_rep,)))  # h_i || m_e || m_c
        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                frames=frame_dict[0 if not self.use_original_gcp else 1],
                use_original_gcp=self.use_original_gcp,
                gcp_forward_kwargs={
                    "edge_index": edge_index,
                    "node_inputs": True,
                    "node_mask": None
                }
            )


        # apply GCP dropout
        node_rep = node_rep + self.gcp_dropout["0"](hidden_residual)

        # apply GCP normalization (2)
        if not self.pre_norm:
            node_rep = self.gcp_norm["0"](node_rep)

        # update only unmasked node representations and residuals
        if node_mask is not None:
            node_rep = node_rep.mask(node_mask.float())

        # bypass updating node positions
        if not self.predict_node_positions:
            return node_rep, node_pos

        # update node positions
        node_pos = node_pos + self.derive_x_update(
            node_rep, edge_index, frame_dict[1].transpose(-1, -2), node_mask=node_mask
        )

        # update only unmasked node positions
        if node_mask is not None:
            node_pos = node_pos * node_mask.float().unsqueeze(-1)

        # TODO: also allow cell_rep update
        return node_rep, node_pos


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
