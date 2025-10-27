#%%
from typing import Literal, Dict

import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.custom_types import ActivationType

from topotein.models.graph_encoders.layers.topotein_net.tpp import TPP
from topotein.models.utils import sv_aggregate


class GeometricMessagePassing(nn.Module):
    def __init__(self, in_dim_dict, out_dim_dict, mapping_dict, agg_reduce='sum',
                 frame_selection: Literal['target', 'source'] = 'target', activation: ActivationType = 'silu', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.target_ranks = set({})
        self.mapping_dict = mapping_dict
        self.frame_selection = frame_selection
        self.agg_reduce = agg_reduce
        self.W_intra = nn.ModuleDict()
        self.msg_num_to_rank = {}
        for from_rank in self.mapping_dict.keys():
            for to_rank in self.mapping_dict[from_rank]:
                self.target_ranks.add(to_rank)
                self.W_intra[f'{from_rank}->{to_rank}'] = nn.ModuleList([
                    TPP(
                        in_dims=ScalarVector(
                            self.in_dim_dict[from_rank].scalar + self.out_dim_dict[to_rank].scalar,
                            self.in_dim_dict[from_rank].vector + self.out_dim_dict[to_rank].vector,
                            ),
                        out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                        rank=self._get_frame_rank(to_rank, from_rank),
                        activation=activation
                    )
                ] + [
                    TPP(
                        in_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                        out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                        rank=self._get_frame_rank(to_rank, from_rank),
                        activation=activation
                    ) for _ in range(1)
                ])

                if to_rank not in self.msg_num_to_rank.keys():
                    self.msg_num_to_rank[to_rank] = 0
                self.msg_num_to_rank[to_rank] += 1

        self.W_inter = nn.ModuleDict()
        self.S_attn = nn.ModuleDict()
        for to_rank in self.target_ranks:
            self.W_inter[f'{to_rank}'] = nn.ModuleList([
               TPP(
                   in_dims=ScalarVector(
                       self.out_dim_dict[to_rank].scalar * self.msg_num_to_rank[to_rank] + self.in_dim_dict[to_rank].scalar,
                       self.out_dim_dict[to_rank].vector * self.msg_num_to_rank[to_rank] + self.in_dim_dict[to_rank].vector,
                       ),
                   out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                   rank=to_rank,
                   activation=activation
               )
           ] + [
                TPP(
                    in_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                    out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                    rank=to_rank,
                    activation=activation
                ) for _ in range(3)
            ])
            self.S_attn[f'{to_rank}'] = nn.Sequential(
                nn.Linear(self.out_dim_dict[to_rank].scalar, 1), nn.Sigmoid()
            )

    def _get_frame_rank(self, to_rank, from_rank):
        if self.frame_selection == 'target':
            return to_rank
        else:
            return from_rank

    def forward(self, X_dict, neighborhood_dict, frame_dict):
        msg_dict = {k: [] for k in self.target_ranks}
        for from_rank in self.mapping_dict.keys():
            for to_rank in self.mapping_dict[from_rank]:
                neighborhood = neighborhood_dict[f"N{from_rank}_{to_rank}"]
                msg = sv_aggregate(X_dict[from_rank], neighborhood, self.agg_reduce)
                if to_rank in msg_dict.keys():
                    msg = msg.concat([X_dict[to_rank]])

                msg = self.W_intra[f'{from_rank}->{to_rank}'][0](msg, frame_dict[to_rank])
                for layer in self.W_intra[f'{from_rank}->{to_rank}'][1:]:
                    new_msg = layer(msg, frame_dict[to_rank])
                    msg = msg + new_msg
                msg_dict[to_rank].append(msg)

        update_dict = {}
        for to_rank in msg_dict.keys():
            msg = ScalarVector(*X_dict[to_rank].concat(msg_dict[to_rank]))
            msg = self.W_inter[f'{to_rank}'][0](msg, frame_dict[to_rank])
            for layer in self.W_inter[f'{to_rank}'][1:]:
                new_msg = layer(msg, frame_dict[to_rank])
                msg = msg + new_msg
            update_dict[to_rank] = ScalarVector(self.S_attn[f'{to_rank}'](msg.scalar) * msg.scalar, msg.vector)

        return update_dict


class TPPMessagePassing(nn.Module):
    def __init__(
            self,
            input_dim_dict: Dict[int, ScalarVector],
            output_dim_dict: Dict[int, ScalarVector],
            reduce_function: str = "sum",
            use_scalar_message_attention: bool = True,
    ):
        super().__init__()

        # hyperparameters
        self.scalar_input_dim, self.vector_input_dim = input_dim_dict[0]
        self.scalar_output_dim, self.vector_output_dim = output_dim_dict[0]
        self.edge_scalar_dim, self.edge_vector_dim = input_dim_dict[1]
        self.reduce_function = reduce_function
        self.use_scalar_message_attention = use_scalar_message_attention

        scalars_in_dim = 2 * self.scalar_input_dim + self.edge_scalar_dim
        vectors_in_dim = 2 * self.vector_input_dim + self.edge_vector_dim

        self.message_fusion = nn.ModuleList([
            TPP(
                ScalarVector(scalars_in_dim, vectors_in_dim),
                output_dim_dict[0],
                rank=1
            )
        ] + [
            TPP(
                output_dim_dict[0],
                output_dim_dict[0],
                feed_forward=True,
                rank=1
            ) for _ in range(3)
        ])

        # learnable scalar message gating
        if use_scalar_message_attention:
            self.scalar_message_attention = nn.Sequential(
                nn.Linear(output_dim_dict[0].scalar, 1), nn.Sigmoid()
            )

    def message(
            self,
            rep_dict: Dict[int, ScalarVector],
            neighbor_dict: Dict[str, torch.Tensor],
            frame_dict: Dict[int, torch.Tensor],
    ) -> ScalarVector:
        node_rep = rep_dict[0]
        edge_rep = rep_dict[1]
        edge_index = neighbor_dict["N0_0_via_1"].indices()
        row, col = edge_index
        message = node_rep.idx(row).concat(
            (node_rep.idx(col), edge_rep)
        )

        message_residual = self.message_fusion[0](
            message, frame_dict
        )
        for module in self.message_fusion[1:]:
            # exchange geometric messages while maintaining residual connection to original message
            new_message = module(message_residual, frame_dict)
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

        return message_residual

    def forward(
            self,
            rep_dict: Dict[int, ScalarVector],
            neighbor_dict: Dict[str, torch.Tensor],
            frame_dict: Dict[int, torch.Tensor],
    ) -> ScalarVector:
        message = self.message(
            rep_dict, frame_dict=frame_dict, neighbor_dict=neighbor_dict
        )
        return sv_aggregate(message, neighbor_dict["N0_0_via_1"], self.reduce_function)



