#%%
from typing import Literal

import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.types import ActivationType

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
                self.W_intra[f'{from_rank}->{to_rank}'] = TPP(
                    in_dims=ScalarVector(
                        self.in_dim_dict[from_rank].scalar + self.out_dim_dict[to_rank].scalar,
                        self.in_dim_dict[from_rank].vector + self.out_dim_dict[to_rank].vector,
                        ),
                    out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                    rank=self._get_frame_rank(to_rank, from_rank),
                    activation=activation
                )

                if to_rank not in self.msg_num_to_rank.keys():
                    self.msg_num_to_rank[to_rank] = 0
                self.msg_num_to_rank[to_rank] += 1

        self.W_inter = nn.ModuleDict()
        for to_rank in self.target_ranks:
            self.W_inter[f'{to_rank}'] = TPP(
                in_dims=ScalarVector(
                    self.out_dim_dict[to_rank].scalar * self.msg_num_to_rank[to_rank] + self.in_dim_dict[to_rank].scalar,
                    self.out_dim_dict[to_rank].vector * self.msg_num_to_rank[to_rank] + self.in_dim_dict[to_rank].vector,
                    ),
                out_dims=ScalarVector(self.out_dim_dict[to_rank].scalar, self.out_dim_dict[to_rank].vector),
                rank=to_rank,
                activation=activation
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
                msg_dict[to_rank].append(
                    self.W_intra[f'{from_rank}->{to_rank}'](msg, frame_dict[from_rank])
                )

        update_dict = {}
        for to_rank in msg_dict.keys():
            msg = ScalarVector(*X_dict[to_rank].concat(msg_dict[to_rank]))
            update_dict[to_rank] = self.W_inter[f'{to_rank}'](msg, frame_dict[to_rank])

        return update_dict




