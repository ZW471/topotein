import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from topotein.models.graph_encoders.layers.topotein_net.tpp import TPP
from topotein.models.utils import DEFAULT_RANK_MAPPING


class TPPEmbedding(torch.nn.Module):
    def __init__(self, in_dims_dict, out_dims_dict, ranks, rank_mapping_dict=None, bottleneck=1, activation="silu"):
        super().__init__()
        self.ranks = ranks
        self.in_dims_dict = in_dims_dict
        self.out_dims_dict = out_dims_dict
        self.models = nn.ModuleDict()
        if rank_mapping_dict is None:
            self.rank_mapping_dict = DEFAULT_RANK_MAPPING
        else:
            self.rank_mapping_dict = rank_mapping_dict
        for rank in ranks:
            self.models[str(rank)] = TPP(
                in_dims=in_dims_dict[rank],
                out_dims=out_dims_dict[rank],
                rank=rank,
                bottleneck=bottleneck,
                activation=activation
            )

    def forward(self, batch):
        out = {}
        for rank in self.ranks:
            out[rank] = self.models[str(rank)](
                s_and_v=ScalarVector(
                    batch[self.rank_mapping_dict[rank].scalar],
                    batch[self.rank_mapping_dict[rank].vector]
                ),
                frame_dict=batch.frame_dict
            )
        return out
