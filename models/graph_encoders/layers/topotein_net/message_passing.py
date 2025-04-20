#%%
from typing import Literal

import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.types import ActivationType
from topotein.models.graph_encoders.layers.topotein_net.normalization import TPPNorm

from topotein.models.graph_encoders.layers.topotein_net.tpp import TPP
from topotein.models.utils import sv_aggregate


class GeometricMessagePassing(nn.Module):
    def __init__(self, in_dim_dict, out_dim_dict, mapping_dict, agg_reduce='sum',
                 frame_selection: Literal['target', 'source'] = 'target', activation: ActivationType = 'silu', **kwargs):
        super().__init__(**kwargs)
        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.target_ranks = self.out_dim_dict.keys()
        self.mapping_dict = mapping_dict
        self.frame_selection = frame_selection
        self.agg_reduce = agg_reduce
        self.W_intra = nn.ModuleDict()
        self.msg_num_to_rank = {}
        for from_rank in self.mapping_dict.keys():
            for to_rank in self.mapping_dict[from_rank]:
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
        for to_rank in self.out_dim_dict.keys():
            self.W_inter[f'{to_rank}'] = TPP(
                in_dims=ScalarVector(
                    self.out_dim_dict[to_rank].scalar * self.msg_num_to_rank[to_rank],
                    self.out_dim_dict[to_rank].vector * self.msg_num_to_rank[to_rank],
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
            if len(msg_dict[to_rank]) == 1:
                msg = msg_dict[to_rank][0]
            else:
                msg = ScalarVector(*msg_dict[to_rank][0].concat(msg_dict[to_rank][1:]))
            update_dict[to_rank] = self.W_inter[f'{to_rank}'](msg, frame_dict[to_rank])

        return update_dict


class TopoteinMessagePassing(nn.Module):
    def __init__(self, in_dim_dict, out_dim_dict, **kwargs):
        super().__init__(**kwargs)
        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.mapping_dict_list = [
            {0: [1, 2, 3], 1: [0], 2: [0, 3], 3: [2]},
            {0: [0, 3], 1: [0, 3], 2: [0, 1, 3], 3: [0, 1, 2]}
        ]
        self.gmp_list = nn.ModuleList([
            GeometricMessagePassing(
                in_dim_dict=self.in_dim_dict,
                out_dim_dict=self.out_dim_dict,
                mapping_dict=mapping_dict,
                agg_reduce='sum',
                frame_selection='target',
                activation='silu'
            ) for mapping_dict in self.mapping_dict_list
        ])
        self.rank_mapping_dict = {
            0: ScalarVector('h', 'chi'),
            1: ScalarVector('e', 'xi'),
            2: ScalarVector('c', 'rho'),
            3: ScalarVector('p', 'pi'),
        }
        self.normalize = TPPNorm(
            dim_dict=self.out_dim_dict
        )


    def forward(self, batch):
        X_dict = batch.embeddings
        for gmp, mapping in zip(self.gmp_list, self.mapping_dict_list):
            neighborhood_dict = self.get_neighborhood_dict(batch, mapping)
            updates = gmp(X_dict, neighborhood_dict, batch.frame_dict)
            for key in X_dict.keys():
                X_dict[key] = X_dict[key] + updates[key]
        X_dict = self.normalize(X_dict)

        return X_dict

    def get_neighborhood_dict(self, batch, mapping_dict):
        neighborhood_dict = {}
        for from_rank in mapping_dict.keys():
            for to_rank in mapping_dict[from_rank]:
                key = f"N{from_rank}_{to_rank}"
                if hasattr(batch, key):
                    neighborhood_dict[key] = batch[key]
                else:
                    raise KeyError(
                        f"Neighborhood {key} not found in batch. Please check your configuration and try again.")
        return neighborhood_dict


if __name__ == '__main__':
#%%
    from graphein.protein.tensor.data import ProteinBatch
    from topotein.models.graph_encoders.layers.topotein_net.embedding import TPPEmbedding
    from topotein.models.utils import localize
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_dims = {
        0: ScalarVector(128, 16),
        1: ScalarVector(32, 4),
        2: ScalarVector(128, 16),
        3: ScalarVector(128, 16)
    }
    batch: ProteinBatch = torch.load('/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_batch_ccc2.pt', weights_only=False).to(device)
    batch.sse_cell_complex.set_proteins(32, batch.batch)
    batch.frame_dict = {i: localize(batch, rank=i) for i in range(4)}
    model = TPPEmbedding(
        in_dims_dict={
            0: ScalarVector(49, 2),
            1: ScalarVector(1, 1),
            2: ScalarVector(15, 8),
        },
        out_dims_dict=hidden_dims,
        ranks=[0, 1, 2],
        bottleneck=1
    )
    embedded = model(batch)
    batch.embeddings = embedded
    batch.embeddings[3] = ScalarVector(
        torch.randn(32, 128).to(device),
        torch.randn(32, 16, 3).to(device)
    )
    for key in batch.embeddings.keys():
        print(f"{key}: {batch.embeddings[key].shape}")

    #%%

    from topotein.features.topotein_neighborhood_calculator import TopoteinNeighborhoodCalculator
    nc = TopoteinNeighborhoodCalculator(batch.sse_cell_complex)
    neighborhoods = nc.calc_equations([
        "N0_3 = B0_3",
        "N0_2 = B0_2",
        "N0_1 = B0_1",
        "N1_0 = B1_0",
        "N2_0 = B0_2.T",
        "N2_3 = B2_3",
        "N3_2 = B2_3.T",
    ])

    for key, val in neighborhoods.items():
        batch[key] = val

    model = TopoteinMessagePassing(
        in_dim_dict=hidden_dims,
        out_dim_dict=hidden_dims,
    )
#%%
    updates = model(batch)
    print("updates: ")
    for key in updates.keys():
        print(f"{key}: {updates[key].shape}")

