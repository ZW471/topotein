from typing import Dict

from torch import nn
import torch
from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from topotein.models.graph_encoders.layers.topotein_net.embedding import TPPEmbedding
from topotein.models.graph_encoders.layers.topotein_net.message_passing import GeometricMessagePassing, \
    TPPMessagePassing
from topotein.models.graph_encoders.layers.topotein_net.normalization import TPPNorm
from topotein.models.graph_encoders.layers.topotein_net.tpp import TPP


class TopoteinInteraction(nn.Module):
    def __init__(self, in_dim_dict, out_dim_dict, **kwargs):
        super().__init__(**kwargs)
        self.in_dim_dict = in_dim_dict
        self.out_dim_dict = out_dim_dict
        self.mapping_dict_list = [
            {0: [1], 1: [1], 2: [1]},
            {0: [0], 1: [0, 2], 2: [0]},
            {0: [3], 1: [3], 2: [3], 3: [2]},
            # {0: [0], 1: [0], 2: [0], 3: [0]},
            # {0: [1, 2, 3], 1: [0], 2: [0, 3], 3: [2]},
            # {0: [3], 1: [3], 2: [3], 3: [0]}
            # {0: [0, 3], 2: [3]}
        ]
        self.out_ranks = set({})
        for mapping_dict in self.mapping_dict_list:
            for from_rank in mapping_dict.keys():
                for to_rank in mapping_dict[from_rank]:
                    self.out_ranks.add(to_rank)
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
        self.ff_list = nn.ModuleList([
            TPPEmbedding(
                in_dims_dict=self.out_dim_dict,
                out_dims_dict=self.out_dim_dict,
                ranks=self.out_ranks,
                bottleneck=4,
                activation='silu',
                is_batch_embedded=True,
            ) for _ in range(2)
        ])
        self.normalize = TPPNorm(
            dim_dict=self.out_dim_dict
        )


    def forward(self, batch):
        X_residual = batch.embeddings.copy()
        X_dict = batch.embeddings
        # interaction layers: with residual connection
        for gmp, mapping in zip(self.gmp_list, self.mapping_dict_list):
            neighborhood_dict = self.get_neighborhood_dict(batch, mapping)
            u = gmp(X_dict, neighborhood_dict, batch.frame_dict)
            for key, value in u.items():
                X_dict[key] = X_dict[key] + value


        batch.embeddings = X_dict
        # feed-forward layers: no residual connection
        for ff in self.ff_list:
            X_dict = ff(batch)
            for key in X_dict:
                batch.embeddings[key] = X_dict[key]

        for key in X_dict.keys():
            X_dict[key] = X_dict[key] + X_residual[key]

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


class TPPInteraction(nn.Module):
    def __init__(
            self,
            dim_dict: Dict[int, ScalarVector],
    ):
        super().__init__()

        # hyperparameters #
        reduce_function = "sum"
        # geometry-complete message-passing neural network
        self.interaction = TPPMessagePassing(
            input_dim_dict=dim_dict,
            output_dim_dict=dim_dict,
            reduce_function=reduce_function,
            use_scalar_message_attention=True
        )

        self.tpp_norm = TPPNorm(dim_dict)

        # build out feedforward (FF) network modules
        hidden_dims = ScalarVector(dim_dict[0].scalar * 4, dim_dict[0].vector * 2)
        self.feedforward_network = nn.ModuleList([
            TPP(
                in_dims=ScalarVector(dim_dict[0].scalar * 2, dim_dict[0].vector * 2),
                out_dims=hidden_dims,
                rank=0
            ),
            TPP(
                in_dims=hidden_dims,
                out_dims=dim_dict[0],
                rank=0,
                activation="none"
            )
        ])

    def forward(
            self,
            X_dict: Dict[int, ScalarVector],
            neighbor_dict: Dict[str, torch.Tensor],
            frame_dict: Dict[int, torch.Tensor],
    ):
        # forward propagate with interaction module
        hidden_residual = self.interaction(
            X_dict, neighbor_dict, frame_dict
        )

        # aggregate input and hidden features
        hidden_residual = ScalarVector(*hidden_residual.concat((X_dict[0],)))

        # propagate with feedforward layers
        for module in self.feedforward_network:
            hidden_residual = module(
                hidden_residual,
                frame_dict
            )

        X_dict[0] = self.tpp_norm({0: X_dict[0] + hidden_residual})[0]

        return X_dict


if __name__ == '__main__':
    import torch
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

    model = TopoteinInteraction(
        in_dim_dict=hidden_dims,
        out_dim_dict=hidden_dims,
    )
    #%%
    updates = model(batch)
    print("updates: ")
    for key in updates.keys():
        print(f"{key}: {updates[key].shape}")