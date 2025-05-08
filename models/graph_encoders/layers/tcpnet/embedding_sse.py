from typing import Union, Tuple

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped, Float
from torch_geometric.data import Batch

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPEmbedding, GCPLayerNorm
from topotein.models.graph_encoders.layers.tcpnet.tcp import TCP


class TCPEmbedding(GCPEmbedding):
    def __init__(self, sse_input_dims, sse_hidden_dims, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.node_input_dims = kwargs.get("node_input_dims")
        self.node_hidden_dims = kwargs.get("node_hidden_dims")
        self.edge_input_dims = kwargs.get("edge_input_dims")
        self.edge_hidden_dims = kwargs.get("edge_hidden_dims")
        self.sse_input_dims = sse_input_dims
        self.sse_hidden_dims = sse_hidden_dims

        use_gcp_norm = kwargs.get("use_gcp_norm", True)
        nonlinearities = kwargs.get("nonlinearities", ("silu", "silu"))
        cfg = kwargs.get("cfg", None)

        self.sse_normalization = GCPLayerNorm(
            self.sse_input_dims if self.pre_norm else self.sse_hidden_dims,
            use_gcp_norm=use_gcp_norm
        )

        self.node_embedding = TCP(
            self.node_input_dims,
            self.node_hidden_dims,
            nonlinearities=("none", "none"),
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.edge_embedding = TCP(
            self.edge_input_dims,
            self.edge_hidden_dims,
            nonlinearities=nonlinearities,
            scalar_gate=cfg.scalar_gate,
            vector_gate=cfg.vector_gate,
            enable_e3_equivariance=cfg.enable_e3_equivariance,
        )

        self.sse_embedding = TCP(
            self.sse_input_dims,
            self.sse_hidden_dims,
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
        ]
    ]:
        if self.atom_embedding is not None:
            node_rep = ScalarVector(self.atom_embedding(batch.h), batch.chi)
        else:
            node_rep = ScalarVector(batch.h, batch.chi)

        edge_rep = ScalarVector(batch.e, batch.xi)
        sse_rep = ScalarVector(batch.c, batch.rho)

        edge_vectors = (
                batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]
        )  # [n_edges, 3]
        edge_lengths = torch.linalg.norm(edge_vectors, dim=-1)  # [n_edges, 1]
        edge_rep = ScalarVector(
            torch.cat((edge_rep.scalar, self.radial_embedding(edge_lengths)), dim=-1),
            edge_rep.vector,
        )

        edge_rep = (
            edge_rep.scalar if not self.edge_embedding.vector_input_dim else edge_rep
        )
        node_rep = (
            node_rep.scalar if not self.node_embedding.vector_input_dim else node_rep
        )
        sse_rep = (
            sse_rep.scalar if not self.sse_embedding.vector_input_dim else sse_rep
        )

        if self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)
            sse_rep = self.sse_normalization(sse_rep)

        edge_rep = self.edge_embedding(
            edge_rep,
            batch.frame_dict[1],
        )
        node_rep = self.node_embedding(
            node_rep,
            batch.frame_dict[0],
        )
        sse_rep = self.sse_embedding(
            sse_rep,
            batch.frame_dict[2]
        )

        if not self.pre_norm:
            edge_rep = self.edge_normalization(edge_rep)
            node_rep = self.node_normalization(node_rep)
            sse_rep = self.sse_normalization(sse_rep)


        return node_rep, edge_rep, sse_rep
