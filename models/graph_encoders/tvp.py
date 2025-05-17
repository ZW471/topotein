from typing import Set, Union

import torch
import torch.nn.functional as F
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

import topotein.models.graph_encoders.layers.tvp as tvp
from proteinworkshop.models.graph_encoders.components import blocks
from proteinworkshop.models.utils import get_aggregation, get_activations
from proteinworkshop.types import EncoderOutput, ActivationType


class TVPGNNModel(torch.nn.Module):
    def __init__(
        self,
        s_in_dim, v_in_dim, s_in_dim_edge, v_in_dim_edge, s_in_dim_sse, v_in_dim_sse,
        s_dim: int = 256,
        v_dim: int = 32,
        s_dim_edge: int = 64,
        v_dim_edge: int = 2,
        s_dim_sse: int = 256,
        v_dim_sse: int = 32,
        num_layers: int = 6,
        pool: str = "sum",
        residual: bool = True,
        activations: ActivationType = "relu",
        **kwargs,
    ):
        """
        Initializes an instance of the GVPGNNModel class with the provided
        parameters.

        :param s_dim: Dimension of the node state embeddings (default: ``128``)
        :type s_dim: int
        :param v_dim: Dimension of the node vector embeddings (default: ``16``)
        :type v_dim: int
        :param s_dim_edge: Dimension of the edge state embeddings
            (default: ``32``)
        :type s_dim_edge: int
        :param v_dim_edge: Dimension of the edge vector embeddings
            (default: ``1``)
        :type v_dim_edge: int
        :param r_max: Maximum distance for Bessel basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_bessel: Number of Bessel basis functions (default: ``8``)
        :type num_bessel: int
        :param num_polynomial_cutoff: Number of polynomial cutoff basis
            functions (default: ``5``)
        :type num_polynomial_cutoff: int
        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param residual: Whether to use residual connections
            (default: ``True``)
        :type residual: bool
        """
        super().__init__()
        s_in_dim = getattr(kwargs, "s_in_dim", s_in_dim)
        v_in_dim = getattr(kwargs, "v_in_dim", v_in_dim)
        s_in_dim_edge = getattr(kwargs, "s_in_dim_edge", s_in_dim_edge)
        v_in_dim_edge = getattr(kwargs, "v_in_dim_edge", v_in_dim_edge)
        s_in_dim_sse = getattr(kwargs, "s_in_dim_sse", s_in_dim_sse)
        v_in_dim_sse = getattr(kwargs, "v_in_dim_sse", v_in_dim_sse)
        
        _V_DIM_IN = (s_in_dim, v_in_dim)
        _E_DIM_IN = (s_in_dim_edge, v_in_dim_edge)
        _SSE_DIM_IN = (s_in_dim_sse, v_in_dim_sse)
        
        s_dim = getattr(kwargs, "s_dim", s_dim)
        v_dim = getattr(kwargs, "v_dim", v_dim)
        s_dim_edge = getattr(kwargs, "s_dim_edge", s_dim_edge)
        v_dim_edge = getattr(kwargs, "v_dim_edge", v_dim_edge)
        s_dim_sse = getattr(kwargs, "s_dim_sse", s_dim_sse)
        v_dim_sse = getattr(kwargs, "v_dim_sse", v_dim_sse)
        _DEFAULT_V_DIM = (s_dim, v_dim)
        _DEFAULT_E_DIM = (s_dim_edge, v_dim_edge)
        _DEFAULT_SSE_DIM = (s_dim_sse, v_dim_sse)
        self.num_layers = num_layers
        activations = (get_activations(activations), None)

        # Node embedding
        self.W_v = torch.nn.Sequential(
            tvp.LayerNorm(_V_DIM_IN),
            tvp.GVP(
                _V_DIM_IN,
                _DEFAULT_V_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )
        self.W_e = torch.nn.Sequential(
            tvp.LayerNorm(_E_DIM_IN),
            tvp.GVP(
                _E_DIM_IN,
                _DEFAULT_E_DIM,
                activations=(None, None),
                vector_gate=True,
            ),
        )
        self.W_sse = torch.nn.Sequential(
            tvp.LayerNorm(_SSE_DIM_IN),
            tvp.GVP(
                _SSE_DIM_IN,
                _DEFAULT_SSE_DIM,
                activations=(None, None),
                vector_gate=True,
            )
        )
        # Stack of GNN layers
        self.layers = torch.nn.ModuleList(
            tvp.TVPConvLayer(
                _DEFAULT_V_DIM,
                _DEFAULT_E_DIM,
                _DEFAULT_SSE_DIM,
                activations=activations,
                vector_gate=True,
                residual=residual,
            )
            for _ in range(num_layers)
        )
        # Output GVP
        self.W_out = torch.nn.Sequential(
            tvp.LayerNorm(_DEFAULT_V_DIM),
            tvp.GVP(
                _DEFAULT_V_DIM,
                (s_dim, 0),
                activations=activations,
                vector_gate=True,
            ),
        )
        # Global pooling/readout function
        self.readout = get_aggregation(pool)

    @property
    def required_batch_attributes(self) -> Set[str]:
        """Required batch attributes for this encoder.

        - ``edge_index`` (shape ``[2, num_edges]``)
        - ``pos`` (shape ``[num_nodes, 3]``)
        - ``x`` (shape ``[num_nodes, num_node_features]``)
        - ``batch`` (shape ``[num_nodes]``)

        :return: _description_
        :rtype: Set[str]
        """
        return {"edge_index", "pos", "x", "batch"}

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GVP-GNN encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        # Edge features

        h_V = self.W_v((batch.x, batch.x_vector_attr))
        h_E = self.W_e((batch.edge_attr, batch.edge_vector_attr))
        h_SSE = self.W_sse((batch.sse_attr, batch.sse_vector_attr))

        for layer in self.layers:
            h_V, h_SSE = layer(batch.sse_cell_complex, h_V, batch.edge_index, h_E, h_SSE)

        out = self.W_out(h_V)

        return EncoderOutput(
            {
                "node_embedding": out,
                "graph_embedding": self.readout(
                    out, batch.batch
                ),  # (n, d) -> (batch_size, d)
                # "pos": pos  # TODO it is possible to output pos with GVP if needed
            }
        )


if __name__ == "__main__":
    import hydra
    import omegaconf

    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.SRC_PATH / "config" / "encoder" / "gvp.yaml"
    )
    enc = hydra.utils.instantiate(cfg)
    print(enc)
