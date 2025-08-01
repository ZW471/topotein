from functools import partial
from typing import Union
import hydra
import torch
import torch.nn as nn
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from torch_geometric.data import Batch

import proteinworkshop.models.graph_encoders.layers.gcp as gcp
from proteinworkshop.models.graph_encoders.components.wrappers import (
    ScalarVector,
)
from proteinworkshop.models.graph_encoders.gcpnet import GCPNetModel
from proteinworkshop.models.utils import (
    centralize,
    decentralize,
    get_aggregation,
)
from topotein.models.graph_encoders.layers.tcpnet.tcp import TCP
from topotein.models.utils import localize
from proteinworkshop.types import EncoderOutput
from topotein.models.graph_encoders.layers.tcpnet.interaction_sse import TCPInteractions
from topotein.models.graph_encoders.layers.tcpnet.embedding_sse import TCPEmbedding


class TCPNetModel(GCPNetModel):
    def __init__(
            self,
            num_layers: int = 5,
            node_s_emb_dim: int = 128,
            node_v_emb_dim: int = 16,
            edge_s_emb_dim: int = 32,
            edge_v_emb_dim: int = 4,
            cell_s_emb_dim: int = 64,
            cell_v_emb_dim: int = 8,
            r_max: float = 10.0,
            num_rbf: int = 8,
            activation: str = "silu",
            pool: str = "sum",
            # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
            # They are simply listed here to highlight key available arguments
            **kwargs,
    ):
        """
        Initializes an instance of the GCPNetModel class with the provided
        parameters.
        Note: Each of the model's keyword arguments listed here
        are also referenced in the corresponding `DictConfigs` within `kwargs`.
        They are simply listed here to highlight some of the key arguments available.
        See `proteinworkshop/config/encoder/gcpnet.yaml` for a full list of all available arguments.

        :param num_layers: Number of layers in the model (default: ``5``)
        :type num_layers: int
        :param node_s_emb_dim: Dimension of the node state embeddings (default: ``128``)
        :type node_s_emb_dim: int
        :param node_v_emb_dim: Dimension of the node vector embeddings (default: ``16``)
        :type node_v_emb_dim: int
        :param edge_s_emb_dim: Dimension of the edge state embeddings
            (default: ``32``)
        :type edge_s_emb_dim: int
        :param edge_v_emb_dim: Dimension of the edge vector embeddings
            (default: ``4``)
        :type edge_v_emb_dim: int
        :param r_max: Maximum distance for radial basis functions
            (default: ``10.0``)
        :type r_max: float
        :param num_rbf: Number of radial basis functions (default: ``8``)
        :type num_rbf: int
        :param activation: Activation function to use in each GCP layer (default: ``silu``)
        :type activation: str
        :param pool: Global pooling method to be used
            (default: ``"sum"``)
        :type pool: str
        :param kwargs: Primary model arguments in the form of the
            `DictConfig`s `module_cfg`, `model_cfg`, and `layer_cfg`, respectively
        :type kwargs: dict
        """
        super().__init__(**kwargs)

        assert all(
            [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        ), "All required GCPNet `DictConfig`s must be provided."
        module_cfg = kwargs["module_cfg"]
        model_cfg = kwargs["model_cfg"]
        layer_cfg = kwargs["layer_cfg"]

        self.predict_node_pos = module_cfg.predict_node_positions
        self.predict_node_rep = module_cfg.predict_node_rep

        self.sse_update = module_cfg.sse_update
        self.use_tnn_pooling = module_cfg.use_tnn_pooling

        # Feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim, model_cfg.chi_input_dim)
        sse_input_dims = ScalarVector(model_cfg.c_input_dim, model_cfg.rho_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.sse_dims = ScalarVector(model_cfg.c_hidden_dim, model_cfg.rho_hidden_dim)

        # Position-wise operations
        self.centralize = partial(centralize, key="pos")
        self.localize = partial(localize, norm_pos_diff=module_cfg.norm_pos_diff)
        self.decentralize = partial(decentralize, key="pos")

        # Input embeddings
        self.gcp_embedding = None
        self.tcp_embedding = TCPEmbedding(
            node_input_dims=node_input_dims,
            edge_input_dims=edge_input_dims,
            sse_input_dims=sse_input_dims,
            node_hidden_dims=self.node_dims,
            edge_hidden_dims=self.edge_dims,
            sse_hidden_dims=self.sse_dims,
            cfg=module_cfg
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            TCPInteractions(
                self.node_dims,
                self.edge_dims,
                self.sse_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        if self.predict_node_rep:
            # Predictions
            self.invariant_node_projection = nn.ModuleList(
                [
                    gcp.GCPLayerNorm(self.node_dims),
                    TCP(
                        self.node_dims,
                        ScalarVector(self.node_dims.scalar, 0),
                        nonlinearities=tuple(module_cfg.nonlinearities),
                        scalar_gate=module_cfg.scalar_gate,
                        vector_gate=module_cfg.vector_gate,
                        enable_e3_equivariance=module_cfg.enable_e3_equivariance
                    ),
                ]
            )

        # Global pooling/readout function
        self.readout = get_aggregation(
            module_cfg.pool
        )

    @jaxtyped(typechecker=typechecker)
    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the GCPNet encoder.

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
        # Centralize node positions to make them translation-invariant
        pos_centroid, batch.pos = self.centralize(batch, batch_index=batch.batch)

        # Install `h`, `chi`, `e`, and `xi` using corresponding features built by the `FeatureFactory`
        batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = (
            batch.x,
            batch.x_vector_attr,
            batch.edge_attr,
            batch.edge_vector_attr,
            batch.sse_attr,
            batch.sse_vector_attr,
        )
        node_mask = getattr(batch, "mask", None)
        # Craft complete local frames corresponding to each edge
        batch.frame_dict = {rank: localize(batch, rank, node_mask) for rank in range(4)}

        # Embed node and edge input features
        (h, chi), (e, xi), (c, rho) = self.tcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            new_reps = layer(  # TODO: also update c, rho?
                node_rep=ScalarVector(h, chi),
                edge_rep=ScalarVector(e, xi),
                sse_rep=ScalarVector(c, rho),
                frame_dict=batch.frame_dict,
                edge_index=batch.edge_index,
                node_mask = node_mask,
                node_pos=batch.pos,
                ccc=batch.sse_cell_complex,
                node_to_sse_mapping=batch.N0_2,
                sse_to_node_mapping=batch.N2_0,
                edge_to_sse_mapping=batch.N1_2,
                sse_to_edge_outer_mapping=batch.N2_1_outer,
                edge_to_sse_outer_mapping=batch.N1_2_outer
            )
            (h, chi), batch.pos = new_reps[0], new_reps[1]
            if self.sse_update:
                (c, rho) = new_reps[2]

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = h, chi, e, xi, c, rho

        # initialize encoder outputs
        encoder_outputs = {}

        # when updating node positions, decentralize updated positions to make their updates translation-equivariant
        if self.predict_node_pos:
            batch.pos = self.decentralize(
                batch, batch_index=batch.batch, entities_centroid=pos_centroid
            )
            if self.predict_node_rep:
                # prior to scalar node predictions, re-derive local frames after performing all node position updates
                _, centralized_node_pos = self.centralize(
                    batch, batch_index=batch.batch
                )
                batch.frame_dict[1] = self.localize(centralized_node_pos, batch.edge_index)
            encoder_outputs["pos"] = batch.pos  # (n, 3) -> (batch_size, 3)

        # Summarize intermediate node representations as final predictions
        out = h
        if self.predict_node_rep:
            out = self.invariant_node_projection[0](
                ScalarVector(h, chi)
            )  # e.g., GCPLayerNorm()
            out = self.invariant_node_projection[1](
                out, batch.frame_dict[0]
            )  # e.g., GCP((h, chi)) -> h'
        encoder_outputs["node_embedding"] = out

        encoder_outputs["graph_embedding"] = self.readout(
                out, batch.batch
            )  # (n, d) -> (batch_size, d)
        return EncoderOutput(encoder_outputs)


