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
from topotein.models.graph_encoders.layers.tcpnet.interaction import TCPInteractions
from topotein.models.graph_encoders.layers.tcpnet.embedding import TCPEmbedding


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
        self.pr_update = module_cfg.pr_update
        self.use_tnn_pooling = module_cfg.use_tnn_pooling

        # Feature dimensionalities
        edge_input_dims = ScalarVector(model_cfg.e_input_dim, model_cfg.xi_input_dim)
        node_input_dims = ScalarVector(model_cfg.h_input_dim, model_cfg.chi_input_dim)
        sse_input_dims = ScalarVector(model_cfg.c_input_dim, model_cfg.rho_input_dim)
        pr_input_dims = ScalarVector(model_cfg.p_input_dim, model_cfg.pi_input_dim)
        self.edge_dims = ScalarVector(model_cfg.e_hidden_dim, model_cfg.xi_hidden_dim)
        self.node_dims = ScalarVector(model_cfg.h_hidden_dim, model_cfg.chi_hidden_dim)
        self.sse_dims = ScalarVector(model_cfg.c_hidden_dim, model_cfg.rho_hidden_dim)
        self.pr_dims = ScalarVector(model_cfg.p_hidden_dim, model_cfg.pi_hidden_dim)

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
            pr_input_dims=pr_input_dims,
            node_hidden_dims=self.node_dims,
            edge_hidden_dims=self.edge_dims,
            sse_hidden_dims=self.sse_dims,
            pr_hidden_dims=self.pr_dims,
            cfg=module_cfg
        )

        # Message-passing layers
        self.interaction_layers = nn.ModuleList(
            TCPInteractions(
                self.node_dims,
                self.edge_dims,
                self.sse_dims,
                self.pr_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        if self.use_tnn_pooling and not self.pr_update:
            module_cfg_with_pr_pooling = module_cfg.copy()
            module_cfg_with_pr_pooling.update({"pr_update": True})
            self.interaction_layers[-1] = TCPInteractions(
                self.node_dims,
                self.edge_dims,
                self.sse_dims,
                self.pr_dims,
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
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
        self.readout = nn.ModuleList([
            gcp.GCPLayerNorm(self.pr_dims),
            TCP(
                self.pr_dims,
                ScalarVector(self.pr_dims.scalar, 0),
                nonlinearities=tuple(module_cfg.nonlinearities),
                scalar_gate=module_cfg.scalar_gate,
                vector_gate=module_cfg.vector_gate,
                enable_e3_equivariance=module_cfg.enable_e3_equivariance
            ),
        ]) if self.use_tnn_pooling else get_aggregation(
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
        batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho, batch.p, batch.pi = (
            batch.x,
            batch.x_vector_attr,
            batch.edge_attr,
            batch.edge_vector_attr,
            batch.sse_attr,
            batch.sse_vector_attr,
            batch.pr_attr,
            batch.pr_vector_attr
        )
        node_mask = getattr(batch, "mask", None)
        # Craft complete local frames corresponding to each edge
        batch.frame_dict = {rank: localize(batch, rank, node_mask) for rank in range(4)}

        # Embed node and edge input features
        (h, chi), (e, xi), (c, rho), (p, pi) = self.tcp_embedding(batch)

        # Update graph features using a series of geometric message-passing layers
        for layer in self.interaction_layers:
            new_reps = layer(  # TODO: also update c, rho?
                node_rep=ScalarVector(h, chi),
                edge_rep=ScalarVector(e, xi),
                sse_rep=ScalarVector(c, rho),
                pr_rep=ScalarVector(p, pi),
                frame_dict=batch.frame_dict,
                edge_index=batch.edge_index,
                node_mask = node_mask,
                node_pos=batch.pos,
                ccc=batch.sse_cell_complex,
                node_to_sse_mapping=batch.N0_2,
                sse_to_node_mapping=batch.N2_0,
                edge_to_sse_mapping=batch.N1_2,
                pr_to_sse_mapping=batch.N3_2,
                node_to_pr_mapping=batch.N0_3,
                sse_to_pr_mapping=batch.N2_3,
                sse_to_edge_outer_mapping=batch.N2_1_outer,
                edge_to_sse_outer_mapping=batch.N1_2_outer
            )
            (h, chi), batch.pos = new_reps[0], new_reps[1]
            if self.sse_update:
                (c, rho) = new_reps[2]
            if self.pr_update:
                (p, pi) = new_reps[3]

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho, batch.p, batch.pi = h, chi, e, xi, c, rho, p, pi

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

        if self.use_tnn_pooling:
            graph_out = self.readout[0](ScalarVector(p, pi))
            encoder_outputs["graph_embedding"] =  self.readout[1](graph_out, batch.frame_dict[3])
        else:
            encoder_outputs["graph_embedding"] = self.readout(
                out, batch.batch
            )  # (n, d) -> (batch_size, d)
        return EncoderOutput(encoder_outputs)


if __name__ == "__main__":

    from omegaconf import OmegaConf
    from proteinworkshop.constants import PROJECT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch: ProteinBatch = torch.load(f'{PROJECT_PATH}/../test/data/sample_batch/sample_batch_for_tcp.pt', weights_only=False).to(device)
    batch.mask = torch.randn(batch.x.shape[0], device=device) > -.3
    print(batch)

    cfg = OmegaConf.create({
        "_target_": "topotein.models.graph_encoders.tcpnet.TCPNetModel",
        "features": {
            # These will be manually injected via validate_gcpnet_config()
            "vector_node_features": ["orientation"],
            "vector_edge_features": ["edge_vectors"],
            "vector_cell_features": ["cell_vectors"],
            "neighborhoods": ["N2_0 = B2.T @ B1.T / 2"],
            # (Optional placeholders for scalar features if needed by resolve_feature_config_dim)
            "scalar_node_features": None,
            "scalar_edge_features": None,
            "scalar_cell_features": None,
        },
        # Global config
        "num_layers": 6,
        "emb_dim": 128,
        "node_s_emb_dim": 128,  # from emb_dim
        "node_v_emb_dim": 16,
        "edge_s_emb_dim": 32,
        "edge_v_emb_dim": 4,
        "cell_s_emb_dim": 64,
        "cell_v_emb_dim": 8,
        "r_max": 10.0,
        "num_rbf": 8,
        "activation": "silu",
        "pool": "sum",
        # Module config
        "module_cfg": {
            "norm_pos_diff": True,
            "scalar_gate": 0,
            "vector_gate": True,
            "scalar_nonlinearity": "silu",  # resolved from activation
            "vector_nonlinearity": "silu",
            "nonlinearities": ["silu", "silu"],
            "r_max": 10.0,
            "num_rbf": 8,
            "bottleneck": 4,
            "vector_linear": True,
            "vector_identity": True,
            "default_bottleneck": 4,
            "predict_node_positions": False,  # input node positions will not be updated
            "predict_node_rep": True,         # final projection of node features will be performed
            "node_positions_weight": 1.0,
            "update_positions_with_vector_sum": False,
            "enable_e3_equivariance": False,
            "pool": "sum",
        },
        # Model config
        "model_cfg": {
            # Input dimensions are resolved via feature config;
            # here we hardcode them based on the provided sample expected dims:
            "h_input_dim": 49,   # scalar node features dimension
            "chi_input_dim": 2,  # vector node features dimension
            # For edge features, note that e_input_dim = scalar_edge_features_dim + num_rbf.
            # Given sample edge_input_dims = [9, 1] and num_rbf = 8, scalar_edge_features_dim is 1.
            "e_input_dim": 9,    # 1 (scalar edge features) + 8 (num_rbf)
            "xi_input_dim": 1,   # vector edge features dimension
            "c_input_dim": 15,   # scalar cell features dimension
            "rho_input_dim": 8,  # vector cell features dimension
            # Hidden dimensions as provided in the sample:
            "h_hidden_dim": 128,   # node state hidden dimension
            "chi_hidden_dim": 16,  # node vector hidden dimension
            "e_hidden_dim": 32,    # edge state hidden dimension
            "xi_hidden_dim": 4,    # edge vector hidden dimension
            "c_hidden_dim": 64,    # cell state hidden dimension
            "rho_hidden_dim": 8,   # cell vector hidden dimension
            "num_layers": 6,
            "dropout": 0.0,
        },
        # Layer config
        "layer_cfg": {
            "pre_norm": False,
            "use_gcp_norm": True,
            "use_gcp_dropout": True,
            "use_scalar_message_attention": True,
            "num_feedforward_layers": 2,
            "dropout": 0.0,
            "nonlinearity_slope": 1e-2,
            "mp_cfg": {
                "edge_encoder": False,
                "edge_gate": False,
                "num_message_layers": 4,
                "message_residual": 0,
                "message_ff_multiplier": 1,
                "self_message": True,
            },
        },
    })

    model = TCPNetModel(module_cfg=cfg["module_cfg"], model_cfg=cfg["model_cfg"], layer_cfg=cfg["layer_cfg"])

    print(model)

    result = model(batch)
    print(result)
    embedding = result["node_embedding"]

        # Step 1: Create a random 3x3 matrix
    A = torch.randn(3, 3)

    # Step 2: QR decomposition
    Q, R = torch.linalg.qr(A)

    # Step 3: Ensure proper orientation (optional)
    # Make determinant = 1 (if necessary, flip a column)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]

    batch.pos = torch.matmul(batch.pos, Q)
    from proteinworkshop import constants

    cfg = OmegaConf.load(
        constants.PROJECT_PATH
        / "proteinworkshop"
        / "config"
        / "features"
        / "ca_bb_sse.yaml"
    )
    cfg['vector_node_features'] += ['orientation']
    cfg['vector_edge_features'] += ["edge_vectors"]
    cfg['vector_sse_features'] += ["sse_vectors"]
    featuriser = hydra.utils.instantiate(cfg)
    batch = featuriser(batch)
    print(batch)
    result2 = model(batch)
    embedding2 = result2["node_embedding"]
    print(embedding)
    print(embedding2)
    assert torch.allclose(embedding, embedding2, atol=1e-5)
