import torch
from beartype.typing import List
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers import gcp
from proteinworkshop.models.utils import get_activations, centralize
from proteinworkshop.types import EncoderOutput
from topotein.models.graph_encoders.layers.topotein_net.embedding import TPPEmbedding
from topotein.models.graph_encoders.layers.topotein_net.interaction import TopoteinInteraction
from topotein.models.graph_encoders.layers.topotein_net.tpp import TPP
from topotein.models.utils import tensorize, localize


class TopoteinNetModel(nn.Module):
    @property
    def required_batch_attributes(self) -> List[str]:
        return ["edge_index", "pos", "x", "batch"]

    def __init__(self, in_dims_dict=None, out_dims_dict=None, num_layers=None, activation=None, **kwargs):
        super().__init__()

        assert all(
            [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        ), "All required TopoteinNet `DictConfig`s must be provided."
        module_cfg = kwargs["module_cfg"]
        model_cfg = kwargs["model_cfg"]
        layer_cfg = kwargs["layer_cfg"]

        if in_dims_dict is None:
            self.in_dims_dict = {
                0: ScalarVector(model_cfg['h_input_dim'], model_cfg['chi_input_dim']),
                1: ScalarVector(model_cfg['e_input_dim'], model_cfg['xi_input_dim']),
                2: ScalarVector(model_cfg['c_input_dim'], model_cfg['rho_input_dim']),
                3: ScalarVector(model_cfg['p_input_dim'], model_cfg['pi_input_dim']),
            }
        else:
            self.in_dims_dict = in_dims_dict
        if out_dims_dict is None:
            self.out_dims_dict = {
                0: ScalarVector(model_cfg['h_hidden_dim'], model_cfg['chi_hidden_dim']),
                1: ScalarVector(model_cfg['e_hidden_dim'], model_cfg['xi_hidden_dim']),
                2: ScalarVector(model_cfg['c_hidden_dim'], model_cfg['rho_hidden_dim']),
                3: ScalarVector(model_cfg['p_hidden_dim'], model_cfg['pi_hidden_dim']),
            }
        else:
            self.out_dims_dict = out_dims_dict
        if num_layers is None:
            self.num_layers = model_cfg['num_layers']
        else:
            self.num_layers = num_layers
        if activation is None:
            self.activation = module_cfg['nonlinearities'][0]
        else:
            self.activation = activation

        self.pr_pre_tensorize = nn.Sequential(
            nn.Linear(model_cfg['p_input_dim'], model_cfg['pi_hidden_dim'] * 3),
            get_activations(self.activation),
            nn.Linear(model_cfg['pi_hidden_dim'] * 3, model_cfg['pi_hidden_dim'] * 3),
            get_activations(self.activation)
        )

        self.embed = TPPEmbedding(
            in_dims_dict=self.in_dims_dict,
            out_dims_dict=self.out_dims_dict,
            ranks=[0, 1, 2, 3],
            bottleneck=1,
            activation=self.activation
        )
        # interactions layers

        self.interaction_layers = nn.ModuleList(
            TopoteinInteraction(
                in_dim_dict=self.out_dims_dict,
                out_dim_dict=self.out_dims_dict
            )
            for _ in range(model_cfg.num_layers)
        )

        self.invariant_node_projection = nn.ModuleList(
            [
                gcp.GCPLayerNorm(self.out_dims_dict[0]),
                TPP(in_dims=self.out_dims_dict[0], out_dims=self.out_dims_dict[0], rank=0, activation=self.activation),
            ]
        )

        self.invariant_protein_projection = nn.ModuleList(
            [
                gcp.GCPLayerNorm(self.out_dims_dict[3]),
                TPP(in_dims=self.out_dims_dict[3], out_dims=self.out_dims_dict[3], rank=3, activation=self.activation),
            ]
        )


    def forward(self, batch):
        pos_centroid, batch.pos = centralize(batch, batch_index=batch.batch, key="pos")
        batch.frame_dict = {i: localize(batch, rank=i) for i in range(4)}
        batch['pr_vector_attr'] = tensorize(self.pr_pre_tensorize(batch.pr_attr), batch.frame_dict[3], flattened=True).transpose(-1, -2)
        batch.embeddings = self.embed(batch)

        for layer in self.interaction_layers:
            X_dict = layer(batch)
            for key in X_dict:
                batch.embeddings[key] = X_dict[key]

        h_out = self.invariant_node_projection[0](batch.embeddings[0])
        h_out, _ = self.invariant_node_projection[1](h_out, batch.frame_dict[0])
        p_out = self.invariant_protein_projection[0](batch.embeddings[3])
        p_out, _ = self.invariant_protein_projection[1](p_out, batch.frame_dict[3])

        encoder_outputs = {"node_embedding": h_out, "graph_embedding": p_out}
        return EncoderOutput(encoder_outputs)
