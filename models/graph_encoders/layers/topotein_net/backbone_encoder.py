import collections

import torch
from omegaconf import OmegaConf
from torch import nn

from proteinworkshop.models.graph_encoders.gcpnet import GCPNetModel
from topotein.models.utils import to_sse_batch


class BackboneEncoder(nn.Module):
    def __init__(self, in_dims_dict, out_dims_dict, num_layers=6, pretrained_ckpt=None, freeze_encoder=False):
        super().__init__()
        gcpnet_cfg = {
            # Global config
            "num_layers": num_layers,
            "emb_dim": out_dims_dict[0].scalar,
            "node_s_emb_dim": out_dims_dict[0].scalar,  # from emb_dim
            "node_v_emb_dim": out_dims_dict[0].vector,
            "edge_s_emb_dim": out_dims_dict[1].scalar,
            "edge_v_emb_dim": out_dims_dict[1].vector,
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
            "model_cfg": {},
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
        }
        gcpnet_cfg["model_cfg"] = {
            # Input dimensions are resolved via feature config;
            # here we hardcode them based on the provided sample expected dims:
            "h_input_dim": in_dims_dict[0].scalar,   # scalar node features dimension
            "chi_input_dim": in_dims_dict[0].vector,  # vector node features dimension
            # For edge features, note that e_input_dim = scalar_edge_features_dim + num_rbf.
            # Given sample edge_input_dims = [9, 1] and num_rbf = 8, scalar_edge_features_dim is 1.
            "e_input_dim": in_dims_dict[1].scalar + gcpnet_cfg['num_rbf'],    # 1 (scalar edge features) + 8 (num_rbf)
            "xi_input_dim": in_dims_dict[1].vector,   # vector edge features dimension
            # Hidden dimensions as provided in the sample:
            "h_hidden_dim": gcpnet_cfg['node_s_emb_dim'],   # node state hidden dimension
            "chi_hidden_dim": gcpnet_cfg['node_v_emb_dim'],  # node vector hidden dimension
            "e_hidden_dim": gcpnet_cfg['edge_s_emb_dim'],    # edge state hidden dimension
            "xi_hidden_dim": gcpnet_cfg['edge_v_emb_dim'],    # edge vector hidden dimension
            "num_layers": gcpnet_cfg['num_layers'],
            "dropout": 0.0,
        }
        gcpnet_cfg = OmegaConf.create(gcpnet_cfg)
        self.encoder = GCPNetModel(module_cfg=gcpnet_cfg["module_cfg"], model_cfg=gcpnet_cfg["model_cfg"], layer_cfg=gcpnet_cfg["layer_cfg"])
        if pretrained_ckpt is not None:
            encoder_weights = collections.OrderedDict()
            try:
                state_dict = torch.load(pretrained_ckpt, weights_only=False)["state_dict"]
            except RuntimeError:
                state_dict = torch.load(pretrained_ckpt, weights_only=False, map_location='cpu')["state_dict"]
            for k, v in state_dict.items():
                if k.startswith("encoder"):
                    encoder_weights[k.replace("encoder.", "")] = v
            self.encoder.load_state_dict(encoder_weights, strict=False)

            if freeze_encoder:
                self.encoder.requires_grad_(False)

    def forward(self, batch, rank):
        if rank == 2:
            sse_batch = to_sse_batch(batch)
            encoder_out = self.encoder(sse_batch)
        elif rank == 3:
            encoder_out = self.encoder(batch)
            for key in ['h', 'chi', 'e', 'xi']:
                encoder_out[key] = batch[key]
        else:
            raise ValueError(f"Invalid rank: {self.rank}, available ranks: 2 (SSE), 3 (Protein).")

        return encoder_out
