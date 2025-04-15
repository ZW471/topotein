import collections

import torch
from omegaconf import OmegaConf
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.gcpnet import GCPNetModel
from proteinworkshop.models.utils import get_activations
from topotein.models.utils import scalarize, tensorize, DEFAULT_RANK_MAPPING, to_sse_batch


class TPP(torch.nn.Module):
    def __init__(self, in_dims: ScalarVector, out_dims: ScalarVector, rank: int, activation='silu', bottleneck=4):
        super().__init__()
        assert (
                in_dims.vector % bottleneck == 0
        ), f"Input channel of vector ({in_dims.vector}) must be divisible with bottleneck factor ({bottleneck})"
        self.rank: int = rank
        self.scaler_in_dim: int = in_dims.scalar
        self.scaler_out_dim: int = out_dims.scalar
        self.vector_in_dim: int = in_dims.vector
        self.vector_hidden_dim: int = self.vector_in_dim // bottleneck
        self.vector_out_dim: int = out_dims.vector
        self.activation = get_activations(activation)

        self.V_down = nn.Sequential(
            nn.Linear(self.vector_in_dim, self.vector_hidden_dim),
            self.activation,
        )
        self.V_up = nn.Sequential(
            nn.Linear(self.vector_hidden_dim, self.vector_out_dim),
            self.activation,
        )
        self.V_out = nn.Sequential(
            nn.Linear(self.vector_out_dim * 2, self.vector_out_dim),
            self.activation,
            nn.Linear(self.vector_out_dim, self.vector_out_dim)
        )

        self.S_out = nn.Sequential(
            nn.Linear(self.scaler_in_dim + self.vector_hidden_dim * 4, self.scaler_out_dim),
            self.activation,
            nn.Linear(self.scaler_out_dim, self.scaler_out_dim)
        )
        self.S_tensorize = nn.Sequential(
            nn.Linear(self.scaler_out_dim, self.vector_out_dim * 3),
            self.activation
        )
        self.S_gate = nn.Sequential(
            nn.Linear(self.scaler_out_dim, self.vector_out_dim),
            nn.Sigmoid(),
        )


    def forward(self, s_and_v: ScalarVector, frame_dict):
        frames = frame_dict[self.rank]
        s, v = s_and_v

        v_t = v.transpose(-1, -2)
        z_t = self.V_down(v_t)
        v_t_up = self.V_up(z_t)
        scalarized_z = scalarize(z_t, frames)
        normalized_z = torch.norm(z_t, dim=1)

        concated_s = torch.cat([s, scalarized_z, normalized_z], dim=-1)
        s_out = self.S_out(concated_s)

        non_linear_s_out = self.activation(s_out)
        s_gate = self.S_gate(non_linear_s_out)
        s_out_to_tensorize = self.S_tensorize(non_linear_s_out)
        tensorized_s_out = tensorize(s_out_to_tensorize, frames, flattened=True)
        concated_v_t_up = torch.cat([v_t_up, tensorized_s_out], dim=-1)
        v_out = self.V_out(concated_v_t_up).transpose(-1, -2) * s_gate.unsqueeze(-1)

        return ScalarVector(s_out, v_out)


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
