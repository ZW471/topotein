import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import ActivationType
from topotein.models.utils import scalarize, tensorize


class TPP(torch.nn.Module):
    def __init__(self, in_dims: ScalarVector, out_dims: ScalarVector, rank: int, activation:ActivationType='silu', bottleneck=4, **kwargs):
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
        self.enable_tensorization = getattr(kwargs, "enable_tensorization", False)
        self.split_V_down = getattr(kwargs, "split_V_down", True)

        self.V_down = nn.Sequential(
            nn.Linear(self.vector_in_dim, self.vector_hidden_dim),
            self.activation,
        )
        if self.split_V_down:
            self.V_down_s = nn.Sequential(
                nn.Linear(self.vector_in_dim, self.vector_hidden_dim),
                self.activation,
            )
        self.V_up = nn.Sequential(
            nn.Linear(self.vector_hidden_dim, self.vector_out_dim),
            self.activation,
        )

        self.S_out = nn.Sequential(
            nn.Linear(self.scaler_in_dim + self.vector_hidden_dim * 4, self.scaler_out_dim),
            self.activation,
            nn.Linear(self.scaler_out_dim, self.scaler_out_dim)
        )
        if self.enable_tensorization:
            self.V_out = nn.Sequential(
                nn.Linear(self.vector_out_dim * 2, self.vector_out_dim),
                self.activation,
                nn.Linear(self.vector_out_dim, self.vector_out_dim)
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
        if self.split_V_down:
            scalarized_z = scalarize(self.V_down_s(v_t), frames)
        else:
            scalarized_z = scalarize(z_t, frames)
        normalized_z = torch.norm(z_t, dim=1)

        concated_s = torch.cat([s, scalarized_z, normalized_z], dim=-1)
        s_out = self.S_out(concated_s)

        non_linear_s_out = self.activation(s_out)
        s_gate = self.S_gate(non_linear_s_out)
        if self.enable_tensorization:
            s_out_to_tensorize = self.S_tensorize(non_linear_s_out)
            tensorized_s_out = tensorize(s_out_to_tensorize, frames, flattened=True)
            concated_v_t_up = torch.cat([v_t_up, tensorized_s_out], dim=-1)
            v_out = self.V_out(concated_v_t_up).transpose(-1, -2) * s_gate.unsqueeze(-1)
        else:
            v_out = v_t_up.transpose(-1, -2) * s_gate.unsqueeze(-1)

        return ScalarVector(s_out, v_out)
