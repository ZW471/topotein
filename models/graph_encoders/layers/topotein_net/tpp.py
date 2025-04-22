import torch
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.utils import get_activations
from proteinworkshop.types import ActivationType
from topotein.models.utils import scalarize, tensorize


class TPP(torch.nn.Module):
    def __init__(self, in_dims: ScalarVector, out_dims: ScalarVector, rank: int, activation:ActivationType='silu', bottleneck=4, **kwargs):
        """
        Initializes a neural network module with configurable layer sizes,
        activation functions, and optional tensorization capabilities.

        This module includes several sequential layers for transforming
        input dimensions across scaler and vector spaces with the ability
        to adjust bottleneck factors, enable or disable tensorization,
        and apply activations dynamically. The architecture integrates
        split operations for downscaling vectors if specified and supports
        gate mechanisms to modulate outputs.

        :param in_dims: Input dimensions including scalar and vector sub-dimensions.
        :param out_dims: Output dimensions including scalar and vector sub-dimensions.
        :param rank: Integer rank parameter used for dimensionality adjustments.
        :param activation: Type of activation function to use in the layers.
        :param bottleneck: Integer factor to reduce vector dimension via the bottleneck mechanism.
        :param kwargs: Additional configuration parameters for enabling tensorization
                       and specifying split behavior in vector dimensions.
        """
        super().__init__()
        assert (
                in_dims.vector % bottleneck == 0
        ), f"Input channel of vector ({in_dims.vector}) must be divisible with bottleneck factor ({bottleneck})"
        self.rank: int = rank
        self.scaler_in_dim: int = in_dims.scalar
        self.scaler_out_dim: int = out_dims.scalar
        self.vector_in_dim: int = in_dims.vector
        self.vector_out_dim: int = out_dims.vector
        self.vector_hidden_dim: int = (
            self.vector_in_dim // bottleneck
            if bottleneck > 1
            else max(self.vector_in_dim, self.vector_out_dim)
        )
        self.activation = get_activations(activation)
        self.enable_tensorization = getattr(kwargs, "enable_tensorization", False)
        self.split_V_down = getattr(kwargs, "split_V_down", True)
        self.scalarization_dim = getattr(kwargs, "scalarization_dim", 3)
        self.feed_forward = getattr(kwargs, "feed_forward", False)

        self.V_down = nn.Linear(self.vector_in_dim, self.vector_hidden_dim, bias=False)
        if self.split_V_down:
            self.V_down_s = nn.Sequential(
                nn.Linear(self.vector_in_dim, self.scalarization_dim),
                self.activation,
            )
        self.V_up = nn.Linear(self.vector_hidden_dim, self.vector_out_dim, bias=False)

        self.S_out = nn.Sequential(
            nn.Linear(
                self.scaler_in_dim + self.vector_hidden_dim + 3 * self.scalarization_dim,
                self.scaler_out_dim
            ),
            self.activation,
            nn.Linear(self.scaler_out_dim, self.scaler_out_dim)
        ) if self.feed_forward else nn.Linear(
            self.scaler_in_dim + self.vector_hidden_dim + 3 * self.scalarization_dim,
            self.scaler_out_dim
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
        """
        Computes the forward pass for a given set of scalar and vector components.

        The function takes a tuple containing scalar and vector components, processes
        them through a series of transformations, and outputs the resulting scalar
        and vector outputs. It performs a variety of operations such as transformations,
        scalarization, normalization, activation, and tensorization (if enabled)
        to derive the final outputs.

        :param s_and_v: A tuple containing scalar (`s`) and vector (`v`) components.
        :param frame_dict: A dictionary mapping rank identifiers to their respective frames,
                           used for operations requiring frame-specific transformations.

        :return: A `ScalarVector` instance containing the computed scalar and vector outputs.
        """
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
        s_gate = self.S_gate(s_out)
        if self.enable_tensorization:
            s_out_to_tensorize = self.S_tensorize(s_out)
            tensorized_s_out = tensorize(s_out_to_tensorize, frames, flattened=True)
            concated_v_t_up = torch.cat([v_t_up, tensorized_s_out], dim=-1)
            v_out = self.V_out(concated_v_t_up).transpose(-1, -2) * s_gate.unsqueeze(-1)
        else:
            v_out = v_t_up.transpose(-1, -2) * s_gate.unsqueeze(-1)

        return ScalarVector(self.activation(s_out), v_out)
