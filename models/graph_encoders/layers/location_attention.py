"""
Geometry Location Attention modules.

This module provides implementations of geometry-aware location attention mechanisms.
"""
import torch
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_softmax
from proteinworkshop.models.utils import get_activations
from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from topotein.models.utils import get_com, sv_attention


class GeometryLocationAttentionHead(nn.Module):
    """
    A single head of geometry-aware location attention.

    This module computes attention weights based on the geometric relationship
    between two entities (from_rank and to_rank) connected by an incidence matrix.
    """

    def __init__(
            self,
            from_vec_dim: int,
            to_vec_dim: int,
            hidden_dim: int = 3,
            activation: str = 'silu',
            higher_to_lower: bool = True,
    ):
        """
        Initialize the GeometryLocationAttentionHead.

        Args:
            from_vec_dim: Dimension of the vector features of the from_rank entities
            to_vec_dim: Dimension of the vector features of the to_rank entities
            hidden_dim: Dimension of the hidden representation (default: 3)
            activation: Activation function to use (default: 'silu')
        """
        super().__init__()
        self.from_vec_dim = from_vec_dim
        self.to_vec_dim = to_vec_dim
        self.hidden_dim = hidden_dim

        # Linear projections for vector features
        if self.hidden_dim is not None:
            self.from_proj = nn.Linear(from_vec_dim, hidden_dim, bias=False)
            self.to_proj = nn.Linear(to_vec_dim, hidden_dim, bias=False)
            self.attn_proj = nn.Linear(hidden_dim * 2 * 3 + 2 * 3, 1, bias=False)
        else:
            self.from_proj = nn.Identity()
            self.to_proj = nn.Identity()
            self.attn_proj = nn.Linear((from_vec_dim + to_vec_dim) * 3 + 2 * 3, 1, bias=False)
        self.activation = get_activations(activation)
        self.higher_to_lower = higher_to_lower

    def forward(
            self,
            from_rank_sv: ScalarVector,
            to_rank_sv: ScalarVector,
            incidence_matrix,
            from_frame: torch.Tensor,
            to_frame: torch.Tensor,
            from_pos: torch.Tensor = None,
            to_pos: torch.Tensor = None,
    ):
        """
        Forward pass of the GeometryLocationAttentionHead.

        Args:
            from_rank_sv: ScalarVector of the from_rank entities
            to_rank_sv: ScalarVector of the to_rank entities
            incidence_matrix: Sparse incidence matrix connecting from_rank to to_rank
            from_frame: Frame of the from_rank entities
            to_frame: Frame of the to_rank entities
            from_pos: Position of the from_rank entities (optional)
            to_pos: Position of the to_rank entities (optional)

        Returns:
            Attention weights for each edge in the incidence matrix
        """
        incidence_index = incidence_matrix.indices()
        incidence_size = incidence_matrix.size()

        # Get vector features for the entities connected by the incidence matrix
        from_v = from_rank_sv.idx(incidence_index[0]).vector
        from_frame_selected = from_frame[incidence_index[0]]
        to_v = to_rank_sv.idx(incidence_index[1]).vector
        to_frame_selected = to_frame[incidence_index[1]]

        # Project vector features
        from_v_trans = self.from_proj(from_v.transpose(-1, -2))
        to_v_trans = self.to_proj(to_v.transpose(-1, -2))

        # Compute position difference if positions are provided
        if from_pos is not None and to_pos is not None:
            from_pos_selected = from_pos[incidence_index[0]]
            to_pos_selected = to_pos[incidence_index[1]]
            pos_diff = (to_pos_selected - from_pos_selected).unsqueeze(-1)

            # Concatenate projected vectors with position difference
            from_v = torch.concat([from_v_trans, pos_diff], dim=-1).transpose(-1, -2)
            to_v = torch.concat([to_v_trans, -pos_diff], dim=-1).transpose(-1, -2)
        else:
            # If positions are not provided, just use the projected vectors
            from_v = from_v_trans.transpose(-1, -2)
            to_v = to_v_trans.transpose(-1, -2)

        # Transform vectors using frames to make them invariant
        from_v_scalarized = torch.bmm(from_v, from_frame_selected).reshape(from_v.shape[0], -1)
        to_v_scalarized = torch.bmm(to_v, to_frame_selected).reshape(to_v.shape[0], -1)

        # Concatenate transformed vectors
        v_scalarized = torch.concat([from_v_scalarized, to_v_scalarized], dim=1)

        # Compute attention weights
        raw = self.attn_proj(self.activation(v_scalarized)).squeeze(-1)
        att = scatter_softmax(raw, incidence_index[0 if self.higher_to_lower else 1], dim=0, dim_size=incidence_size[1])
        att = att.unsqueeze(-1)

        return att


class GeometryLocationAttention(nn.Module):
    """
    Multi-head geometry-aware location attention.

    This module uses multiple GeometryLocationAttentionHead instances to compute
    attention weights, and can either concatenate or average the outputs.
    """

    def __init__(
            self,
            from_vec_dim: int,
            to_vec_dim: int,
            num_heads: int = 4,
            hidden_dim: int = 3,
            higher_to_lower: bool = True,
            activation: str = 'silu',
            concat: bool = True,
    ):
        """
        Initialize the GeometryLocationAttention module.

        Args:
            from_vec_dim: Dimension of the vector features of the from_rank entities
            to_vec_dim: Dimension of the vector features of the to_rank entities
            num_heads: Number of attention heads (default: 4)
            hidden_dim: Dimension of the hidden representation (default: 3)
            activation: Activation function to use (default: 'silu')
            concat: Whether to concatenate or average the outputs of the attention heads (default: True)
        """
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        # Create multiple attention heads
        self.heads = nn.ModuleList([
            GeometryLocationAttentionHead(
                from_vec_dim=from_vec_dim,
                to_vec_dim=to_vec_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                higher_to_lower=higher_to_lower,
            )
            for _ in range(num_heads)
        ])

    def forward(
            self,
            from_rank_sv: ScalarVector,
            to_rank_sv: ScalarVector,
            incidence_matrix,
            from_frame: torch.Tensor = None,
            to_frame: torch.Tensor = None,
            from_pos: torch.Tensor = None,
            to_pos: torch.Tensor = None,
    ):
        """
        Forward pass of the GeometryLocationAttention module. Must from a higher rank to a lower rank.

        Args:
            from_rank_sv: ScalarVector of the from_rank entities
            to_rank_sv: ScalarVector of the to_rank entities
            incidence_matrix: Sparse incidence matrix connecting from_rank to to_rank
            from_frame: Frame of the from_rank entities (optional)
            to_frame: Frame of the to_rank entities (optional)
            from_pos: Position of the from_rank entities (optional)
            to_pos: Position of the to_rank entities (optional)

        Returns:
            Attention weights for each edge in the incidence matrix
        """
        # If frames are not provided, use identity matrices
        if from_frame is None:
            from_frame = torch.eye(3, device=from_rank_sv.scalar.device).unsqueeze(0).expand(from_rank_sv.scalar.shape[0], -1, -1)
        if to_frame is None:
            to_frame = torch.eye(3, device=to_rank_sv.scalar.device).unsqueeze(0).expand(to_rank_sv.scalar.shape[0], -1, -1)

        # Compute attention weights for each head
        head_outputs = [
            head(
                from_rank_sv=from_rank_sv,
                to_rank_sv=to_rank_sv,
                incidence_matrix=incidence_matrix,
                from_frame=from_frame,
                to_frame=to_frame,
                from_pos=from_pos,
                to_pos=to_pos,
            )
            for head in self.heads
        ]

        # Combine outputs from all heads
        if self.concat:
            # Concatenate outputs along a new dimension
            att = torch.stack(head_outputs, dim=-1)
        else:
            # Average outputs
            att = torch.mean(torch.stack(head_outputs, dim=0), dim=0)

        return sv_attention(
            sv = from_rank_sv.idx(incidence_matrix.indices()[0]),
            attention = att,
        )