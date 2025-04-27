from typing import Optional, Tuple

import torch
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Int64, Float, Bool

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCPMessagePassing
from topotein.models.utils import lift_features_with_padding, map_to_cell_index


class TCPMessagePassing(GCPMessagePassing):

    @jaxtyped(typechecker=typechecker)
    def message(
            self,
            node_rep: ScalarVector,
            edge_rep: ScalarVector,
            sse_rep: ScalarVector,
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            edge_frames: Float[torch.Tensor, "batch_num_edges 3 3"],
            node_to_sse_mapping: torch.Tensor = None,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Float[torch.Tensor, "batch_num_edges message_dim"]:
        row, col = edge_index
        node_vector = node_rep.vector.reshape(
            node_rep.vector.shape[0],
            node_rep.vector.shape[1] * node_rep.vector.shape[2],
            )
        vector_reshaped = ScalarVector(node_rep.scalar, node_vector)

        node_s_row, node_v_row = vector_reshaped.idx(row)
        node_s_col, node_v_col = vector_reshaped.idx(col)

        node_v_row = node_v_row.reshape(node_v_row.shape[0], node_v_row.shape[1] // 3, 3)
        node_v_col = node_v_col.reshape(node_v_col.shape[0], node_v_col.shape[1] // 3, 3)

        # cell features for edge
        sse_vector = sse_rep.vector.reshape(
            sse_rep.vector.shape[0],
            sse_rep.vector.shape[1] * sse_rep.vector.shape[2],
            )
        cell_scalar = lift_features_with_padding(sse_rep.scalar, neighborhood=node_to_sse_mapping)
        sse_vector = lift_features_with_padding(sse_vector, neighborhood=node_to_sse_mapping)

        vector_reshaped = ScalarVector(cell_scalar, sse_vector)

        sse_s_row, sse_v_row = vector_reshaped.idx(row)
        sse_s_col, sse_v_col = vector_reshaped.idx(col)

        sse_v_row = sse_v_row.reshape(sse_v_row.shape[0], sse_v_row.shape[1] // 3, 3)
        sse_v_col = sse_v_col.reshape(sse_v_col.shape[0], sse_v_col.shape[1] // 3, 3)


        message = edge_rep.concat((
            ScalarVector(node_s_row, node_v_row),
            ScalarVector(node_s_col, node_v_col),
            ScalarVector(sse_s_row, sse_v_row),
            ScalarVector(sse_s_col, sse_v_col),
        ))

        message_residual = self.message_fusion[0](
            message, edge_index, edge_frames, node_inputs=False, node_mask=node_mask
        )
        # edge message passing
        for module in self.message_fusion[1:]:
            # exchange geometric messages while maintaining residual connection to original message
            new_message = module(
                message_residual,
                edge_index,
                edge_frames,
                node_inputs=False,
                node_mask=node_mask,
            )
            message_residual = message_residual + new_message

        # learn to gate scalar messages
        if self.use_scalar_message_attention:
            message_residual_attn = self.scalar_message_attention(
                message_residual.scalar
            )
            message_residual = ScalarVector(
                message_residual.scalar * message_residual_attn,
                message_residual.vector,
                )

        return message_residual.flatten()


    @jaxtyped(typechecker=typechecker)
    def forward(
            self,
            node_rep: ScalarVector,
            edge_rep: ScalarVector,
            sse_rep: ScalarVector,
            edge_index: Int64[torch.Tensor, "2 batch_num_edges"],
            edge_frames: Float[torch.Tensor, "batch_num_edges 3 3"],
            node_to_sse_mapping: torch.Tensor = None,
            node_mask: Optional[Bool[torch.Tensor, "batch_num_nodes"]] = None,
    ) -> Tuple[ScalarVector, ScalarVector]:
        message = self.message(
            node_rep, edge_rep, sse_rep, edge_index, edge_frames, node_to_sse_mapping, node_mask
        )
        node_aggregate = self.aggregate(
            message, edge_index, dim_size=node_rep.scalar.shape[0]
        )

        cell_edge_index = map_to_cell_index(edge_index, node_to_sse_mapping)
        mask = ((~(cell_edge_index == -1).any(dim=0)) & (cell_edge_index[0] != cell_edge_index[1]))
        sse_aggregate = self.aggregate(
            message[mask], cell_edge_index[:, mask], dim_size=sse_rep.scalar.shape[0]
        )

        node_aggregate = ScalarVector.recover(node_aggregate, self.vector_output_dim)
        sse_aggregate = ScalarVector.recover(sse_aggregate, self.vector_output_dim)


        return node_aggregate, sse_aggregate
