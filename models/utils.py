from functools import partial

from beartype.typing import Optional, Tuple, Union

import torch
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import Bool, jaxtyped
from torch_geometric.data import Batch
from torch_scatter import scatter_mean, scatter, scatter_max

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.utils import safe_norm

DEFAULT_RANK_MAPPING = {
    0: ScalarVector('x', 'x_vector_attr'),
    1: ScalarVector('edge_attr', 'edge_vector_attr'),
    2: ScalarVector('sse_attr', 'sse_vector_attr'),
    3: ScalarVector('pr_attr', 'pr_vector_attr')
}

@jaxtyped(typechecker=typechecker)
def centralize(
        pos: torch.Tensor,
        batch_index: torch.Tensor,
        node_mask: Optional[Bool[torch.Tensor, " n_nodes"]] = None,
) -> Tuple[
    torch.Tensor, torch.Tensor
]:  # note: cannot make assumptions on output shape
    if node_mask is not None:
        # derive centroid of each batch element
        entities_centroid = torch_scatter.scatter(
            pos[node_mask], batch_index[node_mask], dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]

        # center entities using corresponding centroids
        masked_values = torch.ones_like(pos) * torch.inf
        values = pos[node_mask]
        masked_values[node_mask] = (
                values - entities_centroid[batch_index][node_mask]
        )
        entities_centered = masked_values

    else:
        # derive centroid of each batch element, and center entities using corresponding centroids
        entities_centroid = torch_scatter.scatter(
            pos, batch_index, dim=0, reduce="mean"
        )  # e.g., [batch_size, 3]
        entities_centered = pos - entities_centroid[batch_index]

    return entities_centroid, entities_centered


def lift_features_with_padding(features: torch.Tensor, neighborhood: torch.Tensor) -> torch.Tensor:
    """
    Lifts given features with padding based on the provided neighborhood tensor.

    Given an input feature tensor and a neighborhood mapping, the function creates a new
    feature tensor where specific elements are lifted (copied) into a larger tensor
    with padding, according to the neighborhood indices.

    :param features: Tensor containing the feature data, typically in multi-dimensional
        format. It serves as the source from which features will be lifted
        based on the neighborhood mapping.
    :type features: torch.Tensor
    :param neighborhood: Tensor that defines the mapping for lifting the features. It
        contains indices that specify how features from the input tensor will
        be arranged in the output tensor, and its dimensions correspond to
        the mapping logic.
    :type neighborhood: torch.Tensor
    :return: Tensor with lifted features and padding applied, maintaining the necessary
        alignment as defined by the neighborhood tensor.
    :rtype: torch.Tensor
    """
    lifted_size = neighborhood.size()[0]
    lifted_features_values = features[neighborhood.indices()[1, neighborhood.values() == 1]]
    lifted_features = torch.zeros(lifted_size, *features.shape[1:],
                                  device=features.device,
                                  dtype=features.dtype)
    lifted_features[neighborhood.indices()[0]] = lifted_features_values
    return lifted_features


def map_to_cell_index(edge_index: torch.Tensor, node_to_sse_mapping: torch.Tensor) -> torch.Tensor:
    """
    Maps node indices in the edge index of a graph to their corresponding cell indices
    based on a mapping of nodes to SSE (Secondary Structure Elements).

    This function uses the node-to-SSE mapping to construct a lookup table,
    which is then used for mapping the edge index from the node level to
    the cell (SSE) level.

    :param edge_index: A tensor of shape (2, num_edges) where each column
        represents an edge defined by the indices of its two-connected nodes.
    :type edge_index: torch.Tensor
    :param node_to_sse_mapping: A sparse tensor mapping node indices to their
        associated SSE indices. Should be of shape (num_nodes, num_sse).
    :type node_to_sse_mapping: torch.Tensor
    :return: A tensor of shape (2, num_edges) where each column represents an
        edge transformed to the SSE level. The indices now reference SSE indices
        instead of node indices.
    :rtype: torch.Tensor
    """
    sse_mapping = node_to_sse_mapping
    sse_lookup = torch.ones(sse_mapping.size(0), dtype=torch.long, device=sse_mapping.device) * -1
    sse_lookup[sse_mapping.indices()[0]] = sse_mapping.indices()[1]
    cell_edge_index = torch.stack([sse_lookup[edge_index[i]] for i in range(2)], dim=0)
    return cell_edge_index

def get_com(positions, cluster_ids=None, cluster_num=None):
    if cluster_ids is None:
        return torch.mean(positions, dim=0, keepdim=True)
    else:
        return scatter_mean(positions, cluster_ids, dim=0, dim_size=cluster_num)

def pca_reduce(tensor: torch.Tensor, k: int):
    # 1) center
    mean = tensor.mean(dim=0, keepdim=True)   # [1, d]
    X    = tensor - mean                      # [n, d]

    # 2) covariance
    n, d = X.shape
    C    = (X.T @ X) / (n - 1)                # [d, d]

    # 3) eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(C)  # ascending eigvals
    idx = torch.argsort(eigvals, descending=True)
    U   = eigvecs[:, idx][:, :k]             # [d, k]

    # 4) anchor = vector to farthest point
    dist2 = (X**2).sum(dim=1)
    i_max = torch.argmax(dist2)
    ref   = X[i_max]                          # [d]

    # 5) sign‐disambiguation via anchor
    for i in range(k):
        if torch.dot(U[:, i], ref) < 0:
            U[:, i] = -U[:, i]

    # 6) right‐handed check (only for d==3, k==3)
    if k == 3 and d == 3:
        cross = torch.cross(U[:, 0], U[:, 1], dim=0)
        if torch.dot(cross, U[:, 2]) < 0:
            U[:, 2] = -U[:, 2]

    # 7) project
    reduced = X @ U                           # [n, k]
    return reduced, mean.squeeze(0), U        # mean: [d], U: [d, k]

def get_pca_frames(positions: torch.Tensor,
                   cluster_ids: torch.Tensor = None,
                   cluster_num: int = None) -> torch.Tensor:
    """
    Calculate SE(3)-equivariant frames based on PCA for each cluster of positions.

    Returns:
        frames: tensor of shape [cluster_num, 3, 3]
    """
    device = positions.device
    frames = torch.zeros((cluster_num, 3, 3), device=device)

    # Helper to build frame from exactly two points
    def fallback_frame(p1, p2):
        # primary axis: normalized (p2 - p1)
        a = (p2 - p1)
        a = a / a.norm()
        # choose arbitrary orthogonal vector
        # here, cross with world-x unless collinear
        tmp = torch.tensor([1.0, 0.0, 0.0], device=device)
        if torch.allclose(a.abs(), tmp):
            tmp = torch.tensor([0.0, 1.0, 0.0], device=device)
        b = torch.cross(a, tmp)
        b = b / b.norm()
        c = torch.cross(a, b)
        return torch.stack([a, b, c], dim=1)

    if cluster_ids is None:
        # Single cluster
        n, _ = positions.shape
        if n < 2:
            # One or zero points: default axes
            frames[0] = torch.eye(3, device=device)
        elif n == 2:
            # Two points: fallback
            frames[0] = fallback_frame(positions[0], positions[1])
        else:
            # >=3 points: use PCA
            _, mean, U = pca_reduce(positions, 3)
            frames[0] = U
    else:
        # Multiple clusters
        for cid in torch.unique(cluster_ids):
            mask = cluster_ids == cid
            pts = positions[mask]
            m, _ = pts.shape
            if m < 2:
                frames[cid] = torch.eye(3, device=device)
            elif m == 2:
                frames[cid] = fallback_frame(pts[0], pts[1])
            else:
                _, mean, U = pca_reduce(pts, 3)
                frames[cid] = U

    return frames


def localize(batch, rank, node_mask=None, norm_pos_diff=True):
    num_of_frames = batch.sse_cell_complex._get_size_of_rank(rank)
    frames = (
            torch.ones((num_of_frames, 3, 3), device=batch.pos.device)
            * torch.inf
    )
    if rank == 0:
        if node_mask is not None:
            dst_node_mask = node_mask[batch.edge_index[1]]
            neighbor_com = get_com(
                positions=batch.pos[batch.edge_index[1]][dst_node_mask],
                cluster_ids=batch.edge_index[0][dst_node_mask],
                cluster_num=num_of_frames
            )
            frames[node_mask] = get_frames(X_src=batch.pos[node_mask], X_dst=neighbor_com[node_mask], normalize=norm_pos_diff)
        else:
            neighbor_com = get_com(
                positions=batch.pos[batch.edge_index[1]],
                cluster_ids=batch.edge_index[0],
                cluster_num=num_of_frames
            )
            frames = get_frames(X_src=batch.pos, X_dst=neighbor_com, normalize=norm_pos_diff)
    elif rank == 1:
        if node_mask is not None:
            edge_mask = node_mask[batch.edge_index[0]] & node_mask[batch.edge_index[1]]
            X_src = batch.pos[batch.edge_index[0]][edge_mask]
            X_dst = batch.pos[batch.edge_index[1]][edge_mask]
            frames[edge_mask] = get_frames(X_src, X_dst, normalize=norm_pos_diff)
        else:
            X_src = batch.pos[batch.edge_index[0]]
            X_dst = batch.pos[batch.edge_index[1]]
            frames = get_frames(X_src, X_dst, normalize=norm_pos_diff)
    elif rank == 2:
        if not hasattr(batch, 'N0_2'):
            batch.N0_2 = batch.sse_cell_complex.incidence_matrix(from_rank=0, to_rank=2)
        if not hasattr(batch, 'N2_3'):
            batch.N2_3 = batch.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3)
        batch.pos_in_sse = batch.pos[batch.N0_2.indices()[0]]

        if node_mask is not None:
            in_sse_node_mask = node_mask[batch.N0_2.indices()[0]]
            # Use PCA instead of COM for rank 2
            sse_frames = get_pca_frames(
                positions=batch.pos_in_sse[in_sse_node_mask],
                cluster_ids=batch.N0_2.indices()[1][in_sse_node_mask],
                cluster_num=num_of_frames
            )
            sse_mask = torch.any(sse_frames != 0, dim=(1, 2))
            frames[sse_mask] = sse_frames[sse_mask]
        else:
            # Use PCA instead of COM for rank 2
            frames = get_pca_frames(
                positions=batch.pos_in_sse,
                cluster_ids=batch.N0_2.indices()[1],
                cluster_num=num_of_frames
            )
    elif rank == 3:
        if node_mask is not None:
            # Use PCA instead of COM for rank 3
            frames = get_pca_frames(
                positions=batch.pos[node_mask],
                cluster_ids=batch.batch[node_mask],
                cluster_num=len(batch.id)
            )
        else:
            # Use PCA instead of COM for rank 3
            frames = get_pca_frames(
                positions=batch.pos,
                cluster_ids=batch.batch,
                cluster_num=len(batch.id)
            )
    else:
        raise ValueError(f"Invalid rank: {rank}, available ranks are 0, 1, 2, 3")

    return frames

def get_frames(X_src, X_dst, normalize=True):
    # note that when X_src and X_dst is too close to each other, the frame is not accurate, same applies when X is [0, 0, 0]
    norm = lambda x: x / (safe_norm(x, dim=1, keepdim=True) + 1) if normalize else x

    a_vec = norm(X_src - X_dst)
    b_vec = norm(torch.cross(X_src, X_dst))
    c_vec = torch.cross(a_vec, b_vec)

    return torch.stack([a_vec, b_vec, c_vec], dim=1)

def scalarize(vector_reps: torch.Tensor, frames: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Computes a scalarized representation by applying the Einstein summation
    convention on the input tensors. Optionally, flattens the resulting
    tensor along specified dimensions or transposes specific axes based on
    the input flag.

    :param vector_reps: A tensor containing the vector representations. The
       dimensions and shape of this tensor should match that required for
       the Einstein summation with `frames`.
    :type vector_reps: torch.Tensor
    :param frames: A tensor containing frame data for the Einstein
       summation. The dimensions and shape of this tensor should match that
       required for the operation with `vector_reps`.
    :type frames: torch.Tensor
    :param flatten: A boolean flag determining whether the resulting tensor
       should be flattened starting from the second-to-last dimension (-2).
       If set to False, the tensor is instead transposed between the last
       and second-to-last dimensions.
    :type flatten: bool
    :return: The scalarized tensor as a result of the Einstein summation,
       either flattened or transposed based on the `flatten` flag.
    :rtype: torch.Tensor
    """
    result = torch.einsum('...mn,...mk->...nk', vector_reps, frames)
    if flatten:
        result = result.flatten(start_dim=-2)
    else:
        result = result.transpose(-1, -2)
    return result


def tensorize(scalarized_reps: torch.Tensor, frames: torch.Tensor, flattened: bool = False) -> torch.Tensor:
    """
    Reverse the scalarization process to recover the vector representations.

    Parameters:
      scalarized_reps (Tensor): The scalarized tensor. Depending on `flatten`, its shape is either
                                (..., m, n) if flatten is False, or flattened along the second last dimension.
      frames (Tensor): The local frames of shape (..., m, m) (assumed square and orthonormal).
      flatten (bool):  Whether the scalarized representation is flattened along its last two dims.

    Returns:
      Tensor: The reconstructed vector representation of shape (..., m, n).
    """
    # If the scalarized representation was flattened, we need to unflatten it.
    if flattened:
        scalarized_reps = scalarized_reps.view(scalarized_reps.shape[0], -1, 3).transpose(-1, -2)

    # In the scalarize function with flatten=False, the result was transposed so that
    # its shape is (..., m, n). We now reverse the operation by combining the frame vectors:
    vector_reps = torch.einsum('...mk,...kn->...mn', frames, scalarized_reps)
    return vector_reps

def to_sse_batch(batch: ProteinBatch):
    batch.node_frames = localize(batch, 0)
    batch.edge_frames = localize(batch, 1)

    batch.B0_2 = batch.sse_cell_complex.incidence_matrix(from_rank=0, to_rank=2)
    sse_batch = Batch(batch=batch.B0_2.indices()[1], sse_type=batch.sse, num_sse=batch.sse.shape[0])

    for key in ['pos', 'x', 'x_vector_attr', 'node_frames']:
        sse_batch[key] = batch[key][batch.B0_2.indices()[0]]

    batch.B1_2 = batch.sse_cell_complex.incidence_matrix(from_rank=1, to_rank=2)
    edges = batch.B1_2.indices()[0][batch.B1_2.indices()[1]]
    sse_batch['edge_index'] = batch['edge_index'][:, edges]
    for key in ['edge_attr', 'edge_vector_attr', 'edge_frames']:
        sse_batch[key] = batch[key][edges]
    return sse_batch


def sv_scatter(sv: ScalarVector, indices, dim_size, reduce="sum", indexed_input=False) -> ScalarVector:
    vec_dim = sv.vector.shape[1]
    sv_flattened = sv.flatten()[indices[0]] if not indexed_input else sv.flatten()
    sv_flattened = scatter(
        sv_flattened,
        indices[1],
        dim=0,
        dim_size=dim_size,
        reduce=reduce
    )
    sv = ScalarVector.recover(sv_flattened, vec_dim)
    return sv


def sv_aggregate(sv: ScalarVector, neighborhood_matrix, reduce="sum", indexed_input=False) -> ScalarVector:
    return sv_scatter(
        sv,
        neighborhood_matrix.indices(),
        dim_size=neighborhood_matrix.size(-1),
        reduce=reduce,
        indexed_input=indexed_input
    )

def sv_attention(sv: ScalarVector, attention: torch.Tensor):
    v_dim = sv.vector.shape[1]
    sv = sv.flatten()
    if attention.dim() == 2:
        sv = sv * attention
        return ScalarVector.recover(sv, v_dim)
    elif attention.dim() == 3:
        sv = sv.unsqueeze(-1).repeat(1, 1, attention.shape[-1])
        sv = torch.einsum("bfa,bia->bfa", sv, attention)
        sv = [ScalarVector.recover(sv_i.squeeze(-1), v_dim) for sv_i in sv.split(1, dim=-1)]
        sv = sv[0].concat(sv[1:])
        return ScalarVector(*sv)
    else:
        raise ValueError("Attention tensor must have 2 or 3 dimensions.")

def sv_apply_proj(sv: ScalarVector, proj_s: torch.nn.Module, proj_v: torch.nn.Module):
    s, v = sv
    s = proj_s(s)
    v = proj_v(v.transpose(-1, -2)).transpose(-1, -2)
    return ScalarVector(s, v)
