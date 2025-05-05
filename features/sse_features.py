"""Utilities for computing cell features."""
from typing import List, Union

import torch
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
import toponetx as tnx
from torch_geometric.nn import PositionalEncoding
from torch_scatter import scatter_mean, scatter_std

from proteinworkshop.features.utils import _normalize
from torch.nn.utils.rnn import pad_sequence
from proteinworkshop.models.utils import centralize
from topotein.features.topotein_complex import TopoteinComplex
from topotein.features.utils import eigenval_features
from topotein.models.utils import localize, scatter_eigen_decomp, get_pca_frames

SSE_FEATURES: List[str] = [
    "cell_size",
    "node_features",
    "edge_features",
    "cell_type",
    "sse_vector_norms",
    "sse_one_hot",
    "sse_variance_wrt_localized_frame"
]
"""List of cell features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_sse_features(
        x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Computes scalar cell features from a :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` object.

    :param x: :class:`~torch_geometric.data.Data` or :class:`~torch_geometric.data.Batch` protein object.
    :type x: Union[Data, Batch]
    :param features: List of cell features to compute.
    :type features: Union[List[str], ListConfig]

    """
    feats = []
    for feature in features:
        if feature == "sse_size":
            feats.append(compute_sse_sizes(x.sse_cell_complex).to(x.x.device))
        elif feature == "node_features":
            feats.append(get_means_by_group(x.x, x.sse_cell_index))
        elif feature == "edge_features":  # TODO: but unsure if this is needed because edge features contains too much node features
            raise NotImplementedError
        elif feature == "sse_one_hot":
            feats.append(x.sse.to(x.x.device))
        elif feature == "center_pos_encoding":
            pos_encoding = PositionalEncoding(16).to(x.pos.device)
            sse_center_pos = scatter_mean(x.seq_pos[x.N0_2.indices()[0]], x.N0_2.indices()[1], dim_size=x.N0_2.size(1), dim=0)
            sse_center_pos_enc = pos_encoding(sse_center_pos)
            feats.append(sse_center_pos_enc)
        elif feature == "se_pos_encoding":
            pos_encoding = PositionalEncoding(10).to(x.pos.device)
            start_pos, end_pos = x.sse_cell_index_simple[0], x.sse_cell_index_simple[1]
            start_pos = pos_encoding(start_pos)
            end_pos = pos_encoding(end_pos)
            feats.extend([start_pos, end_pos])
        elif feature == "consecutive_angle":
            if not hasattr(x, 'N2_3'):
                x.N2_3 = x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3)
            if not hasattr(x, 'sse_s2e_vec'):
                x.sse_s2e_vec = x.pos[x.sse_cell_index_simple[1]] - x.pos[x.sse_cell_index_simple[0]]
            con_angle = consecutive_angle(x.sse_s2e_vec, x.N2_3.indices()[1])
            feats.append(con_angle)
        elif feature == "torsional_angle":
            if not hasattr(x, 'N2_3'):
                x.N2_3 = x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3)
            if not hasattr(x, 'sse_s2e_vec'):
                x.sse_s2e_vec = x.pos[x.sse_cell_index_simple[1]] - x.pos[x.sse_cell_index_simple[0]]
            torsion_angle = plane_intersection(x.sse_s2e_vec, x.N2_3.indices()[1])
            feats.append(torsion_angle)
        elif feature == "eigenvalues":
            evals, _ = get_sse_eigen_features(x)
            feats.append(torch.concat([
                torch.stack(eigenval_features(evals), dim=-1),
                evals
            ], dim=1))
        elif feature == "sse_vector_norms":
            vectors = vector_features(x)
            feats.append(torch.norm(torch.stack(vectors, dim=2), dim=1))
        elif feature == "std_wrt_localized_frame":
            feats.append(std_wrt_localized_frame(x))
        else:
            raise ValueError(f"Unknown cell feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1).float()


@jaxtyped(typechecker=typechecker)
def compute_vector_sse_features(
        x: Union[Data, Batch], features: Union[List[str], ListConfig]
) -> torch.Tensor:
    """
    Compute vector-based cell features for a given data or batch object.

    This function processes a data or batch object to generate vector-based
    features for cell-centric data representation. It computes vector
    attributes for cells based on positional information provided in the input
    data or batch object. The resulting features are then stored in the input
    object as an attribute. The operation supports only specific known
    features.

    :param x: A data structure, either of type ``Data`` or ``Batch``, which
              contains cell-related positional information as attributes such
              as ``pos`` and ``cell_index``.
    :param features: A list (or ``ListConfig``) of feature names to be
                     generated and added to the ``Data`` or ``Batch`` object.
    :return: The modified data structure of type ``Data`` or ``Batch`` with
             updated cell vector attributes.
    :raises ValueError: If an unknown vector feature is specified in the
                        ``features`` parameter.
    """
    vector_sse_features = []
    for feature in features:
        if feature == "sse_vectors":  # TODO
            sse_vectors = vector_features(x)
            for sse_vector in sse_vectors:
                vector_sse_features.append(_normalize(sse_vector).unsqueeze(-2))
        elif feature == "eigenvectors":
            # not stable - dont use!
            x.sse_pca_frames = localize(x, rank=2, frame_type='pca')
            vector_sse_features.append(x.sse_pca_frames)

        elif feature == "consecutive_diff":
            if not hasattr(x, 'N2_3'):
                x.N2_3 = x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3)
            vector_sse_features.append(consecutive_differences(
                x.sse_cell_complex.get_com(2),
                x.N2_3.indices()[1]
            ))
        elif feature == "pr_com_diff":
            start_diff, end_diff = x.sse_cell_complex.centered_pos[x.sse_cell_index_simple[0]], x.sse_cell_complex.centered_pos[x.sse_cell_index_simple[1]]
            com_diff = x.sse_cell_complex.get_com(2)
            vector_sse_features.append(torch.stack([start_diff, com_diff, end_diff], dim=1))
        else:
            raise ValueError(f"Vector feature {feature} not recognised.")
    return torch.cat(vector_sse_features, dim=1).float()


def consecutive_differences(
        vectors: torch.Tensor,
        batch_index: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    vectors:     (N,3) tensor of 3D vectors (zero‐vector == padding)
    batch_index: (N,)   tensor of ints indicating cluster membership

    returns:     (N,6) tensor whose columns are
                 [Δ_prev_x, Δ_prev_y, Δ_prev_z, Δ_next_x, Δ_next_y, Δ_next_z],
                 where
                   Δ_prev = v[i]   - v[i-1]
                   Δ_next = v[i+1] - v[i]
                 Any entry touching a padding vector or crossing a batch
                 boundary (or at the very first/last index) is zeroed.
    """
    N = vectors.shape[0]
    assert batch_index.shape[0] == N, "batch_index must match vectors length"
    device, dtype = vectors.device, vectors.dtype

    # compute raw diffs between neighbors: v_next - v_prev, length N-1
    v_prev = vectors[:-1]    # (N-1, 3)
    v_next = vectors[1:]     # (N-1, 3)
    diffs  = v_next - v_prev # (N-1, 3)

    # identify invalid pairs
    is_zero_prev = v_prev.norm(dim=1) < eps
    is_zero_next = v_next.norm(dim=1) < eps
    same_batch   = batch_index[1:] == batch_index[:-1]
    invalid      = is_zero_prev | is_zero_next | (~same_batch)

    # zero out invalid diffs
    diffs = diffs.clone()
    diffs[invalid] = 0.0

    # build Δ_prev: zero at i=0, then diffs for i=1..N-1
    zero3 = torch.zeros(1, 3, device=device, dtype=dtype)
    delta_prev = torch.cat([zero3, diffs], dim=0)    # (N,3)

    # build Δ_next: diffs for i=0..N-2, then zero at i=N-1
    delta_next = torch.cat([diffs, zero3], dim=0)    # (N,3)

    # concatenate to (N,6)
    return torch.cat([delta_prev.unsqueeze(1), delta_next.unsqueeze(1)], dim=1)

def plane_intersection(
        vectors: torch.Tensor,
        batch_index: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    vectors:     (N,3) tensor of 3D vectors (zero‐vector == padding)
    batch_index: (N,)   tensor of ints indicating cluster membership

    returns:     (N,2) tensor [sin(phi), cos(phi)] of the dihedral angle
                 between plane({v[i-1],v[i]}) and plane({v[i],v[i+1]}),
                 with zeros at edges, batch‐changes, or padding.
    """
    N = vectors.shape[0]
    assert batch_index.shape[0] == N

    # build triplets v_prev, v_mid, v_next for i=1..N-2
    v_prev = vectors[:-2]    # (N-2,3)
    v_mid  = vectors[1:-1]   # (N-2,3)
    v_next = vectors[2:]     # (N-2,3)

    # normals to each plane
    n1 = torch.cross(v_prev, v_mid,  dim=1)  # (N-2,3)
    n2 = torch.cross(v_mid,  v_next, dim=1)  # (N-2,3)

    # dot and cross‐norm of normals
    dot       = (n1 * n2).sum(dim=1)               # (N-2,)
    cross_norm = torch.cross(n1, n2, dim=1).norm(dim=1)  # (N-2,)

    # norms of normals
    n1_norm = n1.norm(dim=1)                       # (N-2,)
    n2_norm = n2.norm(dim=1)                       # (N-2,)
    denom   = n1_norm * n2_norm                    # (N-2,)

    # raw sin & cos
    cos_raw = dot       / (denom + eps)
    sin_raw = cross_norm / (denom + eps)

    # clamp cos
    cos_clamped = cos_raw.clamp(-1.0, 1.0)

    # mask invalid: padding or batch‐boundary anywhere in the triplet
    is_pad   = denom < eps
    same_batch = (
            (batch_index[:-2] == batch_index[1:-1]) &
            (batch_index[1:-1] == batch_index[2:])
    )
    invalid = is_pad | (~same_batch)

    sin_raw     = sin_raw.clone()
    cos_clamped = cos_clamped.clone()
    sin_raw[invalid]      = 0.0
    cos_clamped[invalid]  = 0.0

    # pad zeros at front and back to make length N
    zero = torch.zeros(1, device=vectors.device, dtype=vectors.dtype)
    sin = torch.cat([zero, sin_raw, zero], dim=0)     # (N,)
    cos = torch.cat([zero, cos_clamped, zero], dim=0) # (N,)

    return torch.stack([sin, cos], dim=1)             # (N,2)

def consecutive_angle(
        vectors: torch.Tensor,
        batch_index: torch.Tensor,
        eps: float = 1e-8
) -> torch.Tensor:
    """
    vectors:     (N,3) tensor of 3D vectors (zero‐vector == padding)
    batch_index: (N,)   tensor of ints indicating cluster membership

    returns:     (N,4) tensor whose columns are
                 [sin_prev, cos_prev, sin_next, cos_next]
    """
    N = vectors.shape[0]
    assert batch_index.shape[0] == N, "batch_index must be length N"

    # pairs (i, i+1)
    v1 = vectors[:-1]          # (N-1, 3)
    v2 = vectors[1:]           # (N-1, 3)

    # compute dot and cross mag
    dot    = (v1 * v2).sum(dim=1)             # (N-1,)
    cross  = torch.cross(v1, v2, dim=1).norm(dim=1)  # (N-1,)

    # norms and denom
    n1     = v1.norm(dim=1)                   # (N-1,)
    n2     = v2.norm(dim=1)                   # (N-1,)
    denom  = n1 * n2                          # (N-1,)

    # raw sin/cos (avoid div-0)
    cos_raw   = dot   / (denom + eps)
    sin_raw   = cross / (denom + eps)

    # clamp cos to valid range
    cos_clamped = cos_raw.clamp(-1.0, 1.0)

    # mask out invalid pairs
    is_padding = denom < eps
    same_batch = batch_index[:-1] == batch_index[1:]
    invalid    = is_padding | (~same_batch)

    sin_raw   = sin_raw.clone()
    cos_clamped = cos_clamped.clone()
    sin_raw[invalid]      = 0.0
    cos_clamped[invalid]  = 0.0

    # build prev angles: pad at front
    zero = torch.zeros(1, device=vectors.device, dtype=vectors.dtype)
    sin_prev = torch.cat([zero, sin_raw],      dim=0)  # (N,)
    cos_prev = torch.cat([zero, cos_clamped],  dim=0)

    # build next angles: pad at end
    sin_next = torch.cat([sin_raw,      zero], dim=0)  # (N,)
    cos_next = torch.cat([cos_clamped,  zero], dim=0)

    # stack into (N,4)
    return torch.stack([sin_prev, cos_prev, sin_next, cos_next], dim=1)

@jaxtyped(typechecker=typechecker)
def compute_sse_sizes(cell_complex: Union[tnx.CellComplex, TopoteinComplex]) -> torch.Tensor:
    """
    Compute the sizes of all cells within the provided cell complex.

    This function iterates over all the cells in the given cell complex
    and computes their sizes. The returned tensor contains the sizes of
    each cell, in order of their iteration.

    :param cell_complex: The cell complex whose cell sizes need to be computed.
    :type cell_complex: tnx.CellComplex

    :return: A tensor containing the sizes of all cells in the given
        cell complex.
    :rtype: torch.Tensor
    """
    if type(cell_complex) == tnx.CellComplex:

        return torch.tensor(list(map(cell_complex.size, iter(cell_complex.cells))))
    elif type(cell_complex) == TopoteinComplex:
        return cell_complex.laplacian_matrix(rank=2, via_rank=0).values()
    else:
        raise ValueError(f"Invalid cell complex type: {type(cell_complex)}")


@jaxtyped(typechecker=typechecker)
def get_means_by_group(features: torch.Tensor, groups: tuple[tuple[int, ...], ...]) -> torch.Tensor:
    """
    Calculate the mean of feature vectors for each specified group in an efficient manner.

    The method processes a tensor of feature vectors and a list of group indices.
    Each group's feature vectors are extracted and padded to handle variable group sizes.
    The mean is then calculated across feature vectors in each group.

    :param features: A tensor containing feature vectors of shape (N, M), where N is
        the number of samples and M is the number of features.
    :type features: torch.Tensor
    :param groups: A list of tuples containing indices specifying the groups. Each tuple
        corresponds to a group, and the indices within the tuple represent the positions
        of rows in the `features` tensor that belong to that group.
    :type groups: list[tuple[int]]
    :return: A tensor of means for each group of shape (J, M), where J is the number of
        groups and M matches the feature size from the input.
    :rtype: torch.Tensor
    """
    # Step 1: Create a list of groups of rows from x.x using indexing
    groups = [features[indices, :] for indices in groups]

    # Handle variable-size groups with padding
    groups_padded = pad_sequence(groups, batch_first=True)  # Shape (J, max_rows, 60)

    # Step 2: Calculate the mean for each group efficiently
    means = groups_padded.mean(dim=1)  # Shape (J, 60)

    return means


def get_sse_eigen_features(batch):
    # Calculate mean positions for each protein using scatter_mean
    if not hasattr(batch, 'N0_2'):
        batch.N0_2 = batch.sse_cell_complex.incidence_matrix(from_rank=0, to_rank=2)
    eigenvec, _, eigenval = scatter_eigen_decomp(
        batch.pos[batch.N0_2.indices()[0]],
        batch.N0_2.indices()[1],
        batch.N0_2.size(1)
    )

    return eigenval, eigenvec



@jaxtyped(typechecker=typechecker)
def std_wrt_localized_frame(batch: ProteinBatch) -> torch.Tensor:
    sse_com = batch.sse_cell_complex.get_com(rank=2, relative=False)
    projected_pos = torch.bmm((batch.pos[batch.N0_2.indices()[0]] - sse_com[batch.N0_2.indices()[1]]).unsqueeze(1), localize(batch, rank=2)[batch.N0_2.indices()[1]]).squeeze(1)
    return scatter_std(projected_pos, batch.N0_2.indices()[1].repeat(3, 1).T, dim=0)


@jaxtyped(typechecker=typechecker)
def vector_features(batch: ProteinBatch) -> list[torch.Tensor]:
    """
    Extracts a list of geometric vector features from 3D spatial coordinates and structured
    secondary element (SSE) index data.

    This function computes various vectors derived from the coordinates of the start, end, and
    median positions of SSEs, as well as their relationships to the center of mass (COM) and
    other intermediate positions. The features include vectors such as start-to-end, median-
    to-start, etc. These vectors can be used as input for downstream processing or modeling.

    :param X: A tensor containing the 3D spatial coordinates of residues.
    :param sse_idx: A tensor containing the indices of the start and end residues for
        SSEs, with shape (2, n) where n is the number of SSEs.
    :return: A list of geometric vector features, where each feature is a tensor.
    """

    com_pos = batch.sse_cell_complex.get_com(2)
    X = batch.sse_cell_complex.centered_pos

    start_residue_idx = batch.sse_cell_index_simple[0, :]
    end_residue_idx = batch.sse_cell_index_simple[1, :]

    start_residue_pos = X[start_residue_idx, :]
    end_residue_pos = X[end_residue_idx, :]
    # middle_pos = (start_residue_pos + end_residue_pos) / 2

    median_residue_idx = (start_residue_idx + end_residue_idx) / 2
    s_median_residue_idx = torch.ceil(median_residue_idx).long()
    e_median_residue_idx = torch.floor(median_residue_idx).long()
    median_residue_pos = ((X[e_median_residue_idx, :] + X[s_median_residue_idx, :]) / 2)

    s_to_e_vec = end_residue_pos - start_residue_pos
    med_to_s_vec = start_residue_pos - median_residue_pos
    med_to_e_vec = end_residue_pos - median_residue_pos
    com_to_s_vec = start_residue_pos - com_pos
    com_to_e_vec = end_residue_pos - com_pos
    med_to_com_vec = com_pos - median_residue_pos
    # med_to_middle_vec = middle_pos - median_residue_pos
    # middle_to_com_vec = com_pos - middle_pos

    features = [
        s_to_e_vec,
        med_to_s_vec,
        med_to_e_vec,
        com_to_s_vec,
        com_to_e_vec,
        med_to_com_vec,
        # med_to_middle_vec,
        # middle_to_com_vec
    ]

    return features
