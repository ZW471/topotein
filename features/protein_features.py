"""Utilities for computing cell features."""
from typing import List, Union
import torch
import torch_scatter
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean, scatter_std

from topotein.features.utils import eigenval_features
from topotein.models.utils import localize, get_com, scatter_eigen_decomp

PROTEIN_FEATURES: List[str] = [
]
"""List of cell features that can be computed."""


@jaxtyped(typechecker=typechecker)
def compute_scalar_protein_features(
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
        pr_dim_size = len(x.id)
        if feature == "pr_size":
            pr_size = torch_scatter.scatter_sum(
                torch.ones_like(x.batch, device=x.x.device),
                x.batch,
                dim_size=pr_dim_size
            ).to_dense()
            feats.append(pr_size)
        elif feature == "aa_freq":
            aa_freq = torch_scatter.scatter_mean(
                x.amino_acid_one_hot().float(),
                x.batch,
                dim=0,
                dim_size=pr_dim_size
            ).to_dense()
            feats.append(aa_freq)
        elif feature == "aa_std":
            aa_freq = torch_scatter.scatter_std(
                x.amino_acid_one_hot().float(),
                x.batch,
                dim=0,
                dim_size=pr_dim_size
            ).to_dense()
            feats.append(aa_freq)
        elif feature == "sse_freq":
            sse_freq = torch_scatter.scatter_mean(
                x.sse.float(),
                x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3).indices()[1],
                dim=0,
                dim_size=pr_dim_size
            ).to_dense()
            feats.append(sse_freq)
        elif feature == "sse_std":
            sse_freq = torch_scatter.scatter_std(
                x.sse.float(),
                x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3).indices()[1],
                dim=0,
                dim_size=pr_dim_size
            ).to_dense()
            feats.append(sse_freq)
        elif feature == "sse_size_mean":
            sse_size = x.sse_cell_index_simple[1] - x.sse_cell_index_simple[0] + 1
            sse_size_mean = torch_scatter.scatter_mean(
                sse_size.float(),
                x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3).indices()[1],
                dim_size=pr_dim_size
            )
            feats.append(sse_size_mean)
        elif feature == "sse_size_std":
            sse_size = x.sse_cell_index_simple[1] - x.sse_cell_index_simple[0] + 1
            sse_size_std = torch_scatter.scatter_std(
                sse_size.float(),
                x.sse_cell_complex.incidence_matrix(from_rank=2, to_rank=3).indices()[1],
                dim_size=pr_dim_size
            )
            feats.append(sse_size_std)
        elif feature == "gyration_r":
            com = x.sse_cell_complex.get_com(rank=3)
            d2 = ((x.pos - com[x.batch])**2).sum(dim=1)
            d_mean = torch_scatter.scatter_mean(
                d2,
                x.batch,
                dim_size=pr_dim_size
            )
            rg = torch.sqrt(d_mean)
            feats.append(rg)
        elif feature == "eigenvalues":
            # projected_pos = getattr(x, "pos_proj", project_node_positions(x))
            # x["pos_proj"] = projected_pos
            evals, _ = get_protein_eigen_features(x)
            feats.append(torch.concat([
                torch.stack(eigenval_features(evals), dim=-1),
                evals
            ], dim=1))
        elif feature == "std_wrt_localized_frame":
            projected_pos = getattr(x, "pos_proj", project_node_positions(x))
            x["pos_proj"] = projected_pos
            feats.append(scatter_std(projected_pos, x.batch.repeat(3, 1).T, dim=0))
        elif feature == "contact_density_and_order":
            batch = x.batch
            num_graphs = int(batch.max().item()) + 1
            per_graph_feats = []

            for graph_id in range(num_graphs):
                mask = (batch == graph_id)
                pos = x.pos[mask]                # [Ni,3] for this graph
                N = pos.size(0)

                # distance / contact‐map for *this* graph
                D = torch.cdist(pos, pos)        # [Ni,Ni]
                contacts = (D < 10.0).float()    # binary map

                # contact density = total contacts ÷ Ni
                contact_density = contacts.sum() / N

                # sequence‐distance matrix
                i = torch.arange(N, device=pos.device)
                seq_dist = (i.unsqueeze(1) - i.unsqueeze(0)).abs().float()  # [Ni,Ni]

                # contact order = (Σ contacts * seq_dist) ÷ Σ contacts
                # (this will include self‐contacts on the diagonal if you want; you
                # can zero them out with contacts.fill_diagonal_(0) if not)
                contact_order = (contacts * seq_dist).sum() / contacts.sum()

                per_graph_feats.append(torch.stack([contact_density, contact_order]))

            # stack into [B,2]
            per_graph_feats = torch.stack(per_graph_feats, dim=0)
            feats.append(per_graph_feats)
        else:
            raise ValueError(f"Unknown protein feature {feature}")
    feats = [feat.unsqueeze(1) if feat.ndim == 1 else feat for feat in feats]
    return torch.cat(feats, dim=1).float()


@jaxtyped(typechecker=typechecker)
def compute_vector_protein_features(
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
    feats = []
    for feature in features:
        if feature == "eigenvectors":  # TODO
            raise ValueError("eigenvectors feature can be unstable")
            projected_pos = getattr(x, "pos_proj", project_node_positions(x))
            x["pos_proj"] = projected_pos
            evals, evecs = get_protein_eigen_features(x, x.pos)
            feats.append(evecs)
        elif feature == "farest_nodes":
            feats.append(batched_top_k_displacement(x.pos, x.batch, 10))
        elif feature == "nearest_nodes":
            feats.append(batched_top_k_displacement(x.pos, x.batch, 10, use_nearest=True))
        else:
            raise ValueError(f"Vector protein feature {feature} not recognised.")
    return torch.cat(feats, dim=1).float()

def batched_top_k_displacement(
        positions: torch.Tensor,           # (N,3)
        batch_index:  torch.LongTensor,    # (N,) with values in [0..B-1]
        K:      int,
        use_nearest: bool = False,
) -> torch.Tensor:
    B = int(batch_index.max().item()) + 1
    device, dtype = positions.device, positions.dtype

    # 1) per-cloud centroid
    sum_coords   = positions.new_zeros((B, 3)).index_add_(0, batch_index, positions)
    counts_long  = torch.bincount(batch_index, minlength=B).to(device)        # (B,)
    counts       = counts_long.to(dtype).unsqueeze(1)                        # (B,1)
    centroids    = sum_coords / counts                                       # (B,3)

    # 2) displacement & squared‐distance
    diffs   = positions - centroids[batch_index]                             # (N,3)
    dist2   = (diffs * diffs).sum(dim=1)                                      # (N,)

    # 3) group via single sort
    sorted_idx = torch.argsort(batch_index, stable=True)                     # (N,)
    diffs_s    = diffs[sorted_idx]                                           # (N,3)
    d2_s       = dist2[sorted_idx]                                           # (N,)

    # 4) compute start‐offset (“head”) of each batch run
    head = torch.cat([
        torch.zeros(1, dtype=torch.long, device=device),
        counts_long.cumsum(0)[:-1]       # ← no dtype/device args here!
    ], dim=0)                                                               # (B,)

    # ...the rest of your code stays exactly the same...

    # 5) make sure we can top‐k up to K by padding with the nearest‐of‐group
    max_cnt = int(counts_long.max().item())
    pad_len = max(max_cnt, K)
    arange  = torch.arange(pad_len, device=device)

    # clamp into valid indices per‐batch
    region_arange = arange.unsqueeze(0).expand(B, pad_len)                   # (B,pad_len)
    region_clamp  = region_arange.clamp(max=(counts_long - 1).unsqueeze(1))  # (B,pad_len)
    region_idx    = head.unsqueeze(1) + region_clamp                         # (B,pad_len)
    region_ds     = d2_s[region_idx]                                         # (B,pad_len)
    nearest_pos   = torch.argmin(region_ds, dim=1)                           # (B,)

    # build full index grid, with pads = nearest_pos
    in_range = region_arange < counts_long.unsqueeze(1)                      # (B,pad_len)
    idx_in   = head.unsqueeze(1) + region_arange                             # (B,pad_len)
    idx_pad  = head.unsqueeze(1) + nearest_pos.unsqueeze(1)                  # (B,pad_len)
    idx2d    = torch.where(in_range, idx_in, idx_pad)                        # (B,pad_len)

    # gather
    grp_diffs = diffs_s[idx2d]                                               # (B,pad_len,3)
    grp_d2    = d2_s[idx2d]                                                  # (B,pad_len)

    # mask pads so they only win when necessary
    fill = float('inf') if use_nearest else float('-inf')
    grp_d2 = grp_d2.masked_fill(~in_range, fill)

    # 6) top-K in one go
    _, topk_inds = torch.topk(grp_d2, K, dim=1, largest=not use_nearest)

    # 7) pull out the vectors
    topk_inds = topk_inds.unsqueeze(-1).expand(-1, -1, 3)                     # (B,K,3)
    return torch.gather(grp_diffs, 1, topk_inds)                              # (B,K,3)


def get_protein_eigen_features(batch):
    # Calculate mean positions for each protein using scatter_mean
    eigenvec, _, eigenval = scatter_eigen_decomp(batch.pos, batch.batch, len(batch.id))

    return eigenval, eigenvec

def project_node_positions(batch):
    pr_com = batch.sse_cell_complex.get_com(rank=3)
    return torch.bmm((batch.pos - pr_com[batch.batch]).unsqueeze(1), localize(batch, rank=3)[batch.batch]).squeeze(1)