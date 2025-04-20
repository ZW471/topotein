"""Utilities for computing cell features."""
from typing import List, Union

import numpy as np
import torch
import torch_scatter
from beartype import beartype as typechecker
from graphein.protein.tensor.data import ProteinBatch
from graphein.protein.tensor.types import CoordTensor, EdgeTensor
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
import toponetx as tnx
from proteinworkshop.features.utils import _normalize
from torch.nn.utils.rnn import pad_sequence
from proteinworkshop.models.utils import centralize
from topotein.features.topotein_complex import TopoteinComplex
from topotein.models.utils import localize
from torch.nn import functional as F

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
    vector_cell_features = []
    for feature in features:
        if feature == "pr_vectors":  # TODO
            pass
        else:
            raise ValueError(f"Vector protein feature {feature} not recognised.")
    return torch.cat(vector_cell_features, dim=1).float()
