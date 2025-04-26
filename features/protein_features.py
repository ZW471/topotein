"""Utilities for computing cell features."""
from typing import List, Union
import torch
import torch_scatter
from beartype import beartype as typechecker
from jaxtyping import jaxtyped
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data
from torch_scatter import scatter_mean, scatter_std
from topotein.models.utils import localize, get_com

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
            projected_pos = getattr(x, "pos_proj", project_node_positions(x))
            x["pos_proj"] = projected_pos
            evals, evecs = get_protein_eigen_features(x, x.pos)
            linearity = (evals[:, 0] - evals[:, 1]) / evals[:, 0]
            planarity = (evals[:, 1] - evals[:, 2]) / evals[:, 1]
            scattering = evals[:, 2] / evals[:, 0]
            omnivariance = (evals[:, 0] * evals[:, 1] * evals[:, 2]) ** (1 / 3)
            anisotropy = (evals[:, 0] - evals[:, 2]) / evals[:, 0]
            feats.append(torch.concat([
                torch.stack([linearity, planarity, scattering, omnivariance, anisotropy], dim=-1),
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
            projected_pos = getattr(x, "pos_proj", project_node_positions(x))
            x["pos_proj"] = projected_pos
            evals, evecs = get_protein_eigen_features(x, x.pos)
            feats.append(evals)
        else:
            raise ValueError(f"Vector protein feature {feature} not recognised.")
    return torch.cat(feats, dim=1).float()


def get_protein_eigen_features(batch, projected_pos):
    # Calculate mean positions for each protein using scatter_mean
    protein_means = scatter_mean(projected_pos, batch.batch, dim=0)

    # Center the data for all proteins at once
    centered_pos = projected_pos - protein_means[batch.batch]

    # Get unique proteins
    unique_proteins = torch.arange(len(batch.id))
    protein_eigenvalues = []
    protein_eigenvectors = []

    # Process all proteins
    for protein_id in unique_proteins:
        protein_mask = batch.batch == protein_id
        # Get centered data for this protein
        protein_centered_data = centered_pos[protein_mask]

        # Calculate covariance matrix
        N = protein_mask.sum()
        protein_cov_matrix = torch.mm(protein_centered_data.T, protein_centered_data) / (N - 1)

        # Calculate eigenvalues and eigenvectors
        protein_evals, protein_evecs = torch.linalg.eigh(protein_cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = torch.argsort(protein_evals, descending=True)
        protein_evals = protein_evals[idx]
        protein_evecs = protein_evecs[:, idx]

        # Ensure right-handed coordinate system
        # Calculate the cross product of first two eigenvectors
        cross_prod = torch.cross(protein_evecs[:, 0], protein_evecs[:, 1])
        # If the dot product with the third eigenvector is negative, flip the third eigenvector
        if torch.dot(cross_prod, protein_evecs[:, 2]) < 0:
            protein_evecs[:, 2] = -protein_evecs[:, 2]

        protein_eigenvalues.append(protein_evals)
        protein_eigenvectors.append(protein_evecs)

    return torch.stack(protein_eigenvalues), torch.stack(protein_eigenvectors)

def project_node_positions(batch):
    pr_com = get_com(
        batch.pos,
        cluster_ids=batch.batch,
        cluster_num=len(batch.id)
    )
    return torch.bmm((batch.pos - pr_com[batch.batch]).unsqueeze(1), localize(batch, rank=3)[batch.batch]).squeeze(1)