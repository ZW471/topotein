from beartype.typing import List, Tuple, Union

import torch
from graphein.protein.tensor.data import Protein, ProteinBatch
from omegaconf import ListConfig
from torch_geometric.data import Batch, Data

from topotein.features.sse import get_sparse_sse_cell_group
from topotein.features.topotein_complex import TopoteinComplex

CATEGORY_DSSP_3 = {"H", "E", "C"}
CATEGORY_DSSP_8 = {"H", "B", "E", "G", "I", "T", "S", "C"}


def compute_sses_pure_torch(
        x: Union[Data, Batch, Protein, ProteinBatch],
        sse_types: Union[ListConfig, List[str]]
) -> Tuple[torch.Tensor, torch.tensor, TopoteinComplex]:
    """
    Compute the secondary structure element (SSE) grouping and associated data for the input structure
    in a PyTorch-based processing pipeline. This function determines the applicable SSE classification
    scheme depending on the input `sse_types`, and processes the data accordingly to provide cell indices,
    cell types, and a `TopoteinComplex` object.

    :param x: Input structure data which can be of types Data, Batch, Protein, or ProteinBatch.
    :param sse_types: List of SSE types provided as either ListConfig or a list of strings.
    :return: A tuple containing:
        - torch.Tensor: The tensor of cell types representing processed SSE classes.
        - torch.Tensor: The tensor of cell indices corresponding to grouped SSE cells.
        - TopoteinComplex: An object containing the processed data, which includes the number
          of nodes, edge index, and cell indices.
    """
    sse_types = set(sse_types)
    if len(sse_types.difference(CATEGORY_DSSP_3)) == 0:  # no type that is not in 3-class category
        is_using_simple_categories = True
    elif len(sse_types.union(CATEGORY_DSSP_8)) > 0:  # some types are in the 8-class category
        is_using_simple_categories = False
    else:  # some types are out of the 8-class category
        raise ValueError(f"invalid sse types, valid sse types are: {CATEGORY_DSSP_8}")

    if is_using_simple_categories:
        groups_dict = get_sparse_sse_cell_group(x, x.sse, x.batch, num_of_classes=3, minimal_group_size=3)
        cell_index = groups_dict.indices()
        cell_type = groups_dict.values()
        return cell_type, cell_index, TopoteinComplex(
            num_nodes=x.num_nodes,
            edge_index=x.edge_index,
            sse_index=cell_index,
            node_pos=x.pos,
            num_proteins=len(x.id),
            protein_batch=x.batch
        )
    else:
        raise NotImplementedError(f"only 3-class scheme implemented")
