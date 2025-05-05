import torch

from topotein.models.utils import get_com


def zero_diagonal(sparse_tensor):
    # Get the indices and values (assumes 2D tensor)
    indices = sparse_tensor._indices()  # shape: [2, nnz]
    values = sparse_tensor._values()  # shape: [nnz]

    # Create a mask that is True for non-diagonal entries (row index != column index)
    mask = indices[0] != indices[1]

    # Filter out the diagonal indices and corresponding values
    new_indices = indices[:, mask]
    new_values = values[mask]

    # Create and return a new sparse tensor with the same shape
    return torch.sparse_coo_tensor(new_indices, new_values, sparse_tensor.shape, device=sparse_tensor.device)


def map_edges_to_cells(edge_indices: torch.Tensor, cell_indices: torch.Tensor):
    """
    Map each edge to the cell that contains it.

    Args:
        edge_indices (torch.Tensor): A tensor of shape [2, a] where:
            - edge_indices[0, :] are the x values of the edges.
            - edge_indices[1, :] are the y values of the edges.
        cell_indices (torch.Tensor): A tensor of shape [2, b] where:
            - cell_indices[0, :] are the lower bounds (m_j) for each cell.
            - cell_indices[1, :] are the upper bounds (n_j) for each cell.

    Returns:
        tuple: A tuple containing:
            - indices (torch.Tensor): A tensor of shape [2, nnz] with the indices of the sparse mapping.
            - values (torch.Tensor): A tensor of shape [nnz] with the corresponding values (1 for each mapping).

    Note:
        An edge is mapped to a cell if both of its endpoints satisfy:
            m_j <= edge <= n_j,
        where m_j and n_j are the lower and upper bounds of the cell, respectively.
    """

    # Expand dimensions to allow broadcasting.
    edges_x = edge_indices[0, :].unsqueeze(1)  # shape: [a, 1]
    edges_y = edge_indices[1, :].unsqueeze(1)  # shape: [a, 1]

    cells_lower = cell_indices[0, :].unsqueeze(0)  # shape: [1, b]
    cells_upper = cell_indices[1, :].unsqueeze(0)  # shape: [1, b]

    # Create boolean masks for both x and y conditions.
    mask_x = (edges_x >= cells_lower) & (edges_x <= cells_upper)  # shape: [a, b]
    mask_y = (edges_y >= cells_lower) & (edges_y <= cells_upper)  # shape: [a, b]

    # Combine masks: an edge is in a cell if both x and y are within the cell bounds.
    mask = mask_x & mask_y  # shape: [a, b]

    # Convert boolean mask to an integer tensor.
    dense_tensor = mask.to(torch.int32)

    # Extract indices and corresponding values where there is a mapping.
    indices = torch.nonzero(dense_tensor, as_tuple=False).t()  # shape: [2, nnz]
    # values = dense_tensor[dense_tensor != 0]  # no need since all will be 1

    return indices


def map_edges_to_cells_searchsorted(edge_indices: torch.Tensor, cell_indices: torch.Tensor):
    """
    Map each edge to the cell that contains it using binary search via torch.searchsorted.

    Args:
        edge_indices (torch.Tensor): A tensor of shape [2, num_edges] where:
            - edge_indices[0, :] are the x values of the edges.
            - edge_indices[1, :] are the y values of the edges.
        cell_indices (torch.Tensor): A tensor of shape [2, num_cells] where:
            - cell_indices[0, :] are the lower bounds for each cell (sorted in ascending order).
            - cell_indices[1, :] are the upper bounds for each cell.

    Returns:
        torch.Tensor: A tensor of shape [2, num_mapped_edges] where:
            - The first row contains the cell indices.
            - The second row contains the corresponding edge indices.
            An edge is mapped to a cell only if both endpoints lie within the cell bounds.
    """
    # Extract cell lower and upper bounds
    cells_lower = cell_indices[0]  # shape: [num_cells]
    cells_upper = cell_indices[1]  # shape: [num_cells]

    # Extract edge coordinates
    edges_x = edge_indices[0]  # shape: [num_edges]
    edges_y = edge_indices[1]  # shape: [num_edges]

    # For each edge coordinate, find the candidate cell index using binary search.
    # torch.searchsorted returns the index where the element should be inserted to maintain order.
    candidate_x = torch.searchsorted(cells_lower, edges_x, right=True) - 1
    candidate_y = torch.searchsorted(cells_lower, edges_y, right=True) - 1

    # Check that the candidate indices are valid and that the edge coordinates are within the cell's upper bound.
    valid_x = (candidate_x >= 0) & (edges_x <= cells_upper[candidate_x])
    valid_y = (candidate_y >= 0) & (edges_y <= cells_upper[candidate_y])

    # Only map an edge if both endpoints are valid and fall in the same cell.
    valid = valid_x & valid_y & (candidate_x == candidate_y)

    # Get indices of the edges that are valid.
    valid_edge_indices = torch.nonzero(valid, as_tuple=False).squeeze()
    # The corresponding cell index (candidate_x equals candidate_y where valid).
    mapped_cell_indices = candidate_x[valid]

    # Stack the cell and edge indices to get the mapping.
    mapping = torch.stack([valid_edge_indices, mapped_cell_indices], dim=0)
    return mapping


class TopoteinComplex:
    def __init__(self, num_nodes, edge_index, sse_index, node_pos, num_proteins=None, protein_batch=None, use_cache=True):
        self.device = edge_index.device

        self.num_nodes = num_nodes
        self.num_edges = edge_index.shape[1]
        self.num_sses = sse_index.shape[1]
        self.num_proteins = num_proteins

        self.nodes = torch.arange(num_nodes, device=self.device)
        self.edge_index = edge_index
        self.sse_index = sse_index
        self.protein_batch = protein_batch

        self.use_cache = use_cache
        self.cache = {}

        self.node_pos = node_pos
        self.com_dict = {}

        self.calculator = None
        self._centered_pos = None

    @property
    def centered_pos(self):
        pos = None
        if self.use_cache:
            pos = self._centered_pos

        if pos is None:
            pos = self.node_pos - self.get_com(rank=3, relative=False)[self.incidence_matrix(from_rank=0, to_rank=3).indices()[1]]
            if self.use_cache:
                self._centered_pos = pos

        return pos

    def get_com(self, rank, relative: bool = True):
        com = None
        dict_name = f"{rank}_{'rel' if relative else 'abs'}"
        if self.use_cache:
            com = self.com_dict.get(dict_name, None)
        if com is None:
            if rank == 0:
                com = self.node_pos
            elif rank == 1:
                com = (self.node_pos[self.edge_index[0]] + self.node_pos[self.edge_index[1]]) / 2
            elif rank == 2:
                B0_2_idx = self.incidence_matrix(from_rank=0, to_rank=2).indices()
                com = get_com(
                    positions=self.node_pos[B0_2_idx[0]],
                    cluster_ids=B0_2_idx[1],
                    cluster_num=self.num_sses,
                )
            elif rank == 3:
                com = get_com(self.node_pos, self.protein_batch, self.num_proteins)
            else:
                raise ValueError(f'Invalid rank: {rank}')

            if self.use_cache:
                self.com_dict[f"{rank}_abs"] = com

            if relative:
                if rank < 3:
                    com = com - self.get_com(rank=3, relative=False)[
                        self.incidence_matrix(from_rank=rank, to_rank=3).indices()[1]
                    ]
                else:
                    com = torch.zeros_like(com, device=self.device)

                if self.use_cache:
                    self.com_dict[dict_name] = com
        return com

    def set_proteins(self, num_proteins, protein_batch):
        self.num_proteins = num_proteins
        self.protein_batch = protein_batch

    def _read_neighborhood_cache(self, neighborhood_type, rank, to_rank):
        if self.use_cache and f'{neighborhood_type}_{rank}_{to_rank}' in self.cache:
            return self.cache[f'{neighborhood_type}_{rank}_{to_rank}']
        else:
            return None

    def _write_neighborhood_cache(self, neighborhood_type, rank, to_rank, neighborhood):
        if self.use_cache:
            self.cache[f'{neighborhood_type}_{rank}_{to_rank}'] = neighborhood

    def _get_size_of_rank(self, rank):
        if rank == 0:
            return self.num_nodes
        elif rank == 1:
            return self.num_edges
        elif rank == 2:
            return self.num_sses
        elif rank == 3:
            return self.num_proteins
        else:
            raise ValueError(f'Invalid rank: {rank}')

    def laplacian_matrix(self, rank, via_rank):
        cache = self._read_neighborhood_cache(neighborhood_type='L', rank=rank, to_rank=via_rank)
        if cache is not None:
            return cache

        if rank == via_rank:
            raise ValueError(f'Invalid rank: rank={rank} equals via_rank={via_rank}')

        # i < j
        i = min(rank, via_rank)
        j = max(rank, via_rank)
        B_i_j = self.incidence_matrix(from_rank=i, to_rank=j)
        if j == 1 and i == 0:
            B_j_i = self.incidence_matrix(from_rank=j, to_rank=i)
        else:
            B_j_i = self.incidence_matrix(from_rank=i, to_rank=j).T.coalesce()

        if rank > via_rank:
            B_i_j, B_j_i = B_j_i, B_i_j

        result = torch.sparse.mm(B_i_j, B_j_i)

        self._write_neighborhood_cache(neighborhood_type='L', rank=rank, to_rank=via_rank, neighborhood=result)

        return result

    def adjacency_matrix(self, rank, via_rank):
        cache = self._read_neighborhood_cache(neighborhood_type='A', rank=rank, to_rank=via_rank)
        if cache is not None:
            return cache

        if rank == via_rank:
            raise ValueError(f'Invalid rank: rank={rank} equals via_rank={via_rank}')

        result = zero_diagonal(self.laplacian_matrix(rank=rank, via_rank=via_rank))

        self._write_neighborhood_cache(neighborhood_type='A', rank=rank, to_rank=via_rank, neighborhood=result)

        return result

    def incidence_matrix(self, from_rank, to_rank):
        neighborhood_type = 'B'
        cache = self._read_neighborhood_cache(neighborhood_type=neighborhood_type, rank=from_rank, to_rank=to_rank)
        if cache is not None:
            return cache

        if from_rank > to_rank and not (from_rank == 1 and to_rank == 0):
            raise ValueError(
                f'Invalid rank: from_rank={from_rank} > to_rank={to_rank} and not (from_rank=1 and to_rank=0)')

        supported_ranks = [0, 1, 2, 3] if self.protein_batch is not None else [0, 1, 2]
        if not (from_rank == 1 and to_rank == 0):
            if from_rank not in supported_ranks[:-1]:
                raise ValueError(f'Invalid rank: from_rank={from_rank}, supported ranks: {supported_ranks}')
            if to_rank not in supported_ranks[1:]:
                raise ValueError(f'Invalid rank: to_rank={to_rank}, supported ranks: {supported_ranks}')

        row, col = None, None
        from_rank_size, to_rank_size = self._get_size_of_rank(from_rank), self._get_size_of_rank(to_rank)
        if to_rank == 0:
            if from_rank == 1:
                # this is needed to allow directions in edges
                col = torch.arange(self.num_edges, device=self.device)
                row = self.edge_index[0]
        elif to_rank == 1:
            if from_rank == 0:
                col = self.edge_index[1]
                row = torch.arange(self.num_edges, device=self.device)
        elif to_rank == 2:
            if from_rank == 0:
                cell_starts = self.sse_index[0, :]  # shape: [b]
                cell_ends = self.sse_index[1, :]  # shape: [b]
                lengths = cell_ends - cell_starts + 1  # shape: [b]

                # Compute cumulative lengths to know the starting index for each cell in the concatenated vector.
                cumulative = torch.cat([torch.zeros(1, device=self.device, dtype=lengths.dtype), lengths.cumsum(dim=0)])
                total_length = cumulative[-1]

                # Create a vector of offsets for all entries combined.
                offsets = torch.arange(total_length, device=self.device)

                # For each offset, determine which cell it belongs to.
                # The cumulative vector is sorted, so we can use searchsorted.
                cell_ids = torch.searchsorted(cumulative, offsets, right=True) - 1

                # Compute the offset within each cell.
                relative_offsets = offsets - cumulative[cell_ids]

                # The column indices are then the cell start plus the relative offset.
                col = cell_starts[cell_ids] + relative_offsets

                # The row indices correspond to the cell id repeated by its length.
                row = torch.repeat_interleave(torch.arange(self.sse_index.shape[1], device=self.device), lengths)

            if from_rank == 1:
                indices = map_edges_to_cells_searchsorted(self.edge_index, self.sse_index)
                row, col = indices[1], indices[0]
        elif to_rank == 3:
            if from_rank == 0:
                col = self.nodes
                row = self.protein_batch
            elif from_rank == 1:
                col = torch.arange(self.num_edges, device=self.device)
                row = self.protein_batch[self.edge_index[0]]
            elif from_rank == 2:
                col = torch.arange(self.num_sses, device=self.device)
                row = self.protein_batch[self.sse_index[0]]

        result = torch.sparse_coo_tensor(
            indices=torch.stack([col, row], dim=0),
            values=torch.ones(col.shape[0], device=self.device),
            size=(from_rank_size, to_rank_size),
            device=self.device,
            dtype=torch.float
        ).coalesce()

        self._write_neighborhood_cache(neighborhood_type=neighborhood_type, rank=from_rank, to_rank=to_rank,
                                       neighborhood=result)

        return result
