import torch
from beartype.typing import Set

from proteinworkshop.models.utils import get_aggregation
from topotein.models.graph_encoders.sotas.etnn.model import ETNNCore


class ETNNModel(torch.nn.Module):
    def __init__(self, in_dim0=57, in_dim1=2, in_dim2=4, emb_dim=512, dropout=0.1, num_layers=6, activation="relu",pool="mean",
                 # Note: Each of the arguments above are stored in the corresponding `kwargs` configs below
                 # They are simply listed here to highlight key available arguments
                 # They will only be used when debug=True
                 **kwargs):
        super().__init__()

        self.debug = kwargs.get("debug", False)

        if self.debug:
            self.in_dim0 = in_dim0
            self.in_dim1 = in_dim1
            self.in_dim2 = in_dim2
            self.emb_dim = emb_dim
            self.dropout = dropout
            self.num_layers = num_layers
            self.activation = activation
            self.pool = pool
        else:
            assert all(
                [cfg in kwargs for cfg in ["model_cfg", "layer_cfg"]]
            ), "All required ETNN `DictConfig`s must be provided."
            module_cfg = kwargs["model_cfg"]

            for k, v in module_cfg.items():
                setattr(self, k, v)

        self.core = ETNNCore(
            num_features_per_rank={0: self.in_dim0, 1: self.in_dim1, 2: self.in_dim2, 3: self.in_dim3},
            num_hidden=self.emb_dim,
            num_out=self.emb_dim,
            num_layers=self.num_layers,
            adjacencies=["0_0_1", "0_0_2", "1_0", "2_0", "0_2", "2_3", "3_2"],
            initial_features=[],
            visible_dims=[0, 1, 2, 3],
            normalize_invariants=True,
            hausdorff_dists=True,
            batch_norm=True,
            sparse_invariant_computation=True,
            pos_update=False,
        )

        self.emb_0 = torch.nn.Linear(self.in_dim0, self.emb_dim)
        self.pool = get_aggregation(self.pool)

    def forward(self, batch):
        X = batch.pos

        device = X.device

        
        N2_0 = batch.N2_0
        N1_0 = batch.N1_0
        N0_0_via_1 = batch.N0_0_via_1
        N0_0_via_2 = batch.N0_0_via_2

        N0_2 = batch.N0_2
        N0_3 = batch.N0_3
        N2_3 = batch.N2_3
        N3_2 = batch.N3_2
        N2_1_outer = batch.N2_1_outer
        N1_2_outer = batch.N1_2_outer

        def cell_list(i, format="list"):
            all_nodes = torch.arange(0, batch.x.shape[0], device=device)
            cell_index_3 = [all_nodes[batch.batch == batch_idx] for batch_idx in range(len(batch.id))]
            cell_index_2 = [torch.arange(inner_list[0], inner_list[1] + 1, device=device) for inner_list in batch.sse_cell_index_simple.T]
            cell_index_1 = list(batch.edge_index.T)
            cell_index_0 = list(torch.arange(0, batch.x.shape[0], device=device).T.unsqueeze(1))
            list_of_cells = [
                cell_index_0,
                cell_index_1,
                cell_index_2,
                cell_index_3,
            ]
            return list_of_cells[i]

        batch.cell_list = cell_list

        def get_adj(neighborhood_matrix):
            return neighborhood_matrix.indices()[:, neighborhood_matrix.values() > 0]

        batch.adj_0_0_1 = get_adj(N0_0_via_1)
        batch.adj_0_0_2 = get_adj(N0_0_via_2)
        batch.adj_1_0 = get_adj(N1_0)
        batch.adj_2_0 = get_adj(N2_0)
        batch.adj_0_2 = get_adj(N0_2)
        batch.adj_2_3 = get_adj(N2_3)
        batch.adj_3_2 = get_adj(N3_2)
        # batch.adj_2_1 = get_adj(N2_1_outer)
        # batch.adj_1_2 = get_adj(N1_2_outer)
        batch.adj_0_3 = get_adj(N0_3)


        out = self.core(batch)
        return out

    @property
    def required_batch_attributes(self) -> Set[str]:
        return {"x", "pos", "edge_attr", "batch", "N2_0", "N1_0", "N0_0_via_1", "N0_0_via_2", "sse_attr"}


#%%
if __name__ == "__main__":
    #%%
    batch = torch.load("/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_featurised_batch_edge_processed_simple.pt", weights_only=False)
    print(batch)
    #%%
    model = ETNNModel(debug=True, activation="silu", num_layers=6, dropout=0, layer_cfg={
        "norm": "layer",
    })
    import time
    tik = time.time()
    out = model(batch)
    tok = time.time()
    print(f"Time taken: {tok-tik} seconds")
    #%%
    print(batch.pos)
    print(out['pos'])
    #%%

    assert torch.allclose(batch.pos, out['pos'], atol=10)
    assert torch.allclose(batch.pos, out['pos'], atol=1)
    print("All close")

