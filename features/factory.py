from typing import List, Union

import networkx as nx
import torch
from graphein.protein.tensor.data import ProteinBatch
from toponetx import CellComplex
from torch_geometric.data import Batch

from proteinworkshop.features.edge_features import compute_scalar_edge_features, compute_vector_edge_features
from proteinworkshop.features.factory import ProteinFeaturiser, StructureRepresentation
from topotein.features.protein_features import compute_vector_protein_features, compute_scalar_protein_features
from topotein.features.sse_features import compute_scalar_sse_features, compute_vector_sse_features
from topotein.features.cell_complex import compute_sses, compute_sses_pure_torch
from topotein.features.neighborhoods import compute_neighborhoods
from topotein.features.sse import sse_onehot
from proteinworkshop.types import ScalarNodeFeature, VectorNodeFeature, ScalarEdgeFeature, VectorEdgeFeature
from topotein.types import ScalarSSEFeature, VectorSSEFeature, ScalarProteinFeature, VectorProteinFeature


# note: remember to add feature dimension in the models.utils.get_input_dim function

class TopoteinFeaturiser(ProteinFeaturiser):
    def __init__(
        self,
        representation: StructureRepresentation,
        scalar_node_features: List[ScalarNodeFeature],
        vector_node_features: List[VectorNodeFeature],
        edge_types: List[str],
        scalar_edge_features: List[ScalarEdgeFeature],
        vector_edge_features: List[VectorEdgeFeature],
        sse_types: List[str],
        scalar_sse_features: List[ScalarSSEFeature],
        vector_sse_features: List[VectorSSEFeature],
        scalar_pr_features: List[ScalarProteinFeature],
        vector_pr_features: List[VectorProteinFeature],
        neighborhoods: List[str],
        directed_edges: bool = False,
        pure_torch: bool = False,
    ):
        if not pure_torch:
            super(TopoteinFeaturiser, self).__init__(representation, scalar_node_features, vector_node_features, edge_types, [], [])
            self.scalar_edge_features_after_sse = scalar_edge_features
            self.vector_edge_features_after_sse = vector_edge_features
        else:
            super(TopoteinFeaturiser, self).__init__(representation, scalar_node_features, vector_node_features, edge_types, scalar_edge_features, vector_edge_features)

        self.sse_types = sse_types
        self.scalar_sse_features = scalar_sse_features
        self.vector_sse_features = vector_sse_features
        self.scalar_pr_features = scalar_pr_features
        self.vector_pr_features = vector_pr_features
        # edge features should be calculated after attaching cells
        self.neighborhoods = neighborhoods
        self.directed_edges = directed_edges
        self.pure_torch = pure_torch

    def forward(
        self, batch: Union[Batch, ProteinBatch]
    ) -> Union[Batch, ProteinBatch]:
        
        # if type(batch) != ProteinBatch:
        #     # If the input is a Batch, we need to convert it to a ProteinBatch
        #     pr_batch = ProteinBatch()
        #     batch = pr_batch.from_batch(batch)
        #     # log.warning(
        #     #     f"Converting Batch to ProteinBatch. Batch size: {batch.batch_size} {type(batch)}"
        #     # )

        batch.sse = sse_onehot(batch)  # this is node sse
        batch = super().forward(batch)

        # cells
        if self.sse_types:
            if self.pure_torch:
                batch.num_nodes = batch.x.shape[0]
                batch.sse, batch.sse_cell_index_simple, batch.sse_cell_complex = compute_sses_pure_torch(
                    batch, self.sse_types
                )
                batch.sse = torch.nn.functional.one_hot(batch.sse, num_classes=len(self.sse_types))

            else:
                # sse: onehot type of sse for groups
                # sse_cell_index: cells that represents sses
                # sse_cell_complex: the whole cell complex that contains structural information of this higher-order graph
                batch.sse, batch.sse_cell_index, batch.sse_cell_complex = compute_sses(
                    batch, self.sse_types, directed_edges=self.directed_edges
                )
                batch.num_sse_type = len(self.sse_types)
                batch.sse_cell_index_simple = torch.tensor(
                    [(t[0], t[-1]) for t in batch.sse_cell_index],
                    dtype=torch.long,
                    device=batch.x.device,
                ).T


                # recreate the edge indices
                cc: CellComplex = batch.sse_cell_complex
                edge_attr_dict = nx.get_edge_attributes(cc._G, "edge_type", default=batch.num_relation)
                device = batch.x.device
                batch.edge_index = torch.tensor(list(edge_attr_dict.keys()), dtype=torch.long, device=device).T
                batch.edge_type = torch.tensor(list(edge_attr_dict.values()), device=device).unsqueeze(0)
                batch.num_relation += 1  # a new kind of edge is introduced by sse cells (connecting SSE start with end)

                # note: this does not align with neighborhood matrices
                # if not self.directed_edges:
                #     batch.edge_index = torch.cat([batch.edge_index, batch.edge_index.flip([0])], dim=1)
                #     batch.edge_type = torch.cat([batch.edge_type, batch.edge_type], dim=1)


        # Scalar cell features
        if self.scalar_sse_features:
            batch.sse_attr = compute_scalar_sse_features(
                batch, self.scalar_sse_features
            )

        # Vector cell features
        if self.vector_sse_features:
            batch.sse_vector_attr = compute_vector_sse_features(
                batch, self.vector_sse_features
            )

        if self.neighborhoods:
            neighborhoods = compute_neighborhoods(
                batch, self.neighborhoods
            )
            for name, value in neighborhoods.items():
                batch[name] = value

        if self.scalar_pr_features:
            batch.pr_attr = compute_scalar_protein_features(
                batch, self.scalar_pr_features
            )

        if self.vector_pr_features:
            batch.pr_vector_attr = compute_vector_protein_features(
                batch, self.vector_pr_features
            )

        if not self.pure_torch:
            # Scalar edge features
            if self.scalar_edge_features_after_sse:
                batch.edge_attr = compute_scalar_edge_features(
                    batch, self.scalar_edge_features_after_sse
                )

            # Vector edge features
            if self.vector_edge_features_after_sse:
                batch = compute_vector_edge_features(
                    batch, self.vector_edge_features_after_sse
                )

        return batch

    def __repr__(self) -> str:
        return f"TopoteinFeaturiser(representation={self.representation}, scalar_node_features={self.scalar_node_features}, vector_node_features={self.vector_node_features}, \
            edge_types={self.edge_types}, scalar_edge_features={self.scalar_edge_features if self.pure_torch else self.scalar_edge_features_after_sse}, vector_edge_features={self.vector_edge_features if self.pure_torch else self.vector_edge_features_after_sse}, sse_types={self.sse_types}, \
            scalar_sse_features={self.scalar_sse_features}, vector_sse_features={self.vector_sse_features}, \
            scalar_pr_features={self.scalar_pr_features}, vector_pr_features={self.vector_pr_features}, neighborhoods={self.neighborhoods}, directed_edges={self.directed_edges}, pure_torch={self.pure_torch})"


#%%

if __name__ == "__main__":
    import hydra
    import omegaconf
    import time
    from proteinworkshop import constants

    cfg = omegaconf.OmegaConf.load(
        constants.PROJECT_PATH
        / "proteinworkshop"
        / "config"
        / "features"
        / "ca_bb_sse.yaml"
    )
    cfg['vector_node_features'] += ['orientation']
    cfg['scalar_edge_features'] += ['rbf']
    cfg['vector_edge_features'] += ["edge_vectors"]
    cfg['vector_sse_features'] += ["sse_vectors", "consecutive_diff", "pr_com_diff"]
    cfg['vector_pr_features'] += ["farest_nodes", "nearest_nodes"]
    cfg['neighborhoods'] = ['N0_2 = B0_2']
    # cfg['pure_torch'] = True
    featuriser = hydra.utils.instantiate(cfg)
    batch: ProteinBatch = torch.load('/Users/dricpro/PycharmProjects/Topotein/test/data/sample_batch/sample_batch_unfeaturised.pt', weights_only=False)
    print(batch)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch.to(device)
    featuriser.to(device)
    tik = time.time()
    batch = featuriser(batch)
    print(batch, f'featurising time: {time.time() - tik}s')

    # torch.save(batch, '../../../test/data/sample_batch/sample_batch_for_tcp.pt')
