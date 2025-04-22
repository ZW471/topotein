from torch import nn

from proteinworkshop.models.graph_encoders.layers.gcp import GCPLayerNorm


class TPPNorm(nn.Module):
    def __init__(self, dim_dict, eps=1e-8, use_norm_ranks=None):
        super().__init__()
        self.eps = eps
        self.gcp_layer_norm = nn.ModuleDict()
        if use_norm_ranks is None:
            use_norm_ranks = []
        for rank in dim_dict:
            self.gcp_layer_norm[str(rank)] = GCPLayerNorm(dim_dict[rank], eps=eps, use_gcp_norm=rank in use_norm_ranks)

    def forward(self, X_dict: dict) -> dict:
        out = {}
        for rank in X_dict:
            out[rank] = self.gcp_layer_norm[str(rank)](X_dict[rank])
        return out