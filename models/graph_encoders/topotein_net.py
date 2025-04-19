import torch
from beartype.typing import List
from torch import nn

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers import gcp
from proteinworkshop.models.utils import get_activations, get_aggregation, centralize
from proteinworkshop.types import EncoderOutput
from topotein.models.graph_encoders.layers.tcp import TCPInteractions
from topotein.models.graph_encoders.layers.topotein_net.backbone_encoder import BackboneEncoder
from topotein.models.graph_encoders.layers.topotein_net.embedding import TPPEmbedding
from topotein.models.utils import tensorize, localize


class TopoteinNetModel(nn.Module):
    @property
    def required_batch_attributes(self) -> List[str]:
        return ["edge_index", "pos", "x", "batch"]

    def __init__(self, in_dims_dict=None, out_dims_dict=None, num_layers=None, backbone_encoder_ckpt=None, freeze_backbone_encoder=None, activation=None, **kwargs):
        super().__init__()

        assert all(
            [cfg in kwargs for cfg in ["module_cfg", "model_cfg", "layer_cfg"]]
        ), "All required TopoteinNet `DictConfig`s must be provided."
        module_cfg = kwargs["module_cfg"]
        model_cfg = kwargs["model_cfg"]
        layer_cfg = kwargs["layer_cfg"]

        if in_dims_dict is None:
            self.in_dims_dict = {
                0: ScalarVector(model_cfg['h_input_dim'], model_cfg['chi_input_dim']),
                1: ScalarVector(model_cfg['e_input_dim'], model_cfg['xi_input_dim']),
                2: ScalarVector(model_cfg['c_input_dim'], model_cfg['rho_input_dim']),
            }
        else:
            self.in_dims_dict = in_dims_dict
        if out_dims_dict is None:
            self.out_dims_dict = {
                0: ScalarVector(model_cfg['h_hidden_dim'], model_cfg['chi_hidden_dim']),
                1: ScalarVector(model_cfg['e_hidden_dim'], model_cfg['xi_hidden_dim']),
                2: ScalarVector(model_cfg['c_hidden_dim'], model_cfg['rho_hidden_dim']),
            }
        else:
            self.out_dims_dict = out_dims_dict
        if num_layers is None:
            self.num_layers = model_cfg['num_layers']
        else:
            self.num_layers = num_layers
        if backbone_encoder_ckpt is None:
            self.backbone_encoder_ckpt = layer_cfg['backbone_encoder_ckpt']
        else:
            self.backbone_encoder_ckpt = backbone_encoder_ckpt
        if freeze_backbone_encoder is None:
            self.freeze_backbone_encoder = layer_cfg['freeze_backbone_encoder']
        else:
            self.freeze_backbone_encoder = freeze_backbone_encoder
        if activation is None:
            self.activation = module_cfg['nonlinearities'][0]
        else:
            self.activation = activation

        self.activation_name = self.activation
        self.activation = get_activations(self.activation)
        self.backbone_encoder = BackboneEncoder(
            in_dims_dict=self.in_dims_dict,
            out_dims_dict=self.out_dims_dict,
            num_layers=kwargs.get("backbone_num_layers", 6),
            pretrained_ckpt=self.backbone_encoder_ckpt,
            freeze_encoder=self.freeze_backbone_encoder
        )

        self.backbone_encoder_sse = BackboneEncoder(
            in_dims_dict=self.in_dims_dict,
            out_dims_dict=self.out_dims_dict,
            num_layers=kwargs.get("backbone_num_layers", 6),
            pretrained_ckpt=self.backbone_encoder_ckpt,
            freeze_encoder=self.freeze_backbone_encoder
        )

        # SSE embedding functions
        self.sse_emb = TPPEmbedding(
            in_dims_dict=self.in_dims_dict,
            out_dims_dict={2: ScalarVector(
                self.out_dims_dict[2].scalar,
                self.out_dims_dict[2].vector
            )},
            ranks=[2],
            bottleneck=1,
            activation=self.activation_name
        )

        self.sse_scalar_pre_recon = nn.Sequential(
            nn.Linear(self.out_dims_dict[0].scalar, self.out_dims_dict[2].scalar),
            self.activation,
            nn.Linear(self.out_dims_dict[2].scalar, self.out_dims_dict[2].scalar),
            self.activation
        )
        self.sse_scalar_down = nn.Sequential(
            nn.Linear(self.out_dims_dict[2].scalar * 2, self.out_dims_dict[2].scalar),
            self.activation,
            nn.Linear(self.out_dims_dict[2].scalar, self.out_dims_dict[2].scalar),
            self.activation
        )
        self.sse_vec_pre_recon = nn.Sequential(
            nn.Linear(self.out_dims_dict[0].scalar, self.out_dims_dict[2].vector * 3),
            self.activation,
            nn.Linear(self.out_dims_dict[2].vector * 3, self.out_dims_dict[2].vector * 3),
            self.activation
        )
        self.sse_vec_down = nn.Sequential(
            nn.Linear(self.out_dims_dict[2].vector * 2, self.out_dims_dict[2].vector),
            self.activation,
            nn.Linear(self.out_dims_dict[2].vector, self.out_dims_dict[2].vector),
            self.activation
        )

        # PR embedding functions
        self.pr_vec_pre_recon = nn.Sequential(
            nn.Linear(self.out_dims_dict[0].scalar, self.out_dims_dict[0].vector * 3),
            self.activation,
            nn.Linear(self.out_dims_dict[0].vector * 3, self.out_dims_dict[0].vector * 3),
            self.activation
        )
        self.pr_vec_post_recon = nn.Sequential(
            nn.Linear(self.out_dims_dict[0].vector, self.out_dims_dict[0].vector),
            self.activation,
            nn.Linear(self.out_dims_dict[0].vector, self.out_dims_dict[0].vector),
            self.activation
        )

        # interactions layers

        self.interaction_layers = nn.ModuleList(
            TCPInteractions(
                self.out_dims_dict[0],
                self.out_dims_dict[1],
                self.out_dims_dict[2],
                cfg=module_cfg,
                layer_cfg=layer_cfg,
                dropout=model_cfg.dropout,
            )
            for _ in range(model_cfg.num_layers)
        )

        self.invariant_node_projection = nn.ModuleList(
            [
                gcp.GCPLayerNorm(self.out_dims_dict[0]),
                gcp.GCP(
                    # Note: `GCPNet` defaults to providing SE(3) equivariance
                    # It is possible to provide E(3) equivariance by instead setting `module_cfg.enable_e3_equivariance=true`
                    self.out_dims_dict[0],
                    ScalarVector(self.out_dims_dict[0].scalar, 0),
                    nonlinearities=tuple(module_cfg.nonlinearities),
                    scalar_gate=module_cfg.scalar_gate,
                    vector_gate=module_cfg.vector_gate,
                    enable_e3_equivariance=module_cfg.enable_e3_equivariance,
                    node_inputs=True,
                ),
            ]
        )

        # Global pooling/readout function
        self.readout = get_aggregation(
            module_cfg.pool
        )  # {"mean": global_mean_pool, "sum": global_add_pool}[pool]


    def get_sse_emb(self, batch):
        sse_s, sse_v = self.sse_emb(batch)[2]
        sse_emb = self.backbone_encoder_sse(batch, rank=2)['graph_embedding']
        sse_s = torch.cat([
            sse_s,
            self.sse_scalar_pre_recon(sse_emb),
        ], dim=-1)
        sse_v = torch.cat([
            tensorize(self.sse_vec_pre_recon(sse_emb), frames=batch.frame_dict[2], flattened=True),
            sse_v.transpose(-1, -2)
        ], dim=-1)
        sse_v = self.sse_vec_down(sse_v)
        sse_s = self.sse_scalar_down(sse_s)
        return ScalarVector(sse_s, sse_v.transpose(-1, -2))

    def get_node_and_edge_emb(self, batch):
        self.backbone_encoder(batch, rank=3)
        (h, chi, e, xi) = batch.h, batch.chi, batch.e, batch.xi
        return h, chi, e, xi

    def get_pr_emb(self, batch):
        raise NotImplementedError
        # pr_emb = self.backbone_encoder(batch, rank=3)['graph_embedding']
        # print(f"pr_emb: {pr_emb.shape}")
        # pr_emb = self.pr_vec_post_recon(
        #     tensorize(
        #         self.pr_vec_pre_recon(pr_emb),
        #         frames=batch.frame_dict[3], flattened=True
        #     ))
        # return ScalarVector(torch.zeros(pr_emb.shape[0], 1), pr_emb)

    def forward(self, batch):
        pos_centroid, batch.pos = centralize(batch, batch_index=batch.batch, key="pos")
        batch.frame_dict = {i: localize(batch, rank=i) for i in range(3)}

        (c, rho) = self.get_sse_emb(batch)
        h, chi, e, xi = self.get_node_and_edge_emb(batch)

        for layer in self.interaction_layers:
            (h, chi), batch.pos = layer(
                node_rep=ScalarVector(h, chi),
                edge_rep=ScalarVector(e, xi),
                cell_rep=ScalarVector(c, rho),
                frames=batch.frame_dict[1],
                cell_frames=batch.frame_dict[2],
                edge_index=batch.edge_index,
                node_mask=getattr(batch, "mask", None),
                node_pos=batch.pos,
                node_to_sse_mapping=batch.N0_2,
            )

        # Record final version of each feature in `Batch` object
        batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = h, chi, e, xi, c, rho

        out = self.invariant_node_projection[0](
            ScalarVector(h, chi)
        )  # e.g., GCPLayerNorm()
        out = self.invariant_node_projection[1](
            out, batch.edge_index, batch.frame_dict[1], node_inputs=True
        )  # e.g., GCP((h, chi)) -> h'

        encoder_outputs = {"node_embedding": out, "graph_embedding": self.readout(
            out, batch.batch
        )}
        return EncoderOutput(encoder_outputs)
