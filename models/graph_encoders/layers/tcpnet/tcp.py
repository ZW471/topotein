from beartype import beartype as typechecker

from proteinworkshop.models.graph_encoders.components.wrappers import ScalarVector
from proteinworkshop.models.graph_encoders.layers.gcp import GCP
from typing import Tuple, Union
from graphein.protein.tensor.data import ProteinBatch
import torch
from jaxtyping import Float, jaxtyped

from topotein.models.utils import centralize

from proteinworkshop.models.utils import safe_norm


class TCP(GCP):

    @jaxtyped(typechecker=typechecker)
    def scalarize(
        self,
        vector_rep: Float[torch.Tensor, "batch_num_entities 3 3"],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"],
        enable_e3_equivariance: bool,
    ) -> Float[torch.Tensor, "effective_batch_num_entities out_scalar_dim"]:
        # important!!! frames in tcp use rows to store vector while frames in gcp use cols
        if enable_e3_equivariance:
            frames[:, 1, :] = frames[:, 1, :].abs()
        return torch.bmm(vector_rep, frames).reshape(vector_rep.shape[0], -1)

    # noinspection PyMethodOverriding
    def forward(
        self,
        s_maybe_v: Union[
            Tuple[
                Float[torch.Tensor, "batch_num_entities scalar_dim"],
                Float[torch.Tensor, "batch_num_entities m vector_dim"],
            ],
            Float[torch.Tensor, "batch_num_entities merged_scalar_dim"],
        ],
        frames: Float[torch.Tensor, "batch_num_edges 3 3"]
    ) -> Union[
        Tuple[
            Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
            Float[torch.Tensor, "batch_num_entities n vector_dim"],
        ],
        Float[torch.Tensor, "batch_num_entities new_scalar_dim"],
    ]:
        if self.vector_input_dim:
            scalar_rep, vector_rep = s_maybe_v
            v_pre = vector_rep.transpose(-1, -2)

            vector_hidden_rep = self.vector_down(v_pre)
            vector_norm = safe_norm(vector_hidden_rep, dim=-2)
            merged = torch.cat((scalar_rep, vector_norm), dim=-1)

            # curate direction-robust and (by default) chirality-aware scalar geometric features
            vector_down_frames_hidden_rep = self.vector_down_frames(v_pre)
            scalar_hidden_rep = self.scalarize(
                vector_down_frames_hidden_rep.transpose(-1, -2),
                frames,
                enable_e3_equivariance=self.enable_e3_equivariance,
            )
            merged = torch.cat((merged, scalar_hidden_rep), dim=-1)
        else:
            # bypass updating scalar features using vector information
            merged = s_maybe_v

        scalar_rep = self.scalar_out(merged)

        if not self.vector_output_dim:
            # bypass updating vector features using scalar information
            return self.scalar_nonlinearity(scalar_rep)
        elif self.vector_output_dim and not self.vector_input_dim:
            # instantiate vector features that are learnable in proceeding GCP layers
            vector_rep = self.create_zero_vector(scalar_rep)
        else:
            # update vector features using either row-wise scalar gating with complete local frames or row-wise self-scalar gating
            vector_rep = self.vectorize(scalar_rep, vector_hidden_rep)

        scalar_rep = self.scalar_nonlinearity(scalar_rep)
        return ScalarVector(scalar_rep, vector_rep)


#%%

if __name__ == "__main__":
    raise NotImplementedError("refactor error unsolved")
    from topotein.models.graph_encoders.layers.tcpnet.embedding import TCPEmbedding
    from topotein.models.graph_encoders.layers.tcpnet.interaction import TCPInteractions
    from proteinworkshop.models.utils import localize
    from omegaconf import OmegaConf
    from proteinworkshop.constants import PROJECT_PATH
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch: ProteinBatch = torch.load(f'{PROJECT_PATH}/../test/data/sample_batch/sample_batch_for_tcp.pt', weights_only=False).to(device)
    batch.mask = torch.randn(batch.x.shape[0], device=device) > -.3
    print(batch)


    batch.f_ij = localize(batch.pos, batch.edge_index)
    batch.f_ij_cell = localize(batch.pos, batch.sse_cell_index_simple)
    batch.node_to_sse_mapping = batch.N2_0.T.coalesce()

    pos_centroid, X_c = centralize(batch.pos, batch.batch)
    ori_pos = batch.pos
    batch.pos = X_c
    batch.h, batch.chi, batch.e, batch.xi, batch.c, batch.rho = (
        batch.x,
        batch.x_vector_attr,
        batch.edge_attr,
        batch.edge_vector_attr,
        batch.sse_attr,
        batch.sse_vector_attr
    )

    cfg = OmegaConf.create({
        "r_max": 10.0,
        "num_rbf": 8,
        "scalar_gate": 0,
        "vector_gate": True,
        "enable_e3_equivariance": False,
        "nonlinearities": ["silu", "silu"],
        "pre_norm": False,
        "use_scalar_message_attention": True,
        "default_bottleneck": 4,
        "mp_cfg": {
            "edge_encoder": False,
            "edge_gate": False,
            "num_message_layers": 4,
            "message_residual": 0,
            "message_ff_multiplier": 1,
            "self_message": True,
        },
        "scalar_nonlinearity": "silu",
        "vector_nonlinearity": "silu",
        "use_gcp_norm": True,
        "use_gcp_dropout": True,
        "num_feedforward_layers": 2
    })


    tcp_embedding = TCPEmbedding(
        node_input_dims=[49, 2],
        edge_input_dims=[9, 1],
        sse_input_dims=[15, 8],
        node_hidden_dims=[128, 16],
        edge_hidden_dims=[32, 4],
        sse_hidden_dims=[128, 16],
        cfg=cfg
    )
    (h, chi), (e, xi), (c, rho) = tcp_embedding(batch)
    print(f'h: {h.shape}, chi: {chi.shape}')
    print(f'e: {e.shape}, xi: {xi.shape}')
    print(f'c: {c.shape}, rho: {rho.shape}')
    # msg_passing = TCPMessagePassing(
    #     ScalarVector(128 + 32, 16 + 4),
    #     ScalarVector(128, 16),
    #     ScalarVector(32, 4),
    #     cfg=cfg,
    #     mp_cfg=cfg.mp_cfg,
    #     reduce_function="sum",
    #     use_scalar_message_attention=cfg.use_scalar_message_attention,
    # )
    # print(msg_passing)
    #
    # msg = msg_passing(
    #     node_rep=ScalarVector(h, chi),
    #     edge_rep=ScalarVector(e, xi),
    #     cell_rep=ScalarVector(c, rho),
    #     edge_index=batch.edge_index,
    #     frames=batch.f_ij,
    #     cell_frames=batch.f_ij_cell,
    #     node_to_sse_mapping=batch.node_to_sse_mapping,
    #     node_mask = batch.mask,
    # )
    #
    # print(msg.scalar.shape)
    # print(msg.vector.shape)

    interactions = TCPInteractions(
        ScalarVector(128, 16),
        ScalarVector(32, 4),
        ScalarVector(128, 16),
        cfg=cfg,
        layer_cfg=cfg,
        dropout=0.0,
        nonlinearities=cfg.nonlinearities,
    )

    node_s_v, pos = interactions(
        node_rep=ScalarVector(h, chi),
        edge_rep=ScalarVector(e, xi),
        cell_rep=ScalarVector(c, rho),
        frames=batch.f_ij,
        cell_frames=batch.f_ij_cell,
        edge_index=batch.edge_index,
        node_mask = batch.mask,
        node_pos=batch.pos,
        node_to_sse_mapping=batch.node_to_sse_mapping,
    )
    print('>>result')
    print(node_s_v.scalar.shape)
    print(node_s_v.vector.shape)
    print(pos.shape)
