from typing import Literal

ScalarSSEFeature = Literal[
    "sse_one_hot",
    "sse_size",
    "sse_vector_norms",
    "sse_variance_wrt_localized_frame",
    "node_features"
]
VectorSSEFeature = Literal["sse_vectors"]
ScalarProteinFeature = Literal[
    "aa_freq",
    "aa_std",
    "sse_freq",
    "sse_std",
    "sse_size_mean",
    "sse_size_std",
    "pr_size"
]
VectorProteinFeature = Literal["aa_vector", "sse_vector"]
