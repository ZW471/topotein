"""Implementation of a transform to remove residues with missing CA atoms."""
import mini3di
import numpy as np
import torch
from graphein.protein import ATOM_NUMBERING_MODIFIED
from torch_geometric import transforms as T


class Attach3Di(T.BaseTransform):
    """Removes residues with missing CA atoms from a protein structure."""

    def __init__(self):
        self.encoder = mini3di.Encoder()

    def __call__(self, data):
        coords = data.coords.numpy().copy()
        coords[coords == 1e-5] = float('nan')

        out = self.encoder.encode_atoms(
            ca=coords[:, ATOM_NUMBERING_MODIFIED["CA"]],
            cb=coords[:, ATOM_NUMBERING_MODIFIED["CB"]],
            n=coords[:, ATOM_NUMBERING_MODIFIED["N"]],
            c=coords[:, ATOM_NUMBERING_MODIFIED["C"]]
        )
        threeDi_type = out.data
        threeDi_type[out.mask] = out.fill_value
        data.threeDi_type = torch.tensor(threeDi_type, dtype=torch.int64).to(data.coords.device)
        return data



