from torch import clamp


def eigenval_features(evals, eps=1e-5):
    evals = clamp(evals, min=eps)
    linearity = (evals[:, 0] - evals[:, 1]) / evals[:, 0]
    planarity = (evals[:, 1] - evals[:, 2]) / evals[:, 1]
    scattering = evals[:, 2] / evals[:, 0]
    omnivariance = (evals[:, 0] * evals[:, 1] * evals[:, 2]) ** (1 / 3)
    anisotropy = (evals[:, 0] - evals[:, 2]) / evals[:, 0]
    return anisotropy, linearity, omnivariance, planarity, scattering
