"""Implements the algorithm defined in
Kullback-Leibler Approach to
Gaussian Mixture Reduction by ANDREW R. RUNNALLS"""

import numpy as np
import numba as nb


def merge_to_2components(weights, means, covs):
    """Entry point

    :param weights: shape N,K
    :param means: shape N,K,L
    :param covs: shape K,L,L
    :return: tuple ( N,2 ; N,2,L ; N,2,L,L)
    """
    pass


@nb.njit
def merge_2_gaussians(w1, w2, m1, m2, C1, C2):
    """Return weight, mean and cov of merged gaussians eq (2-3-4)"""
    w = w1 + w2
    w1s = w1 / w
    w2s = w2 / w
    m = w1s * m1 + w2s * m2
    diff = m1 - m2
    C = w1s * C1 + w2s * C2 + w1s * w2s * diff.reshape((-1, 1)).dot(diff.reshape((1, -1)))
    return w, m, C


@nb.njit
def B(w1, w2, m1, m2, C1, C2):
    """Compute upper bound for discrimination eq (21)

    :param w1: weight 1
    :param w2: weight 2
    :param m1: mean 1
    :param m2: mean 2
    :param C1: cov 1
    :param C2: cov 2
    """
    w, m, C = merge_2_gaussians(w1, w2, m1, m2, C1, C2)
    ld0, ld1, ld2 = np.linalg.slogdet(C), np.linalg.slogdet(C1), np.linalg.slogdet(C2)
    return 0.5 * ((w1 + w2) * ld0 - w1 * ld1 - w2 * ld2)
