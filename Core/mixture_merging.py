"""Implements the algorithm defined in
Kullback-Leibler Approach to
Gaussian Mixture Reduction by ANDREW R. RUNNALLS"""
import logging

import coloredlogs
import numpy as np
import numba as nb
from matplotlib import pyplot

from Core import probas_helper


def merge_to_2components(weights, means, covs):
    """Entry point

    :param weights: shape N,K
    :param means: shape N,K,L
    :param covs: shape K,L,L
    :return: tuple ( N,2 ; N,2,L ; N,2,L,L)
    """
    N, _ = weights.shape
    _merge(weights, means, np.array([covs] * N))


@nb.njit(cache=True, fastmath=True, nogil=True)
def merge_2_gaussians(w1, w2, m1, m2, C1, C2):
    """Return weight, mean and cov of merged gaussians eq (2-3-4)"""
    w = w1 + w2
    w1s = w1 / w
    w2s = w2 / w
    m = w1s * m1 + w2s * m2
    diff = m1 - m2
    C = w1s * C1 + w2s * C2 + w1s * w2s * diff.reshape((-1, 1)).dot(diff.reshape((1, -1)))
    return w, m, C


@nb.njit(cache=True, fastmath=True, nogil=True)
def B(w1, m1, C1, w2, m2, C2):
    """Compute upper bound for discrimination eq (21).
    Returns discrimination value and merged gaussian"""
    w, m, C = merge_2_gaussians(w1, w2, m1, m2, C1, C2)
    (ld0, _), (ld1, _), (ld2, _) = np.linalg.slogdet(C), np.linalg.slogdet(C1), np.linalg.slogdet(C2)
    d = 0.5 * ((w1 + w2) * ld0 - w1 * ld1 - w2 * ld2)
    return d, w, m, C


@nb.njit(cache=True, fastmath=True, nogil=True)
def find_pair_to_merge(weights, means, covs):
    K, L = means.shape
    best_disc, best_merged_w, best_merged_m, best_merged_cov = np.inf, 0, np.zeros(L), np.zeros((L, L))
    best_i, best_j = None, None
    for i in range(K):
        wi, mi, Ci = weights[i], means[i], covs[i]
        for j in range(i + 1, K):
            wj, mj, Cj = weights[j], means[j], covs[j]
            d, merged_w, merged_m, merged_cov = B(wi, mi, Ci, wj, mj, Cj)
            if d < best_disc:
                best_disc = d
                best_merged_w = merged_w
                best_merged_m = merged_m
                best_merged_cov = merged_cov
                best_i, best_j = i, j
    return best_i, best_j, best_merged_w, best_merged_m, best_merged_cov


@nb.njit(fastmath=True, parallel=True)
def _K_step(current_ws, current_ms, current_covs):
    N, K, L = current_ms.shape
    new_weights, new_means, new_covs = np.zeros((N, K - 1)), np.zeros((N, K - 1, L)), np.zeros((N, K - 1, L, L))
    for n in nb.prange(N):
        weights, means, covs = current_ws[n], current_ms[n], current_covs[n]
        i, j, merged_w, merged_m, merged_cov = find_pair_to_merge(weights, means, covs)
        keep_w = [weights[k] for k in range(K) if not (k == i or k == j)]
        keep_m = [means[k] for k in range(K) if not (k == i or k == j)]
        keep_cov = [covs[k] for k in range(K) if not (k == i or k == j)]
        for k in range(K - 2):
            new_weights[n][k] = keep_w[k]
            new_means[n][k] = keep_m[k]
            new_covs[n][k] = keep_cov[k]
        new_weights[n][K - 2] = merged_w
        new_means[n][K - 2] = merged_m
        new_covs[n][K - 2] = merged_cov
    return new_weights, new_means, new_covs


# def bulk_merge(weights, means, covs, threshold):
#     """Merge the weakest components, in term of weights.
#     If threshold is int, only keeps threshold components.
#     If threshold if float, only keeps components  with weights >= threshold"""
#     if type(threshold) is int:
#         sorted(zip(weights, means, covs), key=lambda t: t[0])[threshold:]


def _merge(weightss, meanss, covss, with_plot=False):
    N, K, L = meanss.shape
    current_ws, current_ms, current_covs = weightss, meanss, covss
    while K > 2:
        current_ws, current_ms, current_covs = _K_step(current_ws, current_ms, current_covs)
        K -= 1
        logging.debug("Current mixture size : {}".format(K))
        if with_plot:
            _show_density(current_ws, current_ms, current_covs)
    return current_ws, current_ms, current_covs


# ---------------------- DEBUG Tools ---------------------- #

def _show_density(current_ws, current_ms, current_covs):
    pyplot.clf()
    xlim = ylim = (0, 1)
    RESOLUTION = 200
    x, y = np.meshgrid(np.linspace(*xlim, RESOLUTION, dtype=float),
                       np.linspace(*ylim, RESOLUTION, dtype=float))
    variable = np.array([x.flatten(), y.flatten()]).T
    print("Comuting of density...")
    density = probas_helper.densite_melange(variable, current_ws[0], current_ms[0], current_covs[0])
    z = density.reshape((RESOLUTION, RESOLUTION))
    print("Done.")
    axe = pyplot.gca()
    axe.pcolormesh(x, y, z, cmap="Greens")
    axe.scatter(*current_ms[0].T)
    pyplot.xlim(xlim)
    pyplot.ylim(ylim)
    pyplot.savefig("__tmp_evo.png")


def _test():
    L = 2
    T = np.tril(np.ones((L, L))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD

    N = 10000
    K = 60
    covs = np.array([cov] * K)
    wks = np.arange(N * K, dtype=float).reshape((N, K)) + 1
    wks /= wks.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((N, K, L))

    merge_to_2components(wks, meanss, covs)


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(module)s %(name)s %(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    _test()
