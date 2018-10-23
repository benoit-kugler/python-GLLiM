"""Gradient descent method for noise estimation in Hapke Model.
The objectif function is denoted by J"""
import logging
import time

import coloredlogs
import numpy as np
import numba as nb
import scipy.optimize
import scipy.optimize.linesearch

from tools import context


Ntrain = 100000  # number of samples to estimate d(y, Im F)

TOL = 0.000001  # diff in distance between two iter to stop

INIT_MEAN_NOISE = 0  # initial noise offset


@nb.njit(nogil=True, fastmath=True, parallel=True)
def J(b, Ydiff):
    N, Nobs, D = Ydiff.shape
    s = 0
    for i in nb.prange(Nobs):
        dist_min = np.inf
        for n in range(N):
            dist = np.sum((Ydiff[n, i] + b) ** 2)
            if dist < dist_min:
                dist_min = dist
        s += dist_min
    return s / Nobs


@nb.njit(nogil=True, fastmath=True, parallel=True)
def dJ(b, Ydiff):
    N, Nobs, D = Ydiff.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        dist_min = np.inf
        n_min = 0
        for n in range(N):
            diff = Ydiff[n, i] + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        s += Ydiff[n_min, i]
    return 2 * (s / Nobs + b)


@nb.njit(nogil=True, fastmath=True, parallel=True)
def sigma_estimator_full(b, Ydiff):
    N, Nobs, D = Ydiff.shape
    s = np.zeros((D, D))
    for i in nb.prange(Nobs):
        dist_min = np.inf
        n_min = 0
        for n in range(N):
            diff = Ydiff[n, i] + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        u = Ydiff[n_min, i]
        s += u.T.dot(u)
    return s / Nobs


@nb.njit(nogil=True, fastmath=True, parallel=True)
def sigma_estimator_diag(b, Ydiff):
    N, Nobs, D = Ydiff.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        dist_min = np.inf
        n_min = 0
        for n in range(N):
            diff = Ydiff[n, i] + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        s += (Ydiff[n_min, i] + b) ** 2
    return s / Nobs


def gradient_descent(Ytrain, Yobs, cov_type="full"):
    Ydiff = Ytrain[:, None, :] - Yobs[None, :, :]

    N, Nobs, D = Ydiff.shape
    current_noise_mean = INIT_MEAN_NOISE * np.ones(D)
    logging.info(f"""Starting noise mean estimation with gradient descent.
                    Ntrain = {N}, Nobs = {Nobs}
                    tol. = {TOL}
                    Covariance constraint : {cov_type}""")
    sigma_estimator = sigma_estimator_diag if cov_type == "diag" else sigma_estimator_full
    history = []
    Jinit = J(current_noise_mean, Ydiff)
    while True:
        direction = - dJ(current_noise_mean, Ydiff)
        ti = time.time()
        alpha, *r = scipy.optimize.line_search(J, dJ, current_noise_mean, direction, gfk=-direction, args=(Ydiff,))
        # alpha, *r = scipy.optimize.linesearch.line_search_armijo(J, b, direction, -direction, None, args=(Ydiff,))
        logging.debug(f"Line search performed in {time.time() - ti:.3f} s.")
        if alpha is None:
            break
        new_b = current_noise_mean + alpha * direction
        diff = (J(current_noise_mean, Ydiff) - J(new_b, Ydiff)) / Jinit
        logging.debug(f"J progress : {diff:.8f}")
        current_noise_mean = new_b
        sigma = sigma_estimator(current_noise_mean, Ydiff)
        log_sigma = sigma if cov_type == "diag" else np.diag(sigma)
        logging.info(f""" 
        New estimated OFFSET : {current_noise_mean}
        New estimated COVARIANCE : {log_sigma}""")

        history.append((current_noise_mean.tolist(), sigma.tolist()))
        if diff < TOL:
            break
    return history


def fit(Yobs, cont: context.abstractHapkeModel, cov_type="diag"):
    Xtrain, Ytrain = cont.get_data_training(Ntrain)
    return gradient_descent(Ytrain, Yobs, cov_type=cov_type)


def verifie_J(b, Yobs, Ytrain):
    Ydiff = Ytrain[:, None, :] - Yobs[None, :, :]
    return J(b, Ydiff)


## --------------------- maintenance purpose --------------------- ##
def oldJ(b, Ydiff):
    """Compute distance between observations and translated Im(F)

    :param b: Offset shape (D,)
    :param Ydiff: Difference beetween Obs and F samples. shape (N,Nobs,D)
    :return: scalar
    """
    dist = np.square(Ydiff + b[None, None, :]).sum(axis=2)
    mindist = np.min(dist, axis=0)
    return np.mean(mindist)


def olddJ(b, Ydiff):
    """Compute gradient of J, with respect to b

    :param b: Offset shape (D,)
    :param Ydiff: Difference beetween Obs and F samples. shape (N,Nobs,D)
    :return: shape (D,)
    """
    diff = Ydiff + b[None, None, :]  # N,Nobs,D
    dist = np.square(diff).sum(axis=2)
    ni = np.argmin(dist, axis=0)
    _, Nobs, _ = Ydiff.shape
    choix = diff[ni, np.arange(Nobs)]
    return (2 / Nobs) * choix.sum(axis=0)


def main():
    D = 10
    Ytrain = np.random.random_sample((100000, D)) + 2
    Yobs = np.random.random_sample((1000, D))
    b = np.random.random_sample(D)
    print(gradient_descent(Ytrain, Yobs))


def compare_jit():
    D = 10
    Ytrain = np.random.random_sample((100000, D)) + 2
    Yobs = np.random.random_sample((200, D))
    b = np.random.random_sample(D)
    Ydiff = Ytrain[:, None, :] - Yobs[None, :, :]
    j_ = J(b, Ydiff)  # compilation time
    d_j_ = dJ(b, Ydiff)

    j = oldJ(b, Ydiff)
    j_ = oldJ(b, Ydiff)

    ti = time.time()
    d_j = dJ(b, Ydiff)
    print(f"basic {time.time() - ti}")

    ti = time.time()
    d_j_ = olddJ(b, Ydiff)
    print(f"jitted {time.time() - ti}")

    assert np.allclose(j, j_)
    assert np.allclose(d_j, d_j_)


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
    # compare_jit()
    # res = scipy.optimize.minimize(_J,np.zeros(D),(Ytrain,Yobs),"BFGS",jac=_dJ)
    # print(res)
