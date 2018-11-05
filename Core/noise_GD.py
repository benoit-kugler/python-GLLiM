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

maxIter = 1000

verbosity = 1


# Ydiff = Ytrain[:, None, :] - Yobs[None, :, :]

@nb.njit(nogil=True, fastmath=True, parallel=True)
def J(b, Ytrain, Yobs):
    N, _ = Ytrain.shape
    Nobs, _ = Yobs.shape
    s = 0
    for i in nb.prange(Nobs):
        dist_min = np.inf
        yobs = Yobs[i]
        for n in range(N):
            dist = np.sum((Ytrain[n] - yobs + b) ** 2)
            if dist < dist_min:
                dist_min = dist
        s = s + dist_min
    return s / Nobs


@nb.njit(nogil=True, fastmath=True, parallel=True)
def dJ(b, Ytrain, Yobs):
    """Half of the real gradient"""
    N, D = Ytrain.shape
    Nobs, _ = Yobs.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        dist_min = np.inf
        yobs = Yobs[i]
        n_min = 0
        for n in range(N):
            diff = Ytrain[n] - yobs + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n
        s += Ytrain[n_min] - yobs
    return 1 * (s / Nobs + b)


@nb.njit(nogil=True, fastmath=True, parallel=False)
def sigma_estimator_full(b, Ytrain, Yobs):
    N, D = Ytrain.shape
    Nobs, _ = Yobs.shape
    s = np.zeros((D, D))
    for i in range(Nobs):
        dist_min = np.inf
        yobs = Yobs[i]
        n_min = 0
        for n in range(N):
            diff = Ytrain[n] - yobs + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        u = Ytrain[n_min] - yobs + b
        v = u.reshape((-1, 1)).dot(u.reshape((1, -1)))
        s += v
    return s / Nobs


@nb.njit(nogil=True, fastmath=True, parallel=True)
def sigma_estimator_diag(b, Ytrain, Yobs):
    N, D = Ytrain.shape
    Nobs, _ = Yobs.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        dist_min = np.inf
        yobs = Yobs[i]
        n_min = 0
        for n in range(N):
            diff = Ytrain[n] - yobs + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        s += (Ytrain[n_min] - yobs + b) ** 2
    return s / Nobs


def gradient_descent(Ytrain, Yobs, cov_type="full"):
    """Performs gradient descent

    :param Ytrain: array or callable giving array : shape N,D
    :param Yobs: shape Nobs,D
    :param cov_type:
    :return: history of mean, covariance
    """
    if callable(Ytrain):
        _Ytrain = Ytrain()
    else:
        _Ytrain = Ytrain
    Yobs = np.asarray(Yobs, dtype=float)

    N, D = _Ytrain.shape
    Nobs, _ = Yobs.shape
    current_noise_mean = INIT_MEAN_NOISE * np.ones(D)
    logging.info(f"""
Starting noise mean estimation with gradient descent.
        Ntrain = {N:.1e}, Nobs = {Nobs}
        tol. = {TOL}
        Covariance constraint : {cov_type}""")
    sigma_estimator = sigma_estimator_diag if cov_type == "diag" else sigma_estimator_full
    sigma = sigma_estimator(current_noise_mean, _Ytrain, Yobs)
    Jinit = J(current_noise_mean, _Ytrain, Yobs)
    history = [(current_noise_mean.tolist(), sigma.tolist(), Jinit)]
    c_iter = 0
    while c_iter < maxIter:
        direction = - dJ(current_noise_mean, _Ytrain, Yobs)
        ti = time.time()
        alpha, *r = scipy.optimize.line_search(J, dJ, current_noise_mean, direction, gfk=-direction,
                                               args=(_Ytrain, Yobs,))
        # alpha, *r = scipy.optimize.linesearch.line_search_armijo(J, b, direction, -direction, None, args=(_Ytrain,Yobs,))
        logging.debug(f"Line search performed in {time.time() - ti:.3f} s.")
        if alpha is None:
            break
        new_b = current_noise_mean + alpha * direction
        current_J = J(current_noise_mean, _Ytrain, Yobs)
        diff = (current_J - J(new_b, _Ytrain, Yobs)) / Jinit
        logging.debug(f"J relative progress : {diff:.8f}")
        current_noise_mean = new_b
        sigma = sigma_estimator(current_noise_mean, _Ytrain, Yobs)
        log_sigma = sigma if cov_type == "diag" else np.diag(sigma)
        logging.info(f"Iteration {c_iter}")
        if verbosity >= 2:
            logging.info(f"""
    New estimated OFFSET : {current_noise_mean}
    New estimated COVARIANCE : {log_sigma}""")

        history.append((current_noise_mean.tolist(), sigma.tolist(), current_J))
        if diff < TOL:
            break

        if callable(Ytrain):  # new generation
            _Ytrain = Ytrain()

        c_iter += 1

    return history


def fit(Yobs, cont: context.abstractHapkeModel, cov_type="diag"):
    # Xtrain, Ytrain = cont.get_data_training(Ntrain)

    def Ytrain_gen():
        return cont.get_data_training(Ntrain)[1]

    return gradient_descent(Ytrain_gen, Yobs, cov_type=cov_type)



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

    # j_ = dJ(b, Ytrain, Yobs)  # compilation time
    # d_j_ = dJ3(b, Ytrain, Yobs)
    a = sigma_estimator_diag(b, Ytrain, Yobs)
    print("Jit compile done.")
    # j = oldJ(b, Ytrain, Yobs)
    # j_ = oldJ(b, Ytrain, Yobs)

    ti = time.time()
    # d_j = dJ(b, Ytrain, Yobs)
    S1 = sigma_estimator_diag(b, Ytrain, Yobs)
    print(S1)
    print(f"jitted manual {time.time() - ti}")

    ti = time.time()
    # d_j_ = dJ3(b, Ytrain, Yobs)
    print(f"jitted vector {time.time() - ti}")

    # assert np.allclose(j, j_)
    # assert np.allclose(d_j, d_j_)


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    # main()
    compare_jit()
    # res = scipy.optimize.minimize(_J,np.zeros(D),(Ytrain,Yobs),"BFGS",jac=_dJ)
    # print(res)
