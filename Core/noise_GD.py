"""Gradient descent method for noise estimation in Hapke Model.
The objectif function is denoted by J"""
import logging
import time

import coloredlogs
import numpy as np
import numba as nb
import scipy.optimize


# TODO: optimiser J et dJ pour éviter la répition des calculs

def J(b, Ydiff):
    """Compute distance between observations and translated Im(F)

    :param b: Offset shape (D,)
    :param Ydiff: Difference beetween Obs and F samples. shape (N,Nobs,D)
    :return: scalar
    """
    dist = np.square(Ydiff + b[None, None, :]).sum(axis=2)
    # ni = np.argmax(dist,axis=0)
    mindist = np.min(dist, axis=0)
    return np.mean(mindist)


def dJ(b, Ydiff):
    """Compute gradient of J, with respect to b

    :param b: Offset shape (D,)
    :param Ydiff: Difference beetween Obs and F samples. shape (N,Nobs,D)
    :return: shape (D,)
    """
    diff = Ydiff + b[None, None, :]  # N,Nobs,D
    dist = np.square(diff).sum(axis=2)
    ni = np.argmax(dist, axis=0)
    _, Nobs, _ = Ydiff.shape
    choix = diff[ni, np.arange(Nobs)]
    return (2 / Nobs) * choix.sum(axis=0)


def gradient_descent(Ytrain, Yobs):
    Ydiff = Ytrain[:, None, :] - Yobs[None, :, :]

    Nobs, D = Yobs.shape
    b = np.zeros(D)
    converged = False
    while not converged:
        direction = - dJ(b, Ydiff)
        ti = time.time()
        alpha, *r = scipy.optimize.line_search(J, dJ, b, direction, gfk=-direction, args=(Ydiff,))
        logging.debug(f"Line search performed in {time.time() - ti:.3f} s.")
        new_b = b + alpha * direction
        diff = J(b, Ydiff) - J(new_b, Ydiff)
        logging.debug(f"J progress : {diff:.8f}")
        b = new_b
        if diff < 0.00001:
            converged = True
    return b


def main():
    D = 10
    Ytrain = np.random.random_sample((100000, D)) + 2
    Yobs = np.random.random_sample((200, D))
    b = np.random.random_sample(D)
    print(gradient_descent(Ytrain, Yobs))


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()

    # res = scipy.optimize.minimize(_J,np.zeros(D),(Ytrain,Yobs),"BFGS",jac=_dJ)
    # print(res)
