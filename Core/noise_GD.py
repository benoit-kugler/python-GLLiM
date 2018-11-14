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


# ---------------- Linear case ---------------- #

def _proj_ImF_lin(u, F):
    z2 = (u.reshape((-1, 1)) * F).sum(axis=0) / np.square(F).sum(axis=0)
    return (z2.reshape((1, -1)) * F).sum(axis=1)


def _dist_y_ImFb_lin(yobs, F, b):
    """F is orthogonal : shape D,L"""
    u = yobs - b
    proj = _proj_ImF_lin(u, F)
    return np.square(u).sum() - np.square(proj).sum()


def J_lin(b, F, Yobs):
    Nobs, _ = Yobs.shape
    s = 0
    for i in nb.prange(Nobs):
        dist_min = _dist_y_ImFb_lin(Yobs[i], F, b)
        s = s + dist_min
    return s / Nobs


def _ddist_lin(yobs, F, b):
    u = b - yobs
    proj = _proj_ImF_lin(u, F)
    return u - proj


def dJ_lin(b, F, Yobs):
    """Half of the real gradient"""
    Nobs, D = Yobs.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        s += _ddist_lin(Yobs[i], F, b)
    return s / Nobs


@nb.njit(nogil=True, fastmath=True, parallel=True)
def sigma_estimator_diag_lin(b, F, Yobs):
    Nobs, D = Yobs.shape
    s = np.zeros(D)
    for i in nb.prange(Nobs):
        u = b - Yobs[i]
        z2 = (u.reshape((-1, 1)) * F).sum(axis=0) / np.square(F).sum(axis=0)
        proj = (z2.reshape((1, -1)) * F).sum(axis=1)

        s += (u - proj) ** 2
    return s / Nobs


# ------------------ General case ------------------ #

@nb.njit(nogil=True, fastmath=True, parallel=True)
def J(b, Ytrain, Yobs):
    N, _ = Ytrain.shape
    Nobs, _ = Yobs.shape
    s = 0
    for i in nb.prange(Nobs):
        # dist_min = _dist_y_ImFb(Yobs[i], Ytrain, N, b)
        yobs = Yobs[i]
        dist_min = np.inf
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
        yobs = Yobs[i]
        dist_min = np.inf
        n_min = 0
        for n in range(N):
            diff = Ytrain[n] - yobs + b
            dist = np.sum(diff ** 2)
            if dist < dist_min:
                dist_min = dist
                n_min = n

        s += Ytrain[n_min] - yobs + b
    return s / Nobs


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


class GradientDescent:
    """Base class for noise estimation with Gradient Descent
    Actual computation are made by JIT compiled helpers
    """

    def __init__(self, Yobs, cov_type, resample):
        self.Yobs = Yobs
        self.cov_type = cov_type
        self.resample = resample

    def _get_starting_logging(self, Farg):
        return f"""
        Nobs = {len(self.Yobs)}
        tol. = {TOL}
        Covariance constraint : {self.cov_type}
        """

    def _get_sigma_estimator(self):
        return sigma_estimator_diag if self.cov_type == "diag" else sigma_estimator_full

    def _get_F_arg(self):
        """Should return F related arg needed by sigma_estimator, J, dJ"""
        raise NotImplementedError

    def _get_J_dJ(self):
        """Should return functions to compute J and dJ"""
        raise NotImplementedError

    def run(self):
        Yobs = self.Yobs
        Nobs, D = Yobs.shape
        current_noise_mean = INIT_MEAN_NOISE * np.ones(D)
        Farg = self._get_F_arg()

        log = "Starting noise mean estimation with gradient descent" + self._get_starting_logging(Farg)
        logging.info(log)

        sigma_estimator = self._get_sigma_estimator()

        sigma = sigma_estimator(current_noise_mean, Farg, Yobs)

        Jfunc, dJfunc = self._get_J_dJ()

        Jinit = Jfunc(current_noise_mean, Farg, Yobs)

        history = [(current_noise_mean.tolist(), sigma.tolist(), Jinit)]
        c_iter = 0
        while c_iter < maxIter:
            direction = - dJfunc(current_noise_mean, Farg, Yobs)
            ti = time.time()
            alpha, *r = scipy.optimize.line_search(Jfunc, dJfunc, current_noise_mean, direction, gfk=-direction,
                                                   args=(Farg, Yobs,))
            logging.debug(f"Line search performed in {time.time() - ti:.3f} s.")
            if alpha is None:
                break
            new_b = current_noise_mean + alpha * direction
            current_J = Jfunc(current_noise_mean, Farg, Yobs)
            diff = (current_J - Jfunc(new_b, Farg, Yobs)) / Jinit
            logging.debug(f"J relative progress : {diff:.8f}")
            current_noise_mean = new_b
            sigma = sigma_estimator(current_noise_mean, Farg, Yobs)
            log_sigma = sigma if self.cov_type == "diag" else np.diag(sigma)
            logging.info(f"Iteration {c_iter}")
            if verbosity >= 2:
                logging.info(f"""
        New estimated OFFSET : {current_noise_mean}
        New estimated COVARIANCE : {log_sigma}""")

            history.append((current_noise_mean.tolist(), sigma.tolist(), current_J))
            if diff < TOL:
                break

            if self.resample:  # new generation
                Farg = self._get_F_arg()

            c_iter += 1

        return history


class GradientDescentGeneral(GradientDescent):

    def __init__(self, Yobs, cov_type, Ytrain):
        super().__init__(Yobs, cov_type, True)
        if cov_type == "full":
            logging.warning(f"Full covariance not supported for linear case. "
                            f"Diagonal constraint used")
        self.Ytrain = Ytrain

    def _get_starting_logging(self, Farg):
        s = super()._get_starting_logging(Farg)
        N, D = Farg.shape
        s += f"Ntrain = {N:.1e}"
        return s

    def _get_sigma_estimator(self):
        return sigma_estimator_diag

    def _get_J_dJ(self):
        return J, dJ

    def _get_F_arg(self):
        return self.Ytrain()


class GradientDescentLinear(GradientDescent):

    def __init__(self, Yobs, cov_type, Fmatrix):
        super().__init__(Yobs, cov_type, False)
        self.Fmatrix = Fmatrix

    def _get_starting_logging(self, Farg):
        s = super()._get_starting_logging(Farg)
        s = " (Linear case)" + s
        return s

    def _get_J_dJ(self):
        return J_lin, dJ_lin

    def _get_F_arg(self):
        return self.Fmatrix


def fit(Yobs, cont: context.abstractHapkeModel, cov_type="diag", assume_linear=False):
    if assume_linear:
        fitter = GradientDescentLinear(Yobs, "diag", cont.F_matrix)
    else:
        def Ytrain_gen():
            return cont.get_data_training(Ntrain)[1]

        fitter = GradientDescentGeneral(Yobs, cov_type, Ytrain_gen)

    return fitter.run()



## --------------------- maintenance purpose --------------------- ##
def main():
    c = context.LinearFunction()
    D = 10
    Ytrain = np.random.random_sample((100000, D)) + 2
    Yobs = np.random.random_sample((1000, D))
    b = np.random.random_sample(D)

    # _, Yobs = c.get_data_training(1000)

    # print(J_lin(0, c.F_matrix, Yobs))

    print(fit(Yobs, c, asssume_linear=True))


if __name__ == '__main__':
    coloredlogs.install(level=logging.DEBUG, fmt="%(asctime)s : %(levelname)s : %(message)s",
                        datefmt="%H:%M:%S")
    main()
    # compare_jit()
    # res = scipy.optimize.minimize(_J,np.zeros(D),(Ytrain,Yobs),"BFGS",jac=_dJ)
    # print(res)
