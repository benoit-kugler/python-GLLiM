import autograd.numpy as np
from autograd.scipy.misc import logsumexp
from pymanopt.manifolds import Product, Euclidean, PositiveDefinite
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent

from Core.gllim import jGLLiM


class RiemannianjGLLiM(jGLLiM):

    def get_cost_function(self, T, Y):
        samples = np.concatenate((T, Y), axis=1)
        N, _ = samples.shape

        # (2) Define cost function
        # The parameters must be contained in a list theta.
        def cost(theta):
            # Unpack parameters
            nu = np.concatenate([theta[1], [0]], axis=0)

            S = theta[0]
            logdetS = np.expand_dims(np.linalg.slogdet(S)[1], 1)
            y = np.concatenate([samples.T, np.ones((1, N))], axis=0)

            # Calculate log_q
            y = np.expand_dims(y, 0)

            # 'Probability' of y belonging to each cluster
            log_q = -0.5 * (np.sum(y * np.linalg.solve(S, y), axis=1) + logdetS)

            alpha = np.exp(nu)
            alpha = alpha / np.sum(alpha)
            alpha = np.expand_dims(alpha, 1)

            loglikvec = logsumexp(np.log(alpha) + log_q, axis=0)
            return -np.sum(loglikvec)

        return cost

    @property
    def current_ll(self):
        return 10

    def start_track(self):
        raise NotImplementedError("Manifold optimization can't track thetas")

    def fit(self, T, Y, init, maxIter=100):
        self.init_fit(T, Y, None)

        D = self.D + self.L
        K = self.K

        # (1) Instantiate the manifold
        manifold = Product([PositiveDefinite(D + 1, k=K), Euclidean(K - 1)])

        cost = self.get_cost_function(T, Y)

        problem = Problem(manifold=manifold, cost=cost, verbosity=1)

        # (3) Instantiate a Pymanopt solver
        solver = SteepestDescent(maxiter=3 * maxIter)

        # let Pymanopt do the rest
        Xopt = solver.solve(problem)
        self.Xopt_to_theta(Xopt)

    def Xopt_to_theta(self, Xopt):
        pi = np.exp(np.concatenate([Xopt[1], [0]], axis=0))
        pi = pi / np.sum(pi)
        D = self.D + self.L
        mulist, Sigmalist = [], []
        for psik in Xopt[0]:
            muk = psik[0:D, D:D + 1]
            Sigmak = psik[:D, :D] - muk.dot(muk.T)
            muk = muk[:, 0]
            mulist.append(muk)
            Sigmalist.append(Sigmak)
        theta = self.GMM_to_GLLiM(pi, np.array(mulist), np.array(Sigmalist), self.L)
        self._init_from_dict(theta)
