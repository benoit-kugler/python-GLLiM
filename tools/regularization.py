from itertools import permutations

import numpy as np
from sklearn.cluster.k_means_ import _labels_inertia
from sklearn.utils.extmath import row_norms


class Attractor():

    def __init__(self,K,X,weights):
        self.K = K
        self.x_squared_norms = row_norms(X, squared=True)
        self.X = X
        self.weights = weights

    def _check_empty(self,rnk):
        return rnk.sum(axis=0).prod() == 0

    def _check_empty_labels(self,labels):
        rnk = np.array([labels == k for k in range(self.K)]).T
        return self._check_empty(rnk)

    def modifie_rnk(self,rnk):
        n = np.random.randint(0,len(self.X)-1)
        k = np.random.randint(0,self.K -1)
        l = np.zeros(self.K)
        old_l = rnk[n].copy()
        l[k] = 1
        rnk[n] = l
        if self._check_empty(rnk):
            rnk[n] = old_l
        return rnk

    # def main(self,maxIter=10):
    #     labels = np.random.randint(0, self.K, len(self.X))
    #     rnk = np.array([labels == k for k in range(self.K)]).T
    #     uk = self.compute_uk(rnk.T)
    #     old_intertie = self.inertie(uk)
    #     for i in range(maxIter):
    #         new_rnk = self.modifie_rnk(rnk)
    #         uk = self.compute_uk(new_rnk.T)
    #         new_intertie = self.inertie(uk)
    #         if new_intertie <= old_intertie:
    #             print(new_intertie)
    #             rnk = new_rnk
    #             old_intertie = new_intertie
    #     return rnk.argmax(axis=1) , old_intertie

    def main(self):
        best_intertie,best_labels ,best_uk = np.inf , None, None
        choix = np.array(np.meshgrid(*[list(range(self.K))] * len(self.X))).T.reshape(-1,len(self.X))
        print("Test des {} choix".format(len(choix)))
        for labels in choix:
            if self._check_empty_labels(labels):
                continue
            rnk = np.array([labels == k for k in range(self.K)]).T
            uk = self.compute_uk(rnk.T)
            inertie = self.inertie(uk)
            if inertie < best_intertie:
                best_intertie = inertie
                best_labels = labels
                best_uk = uk
        return best_labels, best_intertie , best_uk

    def compute_uk(self,rkn):
        ar = rkn * self.weights # shape K , N
        Xw = np.array([self.X * ck[:,None] for ck in ar]) # shape (K ,N ,L)
        return Xw.sum(axis=1) / ar.sum(axis=1)[:,None]

    def inertie(self,uk):
        _ , base_inertie = _labels_inertia(self.X, self.x_squared_norms, uk, precompute_distances=True)
        # s = 0
        # for u in uk:
        #     s += np.square(uk - u).sum(axis=1).sum()
        s = set([s for u in uk for s in np.square(uk - u).sum(axis=1)])
        if len(s) == 1:
            s = 0
        else:
            s.discard(0)
            s = min(s)
        base_inertie -= s/200
        return base_inertie


class WeightedKMeans:

    PENALIZATION_CLUSTER = 10

    def __init__(self,K):
        self.K = K

    def init_fit(self,X,weights,init):
        if init is None:
            self.ukList = X[0:self.K,:]
        else:
            self.ukList = init
        self.x_squared_norms = row_norms(X,squared=True)
        self.X = X
        self.weights = weights

    def _e_step(self):
        labels , _ = _labels_inertia(self.X, self.x_squared_norms, self.ukList, precompute_distances=True)
        self.rkn = np.array([labels == k for k in range(self.K)])

    def _m_step(self):
        ar = self.rkn * self.weights # shape K , N
        Xw = np.array([self.X * ck[:,None] for ck in ar]) # shape (K ,N ,L)
        self.ukList = Xw.sum(axis=1) / ar.sum(axis=1)[:,None]

    def fit_predict_score(self, X, weights, init, maxIter=1000):
        self.init_fit(X,weights,init)
        for i in range(maxIter):
            self._e_step()
            self._m_step()
        labels , base_inertia = _labels_inertia(self.X,self.x_squared_norms,self.ukList)
        # inertia =  base_inertia + len(self.X) * self.K * self.X.shape[1] * self.PENALIZATION_CLUSTER
        inertia = 2 * np.log(base_inertia) - np.log(len(self.X)) * self.K
        return labels, - inertia, self.ukList


# def best_K(X,weights,KMax=4):
#     res = []
#     for K in range(1,KMax + 1):
#         w = Attractor(K,X,weights)
#         labels , score , uk = w.main()
#         res.append((labels,score,uk))
#     res = sorted(res,key = lambda d : d[1])
#     best_label, best_score, best_uk = res[0]
#     return  best_uk ,best_label

def best_K(X, weights, KMax=4):
    res = []
    for K in range(1, KMax + 1):
        w = WeightedKMeans(K)
        labels, score, uklist = w.fit_predict_score(X, weights, None)
        print(f"K = {K} score = {score}")
        res.append((labels, score, uklist))
    res = sorted(res, key=lambda d: d[1], reverse=True)
    best_label, best_score, best_uklist = res[0]
    return best_label, best_uklist

def clustered_mean(Xs,weightss):
    """Returns the weighted center cluster by cluster"""
    return [ best_K(X,weights)[0] for X, weights in zip(Xs,weightss)]


### ------------- REGULARIZATION ------ ###

## Regularization by penalization of gradient over wave length
def step_by_step(Xs):
    """Step by step regularization. Xs shape (N,K,L) (K modes).
    Returns N-K-sized permutation (of order K)"""
    N, K, L = Xs.shape
    out = np.empty((N, K), dtype=int)  # choice for sample N and run K
    for k0 in range(K):
        X = np.copy(Xs[:, k0, :])  # init with k0 ieme mode. X is the current choice for this try.
        out[0, k0] = k0  # first choice is identity
        for n in range(N - 1):
            x = X[n,:]
            best_side_k = np.square(x - Xs[n+1,:,:]).sum(axis=1).argmin()
            out[n + 1, k0] = best_side_k  # write index
            X[n + 1, :] = Xs[n + 1, best_side_k]  #update X accordingly
    return out


X1 = np.array([0,1,0,1,0,1,0])[:,None]
X2 = 1 - X1
Xs = np.empty((7,2,1))
Xs[:,0] = X1
Xs[:,1] = X2

def sum_gradient(X):
    grad = X - np.roll(X,-1,axis=0)
    grad[-1] = np.zeros(X.shape[1:3])
    return np.sqrt(np.square(grad).sum(axis=2)).sum(axis=0)


# def global_regularization(Xs,maxiter=10000,tol=None):
#     """Global gradient regularization. Xs shape (N,K,L) (K modes)"""
#     N ,K , L = Xs.shape
#     alea_N = np.random.randint(0,N ,size=(maxiter,K))
#     alea_K = np.random.randint(0,K , size=(maxiter,K))
#     out = np.copy(Xs)
#     gradient_norm = sum_gradient(out)
#     for indexN, indexK in zip(alea_N,alea_K):
#         tmp = out[indexN, list(range(K))]
#         out[indexN, list(range(K))] = Xs[indexN, indexK] # essai
#         new_grad = sum_gradient(out)
#         is_not_best = new_grad > gradient_norm
#         #cancel change where it's not better
#         cancel_N = indexN[[*is_not_best]]
#         cancel_K = np.arange(K)[[*is_not_best]]
#         out[cancel_N, cancel_K] = tmp[is_not_best]
#         final_grad = sum_gradient(out)
#         if tol and np.square(final_grad - gradient_norm).sum() < tol:
#             break
#         gradient_norm = final_grad
#     return out


def global_regularization_exclusion(Xs):
    N ,K , L = Xs.shape
    currentXs = np.copy(Xs)
    out = np.empty((N, K), dtype=int)
    perms = list(permutations(range(K)))
    out[0] = np.arange(K)  # first point is identity
    for n in range(N- 1):
        diff = np.array([np.sqrt(np.square(currentXs[n] - Xs[n + 1, p]).sum(axis=1)).sum() for p in perms])
        best_p = perms[diff.argmin()]
        currentXs[n + 1] = Xs[n + 1, best_p]
        out[n + 1] = best_p
    return out





