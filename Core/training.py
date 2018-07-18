"""Implements severals training pattern, build on gllims classes.
Main functions have signature X (training), Y (training), K, *args, **options -> gllim.
Archiving of model parameters is not done in this module"""
import time
from multiprocessing.pool import Pool

import numpy as np

from Core.gllim import GLLiM

"""Number of iterations to choose rnk. Small"""
NB_ITER_RNK = 10

"""Number of instances to use when choosing rnk."""
NB_INSTANCES_RNK = 6

"""Number of iterations to fit """
NB_MAX_ITER = 100

"""Number of iterations for second learning"""
NB_MAX_ITER_SECOND = 20


"""Number of used processes"""
PROCESSES = 6

def run_gllim(process_index, gllim : GLLiM, Xtrain, Ytrain):
    np.random.seed(process_index * 100)  # Differents seed for different process
    try:
        gllim.fit(Xtrain, Ytrain, 'random', maxIter=NB_ITER_RNK)
    except AssertionError as e: # numerical issu
        print("Fitting interrupted due to numerical issues ({})".format(e))
    ll = gllim.current_ll
    return (ll, gllim.rnk, gllim)


def _best_rnk(Ttrain,Ytrain,K, Lw, sigma_type, gamma_type,
              gllim_cls, verbose):
    """Performs NB_ITER_RNK on NB_INSTANCES_RNK GLLiM instance, after GMM random init.
    Then chooses best rnk (from log-likelihood point of view) and returns it.
    Uses multiprocessing."""
    gllims = [(i,gllim_cls(K, Lw, sigma_type=sigma_type, gamma_type=gamma_type, verbose=verbose),
               Ttrain, Ytrain) for i in range(NB_INSTANCES_RNK)]

    with Pool(processes=PROCESSES) as p:
        r = p.starmap(run_gllim, gllims)

    maxll, rnk, gllim = max(r, key=lambda x: x[0])
    minll = min(g[0] for g in r)
    if verbose:
        print("Likelihood over differents processes : min : {}, max: {}".format(minll, maxll))
    return rnk, gllim


def run_gllim_precisions(process_index, gllim : GLLiM, Xtrain, Ytrain, ck_init, precision_factor):
    """Precisions in K ** precision_rate. ck_init_function may be random."""
    np.random.seed(process_index * 100)  # Differents seed for different process
    rho = np.ones(gllim.K) / gllim.K
    m = ck_init
    precisions = precision_factor * np.array([np.eye(Xtrain.shape[1])] * gllim.K)
    rnk = gllim._T_GMM_init(Xtrain,'random',
                            weights_init=rho,means_init=m,precisions_init=precisions)
    try:
        gllim.fit(Xtrain,Ytrain,{"rnk":rnk},maxIter=NB_ITER_RNK)
    except AssertionError as e:  # numerical issu
        print("Fitting interrupted due to numerical issues ({})".format(e))
    ll = gllim.current_ll
    return (ll, gllim.rnk, gllim)


def _best_rnk_precisions(Xtrain,Ytrain,K,ck_init_function,precision_rate,
                         Lw,sigma_type,gamma_type,gllim_cls,verbose) -> (np.array, GLLiM):
    """Performs NB_ITER_RNK on NB_INSTANCES_RNK GLLiM instance, after GMM init with given precisions.
    Then chooses best rnk (from log-likelihood point of view) and returns it.
    Uses multiprocessing."""

    ck_inits = []
    for i in range(NB_INSTANCES_RNK):
        np.random.seed(i * 100)
        ck_inits.append(ck_init_function())

    gllims =[ (i,gllim_cls(K,Lw,sigma_type=sigma_type,gamma_type=gamma_type,verbose=verbose),
               Xtrain,Ytrain,ck_init,precision_rate) for i , ck_init in enumerate(ck_inits)]

    with Pool(processes=PROCESSES) as p:
        r = p.starmap(run_gllim_precisions,gllims)

    maxll,rnk, gllim = max(r,key= lambda x : x[0])
    return rnk, gllim



def init_local(Ttrain, Ytrain, K, ck_init_function, precision_rate, Lw = 0, sigma_type ="iso", gamma_type ="full",
               track_theta = False, gllim_cls = GLLiM, verbose=False):
    """Initialise with given ck , uniform pik and Gammak (see run_gllim_precisions) on T then run joint GMM fit.
    Accepts covariances constraints and non zero Lw."""
    rnk,gllim = _best_rnk_precisions(Ttrain, Ytrain, K, ck_init_function, precision_rate, Lw, sigma_type, gamma_type, gllim_cls, verbose)
    if track_theta:
        gllim.start_track()
    gllim.fit(Ttrain, Ytrain, {"rnk": rnk}, maxIter=NB_MAX_ITER)
    return gllim




def multi_init(Ttrain, Ytrain, K, Lw = 0, sigma_type = "iso", gamma_type = "full",
               track_theta = False, gllim_cls = GLLiM, verbose=False):
    """Performs several inits to choose best rnk and starts iterations from it.
    Accepts covariances constraints and non zero Lw."""
    rnk,gllim = _best_rnk(Ttrain,Ytrain,K, Lw, sigma_type, gamma_type,gllim_cls, verbose)
    if track_theta:
        gllim.start_track()
    gllim.fit(Ttrain, Ytrain, {"rnk": rnk}, maxIter=NB_MAX_ITER)
    return gllim


def basic_fit(Ttrain, Ytrain, K, Lw = 0, sigma_type = "iso", gamma_type = "full",
              track_theta = False, gllim_cls = GLLiM, verbose=False) :
    """Performs one fit with random GMM initialisation"""
    gllim = gllim_cls(K, Lw, sigma_type=sigma_type, gamma_type=gamma_type, verbose=verbose)
    if track_theta:
        gllim.start_track()
    gllim.fit(Ttrain, Ytrain, 'random', maxIter=NB_MAX_ITER)
    return gllim


##    ----- SECOND LEARNING TOOLS   ------   ##

def job_second_learning(listXYK) -> [GLLiM]:
    """Trains one GLLiM for each tuple (X,Y,rnk)"""
    gllims = []
    for i , (X,Y,K) in enumerate(listXYK):
        print("Second learning {} in process... ({} data, {} clusters)".format(i,len(X),K))

        K = min(K, len(X))  # In case of degenerate sampling
        rho = np.ones(K) / K
        m = X[0:K,:]
        precisions = 10 * K  * np.array([np.eye(X.shape[1])] * K)
        gllim = GLLiM(K, Lw=0, sigma_type='iso', gamma_type='full', verbose=None)
        rnk = gllim._T_GMM_init(X, "random",
                                weights_init=rho, means_init=m, precisions_init=precisions)
        gllim.fit(X, Y, {"rnk": rnk}, maxIter=NB_MAX_ITER_SECOND)
        gllim.inversion()
        gllims.append(gllim)
    return gllims


def second_training_parallel(newXYK):
    print("Second learning starting...")
    chunck = len(newXYK) // PROCESSES
    parts = [newXYK[start * chunck:((start + 1) * chunck)] for start in range(PROCESSES - 1)]
    parts.append(newXYK[(PROCESSES - 1) * chunck:])

    t = time.time()

    with Pool(PROCESSES) as p:
        l = p.map(job_second_learning, parts)

    gllims = [g for sublist in l for g in sublist]
    print("Second learning time for {0} observations : {1:.2f} s".format(len(newXYK), time.time() - t))

    return gllims


