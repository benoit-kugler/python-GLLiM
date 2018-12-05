import time

import numpy as np

from Core import cython, em_is_gllim
from old import em_is_gllim_jit


def test_mu_step_diag(Ny=10000, Ns=40, D=10, L=4, K=40):
    """Compare jit, cython sequential, cython parallel for mu step with IS"""
    print("\nTesting mu_step_diag_IS. Creating datas...")
    T = np.tril(np.ones((L, L))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD
    gllim_covs = np.array([cov] * K)

    weightss = np.random.random_sample((Ny, K))
    weightss /= weightss.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((Ny, K, L)) * 12.2

    current_mean = np.random.random_sample(D)
    current_cov = np.arange(D) + 1.2

    Yobs = np.random.random_sample((Ny, D))
    Xs = np.random.random_sample((Ny, Ns, L))
    FXs = np.random.random_sample((Ny, Ns, D)) * 9
    mask = np.asarray(np.random.random_sample((Ny, Ns)) > 0.4, dtype=int)

    print("Compiling jit...")
    em_is_gllim_jit._mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask,   #compiling
                                                      gllim_covs, current_mean, current_cov)
    print("Starting runs...")
    ti = time.time()
    maximal_mu1, ws1 =  em_is_gllim_jit._mu_step_diag(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov)
    print(f"Jit               : {time.time() - ti:.3f} s")


    ti = time.time()
    maximal_mu2, ws2 = cython.mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov, parallel=False)
    print(f"Cython sequential : {time.time() - ti:.3f} s")


    ti = time.time()
    maximal_mu3, ws3 = cython.mu_step_diag_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov, parallel=True)
    print(f"Cython parallel   : {time.time() - ti:.3f} s")

    em_is_gllim.WITH_THREADS = False
    ti = time.time()
    maximal_mu4, ws4 = em_is_gllim.mu_step_diag_IS_joblib(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov)
    print(f"Joblib (process)  : {time.time() - ti:.3f} s")

    em_is_gllim.WITH_THREADS = True
    ti = time.time()
    maximal_mu5, ws5 = em_is_gllim.mu_step_diag_IS_joblib(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov)
    print(f"Joblib (threads)  : {time.time() - ti:.3f} s")

    assert np.allclose(maximal_mu1, maximal_mu2)
    assert np.allclose(maximal_mu1, maximal_mu3)
    assert np.allclose(maximal_mu3, maximal_mu4)
    assert np.allclose(maximal_mu3, maximal_mu5)
    assert np.allclose(ws1, ws2)
    assert np.allclose(ws1, ws3)
    assert np.allclose(ws3, ws4)
    assert np.allclose(ws3, ws5)



def test_mu_step_full(Ny=100, Ns=4000, D=10, L=4, K=40):
    """Compare jit, cython sequential, cython parallel for mu step with IS"""
    print("\nTesting mu_step_full_IS. Creating datas...")
    T = np.tril(np.ones((L, L))) * 0.456
    cov = np.dot(T, T.T)
    U = np.linalg.cholesky(cov).T  # DxD
    gllim_covs = np.array([cov] * K)

    weightss = np.random.random_sample((Ny, K))
    weightss /= weightss.sum(axis=1, keepdims=True)
    meanss = np.random.random_sample((Ny, K, L)) * 12.2

    current_mean = np.random.random_sample(D)
    T = np.tril(np.ones((D, D))) * 0.389
    current_cov = np.dot(T, T.T)

    Yobs = np.random.random_sample((Ny, D))
    Xs = np.random.random_sample((Ny, Ns, L))
    FXs = np.random.random_sample((Ny, Ns, D)) * 9
    mask = np.asarray(np.random.random_sample((Ny, Ns)) > 0.4, dtype=int)

    # print("Compiling jit...")
    # em_is_gllim_jit._mu_step_full(Yobs, Xs, meanss, weightss, FXs, mask,   #compiling
    #                                                   gllim_covs, current_mean, current_cov)
    # print("Starting runs...")
    # ti = time.time()
    # maximal_mu1, ws1 =  em_is_gllim_jit._mu_step_full(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean, current_cov)
    # print(f"Jit               : {time.time() - ti:.3f} s")


    # ti = time.time()
    # maximal_mu2, ws2 = cython.mu_step_full_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
    #                                         current_cov, parallel=False)
    # print(f"Cython sequential : {time.time() - ti:.3f} s")


    ti = time.time()
    maximal_mu3, ws3 = cython.mu_step_full_IS(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov, parallel=True)
    print(f"Cython parallel   : {time.time() - ti:.3f} s")

    em_is_gllim.WITH_THREADS = False
    # ti = time.time()
    # maximal_mu4, ws4 = em_is_gllim.mu_step_full_IS_joblib(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
    #                                         current_cov)
    # print(f"Joblib (process)  : {time.time() - ti:.3f} s")

    em_is_gllim.WITH_THREADS = True
    ti = time.time()
    maximal_mu5, ws5 = em_is_gllim.mu_step_full_IS_joblib(Yobs, Xs, meanss, weightss, FXs, mask, gllim_covs, current_mean,
                                            current_cov)
    print(f"Joblib (threads)  : {time.time() - ti:.3f} s")

    # assert np.allclose(maximal_mu1, maximal_mu2)
    # assert np.allclose(maximal_mu1, maximal_mu3)
    # assert np.allclose(maximal_mu3, maximal_mu4)
    assert np.allclose(maximal_mu3, maximal_mu5)
    # assert np.allclose(ws1, ws2)
    # assert np.allclose(ws1, ws3)
    # assert np.allclose(ws3, ws4)
    assert np.allclose(ws3, ws5)

if __name__ == '__main__':
    test_mu_step_diag(100,2000)
    test_mu_step_full(100,2000)
