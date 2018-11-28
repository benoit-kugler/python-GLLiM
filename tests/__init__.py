"""Tools to try and test differents implementation (numba, cython, parallelism, ...)."""
import numpy as np

def show_diff(a1, a2, label="diff : "):
    d =  np.max(np.abs(a1 - a2)) / np.max(np.abs(a1))
    print(label, d)


