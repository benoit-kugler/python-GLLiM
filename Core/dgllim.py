"""Uses the knowledge of dF (where F is regular) instead of linear regression to enhance GLLiM"""
import numpy as np
from Core.gllim import GLLiM


class dGLLiM(GLLiM):
    """Only valid with Lw = 0"""

    dF_hook = None
    """Function whih computes dF(x), vectorized"""

    def _compute_Ak(self, Xnk, Y, SkList_X):
        # since Lw = 0, Xnk = Tn , and xk_bar = ck
        ck = (self.rnk[:, :, None] * Xnk).sum(axis=0) / self.rkList[:, None]
        return self.dF_hook(ck)


class ZeroDeltadGLLiM(dGLLiM):
    F_hook = None
    """Function with computes F(x), vectorized"""

    def _compute_bk(self, Y, Xnk, AkList):
        # since Lw = 0, Xnk = Tn , and xk_bar = ck
        ck = (self.rnk[:, :, None] * Xnk).sum(axis=0) / self.rkList[:, None]
        return self.F_hook(ck) - np.matmul(AkList, ck[:, :, None])[:, :, 0]
