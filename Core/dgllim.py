"""Uses the knowledge of dF (where F is regular) instead of linear regression to enhance GLLiM"""

from Core.gllim import GLLiM


class dGLLiM(GLLiM):
    """Only valid with Lw = 0"""

    dF_hook = None # Function whih computes dF(x) not vectorized

    def _compute_Ak(self, Xnk, Y, SkList_X):
        # since Lw = 0, Xnk = Tn , and xk_bar = ck
        ck = (self.rnk[:, :, None] * Xnk).sum(axis=0) / self.rkList[:, None]
        return self.dF_hook(ck)


