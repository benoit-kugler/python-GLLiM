from .multinomial import multinomial_sampling as multinomial_sampling_cython
from .sampling import sampling_sameCov_chols
from .noise_em import (sigma_step_full_NoIS, sigma_step_diag_NoIS, mu_step_NoIS,
                       mu_step_diag_IS, mu_step_full_IS,
                       sigma_step_diag_IS, sigma_step_full_IS,
                       test)

from .gllim import (compute_next_theta_GFull_SFull, compute_next_theta_GDiag_SFull, compute_next_theta_GIso_SFull,
                    compute_next_theta_GFull_SDiag, compute_next_theta_GDiag_SDiag, compute_next_theta_GIso_SDiag,
                    compute_next_theta_GFull_SIso, compute_next_theta_GDiag_SIso, compute_next_theta_GIso_SIso)


class gllim:
    compute_next_theta_GFull_SFull = compute_next_theta_GFull_SFull
    compute_next_theta_GDiag_SFull = compute_next_theta_GDiag_SFull
    compute_next_theta_GIso_SFull = compute_next_theta_GIso_SFull
    compute_next_theta_GFull_SDiag = compute_next_theta_GFull_SDiag
    compute_next_theta_GDiag_SDiag = compute_next_theta_GDiag_SDiag
    compute_next_theta_GIso_SDiag = compute_next_theta_GIso_SDiag
    compute_next_theta_GFull_SIso = compute_next_theta_GFull_SIso
    compute_next_theta_GDiag_SIso = compute_next_theta_GDiag_SIso
    compute_next_theta_GIso_SIso = compute_next_theta_GIso_SIso


from .gllim_para import (compute_next_theta_GFull_SFull, compute_next_theta_GDiag_SFull, compute_next_theta_GIso_SFull,
                         compute_next_theta_GFull_SDiag, compute_next_theta_GDiag_SDiag, compute_next_theta_GIso_SDiag,
                         compute_next_theta_GFull_SIso, compute_next_theta_GDiag_SIso, compute_next_theta_GIso_SIso)


class gllim_para:
    compute_next_theta_GFull_SFull = compute_next_theta_GFull_SFull
    compute_next_theta_GDiag_SFull = compute_next_theta_GDiag_SFull
    compute_next_theta_GIso_SFull = compute_next_theta_GIso_SFull
    compute_next_theta_GFull_SDiag = compute_next_theta_GFull_SDiag
    compute_next_theta_GDiag_SDiag = compute_next_theta_GDiag_SDiag
    compute_next_theta_GIso_SDiag = compute_next_theta_GIso_SDiag
    compute_next_theta_GFull_SIso = compute_next_theta_GFull_SIso
    compute_next_theta_GDiag_SIso = compute_next_theta_GDiag_SIso
    compute_next_theta_GIso_SIso = compute_next_theta_GIso_SIso


from .probas import test_chol
