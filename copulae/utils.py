# -*- coding: utf8 -*-
'''Small utilities with no other home'''


from copulae.typing import Tensor

import scipy.stats as ss


def gauss_copula(
    u: Tensor,
    mean: Tensor,
    cov: Tensor
) -> Tensor:
    ppfs = ss.norm.ppf(u)
    return ss.multivariate_normal(
        mean=mean,
        cov=cov
    ).cdf(ppfs)
