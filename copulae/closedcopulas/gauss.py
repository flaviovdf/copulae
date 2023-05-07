# -*- coding: utf8 -*-
'''
Evaluates a Gaussian Copula (C, capital C) at the given
points `u`. This module is able to evalue gaussian
copulas or arbitrary dimensions. However, given that
it employs regular scipy (not jax), it is not grad
nor jit friendly.
'''

from copulae.typing import Tensor

import scipy.stats as ss


def C(
    u: Tensor,
    mean: Tensor,
    cov: Tensor
) -> Tensor:
    '''
    Evaluates a Gaussian Copula (C, capital C) at the given
    points `u`. Each element in `u` must be a number in
    [0, 1]. The mean of the multivariate Gaussian governing
    the copula is of `mean` and covariance `cov`.

    Note that all dimensions must match, i.e:
    u.shape[0] == mean.shape[0] == cov.shape[0|1]

    Arguments
    ---------
    u: Tensor (1-d array like)
        Input vector where each `u[i]` in [0, 1]
    mean: Tensor (1-d array like)
        The mean of the Gaussian copula. Must be of the
        same dimension of `u`
    cov: Tensor (2-d matrix)
        The covariance of the Gaussian copula. Must be
        a square matrix with the same dimension of `u`

    Returns
    -------
    A 1-d Tensor of the evaluated Copula.
    '''
    ppfs = ss.norm.ppf(u)
    return ss.multivariate_normal(
        mean=mean,
        cov=cov
    ).cdf(ppfs)
