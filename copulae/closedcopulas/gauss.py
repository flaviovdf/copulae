# -*- coding: utf8 -*-
'''
Evaluates a Gaussian Copula (C, capital C) at the given
points `u`. This module is able to evalue gaussian
copulas or arbitrary dimensions.
'''


from copulae.training.cflax.binorm import binorm as \
    binorm_cdf


from copulae.typing import Tensor


import flax.linen as nn


import jax.scipy.stats as jss
import jax.scipy.special as jspecial
import jax.numpy as jnp


class GaussCopula(nn.Module):

    @classmethod
    def c(cls, rho: Tensor, UV: Tensor) -> Tensor:
        u = UV[0]
        v = UV[1]

        a = jnp.sqrt(2) * jspecial.erfinv(2 * u - 1)
        aa = a * a
        b = jnp.sqrt(2) * jspecial.erfinv(2 * v - 1)
        bb = b * b
        rr = rho * rho
        abrho = a * b * rho

        rv = 1.0 / jnp.sqrt(1 - rr)
        rv = rv * jnp.exp(
            -((aa + bb) * rr - 2 * abrho) / (2 * (1 - rr))
        )

        return rv

    @classmethod
    def dC_du(cls, rho: Tensor, UV: Tensor) -> Tensor:
        u = UV[0]
        x1 = jss.norm.ppf(u)
        v = UV[1]
        x2 = jss.norm.ppf(v)
        rr = rho * rho
        return jss.norm.cdf(
            (x2 - rho * x1) / jnp.sqrt(1 - rr)
        )

    @classmethod
    def dC_dv(cls, rho: Tensor, UV: Tensor) -> Tensor:
        u = UV[0]
        x1 = jss.norm.ppf(u)
        v = UV[1]
        x2 = jss.norm.ppf(v)
        rr = rho * rho
        return jss.norm.cdf(
            (x1 - rho * x2) / jnp.sqrt(1 - rr)
        )

    @classmethod
    def C(cls, rho: Tensor, UV: Tensor) -> Tensor:
        u = UV[0]
        v = UV[1]
        x1 = jss.norm.ppf(u)
        x2 = jss.norm.ppf(v)
        return binorm_cdf(x1, x2, rho)

    @nn.compact
    def __call__(self, UV: Tensor) -> Tensor:
        rho = self.param(
            'rho',
            nn.initializers.lecun_normal(),
            (1, 1)
        )
        rho = jnp.clip(nn.tanh(rho), -0.9999, 0.9999)
        return GaussCopula.C(rho[0, 0], UV)[:, jnp.newaxis]


def C(
    u: Tensor,
    mean: Tensor,
    cov: Tensor
) -> Tensor:
    import scipy.stats as ss
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
