# -*- coding: utf8 -*-
# noqa: E501


from copulae.training.cflax.mono_aux import ELUPlusOne
from copulae.training.cflax.mono_aux import \
    integrate_and_set


from copulae.typing import Tensor


from jax.scipy.special import erf
from jax.scipy.stats.norm import cdf as cdf1d


import flax.linen as nn


import jax
import jax.numpy as jnp


@jax.jit
def case1(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 * (erf(q / sqrt2) + erf(b / (sqrt2 * a)))
    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)  # noqa: E501

    line21 = 1 - erf((sqrt2 * b - a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))  # noqa: E501
    aux3 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)  # noqa: E501

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1  # noqa: E501
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))  # noqa: E501

    return line11 + (line12 * line21) - (line22 * (line31 + line32))  # noqa: E501


@jax.jit
def case2(p, q):
    return cdf1d(p) * cdf1d(q)


@jax.jit
def case3(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    aux1 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line11 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)  # noqa: E501

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1  # noqa: E501
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line12 = 1.0 + erf(aux3 / aux4)

    return line11 * line12


@jax.jit
def case4(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 + 0.5 * erf(q / sqrt2)
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)  # noqa: E501

    aux3 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 - a * c1  # noqa: E501
    aux4 = 2 * jnp.sqrt(1 - a2 * c2)
    line2 = 1.0 + erf(aux3 / aux4)

    return line11 - (line12 * line2)


@jax.jit
def case5(p, q, rho, a, b):
    c1 = -1.0950081470333
    c2 = -0.75651138383854
    sqrt2 = jnp.sqrt(2)
    a2 = a * a
    b2 = b * b

    line11 = 0.5 - 0.5 * erf(b / (sqrt2 * a))
    aux1 = a2 * c1 * c1 + 2 * sqrt2 * b * c1 + 2 * b2 * c2
    aux2 = 4 * (1 - a2 * c2)
    line12 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux1 / aux2)  # noqa: E501

    line21 = 1 - erf((sqrt2 * b + a2 * c1) / (2 * a * jnp.sqrt(1 - a2 * c2)))  # noqa: E501
    aux3 = a2 * c1 * c1 - 2 * sqrt2 * b * c1 + 2 * b2 * c2
    line22 = (1.0 / (4 * jnp.sqrt(1 - a2 * c2))) * jnp.exp(aux3 / aux2)  # noqa: E501

    aux4 = sqrt2 * q - sqrt2 * a2 * c2 * q - sqrt2 * a * b * c2 + a * c1  # noqa: E501
    aux5 = 2 * jnp.sqrt(1 - a2 * c2)
    line31 = erf(aux4 / aux5)
    line32 = erf((-a2 * c1 + sqrt2 * b) / (2 * a * jnp.sqrt(1 - a2 * c2)))  # noqa: E501

    return line11 - (line12 * line21) + line22 * (line31 + line32)  # noqa: E501


def binorm(p, q, rho):
    if rho == 0:
        return case2(p, q)

    p = jnp.asarray(jnp.atleast_1d(p), dtype=jnp.float32)
    q = jnp.asarray(jnp.atleast_1d(q), dtype=jnp.float32)

    a = -rho / jnp.sqrt(1 - rho * rho)
    b = p / jnp.sqrt(1 - rho * rho)

    idx_1 = jnp.where(
        (a > 0) & ((a * q + b) >= 0),
    )

    idx_3 = jnp.where(
        (a > 0) & ((a * q + b) < 0),
    )

    idx_4 = jnp.where(
        (a < 0) & ((a * q + b) >= 0),
    )

    idx_5 = jnp.where(
        (a < 0) & ((a * q + b) < 0),
    )

    rv = jnp.zeros_like(p)
    rv = rv.at[idx_1].set(
        case1(p[idx_1], q[idx_1], rho, a, b[idx_1])
    )
    rv = rv.at[idx_3].set(
        case3(p[idx_3], q[idx_3], rho, a, b[idx_3])
    )
    rv = rv.at[idx_4].set(
        case4(p[idx_4], q[idx_4], rho, a, b[idx_4])
    )
    rv = rv.at[idx_5].set(
        case5(p[idx_5], q[idx_5], rho, a, b[idx_5])
    )
    return rv


class NormalBi(nn.Module):

    @nn.compact
    def __call__(self, z0: Tensor, z1: Tensor) -> Tensor:
        mu0 = jnp.mean(z0)
        s0 = jnp.std(z0, ddof=1)

        mu1 = jnp.mean(z1)
        s1 = jnp.std(z1, ddof=1)

        rho = jnp.corrcoef(z0, z1)

        p = (z0 - mu0) / s0
        q = (z1 - mu1) / s1

        return binorm(p, q, rho)


class SiamesePositiveBiNormalCopula(nn.Module):
    left: ELUPlusOne
    right: ELUPlusOne

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z0 = integrate_and_set(U[0], self.left(U).ravel())
        z1 = integrate_and_set(U[1], self.right(U).ravel())
        return NormalBi()(z0, z1)


class PositiveBiNormalCopula(nn.Module):
    base: ELUPlusOne

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z_aux = self.base(U).ravel()
        z0 = integrate_and_set(U[0], z_aux)
        z1 = integrate_and_set(U[1], z_aux)
        return NormalBi()(z0, z1)
