# -*- coding: utf8 -*-


from copulae.training.cflax.mono_aux import \
    integrate_and_set

from copulae.typing import Sequence
from copulae.typing import Tensor

import flax.linen as nn


import jax

import jax.numpy as jnp

import jax.scipy.stats as jss
import jax.scipy.special as jspecial


class LogitPDFNet(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)

        z = nn.Dense(self.layers[0])(a)
        a = nn.relu(z)

        for layer_width in self.layers[1:]:
            z = nn.Dense(layer_width)(a)
            a = nn.relu(z)

        z = nn.Dense(1)(a)
        e = jnp.exp(-z)
        return e / (1 + e)


class NormalPDFNet(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)

        z = nn.Dense(self.layers[0])(a)
        a = nn.relu(z)

        for layer_width in self.layers[1:]:
            z = nn.Dense(layer_width)(a)
            a = nn.relu(z)

        z = nn.Dense(1)(a)
        return jss.norm.pdf(z)


class GaussCopNet(nn.Module):

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        rho = self.param(
            'rho',
            jax.nn.initializers.lecun_normal(),
            (1, 1)
        )
        rho = jnp.clip(nn.tanh(rho), -0.9999, 0.9999)

        U = jnp.clip(U, 0, 1)
        u = U[0]
        v = U[1]

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


class FGMCopNet(nn.Module):

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        theta = self.param(
            'theta',
            jax.nn.initializers.lecun_normal(),
            (1, 1)
        )
        theta = jnp.clip(nn.tanh(theta), -0.9999, 0.9999)

        U = jnp.clip(U, 0, 1)
        u = U[0]
        v = U[1]

        rv = 1.0 + theta * (1 - 2 * u) * (1 - 2 * v)
        return rv


class FrankCopNet(nn.Module):

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        theta = self.param(
            'theta',
            jax.nn.initializers.lecun_normal(),
            (1, 1)
        )
        theta = theta + 1e-6

        U = jnp.clip(U, 0, 1)

        u = U[0]
        v = U[1]

        num = -theta * jnp.exp(-theta * (u + v))
        num = num * (jnp.exp(-theta) - 1)

        den = jnp.exp(-theta)
        den = den + jnp.exp(-theta * u)
        den = den + jnp.exp(-theta * v)
        den = den + jnp.exp(-theta * (u + v))
        den = den * den

        return num / den


class MixtureCopula(nn.Module):
    components: Sequence[
        NormalPDFNet | LogitPDFNet | GaussCopNet | \
        FGMCopNet | FrankCopNet  # noqa: E502
    ]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        weights = self.param(
            'weights',
            jax.nn.initializers.lecun_normal(),
            (len(self.components), 1)
        )

        weights = nn.softmax(weights)

        rv = 0.0
        for i, comp in enumerate(self.components):
            rv += comp(U) * weights[i, 0]
        return rv


class DoubleIntegral(nn.Module):
    base: MixtureCopula | NormalPDFNet | LogitPDFNet | \
        GaussCopNet | FGMCopNet | FrankCopNet

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z_aux = self.base(U).ravel()
        z0 = integrate_and_set(U[0], z_aux)
        z1 = integrate_and_set(U[1], z0)
        return z1.reshape(1, z1.shape[0])
