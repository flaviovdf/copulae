# -*- coding: utf8 -*-


from copulae.training.cflax.mono_aux import PositiveLayer
from copulae.training.cflax.mono_aux import \
    integrate_and_set


from copulae.typing import Tensor


import flax.linen as nn


import jax
import jax.numpy as jnp


@jax.jit
def flexible_bi(
    z0, z1, m0, m1, sigma0, sigma1, alpha, theta
):

    z0 = (z0 - m0) / sigma0
    z1 = (z1 - m1) / sigma1

    base = jnp.exp(-alpha * z0)
    base += jnp.exp(-alpha * z1)
    base += theta * jnp.exp(-alpha * z0 - alpha * z1)
    base = 1.0 + base ** (1.0 / alpha)
    return 1.0 / base


class FlexibleBi(nn.Module):

    @nn.compact
    def __call__(self, z0: Tensor, z1: Tensor) -> Tensor:
        m0 = self.param(
            'm0',
            jax.nn.initializers.constant(0.0),
            (1, 1)
        )
        m1 = self.param(
            'm1',
            jax.nn.initializers.constant(0.0),
            (1, 1)
        )

        s0 = self.param(
            's0',
            jax.nn.initializers.constant(0.0),
            (1, 1)
        )
        s1 = self.param(
            's1',
            jax.nn.initializers.constant(0.0),
            (1, 1)
        )

        alpha = self.param(
            'alpha',
            jax.nn.initializers.constant(1.0),
            (1, 1)
        )
        theta = self.param(
            'theta',
            jax.nn.initializers.constant(0.0),
            (1, 1)
        )

        # alpha >= 1 and theta >= 0 and theta is bounded
        # by eq below. see pg 253 of
        # Handbook of the logistic distribution
        a = 1.0 + jnp.sqrt(alpha * alpha)
        a2 = a * a

        theta = jnp.sqrt(theta * theta)
        upper_theta = (
            (a2 - 1)**(1.0 / a) +  # noqa: W504
            ((1 + a) * (a2 - 1))**((1.0 - a) / a)
        ) ** a
        theta = jnp.clip(theta, 0, upper_theta)

        # deviations are always positive
        s0 = jnp.sqrt(s0 * s0) + 1e-6
        s1 = jnp.sqrt(s1 * s1) + 1e-6
        return flexible_bi(
            z0, z1, m0, m1, s0, s1, a, theta
        )


class SiamesePositiveBiLogitCopula(nn.Module):
    left: PositiveLayer
    right: PositiveLayer

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z0 = integrate_and_set(U[0], self.left(U).ravel())
        z1 = integrate_and_set(U[1], self.right(U).ravel())
        return FlexibleBi()(z0, z1)


class PositiveBiLogitCopula(nn.Module):
    base: PositiveLayer

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z_aux = self.base(U).ravel()
        z0 = integrate_and_set(U[0], z_aux)
        z1 = integrate_and_set(U[1], z_aux)
        return FlexibleBi()(z0, z1)
