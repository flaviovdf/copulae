# -*- coding: utf8 -*-
'''
Code regarding neural networks which are guaranteed to
generate monotonic outputs. Here, we implement the
network defined in [1].

[1] Monotonic Networks. Joseph Sill. NeuRIPS 1997.
'''


from copulae.typing import Tensor
from copulae.typing import Sequence


import flax.linen as nn


import jax
import jax.numpy as jnp


class MLP(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)
        for layer_width in self.layers[:-1]:
            z = nn.Dense(layer_width)(a)
            a = nn.relu(z)
        return nn.Dense(self.layers[-1])(a)


class SingleLogitCopula(nn.Module):
    base: MLP

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        return jax.nn.sigmoid(
            nn.Dense(1)(self.base(U))
        )


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
            jax.nn.initializers.constant(1.0),
            (1, 1)
        )
        s1 = self.param(
            's1',
            jax.nn.initializers.constant(1.0),
            (1, 1)
        )

        alpha = self.param(
            'alpha',
            jax.nn.initializers.constant(1.0),
            (1, 1)
        )
        theta = self.param(
            'theta',
            jax.nn.initializers.constant(1.0),
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
        s0 = jnp.sqrt(s0 * s0)
        s1 = jnp.sqrt(s1 * s1)
        return flexible_bi(
            z0, z1, m0, m1, s0, s1, a, theta
        )


@jax.jit
def cumtrapz(u: Tensor, z: Tensor) -> Tensor:
    # u and z must be ordered according to u

    d = jnp.diff(u, prepend=0)
    s = jnp.zeros(d.shape[0], dtype=jnp.float32)
    s = s.at[0].set(z[0])
    s = s.at[1:].set(z[1:] + z[:-1])
    return (d * s / 2.0).cumsum()


@jax.jit
def integrate_and_set(u: Tensor, z: Tensor) -> Tensor:
    idx = u.argsort()
    u_sorted = u[idx]
    ct = cumtrapz(u_sorted, z[idx])
    reverse_idx = jnp.searchsorted(u_sorted, u)
    return ct[reverse_idx]


class ELUPlusOne(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)
        for layer_width in self.layers:
            z = nn.Dense(layer_width)(a)
            a = nn.elu(z) + 1
        z = nn.Dense(1)(a)
        return nn.elu(z) + 1


class SiamesePositiveBiLogitCopula(nn.Module):
    left: ELUPlusOne
    right: ELUPlusOne

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z0 = integrate_and_set(U[0], self.left(U).ravel())
        z1 = integrate_and_set(U[1], self.right(U).ravel())
        return FlexibleBi()(z0, z1)


class PositiveBiLogitCopula(nn.Module):
    base: ELUPlusOne

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z_aux = self.base(U).ravel()
        z0 = integrate_and_set(U[0], z_aux)
        z1 = integrate_and_set(U[1], z_aux)
        return FlexibleBi()(z0, z1)
