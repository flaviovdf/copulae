# -*- coding: utf8


from copulae.training.cflax.binorm import NormalBi
from copulae.training.cflax.bilogit import FlexibleBi

from copulae.training.cflax.mono_aux import PositiveLayer
from copulae.training.cflax.mono_aux import cumtrapz

from copulae.training.cflax.mlp import (
    MLP,
    PositiveMLP
)

from copulae.typing import Sequence
from copulae.typing import Tensor


from jax.scipy.special import erf


import flax.linen as nn

import jax.numpy as jnp
import jax


@jax.jit
def erfp(x: Tensor) -> Tensor:
    return (erf(x) + 1) / 2


@jax.jit
def erfp_integral(x: Tensor) -> Tensor:
    return 0.5 * (x * erf(x) + (((-x) ** 2) / jnp.pi) + x)


class TransformLayer(nn.Module):
    base: PositiveLayer

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        def trapz_zero(
            u0: float, u1: float, nd: int = 200
        ) -> float:
            u0_aux = jnp.linspace(0, 1, nd)
            ii = jnp.searchsorted(u0_aux, u0)
            u0_aux = jnp.insert(u0_aux, ii, u0)
            nd = u0_aux.shape[0]
            u1_aux = jnp.ones(nd) * u1

            U_aux = jnp.stack((u0_aux, u1_aux))
            z0 = cumtrapz(
                U_aux[0], self.base(U_aux).ravel()
            )
            z0 = z0 / z0[-1]

            ii = jnp.searchsorted(u0_aux, u0)
            return z0[ii]

        def trapz_one(
            u0: float, u1: float, nd: int = 200
        ) -> float:
            u1_aux = jnp.linspace(0, 1, nd)
            ii = jnp.searchsorted(u1_aux, u1)
            u1_aux = jnp.insert(u1_aux, ii, u1)
            nd = u1_aux.shape[0]
            u0_aux = jnp.ones(nd) * u0

            U_aux = jnp.stack((u0_aux, u1_aux))
            z1 = cumtrapz(
                U_aux[1], self.base(U_aux).ravel()
            )
            z1 = z1 / z1[-1]

            ii = jnp.searchsorted(u1_aux, u1)
            return z1[ii]

        z0 = jax.vmap(trapz_zero, in_axes=(0, 0))(
            U[0], U[1]
        )
        z0 = jnp.clip(z0, 1e-6, 1 - 1e-6)
        z1 = jax.vmap(trapz_one, in_axes=(0, 0))(
            U[0], U[1]
        )
        z1 = jnp.clip(z1, 1e-6, 1 - 1e-6)
        return jnp.stack((z0, z1))


class TwoCats(nn.Module):
    base: Sequence[TransformLayer]
    end: NormalBi | FlexibleBi

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        Z = U
        for pl in self.base:
            Z = pl(U)

        z0 = Z[0]
        z1 = Z[1]

        x0 = jnp.log(z0 / (1 - z0))
        x1 = jnp.log(z1 / (1 - z1))

        return self.end(x0, x1)


class TwoCatsTesting(nn.Module):
    base: MLP | PositiveMLP

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z = erfp(self.base(U).ravel())
        s = erfp_integral(1.0)

        z0 = (erfp_integral(z) * U[0]) / s
        z1 = (erfp_integral(z) * U[1]) / s

        x0 = jnp.log(z0 / (1 - z0))
        x1 = jnp.log(z1 / (1 - z1))

        return NormalBi()(x0, x1)
