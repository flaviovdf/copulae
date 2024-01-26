# -*- coding: utf8


from copulae.training.cflax.binorm import NormalBi
from copulae.training.cflax.mlp import MLP

from copulae.typing import Tensor


from jax.scipy.special import erf


import flax.linen as nn

import jax.numpy as jnp
import jax.scipy.stats as jss
import jax


@jax.jit
def erfp(x: Tensor) -> Tensor:
    return (erf(x) + 1) / 2


@jax.jit
def erfp_integral(x: Tensor) -> Tensor:
    return 0.5 * (x * erf(x) + (((-x) ** 2) / jnp.pi) + x)


class TwoCats(nn.Module):
    base: MLP

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        z = erfp(self.base(U).ravel())
        max_ = self.base(jnp.ones((2, 1))).ravel()[0]
        s = erfp_integral(max_)

        t_0 = (erfp_integral(z) * U[0]) / s
        t_1 = (erfp_integral(z) * U[1]) / s

        return NormalBi()(t_0, t_1)
