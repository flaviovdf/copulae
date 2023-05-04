# -*- coding: utf8 -*-


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
