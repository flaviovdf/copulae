# -*- coding: utf8 -*-


from copulae.typing import Tensor
from copulae.typing import PyTree


import jax
import jax.numpy as jnp


@jax.jit
def cross_entropy(
    Y: Tensor,
    logits: Tensor
) -> Tensor:
    logit = jnp.clip(logits, 1e-6, 1 - 1e-6)
    Y = jnp.clip(Y, 0, 1)
    return jnp.mean(
        -Y * jnp.log(logit) - (1 - Y) * jnp.log(1 - logit)
    )


@jax.jit
def l2(
    params: PyTree
):
    return jnp.array(
        jax.tree_map(
            lambda p: (p ** 2).sum(),
            params
        )
    ).sum()
