# -*- coding: utf8 -*-


from copulae.typing import Tensor
from copulae.typing import PyTree


import jax
import jax.numpy as jnp


@jax.jit
def cross_entropy(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    Ŷ = jnp.clip(Ŷ_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(Y_batches, 0, 1)

    ce_batches = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).sum(axis=1)

    return jnp.mean(ce_batches)


@jax.jit
def l2(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    return jnp.array(
        jax.tree_map(
            lambda p: (p ** 2).sum(),
            params
        )
    ).sum()


@jax.jit
def l1(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    return jnp.array(
        jax.tree_map(
            lambda p: (jnp.abs(p)).sum(),
            params
        )
    ).sum()


@jax.jit
def frechet(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    L = jnp.clip(U_batches.sum(axis=1) - 1, 0)
    R = jnp.min(U_batches, axis=1)
    logits = Ŷ_batches.squeeze(-1)  # same dim as L, and R

    # -1 * sign --> penalizes the negative values
    # +1 --> output in the range [0, 2]
    # /2 --> output in the range [0, 1]
    loss = ((-1 * jnp.sign(logits - L)) + 1).mean() / 2
    loss += ((-1 * jnp.sign(R - logits)) + 1).mean() / 2
    return loss


@jax.jit
def valid_partial(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    dC = copula['partial_density'](params, U_batches)
    return (dC < 0).mean() + (dC > 1).mean()


@jax.jit
def valid_density(
    *,
    params: PyTree,
    copula: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    Ŷ_batches: Tensor
) -> float:
    dC = copula['density'](params, U_batches)
    return (dC < 0).mean()
