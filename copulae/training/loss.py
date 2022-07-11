# -*- coding: utf8 -*-


from copulae.training import CopulaTrainingState

from copulae.typing import Tensor

import jax
import jax.numpy as jnp


def cross_entropy(
    state: CopulaTrainingState,
) -> Tensor:
    Ŷ = jnp.clip(state.ŶC_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 0, 1)

    ce_batches = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).sum(axis=1)

    return jnp.mean(ce_batches)


def l2(
    state: CopulaTrainingState,
) -> Tensor:
    return jnp.array(
        jax.tree_map(
            lambda p: (p ** 2).sum(),
            state.params
        )
    ).sum()


def l1(
    state: CopulaTrainingState,
) -> Tensor:
    return jnp.array(
        jax.tree_map(
            lambda p: (jnp.abs(p)).sum(),
            state.params
        )
    ).sum()


def frechet(
    state: CopulaTrainingState,
) -> Tensor:
    L = jnp.clip(state.U_batches.sum(axis=1) - 1, 0)
    R = jnp.min(state.U_batches, axis=1)

    # same dim as L, and R
    logits = state.Ŷ_batches.squeeze(-1)

    # -1 * sign --> penalizes the negative values
    # +1 --> output in the range [0, 2]
    # /2 --> output in the range [0, 1]
    loss = ((-1 * jnp.sign(logits - L)) + 1).mean() / 2
    loss += ((-1 * jnp.sign(R - logits)) + 1).mean() / 2
    return loss


def valid_partial(
    state: CopulaTrainingState,
) -> Tensor:
    dC = state.ŶM_batches
    return (dC < 0).mean() + (dC > 1).mean()


def valid_density(
    state: CopulaTrainingState,
) -> Tensor:
    ddC = state.Ŷc_batches
    return (ddC < 0).mean()
