# -*- coding: utf8 -*-
'''
Loss functions used to train neural copula. These should
be composed (added with respective weights) in order to
guarantee that neural networks mimick the copula behavior
correctly.

Most of theses losses have a regularization effect where
the weights of the network must be adjusted to mimick
copula like behavior.
'''

from copulae.training import CopulaTrainingState

from copulae.typing import PyTree
from copulae.typing import Tensor

import jax
import jax.numpy as jnp


@jax.jit
def cross_entropy(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the cross entropy between the neural copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶC_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶC_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶC_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 0, 1)

    rv = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).mean()

    return rv


@jax.jit
def jsd(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the jensen shannon divergence copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶC_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶC_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶC_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 1e-6, 1 - 1e-6)

    left = Y * jnp.log2(Y / Ŷ)
    left += (1 - Y) * jnp.log2((1 - Y) / (1 - Ŷ))

    right = Ŷ * jnp.log2(Ŷ / Y)
    right += (1 - Ŷ) * jnp.log2((1 - Ŷ) / (1 - Y))
    return (left + right).mean()


@jax.jit
def cross_entropy_partial(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the cross entropy between the neural copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    for each dimension.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶM_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 0, 1)

    rv = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).mean()

    return rv


@jax.jit
def jsd_partial(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the jensen shannon divergence copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶC_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶC_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶC_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 1e-6, 1 - 1e-6)

    left = Y * jnp.log2(Y / Ŷ)
    left += (1 - Y) * jnp.log2((1 - Y) / (1 - Ŷ))

    right = Ŷ * jnp.log2(Ŷ / Y)
    right += (1 - Ŷ) * jnp.log2((1 - Ŷ) / (1 - Y))
    return (left + right).mean()


@jax.jit
def l2(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Simple L2 regularization on the parameters of the net.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    return jnp.array(
        jax.tree_map(
            lambda p: (p ** 2).sum(), params
        )
    ).sum()


@jax.jit
def l1(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Simple L1 regularization on the parameters of the net.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    return jnp.array(
        jax.tree_map(
            lambda p: (jnp.abs(p)).sum(), params
        )
    ).sum()


@jax.jit
def frechet(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    A Copula must respect the Frechet bounds. For a 2d,
    Copula this is:

    max{u + v - 1, 0} <= C(u, v) <= min{u, v}

    For a nd Copula it is:

    max{sum(u_vector)-1, 0} <= C(u_vector) <= min(u_vector)

    This function returns the fraction of values the neural
    copula returns which do not respect the above
    inequality.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    L = jnp.clip(state.U_batches.sum(axis=1) - 1, 0, 1)
    R = jnp.clip(jnp.min(state.U_batches, axis=1), 0, 1)

    # same dim as L, and R
    Ŷ = jnp.clip(state.ŶC_batches, 0, 1).squeeze(-1)

    # -1 * sign --> penalizes the negative values
    # +1 --> output in the range [0, 2]
    # /2 --> output in the range [0, 1]
    loss = ((-1 * jnp.sign(Ŷ - L)) + 1).mean() / 2
    loss += ((-1 * jnp.sign(R - Ŷ)) + 1).mean() / 2
    return loss


@jax.jit
def valid_partial(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    The first derivative of a Copula maps to:

    dC(u, v)/du = F(X < x | Y = y), with
    F(X < x) = u
    F(Y < y) = y

    This value, stored in ŶM_batches must be in [0, 1].
    That is, cumulative distributions are always in [0, 1].
    This method will count the fraction of values outside
    this range.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    dC = state.ŶM_batches
    return (dC < 0).mean() + (dC > 1).mean()


@jax.jit
def valid_density(
    params: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    The second derivative of a Copula is a probability
    density function, thus it must be greater or equal to
    zero. This value is stored in Ŷc_batches.

    This, this function returns the fraction of invalid
    values.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    ddC = state.Ŷc_batches
    return (ddC < 0).mean()
