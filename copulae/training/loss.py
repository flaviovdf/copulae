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
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the cross entropy between the neural copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶY_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶY_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶY_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 0, 1)

    rv = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).mean()

    return rv


@jax.jit
def cross_entropy_partial(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the cross entropy between the neural copula
    output for conditional CDFs (derivatives of C),
    and the empirical conditional CDFs estimated from data.

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
    Y = jnp.clip(state.C_batches, 0, 1)

    rv = (
        -Y * jnp.log2(Ŷ) - (1 - Y) * jnp.log2(1 - Ŷ)
    ).mean()

    return rv


@jax.jit
def jsd(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the jensen shannon divergence copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶY_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶY_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = jnp.clip(state.ŶY_batches, 1e-6, 1 - 1e-6)
    Y = jnp.clip(state.Y_batches, 1e-6, 1 - 1e-6)

    left = Y * jnp.log2(Y / Ŷ)
    left += (1 - Y) * jnp.log2((1 - Y) / (1 - Ŷ))

    right = Ŷ * jnp.log2(Ŷ / Y)
    right += (1 - Ŷ) * jnp.log2((1 - Ŷ) / (1 - Y))
    return (left + right).mean()


@jax.jit
def jsd_partial(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the Jensen Shannon Div between the neural
    copula output for conditional CDFs (derivatives of C),
    and the empirical conditional CDFs estimated from data.

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
    Y = jnp.clip(state.C_batches, 1e-6, 1 - 1e-6)

    left = Y * jnp.log2(Y / Ŷ)
    left += (1 - Y) * jnp.log2((1 - Y) / (1 - Ŷ))

    right = Ŷ * jnp.log2(Ŷ / Y)
    right += (1 - Ŷ) * jnp.log2((1 - Ŷ) / (1 - Y))
    return (left + right).mean()


@jax.jit
def sq_error(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the squared error between the neural copula
    output (capital C, or the Cumulative of the copula),
    and the empirical multivariate cumulative distribution
    function. Below we detail which parameters are used.

    ŶY_batches = C(u, v)
    Y_batches = ECDF(x, y)

    where

    F(X < x) = u
    F(Y < y) = y

    this method this returns the cross-entropy of
    Y_batches and ŶY_batches.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = state.ŶY_batches
    Y = state.Y_batches

    return jnp.power(Y - Ŷ, 2).mean()


@jax.jit
def sq_error_partial(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    Computes the squared error between the neural copula
    output for conditional CDFs (derivatives of C),
    and the empirical conditional CDFs estimated from data.

    Arguments
    ---------
    state: CopulaTrainingState
        The tensors composing the last evaluation of the
        neural copula

    Returns
    -------
    Tensor of size (1, 1) with the loss
    '''
    Ŷ = state.ŶC_batches
    Y = state.C_batches

    return jnp.power(Y - Ŷ, 2).mean()


@jax.jit
def data_likelihood(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    # (n_batches, n_ex)
    copula_density = jnp.clip(state.Ŷc_batches, 1e-6)
    kde_density = jnp.clip(state.I_pdf, 1e-6)
    return -(
        jnp.log2(copula_density) + jnp.log2(kde_density)
    ).mean()


@jax.jit
def copula_likelihood(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    copula_density = state.Ŷc_batches  # (n_batches, n_ex)
    return -jnp.log2(jnp.clip(copula_density, 1e-6)).mean()


@jax.jit
def l2(
    params: PyTree,
    _: CopulaTrainingState,
) -> Tensor:
    '''
    Simple L2 regularization on the parameters of the net.

    Arguments
    ---------
    params: PyTree
        The parameters to compute l2

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
    _: CopulaTrainingState,
) -> Tensor:
    '''
    Simple L1 regularization on the parameters of the net.

    Arguments
    ---------
    params: PyTree
        The parameters to compute l2
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
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    A Copula must respect the Frechet bounds. For a 2d,
    Copula this is:

    max{u + v - 1, 0} <= C(u, v) <= min{u, v}

    For a nd Copula it is:

    max{sum(u_vector)-1, 0} <= C(u_vector) <= min(u_vector)

    This function returns the fraction of values the neural
    copula returns that do not respect the above
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
    Ŷ = jnp.clip(state.ŶY_batches, 0, 1).squeeze(-1)

    # -1 * sign --> penalizes the negative values
    # +1 --> output in the range [0, 2]
    # /2 --> output in the range [0, 1]
    loss = ((-1 * jnp.sign(Ŷ - L)) + 1).mean() / 2
    loss += ((-1 * jnp.sign(R - Ŷ)) + 1).mean() / 2
    return loss


@jax.jit
def sq_frechet(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    A Copula must respect the Frechet bounds. For a 2d,
    Copula this is:

    max{u + v - 1, 0} <= C(u, v) <= min{u, v}

    For a nd Copula it is:

    max{sum(u_vector)-1, 0} <= C(u_vector) <= min(u_vector)

    This function returns the square of values the neural
    copula returns that do not respect the above
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
    Ŷ = jnp.clip(state.ŶY_batches, 0, 1).squeeze(-1)

    rv = jnp.power(Ŷ[Ŷ < L], 2) + jnp.power(Ŷ[Ŷ > R], 2)
    return rv.mean()


@jax.jit
def valid_partial(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    The first derivative of a Copula maps to:

    dC(u, v)/du = F(X < x | Y = y), with
    F(X < x) = u
    F(Y < y) = y

    This value, stored in ŶC_batches must be in [0, 1].
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
    dC = state.ŶC_batches
    return (dC < 0).mean() + (dC > 1).mean()


@jax.jit
def sq_valid_partial(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    The first derivative of a Copula maps to:

    dC(u, v)/du = F(X < x | Y = y), with
    F(X < x) = u
    F(Y < y) = y

    This value, stored in ŶC_batches must be in [0, 1].
    That is, cumulative distributions are always in [0, 1].
    This method will sum the square of values outside
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
    dC = state.ŶC_batches
    rv = jnp.power(dC[dC < 0], 2)
    rv += jnp.power(dC[dC > 0], 2)
    return rv.mean()


@jax.jit
def valid_density(
    _: PyTree,
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


@jax.jit
def sq_valid_density(
    _: PyTree,
    state: CopulaTrainingState,
) -> Tensor:
    '''
    The second derivative of a Copula is a probability
    density function, thus it must be greater or equal to
    zero. This value is stored in Ŷc_batches.

    This, this function returns the squared sum of invalid
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
    return jnp.power(ddC[ddC < 0], 2).mean()
