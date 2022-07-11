# -*- coding: utf8 -*-
'''Unit tests for loss functions'''


from copulae.training.loss import cross_entropy
from copulae.training.loss import l1
from copulae.training.loss import l2
from copulae.training.loss import valid_density
from copulae.training.loss import valid_partial


import jax
import jax.numpy as jnp


def test_l1():
    params = []

    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    loss = l1(params=params)
    assert(loss == 8)


def test_l2():
    params = []

    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    loss = l2(params=params)
    assert(loss == 22)


def test_cross_entropy():
    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    loss = cross_entropy(Y_batches=Y, Ŷ_batches=Ŷ)
    assert(loss > 0)


def test_cross_entropy2():
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    loss = cross_entropy(Y_batches=Y, Ŷ_batches=Ŷ)
    assert(loss > 0)


def test_valid_density():
    U_batches = jnp.zeros((2, 2, 3))
    U_batches = U_batches.at[0].set(
        jnp.array([[1.1, 2.2, 3.3], [0, 0, 0]])
    )
    U_batches = U_batches.at[1].set(
        jnp.array([[1.1, 2.2, 3.3], [2, 2, 2]])
    )

    params = []
    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    @jax.jit
    def density(params, U_batches):
        return jnp.array([-1, -2, 0, 0])

    copula = {
        'density': density
    }

    loss = valid_density(
        params=params, copula=copula, U_batches=U_batches
    )
    assert(loss == 0.5)


def test_valid_partial():
    U_batches = jnp.zeros((2, 2, 3))
    U_batches = U_batches.at[0].set(
        jnp.array([[1.1, 2.2, 3.3], [0, 0, 0]])
    )
    U_batches = U_batches.at[1].set(
        jnp.array([[1.1, 2.2, 3.3], [2, 2, 2]])
    )

    params = []
    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    @jax.jit
    def partial_density(params, U_batches):
        return jnp.array([-1, -2, 0.1, 0.2, 2, 0.1])

    copula = {
        'partial_density': partial_density
    }

    loss = valid_partial(
        params=params, copula=copula, U_batches=U_batches
    )
    assert(loss == 0.5)
