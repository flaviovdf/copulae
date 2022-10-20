# -*- coding: utf8 -*-
'''Unit tests for copula creation'''


from copulae.c import create_copula

from numpy.testing import assert_
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

import jax
import jax.numpy as jnp


def test_with_mlp():
    '''Simple test for the construction'''

    @jax.jit
    def forward_fun(params, U):
        W, b = params[0]
        A = jax.nn.relu(W @ U + b)
        W, b = params[1]
        return jax.nn.sigmoid(W @ A + b).T

    cumulative, _, _ = create_copula(forward_fun)

    params = []
    weights = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    weights = jnp.array([[1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    U = jnp.zeros(shape=(1, 2, 1), dtype=jnp.float32)
    U = U.at[0, 0].set(0.5)
    U = U.at[0, 1].set(0.5)

    assert_(cumulative(params, U) >= 0)
    assert_(cumulative(params, U) <= 1)


def test_marshal_olkin():
    '''
    The Marshall Olkin copula has a closed expression for
    the density. From the reference below:

    http://ajmaa.org/searchroot/files/pdf/v11n1/v11i1p2.pdf
    '''
    @jax.jit
    def forward_fun(params, U):
        a1 = params[0]
        a2 = params[1]
        return jnp.minimum(
            U[1] * jnp.power(U[0], 1.0 - a1),
            U[0] * jnp.power(U[1], 1.0 - a2)
        )

    cumulative, _, density = create_copula(forward_fun)
    params = jnp.array([0.25, 0.75])
    U = jnp.zeros(shape=(1, 2, 1), dtype=jnp.float32)
    U = U.at[0, 0].set(0.6)
    U = U.at[0, 1].set(0.2)

    assert_almost_equal(
        density(params, U)[0, 0], 0.8521645248506244
    )


def test_closed_form_partial():
    '''
    Copulas of the form:

    C(u, v) = uv / (u + v - uv)

    Have a closed form partial of:

    c_u(v) =  (v / (u + v - uv))**2

    See example 2.20 of "An introduction to copulas"

    We shall use this expression to test our partial
    derivatives
    '''
    @jax.jit
    def forward_fun(params, U):
        return (U[0] * U[1]) / (U[0] + U[1] - U[0] * U[1])

    _, partial, _ = create_copula(forward_fun)
    params = jnp.array([])
    U = jnp.zeros(shape=(1, 2, 1), dtype=jnp.float32)
    U = U.at[0, 0].set(0.6)
    U = U.at[0, 1].set(0.2)

    assert_almost_equal(
        partial(params, U)[0, 0, 0],
        (0.2 / (0.6 + 0.2 - 0.6 * 0.2)) ** 2
    )

    assert_almost_equal(
        partial(params, U)[0, 1, 0],
        (0.6 / (0.6 + 0.2 - 0.6 * 0.2)) ** 2
    )


def test_partial_shape():
    '''The shape of the partial is (batches, dim, elems)'''
    
    @jax.jit
    def forward_fun(params, U):
        return (U[0] * U[1]) / (U[0] + U[1] - U[0] * U[1])

    _, partial, _ = create_copula(forward_fun)
    params = jnp.array([])
    U = jnp.zeros(shape=(3, 2, 5), dtype=jnp.float32) + 0.1
    assert_equal((3, 2, 5), partial(params, U).shape)


def test_density_shape():
    '''The shape of the density must be batches by elem'''

    @jax.jit
    def forward_fun(params, U):
        W, b = params[0]
        A = jax.nn.relu(W @ U + b)
        W, b = params[1]
        return jax.nn.sigmoid(W @ A + b).T

    _, _, density = create_copula(forward_fun)

    params = []
    weights = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    weights = jnp.array([[1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    U = jnp.ones(shape=(6, 2, 12), dtype=jnp.float32)

    assert_equal((6, 12), density(params, U).shape)
