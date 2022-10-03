# -*- coding: utf8 -*-
'''Unit tests for copula creation'''


from copulae.c import create_copula
from copulae.input import generate_copula_net_input
from copulae.utils import gauss_copula

from numpy.testing import assert_

import jax
import jax.numpy as jnp


def test_with_mlp():
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


def test_with_gauss_copula():
    '''
    Tests the input generator against a gaussian copula
    '''
    n_batches = 1
    batch_size = 512

    # parameters for the gaussian copula
    rho = 0.65
    mean = jnp.zeros(2)
    E = jnp.zeros(shape=(2, 2)) + rho
    E = E.at[0, 0].set(1)
    E = E.at[1, 1].set(1)

    # dataset
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = jax.random.multivariate_normal(
        subkey, mean=mean, cov=E, shape=(10000, )
    ).T

    key, _ = jax.random.split(key)
    U_batches, M_batches, _, _ = generate_copula_net_input(
        key, D, n_batches=n_batches,
        batch_size=batch_size
    )

    @jax.jit
    def forward_fun(_, U):
        return gauss_copula(U.T, mean, E).T

    cumulative, partial, density = create_copula(
        forward_fun
    )
