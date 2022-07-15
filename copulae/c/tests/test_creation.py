# -*- coding: utf8 -*-
'''Unit tests for copula creation'''


from copulae.c import create_copula


import jax
import jax.numpy as jnp


def test_cumulative():
    @jax.jit
    def forward_fun(params, U):
        W, b = params[0]
        A = jax.nn.relu(W @ U + b)
        W, b = params[1]
        return jax.nn.sigmoid(W @ A + b).T

    cumulative = create_copula(forward_fun)[0]

    params = []
    weights = jnp.array([[1.0, 1.0], [1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    weights = jnp.array([[1.0, 1.0]])
    bias = jnp.array([[1.0]])
    params.append((weights, bias))

    U = jnp.zeros(shape=(2, 1), dtype=jnp.float32)
    U = U.at[0].set(0.5)
    U = U.at[0].set(0.5)

    assert(cumulative(params, U) >= 0)
    assert(cumulative(params, U) <= 1)
