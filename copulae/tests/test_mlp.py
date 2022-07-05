# -*- coding: utf8 -*-
'''Unit tests for our kde estimates'''


from copulae.mlp import init_mlp
from copulae.mlp import mlp

from numpy.testing import assert_equal


import jax
import jax.numpy as jnp


def test_init_mlp():
    key = jax.random.PRNGKey(30091985)
    _, params = init_mlp(key, 2, 3, 8, 1)
    assert_equal(4, len(params))

    weights, bias = params[0]
    assert((weights != 0).any())
    assert(weights.shape[0] == 8)
    assert(weights.shape[1] == 2)
    assert((bias == 1).all())

    for weights, bias in params[1:-1]:
        assert((weights != 0).any())
        assert(weights.shape[0] == 8)
        assert(weights.shape[1] == 8)
        assert((bias == 1).all())

    weights, bias = params[-1]
    assert((weights != 0).any())
    assert(weights.shape[0] == 1)
    assert(weights.shape[1] == 8)
    assert(bias.shape[0] == 1)
    assert(bias.shape[1] == 1)
    assert((bias == 1).all())


def test_init_mlp_2():
    key = jax.random.PRNGKey(30091985)
    _, params = init_mlp(key, 3, 2, 4, 0)
    assert_equal(3, len(params))

    weights, bias = params[0]
    assert((weights != 0).any())
    assert(weights.shape[0] == 4)
    assert(weights.shape[1] == 3)
    assert((bias == 0).all())

    for weights, bias in params[1:-1]:
        assert((weights != 0).any())
        assert(weights.shape[0] == 4)
        assert(weights.shape[1] == 4)
        assert((bias == 0).all())

    weights, bias = params[-1]
    assert((weights != 0).any())
    assert(weights.shape[0] == 1)
    assert(weights.shape[1] == 4)
    assert(bias.shape[0] == 1)
    assert(bias.shape[1] == 1)
    assert((bias == 0).all())


def test_mlp_1():
    key = jax.random.PRNGKey(30091985)
    X = jnp.array([[1, 2, 3],
                   [7, 8, 9]], dtype=jnp.float32)
    _, params = init_mlp(key, X.shape[0], 8, 8, 1)
    output = mlp(params, X)
    assert((output <= 1).all())
    assert((output >= 0).all())
    assert(output.shape[0] == 3)
    assert(output.shape[1] == 1)