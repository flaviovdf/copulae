# -*- coding: utf8 -*-
'''Unit tests for our kde estimates'''


from copulae.mlp import init_mlp

from numpy.testing import assert_equal


import jax


def test_init_mlp():
    key = jax.random.PRNGKey(30091985)
    new_key, params = init_mlp(key, 2, 3, 8, 1)
    assert_equal(4, len(params))

    weights, bias = params[0]
    assert((weights != 0).any())
    assert(weights.shape[0] == 8)
    assert(weights.shape[1] == 3)
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
    assert((bias == 1).all())

    assert(new_key != key)

def test_init_mlp_2():
    key = jax.random.PRNGKey(30091985)
    new_key, params = init_mlp(key, 2, 2, 4, 0)
    assert_equal(3, len(params))

    weights, bias = params[0]
    assert((weights != 0).any())
    assert(weights.shape[0] == 4)
    assert(weights.shape[1] == 2)
    assert((bias == 1).all())

    for weights, bias in params[1:-1]:
        assert((weights != 0).any())
        assert(weights.shape[0] == 4)
        assert(weights.shape[1] == 4)
        assert((bias == 0).all())

    weights, bias = params[-1]
    assert((weights != 0).any())
    assert(weights.shape[0] == 1)
    assert(weights.shape[1] == 4)
    assert((bias == 0).all())

    assert(new_key != key)