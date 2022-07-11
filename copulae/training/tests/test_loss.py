# -*- coding: utf8 -*-
'''Unit tests for loss functions'''


from copulae.loss import l1
from copulae.loss import l2


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
