# -*- coding: utf8 -*-
'''
Tests for the ECDF port. Mostly a copy paste of
statsmodels tests
'''


from copulae.sm.ecdf import ECDF
from copulae.sm.ecdf import StepFunction

from numpy.testing import assert_raises
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_almost_equal

from statsmodels.distributions.empirical_distribution \
    import ECDF as smECDF


import jax
import jax.numpy as jnp


def test_step_function():
    x = jnp.arange(20)
    y = jnp.arange(20)
    f = StepFunction(x, y)
    assert_almost_equal(
        f(jnp.array([[3.2, 4.5], [24, -3.1], [3.0, 4.0]])),
        [[3, 4], [19, 0], [2, 3]]
    )


def test_step_function_bad_shape():
    x = jnp.arange(20)
    y = jnp.arange(21)
    assert_raises(ValueError, StepFunction, x, y)
    x = jnp.zeros((2, 2))
    y = jnp.zeros((2, 2))
    assert_raises(ValueError, StepFunction, x, y)


def test_step_function_value_side_right():
    x = jnp.arange(20)
    y = jnp.arange(20)
    f = StepFunction(x, y, side='right')
    assert_almost_equal(
        f(jnp.array([[3.2, 4.5], [24, -3.1], [3.0, 4.0]])),
        [[3, 4], [19, 0], [3, 4]]
    )


def test_step_function_repeated_values():
    x = [1, 1, 2, 2, 2, 3, 3, 3, 4, 5]
    y = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    f = StepFunction(x, y)

    assert_almost_equal(
        f(jnp.array([1, 2, 3, 4, 5])), [0, 7, 10, 13, 14]
    )
    f2 = StepFunction(x, y, side='right')
    assert_almost_equal(
        f2(jnp.array([1, 2, 3, 4, 5])), [7, 10, 13, 14, 15]
    )


def test_ecdf():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))
    print(data)

    ecdf = ECDF(data)
    ecdf_sm = smECDF(data)

    assert(ecdf.x[0] == data.min() - 1e-6)
    assert_array_almost_equal(ecdf_sm.x[1:], ecdf.x[1:])
    assert_array_almost_equal(ecdf_sm.y, ecdf.y)
    assert_array_almost_equal(ecdf_sm(data), ecdf(data))
