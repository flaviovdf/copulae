# -*- coding: utf8 -*-
'''Unit tests for our kde estimates'''

from numpy.testing import assert_almost_equal


import jax
import jax.numpy as jnp

import scipy.stats as ss


def test_pdf():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))
    skde = ss.gaussian_kde(data)
    pass


def test_cdf():
    pass
