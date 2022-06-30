# -*- coding: utf8 -*-
'''Unit tests for the utilities module'''

from ..utils import ecdf

from statsmodels.distributions.empirical_distribution \
    import ECDF

from numpy.testing import assert_almost_equal


import jax.numpy as jnp


def test_ecdf():
    data = jnp.array([2, 1, 4, 0, 5, 6, 7, 1, 2, 4, 5, 2])
    ecdf_y = ecdf(data)

    ecdf_sm = ECDF(data)
    assert_almost_equal(ecdf_sm(data), ecdf_y)
