# -*- coding: utf8 -*-
'''Unit tests for our kde estimates'''

from copulae.kde import kde_pdf

from numpy.testing import assert_almost_equal


import jax

import scipy.stats as ss


def test_pdf():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))
    kde_ss = ss.gaussian_kde(data)
    y_ss = kde_ss.pdf(data)
    y = kde_pdf(data)
    assert_almost_equal(y_ss, y)


def test_cdf():
    pass
