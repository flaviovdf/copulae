# -*- coding: utf8 -*-
'''Unit tests for our kde estimates'''

from copulae.kde import scotts_method
from copulae.kde import silvermans_method
from copulae.kde import kde_cdf
from copulae.kde import kde_pdf

from numpy.testing import assert_almost_equal


import jax

import scipy.stats as ss


def test_pdf_silverman():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))

    kde_ss = ss.gaussian_kde(data, bw_method='silverman')

    bw = silvermans_method(data.shape[0], 1)
    y_ss = kde_ss.pdf(data, bw)

    y = kde_pdf(data)
    assert_almost_equal(y_ss, y)


def test_pdf_scotts():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))

    kde_ss = ss.gaussian_kde(data, bw_method='scott')

    bw = scotts_method(data.shape[0], 1)
    y_ss = kde_ss.pdf(data, bw)

    y = kde_pdf(data)
    assert_almost_equal(y_ss, y)


def test_cdf_silverman():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))

    kde_ss = ss.gaussian_kde(data, bw_method='silverman')

    bw = silvermans_method(data.shape[0], 1)
    y_ss = kde_ss.cdf(data, bw)

    y = kde_cdf(data)
    assert_almost_equal(y_ss, y)


def test_cdf_scotts():
    key = jax.random.PRNGKey(30091985)
    _, key = jax.random.split(key)
    data = jax.random.normal(key, shape=(100, ))

    kde_ss = ss.gaussian_kde(data, bw_method='scott')

    bw = scotts_method(data.shape[0], 1)
    y_ss = kde_ss.cdf(data, bw)

    y = kde_cdf(data)
    assert_almost_equal(y_ss, y)
