# -*- coding: utf8 -*-
'''Unit tests for the utilities module'''


from copulae.utils import gauss_copula

from numpy.testing import assert_array_equal


import jax.numpy as jnp
import jax.scipy.stats as jss


def test_gausscopula():
    rho = 0
    mean = jnp.zeros(2)
    mean.at[1].set(2)

    E = jnp.zeros(shape=(2, 2)) + rho
    E = E.at[0, 0].set(1)
    E = E.at[1, 1].set(1)

    x1 = jss.norm.ppf(0.25, loc=0, scale=1)
    x2 = jss.norm.ppf(0.5, loc=2, scale=1)

    cdf1 = jss.norm.cdf(x1, loc=0, scale=1)
    cdf2 = jss.norm.cdf(x2, loc=2, scale=1)

    u = jnp.array([0.25, 0.5])
    assert_array_equal(cdf1 * cdf2,
                       gauss_copula(u, mean, E))
