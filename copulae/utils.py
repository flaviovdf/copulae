# -*- coding: utf8 -*-


from .typing import Tensor
from .typing import PyTree

import jax
import jax.numpy as jnp

import scipy.stats as ss


@jax.jit
def ecdf(data: Tensor) -> PyTree:
    '''
    Returns an empirical cumulative distribution estimate
    for the given datapoints. Return values are not
    sorted. In order to plot one should do as follows:

    >>> from copulate.utils import ecdf
    >>> import jax.numpy as jnp
    >>> data = jnp.array([3.0, 1.0, 1.0, 7.0, -2.2])
    >>> ecdf_y = ecdf(data)
    >>> asort = data.argsort()
    >>> x_plot = data[asort]
    >>> y_plot = data[asort]
    >>> # from here you can plot x and y

    Arguments
    ---------
    data: Tensor (1 dimensional array-like)
        the data set to compute the ecdf

    Returns
    -------
    y: Tensor (1 dimensional jnp.array)
        the ecdf estimate for give datapoints
    '''
    xs = jnp.sort(data)
    n = data.shape[0]
    y = (jnp.searchsorted(xs, data, side="right") + 1) / n
    return y


def gauss_copula(
    u: Tensor,
    mean: Tensor,
    cov: Tensor
) -> Tensor:
    ppfs = ss.norm.ppf(u)
    return ss.multivariate_normal(
        mean=mean,
        cov=cov
    ).cdf(ppfs)
