# -*- coding: utf8 -*-
'''
1-D Gaussian Kernel Density Estimators

Employs code from
[jaxkern](https://github.com/IPL-UV/jaxkern).
'''


from copulae.typing import Tensor


import jax
import jax.numpy as jnp


@jax.jit
def scotts_method(n: int, d: int) -> float:
    '''
    Bandwidth estimator based on number of data points (n)
    and dimensions (d).
    '''
    return jnp.power(n, -1.0 / (d + 4))


@jax.jit
def silvermans_method(n: int, d: int) -> float:
    '''
    Bandwidth estimator based on number of data points (n)
    and dimensions (d).
    '''
    return jnp.power(n * (d + 2.0) / 4.0, -1.0 / (d + 4))


@jax.jit
def kde_pdf(
    x: Tensor,
    bandwidth: float
) -> Tensor:
    '''
    Computes a Gaussian KDE estimate of the probability
    density function (pdf) for the given dataset.

    Arguments
    ---------
    x: array like
        data to compute the pdf.
    bandwidth: float
        size of the bandwidth. For better results, make use
        of Scott's or Silverman's method which are provided
        with this module.

    Returns
    -------
    An array with the density of each x[i]. That is,
    pdf(x[i]).

    Examples
    --------

    >>> from copulae.kde import scotts_method
    >>> from copulae.kde import kde_pdf
    >>>
    >>> import jax
    >>>
    >>> _, key = jax.random.split(key)
    >>> key = jax.random.PRNGKey(30091985)
    >>>
    >>> data = jax.random.normal(key, shape=(100, ))
    >>>
    >>> # number of points and number of dimensions
    >>> bw = scotts_method(data.shape[0], 1)
    >>> pdf_data = kde_pdf(data, bw)
    '''
    # distances (use broadcasting)
    rescaled_x = (x - x[:, jnp.newaxis]) / bandwidth

    # compute the gaussian kernel
    kernel = jnp.exp(- 0.5 * rescaled_x ** 2)
    kernel /= jnp.sqrt(2 * jnp.pi)

    # rescale
    return kernel.sum(axis=0) / x.shape[0] / bandwidth


@jax.jit
def kde_cdf(
    x: Tensor,
    bandwidth: float
) -> jnp.ndarray:
    '''
    Computes a Gaussian KDE estimate of the cumulative
    density function (cdf) for the given dataset.

    Arguments
    ---------
    x: array like
        data to compute the cdf.
    bandwidth: float
        size of the bandwidth. For better results, make use
        of Scott's or Silverman's method which are provided
        with this module.

    Returns
    -------
    An array with the density of each x[i]. That is,
    cdf(x[i]).

    Examples
    --------

    >>> from copulae.kde import scotts_method
    >>> from copulae.kde import kde_cdf
    >>>
    >>> import jax
    >>>
    >>> _, key = jax.random.split(key)
    >>> key = jax.random.PRNGKey(30091985)
    >>>
    >>> data = jax.random.normal(key, shape=(100, ))
    >>>
    >>> # number of points and number of dimensions
    >>> bw = scotts_method(data.shape[0], 1)
    >>> cdf_data = kde_cdf(data, bw)
    '''

    n_samples = x.shape[0]

    # normalize samples
    low = (-jnp.inf - x[:, jnp.newaxis]) / bandwidth
    x = (x - x[:, jnp.newaxis]) / bandwidth

    # evaluate integral
    integral = jax.scipy.special.ndtr(x)
    integral -= jax.scipy.special.ndtr(low)

    # normalize distribution
    x_cdf = integral.sum(axis=0) / n_samples
    return x_cdf
