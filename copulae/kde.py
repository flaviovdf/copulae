# -*- coding: utf8 -*-

import jax
import jax.numpy as jnp


@jax.jit
def scotts_method(n: int, d: int) -> float:
    return jnp.power(n, -1./(d+4))


@jax.jit
def silvermans_method(n: int, d: int) -> float:
    return jnp.power(n*(d+2.0)/4.0, -1./(d+4))


@jax.jit
def kde_pdf(
    x: jnp.array,
    bandwidth: float
) -> jnp.array:
    # distances (use broadcasting)
    rescaled_x = (x - x[:, jnp.newaxis]) / bandwidth

    # compute the gaussian kernel
    kernel = jnp.exp(- 0.5 * rescaled_x ** 2)
    kernel /= jnp.sqrt(2 * jnp.pi)

    # rescale
    return kernel.sum(axis=0) / x.shape[0] / bandwidth


@jax.jit
def kde_cdf(
    x: jnp.ndarray,
    bandwidth: float
) -> jnp.ndarray:
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
