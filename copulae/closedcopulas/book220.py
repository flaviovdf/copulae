# -*- coding: utf8 -*-
'''
Copulas of the form:

C(u, v) = uv / (u + v - uv)

Have a closed form partial of:

c_u(v) =  (v / (u + v - uv))**2
c_v(u) =  (u / (v + u - vu))**2

and inverse

u ~ uniform(0, 1)
t ~ uniform(0, 1)
v = i_u(t) = u * sqrt(t) / (1 - (1 - u) sqrt(t))

v ~ uniform(0, 1)
t ~ uniform(0, 1)
u = i_v(t) = v * sqrt(t) / (1 - (1 - v) sqrt(t))

See example 2.20 of "An introduction to copulas"

We shall use these in several of our tests
'''


from copulae.typing import PRNGKey
from copulae.typing import Tensor


import jax
import jax.numpy as jnp


import numpy as np


def C(u: Tensor, v: Tensor) -> Tensor:
    '''
    Evaluates the copula at u, v. This will be
    equivalent to the bivariate CDF.

    Arguments
    ---------
    u, v: numbers in [0, 1]. We do not check this
          so if you feed numbers out of this range
          expect errors.

    Returns
    -------
    The copula evaluated a u, v
    '''
    return u * v / (u + v - u * v)


def dCdv(u: Tensor, v: Tensor) -> Tensor:
    '''
    Evaluates the first derivative dC(u,v)/dv at u, v.

    Arguments
    ---------
    u, v: numbers in [0, 1]. We do not check this
          so if you feed numbers out of this range
          expect errors.

    Returns
    -------
    The first derivative
    '''
    return (u / (u + v - u * v)) ** 2


def dCdu(u: Tensor, v: Tensor) -> Tensor:
    '''
    Evaluates the first derivative dC(u,v)/du at u, v.

    Arguments
    ---------
    u, v: numbers in [0, 1]. We do not check this
          so if you feed numbers out of this range
          expect errors.

    Returns
    -------
    The first derivative
    '''
    return dCdv(v, u)


def c(u: Tensor, v: Tensor) -> Tensor:
    '''
    Evaluates the copula density u, v.

    Arguments
    ---------
    u, v: numbers in [0, 1]. We do not check this
          so if you feed numbers out of this range
          expect errors.

    Returns
    -------
    The copula density
    '''
    num = -2.0 * u * v
    den = (u * (v - 1) - v)**3
    return num / den


def sample(
    key: PRNGKey, size: int, to_numpy=False
) -> Tensor:
    '''
    Generates random data points following the algorithm:

    u ~ uniform(0, 1)
    t ~ uniform(0, 1)
    v = i_u(t) = u * sqrt(t) / (1 - (1 - u) sqrt(t))

    or

    v ~ uniform(0, 1)
    t ~ uniform(0, 1)
    u = i_v(t) = v * sqrt(t) / (1 - (1 - v) sqrt(t))

    See example 2.20 of "An introduction to copulas"

    Arguments
    ---------
    key: PRNGKey
        The random key to use for random number gen
    size: int
        The number of data points to use
    to_numpy: bool
        If the output should be converted to regular
        numpy as not jax numpy.

    Returns
    -------
    A matix of shape (2, n) with the random dataset
    '''
    s1, s2 = jax.random.split(key)
    us = jax.random.uniform(
        s1, minval=0, maxval=1, shape=(size, )
    )
    ts = jax.random.uniform(
        s2, minval=0, maxval=1, shape=(size, )
    )
    vs = us * jnp.sqrt(ts) / (1 - (1 - us) * jnp.sqrt(ts))

    d0 = 2 * us - 1
    d1 = -jnp.log(1 - vs)

    if to_numpy:
        return np.array([d0, d1])
    else:
        return jnp.array([d0, d1])
