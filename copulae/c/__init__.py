# -*- coding: utf8 -*-
'''
Contains the base code to create neural copulas.
'''


from copulae.typing import Callable
from copulae.typing import PyTree
from copulae.typing import Tensor


import jax
import jax.numpy as jnp


class Copula(object):

    def cumulative(self, params, U):
        return self(params, U)

def create_copula(
    forward_fun: Callable,
):
    @jax.jit
    def C(
        params: PyTree,
        U: Tensor
    ) -> Tensor:
        return forward_fun(params, U)

    batched_C = jax.vmap(
        C,
        in_axes=(None, 0),
        out_axes=0
    )

    @jax.jit
    def partial_c(
        params: PyTree,
        U: Tensor
    ) -> Tensor:
        def j(params, u):
            u = jnp.atleast_2d(u).T
            jacobian = jax.jacobian(
                C, argnums=1
            )(
                params,
                u
            ).squeeze((1, -1)).T
            return jacobian
        return jax.vmap(j, in_axes=(None, 1))(params, U)

    batched_partial_c = jax.vmap(
        partial_c,
        in_axes=(None, 0),
        out_axes=0
    )

    @jax.jit
    def c(
        params: PyTree,
        U: Tensor
    ) -> Tensor:
        def h(params, u):
            hessian = jax.hessian(
                C,
                argnums=1
            )(
                params,
                jnp.atleast_2d(u).T
            ).ravel()[-2]
            return hessian
        return jax.vmap(h, in_axes=(None, 1))(params, U)

    batched_c = jax.vmap(
        c,
        in_axes=(None, 0),
        out_axes=0
    )

    return (C, batched_C,
            partial_c, batched_partial_c,
            c, batched_c)
