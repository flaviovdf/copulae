# -*- coding: utf8 -*-
'''
Contains the base code to create neural copulas based on
automatic differentiation.
'''


from copulae.typing import Callable
from copulae.typing import PyTree
from copulae.typing import Tensor


import jax
import jax.numpy as jnp


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
            ).squeeze()
            return jacobian
        aux = jnp.swapaxes(
            jax.vmap(j, in_axes=(None, 1))(params, U),
            -2, -1
        )
        rv = jnp.zeros_like(aux)
        rv = rv.at[0].set(aux[1])
        rv = rv.at[1].set(aux[0])
        print(rv.shape)
        return rv

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

    return (batched_C, batched_partial_c, batched_c)
