# -*- coding: utf8 -*-


from .typing import Tensor
from .typing import PyTree

import jax
import jax.numpy as jnp


@jax.jit
def C(params: PyTree, U: Tensor) -> Tensor:

    a = jnp.clip(U, 0, 1)  # map input to [0, 1]
    for W, b in params[:-1]:
        z = jnp.dot(W, a) + b
        a = jax.nn.swish(z)

    W, b = params[-1]
    z = jnp.dot(W, a) + b
    return jax.nn.sigmoid(z).T


batched_C = jax.vmap(C, in_axes=(None, 0), out_axes=0)


@jax.jit
def partial_c(params: PyTree, U: Tensor) -> Tensor:
    def j(params, u):
        u = jnp.atleast_2d(u).T
        jacobian = jax.jacobian(C, argnums=1)(params, u).squeeze((1, -1)).T
        return jacobian

    return jax.vmap(j, in_axes=(None, 1))(params, U)


batched_partial_c = jax.vmap(partial_c, in_axes=(None, 0), out_axes=0)


@jax.jit
def c(params: PyTree, U: Tensor) -> Tensor:
    def h(params, u):
        hessian = jax.hessian(C, argnums=1)(params, jnp.atleast_2d(u).T).ravel()[-2]
        return hessian

    return jax.vmap(h, in_axes=(None, 1))(params, U)


batched_c = jax.vmap(c, in_axes=(None, 0), out_axes=0)


@jax.jit
def pdf(params: PyTree, X: Tensor) -> Tensor:

    ecdf_0_x, ecdf_0_y = ecdf(X[0])
    ecdf_1_x, ecdf_1_y = ecdf(X[1])

    def F0(data):
        return jnp.interp(data, ecdf_0_x, ecdf_0_y)

    def F1(data):
        return jnp.interp(data, ecdf_1_x, ecdf_1_y)

    def _C(x):
        u = jnp.array([[F0(x[0, 0]), F0(x[1, 0])]]).T
        return C(params, u)

    def h(params, x):
        return jax.hessian(_C)(jnp.atleast_2d(x).T).ravel()[-2]

    p = jax.vmap(h, in_axes=(None, 1))(params, X)
    p = jnp.nan_to_num(p, 0)
    return jnp.clip(p, 1e-6)


batched_pdf = jax.vmap(pdf, in_axes=(None, 0), out_axes=0)


@jax.jit
def cross_entropy(Y: Tensor, logits: Tensor) -> Tensor:
    logit = jnp.clip(logits, 1e-6, 1 - 1e-6)
    Y = jnp.clip(Y, 0, 1)
    return jnp.mean(-Y * jnp.log(logit) - (1 - Y) * jnp.log(1 - logit))


@jax.jit
def C_forward(
    params: PyTree,
    U_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    key: jax.random.PRNGKey,
    alpha: float,
    beta: float,
    gamma: float,
    omega: float,
    tau: float,
) -> Tensor:

    # 1. Basic loss on empirical cdf
    logits = batched_C(params, U_batches)
    loss = cross_entropy(Y_batches, logits)

    # 2. pdf of the data
    data_lhood = batched_pdf(params, X_batches)
    loss += -(jnp.log(data_lhood).mean()) * alpha

    # 3. L2 Regularization
    loss += jnp.array(jax.tree_map(lambda p: (p**2).sum(), params)).sum() * beta

    # 4. Frechet bounds loss
    #    L: max(u + v - 1, 0)
    #    R: min(u, v)
    L = jnp.clip(U_batches.sum(axis=1) - 1, 0)
    R = jnp.min(U_batches, axis=1)
    logits = logits.squeeze(-1)  # same dim as L, and R

    #   -1 * sign --> penalizes the negative values, goes to +1
    loss += ((-1 * jnp.sign(logits - L)) + 1).mean() * gamma * 0.5
    loss += ((-1 * jnp.sign(R - logits)) + 1).mean() * gamma * 0.5

    # 5. Partial derivative loss
    #    First derivative must be >= 0
    dC = batched_partial_c(params, U_batches)
    # loss += ((-1 * jnp.sign(dC) + 1)).mean() * omega * 0.5
    loss += (dC < 0).mean() * omega  # * 0.5

    # 6. First derivative must be <= 1
    loss += (dC > 1).mean() * omega  # * 0.5

    # 7. Second derivative loss
    #    Second derivative must be >= 0
    ddC = batched_c(params, U_batches)
    # loss += ((-1 * jnp.sign(ddC) + 1)).mean() * tau * 0.5
    loss += (ddC < 0).mean() * omega  # * 0.5

    return loss


C_grad_fn = jax.grad(C_forward)
