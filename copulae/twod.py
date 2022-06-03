# -*- coding: utf8 -*-


from .typing import Tensor
from .typing import Tuple
from .typing import PyTree

import jax
import jax.numpy as jnp


@jax.jit
def ecdf(data: Tensor) -> PyTree:
    x = jnp.sort(data)
    n = data.shape[0]
    y = (jnp.searchsorted(x, x, side="right") + 1) / n
    return x, y


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


def init_mlp(
    key: jax.random.PRNGKey,
    input_size: int,
    n_layers: int,
    layer_width: int,
    b_init: int = 0,
) -> PyTree:

    initializer = jax.nn.initializers.lecun_normal()
    params = []
    new_key, *subkeys = jax.random.split(key, n_layers + 2)

    W = initializer(subkeys[0], (layer_width, input_size), jnp.float32)
    b = jnp.zeros(shape=(layer_width, 1)) + b_init
    params.append((W, b))

    for i in range(1, n_layers):
        W = initializer(subkeys[i], (layer_width, layer_width), jnp.float32)
        b = jnp.zeros(shape=(layer_width, 1)) + b_init
        params.append((W, b))

    W = initializer(subkeys[-1], (1, layer_width), jnp.float32)
    b = jnp.zeros(shape=(1, 1)) + b_init
    params.append((W, b))

    return params, new_key


def gauss_copula(u: Tensor, mean: Tensor, E: Tensor) -> Tensor:

    import scipy.stats as ss

    ppfs = ss.norm.ppf(u)
    return ss.multivariate_normal(mean=mean, cov=E).cdf(ppfs)


def generate_copula_net_input(
    key: jax.random.PRNGKey, D: Tensor, n_batches: int = 128, batch_size: int = 64
) -> Tuple[Tensor, Tensor]:

    n_features = D.shape[1]
    ecdfs = []
    for j in range(n_features):
        x, y = ecdf(D[:, j])
        ecdfs.append((x, y))

    # U is used for the copula training
    # M and X are the marginal CDFs used for regularization
    U_batches = jnp.zeros(shape=(n_batches, n_features, batch_size))
    M_batches = jnp.zeros(shape=(n_batches, n_features, batch_size))
    X_batches = jnp.zeros(shape=(n_batches, n_features, batch_size))
    Y_batches = jnp.zeros(shape=(n_batches, batch_size, 1))

    for batch_i in range(n_batches):
        key, subkey = jax.random.split(key)
        Ub = jax.random.uniform(
            subkey, shape=(n_features, batch_size), minval=-1.2, maxval=1.2
        )

        mask = True
        for j, xy in enumerate(ecdfs):
            pos = jnp.searchsorted(xy[1], Ub[j])
            vals_m = xy[1][pos]
            M_batches = M_batches.at[batch_i, j, :].set(vals_m)

            vals_x = xy[0][pos]
            X_batches = X_batches.at[batch_i, j, :].set(vals_x)

            lt = jnp.tile(D[:, j], batch_size).reshape(D.shape[0], batch_size) <= vals_x
            mask = mask & lt

        Yb = mask.mean(axis=0)
        Yb = Yb.reshape(batch_size, 1)

        U_batches = U_batches.at[batch_i].set(Ub)
        Y_batches = Y_batches.at[batch_i].set(Yb)

    return U_batches, M_batches, X_batches, Y_batches
