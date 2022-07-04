# -*- coding: utf8 -*-


from .typing import Tensor
from .typing import Tuple

from .utils import ecdf

import jax
import jax.numpy as jnp


def generate_copula_net_input(
    key: jax.random.PRNGKey,
    dataset: Tensor,
    n_batches: int = 128,
    batch_size: int = 64
) -> Tuple[Tensor, Tensor]:

    n_features = dataset.shape[1]
    ecdfs = []
    for j in range(n_features):
        x, y = ecdf(dataset[:, j])
        asort = x.argosort()
        x = x[asort]
        y = y[asort]
        ecdfs.append((x, y))

    # U is used for the copula training
    # M and X are the marginal CDFs used for regularization
    U_batches = jnp.zeros(
        shape=(n_batches, n_features, batch_size)
    )
    M_batches = jnp.zeros(
        shape=(n_batches,   n_features, batch_size))
    X_batches = jnp.zeros(
        shape=(n_batches, n_features, batch_size))
    Y_batches = jnp.zeros(
        shape=(n_batches, batch_size, 1))

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


@jax.jit
def cross_entropy(
    Y: Tensor,
    logits: Tensor
) -> Tensor:
    logit = jnp.clip(logits, 1e-6, 1 - 1e-6)
    Y = jnp.clip(Y, 0, 1)
    return jnp.mean(
        -Y * jnp.log(logit) - (1 - Y) * jnp.log(1 - logit)
    )
