# -*- coding: utf8 -*-
'''
This module is used to generate inputs for copula
neural networks
'''


from copulae.typing import Tensor
from copulae.typing import Tuple

from copulae.sm.ecdf import ECDF

import jax
import jax.numpy as jnp


def generate_copula_net_input(
    key: jax.random.PRNGKey,
    D: Tensor,
    min_val: int = 0,
    max_val: int = 1,
    n_batches: int = 128,
    batch_size: int = 64
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:

    n_features = D.shape[0]
    ecdfs = []
    for j in range(n_features):
        ecdf = ECDF(D[j])
        ecdfs.append((ecdf.x, ecdf.y))

    # U is used for the copula training
    # M and X are the marginal CDFs used for regularization
    U_batches = jnp.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=jnp.float32
    )
    M_batches = jnp.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=jnp.float32
    )
    X_batches = jnp.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=jnp.float32
    )
    Y_batches = jnp.zeros(
        shape=(n_batches, batch_size, 1),
        dtype=jnp.float32
    )

    for batch_i in range(n_batches):
        Ub = jax.random.uniform(
            key, shape=(n_features, batch_size),
            minval=min_val, maxval=max_val
        )

        mask = True
        for j, xy in enumerate(ecdfs):
            pos = jnp.searchsorted(
                xy[1], Ub[j]
            )

            vals_m = xy[1][pos]
            M_batches = \
                M_batches.at[batch_i, j, :].set(vals_m)

            vals_x = xy[0][pos]
            X_batches = \
                X_batches.at[batch_i, j, :].set(vals_x)

            lt = jnp.tile(
                D[j], batch_size
            ).reshape(
                D.shape[1],
                batch_size
            ) <= vals_x
            mask = mask & lt

        Yb = mask.mean(axis=0)
        Yb = Yb.reshape(batch_size, 1)

        U_batches = U_batches.at[batch_i].set(Ub)
        Y_batches = Y_batches.at[batch_i].set(Yb)

    return U_batches, M_batches, X_batches, Y_batches
