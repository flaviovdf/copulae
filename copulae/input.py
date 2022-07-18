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


def __init_output(n_batches, n_features, batch_size):
    # U is used for the copula training
    # M are the marginal CDFs used for regularization
    # X are the X values related to M
    # Y is the expected copula output
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
    return M_batches, X_batches, U_batches, Y_batches


def generate_copula_net_input(
    key: jax.random.PRNGKey,
    D: Tensor,
    min_val: int = 0,
    max_val: int = 1,
    n_batches: int = 128,
    batch_size: int = 64
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    '''
    Creates the input tensors needed to trai neural
    copulas. These tensors will be organized in
    batches (a total of `n_batches`), each batch containing
    `batch_size` elements.

    Batches are created using a bootstrap sampling
    procedure which generates samples based on the
    empirical cumulative distribution function (ecdf).

    The steps to produce the batches are as follows:
    1. For each dimension in the dataset `D`:
        a. Generate `batch_size` random numbers uniformly
           in [0, 1].
        b. For each of the numbers above, sample
           `batch_size` data points for that by
           sampling from the ecdf of the dimension.
        c. Store the uniform value from (a) in U; the
           data points in X; the ecdf of X in M
           (for marginal);
    2. and, finally store the joint ecdf for every
       dimension in Y.

    Steps (1) and (2) generate a single batch.

    Arguments
    ---------
    key: jax.random.PRNGKey
        The key used for random number generation, must be
        discarded afterwards.
    D: Tensor
        Our dataset of (`n_dimensions`, `n_samples`)
    min_val: int (defaults to 0)
    max_val: int (defaults to 1)
        Min and max values are used to generate the uniform
        random values used as input to the copula. By
        definition a copula receives only values in [0, 1]
        as input. However, we may sample values out of this
        range in order to train corner cases.
    n_batches: int
        Number of batches to generate
    batch_size: int
        The size of each batch

    Returns
    -------
    Four tensors:

    U_batches: Tensor of shape
               (n_batches, n_dimensions, batch_size)
        The tensor that serves as input to trai neural
        copulas.
    M_batches: Tensor
                (n_batches, n_dimensions, batch_size)
        Marginal cumulative distribution functions (ecdf)
        for each dimension.
    X_batches: Tensor
                (n_batches, n_dimensions, batch_size)
        Data points associated with each marginal above.
    Y_batches: Tensor
                (n_batches, n_dimensions, batch_size)
        The output of the neural copula. A joint cumulative
        distribution estimate of the values in `X_batches`.
    '''
    n_features = D.shape[0]
    ecdfs = []
    for j in range(n_features):
        ecdf = ECDF(D[j], side='right')
        ecdfs.append((ecdf.x, ecdf.y))

    @jax.jit
    def populate():
        for batch_i in range(n_batches):
            keys = jax.random.split(key, n_batches)
            M_batches, X_batches, U_batches, Y_batches = \
                __init_output(
                    n_batches, n_features, batch_size)

            Ub = jax.random.uniform(
                keys[batch_i],
                shape=(n_features, batch_size),
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
    return populate()
