# -*- coding: utf8 -*-
'''
This module is used to generate inputs for copula
neural networks.
'''


from collections import namedtuple

from copulae.typing import Sequence
from copulae.typing import Tensor
from copulae.typing import Tuple

from scipy.stats import gaussian_kde

import numpy as np


def __ecdf(x):
    xs = np.zeros(x.shape[0] + 1, dtype=x.dtype)
    ys = np.zeros_like(xs)
    xs[0] = x.min() - 1e-6
    xs[1:] = np.sort(x)
    ys[1:] = np.arange(1, x.shape[0] + 1) / x.shape[0]
    return xs, ys


def __nn_cond(
    dim1_key, dim2_key, dim1_vals, dim2_vals, C
):
    k1 = np.searchsorted(dim1_vals, dim1_key)
    k2 = np.searchsorted(dim2_vals, dim2_key)
    return C[k1, k2]


def __populate_conditionals(
    U_batches, C_batches,
    probs, keys_v, keys_u, dim_v, dim_u
):
    for i in range(U_batches.shape[0]):
        for k in range(U_batches.shape[2]):
            v = U_batches[i, dim_v, k]
            u = U_batches[i, dim_u, k]
            v_idx = np.searchsorted(keys_v, v)

            if v_idx >= len(probs):
                c_vu = 1.0
            else:
                u_idx = np.searchsorted(keys_u[v_idx], u)
                if u_idx >= probs[v_idx].shape[0]:
                    c_vu = 1.0
                else:
                    c_vu = probs[v_idx][u_idx]
            C_batches[i, dim_v, k] = c_vu


def __create_conditionals(x_us, us, vs, data_u):
    x_us = np.array(x_us)
    us = np.array(us)
    vs = np.array(vs)
    data_u = np.array(data_u)

    order_it = np.searchsorted(x_us, data_u)
    us = us[order_it]
    vs = vs[order_it]

    probs = []
    keys_u = []
    keys_v = []
    for i in range(us.shape[0]):
        v = vs[i]
        idx = vs <= v
        to_kde = np.sort(us[idx])
        if to_kde.shape[0] >= 3:
            kde = gaussian_kde(to_kde)
            base = kde.pdf(to_kde)
            probs.append(base * v)
        else:
            probs.append([0] * i)
        keys_u.append(to_kde)
        keys_v.append(v)

    keys_v_np = np.array(keys_v)
    order = np.argsort(keys_v_np)

    keys_u_np_unordered = list(map(np.array, keys_u))
    probs_np_unordered = list(map(np.array, probs))

    keys_u_np = [keys_u_np_unordered[i] for i in order]
    probs_np = [probs_np_unordered[i] for i in order]

    return probs_np, keys_v_np, keys_u_np


def __init_output(n_batches, n_features, batch_size):
    # U is used for the copula training
    # M are the marginal CDFs
    # X are the dataset values related to M
    # Y is the expected copula output
    U_batches = np.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=np.float32
    )
    M_batches = np.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=np.float32
    )
    X_batches = np.zeros(
        shape=(n_batches, n_features, batch_size),
        dtype=np.float32
    )
    Y_batches = np.zeros(
        shape=(n_batches, batch_size, 1),
        dtype=np.float32
    )
    return U_batches, M_batches, X_batches, Y_batches


def __populate(
    D: Tensor,
    bootstrap: bool,
    ecdfs: Sequence[Tuple[Tensor, Tensor]],
    min_val: float,
    max_val: float,
    n_batches: int,
    batch_size: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

    n_features = D.shape[0]
    U_batches, M_batches, X_batches, Y_batches = \
        __init_output(n_batches, n_features, batch_size)

    for batch_i in range(n_batches):
        if bootstrap:
            Ub = np.random.uniform(
                size=(n_features, batch_size),
                low=min_val, high=max_val
            )
        else:
            Ub = np.zeros(
                shape=(n_features, batch_size),
                dtype=np.float32
            )
            for j, xy in enumerate(ecdfs):
                xs = xy[0]
                ys = xy[1]
                idx = np.searchsorted(xs, D[j])
                Ub[j] = ys[idx]

        mask = True
        for j, xy in enumerate(ecdfs):
            pos = np.searchsorted(xy[1], Ub[j])
            vals_m = xy[1][pos]
            M_batches[batch_i, j, :] = vals_m

            vals_x = xy[0][pos]
            X_batches[batch_i, j, :] = vals_x

            lt = np.tile(
                D[j], batch_size
            ).reshape(
                batch_size,
                D.shape[1]
            ).T <= vals_x
            mask = mask & lt

        Yb = mask.mean(axis=0)
        Yb = Yb.reshape(batch_size, 1)

        U_batches[batch_i, :, :] = Ub
        Y_batches[batch_i, :, :] = Yb

    R_batches = np.random.uniform(
        low=min_val, high=max_val - U_batches,
        size=U_batches.shape
    )

    x_us = ecdfs[0][0]
    x_vs = ecdfs[1][0]
    us = ecdfs[0][1]
    vs = ecdfs[0][1]
    p_vu, keys_p_vu_v, keys_p_vu_u = \
        __create_conditionals(x_us, us, vs, D[0])
    p_uv, keys_p_uv_u, keys_p_uv_v = \
        __create_conditionals(x_vs, vs, us, D[1])

    C_batches = np.zeros_like(U_batches)
    __populate_conditionals(
        U_batches, C_batches,
        p_vu, keys_p_vu_v, keys_p_vu_u, 1, 0
    )
    __populate_conditionals(
        U_batches, C_batches,
        p_uv, keys_p_uv_u, keys_p_uv_v, 0, 1
    )

    return U_batches, M_batches, C_batches, R_batches, \
        X_batches, Y_batches


def generate_copula_net_input(
    D: Tensor,
    bootstrap: bool = True,
    n_batches: int = 128,
    batch_size: int = 64
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    '''
    Creates the input tensors needed to train neural
    copulas. If `bootstrap=True` then random inputs are
    generated by empirically sampling from the CDF of the
    data. If `bootstrap=False`, then a single batch is
    returned.

    See the notes below for differences when
    bootstrapping and when not bootstrapping

    **Bootstrapping (bootstrap=True)**

    The returned tensors will be organized in batches
    (a total of `n_batches`), each batch containing
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

    **No Bootstrap (bootstrap=False)**

    A single batch is returned with the size of the
    dataset. The arguments returned are the same as when
    bootstrapping.

    The arguments `min_val`, `max_val`, `n_batches` and
    `batch_size` are ignored when not bootstrapping.

    Arguments
    ---------
    D: Tensor
        Our dataset of (`n_dimensions`, `n_samples`)
    bootstrap: bool
        Bootstrap input or not
    n_batches: int
        Number of batches to generate
    batch_size: int
        The size of each batch

    Returns
    -------
    Six tensors in a namedtuple. Please use
    ReturnValue.NameOfTensor to get each one of the tensors
    below

    U_batches: Tensor (n_batches, n_dimensions, batch_size)
        The tensor that serves as input to train neural
        copulas.
    M_batches: Tensor (n_batches, n_dimensions, batch_size)
        Marginal cumulative distribution functions (ecdf)
        for each dimension. When bootstrap=False, this is
        the same as the U_batches tensor. When it is true,
        there may be some entries with differing, but very
        close, values. This is because the M_batches always
        contains seen data, whereas U are samples when
        bootstrap=True. Thus, M will be the closest value
        to U in the dataset.
    C_batches: Tensor (n_batches, n_dimensions, batch_size)
        Conditional CDFs of the form P[U <= u | V = v] and
        P[V <= v | U = u], u and v are cdf values for each
        dimension
    R_batches: Tensor (n_batches, n_dimensions, batch_size)
        Random width and heights to create rectangles where
        the left corner is the value on U_batches.
    X_batches: Tensor (n_batches, n_dimensions, batch_size)
        Data points associated with each marginal above.
    Y_batches: Tensor (n_batches, n_dimensions, batch_size)
        The output of the neural copula. A joint cumulative
        distribution estimate of the values in `X_batches`.
    '''

    if len(D.shape) != 2 or D.shape[0] != 2:
        raise ValueError('D must be of shape (2, n)')

    if not bootstrap:
        n_batches = 1
        batch_size = D.shape[1]

    ecdfs = []
    xs, ys = __ecdf(D[0])
    ecdfs.append((xs, ys))

    xs, ys = __ecdf(D[1])
    ecdfs.append((xs, ys))

    assert np.all(ecdfs[0][0][:-1] <= ecdfs[0][0][1:])
    assert np.all(ecdfs[0][1][:-1] <= ecdfs[0][1][1:])
    assert np.all(ecdfs[1][0][:-1] <= ecdfs[1][0][1:])
    assert np.all(ecdfs[1][1][:-1] <= ecdfs[1][1][1:])

    TrainingTensors = namedtuple(
        'TrainingTensors',
        ['U_batches', 'M_batches', 'C_batches',
         'R_batches', 'X_batches', 'Y_batches']
    )

    rv = __populate(
        D, bootstrap, ecdfs, 0, 1.0, n_batches, batch_size
    )
    return TrainingTensors(*rv)
