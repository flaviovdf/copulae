# -*- coding: utf8 -*-
'''Unit tests for input generators'''


from copulae.sm.ecdf import ECDF

from copulae.input import generate_copula_net_input

from copulae.utils import gauss_copula


from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


import jax
import jax.numpy as jnp


def test_generate_copula_net_input_shapes():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)

    ndim = 2
    nex = 1024
    D = jax.random.normal(subkey, shape=(ndim, nex))

    _, subkey = jax.random.split(key)
    del key

    U_batches, M_batches, X_batches, Y_batches = \
        generate_copula_net_input(
            subkey, D, n_batches=16, batch_size=8
        )

    assert(U_batches.shape[0] == 16)
    assert(U_batches.shape[1] == ndim)
    assert(U_batches.shape[2] == 8)

    assert(M_batches.shape[0] == 16)
    assert(M_batches.shape[1] == ndim)
    assert(M_batches.shape[2] == 8)

    assert(X_batches.shape[0] == 16)
    assert(X_batches.shape[1] == ndim)
    assert(X_batches.shape[2] == 8)

    assert(Y_batches.shape[0] == 16)
    assert(Y_batches.shape[1] == 8)
    assert(Y_batches.shape[2] == 1)


def test_generate_copula_net_input_values_1():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)

    ndim = 2
    nex = 1024
    D = jax.random.uniform(
        subkey, minval=2, maxval=3, shape=(ndim, nex)
    )

    _, subkey = jax.random.split(key)
    del key

    U_batches, M_batches, X_batches, Y_batches = \
        generate_copula_net_input(
            subkey, D, n_batches=16, batch_size=8
        )

    assert((U_batches <= 1).all())
    assert((U_batches >= 0).all())

    assert((M_batches <= 1).all())
    assert((M_batches >= 0).all())

    assert((X_batches <= 3).all())
    assert((X_batches >= 2).all())

    assert((Y_batches <= 1).all())
    assert((Y_batches >= 0).all())


def test_generate_copula_net_input_values_2():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)

    ndim = 2
    nex = 1024
    D = jax.random.uniform(
        subkey, minval=2, maxval=3, shape=(ndim, nex)
    )

    _, subkey = jax.random.split(key)
    del key

    U_batches, M_batches, X_batches, Y_batches = \
        generate_copula_net_input(
            subkey, D, n_batches=16, batch_size=8,
            min_val=-2, max_val=2,
        )

    assert((U_batches > 1).any())
    assert((U_batches < 0).any())

    assert((M_batches <= 1).all())
    assert((M_batches >= 0).all())

    assert((X_batches <= 3).all())
    assert((X_batches >= 2).all())

    assert((Y_batches <= 1).all())
    assert((Y_batches >= 0).all())


def test_Y_is_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    n_batches = 1
    batch_size = 4096

    # parameters for the synthetic copula
    rho = 0.65
    mean = jnp.zeros(2)
    E = jnp.zeros(shape=(2, 2)) + rho
    E = E.at[0, 0].set(1)
    E = E.at[1, 1].set(1)

    # dataset
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = jax.random.multivariate_normal(
        subkey, mean=mean, cov=E, shape=(10000, )
    ).T

    _, subkey = jax.random.split(key)
    U_batches, _, _, Y_batches = generate_copula_net_input(
        subkey, D, n_batches=n_batches,
        batch_size=batch_size
    )

    assert(U_batches.shape[0] == 1)
    assert(U_batches.shape[1] == 2)
    assert(U_batches.shape[2] == 4096)

    # get the expected values from the copula equation
    C_batches = jnp.zeros(shape=(n_batches, batch_size, 1))
    for batch_i in range(n_batches):
        Cb = gauss_copula(
            U_batches[batch_i].T, mean, E
        ).reshape(batch_size, 1)
        C_batches = C_batches.at[batch_i].set(Cb)

    assert_array_almost_equal(
        Y_batches.ravel(), C_batches.ravel(), 0.01
    )


def test_M_and_X_are_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    n_batches = 1
    batch_size = 4096

    # parameters for the synthetic copula
    rho = 0.65
    mean = jnp.zeros(2)
    E = jnp.zeros(shape=(2, 2)) + rho
    E = E.at[0, 0].set(1)
    E = E.at[1, 1].set(1)

    # dataset
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = jax.random.multivariate_normal(
        subkey, mean=mean, cov=E, shape=(10000, )
    ).T

    _, subkey = jax.random.split(key)
    _, M_batches, X_batches, _ = generate_copula_net_input(
        subkey, D, n_batches=n_batches,
        batch_size=batch_size
    )

    ecdf_0 = ECDF(D[0])
    ecdf_1 = ECDF(D[1])

    data_points_0 = X_batches[0][0]
    data_points_1 = X_batches[0][1]

    marginal_ecdfs_0 = M_batches[0][0]
    marginal_ecdfs_1 = M_batches[0][1]

    assert_array_equal(
        ecdf_0(data_points_0), marginal_ecdfs_0
    )
    assert_array_equal(
        ecdf_1(data_points_1), marginal_ecdfs_1
    )
