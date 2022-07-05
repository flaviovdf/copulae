# -*- coding: utf8 -*-
'''Unit tests for input generators'''


from copulae.input import generate_copula_net_input


import jax


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
