# -*- coding: utf8 -*-
'''Unit tests for input generators'''


from copulae.np_input import generate_copula_net_input

from copulae.utils import gauss_copula

from numpy.testing import assert_
from numpy.testing import assert_array_almost_equal


from statsmodels.distributions.empirical_distribution \
    import ECDF


import numpy as np


def seeded(f):
    np.random.seed(30091985)
    return f()


@seeded
def test_generate_copula_net_input_shapes():
    ndim = 2
    nex = 1024
    D = np.random.normal(size=(ndim, nex))

    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    U_batches = InputTensors.U_batches
    M_batches = InputTensors.M_batches
    C_batches = InputTensors.C_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    Y_batches = InputTensors.Y_batches

    assert_(U_batches.shape[0] == 16)
    assert_(U_batches.shape[1] == ndim)
    assert_(U_batches.shape[2] == 8)

    assert_(M_batches.shape[0] == 16)
    assert_(M_batches.shape[1] == ndim)
    assert_(M_batches.shape[2] == 8)

    assert_(C_batches.shape[0] == 16)
    assert_(C_batches.shape[1] == ndim)
    assert_(C_batches.shape[2] == 8)

    assert_(R_batches.shape[0] == 16)
    assert_(R_batches.shape[1] == ndim)
    assert_(R_batches.shape[2] == 8)

    assert_(X_batches.shape[0] == 16)
    assert_(X_batches.shape[1] == ndim)
    assert_(X_batches.shape[2] == 8)

    assert_(Y_batches.shape[0] == 16)
    assert_(Y_batches.shape[1] == 8)
    assert_(Y_batches.shape[2] == 1)


@seeded
def test_generate_copula_net_input_values_1():
    ndim = 2
    nex = 1024
    D = np.random.uniform(
        low=2, high=3, size=(ndim, nex)
    )

    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    U_batches = InputTensors.U_batches
    M_batches = InputTensors.M_batches
    C_batches = InputTensors.C_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    Y_batches = InputTensors.Y_batches

    assert_((U_batches <= 1).all())
    assert_((U_batches >= 0).all())

    assert_((M_batches <= 1).all())
    assert_((M_batches >= 0).all())

    assert_((C_batches <= 1).all())
    assert_((C_batches >= 0).all())

    assert_((R_batches <= 1).all())
    assert_((R_batches >= 0).all())

    assert_((X_batches <= 3).all())
    assert_((X_batches >= 2).all())

    assert_((Y_batches <= 1).all())
    assert_((Y_batches >= 0).all())


@seeded
def test_generate_copula_net_input_values_2():
    ndim = 2
    nex = 1024
    D = np.random.uniform(
        low=2, high=3, size=(ndim, nex)
    )

    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    U_batches = InputTensors.U_batches
    M_batches = InputTensors.M_batches
    C_batches = InputTensors.C_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    Y_batches = InputTensors.Y_batches

    assert_((U_batches <= 1).all())
    assert_((U_batches >= 0).all())

    assert_((M_batches <= 1).all())
    assert_((M_batches >= 0).all())

    assert_((C_batches <= 1).all())
    assert_((C_batches >= 0).all())

    assert_((R_batches <= 1).all())
    assert_((R_batches >= 0).all())

    assert_((X_batches <= 3).all())
    assert_((X_batches >= 2).all())

    assert_((Y_batches <= 1).all())
    assert_((Y_batches >= 0).all())


@seeded
def test_Y_is_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    n_batches = 1
    batch_size = 4096

    # parameters for the synthetic copula
    rho = 0.65
    mean = np.zeros(2)
    E = np.zeros(shape=(2, 2)) + rho
    E[0, 0] = 1
    E[1, 1] = 1

    # dataset
    D = np.random.multivariate_normal(
        mean=mean, cov=E, size=(10000, )
    ).T

    InputTensors = generate_copula_net_input(
        D, n_batches=n_batches,
        batch_size=batch_size
    )

    assert_(InputTensors.U_batches.shape[0] == 1)
    assert_(InputTensors.U_batches.shape[1] == 2)
    assert_(InputTensors.U_batches.shape[2] == 4096)

    # get the expected values from the copula equation
    E_batches = np.zeros(shape=(n_batches, batch_size, 1))
    for batch_i in range(n_batches):
        Eb = gauss_copula(
            InputTensors.U_batches[batch_i].T, mean, E
        ).reshape(batch_size, 1)
        E_batches[batch_i] = Eb

    assert_array_almost_equal(
        InputTensors.Y_batches.ravel(), E_batches.ravel(),
        0.01
    )


@seeded
def test_M_and_X_are_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    n_batches = 1
    batch_size = 4096

    # parameters for the synthetic copula
    rho = 0.65
    mean = np.zeros(2)
    E = np.zeros(shape=(2, 2)) + rho
    E[0, 0] = 1
    E[1, 1] = 1

    # dataset
    D = np.random.multivariate_normal(
        mean=mean, cov=E, size=(10000, )
    ).T

    InputTensors = generate_copula_net_input(
        D, n_batches=n_batches,
        batch_size=batch_size
    )

    ecdf_0 = ECDF(D[0])
    ecdf_1 = ECDF(D[1])

    data_points_0 = InputTensors.X_batches[0][0]
    data_points_1 = InputTensors.X_batches[0][1]

    marginal_ecdfs_0 = InputTensors.M_batches[0][0]
    marginal_ecdfs_1 = InputTensors.M_batches[0][1]

    assert_array_almost_equal(
        ecdf_0(data_points_0), marginal_ecdfs_0, 4
    )
    assert_array_almost_equal(
        ecdf_1(data_points_1), marginal_ecdfs_1, 4
    )
