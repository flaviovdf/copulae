# -*- coding: utf8 -*-
'''Unit tests for input generators'''


from copulae.input import generate_copula_net_input

from copulae.utils import gauss_copula

from numpy.testing import assert_, assert_almost_equal
from numpy.testing import assert_array_almost_equal


from statsmodels.distributions.empirical_distribution \
    import ECDF


import numpy as np
import scipy.stats as ss


def test_generate_copula_net_input_shapes():
    np.random.seed(30091985)

    ndim = 2
    nex = 1024

    D = np.random.normal(size=(ndim, nex))
    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    UV_batches = InputTensors.UV_batches
    M_batches = InputTensors.M_batches
    YdC_batches = InputTensors.YdC_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    YC_batches = InputTensors.YC_batches

    assert_(UV_batches.shape[0] == 16)
    assert_(UV_batches.shape[1] == ndim)
    assert_(UV_batches.shape[2] == 8)

    assert_(M_batches.shape[0] == 16)
    assert_(M_batches.shape[1] == ndim)
    assert_(M_batches.shape[2] == 8)

    assert_(YdC_batches.shape[0] == 16)
    assert_(YdC_batches.shape[1] == ndim)
    assert_(YdC_batches.shape[2] == 8)

    assert_(R_batches.shape[0] == 16)
    assert_(R_batches.shape[1] == ndim)
    assert_(R_batches.shape[2] == 8)

    assert_(X_batches.shape[0] == 16)
    assert_(X_batches.shape[1] == ndim)
    assert_(X_batches.shape[2] == 8)

    assert_(YC_batches.shape[0] == 16)
    assert_(YC_batches.shape[1] == 8)
    assert_(YC_batches.shape[2] == 1)


def test_generate_copula_net_input_shapes_noboot():
    np.random.seed(30091985)

    ndim = 2
    nex = 1024
    D = np.random.normal(size=(ndim, nex))
    InputTensors = generate_copula_net_input(
        D, bootstrap=False
    )
    UV_batches = InputTensors.UV_batches
    M_batches = InputTensors.M_batches
    YdC_batches = InputTensors.YdC_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    YC_batches = InputTensors.YC_batches

    assert_(UV_batches.shape[0] == 1)
    assert_(UV_batches.shape[1] == ndim)
    assert_(UV_batches.shape[2] == 1024)

    assert_(M_batches.shape[0] == 1)
    assert_(M_batches.shape[1] == ndim)
    assert_(M_batches.shape[2] == 1024)

    assert_(YdC_batches.shape[0] == 1)
    assert_(YdC_batches.shape[1] == ndim)
    assert_(YdC_batches.shape[2] == 1024)

    assert_(R_batches.shape[0] == 1)
    assert_(R_batches.shape[1] == ndim)
    assert_(R_batches.shape[2] == 1024)

    assert_(X_batches.shape[0] == 1)
    assert_(X_batches.shape[1] == ndim)
    assert_(X_batches.shape[2] == 1024)

    assert_(YC_batches.shape[0] == 1)
    assert_(YC_batches.shape[1] == 1024)
    assert_(YC_batches.shape[2] == 1)


def test_generate_copula_net_input_values_1():
    np.random.seed(30091985)

    ndim = 2
    nex = 1024
    D = np.random.uniform(
        low=2, high=3, size=(ndim, nex)
    )

    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    UV_batches = InputTensors.UV_batches
    M_batches = InputTensors.M_batches
    YdC_batches = InputTensors.YdC_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    YC_batches = InputTensors.YC_batches

    assert_((UV_batches <= 1).all())
    assert_((UV_batches >= 0).all())

    assert_((M_batches <= 1).all())
    assert_((M_batches >= 0).all())

    assert_((YdC_batches <= 1).all())
    assert_((YdC_batches >= 0).all())

    assert_((R_batches <= 1).all())
    assert_((R_batches >= 0).all())

    assert_((X_batches <= 3).all())
    assert_((X_batches >= 2).all())

    assert_((YC_batches <= 1).all())
    assert_((YC_batches >= 0).all())


def test_generate_copula_net_input_values_2():
    np.random.seed(30091985)

    ndim = 2
    nex = 1024
    D = np.random.uniform(
        low=2, high=3, size=(ndim, nex)
    )

    InputTensors = generate_copula_net_input(
        D, bootstrap=False
    )
    UV_batches = InputTensors.UV_batches
    M_batches = InputTensors.M_batches
    YdC_batches = InputTensors.YdC_batches
    R_batches = InputTensors.R_batches
    X_batches = InputTensors.X_batches
    YC_batches = InputTensors.YC_batches

    assert_((UV_batches <= 1).all())
    assert_((UV_batches >= 0).all())

    assert_((M_batches <= 1).all())
    assert_((M_batches >= 0).all())

    assert_((YdC_batches <= 1).all())
    assert_((YdC_batches >= 0).all())

    assert_((R_batches <= 1).all())
    assert_((R_batches >= 0).all())

    assert_((X_batches <= 3).all())
    assert_((X_batches >= 2).all())

    assert_((YC_batches <= 1).all())
    assert_((YC_batches >= 0).all())


def test_generate_copula_net_input_rectangles():
    np.random.seed(30091985)

    ndim = 2
    nex = 1024
    D = np.random.uniform(2, 3, size=(ndim, nex))
    InputTensors = generate_copula_net_input(
        D, n_batches=16, batch_size=8
    )
    UV_batches = InputTensors.UV_batches
    R_batches = InputTensors.R_batches

    assert_((UV_batches + R_batches <= 1).all())
    assert_((UV_batches + R_batches >= 0).all())


def test_Y_is_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    np.random.seed(30091985)
    n_batches = 1
    batch_size = 1024

    # parameters for the synthetic copula
    rho = 0.65
    mean = np.zeros(2)
    E = np.zeros(shape=(2, 2)) + rho
    E[0, 0] = 1
    E[1, 1] = 1

    # dataset
    D = np.random.multivariate_normal(
        mean=mean, cov=E, size=(1000, )
    ).T

    InputTensors = generate_copula_net_input(
        D, n_batches=n_batches,
        batch_size=batch_size
    )

    assert_(InputTensors.UV_batches.shape[0] == 1)
    assert_(InputTensors.UV_batches.shape[1] == 2)
    assert_(InputTensors.UV_batches.shape[2] == 1024)

    # get the expected values from the copula equation
    E_batches = np.zeros(shape=(n_batches, batch_size, 1))
    for batch_i in range(n_batches):
        Eb = gauss_copula(
            InputTensors.UV_batches[batch_i].T, mean, E
        ).reshape(batch_size, 1)
        E_batches[batch_i] = Eb

    assert_array_almost_equal(
        InputTensors.YC_batches.ravel(), E_batches.ravel(),
        0.01
    )


def test_M_and_X_are_correct():
    '''
    Tests the input generator against a synthetic copula
    '''
    np.random.seed(30091985)
    n_batches = 1
    batch_size = 1024

    # parameters for the synthetic copula
    rho = 0.65
    mean = np.zeros(2)
    E = np.zeros(shape=(2, 2)) + rho
    E[0, 0] = 1
    E[1, 1] = 1

    # dataset
    D = np.random.multivariate_normal(
        mean=mean, cov=E, size=(1000, )
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
        ecdf_0(data_points_0), marginal_ecdfs_0, 3
    )
    assert_array_almost_equal(
        ecdf_1(data_points_1), marginal_ecdfs_1, 3
    )


def test_with_wikipedia_example_Y():
    np.random.seed(30091985)
    ps = np.array([0.00, 0.10, 0.00, 0.10,
                   0.00, 0.00, 0.20, 0.00,
                   0.30, 0.00, 0.00, 0.15,
                   0.00, 0.00, 0.15, 0.00])

    data = [(1, 2), (1, 4), (1, 6), (1, 8),
            (3, 2), (3, 4), (3, 6), (3, 8),
            (5, 2), (5, 4), (5, 6), (5, 8),
            (7, 2), (7, 4), (7, 6), (7, 8)]

    D = []
    for r in np.random.multinomial(1, ps, size=(1000, )):
        i = np.where(r)[0][0]
        D.append([data[i][0], data[i][1]])
    D = np.array(D, dtype=np.float32)

    InputTensors = generate_copula_net_input(
        D.T, bootstrap=False
    )
    YC_batches = InputTensors.YC_batches
    assert_(((YC_batches != 0) & (YC_batches != 1)).any())


def test_closed_form_partial_large():
    '''
    Copulas of the form:

    C(u, v) = uv / (u + v - uv)

    Have a closed form partial of:

    c_u(v) =  (v / (u + v - uv))**2
    c_v(u) =  (u / (v + u - vu))**2

    and inverse

    u ~ uniform(0, 1)
    t ~ uniform(0, 1)
    v = i_u(t) = u * sqrt(t) / (1 - (1 - u) sqrt(t))

    v ~ uniform(0, 1)
    t ~ uniform(0, 1)
    u = i_v(t) = v * sqrt(t) / (1 - (1 - v) sqrt(t))

    See example 2.20 of "An introduction to copulas"

    We shall use this expression to test our empirical
    partials.
    '''
    np.random.seed(30091985)

    us = np.random.uniform(0, 1, size=(1000, ))
    ts = np.random.uniform(0, 1, size=(1000, ))
    vs = us * np.sqrt(ts) / (1 - (1 - us) * np.sqrt(ts))

    # from the book this is a valid dataset copulated by
    # the copula in the docstring
    d0 = 2 * us - 1
    d1 = -np.log(1 - vs)

    D = np.array([d0, d1])
    InputTensors = generate_copula_net_input(
        D, bootstrap=False
    )

    def C(u, v):
        return u * v / (u + v - u * v)

    UV = InputTensors.UV_batches[0]

    Y_expected = C(UV[0], UV[1]).ravel()
    Y_emp = InputTensors.YC_batches.ravel()

    assert_array_almost_equal(Y_expected, Y_emp, 1e-5)

    r, p = ss.pearsonr(Y_expected, Y_emp)
    assert_almost_equal(1.0, r, 1e-5)
    assert_almost_equal(0.0, p, 1e-5)

    def dC(v, u):  # P[U <= v | U = u]
        return (v / (u + v - u * v)) ** 2

    dC_expected_vu = dC(UV[1], UV[0]).ravel()
    dC_expected_uv = dC(UV[0], UV[1]).ravel()

    dC_emp_uv = InputTensors.YdC_batches[:, 0, :].ravel()
    dC_emp_vu = InputTensors.YdC_batches[:, 1, :].ravel()

    r, p = ss.pearsonr(dC_expected_vu, dC_emp_vu)
    assert_(r >= 0.90)
    assert_almost_equal(0.0, p, 1e-5)

    r, p = ss.pearsonr(dC_expected_uv, dC_emp_uv)
    assert_(r >= 0.90)
    assert_almost_equal(0.0, p, 1e-5)


def test_closed_form_partial_small():
    # same as the previous test, used for debugging
    # easier to follow small arrays

    np.random.seed(30091985)

    us = np.random.uniform(0, 1, size=(100, ))
    ts = np.random.uniform(0, 1, size=(100, ))
    vs = us * np.sqrt(ts) / (1 - (1 - us) * np.sqrt(ts))

    d0 = 2 * us - 1
    d1 = -np.log(1 - vs)

    D = np.array([d0, d1])
    InputTensors = generate_copula_net_input(
        D, bootstrap=False
    )

    def C(u, v):
        return u * v / (u + v - u * v)

    UV = InputTensors.UV_batches[0]

    Y_expected = C(UV[0], UV[1]).ravel()
    Y_emp = InputTensors.YC_batches.ravel()

    assert_array_almost_equal(Y_expected, Y_emp, 1e-5)

    r, p = ss.pearsonr(Y_expected, Y_emp)
    assert_almost_equal(1.0, r, 1e-5)
    assert_almost_equal(0.0, p, 1e-5)

    def dC(v, u):  # P[V <= v | U = u]
        return (v / (u + v - u * v)) ** 2

    dC_expected_vu = dC(UV[1], UV[0]).ravel()
    dC_expected_uv = dC(UV[0], UV[1]).ravel()

    dC_emp_uv = InputTensors.YdC_batches[:, 0, :].ravel()
    dC_emp_vu = InputTensors.YdC_batches[:, 1, :].ravel()

    r, p = ss.pearsonr(dC_expected_vu, dC_emp_vu)
    assert_(r >= 0.9)
    assert_almost_equal(0.0, p, 1e-5)

    r, p = ss.pearsonr(dC_expected_uv, dC_emp_uv)
    assert_(r >= 0.9)
    assert_almost_equal(0.0, p, 1e-5)
