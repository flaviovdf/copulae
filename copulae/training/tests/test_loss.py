# -*- coding: utf8 -*-
'''Unit tests for loss functions'''


from copulae.training import CopulaTrainingState

from copulae.training.loss import cross_entropy
from copulae.training.loss import cross_entropy_partial
from copulae.training.loss import copula_likelihood
from copulae.training.loss import frechet
from copulae.training.loss import jsd
from copulae.training.loss import jsd_partial
from copulae.training.loss import l1
from copulae.training.loss import l2
from copulae.training.loss import sq_error
from copulae.training.loss import sq_error_partial
from copulae.training.loss import sq_frechet
from copulae.training.loss import sq_valid_density
from copulae.training.loss import sq_valid_partial
from copulae.training.loss import valid_density
from copulae.training.loss import valid_partial


from numpy.testing import assert_
from numpy.testing import assert_almost_equal

import jax.numpy as jnp


def test_l1():
    params = []

    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    state = CopulaTrainingState()
    loss = l1(params, state)
    assert_(loss == 8)


def test_l2():
    params = []

    weights = jnp.array([1, 2])[:, jnp.newaxis]
    bias = jnp.array([[-1]])
    params.append((weights, bias))

    weights = jnp.array([0, 0])[:, jnp.newaxis]
    bias = jnp.array([[-4]])
    params.append((weights, bias))

    state = CopulaTrainingState()
    loss = l2(params, state)
    assert_(loss == 22)


def test_sq_error():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.1, 0.0, 0.0]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = sq_error(params, state)
    assert_(loss > 0)


def test_jsd():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_jsd2():
    params = []
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_jsd3():
    params = []
    Y = jnp.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_cross_entropy():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_cross_entropy2():
    params = []
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_cross_entropy3():
    params = []
    Y = jnp.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        YC_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_valid_partial():
    params = []
    ŶdC_batches = jnp.zeros((2, 2, 3))

    ŶdC_batches = ŶdC_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    ŶdC_batches = ŶdC_batches.at[1].set(
        jnp.array([[-1, -3, -0.001], [2.1, 3.2, 3.2]])
    )

    state = CopulaTrainingState(
        ŶdC_batches=ŶdC_batches
    )
    loss = valid_partial(params, state)
    assert_(loss == 0.75)

    loss = sq_valid_partial(params, state)
    assert_(loss > 0)


def test_cross_entropy_partial():
    params = []
    ŶdC_batches = jnp.zeros((2, 2, 3))

    ŶdC_batches = ŶdC_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    ŶdC_batches = ŶdC_batches.at[1].set(
        jnp.array([[-1, -3, -0.001], [2.1, 3.2, 3.2]])
    )

    state = CopulaTrainingState(
        ŶdC_batches=ŶdC_batches
    )
    loss = cross_entropy_partial(params, state)
    assert_(loss > 0)


def test_jsd_partial():
    params = []
    ŶdC_batches = jnp.zeros((2, 2, 3))

    ŶdC_batches = ŶdC_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    ŶdC_batches = ŶdC_batches.at[1].set(
        jnp.array([[-1, -3, -0.001], [2.1, 3.2, 3.2]])
    )

    state = CopulaTrainingState(
        ŶdC_batches=ŶdC_batches
    )
    loss = jsd_partial(params, state)
    assert_(loss > 0)


def test_sq_error_partial():
    params = []
    ŶdC_batches = jnp.zeros((2, 2, 3))

    ŶdC_batches = ŶdC_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    ŶdC_batches = ŶdC_batches.at[1].set(
        jnp.array([[-1, -3, -0.001], [2.1, 3.2, 3.2]])
    )

    state = CopulaTrainingState(
        ŶdC_batches=ŶdC_batches
    )
    loss = sq_error_partial(params, state)
    assert_(loss > 0)


def test_valid_density():
    params = []

    Ŷc_batches = jnp.zeros((2, 2, 3))

    Ŷc_batches = Ŷc_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    Ŷc_batches = Ŷc_batches.at[1].set(
        jnp.array([[0, -3, 0], [2.1, 0.9, 3.2]])
    )

    state = CopulaTrainingState(
        Ŷc_batches=Ŷc_batches
    )
    loss = valid_density(params, state)
    assert_(loss == 2.0 / 12)

    loss = sq_valid_density(params, state)
    assert_(loss > 0)


def test_copula_likelihood():
    params = []

    Ŷc_batches = jnp.zeros((2, 2, 3))

    Ŷc_batches = Ŷc_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    Ŷc_batches = Ŷc_batches.at[1].set(
        jnp.array([[0, -3, 0], [2.1, 0.9, 3.2]])
    )

    state = CopulaTrainingState(
        Ŷc_batches=Ŷc_batches
    )
    loss = copula_likelihood(params, state)
    assert_(loss > 0)


def test_frechet1():
    params = []

    UV_batches = jnp.zeros((2, 2, 3))
    ŶC_batches = jnp.zeros((2, 3, 1))

    # max: 0, 0, 0.2
    # min: 0.1, 0.2, 0.6
    UV_batches = UV_batches.at[0].set(
        jnp.array([[0.1, 0.2, 0.6], [0.4, 0.5, 0.6]])
    )

    # max: 0.2, 0, 0
    # min: 0.4, 0.2, 0.3
    UV_batches = UV_batches.at[1].set(
        jnp.array([[0.8, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )

    ŶC_batches = ŶC_batches.at[0].set(
        jnp.array([[0.7], [0.9], [0.0]])
    )
    ŶC_batches = ŶC_batches.at[1].set(
        jnp.array([[0.6], [0.9], [0.15]])
    )

    state = CopulaTrainingState(
        UV_batches=UV_batches,
        ŶC_batches=ŶC_batches
    )

    loss = frechet(params, state)
    assert_almost_equal(loss, 5 / 6)

    loss = sq_frechet(params, state)
    assert_(loss > 0)
