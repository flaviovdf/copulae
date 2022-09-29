# -*- coding: utf8 -*-
'''Unit tests for loss functions'''


from copulae.training import CopulaTrainingState

from copulae.training.loss import cross_entropy
from copulae.training.loss import frechet
from copulae.training.loss import jsd
from copulae.training.loss import kld
from copulae.training.loss import l1
from copulae.training.loss import l2
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


def test_kld():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = kld(params, state)
    assert_(loss > 0)


def test_kld2():
    params = []
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = kld(params, state)
    assert_(loss > 0)


def test_kld3():
    params = []
    Y = jnp.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = kld(params, state)
    assert_(loss > 0)


def test_jsd():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_jsd2():
    params = []
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_jsd3():
    params = []
    Y = jnp.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = jsd(params, state)
    assert_(loss > 0)


def test_cross_entropy():
    params = []

    Y = jnp.array([0, 1, 0]).reshape((1, 3, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25]).reshape((1, 3, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_cross_entropy2():
    params = []
    Y = jnp.array([0, 1, 0, 0]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_cross_entropy3():
    params = []
    Y = jnp.array([0.1, 0.1, 0.1, 0.1]).reshape((1, 4, 1))
    Ŷ = jnp.array([0.15, 0.6, 0.25, 0]).reshape((1, 4, 1))

    state = CopulaTrainingState(
        Y_batches=Y,
        ŶC_batches=Ŷ
    )
    loss = cross_entropy(params, state)
    assert_(loss > 0)


def test_valid_partial():
    params = []
    ŶM_batches = jnp.zeros((2, 2, 3))

    ŶM_batches = ŶM_batches.at[0].set(
        jnp.array([[1.1, 0.2, 3.3], [0, -7, 0]])
    )
    ŶM_batches = ŶM_batches.at[1].set(
        jnp.array([[-1, -3, -0.001], [2.1, 3.2, 3.2]])
    )

    state = CopulaTrainingState(
        ŶM_batches=ŶM_batches
    )
    loss = valid_partial(params, state)
    assert_(loss == 0.75)


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


def test_frechet1():
    params = []

    U_batches = jnp.zeros((2, 2, 3))
    ŶC_batches = jnp.zeros((2, 3, 1))

    # max: 0, 0, 0.2
    # min: 0.1, 0.2, 0.6
    U_batches = U_batches.at[0].set(
        jnp.array([[0.1, 0.2, 0.6], [0.4, 0.5, 0.6]])
    )

    # max: 0.2, 0, 0
    # min: 0.4, 0.2, 0.3
    U_batches = U_batches.at[1].set(
        jnp.array([[0.8, 0.2, 0.3], [0.4, 0.5, 0.6]])
    )

    ŶC_batches = ŶC_batches.at[0].set(
        jnp.array([[0.7], [0.9], [0.0]])
    )
    ŶC_batches = ŶC_batches.at[1].set(
        jnp.array([[0.6], [0.9], [0.15]])
    )

    state = CopulaTrainingState(
        U_batches=U_batches,
        ŶC_batches=ŶC_batches
    )

    loss = frechet(params, state)
    assert_almost_equal(loss, 5 / 6)
