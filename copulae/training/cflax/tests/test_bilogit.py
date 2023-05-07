# -*- coding: utf8 -*-

from . import nonan


from copulae.closedcopulas import book220

from copulae.input import generate_copula_net_input

from copulae.training.cflax.mono_aux import ELUPlusOne
from copulae.training.cflax.bilogit import \
    (PositiveBiLogitCopula, SiamesePositiveBiLogitCopula)

from copulae.training import setup_training

from copulae.training.loss import sq_error
from copulae.training.loss import sq_error_partial
from copulae.training.loss import copula_likelihood


import jax
import jax.numpy as jnp


from numpy.testing import assert_


def test_pblc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 100, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [200, 200, 200, 200, 200]
    model = PositiveBiLogitCopula(
        ELUPlusOne(net_def)
    )

    _, subkey = jax.random.split(key)
    del key

    UV = TrainingTensors.UV_batches[0]
    Y = TrainingTensors.YC_batches[0]
    init_params = model.init(subkey, UV)

    jax.tree_map(nonan, init_params)

    def loss(params):
        Ŷ = model.apply(params, UV)
        return jnp.power(Y - Ŷ, 2).mean()

    grad = jax.grad(loss)
    dmodel = grad(init_params)
    jax.tree_map(nonan, dmodel)


def test_spblc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 100, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [200, 200, 200, 200, 200]
    model = SiamesePositiveBiLogitCopula(
        ELUPlusOne(net_def), ELUPlusOne(net_def)
    )

    _, subkey = jax.random.split(key)
    del key

    UV = TrainingTensors.UV_batches[0]
    Y = TrainingTensors.YC_batches[0]
    init_params = model.init(subkey, UV)

    jax.tree_map(nonan, init_params)

    def loss(params):
        Ŷ = model.apply(params, UV)
        return jnp.power(Y - Ŷ, 2).mean()

    grad = jax.grad(loss)
    dmodel = grad(init_params)
    jax.tree_map(nonan, dmodel)


def test_lots_of_derivatives_pblc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 500, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [int(x) for x in jnp.ones(16) * 16]
    model = PositiveBiLogitCopula(
        ELUPlusOne(net_def)
    )

    losses = [
        (1.0, sq_error),
        (1.0, sq_error_partial),
        (1.0, copula_likelihood)
    ]

    nn_C, nn_dC, nn_c, cop_state, _, grad = \
        setup_training(
            model, TrainingTensors, losses
        )

    UV = TrainingTensors.UV_batches[0]

    _, subkey = jax.random.split(key)
    del key
    init_params = model.init(subkey, UV)
    jax.tree_map(nonan, init_params)

    x = nn_C(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_dC(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_c(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    jax.tree_map(nonan, grad(init_params, cop_state))


def test_lots_of_derivatives_spblc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 500, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [int(x) for x in jnp.ones(16) * 16]
    model = SiamesePositiveBiLogitCopula(
        ELUPlusOne(net_def), ELUPlusOne(net_def)
    )

    losses = [
        (1.0, sq_error),
        (1.0, sq_error_partial),
        (1.0, copula_likelihood)
    ]

    nn_C, nn_dC, nn_c, cop_state, _, grad = \
        setup_training(
            model, TrainingTensors, losses
        )

    UV = TrainingTensors.UV_batches[0]

    _, subkey = jax.random.split(key)
    del key
    init_params = model.init(subkey, UV)
    jax.tree_map(nonan, init_params)

    x = nn_C(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_dC(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_c(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    jax.tree_map(nonan, grad(init_params, cop_state))
