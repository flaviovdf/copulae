# -*- coding: utf8 -*-

from . import nonan


from copulae.closedcopulas import book220

from copulae.input import generate_copula_net_input

from copulae.training.cflax.mono_aux import ELUPlusOne
from copulae.training.cflax.binorm import \
    (PositiveBiNormalCopula, SiamesePositiveBiNormalCopula)
from copulae.training.cflax.binorm import binorm
from copulae.training.cflax.binorm import vbinorm

from copulae.training import setup_training

from copulae.training.loss import sq_error
from copulae.training.loss import sq_error_partial
from copulae.training.loss import copula_likelihood


import jax
import jax.numpy as jnp


from numpy.testing import assert_
from numpy.testing import assert_array_almost_equal


def test_pbnc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 100, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [200, 200, 200, 200, 200]
    model = PositiveBiNormalCopula(
        ELUPlusOne(net_def)
    )

    UV = TrainingTensors.UV_batches[0]
    Y = TrainingTensors.YC_batches[0]

    _, subkey = jax.random.split(key)
    del key
    init_params = model.init(subkey, UV)

    jax.tree_map(nonan, init_params)

    def loss(params):
        Ŷ = model.apply(params, UV)
        return jnp.power(Y - Ŷ, 2).mean()

    grad = jax.grad(loss)
    dmodel = grad(init_params)
    jax.tree_map(nonan, dmodel)


def test_spbnc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 100, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [200, 200, 200, 200, 200]
    model = SiamesePositiveBiNormalCopula(
        ELUPlusOne(net_def), ELUPlusOne(net_def)
    )

    UV = TrainingTensors.UV_batches[0]
    Y = TrainingTensors.YC_batches[0]

    _, subkey = jax.random.split(key)
    del key
    init_params = model.init(subkey, UV)

    jax.tree_map(nonan, init_params)

    def loss(params):
        Ŷ = model.apply(params, UV)
        return jnp.power(Y - Ŷ, 2).mean()

    grad = jax.grad(loss)
    dmodel = grad(init_params)
    jax.tree_map(nonan, dmodel)


def test_lots_of_derivatives_pbnc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 500, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [int(x) for x in jnp.ones(16) * 16]
    model = PositiveBiNormalCopula(
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
    # print(x)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_dC(init_params, cop_state.UV_batches)
    # print(x)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    x = nn_c(init_params, cop_state.UV_batches)
    assert_(jnp.isfinite(x).all())
    assert_(not jnp.isnan(x).any())

    jax.tree_map(nonan, grad(init_params, cop_state))


def test_lots_of_derivatives_spbnc():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = book220.sample(subkey, 500, True)

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    net_def = [int(x) for x in jnp.ones(16) * 16]
    model = SiamesePositiveBiNormalCopula(
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


def test_binorm_paper_results():
    from scipy.stats import multivariate_normal
    gt = multivariate_normal.cdf

    rows = []
    for p in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for q in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            row = []
            for rho in [-0.99, -0.5, -0.1, 0.1, 0.5, 0.99]:
                row.append(binorm(p, q, rho))
            rows.append(row)
    x = jnp.array(rows)

    rows = []
    for x1 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
        for x2 in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            row = []
            for rho in [-0.99, -0.5, -0.1, 0.1, 0.5, 0.99]:
                means = [0, 0]
                E = [[1, rho],
                     [rho, 1]]
                row.append(gt([x1, x2], means, E))
            rows.append(row)
    e = jnp.array(rows)
    assert_array_almost_equal(e, x, 3)


def test_vbinorm():
    key = jax.random.PRNGKey(30091985)
    _, subkey = jax.random.split(key)
    del key

    D = jax.random.normal(subkey, shape=(2, 20))

    for rho in jnp.linspace(0.00001, 0.99999, 100):
        x = vbinorm(D[0], D[1], rho)
        assert_(jnp.isfinite(x).all())
        assert_(not jnp.isnan(x).any())


def test_vbinorm_grad():
    grad_p = jax.vmap(
        jax.grad(binorm, argnums=0),
        in_axes=(0, 0, None)
    )
    grad_q = jax.vmap(
        jax.grad(binorm, argnums=1),
        in_axes=(0, 0, None)
    )

    key = jax.random.PRNGKey(30091985)
    _, subkey = jax.random.split(key)
    del key

    D = jax.random.normal(subkey, shape=(2, 20))
    for rho in jnp.linspace(0.00001, 0.99999, 100):
        x = grad_p(D[0], D[1], rho)
        assert_(jnp.isfinite(x).all())
        assert_(not jnp.isnan(x).any())

        x = grad_q(D[0], D[1], rho)
        assert_(jnp.isfinite(x).all())
        assert_(not jnp.isnan(x).any())


def test_vbinorm_dgrad():
    dgrad = jax.vmap(
        jax.grad(jax.grad(binorm, argnums=0), argnums=1),
        in_axes=(0, 0, None)
    )

    key = jax.random.PRNGKey(30091985)
    _, subkey = jax.random.split(key)
    del key

    D = jax.random.normal(subkey, shape=(2, 20))
    for rho in jnp.linspace(0.00001, 0.99999, 100):
        x = dgrad(D[0], D[1], rho)
        assert_(jnp.isfinite(x).all())
        assert_(not jnp.isnan(x).any())
