# -*- coding: utf8 -*-
'''Unit tests for our training loops'''


from copulae.input import generate_copula_net_input

from copulae.training import setup_training

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

from copulae.training.mlp import init_mlp
from copulae.training.mlp import mlp

from copulae.typing import Tensor
from copulae.typing import Sequence


from numpy.testing import assert_


import flax.linen as nn


import jax


import numpy as np


def test_if_looses_are_considered():
    '''
    Simply test if our losses are considered in
    the training loop
    '''

    np.random.seed(30091985)

    # generate some points according to
    # example 2.20 of "An introduction to copulas"
    us = np.random.uniform(0, 1, size=(500, ))
    ts = np.random.uniform(0, 1, size=(500, ))
    vs = us * np.sqrt(ts) / (1 - (1 - us) * np.sqrt(ts))

    d0 = 2 * us - 1
    d1 = -np.log(1 - vs)

    D = np.array([d0, d1])

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    losses = [
        (0.001, cross_entropy),
        (0.001, cross_entropy_partial),
        (0.001, copula_likelihood),
        (0.001, frechet),
        (0.001, jsd),
        (0.001, jsd_partial),
        (0.001, l1),
        (0.001, l2),
        (0.001, sq_error),
        (0.001, sq_error_partial),
        (0.001, sq_frechet),
        (0.001, sq_valid_density),
        (0.001, sq_valid_partial),
        (0.001, valid_density),
        (0.001, valid_partial),
    ]

    nn_C, nn_dC, nn_c, cop_state, forward, grad = \
        setup_training(mlp, TrainingTensors, losses)

    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    init_params = init_mlp(subkey, 2, 2, 2, b_init=0)

    assert_(forward(init_params, cop_state)[0] > 0)

    _, _, _, _, forward2, _ = \
        setup_training(mlp, TrainingTensors, losses[:2])
    f1 = forward(init_params, cop_state)[0]
    f2 = forward2(init_params, cop_state)[0]

    assert_(f1 > 0)
    assert_(f2 > 0)
    assert_(f1 > f2)


def test_if_looses_are_considered_w_flax():
    '''
    Simply test if our losses are considered in
    the training loop
    '''
    '''
    Simply test if our losses are considered in
    the training loop
    '''

    np.random.seed(30091985)

    # generate some points according to
    # example 2.20 of "An introduction to copulas"
    us = np.random.uniform(0, 1, size=(500, ))
    ts = np.random.uniform(0, 1, size=(500, ))
    vs = us * np.sqrt(ts) / (1 - (1 - us) * np.sqrt(ts))

    d0 = 2 * us - 1
    d1 = -np.log(1 - vs)

    D = np.array([d0, d1])

    TrainingTensors = generate_copula_net_input(
        D=D,
        bootstrap=False
    )

    losses = [
        (0.001, cross_entropy),
        (0.001, cross_entropy_partial),
        (0.001, copula_likelihood),
        (0.001, frechet),
        (0.001, jsd),
        (0.001, jsd_partial),
        (0.001, sq_error),
        (0.001, sq_error_partial),
        (0.001, sq_frechet),
        (0.001, sq_valid_density),
        (0.001, sq_valid_partial),
        (0.001, valid_density),
        (0.001, valid_partial),
    ]

    class MLP(nn.Module):
        layers: Sequence[int]

        @nn.compact
        def __call__(self, U: Tensor) -> Tensor:
            a = U.T
            for layer_width in self.layers[:-1]:
                z = nn.Dense(layer_width)(a)
                a = nn.relu(z)
            return nn.Dense(self.layers[-1])(a)

    class SingleLogit(nn.Module):
        base: MLP

        @nn.compact
        def __call__(self, U: Tensor, _: Tensor) -> Tensor:
            return jax.nn.sigmoid(
                nn.Dense(1)(self.base(U))
            )

    model = SingleLogit(MLP([2, 2]))
    nn_C, nn_dC, nn_c, cop_state, forward, grad = \
        setup_training(model, TrainingTensors, losses)

    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)

    init_params = model.init(
        subkey,
        cop_state.UV_batches[0],
        cop_state.Or_batches[0]
    )

    _, _, _, _, forward2, _ = \
        setup_training(model, TrainingTensors, losses[:2])
    f1 = forward(init_params, cop_state)[0]
    f2 = forward2(init_params, cop_state)[0]

    assert_(f1 > 0)
    assert_(f2 > 0)
    assert_(f1 > f2)
