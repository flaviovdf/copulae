# -*- coding: utf8 -*-


from collections import namedtuple


import jax.numpy as jnp


CopulaTrainingState = namedtuple(
    'CopulaTrainingState',
    [
        'params',      # the neural copula parameters

        'U_batches',   # the input of the neural copula
        'M_batches',   # the marginal CDFs of the copula
        'X_batches',   # data points associated with U
        'Y_batches',   # the expected output of the copula

        'ŶC_batches',  # the actual output of the copula
        'ŶM_batches',  # the actual marginal CDFs output
        'Ŷc_batches'   # the density output of the copula
    ],
    defaults=[
        tuple(),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1))
    ]
)
