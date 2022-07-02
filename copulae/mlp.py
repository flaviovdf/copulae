# -*- coding: utf8 -*-
'''
Code regarding multi layer perceptrons, mostly network
initialization and definition
'''


from copulae.typing import PyTree

import jax
import jax.numpy as jnp


def init_mlp(
    key: jax.random.PRNGKey,
    input_size: int,
    n_layers: int,
    layer_width: int,
    b_init: int = 0
) -> PyTree:
    '''
    Initializes the layers of a dense multilayer neural
    network. Weights are initialized using a Lecun Normal
    approach.

    Arguments
    ---------
    key: jax.random.PRNGKey
        The key to use for random number generation
    input_size: int
        The size of the input to the network, this will
        be the number of dimensions in the copula
    n_layers: int
        Number of layers in the networks
    layer_width: int
        The width of each layer
    b_init: int
        The initial value for the biases of each neural
        (defaults do zero)

    Returns
    -------
    key: jax.random.PRNGKey
        A new random key, the one used as input must be
        discarded
    params: list
        The parameters (weights, bias) for each layer of
        the network
    '''
    initializer = jax.nn.initializers.lecun_normal()
    params = []
    new_key, *subkeys = jax.random.split(key, n_layers + 2)

    weights = initializer(
        subkeys[0],
        (layer_width, input_size),
        jnp.float32
    )
    b = jnp.zeros(
        shape=(layer_width, 1), dtype=jnp.float32
    ) + b_init
    params.append((weights, b))

    for i in range(1, n_layers):
        weights = initializer(
            subkeys[i],
            (layer_width, layer_width),
            jnp.float32
        )
        b = jnp.zeros(
            shape=(layer_width, 1), dtype=jnp.float32
        ) + b_init
        params.append((weights, b))

    weights = initializer(
        subkeys[-1],
        (1, layer_width),
        jnp.float32
    )
    b = jnp.zeros(
        shape=(layer_width, 1), dtype=jnp.float32
    ) + b_init
    params.append((weights, b))

    return new_key, params
