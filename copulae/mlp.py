# -*- coding: utf8 -*-
'''
Code regarding multi layer perceptrons, mostly network
initialization, definition and loss functions.

Written from scratch instead of using Flax as to have
more control over experiments and architectures.
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


@jax.jit
def mlp(
    params: PyTree,
    X: Tensor,
    middle_activation: Callable = jax.nn.swish,
    end_activation: Callable = jax.nn.sigmoid
) -> Tensor:

    a = X
    for W, b in params[:-1]:
        z = jnp.dot(W, a) + b
        a = middle_activation(z)

    W, b = params[-1]
    z = jnp.dot(W, a) + b
    return end_activation(z).T


@jax.jit
def cross_entropy(
    Y: Tensor,
    logits: Tensor
) -> Tensor:
    logit = jnp.clip(logits, 1e-6, 1 - 1e-6)
    Y = jnp.clip(Y, 0, 1)
    return jnp.mean(
        -Y * jnp.log(logit) - (1 - Y) * jnp.log(1 - logit)
    )