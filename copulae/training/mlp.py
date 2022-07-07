# -*- coding: utf8 -*-

'''
Code regarding multi layer perceptrons, mostly network
initialization, definition and loss functions.

Written from scratch instead of using Flax as to have
more control over experiments and architectures.
'''


from copulae.typing import Tensor
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
    params: list
        The parameters (weights, bias) for each layer of
        the network
    '''
    initializer = jax.nn.initializers.lecun_normal()
    params = []
    subkeys = jax.random.split(key, n_layers + 1)

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
        shape=(1, 1), dtype=jnp.float32
    ) + b_init
    params.append((weights, b))

    return params


@jax.jit
def mlp(
    params: PyTree,
    U: Tensor
) -> Tensor:
    '''
    Feed-forward for a simple multi-layer neural network.
    The parameters of the network should be initialized
    using the `init_mlp` function. `U` is the input
    of the network.

    This network is the one that mimicks the copula. Thus,
    every element in the input matrix `U` will be clipped
    to the range [0, 1].

    In order to create valid copulas, the final activation
    must output a number in [0, 1]. By default, we make
    use of a sigmoid. Middle activations are Swish
    functions.

    Parameters
    ----------
    params: PyTree
        The parameters of the network. If `U` has
        `n_dimensions` (features), then you must
        initialize parameters as:
        >>> n_dimensions = U.shape[0]
        >>> key, params = init_mlp(key, n_dimensions, ...)
    U: Tensor (2d)
        A matrix of shape: (n_dimensions, n_examples). Note
        that this is different from your common numpy data
        matrix where rows are examples. Here, examples are
        columns.

    Returns
    -------
    A column vector with `n_examples` entries. These are
    the activations for example in `X`.
    '''
    a = jnp.clip(U, 0, 1)  # map input to [0, 1]
    for W, b in params[:-1]:
        z = jnp.dot(W, a) + b
        a = jax.nn.swish(z)

    W, b = params[-1]
    z = jnp.dot(W, a) + b
    return jax.nn.sigmoid(z).T
