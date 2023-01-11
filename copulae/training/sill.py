# -*- coding: utf8 -*-

'''
Code regarding neural networks which are guaranteed to
generate monotonic outputs. Here, we implement the
network defined in [1].

[1] Monotonic Networks. Joseph Sill. NeuRIPS 1997.
'''


from copulae.typing import Tensor
from copulae.typing import PyTree


import jax
import jax.numpy as jnp


def init_sill(
    key: jax.random.PRNGKey,
    input_size: int,
    n_layers: int,
    layer_width: int,
    n_groups_per_neuron: int,
    layer_width_per_group: int,
    b_init: int = 0
) -> PyTree:
    '''
    Initializes the layers of monotone neural network.

    Arguments
    ---------
    key: jax.random.PRNGKey
        The key to use for random number generation
    input_size: int
        The size of the input to the network, this will
        be the number of dimensions in the copula
    n_layers: int
        Number of layers in the networks
    n_groups_per_neuron: int
        The number of groups in each sill neuron. These
        groups are neurons which are max-pooled at the end
    layer_width_per_group: int
        The number of neurons to max pool in each group
    layer_width: int
        The width, number of neurons, of each layer
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
        (layer_width, n_groups_per_neuron,
         layer_width_per_group, input_size),
        jnp.float32
    )
    b = jnp.zeros(
        shape=(layer_width, n_groups_per_neuron,
               layer_width_per_group, 1),
        dtype=jnp.float32
    ) + b_init
    params.append((weights, b))

    for i in range(1, n_layers):
        weights = initializer(
            subkeys[i],
            (layer_width, n_groups_per_neuron,
             layer_width_per_group, layer_width),
            jnp.float32
        )
        b = jnp.zeros(
            shape=(layer_width, n_groups_per_neuron,
                   layer_width_per_group, 1),
            dtype=jnp.float32
        ) + b_init
        params.append((weights, b))

    weights = initializer(
        subkeys[-1],
        (1, n_groups_per_neuron,
         layer_width_per_group, layer_width),
        jnp.float32
    )
    b = jnp.zeros(
        shape=(1, n_groups_per_neuron,
               layer_width_per_group, 1),
        dtype=jnp.float32
    ) + b_init
    params.append((weights, b))

    return params


@jax.jit
def sill_neuron(
    Ws: Tensor,
    bs: Tensor,
    U: Tensor
) -> Tensor:
    '''
    For this library, a sill neuron is actually the full
    implementation of the two-layer monotone network
    described in the sill paper. We compose these as
    neurons of a more complex network.

    Arguments
    ---------
    Ws: Tensor
        tensor of shape [num_groups, num_neurons, in_dim]
    bs: Tensor
        bias for each group. array of len == num_groups
    U: Tensor
        The example to compute the output. Shape  of
        [in_dim, 1].
    '''
    # compute W @ U + b for each group
    # params are expotentiated to maintain positivity
    # the swish is used to preserve volume
    A = jax.vmap(
        lambda W: jnp.exp(W[0]) @ jax.nn.swish(U) + W[1]
    )((Ws, bs))

    # get the max for the group, axis=1,
    # then the min of every group group
    return A.max(axis=1).min(axis=0)


@jax.jit
def sill_net(
    params: PyTree,
    U: Tensor
) -> Tensor:
    '''
    Feed-forward for a simple multi-layer sill network.
    The parameters of the network should be initialized
    using the `init_sill` function. `U` is the input
    of the network.

    This network is the one that mimicks the copula. Thus,
    every element in the input matrix `U` will be clipped
    to the range [0, 1].

    In order to create valid copulas, the final activation
    must output a number in [0, 1].

    The hidden layers are composed of monotone networks
    which we call Sill Neurons (see `sill_neuron`). These
    implement the max-min approach described by [1].

    [1] Monotonic Networks. Joseph Sill. NeuRIPS 1997.

    Parameters
    ----------
    params: PyTree
        The parameters of the network. If `U` has
        `n_dimensions` (features), then you must
        initialize parameters as:
        >>> n_dimensions = U.shape[0]
        >>> key, params = init_sill(key, n_dimensions, ...)
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
    a = jnp.clip(U, 0, 1)
    for Wl, bl in params[:-1]:
        Wl = Wl * Wl
        z = jax.vmap(
            lambda W, b: sill_neuron(W, b, a),
            in_axes=[0, 0]
        )(Wl, bl)
        a = jax.nn.relu(z)
    Wl, bl = params[-1]
    z = jax.vmap(
        lambda W, b: sill_neuron(W, b, a),
        in_axes=[0, 0]
    )(Wl, bl)
    return jax.nn.sigmoid(z).T
