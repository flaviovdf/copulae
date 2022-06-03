# -*- coding: utf8 -*-


from .typing import PyTree

import jax
import jax.numpy as jnp


def init_mlp(
    key: jax.random.PRNGKey,
    input_size: int,
    n_layers: int,
    layer_width: int,
    b_init: int = 0,
) -> PyTree:

    initializer = jax.nn.initializers.lecun_normal()
    params = []
    new_key, *subkeys = jax.random.split(key, n_layers + 2)

    weights = initializer(
        subkeys[0],
        (layer_width, input_size),
        jnp.float32
    )
    b = jnp.zeros(shape=(layer_width, 1)) + b_init
    params.append((weights, b))

    for i in range(1, n_layers):
        weights = initializer(
            subkeys[i],
            (layer_width, layer_width),
            jnp.float32
        )
        b = jnp.zeros(shape=(layer_width, 1)) + b_init
        params.append((weights, b))

    weights = initializer(
        subkeys[-1],
        (1, layer_width),
        jnp.float32
    )
    b = jnp.zeros(shape=(1, 1)) + b_init
    params.append((weights, b))

    return params, new_key
