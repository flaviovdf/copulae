# -*- coding: utf8 -*-
'''Unit tests for input generators'''


from copulae.input import generate_copula_net_input


import jax


def test_generate_copula_net_input():
    key = jax.random.PRNGKey(30091985)
    key, subkey = jax.random.split(key)
    D = jax.random.normal(subkey, shape=(2, 100))

    _, subkey = jax.random.split(key)
    del key
    generate_copula_net_input(subkey, D)
