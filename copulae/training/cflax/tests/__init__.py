# -*- coding: utf 8


from numpy.testing import assert_


import jax.numpy as jnp


def nonan(t):
    t = jnp.asarray(t)
    assert_(not jnp.isnan(t).any())
    assert_(jnp.isfinite(t).all())
