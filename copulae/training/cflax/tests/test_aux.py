# -*- coding: utf8 -*-
'''Unit tests for some auxiliaries'''


from copulae.training.cflax.mono_aux import cumtrapz
from copulae.training.cflax.mono_aux import \
    integrate_and_set


from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal


from scipy.integrate import cumtrapz as gt


import jax.numpy as jnp


def test_cumtrapz():
    u = jnp.linspace(0.01, 1, 100)

    #  our code assumes a copua as input, thus C(0) = 0
    #  this is forced in our cumptraz but not in scipy
    #  the appends bellow will append 0 to x and
    #  C(0) = 0 to z (or the image, y in scipy).

    u_sp = [0] + [float(x) for x in u]
    z = u + 1
    z_sp = [0] + [float(x) for x in z]

    assert_array_almost_equal(
        gt(x=u_sp, y=z_sp), cumtrapz(u, z)
    )


def test_cumtrapz2():
    u = jnp.linspace(0.01, 1, 100)

    #  our code assumes a copua as input, thus C(0) = 0
    #  this is forced in our cumptraz but not in scipy
    #  the appends bellow will append 0 to x and
    #  C(0) = 0 to z (or the image, y in scipy).

    u_sp = [0] + [float(x) for x in u]
    z = jnp.log(u + 1)
    z_sp = [0] + [float(x) for x in z]

    assert_array_almost_equal(
        gt(x=u_sp, y=z_sp), cumtrapz(u, z)
    )


def test_cumtrapz3():
    u = jnp.linspace(0.01, 1, 100)

    #  our code assumes a copua as input, thus C(0) = 0
    #  this is forced in our cumptraz but not in scipy
    #  the appends bellow will append 0 to x and
    #  C(0) = 0 to z (or the image, y in scipy).

    u_sp = [0] + [float(x) for x in u]
    z = jnp.exp(u ** 3 + 1)
    z_sp = [0] + [float(x) for x in z]

    assert_array_almost_equal(
        gt(x=u_sp, y=z_sp), cumtrapz(u, z)
    )


def test_integrate_and_set():
    u = [0.1, 0.3, 0.2, 0.4, 0.01]
    z = [1.1, 1.3, 1.2, 1.4, 1.01]

    us = [0.01, 0.1, 0.2, 0.3, 0.4]
    zs = [1.01, 1.1, 1.2, 1.3, 1.4]

    u = jnp.array(u)
    z = jnp.array(z)
    us = jnp.array(us)
    zs = jnp.array(zs)

    a = cumtrapz(us, zs)

    e = jnp.array([a[1], a[3], a[2], a[4], a[0]])
    y = integrate_and_set(u, z)
    assert_array_equal(e, y)
