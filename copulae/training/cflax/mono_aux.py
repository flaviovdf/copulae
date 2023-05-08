# -*- coding: utf8


from copulae.typing import Tensor
from copulae.typing import Sequence


import flax.linen as nn


import jax
import jax.numpy as jnp


@jax.jit
def cumtrapz(u: Tensor, z: Tensor) -> Tensor:
    # u and z must be ordered according to u

    d = jnp.diff(u, prepend=0)
    s = jnp.zeros(d.shape[0], dtype=jnp.float32)
    s = s.at[0].set(z[0])
    s = s.at[1:].set(z[1:] + z[:-1])
    return (d * s / 2.0).cumsum()


@jax.jit
def integrate_and_set(u: Tensor, z: Tensor) -> Tensor:
    idx = u.argsort()
    u_sorted = u[idx]
    ct = cumtrapz(u_sorted, z[idx])
    reverse_idx = jnp.searchsorted(u_sorted, u)
    return ct[reverse_idx]


class ResELUPlusOne(nn.Module):
    layers: Sequence[int]

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)
        z = nn.Dense(self.layers[0])(a)
        a = nn.elu(z) + 1

        for layer_width in self.layers[1:]:
            z = nn.Dense(layer_width)(a)
            a = nn.elu(z) + 1 + a

        z = nn.Dense(1)(a)
        return nn.elu(z) + 1 + a


class EluPOne(nn.Module):
    @nn.compact
    def __call__(self, z: Tensor, _: Tensor) -> Tensor:
        return nn.elu(z) + 1


class ResEluPOne(nn.Module):
    @nn.compact
    def __call__(self, z: Tensor, x: Tensor) -> Tensor:
        return nn.elu(z) + 1 + x


class SoftPlus(nn.Module):
    @nn.compact
    def __call__(self, z: Tensor, _: Tensor) -> Tensor:
        return nn.softplus(z)


class ResSoftPlus(nn.Module):
    @nn.compact
    def __call__(self, z: Tensor, x: Tensor) -> Tensor:
        return nn.softplus(z) + x


class Identity(nn.Module):
    @nn.compact
    def __call__(self, z: Tensor, _: Tensor) -> Tensor:
        return z


class PositiveLayer(nn.Module):
    layers: Sequence[int]
    ini: EluPOne | SoftPlus | Identity
    mid: EluPOne | SoftPlus | ResELUPlusOne | ResSoftPlus
    end: EluPOne | SoftPlus | ResELUPlusOne | ResSoftPlus

    @nn.compact
    def __call__(self, U: Tensor) -> Tensor:
        a = jnp.clip(U.T, 0, 1)
        z = nn.Dense(self.layers[0])(a)
        a = self.ini()(z, a)

        for layer_width in self.layers[1:]:
            z = nn.Dense(layer_width)(a)
            a = self.mid()(z, a)

        z = nn.Dense(1)(a)
        return self.end()(z, a)
