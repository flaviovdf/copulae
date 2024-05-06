# -*- coding: utf8


from flax.linen.dtypes import promote_dtype

from flax.typing import (
    Array,
    Dtype,
    Initializer,
    Optional,
    PrecisionLike
)


import flax.linen as nn

import jax
import jax.numpy as jnp


lecun_normal = nn.initializers.lecun_normal()
zeros_init = nn.initializers.zeros_init()


class PositiveDense(nn.Module):
    """A linear with transformation *positive weights*
    applied over the last dimension of the input.

    Positive weights are simply the regular Dense layer
    weights re-scaled by a elu(w) + 1 activation.

    Follows the same approach as `flax.linen.linear.Dense`.
    Check that documentation for more information.

    Attributes
    ----------
    features: the number of output features.
    use_bias: whether to add a bias to the output
        (default: True).
    dtype: the dtype of the computation
        (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers
        (default: float32).
    precision: numerical precision of the computation see
        ``jax.lax.Precision`` for details.
    kernel_init: initializer function for the weight
        matrix.
    bias_init: initializer function for the bias.
    """

    features: int
    use_bias: bool = True
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    kernel_init: Initializer = lecun_normal
    bias_init: Initializer = zeros_init

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation with positive
        weights to the inputs along the last dimension.

        Args:
            inputs: The nd-array to be transformed.

        Returns:
            The transformed input.
        """
        kernel = self.param(
            'kernel',
            self.kernel_init,
            (jnp.shape(inputs)[-1], self.features),
            self.param_dtype,
        )
        if self.use_bias:
            bias = self.param(
                'bias', self.bias_init,
                (self.features,),
                self.param_dtype
            )
        else:
            bias = None
        inputs, kernel, bias = promote_dtype(
            inputs, kernel, bias, dtype=self.dtype
        )
        kernel = nn.elu(kernel) + 1

        if self.dot_general_cls is not None:
            dot_general = self.dot_general_cls()
        elif self.dot_general is not None:
            dot_general = self.dot_general
        else:
            dot_general = jax.lax.dot_general
        y = dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            y += jnp.reshape(
                bias, (1,) * (y.ndim - 1) + (-1,)
            )
        return y
