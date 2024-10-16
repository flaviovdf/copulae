# -*- coding: utf8 -*-


from copulae.c import CopulaType
from copulae.c import create_copula

from copulae.kde import silvermans_method
from copulae.kde import kde_pdf

from copulae.typing import Callable
from copulae.typing import PyTree
from copulae.typing import Tensor
from copulae.typing import Tuple
from copulae.typing import Sequence


from collections import namedtuple


import flax


import jax
import jax.numpy as jnp


CopulaTrainingState = namedtuple(
    'CopulaTrainingState',
    [
        'UV_batches',   # the input of the neural copula
        'M_batches',    # the marginal CDFs of the copula
        'X_batches',    # data points associated with U
        'R_batches',    # a random rectangle around U

        'YdC_batches',  # conditional CDFs of the copula
        'YC_batches',   # the expected output of the copula

        'ŶC_batches',   # the actual output of the copula
        'ŶdC_batches',  # the actual marginal CDFs output
        'Ŷc_batches',   # the density output of the copula

        'I_pdf'         # the product of the marginal pdfs
    ],
    defaults=[
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),

        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),

        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1)),

        jnp.zeros((1, 1))
    ]
)


def rescale_for_training(UV_batches):
    n = UV_batches.shape[2]
    s = n / (n + 1)
    return jnp.clip(UV_batches, 1e-6, 1) * s


def setup_training(
    forward_fun: CopulaType,
    TrainingTensors: Tuple[Tensor, Tensor, Tensor, Tensor,
                           Tensor, Tensor],
    losses: Sequence[Tuple[float, Callable]],
    rescale: bool = False
):

    if rescale:
        UV_batches = rescale_for_training(
            TrainingTensors.UV_batches
        )
    else:
        UV_batches = TrainingTensors.UV_batches
    M_batches = TrainingTensors.M_batches
    X_batches = TrainingTensors.X_batches
    R_batches = TrainingTensors.R_batches
    YdC_batches = TrainingTensors.YdC_batches
    YC_batches = TrainingTensors.YC_batches

    if isinstance(forward_fun, flax.linen.Module):
        def net(params, U):
            return forward_fun.apply(params, U)
    else:
        net = forward_fun

    cumulative, partial, density = create_copula(net)

    @jax.jit
    def forward(
        params: PyTree,
        state: CopulaTrainingState
    ):
        ŶC_batches = cumulative(
            params, state.UV_batches
        )
        ŶdC_batches = partial(
            params, state.UV_batches
        )
        Ŷc_batches = density(
            params, state.UV_batches
        )

        new_state = CopulaTrainingState(
            UV_batches=state.UV_batches,
            X_batches=state.X_batches,
            M_batches=state.M_batches,
            R_batches=state.R_batches,
            YdC_batches=state.YdC_batches,
            YC_batches=state.YC_batches,

            ŶC_batches=ŶC_batches,
            ŶdC_batches=ŶdC_batches,
            Ŷc_batches=Ŷc_batches,

            I_pdf=state.I_pdf
        )

        loss = jnp.zeros((1,), dtype=jnp.float32)
        for w, loss_func in losses:
            loss += w * loss_func(params, new_state)
        return loss[0], new_state

    U_flat = UV_batches.reshape(
        UV_batches.shape[1],
        UV_batches.shape[0] * UV_batches.shape[2]
    )

    n = U_flat.shape[1]
    bw = silvermans_method(n, 1)
    independence_pdf = 1.0
    for dim in range(U_flat.shape[0]):
        independence_pdf *= kde_pdf(U_flat[dim], bw)
    I_pdf = independence_pdf.reshape(
        UV_batches.shape[0], UV_batches.shape[2]
    )

    state = CopulaTrainingState(
        UV_batches=UV_batches,
        M_batches=M_batches,
        X_batches=X_batches,
        R_batches=R_batches,
        YdC_batches=YdC_batches,
        YC_batches=YC_batches,
        I_pdf=I_pdf
    )
    return cumulative, partial, density, state, \
        forward, jax.grad(forward, has_aux=True)
