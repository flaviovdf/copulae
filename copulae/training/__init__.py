# -*- coding: utf8 -*-


from copulae.c import create_copula

from copulae.kde import silvermans_method
from copulae.kde import kde_pdf

from copulae.typing import Callable
from copulae.typing import PyTree
from copulae.typing import Tensor
from copulae.typing import Tuple
from copulae.typing import Sequence


from collections import namedtuple


import jax
import jax.numpy as jnp


CopulaTrainingState = namedtuple(
    'CopulaTrainingState',
    [
        'U_batches',   # the input of the neural copula
        'M_batches',   # the marginal CDFs of the copula
        'C_batches',   # the conditional CDFs of the copula
        'R_batches',   # a random rectangle around U
        'X_batches',   # data points associated with U
        'Y_batches',   # the expected output of the copula

        'ŶY_batches',  # the actual output of the copula
        'ŶC_batches',  # the actual marginal CDFs output
        'Ŷc_batches',  # the density output of the copula

        'I_pdf'        # the product of the pdf of each dim
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


def setup_training(
    forward_fun: Callable,
    U_batches: Tensor,
    M_batches: Tensor,
    C_batches: Tensor,
    R_batches: Tensor,
    X_batches: Tensor,
    Y_batches: Tensor,
    losses: Sequence[Tuple[float, Callable]]
):
    cumulative, partial, density = create_copula(
        forward_fun
    )
    losses = losses.copy()

    @jax.jit
    def forward(
        params: PyTree,
        state: CopulaTrainingState
    ):
        ŶY_batches = cumulative(params, state.U_batches)
        ŶC_batches = partial(params, state.M_batches)
        Ŷc_batches = density(params, state.M_batches)

        new_state = CopulaTrainingState(
            U_batches=state.U_batches,
            M_batches=state.M_batches,
            C_batches=state.C_batches,
            R_batches=state.R_batches,
            X_batches=state.X_batches,
            Y_batches=state.Y_batches,

            ŶY_batches=ŶY_batches,
            ŶC_batches=ŶC_batches,
            Ŷc_batches=Ŷc_batches,

            I_pdf=state.I_pdf
        )

        loss = jnp.zeros((1,), dtype=jnp.float32)
        for w, loss_func in losses:
            loss += w * loss_func(params, new_state)
        return loss[0], new_state

    U_flat = U_batches.reshape(
        U_batches.shape[1],
        U_batches.shape[0] * U_batches.shape[2]
    )

    n = U_flat.shape[1]
    bw = silvermans_method(n, 1)
    independence_pdf = 1.0
    for dim in range(U_flat.shape[0]):
        independence_pdf *= kde_pdf(U_flat[dim], bw)
    I_pdf = independence_pdf.reshape(
        U_batches.shape[0], U_batches.shape[2]
    )

    state = CopulaTrainingState(
        U_batches=U_batches,
        M_batches=M_batches,
        C_batches=C_batches,
        R_batches=R_batches,
        X_batches=X_batches,
        Y_batches=Y_batches,
        I_pdf=I_pdf
    )
    return cumulative, partial, density, state, \
        forward, jax.grad(forward, has_aux=True)
