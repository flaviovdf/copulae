# -*- coding: utf8 -*-


from copulae.c import create_copula

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
        'params',      # the neural copula parameters

        'U_batches',   # the input of the neural copula
        'M_batches',   # the marginal CDFs of the copula
        'X_batches',   # data points associated with U
        'Y_batches',   # the expected output of the copula

        'ŶC_batches',  # the actual output of the copula
        'ŶM_batches',  # the actual marginal CDFs output
        'Ŷc_batches'   # the density output of the copula
    ],
    defaults=[
        tuple(),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1)),
        jnp.zeros((1, 1, 1))
    ]
)


def update_params(
    state: CopulaTrainingState,
    new_params: PyTree
) -> CopulaTrainingState:
    return CopulaTrainingState(
        params=new_params,
        U_batches=state.U_batches,
        M_batches=state.M_batches,
        X_batches=state.X_batches,
        Y_batches=state.Y_batches,
        ŶC_batches=state.ŶC_batches,
        ŶM_batches=state.ŶM_batches,
        Ŷc_batches=state.Ŷc_batches
    )


def setup_training(
    forward_fun: Callable,
    params: PyTree,
    U_batches: Tensor,
    M_batches: Tensor,
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
        state: CopulaTrainingState
    ):
        ŶC_batches = cumulative(
            state.params, state.U_batches
        )
        ŶM_batches = partial(
            state.params, state.U_batches
        )
        Ŷc_batches = density(
            state.params, state.U_batches
        )

        new_state = CopulaTrainingState(
            params=state.params,
            U_batches=state.U_batches,
            M_batches=state.M_batches,
            X_batches=state.X_batches,
            Y_batches=state.Y_batches,
            ŶC_batches=ŶC_batches,
            ŶM_batches=ŶM_batches,
            Ŷc_batches=Ŷc_batches
        )

        loss = jnp.zeros((1,), dtype=jnp.float32)
        for w, loss_func in losses:
            loss += w * loss_func(new_state)
        return loss[0]

    state = CopulaTrainingState(
        params=params,
        U_batches=U_batches,
        M_batches=M_batches,
        X_batches=X_batches,
        Y_batches=Y_batches
    )
    return cumulative, partial, density, state, \
        forward, jax.grad(forward)
