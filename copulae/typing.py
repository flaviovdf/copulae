# -*- coding: utf8 -*-
'''Simple typing aliases'''


from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Sequence
from typing import Tuple
from typing import Union

import jax.numpy as jnp


Callable = Callable
Tensor = Union[jnp.ndarray, jnp.DeviceArray]
PyTree = Union[
    Tensor,
    Tuple["PyTree", ...],
    Sequence["PyTree"],
    Dict[Hashable, "PyTree"],
]
Sequence = Sequence
Tuple = Tuple
