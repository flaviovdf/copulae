# -*- coding: utf8 -*-
'''Simple typing aliases'''


from typing import Any
from typing import Callable
from typing import Dict
from typing import Hashable
from typing import Sequence
from typing import Tuple
from typing import Union


from jax import Array as JaxArray


import numpy as np
import numpy.typing as npt


PRNGKey = Any

Shape = Tuple[int, ...]
ShapeLike = Union[int, Sequence[int]]

Tensor = Union[
    np.ndarray, JaxArray, npt.NDArray[Any]
]

PyTree = Union[Tensor, Tuple["PyTree", ...],
               Sequence["PyTree"], Dict[Hashable, "PyTree"]
               ]
Callable = Callable
