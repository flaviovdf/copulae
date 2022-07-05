# -*- coding: utf8 -*-

'''
Empirical CDF Functions. Mostly a copy paste from
statsmodels in order to use jax.
'''


import jax.numpy as jnp


class StepFunction(object):
    '''
    A basic step function.
    Values at the ends are handled in the simplest way
    possible: everything to the left of x[0] is set to
    ival; everything to the right of x[-1] is set to y[-1].

    Attributes
    ----------
    x : array_like
    y : array_like
    ival : float
        ival is the value given to the values to the left
        of x[0]. Default is 0.
    sorted : bool
        Default is False.
    side : {'left', 'right'}, optional
        Default is 'left'. Defines the shape of the
        intervals constituting the steps. 'right'
        correspond to [a, b) intervals and 'left' to
        (a, b].

    Examples
    --------
    >>> import numpy as np
    >>> from copulae.sm.ecdf import StepFunction
    >>>
    >>> x = np.arange(20)
    >>> y = np.arange(20)
    >>> f = StepFunction(x, y)
    >>>
    >>> print(f(3.2))
    3.0
    >>> print(f([[3.2,4.5],[24,-3.1]]))
    [[  3.   4.]
     [ 19.   0.]]
    >>> f2 = StepFunction(x, y, side='right')
    >>>
    >>> print(f(3.0))
    2.0
    >>> print(f2(3.0))
    3.0
    '''
    def __init__(
        self, x, y, ival=0., sorted=False, side='left'
    ):
        if side.lower() not in ['right', 'left']:
            msg = "side must be in {'right' or 'left'}"
            raise ValueError(msg)
        self.side = side

        _x = jnp.asarray(x)
        _y = jnp.asarray(y)

        if _x.shape != _y.shape:
            msg = "x and y do not have the same shape"
            raise ValueError(msg)
        if len(_x.shape) != 1:
            msg = 'x and y must be 1-dimensional'
            raise ValueError(msg)

        self.x = jnp.r_[-jnp.inf, _x]
        self.y = jnp.r_[ival, _y]

        if not sorted:
            asort = jnp.argsort(self.x)
            self.x = jnp.take(self.x, asort, 0)
            self.y = jnp.take(self.y, asort, 0)
        self.n = self.x.shape[0]

    def __call__(self, time):
        ind = jnp.searchsorted(self.x, time, self.side) - 1
        return self.y[ind]


class ECDF(StepFunction):
    '''
    Return the Empirical CDF of an array as a step
    function.

    Attributes
    ----------
    x : array_like
        Observations
    side : {'left', 'right'}, optional
        Default is 'right'. Defines the shape of the
        intervals constituting the . 'right' correspond to
        [a, b) intervals and 'left' to (a, b].
    Returns
    -------
    Empirical CDF as a step function.
    Examples
    --------
    >>> import numpy as np
    >>> from copulae.sm.ecdf import ECDF
    >>>
    >>> ecdf = ECDF([3, 3, 1, 4])
    >>>
    >>> ecdf([3, 55, 0.5, 1.5])
    array([ 0.75,  1.  ,  0.  ,  0.25])
    '''
    def __init__(self, x, side='right'):
        x = jnp.array(x)
        x.sort()
        nobs = x.shape[0]
        y = jnp.linspace(1.0 / nobs, 1, nobs)
        super(ECDF, self).__init__(
            x, y, side=side, sorted=True
        )
