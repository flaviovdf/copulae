# -*- coding: utf8 -*-

'''
Port of some of statsmodels functions to jax. Moreover,
we need to make sure that no nans or infs appear,
something statsmodels fallsback to in some cases (e.g.,
the cdf of statsmodels make's use of -np.inf which
we here avoid in order to feed only valid numbers to
the neural networks)
'''
