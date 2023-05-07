# -*- coding: utf8 -*-

'''Unit tests for the book example copula'''


from copulae.closedcopulas.book220 import C
from copulae.closedcopulas.book220 import dCdu
from copulae.closedcopulas.book220 import dCdv
from copulae.closedcopulas.book220 import c


from numpy.testing import assert_


def test_all():
    for u in [0.01, 0.1, 0.5, 0.9, 0.99]:
        for v in [0.01, 0.1, 0.5, 0.9, 0.99]:
            assert_(C(u, v) >= 0)
            assert_(C(u, v) <= 1)

            assert_(dCdu(u, v) >= 0)
            assert_(dCdu(u, v) <= 1)

            assert_(dCdv(u, v) >= 0)
            assert_(dCdv(u, v) <= 1)

            assert_(c(u, v) >= 0)
