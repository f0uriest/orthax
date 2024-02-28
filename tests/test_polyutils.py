"""Tests for polyutils module."""
import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises

import orthax.polyutils as pu


class TestMisc:
    def test_trimseq(self):
        for i in range(5):
            tgt = [1]
            res = pu.trimseq([1] + [0] * 5)
            assert_equal(res, tgt)

    def test_trimcoef(self):
        coef = np.array([2, -1, 1, 0])
        # Test exceptions
        assert_raises(ValueError, pu.trimcoef, coef, -1)
        # Test results
        assert_equal(pu.trimcoef(coef), coef[:-1])
        assert_equal(pu.trimcoef(coef, 1), coef[:-3])
        assert_equal(pu.trimcoef(coef, 2), np.array([0]))

    def test_vander_nd_exception(self):
        # n_dims != len(points)  noqa:E800
        assert_raises(ValueError, pu._vander_nd, (), (1, 2, 3), [90])
        # n_dims != len(degrees)  noqa:E800
        assert_raises(ValueError, pu._vander_nd, (), (), [90.65])
        # n_dims == 0  noqa:E800
        assert_raises(ValueError, pu._vander_nd, (), (), [])

    def test_pow_too_large(self):
        # power > maxpower
        assert_raises(ValueError, pu._pow, (), [1, 2, 3], 5, 4)


class TestDomain:
    def test_getdomain(self):
        # test for real values
        x = [1, 10, 3, -1]
        tgt = [-1, 10]
        res = pu.getdomain(x)
        assert_array_equal(res, tgt)

        # test for complex values
        x = [1 + 1j, 1 - 1j, 0, 2]
        tgt = [-1j, 2 + 1j]
        res = pu.getdomain(x)
        assert_array_equal(res, tgt)

    def test_mapdomain(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = dom2
        res = pu.mapdomain(dom1, dom1, dom2)
        assert_array_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = dom2
        x = dom1
        res = pu.mapdomain(x, dom1, dom2)
        assert_array_equal(res, tgt)

        # test for multidimensional arrays
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = np.array([dom2, dom2])
        x = np.array([dom1, dom1])
        res = pu.mapdomain(x, dom1, dom2)
        assert_array_equal(res, tgt)

    def test_mapparms(self):
        # test for real values
        dom1 = [0, 4]
        dom2 = [1, 3]
        tgt = [1, 0.5]
        res = pu.mapparms(dom1, dom2)
        assert_array_equal(res, tgt)

        # test for complex values
        dom1 = [0 - 1j, 2 + 1j]
        dom2 = [-2, 2]
        tgt = [-1 + 1j, 1 - 1j]
        res = pu.mapparms(dom1, dom2)
        assert_array_equal(res, tgt)
