#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from .util import random_complex, inner_complex
from tike.operators import Convolution

__author__ = "Daniel Ching"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'


class TestConvolution(unittest.TestCase):
    """Test the Convolution operator."""

    def setUp(self):
        """Load a dataset for reconstruction."""
        self.ntheta = 3
        self.nscan = 27
        self.original_shape = (self.ntheta, 128, 128)
        self.probe_shape = 15

    def test_adjoint(self):
        """Check that the diffraction adjoint operator is correct."""
        np.random.seed(0)
        scan = np.random.rand(self.ntheta, self.nscan, 2) * 127 - 15
        original = random_complex(*self.original_shape)
        nearplane = random_complex(self.ntheta, self.nscan, self.probe_shape,
                                   self.probe_shape)

        scan = scan.astype('float32')
        original = original.astype('complex64')
        nearplane = nearplane.astype('complex64')

        with Convolution(
                ntheta=self.ntheta,
                nscan=self.nscan,
                nz=self.original_shape[-2],
                n=self.original_shape[-1],
                probe_shape=self.probe_shape,
        ) as slv:
            d = slv.fwd(
                scan=scan,
                psi=original,
            )
            o = slv.adj(
                nearplane=nearplane,
                scan=scan,
            )
            a = inner_complex(original, o)
            b = inner_complex(d, nearplane)
            print()
            print('<Q , P*ψ> = {:.6f}{:+.6f}j'.format(a.real.item(),
                                                      a.imag.item()))
            print('<QP,   ψ> = {:.6f}{:+.6f}j'.format(b.real.item(),
                                                      b.imag.item()))
            # Test whether Adjoint fixed probe operator is correct
            np.testing.assert_allclose(a.real, b.real, rtol=1e-5)
            np.testing.assert_allclose(a.imag, b.imag, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
