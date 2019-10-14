# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    degrees = np.arange(0, degree+1)
    psi = np.power(x[:, np.newaxis], degrees)
    #psi = np.power(x[:, :, np.newaxis], degrees) multidimensions
    return psi