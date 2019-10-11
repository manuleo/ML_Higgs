# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def build_poly(x, degree):
    pi=np.empty([degree+1,x.shape[0]])
    for i in range(degree+1):
        pi[i] = x**i
    res = np.transpose(pi)
    return res