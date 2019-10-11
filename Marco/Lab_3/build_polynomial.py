# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np

def PI(x,degree):
    pi=np.empty([degree+1,x.shape[0]])
    for i in range(degree+1):
        pi[i] = x**i
    return pi

def build_poly(x, degree):
    res = np.transpose(PI(x,degree))
    return res



