from numba import jit, vectorize, guvectorize
from numba import float64, int64
import numpy as np
from numpy.random import normal, uniform, binomial, multinomial, gamma
from math import factorial
from math import gamma as gamma_f
from math import erf

########################################
############## SAMPLING ################
########################################

@vectorize([float64(float64, float64)], nopython=True)
def samp_normal(loc, scale):
    return normal(loc=loc, scale=scale)

@vectorize([float64(float64, float64)], nopython=True)
def samp_uniform(a, b):
    return uniform(a, b)

@vectorize([int64(int64, float64)], nopython=True)
def samp_binomial(n, p):
    return binomial(n=n, p=p)

@vectorize([float64(float64, float64)], nopython=True)
def samp_invgamma(shape, scale):
    gamma_var = gamma(shape=shape, scale=1/scale)
    return 1/gamma_var

@vectorize([float64(float64, float64, float64)], nopython=True)
def samp_lowtrunc_normal(loc, scale, low=0):
    x = low
    while x <= low:
        x = samp_normal(loc, scale)
    return x

@jit(nopython=True, nogil=True)
def samp_multinoulli(n, pvals):
    result = np.zeros(n, dtype=np.int64)
    for i in range(n):
        u = uniform(0, 1)
        result[i] = (u > pvals.cumsum()).argmin()
    return result  

@jit(nopython=True, nogil=True)
def samp_multinomial(n, pvals):
    return multinomial(n, pvals)

########################################
########### LOG PDFS/PMFS ##############
########################################

@vectorize([float64(float64, float64, float64)], nopython=True)
def normal_logpdf(x, loc=0, scale=1):
    return -1/2 * np.log(2 * np.pi) - np.log(scale) - 1/2 * ((x - loc) / scale) ** 2

@vectorize([float64(float64, float64, float64)], nopython=True)
def uniform_logpdf(x, a, b):
    if a <= x <= b:
        return -np.log(b - a)
    else:
        return -np.inf

@jit(nopython=True, nogil=True)
def binomial_logpmf(x, n, p):
    n_True = x[x == 1].size
    n_False = n - n_True
    const = np.log(factorial(n)) - np.log(factorial(n_True)) - np.log(factorial(n_False))
    return const + n_True * np.log(p) + n_False * np.log(1 - p)

@jit(nopython=True, nogil=True)
def multinomial_logpmf(x, n, pvals):
    n_classes = p_vals.size
    classes, counts = numpy.unique(x, return_counts=True)
    const = np.log(factorial(n)) - np.sum([np.log(factorial(c)) for c in counts])
    return const + np.sum(counts * np.log(pvals))

@jit(nopython=True, nogil=True)
def invgamma_logpdf(x, shape, scale):
    return shape * np.log(scale) - np.log(gamma_f(shape)) - (shape + 1) * np.log(x) - scale / x

@jit(nopython=True, nogil=True)
def lowtrunc_normal_logpdf(x, loc=0, scale=1, low=0):
    if x <= low:
        return -np.inf
    else:
        return normal_logpdf(x, loc, scale) - normal_logcdf(loc, low, scale)

########################################
########### LOG CDFS/CMFS ##############
########################################

@jit(nopython=True, nogil=True)
def normal_logcdf(x, loc, scale):
    z = (x - loc) / scale
    return np.log(1 + erf(z / np.sqrt(2))) - np.log(2)

########################################
########### MISCELLANEOUS ##############
########################################

@jit(nopython=True, nogil=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@jit(nopython=True, nogil=True)
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()