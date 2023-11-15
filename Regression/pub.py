import tables
import numpy as np
from numba import jit

def ReadJPPMT(file=r"./PMT.txt"):
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    PMT_pos = np.loadtxt(file)
    return PMT_pos

@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x