r'''Translation of fpbpl and fpbisp from Fortran to Python

The scipy source includes FITPACK, which implements b-spline interpolation. I
converted its 2D surface interpolation routine (fpbisp) to C (via f2c), and then
semi-manually to Python. I can then feed it sympy symbols, and get out
analytical expressions, which are surprisingly difficult to find in a nice
usable form on the internet.

The original fpbisp implementation is at

  scipy/interpolate/fitpack/fpbisp.f

The analysis that uses this conversion lives in bsplines.py

'''

import sympy

def fpbspl_(t, k, x, l):

    h__ = [None] * 100
    hh  = [None] * 100

    h__[-1 + 1] = sympy.Integer(1)
    i__1 = k
    for j in range(1,i__1+1):
        i__2 = j
        for i__ in range(1,i__2+1):
            hh[i__ - 1] = h__[-1 + i__]

        h__[-1 + 1] = 0
        i__2 = j
        for i__ in range(1,i__2+1):
            li = l + i__
            lj = li - j
            f = hh[i__ - 1] / (t[li] - t[lj])
            h__[-1 + i__] += f * (t[li] - x)
            h__[-1 + i__ + 1] = f * (x - t[lj])

    return h__


# argx is between tx[lx] and tx[lx+1]. Same with y
def fpbisp_(tx, ty, k, c, argx, lx, argy, ly):

    wx = [None] * 100
    wy = [None] * 100

    h__ = fpbspl_(tx, k, argx, lx)
    for j in range(1, (k+1)+1):
        wx[1 + j] = h__[j - 1]

    h__ = fpbspl_(ty, k, argy, ly)
    for j  in range(1,(k+1)+1):
        wy[1 + j]   = h__[j - 1]


    for i1 in range(1,(k+1)+1):
        h__[i1 - 1] = wx[1 + i1]

    sp = 0
    for i1 in range(1,(k+1)+1):
        for j1 in range(1,(k+1)+1):
            sp += c[j1-1,i1-1] * h__[i1 - 1] * wy[1 + j1]

    return sp
