#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import gnuplotlib as gp

import scipy.interpolate
from scipy.interpolate import _fitpack

# I want to use b-splines to drive the generica camera models. These provide
# local support in the control points at the expense of not interpolating the
# control points. I don't NEED an interpolating spline, so this is just fine.
# Let's assume the knots lie at integer coordinates.
#
# I want a function that takes in
#
# - the control points in a neighborhood of the spline segment
# - the query point x, scaled to [0,1] in the spline segment
#
# This is all nontrivial for some reason, and there're several implementations
# floating around with slightly different inputs. I have
#
# - sample_segment_cubic()
#   From a generic calibration research library:
#   https://github.com/puzzlepaint/camera_calibration/blob/master/applications/camera_calibration/scripts/derive_jacobians.py
#   in the EvalUniformCubicBSpline() function. This does what I want (query
#   point in [0,1], control points around it, one segment at a time), but the
#   source of the expression isn't given. I'd like to re-derive it, and then
#   possibly extend it
#
# - splev_local()
#   wraps scipy.interpolate.splev()
#
# - splev_translated()
#   Followed sources of scipy.interpolate.splev() to the core fortran functions
#   in fpbspl.f and splev.f. I then ran these through f2c, pythonified it, and
#   simplified it. The result is sympy-able
#
# - splev_wikipedia()
#   Sample implementation of De Boor's algorithm from wikipedia:
#   https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
# from EvalUniformCubicBSpline() in
# camera_calibration/applications/camera_calibration/scripts/derive_jacobians.py.
# translated to [0,1] from [3,4]
def sample_segment_cubic(x, a,b,c,d):
    A =  (-x**3 + 3*x**2 - 3*x + 1)/6
    B = (3 * x**3/2 - 3*x**2 + 2)/3
    C = (-3 * x**3 + 3*x**2 + 3*x + 1)/6
    D = (x * x * x) / 6

    return A*a + B*b + C*c + D*d

def splev_local(x, t, c, k, der=0, ext=0):
    y = scipy.interpolate.splev(x, (t,c,k), der, ext)
    return y.reshape(x.shape)

def splev_translated(x, t, c, k, l):

    # assumes that t[l] <= x <= t[l+1]

    # print(f"l = {l}")
    # print((t[l], x, t[l+1]))

    l += 1 # l is now a fortran-style index

    hh  = [0] * 19
    h__ = [0] * 19

    h__[-1 + 1] = 1
    i__1 = k
    for j in range(1,i__1+1):
        i__2 = j
        for i__ in range(1,i__2+1):
            hh[-1 + i__] = h__[-1 + i__]
        h__[-1 + 1] = 0
        i__2 = j
        for i__ in range(1,i__2+1):
            li = l + i__
            lj = li - j
            if t[-1 + li] != t[-1 + lj]:
                h__[-1 + i__]    += (t[-1 + li] - x) * hh[-1 + i__] / (t[-1 + li] - t[-1 + lj])
                h__[-1 + i__ + 1] = (x - t[-1 + lj]) * hh[-1 + i__] / (t[-1 + li] - t[-1 + lj])
            else:
                h__[-1 + i__ + 1] = 0

    sp = 0
    ll = l - (k+1)
    i__2 = (k+1)
    for j in range(1,i__2+1):
        ll += 1
        sp += c[-1 + ll] * h__[-1 + j]
    return sp

# from https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
def splev_wikipedia(k: int, x: int, t, c, p: int):
    """Evaluates S(x).

    Arguments
    ---------
    k: Index of knot interval that contains x.
    x: Position.
    t: Array of knot positions, needs to be padded as described above.
    c: Array of control points.
       We will look at c[k-p .. k]
    p: Degree of B-spline.
    """
    # make sure I never reference c out of bounds
    if k-p < 0: raise Exception("c referenced out of min bounds")
    if k >= len(c): raise Exception("c referenced out of max bounds")
    d = [c[j + k - p] for j in range(0, p+1)]

    for r in range(1, p+1):
        for j in range(p, r-1, -1):
            alpha = (x - t[j+k-p]) / (t[j+1+k-r] - t[j+k-p])
            d[j] = (1 - alpha) * d[j-1] + alpha * d[j]

    return d[p]


##############################################################################
# First I confirm that the functions from numpy and from wikipedia produce the
# same results. I don't care about edge cases for now. I query an arbitrary
# point

N = 30
t = np.arange( N, dtype=int)
k = 3
c = np.random.rand(len(t))

x  = 4.3
# may have trouble exactly AT the knots. the edge logic isn't quite what I want
l = np.searchsorted(np.array(t), x)-1
y0 = splev_local(np.array((x,)),t,c,k)[0]
y1 = splev_translated(x, t, c, k, l)
y2 = splev_wikipedia(l, x, t, c, k)

print(f"These should all match: {y0} {y1} {y2}")
print(f"  err1 = {y1-y0}")
print(f"  err2 = {y2-y0}")



##############################################################################
# OK, good. I want to make sure that the spline roughly follows the curve
# defined by the control points. There should be no x offset or anything of that
# sort
N = 30
t = np.arange( N, dtype=int)
k = 3
c = np.random.rand(len(t))
Npad = 10
x = np.linspace(Npad, N-Npad, 1000)

########### sample_segment_cubic()
@nps.broadcast_define(((),), ())
def sample_cubic(x, cp):
    i = int(x//1)
    q = x-i
    try:    return sample_segment_cubic(q, *cp[i-1:i+3])
    except: return 0

y = sample_cubic(x, c)
c2 = c.copy()
c2[int(N//2)] *= 1.1
y2 = sample_cubic(x, c2)

plot1 = gp.gnuplotlib(title = 'sample_segment_cubic')
plot1.plot( (x, nps.cat(y,y2),
             dict(_with='lines',
                  legend=np.array(('Spline: baseline',
                                   'Spline: tweaked one control point')))),
            (t[:len(c)], nps.cat(c,c2),
             dict(_with='linespoints pt 7 ps 2',
                  legend=np.array(('Control points: baseline',
                                   'Control points: tweaked one control point')))))

########### sample_wikipedia()
@nps.broadcast_define(((),), ())
def sample_wikipedia(x, t, c, k):
    return splev_wikipedia(np.searchsorted(np.array(t), x)-1,x,t,c,k)

@nps.broadcast_define(((),), ())
def sample_wikipedia_integer_knots(x, c, k):
    t = np.arange(len(c) + k, dtype=int)
    l = int(x//1)
    offset = int((k+1)//2)
    return splev_wikipedia(l,x,t,c[offset:],k)

if 1:
    y = sample_wikipedia_integer_knots(x,c,k)
else:
    offset = int((k+1)//2)
    y = sample_wikipedia(x,t, c[offset:], k)

plot2 = gp.gnuplotlib(title = 'sample_wikipedia')
plot2.plot( (x, y, dict(_with='lines')),
            (t[:len(c)], c,  dict(_with='linespoints pt 7 ps 2')) )
print("these two plots should look the same: we're using two implementation of the same algorithm to interpolate the same data")
plot2.wait()
plot1.wait()


# Now I use sympy to get the polynomial coefficients from sample_wikipedia.
# These should match the ones in sample_segment_cubic()

import sympy
c = sympy.symbols(f'c:{len(c)}')
x = sympy.symbols('x')

# Which interval we're in. Arbitrary. In the middle somewhere
l = int(N//2)
print("Should match A,B,C,D coefficients in sample_segment_cubic()")
s = splev_wikipedia(l,
                    # I want 'x' to be [0,1] within the interval, but this
                    # function wants x in the whole domain
                    x+l,
                    np.arange( N, dtype=int), c, k).expand()
print(s.coeff(c[12]))
print(s.coeff(c[13]))
print(s.coeff(c[14]))
print(s.coeff(c[15]))

print("Should also match A,B,C,D coefficients in sample_segment_cubic()")
s = splev_translated(# I want 'x' to be [0,1] within the interval, but this
                     # function wants x in the whole domain
                     x+l,
                     np.arange( N, dtype=int), c, k, l).expand()
print(s.coeff(c[12]))
print(s.coeff(c[13]))
print(s.coeff(c[14]))
print(s.coeff(c[15]))


#########################################################
# Great. More questions. Here I have a cubic spline (k==3). And to evaluate the
# spline value I need to have 4 control point values available, 2 on either
# side. Piecewise-linear splines are too rough, but quadratic splines could work
# (k==2). What do the expressions look like? How many neighbors do I need? Here
# the control point values c represent the function value between adjacent
# knots, so each segment uses 3 neighboring control points, and is defined in a
# region [-0.5..0.5] off the center control point
print("======================== k = 2")
N = 30
t = np.arange( N, dtype=int)
k = 2
# c[0,1,2] corresponds to is t[-0.5 0.5 1.5 2.5 ...
c = np.random.rand(len(t))

x  = 4.3
# may have trouble exactly AT the knots. the edge logic isn't quite what I want
l = np.searchsorted(np.array(t), x)-1
y0 = splev_local(np.array((x,)),t,c,k)[0]
y1 = splev_translated(x, t, c, k, l)
y2 = splev_wikipedia(l, x, t, c, k)
print(f"These should all match: {y0} {y1} {y2}")
print(f"  err1 = {y1-y0}")
print(f"  err2 = {y2-y0}")


##############################################################################
# OK, good. I want to make sure that the spline roughly follows the curve
# defined by the control points. There should be no x offset or anything of that
# sort
N = 30
t = np.arange( N, dtype=int)
k = 2
c = np.random.rand(len(t))
Npad = 10
x = np.linspace(Npad, N-Npad, 1000)
offset = int((k+1)//2)

y = sample_wikipedia(x,t-0.5, c[offset:], k)

xm = (x[1:] + x[:-1]) / 2.
d = np.diff(y) / np.diff(x)
plot1 = gp.gnuplotlib(title = 'k==2; sample_wikipedia')
plot1.plot( (x, y, dict(_with='lines', legend='spline')),
            (xm, d, dict(_with='lines', y2=1, legend='diff')),
            (t[:len(c)], c,  dict(_with='linespoints pt 7 ps 2', legend='control points')))

@nps.broadcast_define(((),), ())
def sample_splev_translated(x, t, c, k):
    l = np.searchsorted(np.array(t), x)-1
    return splev_translated(x,t,c,k,l)
y = sample_splev_translated(x,t-0.5, c[offset:], k)
xm = (x[1:] + x[:-1]) / 2.
d = np.diff(y) / np.diff(x)
plot2 = gp.gnuplotlib(title = 'k==2; splev_translated')
plot2.plot( (x, y, dict(_with='lines', legend='spline')),
            (xm, d, dict(_with='lines', y2=1, legend='diff')),
            (t[:len(c)], c,  dict(_with='linespoints pt 7 ps 2', legend='control points')))


# These are the functions I'm going to use. Derived by the sympy steps
# immediately after this
def sample_segment_quadratic(x, a,b,c):
    A = x**2/2 - x/2 + 1/8
    B = 3/4 - x**2
    C = x**2/2 + x/2 + 1/8
    return A*a + B*b + C*c
@nps.broadcast_define(((),), ())
def sample_quadratic(x, cp):
    i = int((x+0.5)//1)
    q = x-i
    try:    return sample_segment_quadratic(q, *cp[i-1:i+2])
    except: return 0
y = sample_quadratic(x, c)
xm = (x[1:] + x[:-1]) / 2.
d = np.diff(y) / np.diff(x)
plot3 = gp.gnuplotlib(title = 'k==2; sample_quadratic')
plot3.plot( (x, y, dict(_with='lines', legend='spline')),
            (xm, d, dict(_with='lines', y2=1, legend='diff')),
            (t[:len(c)], c,  dict(_with='linespoints pt 7 ps 2', legend='control points')))

plot3.wait()
plot2.wait()
plot1.wait()

print("these 3 plots should look the same: we're using different implementation of the same algorithm to interpolate the same data")


# ##################################
# # OK, these match. Let's get the expression of the polynomial in a segment. This
# # was used to construct sample_segment_quadratic() above
# c = sympy.symbols(f'c:{len(c)}')
# x = sympy.symbols('x')
# l = int(N//2)
# print("A,B,C for k==2 using splev_wikipedia()")
# s = splev_wikipedia(l,
#                     # I want 'x' to be [-0.5..0.5] within the interval, but this
#                     # function wants x in the whole domain
#                     x+l,
#                     np.arange( N, dtype=int) - sympy.Rational(1,2), c, k).expand()
# print(s)
# print(s.coeff(c[13]))
# print(s.coeff(c[14]))
# print(s.coeff(c[15]))
# # I see this:
# #   c13*x**2/2 - c13*x/2 + c13/8 - c14*x**2 + 3*c14/4 + c15*x**2/2 + c15*x/2 + c15/8
# #   x**2/2 - x/2 + 1/8
# #   3/4 - x**2
# #   x**2/2 + x/2 + 1/8


####################################################################
# Great. Final set of questions: how do you we make a 2D spline surface? The
# generic calibration research library
# (https://github.com/puzzlepaint/camera_calibration) Does a set of 1d
# interpolations in one dimension, and then interpolates the 4 interpolated
# values along the other dimension. Questions:
#
# Does order matter? I can do x and then y or y then x
#
# And is there a better way? scipy has 2d b-spline interpolation routines. Do
# they do something better?
#
# ############### x-y or y-x?
import sympy
from sympy.abc import x,y
cp = sympy.symbols('cp:4(:4)')

# x then y
xs = [sample_segment_cubic( x,
                            cp[i*4 + 0],
                            cp[i*4 + 1],
                            cp[i*4 + 2],
                            cp[i*4 + 3] ) for i in range(4)]
yxs = sample_segment_cubic(y, *xs)

# y then x
ys = [sample_segment_cubic( y,
                            cp[0*4 + i],
                            cp[1*4 + i],
                            cp[2*4 + i],
                            cp[3*4 + i] ) for i in range(4)]
xys = sample_segment_cubic(x, *ys)

print(f"Bicubic interpolation. x-then-y and y-then-x difference: {(xys - yxs).expand()}")

########### Alright. Apparently either order is ok. Does scipy do something
########### different for 2d interpolation? I compare the 1d-then-1d
########### interpolation above to fitpack results:

from fpbisp import fpbisp_

N = 30
t = np.arange( N, dtype=int)
k = 3

cp = np.array(sympy.symbols('cp:40(:40)'), dtype=np.object).reshape(40,40)

lx = 3
ly = 5

z   = fpbisp_(t, t, k, cp, x+lx, lx, y+ly, ly)
err = z - yxs

print(f"Bicubic interpolation. difference(1d-then-1d, FITPACK): {err.expand()}")

# Apparently chaining 1D interpolations produces identical results to what FITPACK is doing
