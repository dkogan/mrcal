#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import scipy.optimize
import sympy
import sympy.abc
from sympy import Derivative,Function

def sqrt(x):
    try:    return np.sqrt(x)
    except: return sympy.sqrt(x)

def uscalar_func(x,z):
    r'''2D stereographic projection

    2D point (x,z) -> 1D u'''
    denom = z + sqrt(x*x + z*z)
    u = 2 * x / denom
    du_dz = -2 * x / (denom*denom) * (1 + z/sqrt(x*x + z*z))
    return u,du_dz

def err(z_adjusted, # hypothesis
        k,x,z,      # known
        get_gradients = False
       ):

    r'''Reports the fit error for a given hypothesis z_adjusted

    I compute a scalar u (looking at the (xy,z_adjusted) direction vector) and
    use a simple polynomial to then get the z shift z_adjusted-z. This creates a
    nonlinear equation that I can solve for z_adjusted.

    This is the error function that can be minimized to solve this nonlinear
    equation

    '''

    u,du_dzadj = uscalar_func(x,z_adjusted)

    e = \
        (z_adjusted-z) - \
        u*(k[0] + u*k[1])

    if not get_gradients:
        return e

    de_dzadj = 1 - du_dzadj*(k[0] + 2*k[1])
    return e, de_dzadj


k = (1e-2, 1e-1)
x = 10.
z = 1.

z_adjusted = \
    scipy.optimize.newton(err,
                          x0   = z,
                          args = (k,x,z),)

# one step: 0 = err0 + dderr/dzadj dzadj -> dzadj = -err0 / dderr/dzadj
e, de_dzadj = \
    err( z,
         k,x,z,
         get_gradients = True)
zadj_onestep = z - e/de_dzadj

print(f"relative error in z_adjusted: {(z_adjusted - zadj_onestep)/ ((np.abs(z_adjusted)+np.abs(zadj_onestep))/2):.2g}")


####################


def zadj_func(x,y,z, k):
    r'''Reports zadj based on the observation vector (x,y,z)

    I want to find z_adjusted that satisfies

      (z_adjusted-z) = u*(k[0] + u*k[1])

    Where u is u(x,y,z_adjusted)

    This is nonlinear, and I approximate the solution with a single Newton step,
    starting at z0 = z:

      E(z_adjusted) = (z_adjusted-z) - u(z_adjusted)*(k[0] + u(z_adjusted)*k[1])

      E(z0 + dz) ~ E(z0) + dE/dz dz = 0 ->
      dz = -E(z0) / dE/dz ->

      z_adjusted ~ z0 + dz =
                 = z0 - E(z0) / dE/dz
                 = z  - (0 - u*(k[0] + u*k[1])) / (1 - du/dz*(k[0] + 2*u*k[1]))
                 = z  + u*(k[0] + u*k[1]) / (1 - du/dz*(k[0] + 2*u*k[1]))

    Here "u" is the scalar u that considers a 2D observation vector (mag(xy), z).

    We have

      u = 2 * sqrt(x*x + y*y) / (z + sqrt(x*x + y*y + z*z))

    '''

    magxy  = sqrt(x*x + y*y)
    magxyz = sqrt(x*x + y*y + z*z)

    u,du_dz = uscalar_func(magxy, z)
    return z + (u*(k[0] + u*k[1])) / (1 - du_dz*(k[0] + 2*k[1]*u) )

def disp(x):
    sympy.preview(x, output='pdf', viewer='mupdf')


k     = sympy.symbols('k:2',   real = True)
k0,k1 = k
x,y,z = sympy.symbols('x,y,z', real = True, positive = True)

u      = sympy.symbols('u',      real = True, positive = True)
d      = sympy.symbols('d',      real = True, positive = True)
magxyz = sympy.symbols('magxyz', real = True, positive = True)
magxy  = sympy.symbols('magxy',  real = True, positive = True)
du_dx  = sympy.symbols('du_dx',  real = True, positive = True)
dd_dx  = sympy.symbols('dd_dx',  real = True, positive = True)
du_dy  = sympy.symbols('du_dy',  real = True, positive = True)
dd_dy  = sympy.symbols('dd_dy',  real = True, positive = True)
du_dz  = sympy.symbols('du_dz',  real = True, positive = True)
dd_dz  = sympy.symbols('dd_dz',  real = True, positive = True)


d_expr = 1 + u*(k0 + 2*k1*u)/magxyz


zadj = zadj_func(x,y,z,k)                                         \
           .subs(z + sqrt(x**2 + y**2 + z**2), 2*sqrt(x*x+y*y)/u) \
           .subs(sqrt(x**2 + y**2 + z**2),     magxyz)            \
           .subs(sqrt(x**2 + y**2),            magxy)             \
           .subs(1 + z/magxyz,                 2*magxy/u/magxyz)  \
           .subs(d_expr, d)
print(f"zadj = {zadj}")

u_expr = uscalar_func(sqrt(x*x + y*y), z)[0]
du_dx_expr = u_expr                                                        \
                 .subs(sqrt(x**2 + y**2 + z**2),                 magxyz)   \
                 .subs(sqrt(x**2 + y**2),                        magxy)    \
                 .subs(magxy , Function('magxy') (x,y))                    \
                 .subs(magxyz, Function('magxyz')(x,y,z))                  \
                 .diff(x)                                                  \
                 .subs(Derivative(Function('magxy') (x,y), x),   x/magxy)  \
                 .subs(Derivative(Function('magxyz')(x,y,z), x), x/magxyz) \
                 .subs(Function('magxy') (x,y),                  magxy)    \
                 .subs(Function('magxyz')(x,y,z),                magxyz)   \
                 .subs(z + magxyz, 2*magxy/u)
# manual factor
_du_dx_expr = u*x/magxy*( -u/(2*magxyz) + 1/magxy )
if (_du_dx_expr - du_dx_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _du_dx_expr")
du_dx_expr = _du_dx_expr

du_dz_expr = u_expr                                                        \
                 .subs(sqrt(x**2 + y**2 + z**2),                 magxyz)   \
                 .subs(sqrt(x**2 + y**2),                        magxy)    \
                 .subs(magxyz, Function('magxyz')(x,y,z))                  \
                 .diff(z)                                                  \
                 .subs(Derivative(Function('magxyz')(x,y,z), z), z/magxyz) \
                 .subs(Function('magxyz')(x,y,z),                magxyz)   \
                 .subs(z + magxyz, 2*magxy/u)
# manual factor
_du_dz_expr = -u/magxyz
if (du_dz_expr.factor() - _du_dz_expr).subs(u, 2*magxy/(z+magxyz)).expand() != 0:
    raise Exception("manual factor is wrong. Fix _du_dz_expr")
du_dz_expr = _du_dz_expr


dd_dx_expr = d_expr.subs(u,      Function('u')     (x,y,z))                  \
                   .subs(magxyz, Function('magxyz')(x,y,z))                  \
                   .diff(x)                                                  \
                   .subs(Derivative(Function('magxyz')(x,y,z), x), x/magxyz) \
                   .subs(Derivative(Function('u')(x,y,z), x),      du_dx)    \
                   .subs(Function('magxyz')(x,y,z),                magxyz)   \
                   .subs(Function('u')     (x,y,z),                u)
# manual factor
_dd_dx_expr = du_dx*(k0 + 4*k1*u)/magxyz - x*(d-1)/magxyz**2
if (_dd_dx_expr.subs(d,d_expr) - dd_dx_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dd_dx_expr")
dd_dx_expr = _dd_dx_expr

dd_dz_expr = d_expr.subs(u,      Function('u')     (x,y,z))                  \
                   .subs(magxyz, Function('magxyz')(x,y,z))                  \
                   .diff(z)                                                  \
                   .subs(Derivative(Function('magxyz')(x,y,z), z), z/magxyz) \
                   .subs(Derivative(Function('u')(x,y,z), z),      du_dz)    \
                   .subs(Function('magxyz')(x,y,z),                magxyz)   \
                   .subs(Function('u')     (x,y,z),                u)
# manual factor
_dd_dz_expr = -1/magxyz**2*( u*(k0 + 4*k1*u) + z*(d-1))
if (_dd_dz_expr.subs(d,d_expr) - dd_dz_expr.subs(du_dz,du_dz_expr)).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dd_dz_expr")
dd_dz_expr = _dd_dz_expr

dzadj_dx = zadj \
    .subs(u,      Function('u')     (x,y,z)) \
    .subs(d,      Function('d')     (x,y,z)) \
    .subs(magxyz, Function('magxyz')(x,y,z)) \
    .diff(x) \
    .subs(Derivative(Function('magxyz')(x,y,z), x), x/magxyz) \
    .subs(Derivative(Function('u')(x,y,z), x),      du_dx) \
    .subs(Derivative(Function('d')(x,y,z), x),      dd_dx) \
    .subs(Function('magxyz')(x,y,z),                magxyz) \
    .subs(Function('u')     (x,y,z),                u) \
    .subs(Function('d')     (x,y,z),                d) \
    .subs(magxy,                                    u*(z+magxyz)/2)
# manual factor
_dzadj_dx = du_dx*(k0 + 2*k1*u)/d - dd_dx*u*(k0 + k1*u)/d**2
if (_dzadj_dx - dzadj_dx).subs(d,d_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dzadj_dx")
dzadj_dx = _dzadj_dx

dzadj_dz = zadj \
    .subs(u,      Function('u')     (x,y,z)) \
    .subs(d,      Function('d')     (x,y,z)) \
    .subs(magxyz, Function('magxyz')(x,y,z)) \
    .diff(z) \
    .subs(Derivative(Function('magxyz')(x,y,z), z), z/magxyz) \
    .subs(Derivative(Function('u')(x,y,z), z),      du_dz) \
    .subs(Derivative(Function('d')(x,y,z), z),      dd_dz) \
    .subs(Function('magxyz')(x,y,z),                magxyz) \
    .subs(Function('u')     (x,y,z),                u) \
    .subs(Function('d')     (x,y,z),                d) \
    .subs(magxy,                                    u*(z+magxyz)/2)
# manual factor
_dzadj_dz = 1 + du_dz*(k0 + 2*k1*u)/d - dd_dz*u*(k0 + k1*u)/d**2
if (_dzadj_dz - dzadj_dz).subs(d,d_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dzadj_dz")
dzadj_dz = _dzadj_dz

dzadj_dk0 = zadj                                                 \
    .subs(d,         Function('d')(k0,k1))                       \
    .diff(k0)                                                    \
    .subs(Derivative(Function('d')(k0,k1), k0), d_expr.diff(k0)) \
    .subs(Function('d')(k0,k1),                 d)
# manual factor
_dzadj_dk0 = u/d*(  1 - u*(k0 + k1*u)/(d*magxyz) )
if (_dzadj_dk0 - dzadj_dk0).subs(d,d_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dzadj_dk0")
dzadj_dk0 = _dzadj_dk0

dzadj_dk1 = zadj                                                 \
    .subs(d,         Function('d')(k0,k1))                       \
    .diff(k1)                                                    \
    .subs(Derivative(Function('d')(k0,k1), k1), d_expr.diff(k1)) \
    .subs(Function('d')(k0,k1),                 d)
# manual factor
_dzadj_dk1 = u**2/d*(  1 - 2*u*(k0 + k1*u)/(d*magxyz))
if (_dzadj_dk1 - dzadj_dk1).subs(d,d_expr).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dzadj_dk1")
dzadj_dk1 = _dzadj_dk1

# All the gradients are done, except dzadj/dy. This works just like dzadj/dx, so
# I do this via a substitution
du_dy_expr = du_dx_expr \
    .subs(x,y)
dd_dy_expr = dd_dx_expr \
    .subs(x,y)          \
    .subs(du_dx, du_dy)
dzadj_dy = dzadj_dx \
    .subs(x,y)          \
    .subs(du_dx, du_dy) \
    .subs(dd_dx, dd_dy) \


####################
# Got all the expressions. Check the gradients using central differences
def subs_into(f,
              vars0,
              dx  = 0,
              dy  = 0,
              dz  = 0,
              dk0 = 0,
              dk1 = 0):

    v = dict(vars0)
    v['x']  += dx
    v['y']  += dy
    v['z']  += dz
    v['k0'] += dk0
    v['k1'] += dk1

    def make_symbol_table(v):
        r'''Rebuilt table from string keys to sympy object keys'''
        g  = globals()
        vv = dict()
        for k in v.keys():
            vv[g[k]] = v[k]
        return vv

    v['magxy' ] = sqrt(x*x + y*y      ).subs(make_symbol_table(v))
    v['magxyz'] = sqrt(x*x + y*y + z*z).subs(make_symbol_table(v))
    v['u']      = u_expr.subs(make_symbol_table(v))
    v['d']      = d_expr.subs(make_symbol_table(v))
    v['du_dx']  = du_dx_expr.subs(make_symbol_table(v))
    v['dd_dx']  = dd_dx_expr.subs(make_symbol_table(v))
    v['du_dy']  = du_dy_expr.subs(make_symbol_table(v))
    v['dd_dy']  = dd_dy_expr.subs(make_symbol_table(v))
    v['du_dz']  = du_dz_expr.subs(make_symbol_table(v))
    v['dd_dz']  = dd_dz_expr.subs(make_symbol_table(v))

    return float(f.subs(make_symbol_table(v)))


delta = 1e-6
vars0 = dict( x =  1.1,
              y =  2.3,
              z =  4.2,
              k0 = 0.1,
              k1 = 0.3 )


zadj0 = subs_into(zadj, vars0, dx = -delta)
zadj1 = subs_into(zadj, vars0, dx =  delta)
dzadj_dx_observed = (zadj1 - zadj0) / (2 * delta)
dzadj_dx_expected = subs_into(dzadj_dx, vars0)
print(f"dzadj/dx relative error: {(dzadj_dx_observed-dzadj_dx_expected)/((np.abs(dzadj_dx_observed)+np.abs(dzadj_dx_expected))/2.):.2g}")

zadj0 = subs_into(zadj, vars0, dy = -delta)
zadj1 = subs_into(zadj, vars0, dy =  delta)
dzadj_dy_observed = (zadj1 - zadj0) / (2 * delta)
dzadj_dy_expected = subs_into(dzadj_dy, vars0)
print(f"dzadj/dy relative error: {(dzadj_dy_observed-dzadj_dy_expected)/((np.abs(dzadj_dy_observed)+np.abs(dzadj_dy_expected))/2.):.2g}")

zadj0 = subs_into(zadj, vars0, dz = -delta)
zadj1 = subs_into(zadj, vars0, dz =  delta)
dzadj_dz_observed = (zadj1 - zadj0) / (2 * delta)
dzadj_dz_expected = subs_into(dzadj_dz, vars0)
print(f"dzadj/dz relative error: {(dzadj_dz_observed-dzadj_dz_expected)/((np.abs(dzadj_dz_observed)+np.abs(dzadj_dz_expected))/2.):.2g}")

zadj0 = subs_into(zadj, vars0, dk0 = -delta)
zadj1 = subs_into(zadj, vars0, dk0 =  delta)
dzadj_dk0_observed = (zadj1 - zadj0) / (2 * delta)
dzadj_dk0_expected = subs_into(dzadj_dk0, vars0)
print(f"dzadj/dk0 relative error: {(dzadj_dk0_observed-dzadj_dk0_expected)/((np.abs(dzadj_dk0_observed)+np.abs(dzadj_dk0_expected))/2.):.2g}")

zadj0 = subs_into(zadj, vars0, dk1 = -delta)
zadj1 = subs_into(zadj, vars0, dk1 =  delta)
dzadj_dk1_observed = (zadj1 - zadj0) / (2 * delta)
dzadj_dk1_expected = subs_into(dzadj_dk1, vars0)
print(f"dzadj/dk1 relative error: {(dzadj_dk1_observed-dzadj_dk1_expected)/((np.abs(dzadj_dk1_observed)+np.abs(dzadj_dk1_expected))/2.):.2g}")


#################################
# And finally, I use the final expressions to once again confirm that they work
# as an approximation to the solution of the nonlinear equation above

dz_adj_perfect = \
    scipy.optimize.newton(err,
                          x0   = vars0['z'],
                          args = ((vars0['k0'],vars0['k1']),
                                  np.sqrt(vars0['x']*vars0['x'] +
                                          vars0['y']*vars0['y']),
                                  vars0['z'])) - vars0['z']

dz_adjusted_approximation = subs_into(zadj, vars0) - vars0['z']
print(f"relative error in dz_adjusted: {(dz_adj_perfect - dz_adjusted_approximation)/ ((np.abs(dz_adj_perfect)+np.abs(dz_adjusted_approximation))/2):.2g}")




# All done. Output the expressions
from sympy.printing import ccode
from sympy.codegen.rewriting import create_expand_pow_optimization

print('')

print(f"        const double magxy    = {ccode(create_expand_pow_optimization(2)(sqrt(x*x + y*y)))};")
print(f"        const double magxyz   = {ccode(create_expand_pow_optimization(2)(sqrt(x*x + y*y + z*z)))};")
print(f"        const double u        = {ccode(create_expand_pow_optimization(2)(u_expr))};")
print(f"        const double d        = {ccode(create_expand_pow_optimization(2)(d_expr))};")
print(f"        const double zadj     = {ccode(create_expand_pow_optimization(2)(zadj))};")

print(f"        const double du_dx    = {ccode(create_expand_pow_optimization(2)(du_dx_expr))};")
print(f"        const double du_dy    = {ccode(create_expand_pow_optimization(2)(du_dy_expr))};")
print(f"        const double du_dz    = {ccode(create_expand_pow_optimization(2)(du_dz_expr))};")

print(f"        const double dd_dx    = {ccode(create_expand_pow_optimization(2)(dd_dx_expr))};")
print(f"        const double dd_dy    = {ccode(create_expand_pow_optimization(2)(dd_dy_expr))};")
print(f"        const double dd_dz    = {ccode(create_expand_pow_optimization(2)(dd_dz_expr))};")

print(f"        const double dzadj_dx = {ccode(create_expand_pow_optimization(2)(dzadj_dx))};")
print(f"        const double dzadj_dy = {ccode(create_expand_pow_optimization(2)(dzadj_dy))};")
print(f"        const double dzadj_dz = {ccode(create_expand_pow_optimization(2)(dzadj_dz))};")

print(f"        const double dzadj_dk0 = {ccode(create_expand_pow_optimization(2)(dzadj_dk0))};")
print(f"        const double dzadj_dk1 = {ccode(create_expand_pow_optimization(2)(dzadj_dk1))};")
