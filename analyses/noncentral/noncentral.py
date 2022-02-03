#!/usr/bin/python3

import sys
import numpy as np
import numpysane as nps
import re
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

    # Equivalently, I can do this. Looks nice, and possibly IS nicer. However,
    # for small x, this has u -> 0/0, so perhaps the above expressions ARE
    # better

    # magxz = sqrt(x*x + z*z)
    # u = 2 * (magxz - z) / x
    # du_dz = -u/magxz

    return u,du_dz

def dz_nonlinearity(k,u):
    r'''Computings the nonlinear z(u) function

    k is the vector of coefficients. Returns a tuple z, dz/du, dz/dk'''
    if order == 2:
        dz     = u*(k[0] + u*k[1])
        ddz_du = k[0] + 2*u*k[1]
        ddz_dk = sympy.Array((u, u*u))
    elif order == 3:
        dz     = u*(k[0] + u*(k[1] + u*k[2]))
        ddz_du = k[0] + u*(2*k[1] + u*3*k[2])
        ddz_dk = sympy.Array((u, u*u, u*u*u))
    elif order == 4:
        dz     = u*(k[0] + u*(k[1] + u*(k[2] + u*k[3])))
        ddz_du = k[0] + u*(2*k[1] + u*(3*k[2] + u*4*k[3]))
        ddz_dk = sympy.Array((u, u*u, u*u*u, u*u*u*u))
    elif order == 5:
        dz     = u*(k[0] + u*(k[1] + u*(k[2] + u*(k[3] + u*k[4]))))
        ddz_du = k[0] + u*(2*k[1] + u*(3*k[2] + u*(4*k[3] + u*5*k[4])))
        ddz_dk = sympy.Array((u, u*u, u*u*u, u*u*u*u, u*u*u*u*u))
    else:
        raise
    return dz, ddz_du, ddz_dk


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

    dz_approx,ddz_approx,_ = dz_nonlinearity(k,u)

    e = (z_adjusted-z) - dz_approx

    if not get_gradients:
        return e

    de_dzadj = 1 - du_dzadj*ddz_approx

    return e, de_dzadj


order = 5
k = (1e-2, 1e-1, -2e-2, 1e-2, 1e-3)
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

      dz = (z_adjusted-z) = dz(k,u)

    dz is currently a polynomial, such as u*(k[0] + u*k[1]), but it could
    potentially be anything

    This is nonlinear, and I approximate the solution with a single Newton step,
    starting at z0 = z:

      E(zadj) = (zadj-z) - dz(k,u(zadj))

      E(zadj0 + dzadj) ~ E(zadj0) + dE/dzadj dzadj = 0 ->
      dzadj = -E(zadj0) / dE/dzadj ->

      zadj ~ zadj0 + dzadj =
           = zadj0 - E(zadj0) / dE/dzadj
           = z  - (0 - dz(k,u(z))) / (1 - ddz/du du/dz)
           = z  + dz(k,u(z)) / (1 - ddz/du du/dz)

    Here "u" is the scalar u that considers a 2D observation vector (mag(xy), z).

    We have

      u = 2 * sqrt(x*x + y*y) / (z + sqrt(x*x + y*y + z*z))

    '''

    magxy  = sqrt(x*x + y*y)
    magxyz = sqrt(x*x + y*y + z*z)

    u,du_dz = uscalar_func(magxy, z)
    dz, ddz_du, _ = dz_nonlinearity(k,u)

    return z + dz / (1 - ddz_du*du_dz)


def disp(x):
    sympy.preview(x, output='pdf', viewer='mupdf')


k = sympy.symbols(f'k:{order}', real = True)
x,y,z  = sympy.symbols('x,y,z', real = True, positive = True)
f      = sympy.symbols('f',      real = True, positive = True)
u      = sympy.symbols('u',      real = True, positive = True)
d      = sympy.symbols('d',      real = True, positive = True)
dz     = sympy.symbols('dz',     real = True, positive = True)
ddz_du = sympy.symbols('ddz_du', real = True, positive = True)
ddz_dk = sympy.symbols('ddz_dk', real = True, positive = True)
magxyz = sympy.symbols('magxyz', real = True, positive = True)
magxy  = sympy.symbols('magxy',  real = True, positive = True)
du_dx  = sympy.symbols('du_dx',  real = True, positive = True)
dd_dx  = sympy.symbols('dd_dx',  real = True, positive = True)
du_dy  = sympy.symbols('du_dy',  real = True, positive = True)
dd_dy  = sympy.symbols('dd_dy',  real = True, positive = True)
du_dz  = sympy.symbols('du_dz',  real = True, positive = True)
dd_dz  = sympy.symbols('dd_dz',  real = True, positive = True)

dz_expr, ddz_du_expr, ddz_dk_expr = dz_nonlinearity(k,u)
d_expr = 1 + u * ddz_du_expr / magxyz

zadj = zadj_func(x,y,z,k)                                         \
           .subs(z + sqrt(x**2 + y**2 + z**2), 2*sqrt(x*x+y*y)/u) \
           .subs(sqrt(x**2 + y**2 + z**2),     magxyz)            \
           .subs(sqrt(x**2 + y**2),            magxy)             \
           .subs(1 + z/magxyz,                 2*magxy/u/magxyz)  \
           .subs(dz_expr, dz)                                     \
           .subs(d_expr,  d)
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

# factor common in the expressions
f_expr = 0
for p in range(order-1,-1,-1):
    f_expr = ((p+1)*(p+1)*k[p] + f_expr)*u
f_expr /= u

# manual factor
_dd_dx_expr = du_dx*f/magxyz - x*(d-1)/magxyz**2
if (_dd_dx_expr.subs(d,d_expr).subs(f,f_expr) - dd_dx_expr).expand() != 0:
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
_dd_dz_expr = -1/magxyz**2*( u*f + z*(d-1))
if (_dd_dz_expr.subs(d,d_expr).subs(f,f_expr) - dd_dz_expr.subs(du_dz,du_dz_expr)).expand() != 0:
    raise Exception("manual factor is wrong. Fix _dd_dz_expr")
dd_dz_expr = _dd_dz_expr

dzadj_dx = zadj                          \
    .subs(dz,     Function('dz')(u))     \
    .subs(u,      Function('u' )(x,y,z)) \
    .subs(d,      Function('d' )(x,y,z)) \
    .diff(x)                             \
    .subs(Function('d' )(x,y,z), d)      \
    .subs(Function('u' )(x,y,z), u)      \
    .subs(Function('dz')(u),     dz)     \
    .subs(Derivative(d,  x), dd_dx)      \
    .subs(Derivative(u,  x), du_dx)      \
    .subs(Derivative(dz, u), ddz_du)     \
    .factor()

dzadj_dz = zadj                          \
    .subs(dz,     Function('dz')(u))     \
    .subs(u,      Function('u' )(x,y,z)) \
    .subs(d,      Function('d' )(x,y,z)) \
    .diff(z)                             \
    .subs(Function('d' )(x,y,z), d)      \
    .subs(Function('u' )(x,y,z), u)      \
    .subs(Function('dz')(u),     dz)     \
    .subs(Derivative(d,  z), dd_dz)      \
    .subs(Derivative(u,  z), du_dz)      \
    .subs(Derivative(dz, u), ddz_du)     \
    .factor()

dzadj_dk =                                                    \
    [ zadj                                                    \
      .subs(dz, Function('dz')(*k))                           \
      .subs(d,  Function('d')(*k))                            \
      .diff(k[i])                                             \
      .subs(Derivative(Function('d' )(*k), k[i]), d_expr.diff(k[i])) \
      .subs(Derivative(Function('dz')(*k), k[i]), ddz_dk_expr[i])    \
      .subs(Function('d' )(*k), d)                            \
      .subs(Function('dz')(*k), dz)                           \
      for i in range(len(k)) ]

# All the gradients are now done, except dzadj/dy. This works just like
# dzadj/dx, so I do this via a substitution
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
              dk1 = 0,
              dk2 = 0,
              dk3 = 0,
              dk4 = 0):

    v = dict(vars0)
    v['x']  += dx
    v['y']  += dy
    v['z']  += dz
    v['k0'] += dk0
    v['k1'] += dk1
    v['k2'] += dk2
    v['k3'] += dk3
    v['k4'] += dk4

    def make_symbol_table(v):
        r'''Rebuilt table from string keys to sympy object keys'''
        g  = globals()
        vv = dict()
        for k in v.keys():
            m = re.match("k([0-9]+)$", k)
            if m:
                ik = int(m.group(1))
                try:
                    vv[g['k'][ik]] = v[k]
                except:
                    # I don't have this k. Probably this is is past the order
                    # I'm using
                    pass
            else:
                vv[g[k]] = v[k]
        return vv

    v['magxy' ] = sqrt(x*x + y*y      ).subs(make_symbol_table(v))
    v['magxyz'] = sqrt(x*x + y*y + z*z).subs(make_symbol_table(v))
    v['u']      = u_expr .subs(make_symbol_table(v))
    v['d']      = d_expr .subs(make_symbol_table(v))
    v['f']      = f_expr .subs(make_symbol_table(v))
    v['dz']     = dz_expr.subs(make_symbol_table(v))
    v['du_dx']  = du_dx_expr.subs(make_symbol_table(v))
    v['dd_dx']  = dd_dx_expr.subs(make_symbol_table(v))
    v['du_dy']  = du_dy_expr.subs(make_symbol_table(v))
    v['dd_dy']  = dd_dy_expr.subs(make_symbol_table(v))
    v['du_dz']  = du_dz_expr.subs(make_symbol_table(v))
    v['dd_dz']  = dd_dz_expr.subs(make_symbol_table(v))
    v['ddz_du'] = ddz_du_expr.subs(make_symbol_table(v))

    return float(f.subs(make_symbol_table(v)))


delta = 1e-6
vars0 = dict( x =  1.1,
              y =  2.3,
              z =  4.2,
              k0 = 0.1,
              k1 = 0.3,
              k2 = 0.5,
              k3 = 0.03,
              k4 = 0.01 )


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

for i in range(order):
    dki = f"dk{i}"
    zadj0 = subs_into(zadj, vars0, **{dki: -delta})
    zadj1 = subs_into(zadj, vars0, **{dki:  delta})
    dzadj_dki_observed = (zadj1 - zadj0) / (2 * delta)
    dzadj_dki_expected = subs_into(dzadj_dk[i], vars0)
    print(f"dzadj/dk{i} relative error: {(dzadj_dki_observed-dzadj_dki_expected)/((np.abs(dzadj_dki_observed)+np.abs(dzadj_dki_expected))/2.):.2g}")


#################################
# And finally, I use the final expressions to once again confirm that they work
# as an approximation to the solution of the nonlinear equation above
if order == 2:
    vars0k = (vars0['k0'],vars0['k1'])
elif order == 3:
    vars0k = (vars0['k0'],vars0['k1'],vars0['k2'])
elif order == 4:
    vars0k = (vars0['k0'],vars0['k1'],vars0['k2'],vars0['k3'])
elif order == 5:
    vars0k = (vars0['k0'],vars0['k1'],vars0['k2'],vars0['k3'],vars0['k4'])
else:
    raise
dz_adj_perfect = \
    scipy.optimize.newton(err,
                          x0   = vars0['z'],
                          args = (vars0k,
                                  np.sqrt(vars0['x']*vars0['x'] +
                                          vars0['y']*vars0['y']),
                                  vars0['z'])) - vars0['z']

dz_adjusted_approximation = subs_into(zadj, vars0) - vars0['z']
print(f"relative error in dz_adjusted: {(dz_adj_perfect - dz_adjusted_approximation)/ ((np.abs(dz_adj_perfect)+np.abs(dz_adjusted_approximation))/2):.2g}")




# All done. Output the expressions
from sympy.printing import ccode
from sympy.codegen.rewriting import create_expand_pow_optimization

print('')

print(f"        const double magxy    = {ccode(create_expand_pow_optimization(5)(sqrt(x*x + y*y)))};")
print(f"        const double magxyz   = {ccode(create_expand_pow_optimization(5)(sqrt(x*x + y*y + z*z)))};")
print(f"        const double u        = {ccode(create_expand_pow_optimization(5)(u_expr))};")
print(f"        const double d        = {ccode(create_expand_pow_optimization(5)(d_expr))};")
print(f"        const double f        = {ccode(create_expand_pow_optimization(5)(f_expr))};")
print(f"        const double dz       = {ccode(create_expand_pow_optimization(5)(dz_expr))};")
print(f"        zadj                  = {ccode(create_expand_pow_optimization(5)(zadj))};")

print(f"        const double ddz_du   = {ccode(create_expand_pow_optimization(5)(ddz_du_expr))};")
print(f"        const double du_dx    = {ccode(create_expand_pow_optimization(5)(du_dx_expr))};")
print(f"        const double du_dy    = {ccode(create_expand_pow_optimization(5)(du_dy_expr))};")
print(f"        const double du_dz    = {ccode(create_expand_pow_optimization(5)(du_dz_expr))};")

print(f"        const double dd_dx    = {ccode(create_expand_pow_optimization(5)(dd_dx_expr))};")
print(f"        const double dd_dy    = {ccode(create_expand_pow_optimization(5)(dd_dy_expr))};")
print(f"        const double dd_dz    = {ccode(create_expand_pow_optimization(5)(dd_dz_expr))};")

print(f"        const double dzadj_dx = {ccode(create_expand_pow_optimization(5)(dzadj_dx))};")
print(f"        const double dzadj_dy = {ccode(create_expand_pow_optimization(5)(dzadj_dy))};")
print(f"        const double dzadj_dz = {ccode(create_expand_pow_optimization(5)(dzadj_dz))};")

for i in range(order):
    print(f"        const double dzadj_dk{i} = {ccode(create_expand_pow_optimization(5)(dzadj_dk[i]))};")
