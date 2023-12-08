// Apparently I need this in MSVC to get constants
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "../minimath/minimath.h"

#include "test-harness.h"

/* For everything that is a part of my cost function I need to compute gradients
   in respect to the parameters. Thus I need to re-implement the CAHVOR
   projection. I re-worked the math and simplified the implementation. This file
   is a small test program that makes sure that my simpler implementation
   produces the same result as the reference function.
 */



#define bool_t bool
// basic linear algebra used by the reference cahvor projection function. Comes
// from m-jplv/libmat3/mat3.c
double dot3(
    const double a[3],	/* input vector */
    const double b[3])	/* input vector */
{
    double f;

    /* Check for NULL inputs */
    if ((a == NULL) || (b == NULL))
	return 0.0;

    /* Dot the two vectors */
    f = a[0] * b[0] +
	a[1] * b[1] +
	a[2] * b[2];

    /* Return the dot product */
    return f;
    }

double *sub3(
    const double a[3],	/* input minuend vector */
    const double b[3],	/* input subtrahend vector */
    double c[3])	/* output difference vector */
{
    /* Check for NULL inputs */
    if ((a == NULL) || (b == NULL) || (c == NULL))
	return NULL;

    /* Subtract the two vectors */
    c[0] = a[0] - b[0];
    c[1] = a[1] - b[1];
    c[2] = a[2] - b[2];

    /* Return a pointer to the result */
    return c;
    }
double *scale3(
    double s,		/* input scalar */
    const double a[3],	/* input vector */
    double b[3])	/* output vector */
{
    /* Check for NULL inputs */
    if ((a == NULL) || (b == NULL))
	return NULL;

    /* Perform the scalar multiplication */
    b[0]  =  s * a[0];
    b[1]  =  s * a[1];
    b[2]  =  s * a[2];

    /* Return a pointer to the result */
    return b;
    }
double *add3(
    const double a[3],		/* input addend vector */
    const double b[3],		/* input addend vector */
    double c[3])		/* output sum vector */
{
    /* Check for NULL inputs */
    if ((a == NULL) || (b == NULL) || (c == NULL))
	return NULL;

    /* Add the two vectors */
    c[0] = a[0] + b[0];
    c[1] = a[1] + b[1];
    c[2] = a[2] + b[2];

    /* Return a pointer to the result */
    return c;
    }







// original implementation from m-jplv/libcmod/cmod_cahvor.c, with the gradient
// stuff culled out
static
void cmod_cahvor_3d_to_2d(
    const double pos3[3], /* input 3D position */
    const double c[3],	  /* input model center vector C */
    const double a[3],	  /* input model axis   vector A */
    const double h[3],	  /* input model horiz. vector H */
    const double v[3],	  /* input model vert.  vector V */
    const double o[3],	  /* input model optical axis  O */
    const double r[3],	  /* input model radial-distortion terms R */
    bool_t approx,	  /* input flag to use fast approximation */
    double *range,	  /* output range along A (same units as C) */
    double pos2[2],	  /* output 2D image-plane projection */
    double par[2][3])	  /* output partial-derivative matrix of pos2 to pos3 */
{
    double alpha, beta, gamma, xh, yh;
    double omega, omega_2, tau, mu;
    double p_c[3], pp[3], pp_c[3], wo[3], lambda[3];
    double dldp[3][3], dppdp[3][3], m33[3][3], n33[3][3];
    double dxhdpp[3], dyhdpp[3], v3[3], u3[3];
    double dudt;

    /* Calculate p' and other necessary quantities */
    sub3(pos3, c, p_c);
    omega = dot3(p_c, o);
    omega_2 = omega * omega;
    scale3(omega, o, wo);
    sub3(p_c, wo, lambda);
    tau = dot3(lambda, lambda) / omega_2;
    mu = r[0] + (r[1] * tau) + (r[2] * tau * tau);
    scale3(mu, lambda, pp);
    add3(pos3, pp, pp);

    /* Calculate alpha, beta, gamma, which are (p' - c) */
    /* dotted with a, h, v, respectively                */
    sub3(pp, c, pp_c);
    alpha  = dot3(pp_c, a);
    beta   = dot3(pp_c, h);
    gamma  = dot3(pp_c, v);

    /* Calculate the projection */
    pos2[0] = xh = beta  / alpha;
    pos2[1] = yh = gamma / alpha;
    *range = alpha;
}


// my simplified version
static
void project_cahvor(
                    const double pos3[3], /* input 3D position */
                    const double c[3],	  /* input model center vector C */
                    const double a[3],	  /* input model axis   vector A */
                    const double h[3],	  /* input model horiz. vector H */
                    const double v[3],	  /* input model vert.  vector V */
                    const double o[3],	  /* input model optical axis  O */
                    const double r[3],	  /* input model radial-distortion terms R */
                    double pos2[2]	  /* output 2D image-plane projection */
                    )	  /* output partial-derivative matrix of pos2 to pos3 */
{
    // canonical equivalent:
    // if canonical, I have
    // c = 0,   0,  0
    // a = 0,   0,  1
    // h = fx,  0, cx
    // v =  0, fy, cy
    // o = sin(ph)*cos(th), sin(ph)*sin(th), cos(ph)
    // r = r0, r1, r2

    assert(c[0] == 0 && c[1] == 0 && c[2] == 0);
    assert(a[0] == 0 && a[1] == 0 && a[2] == 1);
    assert(h[1] == 0);
    assert(v[0] == 0);

    double norm2o = norm2_vec(3, o);
    assert( norm2o > 1 - 1e-6 &&
            norm2o < 1 + 1e-6 );

    double norm2p = norm2_vec(3, pos3);
    double omega  = dot_vec(3, pos3, o);
    double tau    = norm2p / (omega*omega) - 1.0;
    double mu     = r[0] + tau*(r[1] + r[2] * tau);

    double px = pos3[0] * (mu + 1.0) - mu*omega*o[0];
    double py = pos3[1] * (mu + 1.0) - mu*omega*o[1];
    double pz = pos3[2] * (mu + 1.0) - mu*omega*o[2];

    double fx = h[0];
    double fy = v[1];
    double cx = h[2];
    double cy = v[2];

    pos2[0] = fx*px/pz + cx;
    pos2[1] = fy*py/pz + cy;
}


int main(int argc, char* argv[])
{
    double c[] = {  0,  0,  0 };
    double a[] = {  0,  0,  1 };
    double h[] = { 11,  0, 23 };
    double v[] = {  0, 13,  7 };
    double o[] = {0.1,0.2,0.9 };
    double r[] = {  7,  8,  2 };

    double o_len = sqrt(norm2_vec(3, o));
    for(int i=0; i<3; i++)
        o[i] /= o_len;

    double p[] = {12, 13, 99};


    double range;
    double projection_orig[2];
    cmod_cahvor_3d_to_2d( p,
                          c,a,h,v,o,r, false,
                          &range, projection_orig, NULL );

    double projection_new[2];
    project_cahvor( p,
                    c,a,h,v,o,r,
                    projection_new );

    // out2: new implementation, with scaled up 3d vector. The scaling should
    // have NO effect
    double projection_new_scaled3d[2];
    for(int i=0;i<3;i++) p[i] *= 10.0;
    project_cahvor( p,
                    c,a,h,v,o,r,
                    projection_new_scaled3d );

    confirm_eq_double(projection_new[0], projection_orig[0], 1e-6);
    confirm_eq_double(projection_new[1], projection_orig[1], 1e-6);

    confirm_eq_double(projection_new_scaled3d[0], projection_orig[0], 1e-6);
    confirm_eq_double(projection_new_scaled3d[1], projection_orig[1], 1e-6);

    TEST_FOOTER();
}
