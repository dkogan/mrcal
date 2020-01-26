#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include "minimath.h"

#include "poseutils.h"

// All arrays stored in contiguous matrices in row-major order
//
// I have two different representations of pose transformations:
//
// - Rt is a concatenated (4,3) array: Rt = nps.glue(R,t, axis=-2). The
//   transformation is R*x+t
//
// - rt is a concatenated (6) array: rt = nps.glue(r,t, axis=-1). The
//   transformation is R*x+t where R = R_from_r(r)

// Make an identity rotation or transformation
void mrcal_identity_R (double* R  /* (3,3) array */)
{
    R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
    R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
    R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;
}
void mrcal_identity_r (double* r  /* (3)   array */)
{
    for(int i=0; i<3; i++) r[i] = 0.0;
}
void mrcal_identity_Rt(double* Rt /* (4,3) array */)
{
    mrcal_identity_R(&Rt[0]);
    for(int i=0; i<3; i++) Rt[i+9] = 0.0;
}
void mrcal_identity_rt(double* rt /* (6)   array */)
{
    mrcal_identity_r(&rt[0]);
    for(int i=0; i<3; i++) rt[i+3] = 0.0;
}

void mrcal_rotate_point_R( // output
                          double* x_out, // (3) array
                          double* J_R,   // (3,3,3) array. May be NULL
                          double* J_x,   // (3,3) array. May be NULL

                          // input
                          const double* R,
                          const double* x_in
                         )
{
    // R*x
    mul_vec3_gen33t_vout(x_in, R, x_out);
    if(J_R)
        for(int i=0; i<3; i++)
        {
            int j=0;
            for(; j<i*3; j++)
                J_R[i*9 + j] = 0.0;
            for(int k=0; k<3; k++)
                J_R[i*9 + j + k] = x_in[k];
            for(j+=3; j<9; j++)
                J_R[i*9 + j] = 0.0;
        }
    if(J_x)
        for(int i=0; i<9; i++)
            J_x[i] = R[i];
}


// mrcal_rotate_point_r() uses auto-differentiation, so it's implemented in C++
// in poseutils-hasgrad.cc


// Apply a transformation to a point
void mrcal_transform_point_Rt(// output
                              double* x_out, // (3) array
                              double* J_R,   // (3,3,3) array. Gradient.
                                             // Flattened R. May be NULL
                              double* J_t,   // (3,3) array. Gradient.
                                             // Flattened Rt. May be NULL
                              double* J_x,   // (3,3) array. Gradient. May be
                                             // NULL
                              // input
                              const double* Rt,  // (4,3) array
                              const double* x_in // (3) array
                              )
{
    // I want R*x + t
    // First R*x
    mrcal_rotate_point_R(x_out, J_R, J_x,
                         Rt, x_in);

    // And now +t. The J_R, J_x gradients are unaffected. J_t is identity
    add_vec(3, x_out, &Rt[9]);
    if(J_t)
    {
        J_t[0] = 1.0; J_t[1] = 0.0; J_t[2] = 0.0;
        J_t[3] = 0.0; J_t[4] = 1.0; J_t[5] = 0.0;
        J_t[6] = 0.0; J_t[7] = 0.0; J_t[8] = 1.0;
    }
}

void mrcal_transform_point_rt(// output
                              double* x_out, // (3) array
                              double* J_r,   // (3,3) array. Gradient. May be
                                             // NULL
                              double* J_t,   // (3,3) array. Gradient. May be
                                             // NULL
                              double* J_x,   // (3,3) array. Gradient. May be
                                             // NULL
                              // input
                              const double* rt,  // (6) array
                              const double* x_in // (3) array
                              )
{
    // I want rotate(x) + t
    // First rotate(x)
    mrcal_rotate_point_r(x_out, J_r, J_x,
                         rt, x_in);

    // And now +t. The J_r, J_x gradients are unaffected. J_t is identity
    add_vec(3, x_out, &rt[3]);
    if(J_t)
    {
        J_t[0] = 1.0; J_t[1] = 0.0; J_t[2] = 0.0;
        J_t[3] = 0.0; J_t[4] = 1.0; J_t[5] = 0.0;
        J_t[6] = 0.0; J_t[7] = 0.0; J_t[8] = 1.0;
    }
}

// The implementations of mrcal_r_from_R and mrcal_R_from_r are based on opencv.
// The sources have been heavily modified, but the opencv logic remains.
//
// from opencv-4.1.2+dfsg/modules/calib3d/src/calibration.cpp
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of

// Convert a rotation representation from a matrix to a Rodrigues vector
void mrcal_r_from_R_cv( // output
                    double* r, // (3) vector
                    double* J, // (3,3,3) array. Gradient. May be NULL

                    // input
                    const double* R // (3,3) array
                   )
{
    if( J )
        memset( J, 0, 3*9*sizeof(J[0]) );


    // We're given a rotation matrix. It might not exactly be a rotation, so
    // opencv was finding a nearby rotation by computing the SVD: R = U S Vt,
    // and using R' = U Vt. This R' is guaranteed to be orthonormal (although
    // opencv wasn't checking for mirroring). For my purposes I don't bother to
    // do any of this, and I simply assume that I was given a valid rotations,
    // and I just use that. Enable this code block to take this extra step
#if 0
    // SVD from LAPACK
    int dgesvd_(char *jobu, char *jobvt, int *m, int *n,
                double *a, int *lda, double *s, double *u, int *
                ldu, double *vt, int *ldvt, double *work, int *lwork,
                int *info, int jobu_len, int jobvt_len);

    // I compute the SVD. LAPACK uses col-major ordering, so I must judiciously
    // think about transposes. I'm actually giving it transpose(R), so it's
    // computing v s ut. It's returning v and ut as transposes, so I'm actually
    // getting vt and u
    double U[9];
    double Vt[9];
    double d[3]; // singular values
    int lwork = 64; // arbitrary. docs say at least 5*3, but more is better
    double work[lwork];
    int info;
    double UVt[9];

    // dgesvd destroys the input, so I make a copy, and operate on the copy
    memcpy(UVt, R, sizeof(UVt));
    dgesvd_( &(char){'A'}, &(char){'A'}, &(int){3}, &(int){3},
             UVt,
             &(int){3},
             d,
             Vt, &(int){3},
             U,  &(int){3},
             work, &lwork, &info, 1, 1 );
    if(info != 0)
    {
        fprintf(stderr, "%s(): SVD(R) failed!\n", __func__);
        r[0] = r[0] = r[2] = 1e20;
        return;
    }

    // R = U Vt
    mul_genN3_gen33_vout(3, U, Vt, UVt);

#else
    const double* UVt = R;
#endif

    r[0] = UVt[2*3 + 1] - UVt[1*3 + 2];
    r[1] = UVt[0*3 + 2] - UVt[2*3 + 0];
    r[2] = UVt[1*3 + 0] - UVt[0*3 + 1];

    double s2 =
        (r[0]*r[0] +
         r[1]*r[1] +
         r[2]*r[2]) / 4.;

    if( s2 < 1e-10 )
    {
        double c_unscaled = UVt[0*3 + 0] + UVt[1*3 + 1] + UVt[2*3 + 2] - 1.0;

        if( c_unscaled > 0 )
            r[0] = r[1] = r[2] = 0.0;
        else
        {
            double c =
                fmax(-1.,
                     fmin(1.,
                          c_unscaled*0.5));
            double t;

            t = (UVt[0*3 + 0] + 1)*0.5;
            r[0] = sqrt(fmax(t,0.));

            t = (UVt[1*3 + 1] + 1)*0.5;
            r[1] = copysign( sqrt(fmax(t,0.)), UVt[0*3 + 1] );

            t = (UVt[2*3 + 2] + 1)*0.5;
            r[2] = copysign( sqrt(fmax(t,0.)), UVt[0*3 + 2] );

            if( fabs(r[0]) < fabs(r[1]) &&
                fabs(r[0]) < fabs(r[2]) &&
                (UVt[1*3 + 2] > 0) != (r[1]*r[2] > 0) )
                r[2] = -r[2];

            double scale =
                acos(c) /
                sqrt( r[0]*r[0] +
                      r[1]*r[1] +
                      r[2]*r[2] );
            for(int i=0; i<3; i++)
                r[i] *= scale;
        }

        if( J )
            if( c_unscaled > 0 )
            {
                J[5] = J[15] = J[19] = -0.5;
                J[7] = J[11] = J[21] =  0.5;
            }
    }
    else
    {
        double s = sqrt(s2);
        double c = fmax(-1.,
                        fmin(1.,
                             (UVt[0*3 + 0] + UVt[1*3 + 1] + UVt[2*3 + 2] - 1)*0.5));
        double theta = acos(c);
        double vth = 1. / (2.*s);

        if( J )
        {
            double dtheta_dtr = -2. * vth;
            // var1 = [vth;theta]
            // var = [om1;var1] = [om1;vth;theta]
            double dvth_dtheta = -vth*c/s;
            double d1 = 0.5*dvth_dtheta*dtheta_dtr;
            double d2 = 0.5*dtheta_dtr;

            /*
              The opencv implementation looks really inefficient. I used maxima to
              analytically simplify this thing. Opencv:

                double domegadvar2[] =
                {
                    theta, 0,     0,     r[0]*vth,
                    0,     theta, 0,     r[1]*vth,
                    0,     0,     theta, r[2]*vth
                };
                // var2 = [om;theta]
                double dvar2dvar[] =
                {
                    vth, 0,   0,   r[0], 0,
                    0,   vth, 0,   r[1], 0,
                    0,   0,   vth, r[2], 0,
                    0,   0,   0,   0,    1
                };
                // dvar1/dR = dvar1/dtheta*dtheta/dR = [dvth/dtheta; 1] * dtheta/dtr * dtr/dR
                double dvardR[5*9] =
                {
                    0,  0, 0,  0,  0,  1, 0, -1, 0,
                    0,  0, -1, 0,  0,  0, 1, 0,  0,
                    0,  1, 0,  -1, 0,  0, 0, 0,  0,
                    d1, 0, 0,  0,  d1, 0, 0, 0,  d1,
                    d2, 0, 0,  0,  d2, 0, 0, 0,  d2
                };

                // (3,4) (4,5) (5,9)
                cvMatMul( &_domegadvar2, &_dvar2dvar, &_t0 );
                cvMatMul( &_t0, &_dvardR, &matJ );

                // transpose every row of matJ (treat the rows as 3x3 matrices)
                double t;
                CV_SWAP(J[1],  J[3],  t); CV_SWAP(J[2],  J[6], t);  CV_SWAP(J[5],  J[7], t);
                CV_SWAP(J[10], J[12], t); CV_SWAP(J[11], J[15], t); CV_SWAP(J[14], J[16], t);
                CV_SWAP(J[19], J[21], t); CV_SWAP(J[20], J[24], t); CV_SWAP(J[23], J[25], t);

            Maxima code:

                (%i2) A : matrix([theta, 0,     0,     r0*vth],[0,     theta, 0,     r1*vth],[0,     0,     theta, r2*vth]);
                                        [ theta    0      0    r0 vth ]
                                        [                             ]
                (%o2)                   [   0    theta    0    r1 vth ]
                                        [                             ]
                                        [   0      0    theta  r2 vth ]

                (%i4) B : matrix([vth, 0,   0,   r0, 0],[0,   vth, 0,   r1, 0],[0,   0,   vth, r2, 0], [0,   0,   0,   0,    1]);

                                           [ vth   0    0   r0  0 ]
                                           [                      ]
                                           [  0   vth   0   r1  0 ]
                (%o4)                      [                      ]
                                           [  0    0   vth  r2  0 ]
                                           [                      ]
                                           [  0    0    0   0   1 ]

                (%i6) C : matrix([0,  0, 0,  0,  0,  1, 0, -1, 0], [0,  0, -1, 0,  0,  0, 1, 0,  0], [0,  1, 0,  -1, 0,  0, 0, 0,  0], [d1, 0, 0,  0,  d1, 0, 0, 0,  d1], [d2, 0, 0,  0,  d2, 0, 0, 0,  d2]);

                                    [ 0   0   0    0   0   1  0  - 1  0  ]
                                    [                                    ]
                                    [ 0   0  - 1   0   0   0  1   0   0  ]
                                    [                                    ]
                (%o6)               [ 0   1   0   - 1  0   0  0   0   0  ]
                                    [                                    ]
                                    [ d1  0   0    0   d1  0  0   0   d1 ]
                                    [                                    ]
                                    [ d2  0   0    0   d2  0  0   0   d2 ]

                (%i8) display2d : false;

                (%o8) false
                (%i9) A . B . C;

                (%o9) matrix([d2*r0*vth+d1*r0*theta,0,0,0,d2*r0*vth+d1*r0*theta,theta*vth,0,
                              -theta*vth,d2*r0*vth+d1*r0*theta],
                             [d2*r1*vth+d1*r1*theta,0,-theta*vth,0,d2*r1*vth+d1*r1*theta,0,
                              theta*vth,0,d2*r1*vth+d1*r1*theta],
                             [d2*r2*vth+d1*r2*theta,theta*vth,0,-theta*vth,
                              d2*r2*vth+d1*r2*theta,0,0,0,d2*r2*vth+d1*r2*theta])
            */

            // copied maxima results, applied the SWAP operations manually,
            // removed the 0 values, consolidated identical results
            J[ 0] =  d2*r[0]*vth + d1*r[0]*theta;
            J[ 4] =  J[0];
            J[ 5] = -theta*vth;
            J[ 7] = -J[5];
            J[ 8] =  J[0];
            J[ 9] =  d2*r[1]*vth + d1*r[1]*theta;
            J[11] =  J[7];
            J[13] =  J[9];
            J[15] =  J[5];
            J[17] =  J[9];
            J[18] =  d2*r[2]*vth + d1*r[2]*theta;
            J[19] =  J[5];
            J[21] =  J[7];
            J[22] =  J[18];
            J[26] =  J[18];
        }

        for(int i=0; i<3; i++)
            r[i] *= vth * theta;
    }
}

void mrcal_R_from_r( // outputs
                     double* R, // (3,3) array
                     double* J, // (3,3,3) array. Gradient. May be NULL

                     // input
                     const double* r // (3) vector
                    )
{
    double norm2r = 0.0;
    for(int i=0; i<3; i++)
        norm2r += r[i]*r[i];

    if( norm2r < DBL_EPSILON*DBL_EPSILON )
    {
        R[0] = 1.0; R[1] = 0.0; R[2] = 0.0;
        R[3] = 0.0; R[4] = 1.0; R[5] = 0.0;
        R[6] = 0.0; R[7] = 0.0; R[8] = 1.0;

        if( J )
        {
            memset( J, 0, 3*9*sizeof(J[0]) );
            J[0+5*3] = J[1+6*3] = J[2+1*3] = -1.;
            J[0+7*3] = J[1+2*3] = J[2+3*3] =  1.;
        }
    }
    else
    {
        double theta = sqrt(norm2r);

        double c,s;
        sincos(theta, &s, &c);
        double c1 = 1. - c;
        double itheta = 1./theta;

        double r_unit[3];
        for(int i=0; i<3; i++)
            r_unit[i] = r[i] * itheta;

        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        R[0] = c + c1*r_unit[0]*r_unit[0];
        R[1] =     c1*r_unit[0]*r_unit[1] - s*r_unit[2];
        R[2] =     c1*r_unit[0]*r_unit[2] + s*r_unit[1];
        R[3] =     c1*r_unit[0]*r_unit[1] + s*r_unit[2];
        R[4] = c + c1*r_unit[1]*r_unit[1];
        R[5] =     c1*r_unit[1]*r_unit[2] - s*r_unit[0];
        R[6] =     c1*r_unit[0]*r_unit[2] - s*r_unit[1];
        R[7] =     c1*r_unit[1]*r_unit[2] + s*r_unit[0];
        R[8] = c + c1*r_unit[2]*r_unit[2];

        if( J )
        {
            // opencv had some logic with lots of 0s. I unrolled all of the
            // loops, and removed all the resulting 0 terms
            double a0, a1, a3;
            double a2 = itheta * c1;
            double a4 = itheta * s;

            a0 = -s        *r_unit[0];
            a1 = (s - 2*a2)*r_unit[0];
            a3 = (c -   a4)*r_unit[0];
            J[0+0*3] = a0 + a1*r_unit[0]*r_unit[0] + a2*(r_unit[0]+r_unit[0]);
            J[0+1*3] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   - a3*r_unit[2];
            J[0+2*3] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[1];
            J[0+3*3] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   + a3*r_unit[2];
            J[0+4*3] = a0 + a1*r_unit[1]*r_unit[1];
            J[0+5*3] =      a1*r_unit[1]*r_unit[2]                  - a3*r_unit[0] - a4;
            J[0+6*3] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[1];
            J[0+7*3] =      a1*r_unit[1]*r_unit[2]                  + a3*r_unit[0] + a4;
            J[0+8*3] = a0 + a1*r_unit[2]*r_unit[2];

            a0 = -s        *r_unit[1];
            a1 = (s - 2*a2)*r_unit[1];
            a3 = (c -   a4)*r_unit[1];
            J[1+0*3] = a0 + a1*r_unit[0]*r_unit[0];
            J[1+1*3] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   - a3*r_unit[2];
            J[1+2*3] =      a1*r_unit[0]*r_unit[2]                  + a3*r_unit[1] + a4;
            J[1+3*3] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   + a3*r_unit[2];
            J[1+4*3] = a0 + a1*r_unit[1]*r_unit[1] + a2*(r_unit[1]+r_unit[1]);
            J[1+5*3] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[0];
            J[1+6*3] =      a1*r_unit[0]*r_unit[2]                  - a3*r_unit[1] - a4;
            J[1+7*3] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[0];
            J[1+8*3] = a0 + a1*r_unit[2]*r_unit[2];

            a0 = -s        *r_unit[2];
            a1 = (s - 2*a2)*r_unit[2];
            a3 = (c -   a4)*r_unit[2];
            J[2+0*3] = a0 + a1*r_unit[0]*r_unit[0];
            J[2+1*3] =      a1*r_unit[0]*r_unit[1]                  - a3*r_unit[2] - a4;
            J[2+2*3] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   + a3*r_unit[1];
            J[2+3*3] =      a1*r_unit[0]*r_unit[1]                  + a3*r_unit[2] + a4;
            J[2+4*3] = a0 + a1*r_unit[1]*r_unit[1];
            J[2+5*3] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   - a3*r_unit[0];
            J[2+6*3] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   - a3*r_unit[1];
            J[2+7*3] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   + a3*r_unit[0];
            J[2+8*3] = a0 + a1*r_unit[2]*r_unit[2] + a2*(r_unit[2]+r_unit[2]);
        }
    }
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_r_from_R(). No
// gradients available here. If you need gradients, call mrcal_r_from_R()
// directly
void mrcal_rt_from_Rt( // output
                      double* rt,  // (6) vector

                      // input
                      const double* Rt // (4,3) array
                     )
{
    mrcal_r_from_R(rt, NULL, Rt);
    for(int i=0; i<3; i++) rt[i+3] = Rt[i+9];
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_R_from_r(). No
// gradients available here. If you need gradients, call mrcal_R_from_r()
// directly
void mrcal_Rt_from_rt( // output
                      double* Rt, // (4,3) array

                      // input
                      const double* rt // (6) vector
                     )
{
    mrcal_R_from_r(Rt, NULL, rt);
    for(int i=0; i<3; i++) Rt[i+9] = rt[i+3];

}

// Invert an Rt transformation
//
// b = Ra + t  -> a = R'b - R't
void mrcal_invert_Rt( // output
                     double* Rt_out, // (4,3) array

                     // input
                     const double* Rt_in // (4,3) array
                    )
{
    // transpose(R)
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            Rt_out[i*3 + j] = Rt_in[i + 3*j];

    // -transpose(R)*t
    mul_vec3_gen33_vout_scaled(&Rt_in[9], &Rt_in[0],
                               &Rt_out[9], -1.0);
}

// Invert an rt transformation
//
// b = rotate(a) + t  -> a = invrotate(b) - invrotate(t)
void mrcal_invert_rt( // output
                     double* rt_out, // (6) array

                     // input
                     const double* rt_in // (6) array
                    )
{
    // r uses an angle-axis representation, so to undo a rotation r, I can apply
    // a rotation -r (same axis, equal and opposite angle)
    for(int i=0; i<3; i++)
        rt_out[i] = -rt_in[i];

    mrcal_rotate_point_r( &rt_out[3],
                          NULL, NULL,

                          // input
                          rt_out,
                          &rt_in[3]
                          );
    for(int i=0; i<3; i++)
        rt_out[3+i] *= -1.;
}


// Compose two Rt transformations
//   R0*(R1*x + t1) + t0 =
//   R0*R1*x + R0*t1+t0
void mrcal_compose_Rt( // output
                      double* Rt_out, // (4,3) array

                      // input
                      const double* Rt_0, // (4,3) array
                      const double* Rt_1  // (4,3) array
                     )
{
    // R0*R1
    mul_genN3_gen33_vout(3,
                         &Rt_0[0], &Rt_1[0],
                         &Rt_out[0]);

    // R0*t1+t0
    mul_vec3_gen33t_vout(&Rt_1[9], &Rt_0[0],
                         &Rt_out[9]);
    add_vec(3, &Rt_out[9], &Rt_0[9]);
}

// Compose two rt transformations
void mrcal_compose_rt( // output
                      double* rt_out, // (6) array

                      // input
                      const double* rt_0, // (6) array
                      const double* rt_1  // (6) array
                     )
{
    // I convert this to Rt to get the composition, and then convert back
    double Rt_out[12];
    double Rt_0  [12];
    double Rt_1  [12];
    mrcal_Rt_from_rt(Rt_0,   rt_0);
    mrcal_Rt_from_rt(Rt_1,   rt_1);
    mrcal_compose_Rt(Rt_out, Rt_0, Rt_1);
    mrcal_rt_from_Rt(rt_out, Rt_out);
}
