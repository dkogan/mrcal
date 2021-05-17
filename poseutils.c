#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "poseutils.h"
#include "strides.h"

// All arrays stored in row-major order
//
// I have two different representations of pose transformations:
//
// - Rt is a concatenated (4,3) array: Rt = nps.glue(R,t, axis=-2). The
//   transformation is R*x+t
//
// - rt is a concatenated (6) array: rt = nps.glue(r,t, axis=-1). The
//   transformation is R*x+t where R = R_from_r(r)


// row vectors: vout = matmult(v,Mt)
// equivalent col vector expression: vout = matmult(M,v)
#define mul_vec3_gen33t_vout_scaled_full(vout, vout_stride0,            \
                                         v,    v_stride0,               \
                                         Mt,   Mt_stride0, Mt_stride1,  \
                                         scale)                         \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,0,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,0,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,0,2)*_P1(v,v_stride0,2) ),    \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,1,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,1,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,1,2)*_P1(v,v_stride0,2) ),    \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,2,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,2,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,2,2)*_P1(v,v_stride0,2) ) };  \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#define mul_vec3_gen33t_vout_full(vout, vout_stride0,                   \
                                  v,    v_stride0,                      \
                                  Mt,   Mt_stride0, Mt_stride1)         \
    mul_vec3_gen33t_vout_scaled_full(vout, vout_stride0,                \
                                     v,    v_stride0,                   \
                                     Mt,   Mt_stride0, Mt_stride1, 1.0)
// row vectors: vout = scale*matmult(v,M)
#define mul_vec3_gen33_vout_scaled_full(vout, vout_stride0,             \
                                        v,    v_stride0,                \
                                        M,    M_stride0, M_stride1,     \
                                        scale)                          \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,0)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,0)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,0)*_P1(v,v_stride0,2)),        \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,1)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,1)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,1)*_P1(v,v_stride0,2)),        \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,2)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,2)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,2)*_P1(v,v_stride0,2)) };      \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#define mul_vec3_gen33_vout_full(vout, vout_stride0,                   \
                                 v,    v_stride0,                       \
                                 Mt,   Mt_stride0, Mt_stride1)          \
    mul_vec3_gen33_vout_scaled_full(vout, vout_stride0,                 \
                                    v,    v_stride0,                    \
                                    Mt,   Mt_stride0, Mt_stride1, 1.0)

// row vectors: vout = matmult(v,Mt)
// equivalent col vector expression: vout = matmult(M,v)
#define mul_vec3_gen33t_vaccum_full(vout, vout_stride0,                 \
                                    v,    v_stride0,                    \
                                    Mt,   Mt_stride0, Mt_stride1)       \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            _P1(vout,vout_stride0,0) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,0,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,0,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,0,2)*_P1(v,v_stride0,2),       \
            _P1(vout,vout_stride0,1) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,1,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,1,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,1,2)*_P1(v,v_stride0,2),       \
            _P1(vout,vout_stride0,2) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,2,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,2,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,2,2)*_P1(v,v_stride0,2) };     \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#warning the above didnt fix any tests

// row vectors: vout = scale*matmult(v,M)
#define mul_vec3_gen33_vaccum_scaled_full(vout, vout_stride0,           \
                                          v,    v_stride0,              \
                                          M,    M_stride0, M_stride1,   \
                                          scale)                        \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            _P1(vout,vout_stride0,0) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,0)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,0)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,0)*_P1(v,v_stride0,2)),        \
            _P1(vout,vout_stride0,1) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,1)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,1)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,1)*_P1(v,v_stride0,2)),        \
            _P1(vout,vout_stride0,2) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,2)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,2)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,2)*_P1(v,v_stride0,2)) };      \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#warning did nt fix

// multiply two (3,3) matrices
static inline
void mul_gen33_gen33_vout_full(// output
                               double* restrict m0m1,
                               int m0m1_stride0, int m0m1_stride1,

                               // input
                               const double* restrict m0,
                               int m0_stride0, int m0_stride1,
                               const double* restrict m1,
                               int m1_stride0, int m1_stride1)
{
    for(int i=0; i<3; i++)
        // one row at a time
        mul_vec3_gen33_vout_scaled_full(
                                        &_P2(m0m1,m0m1_stride0,m0m1_stride1,  i,0),   m0m1_stride1,
                                        &_P2(m0  ,  m0_stride0,  m0_stride1,  i,0),   m0_stride1,
                                        m1, m1_stride0, m1_stride1,
                                        1.0);
}
// multiply two (3,3) matrices, but accumulate the result instead of setting
static inline
void mul_gen33_gen33_vaccum_full(// output
                                 double* restrict m0m1,
                                 int m0m1_stride0, int m0m1_stride1,

                                 // input
                                 const double* restrict m0,
                                 int m0_stride0, int m0_stride1,
                                 const double* restrict m1,
                                 int m1_stride0, int m1_stride1)
{
    for(int i=0; i<3; i++)
        // one row at a time
        mul_vec3_gen33_vaccum_scaled_full(
                                          &_P2(m0m1,m0m1_stride0,m0m1_stride1,  i,0),   m0m1_stride1,
                                          &_P2(m0  ,  m0_stride0,  m0_stride1,  i,0),   m0_stride1,
                                          m1, m1_stride0, m1_stride1,
                                          1.0);
}

static inline
double inner3(const double* restrict a,
              const double* restrict b)
{
    double s = 0.0;
    for (int i=0; i<3; i++) s += a[i]*b[i];
    return s;
}




// Make an identity rotation or transformation
void mrcal_identity_R_full(double* R,      // (3,3) array
                           int R_stride0,  // in bytes. <= 0 means "contiguous"
                           int R_stride1   // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_2D(R, 3,3);
    P2(R, 0,0) = 1.0; P2(R, 0,1) = 0.0; P2(R, 0,2) = 0.0;
    P2(R, 1,0) = 0.0; P2(R, 1,1) = 1.0; P2(R, 1,2) = 0.0;
    P2(R, 2,0) = 0.0; P2(R, 2,1) = 0.0; P2(R, 2,2) = 1.0;
}
void mrcal_identity_r_full(double* r,      // (3,) array
                           int r_stride0   // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_1D(r, 3);
    P1(r, 0) = 0.0; P1(r, 1) = 0.0; P1(r, 2) = 0.0;
}
void mrcal_identity_Rt_full(double* Rt,      // (4,3) array
                            int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                            int Rt_stride1   // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_2D(Rt, 4,3);
    mrcal_identity_R_full(Rt, Rt_stride0, Rt_stride1);
    for(int i=0; i<3; i++) P2(Rt, 3, i) = 0.0;
}
void mrcal_identity_rt_full(double* rt,      // (6,) array
                            int rt_stride0   // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_1D(rt, 6);
    mrcal_identity_r_full(rt, rt_stride0);
    for(int i=0; i<3; i++) P1(rt, i+3) = 0.0;
}

void mrcal_rotate_point_R_full( // output
                               double* x_out,      // (3,) array
                               int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                               double* J_R,        // (3,3,3) array. May be NULL
                               int J_R_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_R_stride1,    // in bytes. <= 0 means "contiguous"
                               int J_R_stride2,    // in bytes. <= 0 means "contiguous"
                               double* J_x,        // (3,3) array. May be NULL
                               int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                               // input
                               const double* R,    // (3,3) array. May be NULL
                               int R_stride0,      // in bytes. <= 0 means "contiguous"
                               int R_stride1,      // in bytes. <= 0 means "contiguous"
                               const double* x_in, // (3,) array. May be NULL
                               int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                )
{
    init_stride_1D(x_out, 3);
    init_stride_3D(J_R,   3,3,3 );
    init_stride_2D(J_x,   3,3 );
    init_stride_2D(R,     3,3 );
    init_stride_1D(x_in,  3 );

    if(J_R)
    {
        // out[i] = inner(R[i,:],in)
        for(int i=0; i<3; i++)
        {
            int j=0;
            for(; j<i; j++)
                for(int k=0; k<3; k++)
                    P3(J_R, i,j,k) = 0.0;
            for(int k=0; k<3; k++)
                P3(J_R, i,j,k) = P1(x_in, k);
            for(j++; j<3; j++)
                for(int k=0; k<3; k++)
                    P3(J_R, i,j,k) = 0.0;
        }
    }
    if(J_x)
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                P2(J_x, i,j) = P2(R, i,j);

    // R*x
    mul_vec3_gen33t_vout_full(x_out, x_out_stride0,
                              x_in,  x_in_stride0,
                              R,     R_stride0, R_stride1);
}


// mrcal_rotate_point_r() uses auto-differentiation, so it's implemented in C++
// in poseutils-uses-autodiff.cc


// Apply a transformation to a point
void mrcal_transform_point_Rt_full( // output
                                   double* x_out,      // (3,) array
                                   int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                   double* J_Rt,       // (3,4,3) array. May be NULL
                                   int J_Rt_stride0,   // in bytes. <= 0 means "contiguous"
                                   int J_Rt_stride1,   // in bytes. <= 0 means "contiguous"
                                   int J_Rt_stride2,   // in bytes. <= 0 means "contiguous"
                                   double* J_x,        // (3,3) array. May be NULL
                                   int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                   int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                   // input
                                   const double* Rt,   // (4,3) array. May be NULL
                                   int Rt_stride0,     // in bytes. <= 0 means "contiguous"
                                   int Rt_stride1,     // in bytes. <= 0 means "contiguous"
                                   const double* x_in, // (3,) array. May be NULL
                                   int x_in_stride0    // in bytes. <= 0 means "contiguous"
                                    )
{
    init_stride_1D(x_out, 3);
    init_stride_3D(J_Rt,  3,4,3 );
    // init_stride_2D(J_x,   3,3 );
    init_stride_2D(Rt,    4,3 );
    // init_stride_1D(x_in,  3 );

    // for in-place operation
    double t[] = { P2(Rt,3,0), P2(Rt,3,1), P2(Rt,3,2) };

    // I want R*x + t
    // First R*x
    mrcal_rotate_point_R_full(x_out, x_out_stride0,
                              J_Rt,  J_Rt_stride0,  J_Rt_stride1, J_Rt_stride2,
                              J_x,   J_x_stride0,   J_x_stride1,
                              Rt,    Rt_stride0,    Rt_stride1,
                              x_in,  x_in_stride0);

    // And now +t. The J_R, J_x gradients are unaffected. J_t is identity
    for(int i=0; i<3; i++)
        P1(x_out,i) += t[i];
    if(J_Rt)
        mrcal_identity_R_full(&P3(J_Rt,0,3,0), J_Rt_stride0, J_Rt_stride2);
}

// The implementation of mrcal_R_from_r is based on opencv.
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
void mrcal_R_from_r_full( // outputs
                         double* R,       // (3,3) array
                         int R_stride0,   // in bytes. <= 0 means "contiguous"
                         int R_stride1,   // in bytes. <= 0 means "contiguous"
                         double* J,       // (3,3,3) array. Gradient. May be NULL
                         int J_stride0,   // in bytes. <= 0 means "contiguous"
                         int J_stride1,   // in bytes. <= 0 means "contiguous"
                         int J_stride2,   // in bytes. <= 0 means "contiguous"

                         // input
                         const double* r, // (3,) vector
                         int r_stride0    // in bytes. <= 0 means "contiguous"
                          )
{
    init_stride_2D(R, 3,3);
    init_stride_3D(J, 3,3,3 );
    init_stride_1D(r, 3 );

    double norm2r = 0.0;
    for(int i=0; i<3; i++)
        norm2r += P1(r,i)*P1(r,i);

    if( norm2r < DBL_EPSILON*DBL_EPSILON )
    {
        mrcal_identity_R_full(R, R_stride0, R_stride1);

        if( J )
        {
            for(int i=0; i<3; i++)
                for(int j=0; j<3; j++)
                    for(int k=0; k<3; k++)
                        P3(J,i,j,k) = 0.;

            P3(J,1,2,0) = -1.;
            P3(J,2,0,1) = -1.;
            P3(J,0,1,2) = -1.;

            P3(J,2,1,0) =  1.;
            P3(J,0,2,1) =  1.;
            P3(J,1,0,2) =  1.;
        }
        return;
    }

    double theta = sqrt(norm2r);

    double c,s;
    sincos(theta, &s, &c);
    double c1 = 1. - c;
    double itheta = 1./theta;

    double r_unit[3];
    for(int i=0; i<3; i++)
        r_unit[i] = P1(r,i) * itheta;

    // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
    P2(R, 0,0) = c + c1*r_unit[0]*r_unit[0];
    P2(R, 0,1) =     c1*r_unit[0]*r_unit[1] - s*r_unit[2];
    P2(R, 0,2) =     c1*r_unit[0]*r_unit[2] + s*r_unit[1];
    P2(R, 1,0) =     c1*r_unit[0]*r_unit[1] + s*r_unit[2];
    P2(R, 1,1) = c + c1*r_unit[1]*r_unit[1];
    P2(R, 1,2) =     c1*r_unit[1]*r_unit[2] - s*r_unit[0];
    P2(R, 2,0) =     c1*r_unit[0]*r_unit[2] - s*r_unit[1];
    P2(R, 2,1) =     c1*r_unit[1]*r_unit[2] + s*r_unit[0];
    P2(R, 2,2) = c + c1*r_unit[2]*r_unit[2];

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
        P3(J,0,0,0) = a0 + a1*r_unit[0]*r_unit[0] + a2*(r_unit[0]+r_unit[0]);
        P3(J,0,1,0) =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   - a3*r_unit[2];
        P3(J,0,2,0) =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[1];
        P3(J,1,0,0) =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   + a3*r_unit[2];
        P3(J,1,1,0) = a0 + a1*r_unit[1]*r_unit[1];
        P3(J,1,2,0) =      a1*r_unit[1]*r_unit[2]                  - a3*r_unit[0] - a4;
        P3(J,2,0,0) =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[1];
        P3(J,2,1,0) =      a1*r_unit[1]*r_unit[2]                  + a3*r_unit[0] + a4;
        P3(J,2,2,0) = a0 + a1*r_unit[2]*r_unit[2];

        a0 = -s        *r_unit[1];
        a1 = (s - 2*a2)*r_unit[1];
        a3 = (c -   a4)*r_unit[1];
        P3(J,0,0,1) = a0 + a1*r_unit[0]*r_unit[0];
        P3(J,0,1,1) =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   - a3*r_unit[2];
        P3(J,0,2,1) =      a1*r_unit[0]*r_unit[2]                  + a3*r_unit[1] + a4;
        P3(J,1,0,1) =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   + a3*r_unit[2];
        P3(J,1,1,1) = a0 + a1*r_unit[1]*r_unit[1] + a2*(r_unit[1]+r_unit[1]);
        P3(J,1,2,1) =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[0];
        P3(J,2,0,1) =      a1*r_unit[0]*r_unit[2]                  - a3*r_unit[1] - a4;
        P3(J,2,1,1) =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[0];
        P3(J,2,2,1) = a0 + a1*r_unit[2]*r_unit[2];

        a0 = -s        *r_unit[2];
        a1 = (s - 2*a2)*r_unit[2];
        a3 = (c -   a4)*r_unit[2];
        P3(J,0,0,2) = a0 + a1*r_unit[0]*r_unit[0];
        P3(J,0,1,2) =      a1*r_unit[0]*r_unit[1]                  - a3*r_unit[2] - a4;
        P3(J,0,2,2) =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   + a3*r_unit[1];
        P3(J,1,0,2) =      a1*r_unit[0]*r_unit[1]                  + a3*r_unit[2] + a4;
        P3(J,1,1,2) = a0 + a1*r_unit[1]*r_unit[1];
        P3(J,1,2,2) =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   - a3*r_unit[0];
        P3(J,2,0,2) =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   - a3*r_unit[1];
        P3(J,2,1,2) =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   + a3*r_unit[0];
        P3(J,2,2,2) = a0 + a1*r_unit[2]*r_unit[2] + a2*(r_unit[2]+r_unit[2]);
    }
}

// Invert a rotation matrix. This is a transpose
//
// The input is given in R_in in a (3,3) array
//
// The result is returned in a (3,3) array R_out. In-place operation is
// supported
void mrcal_invert_R_full( // output
                         double* R_out,      // (3,3) array
                         int R_out_stride0,  // in bytes. <= 0 means "contiguous"
                         int R_out_stride1,  // in bytes. <= 0 means "contiguous"

                         // input
                         const double* R_in, // (3,3) array
                         int R_in_stride0,   // in bytes. <= 0 means "contiguous"
                         int R_in_stride1    // in bytes. <= 0 means "contiguous"
                         )
{
    init_stride_2D(R_out, 3,3);
    init_stride_2D(R_in,  3,3);

    // transpose(R). Extra stuff to make in-place operations work
    for(int i=0; i<3; i++)
        P2(R_out,i,i) = P2(R_in,i,i);
    for(int i=0; i<3; i++)
        for(int j=i+1; j<3; j++)
        {
            double tmp = P2(R_in,i,j);
            P2(R_out,i,j) = P2(R_in,j,i);
            P2(R_out,j,i) = tmp;
        }
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_r_from_R().
void mrcal_rt_from_Rt_full(// output
                           double* rt,      // (6,) vector
                           int rt_stride0,  // in bytes. <= 0 means "contiguous"
                           double* J_R,     // (3,3,3) array. Gradient. May be NULL
                           // No J_t. It's always the identity
                           int J_R_stride0, // in bytes. <= 0 means "contiguous"
                           int J_R_stride1, // in bytes. <= 0 means "contiguous"
                           int J_R_stride2, // in bytes. <= 0 means "contiguous"

                           // input
                           const double* Rt,  // (4,3) array
                           int Rt_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_stride1     // in bytes. <= 0 means "contiguous"
                           )
{
    mrcal_r_from_R_full(rt,  rt_stride0,
                        J_R, J_R_stride0, J_R_stride1, J_R_stride2,
                        Rt,  Rt_stride0,  Rt_stride1);

    init_stride_1D(rt,  6);
    // init_stride_3D(J_R, 3,3,3);
    init_stride_2D(Rt,  4,3);

    for(int i=0; i<3; i++)
        P1(rt, i+3) = P2(Rt,3,i);
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_R_from_r().
void mrcal_Rt_from_rt_full(// output
                           double* Rt,      // (4,3) array
                           int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                           int Rt_stride1,  // in bytes. <= 0 means "contiguous"
                           double* J_r,     // (3,3,3) array. Gradient. May be NULL
                           // No J_t. It's just the identity
                           int J_r_stride0, // in bytes. <= 0 means "contiguous"
                           int J_r_stride1, // in bytes. <= 0 means "contiguous"
                           int J_r_stride2, // in bytes. <= 0 means "contiguous"

                           // input
                           const double* rt, // (6,) vector
                           int rt_stride0    // in bytes. <= 0 means "contiguous"
                           )
{
    mrcal_R_from_r_full(Rt,  Rt_stride0,  Rt_stride1,
                        J_r, J_r_stride0, J_r_stride1, J_r_stride2,
                        rt,  rt_stride0);

    init_stride_1D(rt,  6);
    // init_stride_3D(J_r, 3,3,3);
    init_stride_2D(Rt,  4,3);

    for(int i=0; i<3; i++)
        P2(Rt,3,i) = P1(rt,i+3);
}

// Invert an Rt transformation
//
// b = Ra + t  -> a = R'b - R't
void mrcal_invert_Rt_full( // output
                          double* Rt_out,      // (4,3) array
                          int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                          int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                          // input
                          const double* Rt_in, // (4,3) array
                          int Rt_in_stride0,   // in bytes. <= 0 means "contiguous"
                          int Rt_in_stride1    // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_2D(Rt_out, 4,3);
    init_stride_2D(Rt_in,  4,3);

    // transpose(R). Extra stuff to make in-place operations work
    for(int i=0; i<3; i++)
        P2(Rt_out,i,i) = P2(Rt_in,i,i);
    for(int i=0; i<3; i++)
        for(int j=i+1; j<3; j++)
        {
            double tmp = P2(Rt_in,i,j);
            P2(Rt_out,i,j) = P2(Rt_in,j,i);
            P2(Rt_out,j,i) = tmp;
        }

    // -transpose(R)*t
    mul_vec3_gen33t_vout_scaled_full(&P2(Rt_out,3,0), Rt_out_stride1,
                                     &P2(Rt_in, 3,0), Rt_in_stride1,
                                     Rt_out, Rt_out_stride0, Rt_out_stride1,
                                     -1.0);
}

// Invert an rt transformation
//
// b = rotate(a) + t  -> a = invrotate(b) - invrotate(t)
//
// drout_drin is not returned: it is always -I
// drout_dtin is not returned: it is always 0
void mrcal_invert_rt_full( // output
                          double* rt_out,          // (6,) array
                          int rt_out_stride0,      // in bytes. <= 0 means "contiguous"
                          double* dtout_drin,      // (3,3) array
                          int dtout_drin_stride0,  // in bytes. <= 0 means "contiguous"
                          int dtout_drin_stride1,  // in bytes. <= 0 means "contiguous"
                          double* dtout_dtin,      // (3,3) array
                          int dtout_dtin_stride0,  // in bytes. <= 0 means "contiguous"
                          int dtout_dtin_stride1,  // in bytes. <= 0 means "contiguous"

                          // input
                          const double* rt_in,     // (6,) array
                          int rt_in_stride0        // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_1D(rt_out, 6);
    // init_stride_2D(dtout_drin, 3,3);
    init_stride_2D(dtout_dtin, 3,3);
    init_stride_1D(rt_in,  6);

    // r uses an angle-axis representation, so to undo a rotation r, I can apply
    // a rotation -r (same axis, equal and opposite angle)
    for(int i=0; i<3; i++)
        P1(rt_out,i) = -P1(rt_in,i);

    mrcal_rotate_point_r_full( &P1(rt_out,3), rt_out_stride0,
                               dtout_drin, dtout_drin_stride0, dtout_drin_stride1,
                               dtout_dtin, dtout_dtin_stride0, dtout_dtin_stride1,

                               // input
                               rt_out, rt_out_stride0,
                               &P1(rt_in,3), rt_in_stride0,
                               false);
    for(int i=0; i<3; i++)
        P1(rt_out,3+i) *= -1.;

    if(dtout_dtin)
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                P2(dtout_dtin,i,j) *= -1.;
}


// Compose two Rt transformations
//   R0*(R1*x + t1) + t0 =
//   (R0*R1)*x + R0*t1+t0
void mrcal_compose_Rt_full( // output
                           double* Rt_out,      // (4,3) array
                           int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                           int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                           // input
                           const double* Rt_0,  // (4,3) array
                           int Rt_0_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_0_stride1,    // in bytes. <= 0 means "contiguous"
                           const double* Rt_1,  // (4,3) array
                           int Rt_1_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_1_stride1     // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_2D(Rt_out, 4,3);
    init_stride_2D(Rt_0,   4,3);
    init_stride_2D(Rt_1,   4,3);

    // for in-place operation
    double t0[] = { P2(Rt_0,3,0),
                    P2(Rt_0,3,1),
                    P2(Rt_0,3,2) };

    // t <- R0*t1
    mul_vec3_gen33t_vout_full(&P2(Rt_out,3,0), Rt_out_stride1,
                              &P2(Rt_1,  3,0), Rt_1_stride1,
                              Rt_0, Rt_0_stride0, Rt_0_stride1);

    // R <- R0*R1
    mul_gen33_gen33_vout_full( Rt_out, Rt_out_stride0, Rt_out_stride1,
                               Rt_0,   Rt_0_stride0,   Rt_0_stride1,
                               Rt_1,   Rt_1_stride0,   Rt_1_stride1 );

    // t <- R0*t1+t0
    for(int i=0; i<3; i++)
        P2(Rt_out,3,i) += t0[i];
}

// Compose two rt transformations. It is assumed that we're getting no gradients
// at all or we're getting ALL the gradients: only dr_r0 is checked for NULL
//
// dr_dt0 is not returned: it is always 0
// dr_dt1 is not returned: it is always 0
// dt_dr1 is not returned: it is always 0
// dt_dt0 is not returned: it is always the identity matrix
void mrcal_compose_rt_full( // output
                           double* rt_out,       // (6,) array
                           int rt_out_stride0,   // in bytes. <= 0 means "contiguous"
                           double* dr_r0,        // (3,3) array; may be NULL
                           int dr_r0_stride0,    // in bytes. <= 0 means "contiguous"
                           int dr_r0_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dr_r1,        // (3,3) array; may be NULL
                           int dr_r1_stride0,    // in bytes. <= 0 means "contiguous"
                           int dr_r1_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dt_r0,        // (3,3) array; may be NULL
                           int dt_r0_stride0,    // in bytes. <= 0 means "contiguous"
                           int dt_r0_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dt_t1,        // (3,3) array; may be NULL
                           int dt_t1_stride0,    // in bytes. <= 0 means "contiguous"
                           int dt_t1_stride1,    // in bytes. <= 0 means "contiguous"

                           // input
                           const double* rt_0,   // (6,) array
                           int rt_0_stride0,     // in bytes. <= 0 means "contiguous"
                           const double* rt_1,   // (6,) array
                           int rt_1_stride0      // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_1D(rt_out, 6);
    init_stride_2D(dr_r0,  3,3);
    init_stride_2D(dr_r1,  3,3);
    init_stride_2D(dt_r0,  3,3);
    init_stride_2D(dt_t1,  3,3);
    init_stride_1D(rt_0,   6);
    init_stride_1D(rt_1,   6);

    // I convert this to Rt to get the composition, and then convert back
    if(dr_r0 == NULL)
    {
        // no gradients
        double Rt_0  [4*3];
        double Rt_1  [4*3];
        mrcal_Rt_from_rt_full(Rt_0, 0,0,
                              NULL,0,0,0,
                              rt_0, rt_0_stride0);
        mrcal_Rt_from_rt_full(Rt_1, 0,0,
                              NULL,0,0,0,
                              rt_1, rt_1_stride0);

        double Rt_out[4*3];
        mrcal_compose_Rt(Rt_out, Rt_0, Rt_1);
        mrcal_rt_from_Rt_full(rt_out, rt_out_stride0,
                              NULL,0,0,0,
                              Rt_out,0,0);
        return;
    }

    // Alright. gradients!
    // I have (R0*R1)*x + R0*t1+t0
    //   r = r_from_R(R0*R1)
    //   t = R0*t1+t0

    double* R0 = dt_t1; // this one is easy!

    double dR0_dr0[3*3*3];
    mrcal_R_from_r_full( R0, dt_t1_stride0, dt_t1_stride1,
                         dR0_dr0, 0,0,0,
                         rt_0, rt_0_stride0 );

    // to make in-place operations work
    double t[3];
    mul_vec3_gen33t_vout_full( t, sizeof(double),
                               &P1(rt_1, 3), rt_1_stride0,
                               R0, dt_t1_stride0, dt_t1_stride1);
    for(int i=0; i<3; i++)
        t[i] += P1(rt_0,3+i);

    double R1[3*3];
    double dR1_dr1[3*3*3];
    mrcal_R_from_r_full( R1, 0,0,
                         dR1_dr1, 0,0,0,
                         rt_1, rt_1_stride0 );

    double R[3*3];
    mul_gen33_gen33_vout_full(R,  3*sizeof(R [0]), sizeof(R [0]),
                              R0, dt_t1_stride0, dt_t1_stride1,
                              R1, 3*sizeof(R1[0]), sizeof(R1[0]));

    double dr_dR[3*3*3];
    mrcal_r_from_R_full( rt_out, rt_out_stride0,
                         dr_dR, 0,0,0,
                         R, 0,0);

    // dr/dvecr0 = dr/dvecR dvecR/dvecR0 dvecR0/dvecr0
    // R = R0*R1
    // vecR = [ inner(R0[0:3], R1t[0:3]),inner(R0[0:3], R1t[4:6]),inner(R0[0:3], R1t[7:9]),
    //          inner(R0[4:6], R1t[0:3]),inner(R0[4:6], R1t[4:6]),inner(R0[4:6], R1t[7:9]),
    //          inner(R0[7:9], R1t[0:3]),inner(R0[7:9], R1t[4:6]),inner(R0[7:9], R1t[7:9]), ]
    //      = [ R1t * R0[0:3]t ]
    //      = [ R1t * R0[3:6]t ]
    //      = [ R1t * R0[6:9]t ]
    //                     [R1t]
    // dvecR/dvecR0[0:3] = [ 0 ]
    //                     [ 0 ]
    //                     [R1t]
    // dvecR/dvecR0[3:6] = [ 0 ]
    //                     [ 0 ]
    // ... -> dvecR/dvecR0 = blockdiag(R1t,R1t,R1t) ->
    //                         [D]                         [D]
    // -> [A B C] dvecR/dvecR0 [E] = [A R1t  B R1t  C R1t] [E] =
    //                         [F]                         [F]
    //
    // = A R1t D + B R1t E + C R1t F
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            P2(dr_r0,i,j) = 0.0;
    for(int i=0; i<3; i++)
    {
        // compute dr_dR R1t dR0_dr0 for submatrix i
        double dr_dR_R1t[3*3];
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
                dr_dR_R1t[j*3+k] = inner3(&dr_dR[3*i + j*9], &R1[3*k] );

        mul_gen33_gen33_vaccum_full(dr_r0, dr_r0_stride0, dr_r0_stride1,
                                    dr_dR_R1t, 3*sizeof(dr_dR_R1t[0]), sizeof(dr_dR_R1t[0]),
                                    &dR0_dr0[9*i], 3*sizeof(dR0_dr0[0]), sizeof(dR0_dr0[0]));
    }

    // dr/dvecr1 = dr/dvecR dvecR/dvecR1 dvecR1/dvecr1
    // R = R0*R1
    // dvecR/dvecR1 = [ R0_00 I   R0_01 I   R0_02 I]
    //              = [ R0_10 I   R0_11 I   R0_12 I]
    //              = [ R0_20 I   R0_21 I   R0_22 I]
    //                         [D]
    // -> [A B C] dvecR/dvecR1 [E] =
    //                         [F]
    // = AD R0_00 + AE R0_01 + AF R0_02 + ...
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            P2(dr_r1,i,j) = 0.0;
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
                mul_vec3_gen33_vaccum_scaled_full(&P2(dr_r1,k,0), dr_r1_stride1,

                                                  &dr_dR[i*3 + 9*k], sizeof(dr_dR[0]),
                                                  &dR1_dr1[9*j],     3*sizeof(dR1_dr1[0]), sizeof(dR1_dr1[0]),

                                                  // scale
                                                  _P2(R0,dt_t1_stride0,dt_t1_stride1, i,j));

    // t = R0*t1+t0
    // t[0] = inner(R0[0:3], t1)
    // -> dt[0]/dr0 = t1t dR0[0:3]/dr0
    for(int i=0; i<3; i++)
        mul_vec3_gen33_vout_scaled_full(&P2(dt_r0,i,0), dt_r0_stride1,
                                        &P1(rt_1,3), rt_1_stride0,
                                        &dR0_dr0[9*i], 3*sizeof(dR0_dr0[0]), sizeof(dR0_dr0[0]),
                                        1.0);

    for(int i=0; i<3; i++)
        P1(rt_out,3+i) = t[i];
}
