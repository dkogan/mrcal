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
// in poseutils-uses-autodiff.cc


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
void mrcal_R_from_r_noncontiguous( // outputs
                     double* R,     // (3,3) array
                     int R_stride0, // in bytes. <= 0 means "contiguous"
                     int R_stride1, // in bytes. <= 0 means "contiguous"
                     double* J,     // (3,3,3) array. Gradient. May be NULL
                     int J_stride0, // in bytes. <= 0 means "contiguous"
                     int J_stride1, // in bytes. <= 0 means "contiguous"
                     int J_stride2, // in bytes. <= 0 means "contiguous"

                     // input
                     const double* r, // (3,) vector
                     int r_stride0    // in bytes. <= 0 means "contiguous"
                    )
{
    if(R_stride0 > 0) R_stride0 /= sizeof(R[0]);
    else              R_stride0 =  3;
    if(R_stride1 > 0) R_stride1 /= sizeof(R[0]);
    else              R_stride1 =  1;
    if(J_stride0 > 0) J_stride0 /= sizeof(J[0]);
    else              J_stride0 =  3*3;
    if(J_stride1 > 0) J_stride1 /= sizeof(J[0]);
    else              J_stride1 =  3;
    if(J_stride2 > 0) J_stride2 /= sizeof(J[0]);
    else              J_stride2 =  1;
    if(r_stride0 > 0) r_stride0 /= sizeof(r[0]);
    else              r_stride0 =  1;

    double norm2r = 0.0;
    for(int i=0; i<3; i++)
        norm2r += r[r_stride0*i]*r[r_stride0*i];

    if( norm2r < DBL_EPSILON*DBL_EPSILON )
    {
        R[0*R_stride0 + 0*R_stride1] = 1.0;
        R[0*R_stride0 + 1*R_stride1] = 0.0;
        R[0*R_stride0 + 2*R_stride1] = 0.0;
        R[1*R_stride0 + 0*R_stride1] = 0.0;
        R[1*R_stride0 + 1*R_stride1] = 1.0;
        R[1*R_stride0 + 2*R_stride1] = 0.0;
        R[2*R_stride0 + 0*R_stride1] = 0.0;
        R[2*R_stride0 + 1*R_stride1] = 0.0;
        R[2*R_stride0 + 2*R_stride1] = 1.0;

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
            r_unit[i] = r[r_stride0*i] * itheta;

        // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
        R[0*R_stride0 + 0*R_stride1] = c + c1*r_unit[0]*r_unit[0];
        R[0*R_stride0 + 1*R_stride1] =     c1*r_unit[0]*r_unit[1] - s*r_unit[2];
        R[0*R_stride0 + 2*R_stride1] =     c1*r_unit[0]*r_unit[2] + s*r_unit[1];
        R[1*R_stride0 + 0*R_stride1] =     c1*r_unit[0]*r_unit[1] + s*r_unit[2];
        R[1*R_stride0 + 1*R_stride1] = c + c1*r_unit[1]*r_unit[1];
        R[1*R_stride0 + 2*R_stride1] =     c1*r_unit[1]*r_unit[2] - s*r_unit[0];
        R[2*R_stride0 + 0*R_stride1] =     c1*r_unit[0]*r_unit[2] - s*r_unit[1];
        R[2*R_stride0 + 1*R_stride1] =     c1*r_unit[1]*r_unit[2] + s*r_unit[0];
        R[2*R_stride0 + 2*R_stride1] = c + c1*r_unit[2]*r_unit[2];

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
            J[0*J_stride2 + 0*J_stride1 + 0*J_stride0] = a0 + a1*r_unit[0]*r_unit[0] + a2*(r_unit[0]+r_unit[0]);
            J[0*J_stride2 + 1*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   - a3*r_unit[2];
            J[0*J_stride2 + 2*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[1];
            J[0*J_stride2 + 0*J_stride1 + 1*J_stride0] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[1]   + a3*r_unit[2];
            J[0*J_stride2 + 1*J_stride1 + 1*J_stride0] = a0 + a1*r_unit[1]*r_unit[1];
            J[0*J_stride2 + 2*J_stride1 + 1*J_stride0] =      a1*r_unit[1]*r_unit[2]                  - a3*r_unit[0] - a4;
            J[0*J_stride2 + 0*J_stride1 + 2*J_stride0] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[1];
            J[0*J_stride2 + 1*J_stride1 + 2*J_stride0] =      a1*r_unit[1]*r_unit[2]                  + a3*r_unit[0] + a4;
            J[0*J_stride2 + 2*J_stride1 + 2*J_stride0] = a0 + a1*r_unit[2]*r_unit[2];

            a0 = -s        *r_unit[1];
            a1 = (s - 2*a2)*r_unit[1];
            a3 = (c -   a4)*r_unit[1];
            J[1*J_stride2 + 0*J_stride1 + 0*J_stride0] = a0 + a1*r_unit[0]*r_unit[0];
            J[1*J_stride2 + 1*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   - a3*r_unit[2];
            J[1*J_stride2 + 2*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[2]                  + a3*r_unit[1] + a4;
            J[1*J_stride2 + 0*J_stride1 + 1*J_stride0] =      a1*r_unit[0]*r_unit[1] + a2*r_unit[0]   + a3*r_unit[2];
            J[1*J_stride2 + 1*J_stride1 + 1*J_stride0] = a0 + a1*r_unit[1]*r_unit[1] + a2*(r_unit[1]+r_unit[1]);
            J[1*J_stride2 + 2*J_stride1 + 1*J_stride0] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   - a3*r_unit[0];
            J[1*J_stride2 + 0*J_stride1 + 2*J_stride0] =      a1*r_unit[0]*r_unit[2]                  - a3*r_unit[1] - a4;
            J[1*J_stride2 + 1*J_stride1 + 2*J_stride0] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[2]   + a3*r_unit[0];
            J[1*J_stride2 + 2*J_stride1 + 2*J_stride0] = a0 + a1*r_unit[2]*r_unit[2];

            a0 = -s        *r_unit[2];
            a1 = (s - 2*a2)*r_unit[2];
            a3 = (c -   a4)*r_unit[2];
            J[2*J_stride2 + 0*J_stride1 + 0*J_stride0] = a0 + a1*r_unit[0]*r_unit[0];
            J[2*J_stride2 + 1*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[1]                  - a3*r_unit[2] - a4;
            J[2*J_stride2 + 2*J_stride1 + 0*J_stride0] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   + a3*r_unit[1];
            J[2*J_stride2 + 0*J_stride1 + 1*J_stride0] =      a1*r_unit[0]*r_unit[1]                  + a3*r_unit[2] + a4;
            J[2*J_stride2 + 1*J_stride1 + 1*J_stride0] = a0 + a1*r_unit[1]*r_unit[1];
            J[2*J_stride2 + 2*J_stride1 + 1*J_stride0] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   - a3*r_unit[0];
            J[2*J_stride2 + 0*J_stride1 + 2*J_stride0] =      a1*r_unit[0]*r_unit[2] + a2*r_unit[0]   - a3*r_unit[1];
            J[2*J_stride2 + 1*J_stride1 + 2*J_stride0] =      a1*r_unit[1]*r_unit[2] + a2*r_unit[1]   + a3*r_unit[0];
            J[2*J_stride2 + 2*J_stride1 + 2*J_stride0] = a0 + a1*r_unit[2]*r_unit[2] + a2*(r_unit[2]+r_unit[2]);
        }
    }
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_r_from_R(). No
// gradients available here. If you need gradients, call mrcal_r_from_R()
// directly
void mrcal_rt_from_Rt_noncontiguous( // output
                      double* rt,  // (6) vector
                      int rt_stride0, // in bytes. <= 0 means "contiguous"

                      // input
                      const double* Rt, // (4,3) array
                      int Rt_stride0,    // in bytes. <= 0 means "contiguous"
                      int Rt_stride1     // in bytes. <= 0 means "contiguous"
                     )
{
    mrcal_r_from_R_noncontiguous(rt,rt_stride0,
                                 NULL,0,0,0,
                                 Rt, Rt_stride0, Rt_stride1);

    if(Rt_stride0 > 0) Rt_stride0 /= sizeof(Rt[0]);
    else               Rt_stride0 =  3;
    if(Rt_stride1 > 0) Rt_stride1 /= sizeof(Rt[0]);
    else               Rt_stride1 =  1;
    if(rt_stride0 > 0) rt_stride0 /= sizeof(rt[0]);
    else               rt_stride0 =  1;

    for(int i=0; i<3; i++)
        rt[(i + 3)*rt_stride0] = Rt[3*Rt_stride0 + i*Rt_stride1];
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_R_from_r(). No
// gradients available here. If you need gradients, call mrcal_R_from_r()
// directly
void mrcal_Rt_from_rt_noncontiguous( // output
                      double* Rt, // (4,3) array
                      int Rt_stride0, // in bytes. <= 0 means "contiguous"
                      int Rt_stride1, // in bytes. <= 0 means "contiguous"

                      // input
                      const double* rt, // (6) vector
                      int rt_stride0    // in bytes. <= 0 means "contiguous"
                     )
{
    mrcal_R_from_r_noncontiguous(Rt,Rt_stride0,Rt_stride1,
                   NULL,  0,0,0,
                   rt, rt_stride0);


    if(Rt_stride0 > 0) Rt_stride0 /= sizeof(Rt[0]);
    else               Rt_stride0 =  3;
    if(Rt_stride1 > 0) Rt_stride1 /= sizeof(Rt[0]);
    else               Rt_stride1 =  1;
    if(rt_stride0 > 0) rt_stride0 /= sizeof(rt[0]);
    else               rt_stride0 =  1;

    for(int i=0; i<3; i++)
        Rt[3*Rt_stride0 + i*Rt_stride1] = rt[(i + 3)*rt_stride0];
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
//   (R0*R1)*x + R0*t1+t0
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

// Compose two rt transformations. It is assumed that we're getting no gradients
// at all or we're getting ALL the gradients: only dr_r0 is checked for NULL
//
// dt_dr1 is not returned: it is always 0
// dt_dt0 is not returned: it is always the identity matrix
void mrcal_compose_rt( // output
                      double* rt_out, // (6) array
                      double* dr_r0,  // (3,3) array; may be NULL
                      double* dr_r1,  // (3,3) array; may be NULL
                      double* dt_r0,  // (3,3) array; may be NULL
                      double* dt_t1,  // (3,3) array; may be NULL

                      // input
                      const double* rt_0, // (6) array
                      const double* rt_1  // (6) array
                     )
{
    // I convert this to Rt to get the composition, and then convert back
    if(dr_r0 == NULL)
    {
        // no gradients
        double Rt_out[4*3];
        double Rt_0  [4*3];
        double Rt_1  [4*3];
        mrcal_Rt_from_rt(Rt_0,   rt_0);
        mrcal_Rt_from_rt(Rt_1,   rt_1);
        mrcal_compose_Rt(Rt_out, Rt_0, Rt_1);
        mrcal_rt_from_Rt(rt_out, Rt_out);
        return;
    }

    // Alright. gradients!
    // I have (R0*R1)*x + R0*t1+t0
    //   r = r_from_R(R0*R1)
    //   t = R0*t1+t0

    double* R0 = dt_t1; // this one is easy!

    double dR0_dr0[3*3*3];
    mrcal_R_from_r( R0, dR0_dr0, rt_0 );

    double R1[3*3];
    double dR1_dr1[3*3*3];
    mrcal_R_from_r( R1, dR1_dr1, rt_1 );

    double R[3*3];
    mul_genN3_gen33_vout(3, R0, R1,  R);

    double dr_dR[3*3*3];
    mrcal_r_from_R( rt_out, dr_dR, R);

    mul_vec3_gen33t_vout( &rt_1[3], R0, &rt_out[3]);
    add_vec(3, &rt_out[3], &rt_0[3]);

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
    memset(dr_r0, 0, 3*3*sizeof(dr_r0[0]));
    for(int i=0; i<3; i++)
    {
        // compute dr_dR R1t dR0_dr0 for submatrix i
        double dr_dR_R1t[3*3];
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
                dr_dR_R1t[j*3+k] = dot_vec( 3, &dr_dR[3*i + j*9], &R1[3*k] );

        mul_genN3_gen33_vaccum(3, dr_dR_R1t, &dR0_dr0[9*i], dr_r0);
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
    memset(dr_r1, 0, 3*3*sizeof(dr_r1[0]));
    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
                mul_vec3_gen33_vaccum_scaled(&dr_dR[i*3 + 9*k], &dR1_dr1[9*j], &dr_r1[3*k], R0[i*3+j]);


    // t = R0*t1+t0
    // t[0] = inner(R0[0:3], t1)
    // -> dt[0]/dr0 = t1t dR0[0:3]/dr0
    for(int i=0; i<3; i++)
        mul_vec3_gen33_vout(&rt_1[3], &dR0_dr0[9*i], &dt_r0[i*3]);
}
