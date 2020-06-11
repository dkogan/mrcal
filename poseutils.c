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
void mrcal_R_from_r( // outputs
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
    else              R_stride0 =  sizeof(R[0])*3;

    if(R_stride1 > 0) R_stride1 /= sizeof(R[0]);
    else              R_stride1 =  sizeof(R[0]);

    if(J_stride0 > 0) J_stride0 /= sizeof(J[0]);
    else              J_stride0 =  sizeof(J[0])*3*3;

    if(J_stride1 > 0) J_stride1 /= sizeof(J[0]);
    else              J_stride1 =  sizeof(J[0])*3;

    if(J_stride2 > 0) J_stride2 /= sizeof(J[0]);
    else              J_stride2 =  sizeof(J[0]);

    if(r_stride0 > 0) r_stride0 /= sizeof(r[0]);
    else              r_stride0 =  sizeof(r[0]);


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
void mrcal_r_from_R( // output
                    double* r, // (3) vector

                    // input
                    const double* R // (3,3) array
                   )
{
    double tr    = R[0] + R[4] + R[8];
    double costh = (tr - 1.) / 2.;

    double th = acos(costh);
    double axis[3] =
        {
            R[2*3 + 1] - R[1*3 + 2],
            R[0*3 + 2] - R[2*3 + 0],
            R[1*3 + 0] - R[0*3 + 1]
        };

    if(th > 1e-10)
    {
        // normal path
        double mag_axis_recip =
            1. /
            sqrt(axis[0]*axis[0] +
                 axis[1]*axis[1] +
                 axis[2]*axis[2]);
        for(int i=0; i<3; i++)
            r[i] = axis[i] * mag_axis_recip * th;
    }
    else
    {
        // small th. Can't divide by it. But I can look at the limit.
        //
        // axis / (2 sinth)*th = axis/2 *th/sinth ~ axis/2
        for(int i=0; i<3; i++)
            r[i] = axis[i] / 2.;
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
    mrcal_r_from_R(rt, Rt);
    for(int i=0; i<3; i++) rt[i+3] = Rt[i+9];
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_R_from_r(). No
// gradients available here. If you need gradients, call mrcal_R_from_r()
// directly
void mrcal_Rt_from_rt( // output
                      double* Rt, // (4,3) array
                      int Rt_stride0, // in bytes. <= 0 means "contiguous"
                      int Rt_stride1, // in bytes. <= 0 means "contiguous"

                      // input
                      const double* rt, // (6) vector
                      int rt_stride0    // in bytes. <= 0 means "contiguous"
                     )
{
    mrcal_R_from_r(Rt,Rt_stride0,Rt_stride1,
                   NULL,  0,0,0,
                   rt, rt_stride0);
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
    mrcal_Rt_from_rt(Rt_0,0,0,   rt_0,0);
    mrcal_Rt_from_rt(Rt_1,0,0,   rt_1,0);
    mrcal_compose_Rt(Rt_out, Rt_0, Rt_1);
    mrcal_rt_from_Rt(rt_out, Rt_out);
}
