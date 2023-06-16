// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <suitesparse/cholmod.h>

#include <string.h>
#include <math.h>
#include <stdarg.h>

#include "mrcal.h"
#include "minimath/minimath-extra.h"
#include "util.h"
#include "strides.h"


#warning "don't duplicate these"
#define SCALE_INTRINSICS_FOCAL_LENGTH 500.0
#define SCALE_INTRINSICS_CENTER_PIXEL 20.0
#define SCALE_ROTATION_CAMERA         (0.1 * M_PI/180.0)
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          (15.0 * M_PI/180.0)
#define SCALE_TRANSLATION_FRAME       1.0
#define SCALE_POSITION_POINT          SCALE_TRANSLATION_FRAME
#define SCALE_CALOBJECT_WARP          0.01
#define SCALE_DISTORTION              1.0




/*
docs in docstring of
reproject_perturbed__optimize_cross_reprojection_error()

I'm looking at a cross-reprojection. This is a least-squares optimization
of a measurement vector x_cross: a perturbation of the already-optimized
calibration measurement vector x. Changes:

- camera intrinsics unperturbed

- camera extrinsics unperturbed

- frame poses perturbed with

  - perturbed pixel observations qref -> qref + dqref, resulting in
    rt_ref_frame -> rt_ref_frame + M dqref =
    rt_refperturbed_frameperturbed

  - reference-correcting transform rt_ref_refperturbed. So the frame
    transform we use is rt_ref_frameperturbed = compose(rt_ref_refperturbed,
    rt_refperturbed_frameperturbed). rt_ref_refperturbed is tiny, and grows
    with dqref: dqref=0 -> rt_ref_refperturbed=0. So I use a linearization
    at rt_ref_refperturbed=0

    rt_ref_frameperturbed ~
      rt_refperturbed_frameperturbed  + drt_ref_frameperturbed__drt_ref_refperturbed drt_ref_refperturbed ~
      rt_ref_frame + M[frame_i] dqref + drt_ref_frameperturbed__drt_ref_refperturbed drt_ref_refperturbed

- calobject_warp perturbed due to the perturbations in pixel observations
  qref -> qref + dqref, resulting in calobject_warp -> calobject_warp +
  M[calobject_warp] dqref

So

  x_cross[i] =
    x[i] +
    J_frame[i]          (M[frame_i] dqref + drt_ref_frameperturbed__drt_ref_refperturbed drt_ref_refperturbed)
    J_calobject_warp[i] M[calobject_warp] dqref

And I define its gradient:

  Jcross[i] = dx_cross[i]/drt_ref_refperturbed
            = J_frame[i] drt_ref_frameperturbed__drt_ref_refperturbed

I reoptimize norm2(x_cross) by varying rt_ref_refperturbed by taking a
single Newton step. I minimize

  E = norm2(x_cross0 + Jcross drt_ref_refperturbed)

I set the derivative to 0:

  0 = dE/drt_ref_refperturbed ~
    ~ (x_cross0 + Jcross drt_ref_refperturbed)t dx_cross/drt_ref_refperturbed
    = (x_cross0 + Jcross drt_ref_refperturbed)t Jcross

->
  drt_ref_refperturbed =
    rt_ref_refperturbed =
    -inv(Jcross_t Jcross) Jcross_t x_cross0

  x_cross0[i] = x_cross(rt_ref_refperturbed = 0) =
                x[i] +
                J_frame[i]          M[frame_i] dqref
                J_calobject_warp[i] M[calobject_warp] dqref

Note that if dqref = 0, then x_cross0 = x: the cross-reprojection is
equivalent to the baseline projection error, which is as expected.

If dqref = 0, the rt_ref_refperturbed = -inv() Jcross_t x, which is 0 if we
ignore regularization. Let's do that.

rt_ref_refperturbed is a random variable, since dqref is a random
variable. The relationship is nicely linear, so I can compute:

  mean(rt_ref_refperturbed) = -inv(Jcross_t Jcross) Jcross_t x

I have expressions with J, but in reality I have J*: gradients in respect to
PACKED variables. I have a variable scaling diagonal matrix D: J = J* D

From the finished uncertainty documentation

  M = inv(JtJ) Jobservationst W
    = inv( Dt J*t J* D ) D Jobservations*t W
    = invD inv(J*t J*) Jobservations*t W

  Var(qref) = s^2 W^-2

So all the W cancel each other out.

Let's explicate the matrices. (J_frame[] M[frame] + J_calobject_warp[]
M[calobject_warp]) is J_no_ie M where

              i  e        f  calobject_warp
              |  |        |        |
              V  V        V        V
            [ 0 | 0 | ---       | --- ]
            [ 0 | 0 | ---       | --- ]
            [ 0 | 0 | ---       | --- ]
            [ 0 | 0 |    ---    | --- ]
  J_no_ie = [ 0 | 0 |    ---    | --- ]
            [ 0 | 0 |    ---    | --- ]
            [ 0 | 0 |       --- | --- ]
            [ 0 | 0 |       --- | --- ]
            [ 0 | 0 |       --- | --- ]

And

           [ --- drr0 ]
           [ --- drr0 ]
           [ --- drr0 ]
           [ --- drr1 ]
  Jcross = [ --- drr1 ]
           [ --- drr1 ]
           [ --- drr2 ]
           [ --- drr2 ]
           [ --- drr2 ]

Where the --- terms are the flattened "frame" terms from J_no_ie that use
the unpacked state. And drr are the
drt_ref_frameperturbed__drt_ref_refperturbed matrices for the different
rt_ref_frame vectors.

Putting everything together, we have

  rt_ref_refperturbed = -inv(Jcross_t Jcross) Jcross_t x_cross0
                      = -inv(Jcross_t Jcross) Jcross_t J[frame,calobject_warp] db[frame,calobject_warp]
                      = -inv(Jcross_t Jcross) Jcross_t J_no_ie db
                      = -inv(Jcross_t Jcross) Jcross_t J_no_ie* Dinv db
                      = K Dinv db

where

  K = -inv(Jcross_t Jcross)    Jcross_t          J_no_ie*
               (6,6)        (6, Nmeas_obs)  (Nmeas_obs,Nstate)

Finishing it:

  rt_ref_refperturbed = K Dinv db
                      = K Dinv D inv(J*t J*) Jobservations*t W dqref
                      = K inv(J*t J*) Jobservations*t W dqref

I need to compute Jcross_t J_no_ie* (shape (6,Nstate)). Its transpose, for convenience;

  J_no_ie_t* Jcross (shape=(Nstate,6)) =
    [ 0                                                                                     ] <- intrinsics
    [ 0                                                                                     ] <- extrinsics
    [ sum_measi(outer(j_frame0*, j_frame0*)) Dinv drr_frame0                                ]
    [ sum_measi(outer(j_frame1*, j_frame1*)) Dinv drr_frame1                                ] <- frames
    [                         ...                                                           ]
    [ sum_framei(sum_measi(outer(j_calobject_warp_measi*, j_frame_measi*) Dinv drr_framei)) ] <- calobject_warp

  Jcross_t Jcross = sum(outer(jcross, jcross))
                  = sum_framei( drr_framei_t Dinv sum_measi(outer(j_frame_measi*, j_frame_measi*)) Dinv drr_framei )

For each frame, both of these expressions need

  sum_measi(outer(j_frame_measi*, j_frame_measi*)) Dinv drr_framei

I compute this in a loop, and accumulate in finish_Jcross_computations()

*/

static
void finish_Jcross_computations(// output
                                double* Jcross_t__J_fcw,
                                int     Jcross_t__J_fcw_stride0_elems,
                                // rows are assumed stored densely, so there is
                                // no Jcross_t__J_fcw_stride1
                                double* Jcross_t__Jcross,

                                // input

                                // Would be const, but I memset(0) at the end
                                double* sum_outer_jf_jf_packed,
                                double* sum_outer_jf_jcw_packed,
                                const double* rt1_packed,
                                int state_index_frame_current,
                                int state_index_frame0,
                                int state_index_calobject_warp0,
                                int Nstate_noi_noe)
{
    // I accumulated sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) into
    // sum_outer_jf_jf_packed. This is needed to compute both Jcross_t
    // J_fcw* and Jcross_t Jcross, which I do here.
    //
    // sum_outer_jf_jf_packed stores only the upper triangle is stored, in
    // the usual row-major order. sum_outer_jf_jf_packed uses PACKED
    // gradients, which need to be unpacked in some cases. These SCALE
    // factors explained further down
    //
    // Jcross_t__J_fcw[:, iframe0:iframe+6] =
    //   drt_ref_frameperturbed/drt_ref_refperturbed__t sum_outer_jf_jf_packed /SCALE
    //
    // Jcross_t Jcross = sum(outer(jcross, jcross))
    //                   = sum_i( drr[i]t sum_outer_jf_jf_packed drr[i] ) /SCALE/SCALE
    //
    // Jcross has full state, but J_fcw* has packed state, so I need
    // different number of SCALE factors.
    //
    // Jcross_t__J_fcw ~ drr_t j jpt = drr_t Dinv jp jpt
    // Jcross_t Jcross ~ drr_t j jt drr ~ Jcross_t__J_fcw Dinv drr
    //
    // In the code I have sum_outer_jf_jf_packed ~ jp jpt
    //
    // &Jcross_t__J_fcw[state_index_frame_current] is the first element of
    // the output for this frame
    //
    // I have 4 triangles to process with the different gradients, as
    // described above.
    //
    //   drr = [dr/dr      0]
    //         [ -skew(t1) I]
    //
    //   drr_t S = [dr/dr_t skew_t1] [ S00 S01 ] = [ dr/dr_t S00 + skew_t1 S10    dr/dr_t S01 + skew_t1 S11]
    //             [ 0      I      ] [ S10 S11 ]   [ S10                          S11                      ]
    //
    // In the case of the frames, each Sxx block has shape (3,3). For
    // calobject_warp, S has shape (6,2) so I only have S00 and S10, each
    // with shape (3,2)

    // I will need composition gradients assuming tiny rt0. I have
    //   compose(rt0, rt1) = compose(r0,r1), rotate(r0,t1)+t0
    // I need to get gradient drt_ref_frameperturbed/drt_ref_refperturbed. Let's
    // look at the r,t separately. I have:
    //   dr/dr0: This is complex. I compute it and store it into this matrix
    //   dr/dt0 = 0
    //   dt/dr0 = -skew(t1)
    //   dt/dt0 = I
    //
    // where
    //             [  0 -t2  t1]
    //   skew(t) = [ t2   0 -t0]
    //             [-t1  t0   0]

    double dr_ref_frameperturbed__dr_ref_refperturbed[3*3];

#warning UNJUSTIFIED ASSUMPTION
    // UNJUSTIFIED ASSUMPTION HERE. This should use
    // r_refperturbed_frameperturbed = r_ref_frame + M[] dqref, but that makes
    // my life much more complex, so I just use the unperturbed
    // r_ref_frame. I'll try to show empirically that this is just
    // as good
    const double r_ref_frame[3] =
        { rt1_packed[0] * SCALE_ROTATION_FRAME,
          rt1_packed[1] * SCALE_ROTATION_FRAME,
          rt1_packed[2] * SCALE_ROTATION_FRAME };
    mrcal_compose_r_tinyr0_gradientr0(dr_ref_frameperturbed__dr_ref_refperturbed,
                                      r_ref_frame);

    // Jcross_t__J_fcw output goes into [A B]
    //                                  [C D]
    double* A = &Jcross_t__J_fcw[0*Jcross_t__J_fcw_stride0_elems + state_index_frame_current-state_index_frame0 + 0];
    double* B = &Jcross_t__J_fcw[0*Jcross_t__J_fcw_stride0_elems + state_index_frame_current-state_index_frame0 + 3];
    double* C = &Jcross_t__J_fcw[3*Jcross_t__J_fcw_stride0_elems + state_index_frame_current-state_index_frame0 + 0];
    double* D = &Jcross_t__J_fcw[3*Jcross_t__J_fcw_stride0_elems + state_index_frame_current-state_index_frame0 + 3];

    // for calobject_warp
    double* Acw = &Jcross_t__J_fcw[0*Jcross_t__J_fcw_stride0_elems + state_index_calobject_warp0-state_index_frame0];
    double* Ccw = &Jcross_t__J_fcw[3*Jcross_t__J_fcw_stride0_elems + state_index_calobject_warp0-state_index_frame0];

    // I can compute Jcross_t Jcross from the blocks comprising Jcross_t
    // J_fcw. From above:
    //
    // Jcross_t Jcross ~
    //   ~ Jcross_t__J_fcw Dinv drr
    //
    //   ~ [A B] Dinv drr
    //     [C D]
    //
    //   = [A/SCALE_R B/SCALE_T] [dr/dr      0]
    //     [C/SCALE_R D/SCALE_T] [ -skew(t1) I]
    //
    //   = [A/SCALE_R dr/dr - B/SCALE_T skew(t1)    B/SCALE_T]
    //     [...                                     D/SCALE_T]
    //
    // Jcross_t__Jcross is symmetric, so I just compute the upper triangle,
    // and I don't care about the ... block
    const double t0 = rt1_packed[3+0] * SCALE_TRANSLATION_FRAME;
    const double t1 = rt1_packed[3+1] * SCALE_TRANSLATION_FRAME;
    const double t2 = rt1_packed[3+2] * SCALE_TRANSLATION_FRAME;

    // A <- dr/dr_t sum_outer[:3,:3] + skew_t1 sum_outer[3:,:3]
    {
        mul_gen33_gen33insym66(A, Jcross_t__J_fcw_stride0_elems, 1,
                               // transposed, so 1,3 and not 3,1
                               dr_ref_frameperturbed__dr_ref_refperturbed, 1,3,
                               sum_outer_jf_jf_packed, 0, 0,
                               1./SCALE_ROTATION_FRAME);

        // and similar for calobject_warp
        // Acw = drr_t Dinv S; ~
        // -> Acwt = St Dinv drr;
        mul_genNM_genML_accum(// transposed
                              Acw, 1, Jcross_t__J_fcw_stride0_elems,

                              2,3,3,
                              // transposed
                              &sum_outer_jf_jcw_packed[0*2 + 0], 1,2,
                              dr_ref_frameperturbed__dr_ref_refperturbed, 3,1,
                              1./SCALE_ROTATION_FRAME);

        for(int j=0; j<3; j++)
        {
            int i;

            i = 0;
            A[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]   + (  0)*sum_outer_jf_jf_packed[index_sym66(0+3,j)] */
                 /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jf_jf_packed[index_sym66(1+3,j)]
                 /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jf_jf_packed[index_sym66(2+3,j)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 1;
            A[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jf_jf_packed[index_sym66(0+3,j)]
                 /*skew[i*3 + 1]   + (  0)*sum_outer_jf_jf_packed[index_sym66(1+3,j)] */
                 /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jf_jf_packed[index_sym66(2+3,j)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 2;
            A[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jf_jf_packed[index_sym66(0+3,j)]
                 /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jf_jf_packed[index_sym66(1+3,j)]
                 /*skew[i*3 + 2]   + (  0)*sum_outer_jf_jf_packed[index_sym66(2+3,j)] */
                 ) / SCALE_TRANSLATION_FRAME;

            // and similar for calobject_warp
            if(j<2)
            {
                i = 0;
                Acw[i*Jcross_t__J_fcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]   + (  0)*sum_outer_jf_jcw_packed[(0+3)*2 + j] */
                     /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jf_jcw_packed[(1+3)*2 + j]
                     /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jf_jcw_packed[(2+3)*2 + j]
                     ) / SCALE_TRANSLATION_FRAME;

                i = 1;
                Acw[i*Jcross_t__J_fcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jf_jcw_packed[(0+3)*2 + j]
                     /*skew[i*3 + 1]   + (  0)*sum_outer_jf_jcw_packed[(1+3)*2 + j] */
                     /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jf_jcw_packed[(2+3)*2 + j]
                     ) / SCALE_TRANSLATION_FRAME;

                i = 2;
                Acw[i*Jcross_t__J_fcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jf_jcw_packed[(0+3)*2 + j]
                     /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jf_jcw_packed[(1+3)*2 + j]
                     /*skew[i*3 + 2]   + (  0)*sum_outer_jf_jcw_packed[(2+3)*2 + j] */
                     ) / SCALE_TRANSLATION_FRAME;
            }
        }
    }

    // B <- dr/dr_t sum_outer[:3,3:] + skew_t1 sum_outer[3:,3:]
    {
        mul_gen33_gen33insym66(B, Jcross_t__J_fcw_stride0_elems, 1,
                               // transposed, so 1,3 and not 3,1
                               dr_ref_frameperturbed__dr_ref_refperturbed, 1,3,
                               sum_outer_jf_jf_packed, 0, 3,
                               1./SCALE_ROTATION_FRAME);

        for(int j=0; j<3; j++)
        {
            int i;

            i = 0;
            B[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]   + (  0)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)] */
                 /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)]
                 /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 1;
            B[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)]
                 /*skew[i*3 + 1]   + (  0)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)] */
                 /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 2;
            B[i*Jcross_t__J_fcw_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)]
                 /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)]
                 /*skew[i*3 + 2]   + (  0)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)] */
                 ) / SCALE_TRANSLATION_FRAME;
        }
    }

    // C <- sum_outer[3:,:3]
    {
        set_gen33_from_gen33insym66(C, Jcross_t__J_fcw_stride0_elems, 1,
                                    sum_outer_jf_jf_packed, 3, 0,
                                    1./SCALE_TRANSLATION_FRAME);

        // and similar for calobject_warp
        for(int i=0; i<3; i++)
            for(int j=0; j<2; j++)
                Ccw[i*Jcross_t__J_fcw_stride0_elems + j] +=
                    sum_outer_jf_jcw_packed[(3+i)*2 + j]/SCALE_TRANSLATION_FRAME;

    }

    // D <- sum_outer[3:,3:]
    {
        set_gen33_from_gen33insym66(D, Jcross_t__J_fcw_stride0_elems, 1,
                                    sum_outer_jf_jf_packed, 3, 3,
                                    1./SCALE_TRANSLATION_FRAME);
    }

    // Jcross_t__Jcross[rr] <- A/SCALE_R dr/dr - B/SCALE_T skew(t1)
    {
        mul_gen33_gen33_into33insym66_accum(Jcross_t__Jcross, 0, 0,
                                            A, Jcross_t__J_fcw_stride0_elems, 1,
                                            dr_ref_frameperturbed__dr_ref_refperturbed, 3,1,
                                            1./SCALE_ROTATION_FRAME);

        int ivalue = 0;
        for(int i=0; i<3; i++)
        {
            for(int j=i; j<3; j++, ivalue++)
            {
                if(j == 0)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]   + B[i*Jcross_t__J_fcw_stride0_elems+0]*(  0) */
                         /*skew[j + 1*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+1]*( t2)
                         /*skew[j + 2*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+2]*(-t1)
                         ) / SCALE_TRANSLATION_FRAME;

                if(j == 1)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+0]*(-t2)
                         /*skew[j + 1*3]   + B[i*Jcross_t__J_fcw_stride0_elems+1]*(  0) */
                         /*skew[j + 2*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+2]*( t0)
                         ) / SCALE_TRANSLATION_FRAME;

                if(j == 2)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+0]*( t1)
                         /*skew[j + 1*3]*/ + B[i*Jcross_t__J_fcw_stride0_elems+1]*(-t0)
                         /*skew[j + 2*3]   + B[i*Jcross_t__J_fcw_stride0_elems+2]*(  0) */
                         ) / SCALE_TRANSLATION_FRAME;
            }
            ivalue += 3;
        }
    }

    // Jcross_t__Jcross[rt] <- B/SCALE_T
    {
        set_33insym66_from_gen33_accum(Jcross_t__Jcross, 0, 3,
                                       B, Jcross_t__J_fcw_stride0_elems, 1,
                                       1./SCALE_TRANSLATION_FRAME);
    }

    // Jcross_t__Jcross[tr] doesn't need to be set: I only have values in
    // the upper triangle

    // Jcross_t__Jcross[tt] <- D/SCALE_T = sum_outer[3:,3:]/SCALE_T/SCALE_T
    {
        const int N = (6+1)*6/2;
        const int i0 = index_sym66_assume_upper(3,3);
        for(int i=i0; i<N; i++)
            Jcross_t__Jcross[i] +=
                sum_outer_jf_jf_packed[i] /
                (SCALE_TRANSLATION_FRAME*SCALE_TRANSLATION_FRAME);
    }

    memset(sum_outer_jf_jf_packed,  0, (6+1)*6/2*sizeof(double));
    memset(sum_outer_jf_jcw_packed, 0, 6*2      *sizeof(double));
}


bool mrcal_drt_ref_refperturbed__dbpacked_no_ie(// output
                                                // Shape (6,Nstate_noi_noe)
                                                double* K,
                                                int K_stride0, // in bytes. <= 0 means "contiguous"
                                                int K_stride1, // in bytes. <= 0 means "contiguous"

                                                // inputs
                                                // stuff that describes this solve
                                                const double* b_packed,
                                                // used only to confirm that the user passed-in the buffer they
                                                // should have passed-in. The size must match exactly
                                                int buffer_size_b_packed,

                                                // The unitless Jacobian, used by the internal
                                                // optimization routines
                                                // cholmod_analyze() and cholmod_factorize()
                                                // require non-const
                                                /* const */
                                                cholmod_sparse* Jt,

                                                // meta-parameters
                                                int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                                                int Npoints, int Npoints_fixed, // at the end of points[]
                                                int Nobservations_board,
                                                int Nobservations_point,

                                                const mrcal_lensmodel_t* lensmodel,
                                                mrcal_problem_selections_t problem_selections,

                                                int calibration_object_width_n,
                                                int calibration_object_height_n)
{
    const int Nmeas_boards =
        mrcal_num_measurements_boards(Nobservations_board,
                                      calibration_object_width_n,
                                      calibration_object_height_n);
    const int Nmeas_points =
        mrcal_num_measurements_points(Nobservations_point);

    const int Nmeas_obs = Nmeas_boards + Nmeas_points;

    const int state_index_frame0 =
        mrcal_state_index_frames(0,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
    const int state_index_calobject_warp0 =
        mrcal_state_index_calobject_warp(Ncameras_intrinsics, Ncameras_extrinsics,
                                         Nframes,
                                         Npoints, Npoints_fixed, Nobservations_board,
                                         problem_selections,
                                         lensmodel);
    const int Nstate =
        mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                         Nframes,
                         Npoints, Npoints_fixed, Nobservations_board,
                         problem_selections,
                         lensmodel);

    const int num_states_frames =
        mrcal_num_states_frames(Nframes,
                                problem_selections);

    const int num_states_intrinsics =
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_selections,
                                    lensmodel);
    const int num_states_extrinsics =
        mrcal_num_states_extrinsics(Ncameras_extrinsics,
                                    problem_selections);
    const int num_states_calobject_warp =
        mrcal_num_states_calobject_warp(problem_selections,
                                        Nobservations_board);
    const int Nstate_i_e = num_states_intrinsics + num_states_extrinsics;
    const int Nstate_noi_noe = Nstate - Nstate_i_e;


    double Jcross_t__Jcross[(6+1)*6/2] = {};

    if( buffer_size_b_packed != Nstate*(int)sizeof(double) )
    {
        MSG("The buffer b_packed has the wrong size. Needed exactly %d bytes, but got %d bytes",
            Nstate*(int)sizeof(double),buffer_size_b_packed);
        return false;
    }

    if(Nmeas_points != 0)
    {
        MSG("ERROR: %s() currently is not implemented for point observations", __func__);
        return false;
    }
    if(Nstate != (int)Jt->nrow)
    {
        MSG("Inconsistent inputs. I have Nstate=%d, but Jt->nrow=%d. Giving up",
            Nstate, (int)Jt->nrow);
        return false;
    }
    if(state_index_frame0 < 0)
    {
        MSG("Uncertainty computation is currently implemented only if frames are being optimized");
        return false;
    }
    if(state_index_calobject_warp0 < 0)
    {
        MSG("Uncertainty computation is currently implemented only if calobject-warp is being optimized");
        return false;
    }
    if(state_index_frame0 + num_states_frames != state_index_calobject_warp0)
    {
        MSG("I'm assuming that the calobject-warp state directly follows the frames, but here it does not. Giving up");
        return false;
    }
    if(state_index_calobject_warp0 + num_states_calobject_warp != Nstate)
    {
        MSG("I'm assuming calobject_warp is the last set of states, but here it is not. Giving up");
        return false;
    }
    if( state_index_frame0 != Nstate_i_e)
    {
        MSG("Unexpected state vector layout. Giving up");
        return false;
    }

    init_stride_2D(K, 6, Nstate_noi_noe);

    const int K_stride0_elems = K_stride0 / sizeof(double);
    if(K_stride0_elems*(int)sizeof(double) != K_stride0)
    {
        MSG("Currently the implementation assumes that K_stride0 is a multiple of sizeof(double): all elements of K are aligned. Got K_stride0 = %d",
            K_stride0);
        return false;
    }


    if(K_stride1 == sizeof(double))
    {
        // each row is stored densely
        if(K_stride0 == (int)sizeof(double)*Nstate_noi_noe)
            // each column is stored densely as well. I can memset() the whole
            // block of memory
            memset(K, 0, 6*Nstate_noi_noe*sizeof(double));
        else
            for(int i=0; i<6; i++)
                memset(&K[i*K_stride0_elems], 0, Nstate_noi_noe*sizeof(double));
    }
    else
    {
        MSG("Currently the implementation assumes that K has densely-stored rows: K_stride1 must be sizeof(double). Instead I got K_stride1 = %d",
            K_stride1);
        return false;
    }

#warning "I should reuse some other memory for this. Chunks of K ?"
    // sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) for this frame. I sum over
    // all the observations. Uses PACKED gradients. Only the upper triangle is
    // stored, in the usual row-major order
    double sum_outer_jf_jf_packed[(6+1)*6/2] = {};

    // sum(outer(j_frame_measi*, j_calobject_warp_measi*)) for this frame. Uses
    // PACKED gradients. Stored densely, since it isn't symmetric. Shape (6,2)
    double sum_outer_jf_jcw_packed[6*2] = {};

    int state_index_frame_current = -1;

    const int*    Jrowptr = (int*)   Jt->p;
    const int*    Jcolidx = (int*)   Jt->i;
    const double* Jval    = (double*)Jt->x;
    for(int imeas=0; imeas<Nmeas_obs; imeas++)
    {
        // I have dx/drt_ref_frame for this frame. This is 6 numbers
        const double* dx_drt_ref_frame_packed = NULL;

        for(int32_t ival = Jrowptr[imeas]; ival < Jrowptr[imeas+1]; ival++)
        {
            int32_t icol = Jcolidx[ival];
#warning "I can do better than a linear search here. I know the structure of J."
            if(icol < state_index_frame0)
                // not a rt_ref_frame gradient. Ignore
                continue;

            // We're looking at SOME rt_ref_frame gradient. I expect these to be
            // non-decreasing: consecutive chunks of Nw*Nh*2 measurements will
            // represent the same board pose, and the same rt_ref_frame
            if(icol < state_index_frame_current)
            {
                MSG("Unexpected jacobian structure. I'm assuming non-decreasing frame references");
                return false;
            }
            else if( state_index_frame0 <= icol &&
                     icol < state_index_calobject_warp0 )
            {
                // Looking at a new frame. Finish the previous frame
                if(icol != state_index_frame_current)
                {
                    if(state_index_frame_current >= 0)
                    {
                        finish_Jcross_computations( K, K_stride0_elems,
                                                    Jcross_t__Jcross,
                                                    sum_outer_jf_jf_packed,
                                                    sum_outer_jf_jcw_packed,
                                                    &b_packed[state_index_frame_current],
                                                    state_index_frame_current,
                                                    state_index_frame0,
                                                    state_index_calobject_warp0,
                                                    Nstate_noi_noe);
                    }
                    state_index_frame_current = icol;
                }

                // I have dx/drt_ref_frame for this frame. This is 6 numbers
                dx_drt_ref_frame_packed = &Jval[ival];

                // sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) into sum_outer_jf_jf_packed
                {
                    // This is used to compute Jcross_t J_no_ie* and Jcross_t
                    // Jcross. This result is used in finish_Jcross_computations()
                    //
                    // Uses PACKED gradients. Only the upper triangle is stored, in
                    // the usual row-major order
                    int ivalue = 0;
                    for(int i=0; i<6; i++)
                        for(int j=i; j<6; j++, ivalue++)
                            sum_outer_jf_jf_packed[ivalue] +=
                                dx_drt_ref_frame_packed[i]*dx_drt_ref_frame_packed[j];
                }

                // fast-forward past the frame gradients
                ival += 6-1;
            }
            else
            {
                // if() statements above guarantee that this is calobject_warp.
                const double* dx_dcalobject_warp_packed = &Jval[ival];

                // Similar to the above, but this isn't symmetric, so I store it
                // densely
                int ivalue = 0;
                for(int i=0; i<6; i++)
                    for(int j=0; j<2; j++, ivalue++)
                        sum_outer_jf_jcw_packed[ivalue] +=
                            dx_drt_ref_frame_packed[i]*
                            dx_dcalobject_warp_packed[j];

                ival += num_states_calobject_warp-1;
            }
        }
    }

    if(state_index_frame_current >= 0)
        finish_Jcross_computations( K, K_stride0_elems,
                                    Jcross_t__Jcross,
                                    sum_outer_jf_jf_packed,
                                    sum_outer_jf_jcw_packed,
                                    &b_packed[state_index_frame_current],
                                    state_index_frame_current,
                                    state_index_frame0,
                                    state_index_calobject_warp0,
                                    Nstate_noi_noe);


    // I now have filled Jcross_t__Jcross and K. I can
    // compute
    //
    //   inv(Jcross_t Jcross) Jcross_t J_no_ie
    //
    // I actually compute the transpose:
    //
    //   (Jcross_t J_no_ie)t inv(Jcross_t Jcross)
    //
    // in-place: input and output both use the K array
    double inv_JcrosstJcross_det[(6+1)*6/2];
    double det = cofactors_sym6(Jcross_t__Jcross,
                                inv_JcrosstJcross_det);

    // Overwrite K in place
    mul_genN6_sym66_scaled_strided(Nstate_noi_noe,
                                   K, 1, K_stride0_elems,
                                   inv_JcrosstJcross_det,
                                   -1. / det);

    return true;
}
