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

#include "mrcal.h"
#include "minimath/minimath-extra.h"
#include "_util.h"
#include "_strides.h"
#include "scales.h"



/*
Detailed docs appear in the docstring of

  https://mrcal.secretsauce.net/uncertainty-cross-reprojection.html

The punchline:

  Jcross_e = J_extrinsics d(compose_rt(rt_cam_ref,rt_ref_ref*))  /drt_ref_ref*
  Jcross_f = J_frame      d(compose_rt(rt_ref_ref*,rt_ref_frame))/drt_ref_ref*
  Jcross_p = J_p          d(transform(rt_ref_ref*,p ))           /drt_ref_ref*

  For each mesurement I pick one of these expressions.

  We have rt_ref_ref* = K db for some K that depends on the various J matrices
  that are constant for each solve:

    K = -pinv(Jcross) J[frames,points,calobject_warp]

THIS function computes

  Kpacked = K D

Let's explicate the matrices.

              i   e           f                  p     calobject_warp
              |   |           |                  |           |
              V   V           V                  V           V
            [ 0 | 0 | J0_f0             |                | J0_cw ]
            [ 0 | 0 | J1_f0             |                | J1_cw ]
            [ 0 | 0 | J2_f0             |                | J2_cw ]
            [ 0 | 0 |       J3_f1       |                | J3_cw ]
            [ 0 | 0 |       J4_f1       |                | J4_cw ]
            [ 0 | 0 |       J5_f1       |                | J5_cw ]
  J_fpcw =  [ 0 | 0 |             J6_f2 |                | J6_cw ]
            [ 0 | 0 |             J7_f2 |                | J7_cw ]
            [ 0 | 0 |             J8_f2 |                | J8_cw ]
            [ 0 | 0 |                   | J9_p0          |       ]
            [ 0 | 0 |                   | J10_p0         |       ]
            [ 0 | 0 |                   |         J11_p1 |       ]


And

           [ J0_f0  drt_rf0__drt_rrp ]
           [ J1_f0  drt_rf0__drt_rrp ]
           [        ....             ]
           [ J3_f1  drt_rf1__drt_rrp ]
           [ J4_f1  drt_rf1__drt_rrp ]
  Jcross = [        ....             ]
           [ J6_e0  drt_cr0__drt_rrp ]
           [ J7_e0  drt_cr0__drt_rrp ]
           [        ....             ]
           [ J9_p0  dp0__drt_rrp     ]
           [ J10_p0 dp0__drt_rrp     ]
           [        ....             ]

Here I used Jcross_e in measurement blocks 6,7 and Jcross_f for the rest.

Note: these all depend ONLY on rt_cam_ref and rt_ref_frame and p, which are
quantities we have. These do NOT depend on rt_ref_ref*.

Putting everything together, we have

  rt_ref_ref* = K db
              = -pinv(Jcross) J_fpcw db
              = -pinv(Jcross) J_packedfpcw dbpacked
              = Kpacked dbpacked

so

  Kpacked = -inv(Jcross_t Jcross)    Jcross_t        J_packedfpcw
                     (6,6)        (6, Nmeas_obs)   (Nmeas_obs,Nstate)

Usually Nmeas_obs >> Nstate, so I start by computing Jcross_t J_packedfpcw
(shape (6,Nstate)). Its transpose, for convenience;

  J_packedfpcw_t Jcross (shape=(Nstate,6)) =
    [ 0                                                                                     ] <- intrinsics
    [ 0                                                                                     ] <- extrinsics
    [ J0_packedf0_t J0_packedf0   Dinv drt_rf0__drt_rrp                                     ]
    [ J1_packedf0_t J1_packedf0   Dinv drt_rf0__drt_rrp                                     ]
    [                         ...                                                           ]
    [ J3_packedf1_t J3_packedf1   Dinv drt_rf1__drt_rrp                                     ]
    [ J4_packedf1_t J4_packedf1   Dinv drt_rf1__drt_rrp                                     ] <- frames
    [                         ...                                                           ]
    [ J6_packedf2_t J0_packede0   Dinv drt_cr0__drt_rrp                                     ]
    [ J7_packedf2_t J1_packede0   Dinv drt_cr0__drt_rrp                                     ]
    [                         ...                                                           ]
    [ J9_packedp0_t  J9_packedp0  Dinv dp0__drt_rrp                                         ] <- points
    [ J10_packedp0_t J10_packedp0 Dinv dp0__drt_rrp                                         ]
    [                         ...                                                           ]
    [ sum_i(Ji_cw_t Ji_packedfj Dinv drt_rfj__drt_rrp ) +                                   ] <- calobject_warp
    [ sum_i(Ji_cw_t Ji_packedej Dinv drt_crj__drt_rrp )                                     ]

  Jcross_t Jcross = sum(outer(jcross, jcross))
                  = sum_i( drt_rfj__drt_rrp_t Ji_fj_t Ji_fj drt_rfj__drt_rrp ) +
                    sum_i( drt_crj__drt_rrp_t Ji_ej_t Ji_ej drt_crj__drt_rrp ) +
                    sum_i( dpj__drt_rrp_t     Ji_fj_t Ji_fj dpj__drt_rrp )

For each frame, both of these expressions need

  Ji.._t Ji Dinv D...__drt_rtp

I compute this in a loop, and accumulate in accumulate_frame() and
accumulate_point()

*/

static
void accumulate_frame(// output
                      // shape (6,6)
                      double*   Jcross_t__Jpackedf, // THIS one frame, many measurements
                      const int Jcross_t__Jpackedf_stride0_elems,
                      // rows are assumed stored densely, so there is
                      // no Jcross_t__Jpackedf_stride1

                      // shape (6,2)
                      double*   Jcross_t__Jpackedcw, // THIS one frame, many measurements
                      const int Jcross_t__Jpackedcw_stride0_elems,
                      // rows are assumed stored densely, so there is
                      // no Jcross_t__Jpackedcw_stride1

                      // shape (6,6)
                      double* Jcross_t__Jcross,

                      // input
                      // shape (6,6); symmetric, upper-triangle-only is stored
                      const double* sum_outer_jpackedf_jpackedf,
                      // shape (6,2)
                      const double* sum_outer_jpackedf_jpackedcw,
                      // shape (6,)
                      const double* rt_ref_frame_packed)
{
    // sum_outer_jpackedf_jpackedf stores only the upper triangle, in the usual
    // row-major order.
    //
    // Jcross_t__Jpackedf[:, iframe0:iframe+6] =
    //   drtrfp_drtrrp_t sum_outer_jpackedf_jpackedf /SCALE
    //
    // where drtrfp_drtrrp = d(compose_rt(rt_ref_ref*,rt_ref_frame)) / drt_ref_ref*
    //
    // Jcross_t Jcross = sum(outer(jcross, jcross))
    //                 = sum_i( drtrfp_drtrrp_t[i]
    //                          sum_outer_jpackedf_jpackedf
    //                          drtrfp_drtrrp[i] ) /SCALE/SCALE
    //
    // Jcross has full state, but J_packed has packed state, so I need different
    // number of SCALE factors.
    //
    // drtrfp_drtrrp = d(compose(rt0,rt1)/drt0) where rt0 is tiny. Derivation:
    //
    //   R0 (R1 p + t1) + t0 = R0 R1 p + (R0 t1 + t0)
    //   -> R01 = R0 R1
    //   -> t01 = R0 t1 + t0
    //
    //   At rt0 ~ identity we have:
    //     dt01/dr0 = d(R0 t1)/dr0
    //
    //   rotate_point_r_core() says that
    //     const val_withgrad_t<N> cross[3] =
    //         {
    //             (rg[1]*x_ing[2] - rg[2]*x_ing[1])*sign,
    //             (rg[2]*x_ing[0] - rg[0]*x_ing[2])*sign,
    //             (rg[0]*x_ing[1] - rg[1]*x_ing[0])*sign
    //         };
    //     const val_withgrad_t<N> inner =
    //         rg[0]*x_ing[0] +
    //         rg[1]*x_ing[1] +
    //         rg[2]*x_ing[2];
    //     // Small rotation. I don't want to divide by 0, so I take the limit
    //     //   lim(th->0, xrot) =
    //     //     = x + cross(r, x) + r rt x lim(th->0, (1 - cos(th)) / (th*th))
    //     //     = x + cross(r, x) + r rt x lim(th->0, sin(th) / (2*th))
    //     //     = x + cross(r, x) + r rt x/2
    //     for(int i=0; i<3; i++)
    //         x_outg[i] =
    //             x_ing[i] +
    //             cross[i] +
    //             rg[i]*inner / 2.;
    //
    //   So t01 = t0 + t1 + linear(r0) + quadratic(r0)
    //   r0 ~ 0 so I ignore the quadratic term:
    //     dt01/dr0 = d(cross(r0,t1))/dr0
    //              = -d(cross(t1,r0))/dr0
    //              = -d(skew_symmetric(t1) r0))/dr0
    //              = -skew_symmetric(t1)
    //   Thus
    //     drt01/drt0 = [ dr01/dr0  dr01/dt0  ] = [ dr01/dr0              0 ]
    //                  [ dt01/dr0  dt01/dt0  ] = [ -skew_symmetric(t1)   I ]
    //
    // In the above expressions I have drtrfp_drtrrp_t S for some matrix S. Expanded:
    //
    //   drtrfp_drtrrp_t S = [dr/dr_t skew_t1] [ S00 S01 ] = [ dr/dr_t S00 + skew_t1 S10    dr/dr_t S01 + skew_t1 S11]
    //                       [ 0      I      ] [ S10 S11 ]   [ S10                          S11                      ]
    //
    // In the case of the frames, each Sxx block has shape (3,3). For
    // calobject_warp, S has shape (6,2) so I only have S00 and S10, each
    // with shape (3,2)
    double drrfp_drrrp[3*3];

    const double r_ref_frame[3] =
        { rt_ref_frame_packed[0] * SCALE_ROTATION_FRAME,
          rt_ref_frame_packed[1] * SCALE_ROTATION_FRAME,
          rt_ref_frame_packed[2] * SCALE_ROTATION_FRAME };
    mrcal_compose_r_tinyr0_gradientr0(drrfp_drrrp,
                                      r_ref_frame);

    // Jcross_t__Jpackedf output goes into [Af Bf]
    //                                     [Cf Df]

    double* Af = &Jcross_t__Jpackedf[Jcross_t__Jpackedf_stride0_elems*0 + 0];
    double* Bf = &Jcross_t__Jpackedf[Jcross_t__Jpackedf_stride0_elems*0 + 3];
    double* Cf = &Jcross_t__Jpackedf[Jcross_t__Jpackedf_stride0_elems*3 + 0];
    double* Df = &Jcross_t__Jpackedf[Jcross_t__Jpackedf_stride0_elems*3 + 3];

    double* Acw = &Jcross_t__Jpackedcw[Jcross_t__Jpackedcw_stride0_elems*0];
    double* Ccw = &Jcross_t__Jpackedcw[Jcross_t__Jpackedcw_stride0_elems*3];

    // I can compute Jcross_t Jcross from the blocks comprising Jcross_t
    // Jpackedfpcw. From above:
    //
    // Jcross_t Jcross ~
    //   ~ Jcross_t__Jpackedf Dinv drtrfp_drtrrp
    //
    //   ~ [Af Bf] Dinv drtrfp_drtrrp
    //     [Cf Df]
    //
    //   = [Af/SCALE_R Bf/SCALE_T] [dr/dr      0]
    //     [Cf/SCALE_R Df/SCALE_T] [ -skew(t1) I]
    //
    //   = [Af/SCALE_R dr/dr - Bf/SCALE_T skew(t1)    Bf/SCALE_T]
    //     [...                                       Df/SCALE_T]
    const double t0 = rt_ref_frame_packed[3+0] * SCALE_TRANSLATION_FRAME;
    const double t1 = rt_ref_frame_packed[3+1] * SCALE_TRANSLATION_FRAME;
    const double t2 = rt_ref_frame_packed[3+2] * SCALE_TRANSLATION_FRAME;

    // Af <- dr/dr_t sum_outer[:3,:3] + skew_t1 sum_outer[3:,:3]
    {
        mul_gen33_gen33insym66(Af, Jcross_t__Jpackedf_stride0_elems, 1,
                               // transposed, so 1,3 and not 3,1
                               drrfp_drrrp, 1,3,
                               sum_outer_jpackedf_jpackedf, 0, 0,
                               1./SCALE_ROTATION_FRAME);

        // and similar for calobject_warp
        // Acw = drtrfp_drtrrp_t Dinv S; ~
        // -> Acwt = St Dinv drtrfp_drtrrp;
        mul_genNM_genML_accum(// transposed
                              Acw, 1, Jcross_t__Jpackedcw_stride0_elems,

                              2,3,3,
                              // transposed
                              &sum_outer_jpackedf_jpackedcw[0*2 + 0], 1,2,
                              drrfp_drrrp, 3,1,
                              1./SCALE_ROTATION_FRAME);

        for(int j=0; j<3; j++)
        {
            int i;

            i = 0;
            Af[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j)] */
                 /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j)]
                 /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 1;
            Af[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j)]
                 /*skew[i*3 + 1]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j)] */
                 /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 2;
            Af[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j)]
                 /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j)]
                 /*skew[i*3 + 2]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j)] */
                 ) / SCALE_TRANSLATION_FRAME;

            // and similar for calobject_warp
            if(j<2)
            {
                i = 0;
                Acw[i*Jcross_t__Jpackedcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]   + (  0)*sum_outer_jpackedf_jpackedcw[(0+3)*2 + j] */
                     /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jpackedf_jpackedcw[(1+3)*2 + j]
                     /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jpackedf_jpackedcw[(2+3)*2 + j]
                     ) / SCALE_TRANSLATION_FRAME;

                i = 1;
                Acw[i*Jcross_t__Jpackedcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jpackedf_jpackedcw[(0+3)*2 + j]
                     /*skew[i*3 + 1]   + (  0)*sum_outer_jpackedf_jpackedcw[(1+3)*2 + j] */
                     /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jpackedf_jpackedcw[(2+3)*2 + j]
                     ) / SCALE_TRANSLATION_FRAME;

                i = 2;
                Acw[i*Jcross_t__Jpackedcw_stride0_elems + j] +=
                    (
                     /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jpackedf_jpackedcw[(0+3)*2 + j]
                     /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jpackedf_jpackedcw[(1+3)*2 + j]
                     /*skew[i*3 + 2]   + (  0)*sum_outer_jpackedf_jpackedcw[(2+3)*2 + j] */
                     ) / SCALE_TRANSLATION_FRAME;
            }
        }
    }

    // Bf <- dr/dr_t sum_outer[:3,3:] + skew_t1 sum_outer[3:,3:]
    {
        mul_gen33_gen33insym66(Bf, Jcross_t__Jpackedf_stride0_elems, 1,
                               // transposed, so 1,3 and not 3,1
                               drrfp_drrrp, 1,3,
                               sum_outer_jpackedf_jpackedf, 0, 3,
                               1./SCALE_ROTATION_FRAME);

        for(int j=0; j<3; j++)
        {
            int i;

            i = 0;
            Bf[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j+3)] */
                 /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j+3)]
                 /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j+3)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 1;
            Bf[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j+3)]
                 /*skew[i*3 + 1]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j+3)] */
                 /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j+3)]
                 ) / SCALE_TRANSLATION_FRAME;

            i = 2;
            Bf[i*Jcross_t__Jpackedf_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jpackedf_jpackedf[index_sym66(0+3,j+3)]
                 /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jpackedf_jpackedf[index_sym66(1+3,j+3)]
                 /*skew[i*3 + 2]   + (  0)*sum_outer_jpackedf_jpackedf[index_sym66(2+3,j+3)] */
                 ) / SCALE_TRANSLATION_FRAME;
        }
    }

    // Cf <- sum_outer[3:,:3]
    {
        set_gen33_from_gen33insym66(Cf, Jcross_t__Jpackedf_stride0_elems, 1,
                                    sum_outer_jpackedf_jpackedf, 3, 0,
                                    1./SCALE_TRANSLATION_FRAME);

        // and similar for calobject_warp
        for(int i=0; i<3; i++)
            for(int j=0; j<2; j++)
                Ccw[i*Jcross_t__Jpackedcw_stride0_elems + j] +=
                    sum_outer_jpackedf_jpackedcw[(3+i)*2 + j]/SCALE_TRANSLATION_FRAME;

    }

    // Df <- sum_outer[3:,3:]
    {
        set_gen33_from_gen33insym66(Df, Jcross_t__Jpackedf_stride0_elems, 1,
                                    sum_outer_jpackedf_jpackedf, 3, 3,
                                    1./SCALE_TRANSLATION_FRAME);
    }

    // Jcross_t__Jcross is symmetric, so I just compute the upper triangle,
    // and I don't care about the ... block

    // Jcross_t__Jcross[rr] <- Af/SCALE_R dr/dr - Bf/SCALE_T skew(t1)
    {
        mul_gen33_gen33_into33insym66_accum(Jcross_t__Jcross, 0, 0,
                                            Af, Jcross_t__Jpackedf_stride0_elems, 1,
                                            drrfp_drrrp, 3,1,
                                            1./SCALE_ROTATION_FRAME);

        int ivalue = 0;
        for(int i=0; i<3; i++)
        {
            for(int j=i; j<3; j++, ivalue++)
            {
                if(j == 0)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]   + Bf[i*Jcross_t__Jpackedf_stride0_elems+0]*(  0) */
                         /*skew[j + 1*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+1]*( t2)
                         /*skew[j + 2*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+2]*(-t1)
                         ) / SCALE_TRANSLATION_FRAME;

                if(j == 1)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+0]*(-t2)
                         /*skew[j + 1*3]   + Bf[i*Jcross_t__Jpackedf_stride0_elems+1]*(  0) */
                         /*skew[j + 2*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+2]*( t0)
                         ) / SCALE_TRANSLATION_FRAME;

                if(j == 2)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+0]*( t1)
                         /*skew[j + 1*3]*/ + Bf[i*Jcross_t__Jpackedf_stride0_elems+1]*(-t0)
                         /*skew[j + 2*3]   + Bf[i*Jcross_t__Jpackedf_stride0_elems+2]*(  0) */
                         ) / SCALE_TRANSLATION_FRAME;
            }
            ivalue += 3;
        }
    }

    // Jcross_t__Jcross[rt] <- Bf/SCALE_T
    {
        set_33insym66_from_gen33_accum(Jcross_t__Jcross, 0, 3,
                                       Bf, Jcross_t__Jpackedf_stride0_elems, 1,
                                       1./SCALE_TRANSLATION_FRAME);
    }

    // Jcross_t__Jcross[tr] doesn't need to be set: I only have values in
    // the upper triangle

    // Jcross_t__Jcross[tt] <- Df/SCALE_T = sum_outer[3:,3:]/SCALE_T/SCALE_T
    {
        const int N = (6+1)*6/2;
        const int i0 = index_sym66_assume_upper(3,3);
        for(int i=i0; i<N; i++)
            Jcross_t__Jcross[i] +=
                sum_outer_jpackedf_jpackedf[i] /
                (SCALE_TRANSLATION_FRAME*SCALE_TRANSLATION_FRAME);
    }
}

static
void accumulate_point(// output
                      // shape (6,3)
                      double*   Jcross_t__Jpackedp, // THIS one point, many measurements
                      const int Jcross_t__Jpackedp_stride0_elems,
                      // rows are assumed stored densely, so there is
                      // no Jcross_t__Jpackedp_stride1

                      // shape (6,6)
                      double* Jcross_t__Jcross,

                      // input
                      // shape (3,3); symmetric, upper-triangle-only is stored
                      const double* sum_outer_jpackedp_jpackedp,
                      // shape (3,)
                      const double* ppacked)
{
    // sum_outer_jpackedp_jpackedp stores only the upper triangle, in the usual
    // row-major order.
    //
    // Jcross_t__Jpackedp[:, ipoint0:ipoint+3] =
    //   dpref_drrp_t sum_outer_jpackedp_jpackedp /SCALE
    //
    // where dpref_drrp = d(transform(rt_ref_ref*,p*)) / drt_ref_ref*
    //
    // Jcross_t Jcross = sum(outer(jcross, jcross))
    //                 = sum_i( dpref_drrp_t[i]
    //                          sum_outer_jpackedp_jpackedp
    //                          dpref_drrp[i] ) /SCALE/SCALE
    //
    // Jcross has full state, but J_packed has packed state, so I need different
    // number of SCALE factors.
    //
    // dpref_drrp = d(transform(rt_ref_ref*,p)/drt_ref_ref*) where drt_ref_ref*
    // is tiny. Derivation using the Rodrigues rotation formula:
    //
    //     rotate(r,p)
    //     ~ p cos(th) + cross(r/th,p) sin(th) + r/th inner(r/th,p) (1-cos(th))
    //     ~ p + cross(r,p) + r inner(r,p) / (th^2)*(1-cos(th))
    //     ~ p + cross(r,p) + r inner(r,p) / (th^2)*(1- (1-th^2))
    //     ~ p + cross(r,p) + r inner(r,p)
    //     ~ p + linear(r) + quadratic(r)
    //
    //   I assume r is tiny so I only look at the linear term:
    //
    //     d/dr = -skew_symmetric(p)
    //     d/dt = I
    //
    // In the above expressions I have dpref_drrp_t:
    //
    //   dpref_drrp_t = [skew_p]
    //                  [I     ]

    // Jcross_t__Jpackedp output goes into [A]
    //                                     [B]
    //
    // A = dpref_drrp_t[:3,:] sum_outer_jpackedp_jpackedp /SCALE
    // B = dpref_drrp_t[3:,:] sum_outer_jpackedp_jpackedp /SCALE
    //
    // A = skew_p sum_outer_jpackedp_jpackedp /SCALE
    // B =        sum_outer_jpackedp_jpackedp /SCALE
    double* A = &Jcross_t__Jpackedp[Jcross_t__Jpackedp_stride0_elems*0 + 0];
    double* B = &Jcross_t__Jpackedp[Jcross_t__Jpackedp_stride0_elems*3 + 0];

    // A <- skew_p sum_outer_jpackedp_jpackedp /SCALE
    {
        // p = ppacked * SCALE_POSITION_POINT,

        //          [  0 -p2  p1]
        // skew_p = [ p2   0 -p0]
        //          [-p1  p0   0]

        for(int j=0; j<3; j++)
        {
            int i;

            i = 0;
            A[i*Jcross_t__Jpackedp_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]   + (          0)*sum_outer_jpackedp_jpackedp[index_sym33(0,j)] */
                 /*skew[i*3 + 1]*/ + (-ppacked[2])*sum_outer_jpackedp_jpackedp[index_sym33(1,j)]
                 /*skew[i*3 + 2]*/ + ( ppacked[1])*sum_outer_jpackedp_jpackedp[index_sym33(2,j)]
                 );

            i = 1;
            A[i*Jcross_t__Jpackedp_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + ( ppacked[2])*sum_outer_jpackedp_jpackedp[index_sym33(0,j)]
                 /*skew[i*3 + 1]   + (          0)*sum_outer_jpackedp_jpackedp[index_sym33(1,j)] */
                 /*skew[i*3 + 2]*/ + (-ppacked[0])*sum_outer_jpackedp_jpackedp[index_sym33(2,j)]
                 );

            i = 2;
            A[i*Jcross_t__Jpackedp_stride0_elems + j] +=
                (
                 /*skew[i*3 + 0]*/ + (-ppacked[1])*sum_outer_jpackedp_jpackedp[index_sym33(0,j)]
                 /*skew[i*3 + 1]*/ + ( ppacked[0])*sum_outer_jpackedp_jpackedp[index_sym33(1,j)]
                 /*skew[i*3 + 2]   + (          0)*sum_outer_jpackedp_jpackedp[index_sym33(2,j)] */
                 );

        }
    }

    // B <=        sum_outer_jpackedp_jpackedp /SCALE
    {
        for(int j=0; j<3; j++)
            for(int i=0; i<3; i++)
                B[i*Jcross_t__Jpackedp_stride0_elems + j] +=
                    sum_outer_jpackedp_jpackedp[index_sym33(i,j)]
                    / SCALE_POSITION_POINT;
    }

    // Jcross_t__Jcross is symmetric, so I just compute the upper triangle,
    // and I don't care about the ... block

    // Jcross_t__Jcross <- [A] [-skew_p I]
    //                     [B]
    //                  = [-A skew_p    A]
    //                    [-B skew_p    B]

    // Jcross_t__Jcross[00] <- -A skew_p / SCALE
    {
        int ivalue = 0;
        for(int i=0; i<3; i++)
        {
            for(int j=i; j<3; j++, ivalue++)
            {
                if(j == 0)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]   + A[i*Jcross_t__Jpackedp_stride0_elems+0]*(          0) */
                         /*skew[j + 1*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+1]*( ppacked[2])
                         /*skew[j + 2*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+2]*(-ppacked[1])
                         );

                if(j == 1)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+0]*(-ppacked[2])
                         /*skew[j + 1*3]   + A[i*Jcross_t__Jpackedp_stride0_elems+1]*(          0) */
                         /*skew[j + 2*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+2]*( ppacked[0])
                         );

                if(j == 2)
                    Jcross_t__Jcross[ivalue] -=
                        (
                         /*skew[j + 0*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+0]*( ppacked[1])
                         /*skew[j + 1*3]*/ + A[i*Jcross_t__Jpackedp_stride0_elems+1]*(-ppacked[0])
                         /*skew[j + 2*3]   + A[i*Jcross_t__Jpackedp_stride0_elems+2]*(          0) */
                         );
            }
            ivalue += 3;
        }
    }

    // Jcross_t__Jcross[01] <- A/SCALE
    {
        set_33insym66_from_gen33_accum(Jcross_t__Jcross, 0, 3,
                                       A, Jcross_t__Jpackedp_stride0_elems, 1,
                                       1./SCALE_POSITION_POINT);
    }

    // Jcross_t__Jcross[10] doesn't need to be set: I only have values in
    // the upper triangle

    // Jcross_t__Jcross[11] <- B / SCALE
    {
        const int N = (6+1)*6/2;
        const int i0 = index_sym66_assume_upper(3,3);
        for(int i=i0; i<N; i++)
            Jcross_t__Jcross[i] +=
                sum_outer_jpackedp_jpackedp[i-i0] /
                (SCALE_POSITION_POINT*SCALE_POSITION_POINT);
    }
}

static int get_Nstate_intrinsics_in_jacobian_row(const int* Jrowptr,
                                                 const int* Jcolidx,
                                                 const int state_index_intrinsics0,
                                                 const int Nstate_intrinsics)
{
    // I linearly search through the first row of the jacobian to count the
    // number of active variables in the jacobian. I'm assuming that every
    // jacobian row has exactly the same number of intrinsics variables that
    // affect it
    const int imeas = 0;
    int ival = Jrowptr[imeas];

    while(true)
    {
        const int state_index = Jcolidx[ival];

        if( ival < Jrowptr[imeas+1] &&
            state_index_intrinsics0 <= state_index &&
            state_index < state_index_intrinsics0 + Nstate_intrinsics )
        {
            ival++;
            continue;
        }
        return ival;
    }
}

// LAPACK prototypes for a packed cholesky factorization and a linear solve
// using that factorization, respectively
int dpptrf_(char* uplo, int* n, double* ap,
            int* info);
int dpptrs_(char* uplo, int* n, int* nrhs,
            double* ap, double* b, int* ldb, int* info);

bool _mrcal_drt_ref_refperturbed__dbpacked(// output
                                          // Shape (6,Nstate_frames)
                                          double* Kpackedf,
                                          int Kpackedf_stride0, // in bytes. <= 0 means "contiguous"
                                          int Kpackedf_stride1, // in bytes. <= 0 means "contiguous"

                                          // Shape (6,Nstate_points)
                                          double* Kpackedp,
                                          int Kpackedp_stride0, // in bytes. <= 0 means "contiguous"
                                          int Kpackedp_stride1, // in bytes. <= 0 means "contiguous"

                                          // Shape (6,Nstate_calobject_warp)
                                          double* Kpackedcw,
                                          int Kpackedcw_stride0, // in bytes. <= 0 means "contiguous"
                                          int Kpackedcw_stride1, // in bytes. <= 0 means "contiguous"

                                          // inputs
                                          // stuff that describes this solve
                                          const double* b_packed,
                                          // used only to confirm that the user passed-in the buffer they
                                          // should have passed-in. The size must match exactly
                                          int buffer_size_b_packed,

                                          // The unitless (packed) Jacobian,
                                          // used by the internal optimization
                                          // routines cholmod_analyze() and
                                          // cholmod_factorize() require
                                          // non-const
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

    const int*    Jrowptr = (int*)   Jt->p;
    const int*    Jcolidx = (int*)   Jt->i;
    const double* Jval    = (double*)Jt->x;

    const int Nmeas_boards =
        mrcal_num_measurements_boards(Nobservations_board,
                                      calibration_object_width_n,
                                      calibration_object_height_n);
    const int Nmeas_points =
        mrcal_num_measurements_points(Nobservations_point);

    const int Nmeas_obs = Nmeas_boards + Nmeas_points;

    const int state_index_intrinsics0 =
        mrcal_state_index_intrinsics(0,
                                     Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes,
                                     Npoints, Npoints_fixed, Nobservations_board,
                                     problem_selections,
                                     lensmodel);
    const int state_index_extrinsics0 =
        mrcal_state_index_extrinsics(0,
                                     Ncameras_intrinsics, Ncameras_extrinsics,
                                     Nframes,
                                     Npoints, Npoints_fixed, Nobservations_board,
                                     problem_selections,
                                     lensmodel);
    const int state_index_frames0 =
        mrcal_state_index_frames(0,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
    const int state_index_points0 =
        mrcal_state_index_points(0,
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

    const int Nstate_intrinsics =
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_selections,
                                    lensmodel);
    const int Nstate_intrinsics_in_jacobian_row =
        get_Nstate_intrinsics_in_jacobian_row(Jrowptr,
                                              Jcolidx,
                                              state_index_intrinsics0,
                                              Nstate_intrinsics);
    const int Nstate_extrinsics =
        mrcal_num_states_extrinsics(Ncameras_extrinsics,
                                    problem_selections);
    const int Nstate_frames =
        mrcal_num_states_frames(Nframes,
                                problem_selections);
    const int Nstate_points =
        mrcal_num_states_points(Npoints,Npoints_fixed,
                                problem_selections);
    const int Nstate_calobject_warp =
        mrcal_num_states_calobject_warp(problem_selections,
                                        Nobservations_board);

#warning check all Nstate_ and state_index_ references; some of those could be invalid (if some variables are locked for instance). Figure out what makes sense and what we should BARF() against

#warning Do I need this? Where do I assume it?
    if(state_index_frames0 >= 0 &&
       state_index_calobject_warp0 >= 0 &&
       !(state_index_calobject_warp0 == state_index_frames0 + Nstate_frames))
    {
        MSG("I assume that the calobject_warp state variables follow the frame state variables immediately");
        return false;
    }

#warning Do I need this? Where do I assume it?
    if(state_index_calobject_warp0 >= 0 &&
       !(Nstate_calobject_warp == 2))
    {
        MSG("I assume that the calobject_warp has exactly 2 state variables");
        return false;
    }

    if( buffer_size_b_packed != Nstate*(int)sizeof(double) )
    {
        MSG("The buffer b_packed has the wrong size. Needed exactly %d bytes, but got %d bytes",
            Nstate*(int)sizeof(double),buffer_size_b_packed);
        return false;
    }

    if(Nstate != (int)Jt->nrow)
    {
        MSG("Inconsistent inputs. I have Nstate=%d, but Jt->nrow=%d. Giving up",
            Nstate, (int)Jt->nrow);
        return false;
    }

    if(state_index_frames0 < 0 &&
       state_index_points0 < 0)
    {
        MSG("Neither board poses nor points are being optimized. Cannot compute uncertainty if we're not optimizing any observations");
        return false;
    }

#define INIT_ARRAY(Kpacked, N)                                          \
    init_stride_2D(Kpacked, 6, N);                                      \
    const int Kpacked ## _stride0_elems = Kpacked ## _stride0 / sizeof(double); \
    if(Kpacked != NULL)                                                 \
    {                                                                   \
        if(Kpacked ## _stride0_elems*(int)sizeof(double) != Kpacked ## _stride0) \
        {                                                               \
            MSG("Currently the implementation assumes that " #Kpacked "_stride0 is a multiple of sizeof(double): all elements of Kpacked are aligned. Got Kpacked ## _stride0 = %d", \
                Kpacked ## _stride0);                                   \
            return false;                                               \
        }                                                               \
                                                                        \
        if(Kpacked ## _stride1 == sizeof(double))                       \
        {                                                               \
            /* each row is stored densely */                            \
            if(Kpacked ## _stride0 == (int)sizeof(double)*N)            \
                /* each column is stored densely as well. I can memset() the whole */ \
                /* block of memory */                                   \
                memset(Kpacked, 0, 6*N*sizeof(double));                 \
            else                                                        \
                for(int i=0; i<6; i++)                                  \
                    memset(&Kpacked[i*Kpacked ## _stride0_elems], 0, N*sizeof(double)); \
        }                                                               \
        else                                                            \
        {                                                               \
            MSG("Currently the implementation assumes that Kpacked has densely-stored rows: " #Kpacked "_stride1 must be sizeof(double). Instead I got Kpacked ## _stride1 = %d", \
                Kpacked ## _stride1);                                   \
            return false;                                               \
        }                                                               \
    }

    INIT_ARRAY(Kpackedf,  Nstate_frames);
    INIT_ARRAY(Kpackedp,  Nstate_points);
    INIT_ARRAY(Kpackedcw, Nstate_calobject_warp);


    // sum(outer(dx[i]/drt_cam_ref,dx[i]/drt_cam_ref)) for this observed object.
    // I sum over all the observations. Uses PACKED gradients. Only the upper
    // triangle is stored, in the usual row-major order
    double sum_outer_jpackede_jpackede[(6+1)*6/2] = {};

    // sum(outer(dx[i]/drt_cam_ref, dx[i]/drt_ref_frame)) for this observed
    // object. Uses PACKED gradients. Stored densely, since it isn't symmetric.
    // Shape (6,6)
    double sum_outer_jpackede_jpackedf[6*6] = {};

    // sum(outer(dx[i]/drt_cam_ref, dp[i)) for this observed object. Uses PACKED
    // gradients. Stored densely, since it isn't symmetric. Shape (6,3)
    double sum_outer_jpackede_jpackedp[6*3] = {};

    // sum(outer(dx[i]/drt_cam_ref, dx[i]/dcalobject_warp)) for this observed
    // object. Uses PACKED gradients. Stored densely, since it isn't symmetric.
    // Shape (6,2)
    double sum_outer_jpackede_jpackedcw[6*2] = {};

    // sum(outer(dx[i]/drt_ref_frame,dx[i]/drt_ref_frame)) for this frame. I sum
    // over all the observations. Uses PACKED gradients. Only the upper triangle
    // is stored, in the usual row-major order
    double sum_outer_jpackedf_jpackedf[(6+1)*6/2] = {};

    // sum(outer(dx[i]/dpoint,dx[i]/dpoint)) for this point. I sum over all the
    // observations. Uses PACKED gradients. Only the upper triangle is stored,
    // in the usual row-major order
    double sum_outer_jpackedp_jpackedp[(3+1)*3/2] = {};

    // sum(outer(dx[i]/rt_ref_frame, /dcalobject_warp)) for this frame. Uses
    // PACKED gradients. Stored densely, since it isn't symmetric. Shape (6,2)
    double sum_outer_jpackedf_jpackedcw[6*2] = {};

    // There's no sum_outer_jpackedf_jpackede. if(cross_reprojection_ccp), we
    // would use the Jcrossf path ONLY if no extrinsics were available: Je does
    // not exist in that case, and I do NOT need to compute
    // sum_outer_jpackedf_jpackede. if(!cross_reprojection_ccp), then I don't
    // NEED sum_outer_jpackedf_jpackede at all




    double Jcross_t__Jcross[(6+1)*6/2] = {};

    // The current chunk we're processing. We aggregate the sum_outer_...
    // variables for each chunk (set of measurements with identical
    // cam,frame,point). For instance, all 2*N*N measurements for a single
    // chessboard observation come from the same chunk
    int state_index_accumulating_frame = -1;
    int state_index_accumulating_point = -1;


    for(int imeas=0; imeas<Nmeas_obs; imeas++)
    {
        // For each measurement, we will have gradients for some set of variables:
        int ival_intrinsics     = -1;
        int ival_extrinsics     = -1;
        int ival_frames         = -1;
        int ival_points         = -1;
        int ival_calobject_warp = -1;

        // I map out this row
        {
            int ival;
            do
            {
                ival = Jrowptr[imeas];

                int state_index = Jcolidx[ival];

                if( state_index_intrinsics0 <= state_index &&
                    state_index < state_index_intrinsics0 + Nstate_intrinsics )
                {
                    ival_intrinsics = ival;
                    ival += Nstate_intrinsics_in_jacobian_row;
                    if(!(ival < Jrowptr[imeas+1]))
                        break;
                    state_index = Jcolidx[ival];
                }

                if( state_index_extrinsics0 <= state_index &&
                    state_index < state_index_extrinsics0 + Nstate_extrinsics )
                {
                    ival_extrinsics = ival;
                    ival += 6;
                    if(!(ival < Jrowptr[imeas+1]))
                        break;
                    state_index = Jcolidx[ival];
                }

                if( state_index_frames0 <= state_index &&
                    state_index < state_index_frames0 + Nstate_frames )
                {
                    ival_frames = ival;
                    ival += 6;
                    if(!(ival < Jrowptr[imeas+1]))
                        break;
                    state_index = Jcolidx[ival];
                }

                if( state_index_points0 <= state_index &&
                    state_index < state_index_points0 + Nstate_points )
                {
                    ival_points = ival;
                    ival += 3;
                    if(!(ival < Jrowptr[imeas+1]))
                        break;
                    state_index = Jcolidx[ival];
                }

                if( state_index_calobject_warp0 <= state_index &&
                    state_index < state_index_calobject_warp0 + Nstate_calobject_warp )
                {
                    ival_calobject_warp = ival;
                    ival += Nstate_calobject_warp;
                    if(!(ival < Jrowptr[imeas+1]))
                        break;
                    state_index = Jcolidx[ival];
                }
            } while(false);

            if(ival != Jrowptr[imeas+1])
            {
                MSG("ERROR: unexpected jacobian structure");
                return false;
            }

            if(ival_frames >= 0 && ival_points >= 0)
            {
                MSG("ERROR: both points and frames exist in this measuremnet. This is not supported");
                return false;
            }
            if(ival_frames < 0 && ival_calobject_warp >= 0)
            {
                MSG("ERROR: this measurement has no frames but DOES use a calobject_warp. That makes no sense");
                return false;
            }
            if(ival_frames<0 && ival_points<0)
            {
                MSG("ERROR: neither frames nor points are used in this measurement; do I support that?");
                return false;
            }
        }

        if(ival_calobject_warp < 0 &&
           Kpackedcw != NULL)
        {
            MSG("Unexpected jacobian structure. There's no calobject_warp gradient in measurement %d, but the user asked for it",
                imeas);
            return false;
        }
        if(ival_calobject_warp >= 0 &&
           Kpackedcw == NULL)
        {
            MSG("Unexpected jacobian structure. There's a calobject_warp gradient in measurement %d, but the user didn't ask for it",
                imeas);
            return false;
        }


        // Finish up existing accumulations
        if(ival_frames >= 0 &&
           Jcolidx[ival_frames] < state_index_accumulating_frame)
        {
            MSG("Unexpected jacobian structure. I'm assuming non-decreasing frame references. accumulate_frame() re-uses chunks of Kpackedf; it assumes that once the chunk is computed, it is DONE, and never revisited. Non-monotonic frame indices break that");
            return false;
        }
        if(state_index_accumulating_frame >= 0 &&
           ival_frames >= 0 &&
           state_index_accumulating_frame != Jcolidx[ival_frames])
        {
            // New measurement is different. Accumulate.
            accumulate_frame( // output
                             &Kpackedf[state_index_accumulating_frame-state_index_frames0],
                             Kpackedf_stride0_elems,
                             Kpackedcw,
                             Kpackedcw_stride0_elems,
                             Jcross_t__Jcross,

                             // input
                             sum_outer_jpackedf_jpackedf,
                             sum_outer_jpackedf_jpackedcw,
                             &b_packed[state_index_accumulating_frame]);
            memset(sum_outer_jpackedf_jpackedf,  0, sizeof(sum_outer_jpackedf_jpackedf));
            memset(sum_outer_jpackedf_jpackedcw, 0, sizeof(sum_outer_jpackedf_jpackedcw));
        }
        state_index_accumulating_frame = (ival_frames < 0) ? -1 : Jcolidx[ival_frames];


        if(ival_points >= 0 &&
           Jcolidx[ival_points] < state_index_accumulating_point)
        {
            MSG("Unexpected jacobian structure. I'm assuming non-decreasing point references. accumulate_point() re-uses chunks of Kpackedp; it assumes that once the chunk is computed, it is DONE, and never revisited. Non-monotonic point indices break that");
            return false;
        }
        if(state_index_accumulating_point >= 0 &&
           ival_points >= 0 &&
           state_index_accumulating_point != Jcolidx[ival_points])
        {
            // New measurement is different. Accumulate.
            accumulate_point( // output
                             &Kpackedp[state_index_accumulating_point-state_index_points0],
                             Kpackedp_stride0_elems,
                             Jcross_t__Jcross,

                             // input
                             sum_outer_jpackedp_jpackedp,
                             &b_packed[state_index_accumulating_point]);
            memset(sum_outer_jpackedp_jpackedp,  0, sizeof(sum_outer_jpackedp_jpackedp));
        }
        state_index_accumulating_point = (ival_points < 0) ? -1 : Jcolidx[ival_points];


        if(ival_frames >= 0)
        {
            // We're looking at SOME rt_ref_frame gradient. I expect 6 values
            // for the rt_ref_frame gradient followed by 2 values for the
            // calobject_warp gradient
            //
            // Consecutive chunks of Nw*Nh*2 measurements will represent the
            // same board pose, and the same rt_ref_frame

            // I have dx/drt_ref_frame for this frame. This is 6 numbers
            const double* dx_drt_ref_frame_packed = &Jval[ival_frames];

            // sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) into sum_outer_jpackedf_jpackedf

            // This is used to compute Jcross_t J_packedfpcw and Jcross_t
            // Jcross. This result is used in accumulate_frame()
            //
            // Uses PACKED gradients. Only the upper triangle is stored, in
            // the usual row-major order
            for(int i=0, ivalue=0; i<6; i++)
                for(int j=i; j<6; j++, ivalue++)
                    sum_outer_jpackedf_jpackedf[ivalue] +=
                        dx_drt_ref_frame_packed[i]*dx_drt_ref_frame_packed[j];


            if(!(ival_calobject_warp >= 0))
                continue; // next measurement
            const double* dx_dcalobject_warp_packed = &Jval[ival_calobject_warp];
            // Similar to the above, but this isn't symmetric, so I store it
            // densely
            int ivalue = 0;
            for(int i=0; i<6; i++)
                for(int j=0; j<2; j++, ivalue++)
                    sum_outer_jpackedf_jpackedcw[ivalue] +=
                        dx_drt_ref_frame_packed[i]*
                        dx_dcalobject_warp_packed[j];
        }

        // if(point gradient). If these don't exist in this problem,
        // Nstate_points==0, and this will always be false
        if(ival_points >= 0)
        {
            // We're looking at SOME point gradient: 3 values

            // I have dx/dpoint for this point. This is 3 numbers
            const double* dx_dpoint_packed = &Jval[ival_points];

            // sum(outer(dx/dpoint,dx/dpoint)) into sum_outer_jpackedp_jpackedp

            // This is used to compute Jcross_t J_packedfpcw and Jcross_t
            // Jcross. This result is used in accumulate_point()
            //
            // Uses PACKED gradients. Only the upper triangle is stored, in
            // the usual row-major order
            for(int i=0, ivalue=0; i<3; i++)
                for(int j=i; j<3; j++, ivalue++)
                    sum_outer_jpackedp_jpackedp[ivalue] +=
                        dx_dpoint_packed[i]*dx_dpoint_packed[j];
        }
    }

    // Finish up existing accumulations
    if(state_index_accumulating_frame >= 0)
        accumulate_frame( // output
                         &Kpackedf[state_index_accumulating_frame-state_index_frames0],
                         Kpackedf_stride0_elems,
                         Kpackedcw,
                         Kpackedcw_stride0_elems,
                         Jcross_t__Jcross,

                         // input
                         sum_outer_jpackedf_jpackedf,
                         sum_outer_jpackedf_jpackedcw,
                         &b_packed[state_index_accumulating_frame]);

    if(state_index_accumulating_point >= 0)
        accumulate_point( // output
                         &Kpackedp[state_index_accumulating_point-state_index_points0],
                         Kpackedp_stride0_elems,
                         Jcross_t__Jcross,

                         // input
                         sum_outer_jpackedp_jpackedp,
                         &b_packed[state_index_accumulating_point]);



    // I now have filled Jcross_t__Jcross and Kpacked. I can
    // compute
    //
    //   inv(Jcross_t Jcross) Jcross_t J_fpcw
    //
    // I actually compute the transpose:
    //
    //   (Jcross_t J_fpcw)t inv(Jcross_t Jcross)
    //
    // in-place: input and output both use the Kpacked array

    int info;
    dpptrf_(&(char){'L'}, &(int){6}, Jcross_t__Jcross,
            &info);
    if(info != 0)
    {
        MSG("Singular Jcross_t Jcross!");
        return false;
    }

#define FINALIZE(Kpacked, N)                                    \
    if(Kpacked)                                                         \
    {                                                                   \
        /* lapack cannot handle non-contiguous rhs vectors, so I make a copy first */ \
        double x[6];                                                    \
        for(int j=0; j<N; j++)                                          \
        {                                                               \
            for(int i=0; i<6; i++)                                      \
                x[i] = Kpacked[Kpacked ## _stride0_elems*i + j];        \
                                                                        \
            dpptrs_(&(char){'L'}, &(int){6}, &(int){1},                 \
                    Jcross_t__Jcross,                                   \
                    x, &(int){6}, &info);                               \
            if(info != 0)                                               \
            {                                                           \
                MSG("dpptrs() failed. This shouldn't happen");          \
                return false;                                           \
            }                                                           \
                                                                        \
            for(int i=0; i<6; i++)                                      \
                Kpacked[Kpacked ## _stride0_elems*i + j] = -x[i];       \
        }                                                               \
    }

    FINALIZE(Kpackedf,  Nstate_frames);
    FINALIZE(Kpackedp,  Nstate_points);
    FINALIZE(Kpackedcw, Nstate_calobject_warp);
#undef FINALIZE


    return true;
}
