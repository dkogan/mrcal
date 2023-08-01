// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include "autodiff.hh"
#include "strides.h"

extern "C" {
#include "poseutils.h"
}

template<int N>
static void
rotate_point_r_core(// output
                    val_withgrad_t<N>* x_outg,

                    // inputs
                    const val_withgrad_t<N>* rg,
                    const val_withgrad_t<N>* x_ing,
                    bool inverted)
{
    // Rodrigues rotation formula:
    //   xrot = x cos(th) + cross(axis, x) sin(th) + axis axist x (1 - cos(th))
    //
    // I have r = axis*th -> th = norm(r) ->
    //   xrot = x cos(th) + cross(r, x) sin(th)/th + r rt x (1 - cos(th)) / (th*th)

    // an inversion would flip the sign on:
    // - rg
    // - cross
    // - inner
    // But inner is always multiplied by rg, making the sign irrelevant. So an
    // inversion only flips the sign on the cross
    double sign = inverted ? -1.0 : 1.0;

    const val_withgrad_t<N> th2 =
        rg[0]*rg[0] +
        rg[1]*rg[1] +
        rg[2]*rg[2];
    const val_withgrad_t<N> cross[3] =
        {
            (rg[1]*x_ing[2] - rg[2]*x_ing[1])*sign,
            (rg[2]*x_ing[0] - rg[0]*x_ing[2])*sign,
            (rg[0]*x_ing[1] - rg[1]*x_ing[0])*sign
        };
    const val_withgrad_t<N> inner =
        rg[0]*x_ing[0] +
        rg[1]*x_ing[1] +
        rg[2]*x_ing[2];

    if(th2.x < 1e-10)
    {
        // Small rotation. I don't want to divide by 0, so I take the limit
        //   lim(th->0, xrot) =
        //     = x + cross(r, x) + r rt x lim(th->0, (1 - cos(th)) / (th*th))
        //     = x + cross(r, x) + r rt x lim(th->0, sin(th) / (2*th))
        //     = x + cross(r, x) + r rt x/2
        for(int i=0; i<3; i++)
            x_outg[i] =
                x_ing[i] +
                cross[i] +
                rg[i]*inner / 2.;
    }
    else
    {
        // Non-small rotation. This is the normal path

        const val_withgrad_t<N> th = th2.sqrt();
        const vec_withgrad_t<N, 2> sc = th.sincos();

        for(int i=0; i<3; i++)
            x_outg[i] =
                x_ing[i]*sc.v[1] +
                cross[i] * sc.v[0]/th +
                rg[i] * inner * (val_withgrad_t<N>(1.) - sc.v[1]) / th2;
    }
}

template<int N>
static void
r_from_R_core(// output
              val_withgrad_t<N>* rg,

              // inputs
              const val_withgrad_t<N>* Rg)
{
    val_withgrad_t<N> tr = Rg[0] + Rg[4] + Rg[8];
    val_withgrad_t<N> u[3] =
        {
            Rg[2*3 + 1] - Rg[1*3 + 2],
            Rg[0*3 + 2] - Rg[2*3 + 0],
            Rg[1*3 + 0] - Rg[0*3 + 1]
        };

    val_withgrad_t<N> costh = (tr - 1.) / 2.;

    if( (fabs(u[0].x) > 1e-10 ||
         fabs(u[1].x) > 1e-10 ||
         fabs(u[2].x) > 1e-10) &&
        fabs(costh.x)   < (1. - 1e-10) )
    {
        // normal path
        val_withgrad_t<N> th = costh.acos();
        val_withgrad_t<N> mag_axis_recip =
            val_withgrad_t<N>(1.) /
            ((u[0]*u[0] +
              u[1]*u[1] +
              u[2]*u[2]).sqrt());
        for(int i=0; i<3; i++)
            rg[i] = u[i] * mag_axis_recip * th;
    }
    else
    {
        // small th. Can't divide by it. But I can look at the limit.
        //
        // u / (2 sinth)*th = u/2 *th/sinth ~ u/2
        for(int i=0; i<3; i++)
            rg[i] = u[i] / 2.;
    }
}

extern "C"
void mrcal_rotate_point_r_full( // output
                               double* x_out,      // (3,) array
                               int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                               double* J_r,        // (3,3) array. May be NULL
                               int J_r_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_r_stride1,    // in bytes. <= 0 means "contiguous"
                               double* J_x,        // (3,3) array. May be NULL
                               int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                               // input
                               const double* r,    // (3,) array. May be NULL
                               int r_stride0,      // in bytes. <= 0 means "contiguous"
                               const double* x_in, // (3,) array. May be NULL
                               int x_in_stride0,   // in bytes. <= 0 means "contiguous"

                               bool inverted       // if true, I apply a
                                                   // rotation in the opposite
                                                   // direction. J_r corresponds
                                                   // to the input r
                                )
{
    init_stride_1D(x_out, 3);
    init_stride_2D(J_r,   3,3);
    init_stride_2D(J_x,   3,3);
    init_stride_1D(r,     3);
    init_stride_1D(x_in,  3);

    if(J_r == NULL && J_x == NULL)
    {
        vec_withgrad_t<0, 3> rg   (r,    -1, r_stride0);
        vec_withgrad_t<0, 3> x_ing(x_in, -1, x_in_stride0);
        vec_withgrad_t<0, 3> x_outg;
        rotate_point_r_core<0>(x_outg.v,
                               rg.v, x_ing.v,
                               inverted);
        x_outg.extract_value(x_out, x_out_stride0);
    }
    else if(J_r != NULL && J_x == NULL)
    {
        vec_withgrad_t<3, 3> rg   (r,     0, r_stride0);
        vec_withgrad_t<3, 3> x_ing(x_in, -1, x_in_stride0);
        vec_withgrad_t<3, 3> x_outg;
        rotate_point_r_core<3>(x_outg.v,
                               rg.v, x_ing.v,
                               inverted);
        x_outg.extract_value(x_out, x_out_stride0);
        x_outg.extract_grad (J_r, 0, 3, 0, J_r_stride0, J_r_stride1);
    }
    else if(J_r == NULL && J_x != NULL)
    {
        vec_withgrad_t<3, 3> rg   (r,   -1, r_stride0);
        vec_withgrad_t<3, 3> x_ing(x_in, 0, x_in_stride0);
        vec_withgrad_t<3, 3> x_outg;
        rotate_point_r_core<3>(x_outg.v,
                               rg.v, x_ing.v,
                               inverted);
        x_outg.extract_value(x_out, x_out_stride0);
        x_outg.extract_grad (J_x, 0, 3, 0, J_x_stride0,J_x_stride1);
    }
    else
    {
        vec_withgrad_t<6, 3> rg   (r,    0, r_stride0);
        vec_withgrad_t<6, 3> x_ing(x_in, 3, x_in_stride0);
        vec_withgrad_t<6, 3> x_outg;
        rotate_point_r_core<6>(x_outg.v,
                               rg.v, x_ing.v,
                               inverted);
        x_outg.extract_value(x_out, x_out_stride0);
        x_outg.extract_grad (J_r, 0, 3, 0, J_r_stride0, J_r_stride1);
        x_outg.extract_grad (J_x, 3, 3, 0, J_x_stride0, J_x_stride1);
    }
}

extern "C"
void mrcal_transform_point_rt_full( // output
                                   double* x_out,      // (3,) array
                                   int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                   double* J_rt,       // (3,6) array. May be NULL
                                   int J_rt_stride0,   // in bytes. <= 0 means "contiguous"
                                   int J_rt_stride1,   // in bytes. <= 0 means "contiguous"
                                   double* J_x,        // (3,3) array. May be NULL
                                   int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                   int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                   // input
                                   const double* rt,   // (6,) array. May be NULL
                                   int rt_stride0,     // in bytes. <= 0 means "contiguous"
                                   const double* x_in, // (3,) array. May be NULL
                                   int x_in_stride0,   // in bytes. <= 0 means "contiguous"

                                   bool inverted       // if true, I apply the
                                                       // transformation in the
                                                       // opposite direction.
                                                       // J_rt corresponds to
                                                       // the input rt
                                    )
{
    if(!inverted)
    {
        init_stride_1D(x_out, 3);
        init_stride_2D(J_rt,  3,6);
        // init_stride_2D(J_x,   3,3 );
        init_stride_1D(rt,    6 );
        // init_stride_1D(x_in,  3 );

        // to make in-place operations work
        double t[3] = { P1(rt, 3),
                        P1(rt, 4),
                        P1(rt, 5) };
        // I want rotate(x) + t
        // First rotate(x)
        mrcal_rotate_point_r_full(x_out, x_out_stride0,
                                  J_rt,  J_rt_stride0,  J_rt_stride1,
                                  J_x,   J_x_stride0,   J_x_stride1,
                                  rt,    rt_stride0,
                                  x_in,  x_in_stride0, false);

        // And now +t. The J_r, J_x gradients are unaffected. J_t is identity
        for(int i=0; i<3; i++)
            P1(x_out,i) += t[i];
        if(J_rt)
            mrcal_identity_R_full(&P2(J_rt,0,3), J_rt_stride0, J_rt_stride1);
    }
    else
    {
        // I use the special-case rx_minus_rt() to efficiently rotate both x and
        // t by the same r
        init_stride_1D(x_out, 3);
        init_stride_2D(J_rt,  3,6);
        init_stride_2D(J_x,   3,3 );
        init_stride_1D(rt,    6 );
        init_stride_1D(x_in,  3 );

        if(J_rt == NULL && J_x == NULL)
        {
            vec_withgrad_t<0, 3> x_minus_t(x_in,      -1, x_in_stride0);
            x_minus_t -= vec_withgrad_t<0, 3>(&P1(rt,3), -1, rt_stride0);

            vec_withgrad_t<0, 3> rg   (&rt[0],    -1, rt_stride0);
            vec_withgrad_t<0, 3> x_outg;
            rotate_point_r_core<0>(x_outg.v,
                                   rg.v, x_minus_t.v,
                                   true);
            x_outg.extract_value(x_out, x_out_stride0);
        }
        else if(J_rt != NULL && J_x == NULL)
        {
            vec_withgrad_t<6, 3> x_minus_t(x_in,     -1, x_in_stride0);
            x_minus_t -= vec_withgrad_t<6, 3>(&P1(rt,3), 3, rt_stride0);

            vec_withgrad_t<6, 3> rg   (&rt[0],    0, rt_stride0);
            vec_withgrad_t<6, 3> x_outg;
            rotate_point_r_core<6>(x_outg.v,
                                   rg.v, x_minus_t.v,
                                   true);
            x_outg.extract_value(x_out, x_out_stride0);
            x_outg.extract_grad (J_rt,          0, 3, 0, J_rt_stride0, J_rt_stride1);
            x_outg.extract_grad (&P2(J_rt,0,3), 3, 3, 0, J_rt_stride0, J_rt_stride1);
        }
        else if(J_rt == NULL && J_x != NULL)
        {
            vec_withgrad_t<3, 3> x_minus_t(x_in,      0, x_in_stride0);
            x_minus_t -= vec_withgrad_t<3, 3>(&P1(rt,3),-1, rt_stride0);

            vec_withgrad_t<3, 3> rg   (&rt[0],   -1, rt_stride0);
            vec_withgrad_t<3, 3> x_outg;
            rotate_point_r_core<3>(x_outg.v,
                                   rg.v, x_minus_t.v,
                                   true);
            x_outg.extract_value(x_out, x_out_stride0);
            x_outg.extract_grad (J_x, 0, 3, 0, J_x_stride0,  J_x_stride1);
        }
        else
        {
            vec_withgrad_t<9, 3> x_minus_t(x_in,      3, x_in_stride0);
            x_minus_t -= vec_withgrad_t<9, 3>(&P1(rt,3), 6, rt_stride0);

            vec_withgrad_t<9, 3> rg   (&rt[0],    0, rt_stride0);
            vec_withgrad_t<9, 3> x_outg;


            rotate_point_r_core<9>(x_outg.v,
                                   rg.v, x_minus_t.v,
                                   true);

            x_outg.extract_value(x_out, x_out_stride0);
            x_outg.extract_grad (J_rt,          0, 3, 0, J_rt_stride0, J_rt_stride1);
            x_outg.extract_grad (&P2(J_rt,0,3), 6, 3, 0, J_rt_stride0, J_rt_stride1);
            x_outg.extract_grad (J_x,           3, 3, 0, J_x_stride0,  J_x_stride1);
        }
    }
}


extern "C"
void mrcal_r_from_R_full( // output
                         double* r,       // (3,) vector
                         int r_stride0,   // in bytes. <= 0 means "contiguous"
                         double* J,       // (3,3,3) array. Gradient. May be NULL
                         int J_stride0,   // in bytes. <= 0 means "contiguous"
                         int J_stride1,   // in bytes. <= 0 means "contiguous"
                         int J_stride2,   // in bytes. <= 0 means "contiguous"

                         // input
                         const double* R, // (3,3) array
                         int R_stride0,   // in bytes. <= 0 means "contiguous"
                         int R_stride1    // in bytes. <= 0 means "contiguous"
                          )
{
    init_stride_1D(r, 3);
    init_stride_3D(J, 3,3,3);
    init_stride_2D(R, 3,3);

    if(J == NULL)
    {
        vec_withgrad_t<0, 3> rg;
        vec_withgrad_t<0, 9> Rg;
        Rg.init_vars(&P2(R,0,0), 0,3, -1, R_stride1);
        Rg.init_vars(&P2(R,1,0), 3,3, -1, R_stride1);
        Rg.init_vars(&P2(R,2,0), 6,3, -1, R_stride1);

        r_from_R_core<0>(rg.v, Rg.v);
        rg.extract_value(r, r_stride0);
    }
    else
    {
        vec_withgrad_t<9, 3> rg;
        vec_withgrad_t<9, 9> Rg;
        Rg.init_vars(&P2(R,0,0), 0,3, 0, R_stride1);
        Rg.init_vars(&P2(R,1,0), 3,3, 3, R_stride1);
        Rg.init_vars(&P2(R,2,0), 6,3, 6, R_stride1);

        r_from_R_core<9>(rg.v, Rg.v);
        rg.extract_value(r, r_stride0);

        // J is dr/dR of shape (3,3,3). autodiff.h has a gradient of shape
        // (3,9): the /dR part is flattened. I pull it out in 3 chunks that scan
        // the middle dimension. So I fill in J[:,0,:] then J[:,1,:] then J[:,2,:]
        rg.extract_grad(&P3(J,0,0,0), 0,3, 0,J_stride0,J_stride2,3);
        rg.extract_grad(&P3(J,0,1,0), 3,3, 0,J_stride0,J_stride2,3);
        rg.extract_grad(&P3(J,0,2,0), 6,3, 0,J_stride0,J_stride2,3);
    }
}

template<int N>
static void
compose_r_core(// output
               vec_withgrad_t<N, 3>* r,

               // inputs
               const vec_withgrad_t<N, 3>* r0,
               const vec_withgrad_t<N, 3>* r1)
{
    // Described here:
    //
    //     Altmann, Simon L. "Hamilton, Rodrigues, and the Quaternion Scandal."
    //     Mathematics Magazine, vol. 62, no. 5, 1989, pp. 291–308
    //
    // Available here:
    //
    //     https://www.jstor.org/stable/2689481
    //
    // I use Equations (19) and (20) on page 302 of this paper. These equations say
    // that
    //
    //   R(angle=gamma, axis=n) =
    //   compose( R(angle=alpha, axis=l), R(angle=beta, axis=m) )
    //
    // I need to compute gamma*n, and these are given as solutions to:
    //
    //   cos(gamma/2) =
    //     cos(alpha/2)*cos(beta/2) -
    //     sin(alpha/2)*sin(beta/2) * inner(l,m)
    //   sin(gamma/2) n =
    //     sin(alpha/2)*cos(beta/2)*l +
    //     cos(alpha/2)*sin(beta/2)*m +
    //     sin(alpha/2)*sin(beta/2) * cross(l,m)
    //
    // For nicer notation, I define
    //
    //   A = alpha/2
    //   B = beta /2
    //   C = gamma/2
    //
    //   l = r0 /(2A)
    //   m = r1 /(2B)
    //   n = r01/(2C)
    //
    // I rewrite:
    //
    //   cos(C) =
    //     cos(A)*cos(B) -
    //     sin(A)*sin(B) * inner(r0,r1) / 4AB
    //   sin(C) r01 / 2C =
    //     sin(A)*cos(B)*r0 / 2A +
    //     cos(A)*sin(B)*r1 / 2B +
    //     sin(A)*sin(B) * cross(r0,r1) / 4AB


    const val_withgrad_t<N> A = r0->mag() / 2.;
    const val_withgrad_t<N> B = r1->mag() / 2.;

    const val_withgrad_t<N>    inner = r0->dot(*r1);
    const vec_withgrad_t<N, 3> cross = r0->cross(*r1);

    if(A.x < 1e-8)
    {
        // A ~ 0. I simplify
        //
        //   cosC ~
        //     + cosB
        //     - A*sinB * inner(r0,r1) / 4AB
        //   sinC r01 / 2C ~
        //     + A*cosB* r0 / 2A
        //     + sinB  * r1 / 2B
        //     + A*sinB * cross(r0,r1) / 4AB
        //
        // I have C = B + dB where dB ~ 0, so
        //
        //   cosC ~ cos(B + dB) ~ cosB - dB sinB
        //   -> dB = A * inner(r0,r1) / 4AB =
        //           inner(r0,r1) / 4B
        //   -> C = B + inner(r0,r1) / 4B
        //
        // Now let's look at the axis expression. Assuming A ~ 0
        //
        //   sinC r01 / 2C ~
        //     + A*cosB r0 / 2A
        //     + sinB   r1 / 2B
        //     + A*sinB * cross(r0,r1) / 4AB
        // ->
        //   sinC/C * r01 ~
        //     + cosB r0
        //     + sinB r1 / B
        //     + sinB * cross(r0,r1) / 2B
        //
        // I linearize the left-hand side:
        //
        //   sinC/C = sin(B+dB)/(B+dB) ~
        //     sinB/B + d( sinB/B )/dB dB =
        //     sinB/B + dB  (B cosB - sinB) / B^2
        //
        // So
        //
        //   (sinB/B + dB  (B cosB - sinB) / B^2) r01 ~
        //     + cosB r0
        //     + sinB r1 / B
        //     + sinB * cross(r0,r1) / 2B
        // ->
        //   (sinB + dB  (B cosB - sinB) / B) r01 ~
        //     + cosB*B r0
        //     + sinB r1
        //     + sinB * cross(r0,r1) / 2
        // ->
        //   sinB (r01 - r1) + dB (B cosB - sinB) / B r01 ~
        //     + cosB*B r0
        //     + sinB * cross(r0,r1) / 2
        //
        // I want to find the perturbation to give me r01 ~ r1 + deltar ->
        //
        //   sinB deltar + dB (B cosB - sinB) / B (r1 + deltar) ~
        //     + cosB*B r0
        //     + sinB * cross(r0,r1) / 2
        //
        // All terms here are linear or quadratic in r0. For tiny r0, I can
        // ignore the quadratic terms:
        //
        //   sinB deltar + dB (B cosB - sinB) / B r1 ~
        //     + cosB*B r0
        //     + sinB * cross(r0,r1) / 2
        // ->
        //   deltar ~
        //     - dB (B/tanB - 1) / B r1
        //     + B/tanB r0
        //     + cross(r0,r1) / 2
        //
        // I substitute in the dB from above, and I simplify:
        //
        //   deltar ~
        //     - inner(r0,r1) / 4B (B/tanB - 1) / B r1
        //     + B/tanB r0
        //     + cross(r0,r1) / 2
        //   =
        //     - inner(r0,r1) (B/tanB - 1) / 4B^2 r1
        //     + B/tanB r0
        //     + cross(r0,r1) / 2
        //
        // So
        //
        // r01 = r1
        //     - inner(r0,r1) (B/tanB - 1) / 4B^2 r1
        //     + B/tanB r0
        //     + cross(r0,r1) / 2
        if(B.x < 1e-8)
        {
            // what if B is ALSO near 0? I simplify further
            //
            // lim(B->0) (B/tanB) = lim( 1 / sec^2 B) = 1.
            // lim(B->0) d(B/tanB)/dB = 0
            //
            // (B/tanB - 1) / 4B^2 =
            // (B - tanB) / (4 B^2 tanB)
            // lim(B->0) = 0
            // lim(B->0) d/dB = 0
            //
            // So
            // r01 = r1
            //     + r0
            //     + cross(r0,r1) / 2
            //
            // Here I have linear and quadratic terms. With my tiny numbers, the
            // quadratic terms can be ignored, so simply
            //
            //   r01 = r0 + r1
            *r = *r0 + *r1;
            return;
        }

        const val_withgrad_t<N>& B_over_tanB = B / B.tan();
        for(int i=0; i<3; i++)
            (*r)[i] =
                (*r1)[i] * (val_withgrad_t<N>(1.0)
                            - inner * (B_over_tanB - 1.) / (B*B*4.))
                + (*r0)[i] * B_over_tanB
                + cross[i] / 2.;
        return;
    }
    else if(B.x < 1e-8)
    {
        // B ~ 0. I simplify
        //
        //   cosC =
        //     cosA -
        //     sinA*B * inner(r0,r1) / 4AB
        //   sinC r01 / 2C =
        //     sinA*r0 / 2A +
        //     cosA*B*r1 / 2B +
        //     sinA*B * cross(r0,r1) / 4AB
        //
        // I have C = A + dA where dA ~ 0, so
        //
        //   cosC ~ cos(A + dA) ~ cosA - dA sinA
        //   -> dA = B * inner(r0,r1) / 4AB =
        //         = inner(r0,r1) / 4A
        //   -> C = A + inner(r0,r1) / 4A
        //
        // Now let's look at the axis expression. Assuming B ~ 0
        //
        //   sinC r01 / 2C =
        //     + sinA*r0 / 2A
        //     + cosA*B*r1 / 2B
        //     + sinA*B * cross(r0,r1) / 4AB
        // ->
        //   sinC/C r01 =
        //     + sinA*r0 / A
        //     + cosA*r1
        //     + sinA * cross(r0,r1) / 2A
        //
        // I linearize the left-hand side:
        //
        //   sinC/C = sin(A+dA)/(A+dA) ~
        //     sinA/A + d( sinA/A )/dA dA =
        //     sinA/A + dA  (A cosA - sinA) / A^2
        //
        // So
        //
        //   (sinA/A + dA  (A cosA - sinA) / A^2) r01 ~
        //     + sinA*r0 / A
        //     + cosA*r1
        //     + sinA * cross(r0,r1) / 2A
        // ->
        //   (sinA + dA  (A cosA - sinA) / A) r01 ~
        //     + sinA*r0
        //     + cosA*r1*A
        //     + sinA * cross(r0,r1) / 2
        // ->
        //   sinA (r01 - r0) + dA (A cosA - sinA) / A r01 ~
        //     + cosA*A r1
        //     + sinA * cross(r0,r1) / 2
        //
        //
        // I want to find the perturbation to give me r01 ~ r0 + deltar ->
        //
        //   sinA deltar + dA (A cosA - sinA) / A (r0 + deltar) ~
        //     + cosA*A r1
        //     + sinA * cross(r0,r1) / 2
        //
        // All terms here are linear or quadratic in r1. For tiny r1, I can
        // ignore the quadratic terms:
        //
        //   sinA deltar + dA (A cosA - sinA) / A r0 ~
        //     + cosA*A r1
        //     + sinA * cross(r0,r1) / 2
        // ->
        //   deltar ~
        //     - dA (A/tanA - 1) / A r0
        //     + A/tanA r1
        //     + cross(r0,r1) / 2
        //
        // I substitute in the dA from above, and I simplify:
        //
        //   deltar ~
        //     - inner(r0,r1) / 4A (A/tanA - 1) / A r0
        //     + A/tanA r1
        //     + cross(r0,r1) / 2
        //   =
        //     - inner(r0,r1) (A/tanA - 1) / 4A^2 r0
        //     + A/tanA r1
        //     + cross(r0,r1) / 2
        //
        // So
        //
        // r01 = r0
        //     - inner(r0,r1) (A/tanA - 1) / 4A^2 r0
        //     + A/tanA r1
        //     + cross(r0,r1) / 2

        // I don't have an if(A.x < 1e-8){} case here; this is handled in
        // the above if() branch

        const val_withgrad_t<N>& A_over_tanA = A / A.tan();
        for(int i=0; i<3; i++)
            (*r)[i] =
                (*r0)[i] * (val_withgrad_t<N>(1.0)
                            - inner * (A_over_tanA - 1.) / (A*A*4.))
                + (*r1)[i] * A_over_tanA
                + cross[i] / 2.;
        return;
    }

    const vec_withgrad_t<N, 2> sincosA = A.sincos();
    const vec_withgrad_t<N, 2> sincosB = B.sincos();

    const val_withgrad_t<N>& sinA = sincosA.v[0];
    const val_withgrad_t<N>& cosA = sincosA.v[1];
    const val_withgrad_t<N>& sinB = sincosB.v[0];
    const val_withgrad_t<N>& cosB = sincosB.v[1];

    const val_withgrad_t<N>& sinA_over_A = A.sinx_over_x(sinA);
    const val_withgrad_t<N>& sinB_over_B = B.sinx_over_x(sinB);

    val_withgrad_t<N> cosC =
        cosA*cosB -
        sinA_over_A*sinB_over_B*inner/4.;

    // To handle numerical fuzz
    if     (cosC.x >  1.0) cosC.x =  1.0;
    else if(cosC.x < -1.0) cosC.x = -1.0;
    const val_withgrad_t<N> C    = cosC.acos();
    const val_withgrad_t<N> sinC = (val_withgrad_t<N>(1.) - cosC*cosC).sqrt();
    const val_withgrad_t<N> sinC_over_C_recip = val_withgrad_t<N>(1.) / C.sinx_over_x(sinC);

    for(int i=0; i<3; i++)
        (*r)[i] =
            ( sinA_over_A*cosB*(*r0)[i] +
              sinB_over_B*cosA*(*r1)[i] +
              sinA_over_A*sinB_over_B*cross[i]/2. ) *
            sinC_over_C_recip;
}

extern "C"
void mrcal_compose_r_full( // output
                           double* r_out,       // (3,) array
                           int r_out_stride0,   // in bytes. <= 0 means "contiguous"
                           double* dr_dr0,      // (3,3) array; may be NULL
                           int dr_dr0_stride0,  // in bytes. <= 0 means "contiguous"
                           int dr_dr0_stride1,  // in bytes. <= 0 means "contiguous"
                           double* dr_dr1,      // (3,3) array; may be NULL
                           int dr_dr1_stride0,  // in bytes. <= 0 means "contiguous"
                           int dr_dr1_stride1,  // in bytes. <= 0 means "contiguous"

                           // input
                           const double* r_0,   // (3,) array
                           int r_0_stride0,     // in bytes. <= 0 means "contiguous"
                           const double* r_1,   // (3,) array
                           int r_1_stride0      // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_1D(r_out,  3);
    init_stride_2D(dr_dr0, 3,3);
    init_stride_2D(dr_dr1, 3,3);
    init_stride_1D(r_0,    3);
    init_stride_1D(r_1,    3);

    if(dr_dr0 == NULL && dr_dr1 == NULL)
    {
        // no gradients
        vec_withgrad_t<0, 3> r0g, r1g;

        r0g.init_vars(r_0,
                      0, 3, -1,
                      r_0_stride0);
        r1g.init_vars(r_1,
                      0, 3, -1,
                      r_1_stride0);

        vec_withgrad_t<0, 3> r01g;
        compose_r_core<0>( &r01g,
                           &r0g, &r1g );

        r01g.extract_value(r_out, r_out_stride0,
                           0, 3);
    }
    else if(dr_dr0 != NULL && dr_dr1 == NULL)
    {
        // r0 gradient only
        vec_withgrad_t<3, 3> r0g, r1g;

        r0g.init_vars(r_0,
                      0, 3, 0,
                      r_0_stride0);
        r1g.init_vars(r_1,
                      0, 3, -1,
                      r_1_stride0);

        vec_withgrad_t<3, 3> r01g;
        compose_r_core<3>( &r01g,
                           &r0g, &r1g );

        r01g.extract_value(r_out, r_out_stride0,
                           0, 3);
        r01g.extract_grad(dr_dr0,
                          0,3,
                          0,
                          dr_dr0_stride0, dr_dr0_stride1,
                          3);
    }
    else if(dr_dr0 == NULL && dr_dr1 != NULL)
    {
        // r1 gradient only
        vec_withgrad_t<3, 3> r0g, r1g;

        r0g.init_vars(r_0,
                      0, 3, -1,
                      r_0_stride0);
        r1g.init_vars(r_1,
                      0, 3, 0,
                      r_1_stride0);

        vec_withgrad_t<3, 3> r01g;
        compose_r_core<3>( &r01g,
                           &r0g, &r1g );

        r01g.extract_value(r_out, r_out_stride0,
                           0, 3);
        r01g.extract_grad(dr_dr1,
                          0,3,
                          0,
                          dr_dr1_stride0, dr_dr1_stride1,
                          3);
    }
    else
    {
        // r0 AND r1 gradients
        vec_withgrad_t<6, 3> r0g, r1g;

        r0g.init_vars(r_0,
                      0, 3, 0,
                      r_0_stride0);
        r1g.init_vars(r_1,
                      0, 3, 3,
                      r_1_stride0);

        vec_withgrad_t<6, 3> r01g;
        compose_r_core<6>( &r01g,
                           &r0g, &r1g );

        r01g.extract_value(r_out, r_out_stride0,
                           0, 3);
        r01g.extract_grad(dr_dr0,
                          0,3,
                          0,
                          dr_dr0_stride0, dr_dr0_stride1,
                          3);
        r01g.extract_grad(dr_dr1,
                          3,3,
                          0,
                          dr_dr1_stride0, dr_dr1_stride1,
                          3);
    }
}
