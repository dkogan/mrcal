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
    val_withgrad_t<N> axis[3] =
        {
            Rg[2*3 + 1] - Rg[1*3 + 2],
            Rg[0*3 + 2] - Rg[2*3 + 0],
            Rg[1*3 + 0] - Rg[0*3 + 1]
        };

    val_withgrad_t<N> costh = (tr - 1.) / 2.;

    if( (fabs(axis[0].x) > 1e-10 ||
         fabs(axis[1].x) > 1e-10 ||
         fabs(axis[2].x) > 1e-10) &&
        fabs(costh.x)   < (1. - 1e-10) )
    {
        // normal path
        val_withgrad_t<N> th = costh.acos();
        val_withgrad_t<N> mag_axis_recip =
            val_withgrad_t<N>(1.) /
            ((axis[0]*axis[0] +
              axis[1]*axis[1] +
              axis[2]*axis[2]).sqrt());
        for(int i=0; i<3; i++)
            rg[i] = axis[i] * mag_axis_recip * th;
    }
    else
    {
        // small th. Can't divide by it. But I can look at the limit.
        //
        // axis / (2 sinth)*th = axis/2 *th/sinth ~ axis/2
        for(int i=0; i<3; i++)
            rg[i] = axis[i] / 2.;
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
