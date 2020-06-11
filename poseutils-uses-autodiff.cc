#include "autodiff.hh"

template<int N>
static void
rotate_point_r_core(// output
                    val_withgrad_t<N>* x_outg,

                    // inputs
                    const val_withgrad_t<N>* rg,
                    const val_withgrad_t<N>* x_ing)
{
    // Rodrigues rotation formula:
    //   xrot = x cos(th) + cross(axis, x) sin(th) + axis axist x (1 - cos(th))
    //
    // I have r = axis*th -> th = norm(r) ->
    //   xrot = x cos(th) + cross(r, x) sin(th)/th + r rt x (1 - cos(th)) / (th*th)
    const val_withgrad_t<N> th2 =
        rg[0]*rg[0] +
        rg[1]*rg[1] +
        rg[2]*rg[2];
    const val_withgrad_t<N> cross[3] =
        {
            rg[1]*x_ing[2] - rg[2]*x_ing[1],
            rg[2]*x_ing[0] - rg[0]*x_ing[2],
            rg[0]*x_ing[1] - rg[1]*x_ing[0]
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
    val_withgrad_t<N> tr    = Rg[0] + Rg[4] + Rg[8];
    val_withgrad_t<N> costh = (tr - 1.) / 2.;

    val_withgrad_t<N> th = costh.acos();
    val_withgrad_t<N> axis[3] =
        {
            Rg[2*3 + 1] - Rg[1*3 + 2],
            Rg[0*3 + 2] - Rg[2*3 + 0],
            Rg[1*3 + 0] - Rg[0*3 + 1]
        };

    if(th.x > 1e-10)
    {
        // normal path
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
void mrcal_rotate_point_r( // output
                          double* x_out, // (3) array
                          double* J_r,   // (3,3) array. May be NULL
                          double* J_x,   // (3,3) array. May be NULL

                          // input
                          const double* r,
                          const double* x_in
                         )
{
    if(J_r == NULL && J_x == NULL)
    {
        vec_withgrad_t<0, 3> rg   (r);
        vec_withgrad_t<0, 3> x_ing(x_in);
        vec_withgrad_t<0, 3> x_outg;
        rotate_point_r_core<0>(x_outg.v,
                               rg.v, x_ing.v);
        x_outg.extract_value(x_out);
    }
    else if(J_r != NULL && J_x == NULL)
    {
        vec_withgrad_t<3, 3> rg   (r, 0);
        vec_withgrad_t<3, 3> x_ing(x_in);
        vec_withgrad_t<3, 3> x_outg;
        rotate_point_r_core<3>(x_outg.v,
                               rg.v, x_ing.v);
        x_outg.extract_value(x_out);
        x_outg.extract_grad (J_r, 0, 3);
    }
    else if(J_r == NULL && J_x != NULL)
    {
        vec_withgrad_t<3, 3> rg   (r);
        vec_withgrad_t<3, 3> x_ing(x_in, 0);
        vec_withgrad_t<3, 3> x_outg;
        rotate_point_r_core<3>(x_outg.v,
                               rg.v, x_ing.v);
        x_outg.extract_value(x_out);
        x_outg.extract_grad (J_x, 0, 3);
    }
    else
    {
        vec_withgrad_t<6, 3> rg   (r,    0);
        vec_withgrad_t<6, 3> x_ing(x_in, 3);
        vec_withgrad_t<6, 3> x_outg;
        rotate_point_r_core<6>(x_outg.v,
                               rg.v, x_ing.v);
        x_outg.extract_value(x_out);
        x_outg.extract_grad (J_r, 0, 3);
        x_outg.extract_grad (J_x, 3, 3);
    }
}

extern "C"
void mrcal_r_from_R( // output
                    double* r, // (3) vector
                    double* J, // (3,3,3) array. Gradient. May be NULL

                    // input
                    const double* R // (3,3) array
                   )
{
    if(J == NULL)
    {
        vec_withgrad_t<0, 3> rg;
        vec_withgrad_t<0, 9> Rg(R);
        r_from_R_core<0>(rg.v, Rg.v);
        rg.extract_value(r);
    }
    else
    {
        vec_withgrad_t<9, 3> rg;
        vec_withgrad_t<9, 9> Rg(R, 0);
        r_from_R_core<9>(rg.v, Rg.v);
        rg.extract_value(r);
        rg.extract_grad(J, 0, 9);
    }
}

