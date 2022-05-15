#include "minimath-extra.h"

bool get_rt_ref_refperturbed(// inputs
                             // stuff that describes this solve
                             const double* bpacked,
                             const double* x,
                             // transpose(dx/dbpacked)
                             const cholmod_sparse* Jt,

                             // meta-parameters
                             int Ncameras_intrinsics, int Ncameras_extrinsics,
                             int Nframes,
                             int Npoints, int Npoints_fixed, int Nobservations_board,
                             mrcal_problem_selections_t problem_selections,
                             const mrcal_lensmodel_t* lensmodel)
{
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

    If dqref = 0, the rt_ref_refperturbed = -inv() Jcross_t x, which is NOT 0,
    and which is NOT what we want. It makes sense somewhat, however: there's
    some rt_ref_refperturbed that can improve our original optimization
    solution. We're at an optimum (Jt x = 0), but the cross optimization applies
    the same transform to ALL the frames, which .....

#error revisit. Jt x = 0, but this doesn't imply that Jcross_t x = 0. BUT we're
at an optimum, and applying a transform to all the frames is still in my
original optimization state, so it should make the solution worse. What's going
on? Am I applying a transform to all the frames, or am I doing something else?
In any case...

    rt_ref_refperturbed is a random variable, since dqref is a random
    variable. The relationship is nicely linear, so I can compute:

      mean(rt_ref_refperturbed) = -inv(Jcross_t Jcross) Jcross_t x
#error THIS IS THE THING THAT I WANT TO BE 0

    I have expressions with J, but in reality I have J*: gradients in respect to
    PACKED variables. I have a variable scaling diagonal matrix D: J = J* D

    From the finished uncertainty documentation

      M = inv(JtJ) Jobservationst W
        = inv( Dt J*t J* D ) D Jobservations*t W
        = invD inv(J*t J*) Jobservations*t W

      Var(qref) = s^2 W^-2

    So all the W cancel each other out.

    Let's explicate the matrices. (J_frame[] M[frame] + J_calobject_warp[]
    M[calobject_warp]) is J_fcw M where

                i  e        f  calobject_warp
                |  |        |        |
                V  V        V        V
              [ 0 | 0 | xxx       | xxx ]
              [ 0 | 0 | xxx       | xxx ]
              [ 0 | 0 | xxx       | xxx ]
              [ 0 | 0 |    xxx    | xxx ]
      J_fcw = [ 0 | 0 |    xxx    | xxx ]
              [ 0 | 0 |    xxx    | xxx ]
              [ 0 | 0 |       xxx | xxx ]
              [ 0 | 0 |       xxx | xxx ]
              [ 0 | 0 |       xxx | xxx ]

    And

                [ xxx drr0 ]
                [ xxx drr0 ]
                [ xxx drr0 ]
                [ xxx drr1 ]
      Jcross = [ xxx drr1 ]
                [ xxx drr1 ]
                [ xxx drr2 ]
                [ xxx drr2 ]
                [ xxx drr2 ]

    Where the xxx terms are the flattened "frame" terms from J_fcw that use the
    unpacked state. And drr are the drt_ref_frameperturbed__drt_ref_refperturbed
    matrices for the different rt_ref_frame vectors.

    Putting everything together, we have

      K = inv(Jcross_t Jcross) Jcross_t       J_fcw*              inv(J*tJ*)       Jobservations*t
                  (6,6)          (6, Nmeas_obs)  (Nmeas_obs,Nstate)  (Nstate,Nstate)  (Nstate, Nmeas_obs)

      Var(rt_ref_refperturbed) = s^2 K Kt



      Jcross_t J_fcw* =

      [      x         x         x         x         x         x      ]
      [drr0t x   drr0t x   drr0t x   drr1t x   drr1t x   drr1t x .... ] J_fcw* =
      [      x         x         x         x         x         x      ]

      [ 0 | 0 | drt0t sum(outer(xxx,xxx)) | drt1t sum(outer(xxx,xxx)) ... | sum(drtit, outer(xxx_i,xxx_calobject_warp)) ]


      Jcross_t Jcross = sum(outer(jcross, jcross))
                        = sum_j( drr[j]t sum_i(outer(xxx[i], xxx[i])) drr[j] )

      I need sum(outer(...)) for both Jcross_t J_fcw* and Jcross_t Jcross, I
      use this computed sum(outer(...)) in finish_Jcross_computations()

    */


    if(have points)
    {
        barf;
    }

    int state_index_frame0 =
        mrcal_state_index_frames(0,
                                 Ncameras_intrinsics, Ncameras_extrinsics,
                                 Nframes,
                                 Npoints, Npoints_fixed, Nobservations_board,
                                 problem_selections,
                                 lensmodel);
    int num_states_frames =
        mrcal_num_states_frames(Nframes,
                                problem_selections);

    const int*    Jrowptr = (int*)   Jt->p;
    const int*    Jcolidx = (int*)   Jt->i;
    const double* Jval    = (double*)Jt->x;

    double Jcross_t__Jcross[(6+1)*3] = {};

    // Jcross_t J_fcw* has shape (6,Nstate), but the columns corresponding to
    // the camera intrinsics, extrinsics are 0
#error "do I need to store the 0 columns: intrinsics, extrinsics"
    double Jcross_t__J_fcw[6*Nstate] = {};

#error "I should reuse some other memory for this. Chunks of Jcross_t__J_fcw ?"
    // sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) for this frame. I sum over
    // all the observations. Uses PACKED gradients. Only the upper triangle is
    // stored, in the usual row-major order
    double sum_outer_jf_jf_packed[(6+1)*3] = {};

    // I will need composition gradients assuming tiny rt0. I have
    //   compose(rt0, rt1) = compose(r0,r1), rotate(r0,t1)+t0
    // I need to get gradient drt_ref_frameperturbed/drt_ref_refperturbed. Let's
    // look at the r,t separately. I have:
    //   dr/dr0: This is complex. I compute it and store it into this matrix
    //   dr/dt0 = 0
    //   dt/dr0 = -skew_symmetric(t1)
    //   dt/dt0 = I
    double dr_ref_frameperturbed__dr_ref_refperturbed[3*3];
    int state_index_frame_current = -1;

    void finish_Jcross_computations(const int state_index_frame_current,
                                    const double* drr,
                                    const double* t1_packed)
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
        // I have Xp = sum_outer_jf_jf_packed ~ jp jpt
        // Jcross_t__J_fcw ~ drr_t j jpt = drr_t Dinv jp jpt = drr_t Dinv Xp
        // Jcross_t Jcross ~ drr_t j jt drr ~ Jcross_t__J_fcw Dinv drr
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
        //   drr_t X = [dr/dr_t skew_t1] [ S00 S01 ] = [ dr/dr_t S00 + skew_t1 S10    dr/dr_t S01 + skew_t1 S11]
        //             [ 0      I      ] [ S10 S11 ]   [ S10                          S11                      ]
        //
        // Jcross_t__J_fcw output goes into [A B]
        //                                  [C D]
        double* A = &Jcross_t__J_fcw[0*Nstate + state_index_frame_current + 0];
        double* B = &Jcross_t__J_fcw[0*Nstate + state_index_frame_current + 3];
        double* C = &Jcross_t__J_fcw[3*Nstate + state_index_frame_current + 0];
        double* D = &Jcross_t__J_fcw[3*Nstate + state_index_frame_current + 3];

        // From above:
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
        // and I don't care about the ...
        //
        //           [  0 -t2  t1]
        // skew(t) = [ t2   0 -t0]
        //           [-t1  t0   0]
        const double t0 = t1_packed[0] * SCALE_TRANSLATION_FRAME;
        const double t1 = t1_packed[1] * SCALE_TRANSLATION_FRAME;
        const double t2 = t1_packed[2] * SCALE_TRANSLATION_FRAME;

        // A <- dr/dr_t sum_outer[:3,:3] + skew_t1 sum_outer[3:,:3]
        {
            mul_gen33t_gen33insym66(A, Nstate,
                                    drr, -1,
                                    sum_outer_jf_jf_packed, 0, 0,
                                    1./SCALE_ROTATION_FRAME);

            for(int j=0; j<3; j++)
            {
                int i;

                i = 0;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]   + (  0)*sum_outer_jf_jf_packed[index_sym66(0+3,j)] */
                    /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jf_jf_packed[index_sym66(1+3,j)]
                    /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jf_jf_packed[index_sym66(2+3,j)]
                    ) / SCALE_TRANSLATION_FRAME;

                i = 1;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jf_jf_packed[index_sym66(0+3,j)]
                    /*skew[i*3 + 1]   + (  0)*sum_outer_jf_jf_packed[index_sym66(1+3,j)] */
                    /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jf_jf_packed[index_sym66(2+3,j)]
                    ) / SCALE_TRANSLATION_FRAME;

                i = 2;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jf_jf_packed[index_sym66(0+3,j)]
                    /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jf_jf_packed[index_sym66(1+3,j)]
                    /*skew[i*3 + 2]   + (  0)*sum_outer_jf_jf_packed[index_sym66(2+3,j)] */
                    ) / SCALE_TRANSLATION_FRAME;
            }
        }

        // B <- dr/dr_t sum_outer[:3,3:] + skew_t1 sum_outer[3:,3:]
        {
            mul_gen33t_gen33insym66(B, Nstate,
                                    drr, -1,
                                    sum_outer_jf_jf_packed, 0, 3,
                                    1./SCALE_ROTATION_FRAME);

            for(int j=0; j<3; j++)
            {
                int i;

                i = 0;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]   + (  0)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)] */
                    /*skew[i*3 + 1]*/ + (-t2)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)]
                    /*skew[i*3 + 2]*/ + ( t1)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)]
                    ) / SCALE_TRANSLATION_FRAME;

                i = 1;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]*/ + ( t2)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)]
                    /*skew[i*3 + 1]   + (  0)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)] */
                    /*skew[i*3 + 2]*/ + (-t0)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)]
                    ) / SCALE_TRANSLATION_FRAME;

                i = 2;
                A[i*Nstate + j] +=
                    (
                    /*skew[i*3 + 0]*/ + (-t1)*sum_outer_jf_jf_packed[index_sym66(0+3,j+3)]
                    /*skew[i*3 + 1]*/ + ( t0)*sum_outer_jf_jf_packed[index_sym66(1+3,j+3)]
                    /*skew[i*3 + 2]   + (  0)*sum_outer_jf_jf_packed[index_sym66(2+3,j+3)] */
                    ) / SCALE_TRANSLATION_FRAME;
            }
        }

        // C <- sum_outer[3:,:3]
        {
            set_gen33_from_gen33insym66(C, Nstate,
                                        sum_outer_jf_jf_packed, 3, 0,
                                        1./SCALE_TRANSLATION_FRAME);
        }

        // D <- sum_outer[3:,3:]
        {
            set_gen33_from_gen33insym66(D, Nstate,
                                        sum_outer_jf_jf_packed, 3, 3,
                                        1./SCALE_TRANSLATION_FRAME);
        }


        // Jcross_t__Jcross[rr] <- A/SCALE_R dr/dr - B/SCALE_T skew(t1)
        {
            mul_gen33_gen33_into33insym66_accum(Jcross_t__Jcross, 0, 0,
                                                A, Nstate,
                                                drr, -1,
                                                1./SCALE_ROTATION_FRAME);

            int ivalue = 0;
            for(int i=0; i<3; i++)
            {
                for(int j=i; j<3; j++, ivalue++)
                {
                    if(j == 0)
                        Jcross_t__Jcross[ivalue] -=
                            (
                             /*skew[j + 0*3]   + B[i*Nstate+0]*(  0) */
                             /*skew[j + 1*3]*/ + B[i*Nstate+1]*( t2)
                             /*skew[j + 2*3]*/ + B[i*Nstate+2]*(-t1)
                             ) / SCALE_TRANSLATION_FRAME;

                    if(j == 1)
                        Jcross_t__Jcross[ivalue] -=
                            (
                             /*skew[j + 0*3]*/ + B[i*Nstate+0]*(-t2)
                             /*skew[j + 1*3]   + B[i*Nstate+1]*(  0) */
                             /*skew[j + 2*3]*/ + B[i*Nstate+2]*( t0)
                             ) / SCALE_TRANSLATION_FRAME;

                    if(j == 2)
                        Jcross_t__Jcross[ivalue] -=
                            (
                             /*skew[j + 0*3]*/ + B[i*Nstate+0]*( t1)
                             /*skew[j + 1*3]*/ + B[i*Nstate+1]*(-t0)
                             /*skew[j + 2*3]   + B[i*Nstate+2]*(  0) */
                             ) / SCALE_TRANSLATION_FRAME;
                }
                ivalue += 3;
            }
        }

        // Jcross_t__Jcross[rt] <- B/SCALE_T
        {
            set_33insym66_from_gen33_accum(Jcross_t__Jcross, 0, 3,
                                           B, Nstate,
                                           1./SCALE_TRANSLATION_FRAME);
        }

        // Jcross_t__Jcross[tr] doesn't need to be set: I only have values in
        // the upper triangle

        // Jcross_t__Jcross[tt] <- D/SCALE_T = sum_outer[3:,3:]/SCALE_T/SCALE_T
        {
            const int N = sizeof(sum_outer_jf_jf_packed)/sizeof(sum_outer_jf_jf_packed[0]);
            const int i0 = index_sym66_assume_upper(3,3);
            for(int i=i0; i<N; i++)
                Jcross_t__Jcross[i] =
                    sum_outer_jf_jf_packed[i] /
                    (SCALE_TRANSLATION_FRAME*SCALE_TRANSLATION_FRAME);
        }

        memset(sum_outer_jf_jf_packed, 0, sizeof(sum_outer_jf_jf_packed));
    }




    for(int imeas=0; imeas<Nmeas_obs; imeas++)
    {
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
            else if(icol == state_index_frame_current)
            {
                // I already have dr_ref_frameperturbed__dr_ref_refperturbed[] computed
            }
            else
            {
                // Looking at a new frame. Need to
                //   1. Apply old dr_ref_frameperturbed__dr_ref_refperturbed
                //   2. Compute new dr_ref_frameperturbed__dr_ref_refperturbed
                if(state_index_frame_current >= 0)
                    finish_Jcross_computations(state_index_frame_current,
                                                dr_ref_frameperturbed__dr_ref_refperturbed,
                                                &bpacked[state_index_frame_current + 3]);
                state_index_frame_current = icol;

#error UNJUSTIFIED ASSUMPTION
                // UNJUSTIFIED ASSUMPTION HERE. This should use
                // r_refperturbed_frameperturbed = r_ref_frame + M[] dqref, but that makes
                // my life much more complex, so I just use the unperturbed
                // r_ref_frame. I'll try to show empirically that this is just
                // as good
                const double r_ref_frame[3] =
                    { bpacked[state_index_frame_current + 0] * SCALE_ROTATION_FRAME,
                      bpacked[state_index_frame_current + 1] * SCALE_ROTATION_FRAME,
                      bpacked[state_index_frame_current + 2] * SCALE_ROTATION_FRAME };
                mrcal_compose_r_tinyr0_gradientr0(dr_ref_frameperturbed__dr_ref_refperturbed,
                                                  r_ref_frame)
            }

            // Got dr_ref_frameperturbed__dr_ref_refperturbed
            // I have dx/drt_ref_frame for this frame. This is 6 numbers
            const double* dx_drt_ref_frame_packed = &Jval[ival];

            // sum(outer(dx/drt_ref_frame,dx/drt_ref_frame)) into sum_outer_jf_jf_packed
            {
                // This is used to compute Jcross_t J_fcw* and Jcross_t
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

            // Accumulate Jcrosst Jcross = sum(outer(jcross, jcross))
            Jcross_t__Jcross
            for(int i=0; i<6; i++)
                Jcrosst_x[i] += x[imeas]
        }
    }

#error finish_Jcross_computations() here too.
#error barf if I see a point gradient
#error handle calobject_warp gradient properly









    cholmod_dense b = {
        .nrow  = Jt->nrow,
        .ncol  = Nobservations,
        .nzmax = Nobservations * Jt->nrow,
        .d     = Jt->nrow,
        .x     = PyArray_DATA((PyArrayObject*)Py_bt),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };

    Py_out = PyArray_SimpleNew(ndim,
                               PyArray_DIMS((PyArrayObject*)Py_bt),
                               NPY_DOUBLE);
    if(Py_out == NULL)
        {
            BARF("Couldn't allocate Py_out");
            goto done;
        }

    cholmod_dense out = {
        .nrow  = Jt->nrow,
        .ncol  = Nobservations,
        .nzmax = Nobservations * Jt->nrow,
        .d     = Jt->nrow,
        .x     = PyArray_DATA((PyArrayObject*)Py_out),
        .xtype = CHOLMOD_REAL,
        .dtype = CHOLMOD_DOUBLE };

    cholmod_dense* M = &out;
    cholmod_dense* Y = NULL;
    cholmod_dense* E = NULL;

    if(!cholmod_solve2( CHOLMOD_A, self->factorization,
                        &b, NULL,
                        &M, NULL, &Y, &E,
                        &self->common))
        {
            BARF("cholmod_solve2() failed");
            goto done;
        }
    if( M != &out )
        {
            BARF("cholmod_solve2() reallocated out! We leaked memory");
            goto done;
        }
    cholmod_free_dense (&E, &self->common);
    cholmod_free_dense (&Y, &self->common);

    // cholmod_sparse Jt = {
    //     .nrow   = Nstate,
    //     .ncol   = Nmeasurements,
    //     .nzmax  = N_j_nonzero,
    //     .stype  = 0,
    //     .itype  = CHOLMOD_INT,
    //     .xtype  = CHOLMOD_REAL,
    //     .dtype  = CHOLMOD_DOUBLE,
    //     .sorted = 1,
    //     .packed = 1 };

}
