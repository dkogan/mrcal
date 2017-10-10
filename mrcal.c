#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

#include <dogleg.h>
#include <minimath.h>
#include <assert.h>
#include <stdbool.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <dogleg.h>

#include "mrcal.h"

#define VERBOSE 1


#warning terminology: measurement and observation are the same?

#warning kill this comment?
/*

calibration and sfm formulations are a little different

- calibration

  cameras stationary, observed objects move

  ref coord system: cam0

  state:
    for cameras: poses of cameras (except cam0, which is ref)
    for frame:   poses of cal object


  measurements:
    for cameras:
      for frame:
        observations


- sfm

  just like calibration, but I have Nframes cameras, sparsely observing one
  giant calibration object. I have only one frame

  ref coord system: cam0

  state:
    for frame: maybe 3d positions of points. Not required if only two
      cameras observe the point.
    for cameras: poses of cameras (except cam0, which is ref)

  measurements:
    for cameras:
      for frame:
        observations
 */


// These are parameter variable scales. They have the units of the parameters
// themselves, so the optimizer sees x/SCALE_X for each parameter. I.e. as far
// as the optimizer is concerned, the scale of each variable is 1. This doesn't
// need to be precise; just need to get all the variables to be within the same
// order of magnitute. This is important because the dogleg solve treats the
// trust region as a ball in state space, and this ball is isotrophic, and has a
// radius that applies in every direction
#define SCALE_INTRINSICS_FOCAL_LENGTH 10.0
#define SCALE_INTRINSICS_CENTER_PIXEL 2.0
#define SCALE_ROTATION_CAMERA         (1.0 * M_PI/180.0)
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          (2.0 * M_PI/180.0)
#define SCALE_TRANSLATION_FRAME       2.0
#define SCALE_POSITION_POINT          SCALE_TRANSLATION_FRAME

// I need to constrain the point motion since it's not well-defined, and cal
// jump to clearly-incorrect values. This is the distance in front of camera0. I
// make sure this is positive and not unreasonably high
#define POINT_MAXZ                    50000

#warning make this not arbitrary
#define SCALE_DISTORTION              2.0



const char* mrcal_distortion_model_name( enum distortion_model_t model )
{
    switch(model)
    {
#define CASE_STRING(s,n) case s: return #s;
        DISTORTION_LIST( CASE_STRING )

    case DISTORTION_INVALID:
        assert(0);
    }
    return NULL;
}
enum distortion_model_t mrcal_distortion_model_from_name( const char* name )
{
#define CHECK_AND_RETURN(s,n) if( 0 == strcmp( name, #s) ) return s;
    DISTORTION_LIST( CHECK_AND_RETURN );

    return DISTORTION_INVALID;
}

int mrcal_getNdistortionParams(const enum distortion_model_t m)
{
#define SET_NDIST_PARAMS(s,n) [s] = n,
    const signed char numparams[] = { DISTORTION_LIST( SET_NDIST_PARAMS) [DISTORTION_INVALID] = -1 };
    return (int)numparams[m];
}
static int getNintrinsicParams(enum distortion_model_t distortion_model)
{
    return
        N_INTRINSICS_CORE +
        mrcal_getNdistortionParams(distortion_model);
}
static int getNintrinsicOptimizationParams(struct mrcal_variable_select optimization_variable_choice,
                                           enum distortion_model_t distortion_model)
{
    int N = 0;
    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
        N += mrcal_getNdistortionParams(distortion_model);
    if( optimization_variable_choice.do_optimize_intrinsic_core )
        N += N_INTRINSICS_CORE;
    return N;
}

static union point3_t get_refobject_point(int i_pt)
{
    int y = i_pt / CALOBJECT_W;
    int x = i_pt - y*CALOBJECT_W;

    union point3_t pt = {.x = (double)x* CALIBRATION_OBJECT_DOT_SPACING,
                         .y = (double)y* CALIBRATION_OBJECT_DOT_SPACING,
                         .z = 0.0 };
    return pt;
}

static int get_Nstate(int Ncameras, int Nframes, int Npoints,
                      struct mrcal_variable_select optimization_variable_choice,
                      enum distortion_model_t distortion_model)
{
    return
        (Ncameras-1) * 6 + // camera extrinsics
        Nframes      * 6 + // frame poses
        Npoints      * 3 + // individual observed points
        Ncameras * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model); // camera intrinsics
}

static int get_Nmeasurements(int NobservationsBoard,
                             const struct observation_point_t* observations_point,
                             int NobservationsPoint)
{
    // *2 because I have separate x and y measurements
    int Nmeas = NobservationsBoard * NUM_POINTS_IN_CALOBJECT * 2;

    // *2 because I have separate x and y measurements
    Nmeas += NobservationsPoint * 2;

    // known-distance measurements
    for(int i=0; i<NobservationsPoint; i++)
        if(observations_point[i].dist > 0.0) Nmeas++;
    return Nmeas;
}

static int get_N_j_nonzero( const struct observation_board_t* observations_board,
                            int NobservationsBoard,
                            const struct observation_point_t* observations_point,
                            int NobservationsPoint,
                            struct mrcal_variable_select optimization_variable_choice,
                            enum distortion_model_t distortion_model)
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // initial estimate counts extrinsics for camera0, which need to be
    // subtracted off
    int Nintrinsics = getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
    int N = NobservationsBoard * (6 + 6 + Nintrinsics);
    for(int i=0; i<NobservationsBoard; i++)
        if(observations_board[i].i_camera == 0)
            N -= 6;
    N *= NUM_POINTS_IN_CALOBJECT*2; // *2 because I have separate x and y measurements

    // Now the point observations
    for(int i=0; i<NobservationsPoint; i++)
    {
        if(observations_point[i].i_camera == 0)
            N += 2*(Nintrinsics + 3);
        else
            N += 2*(6+Nintrinsics + 3);

        if(observations_point[i].dist > 0)
        {
            if(observations_point[i].i_camera == 0)
                N += 3;
            else
                N += 6 + 3;
        }
    }
    return N;
}


#warning maybe this should project multiple points at a time?
static union point2_t project( // out
                              double*         dxy_dintrinsic_core,
                              double*         dxy_dintrinsic_distortions,
                              union point3_t* dxy_drcamera,
                              union point3_t* dxy_dtcamera,
                              union point3_t* dxy_drframe,
                              union point3_t* dxy_dtframe,

                              // in
                              // these are the unit-scale parameter vectors touched by the solver
                              const double* p_intrinsic_core_unitscale,
                              const double* p_intrinsic_distortions_unitscale,
                              const double* p_camera_rt_unitscale,
                              const double* p_frame_rt_unitscale,
                              bool camera_at_identity,
                              enum distortion_model_t distortion_model,
                              struct mrcal_variable_select optimization_variable_choice,

                              // point index. If <0, a point at the origin is
                              // assumed, dxy_drframe is expected to be NULL and
                              // thus not filled-in, and
                              // p_frame_rt_unitscale[0..2] will not be
                              // referenced
                              int i_pt )
{
    int NdistortionParams = mrcal_getNdistortionParams(distortion_model);

    // I need to compose two transformations
    //
    // (object in reference frame) -> [frame transform] -> (object in camera0 frame) ->
    // -> [camera rt] -> (camera frame)
    //
    // Note that here the frame transform transforms TO the camera0 frame and
    // the camera transform transforms FROM the camera0 frame. This is how my
    // data is expected to be set up
    //
    // [Rc tc] [Rf tf] = [Rc*Rf  Rc*tf + tc]
    // [0  1 ] [0  1 ]   [0      1         ]
    //
    // This transformation (and its gradients) is handled by cvComposeRT() I
    // refer to the camera*frame transform as the "joint" transform, or the
    // letter j

    double _d_rj_rf[3*3];
    double _d_rj_tf[3*3];
    double _d_rj_rc[3*3];
    double _d_rj_tc[3*3];
    double _d_tj_rf[3*3];
    double _d_tj_tf[3*3];
    double _d_tj_rc[3*3];
    double _d_tj_tc[3*3];

#warning some of these are almost certainly zero
    CvMat d_rj_rf = cvMat(3,3, CV_64FC1, _d_rj_rf);
    CvMat d_rj_tf = cvMat(3,3, CV_64FC1, _d_rj_tf);
    CvMat d_rj_rc = cvMat(3,3, CV_64FC1, _d_rj_rc);
    CvMat d_rj_tc = cvMat(3,3, CV_64FC1, _d_rj_tc);
    CvMat d_tj_rf = cvMat(3,3, CV_64FC1, _d_tj_rf);
    CvMat d_tj_tf = cvMat(3,3, CV_64FC1, _d_tj_tf);
    CvMat d_tj_rc = cvMat(3,3, CV_64FC1, _d_tj_rc);
    CvMat d_tj_tc = cvMat(3,3, CV_64FC1, _d_tj_tc);

    double _rj[3];
    CvMat  rj = cvMat(3,1,CV_64FC1, _rj);
    double _tj[3];
    CvMat  tj = cvMat(3,1,CV_64FC1, _tj);
    CvMat* p_rj;
    CvMat* p_tj;

    double _tf[3];
    for(int i=0; i<3; i++)
        _tf[i] = p_frame_rt_unitscale[i+3] * SCALE_TRANSLATION_FRAME;

    double _rf[3];
    if( i_pt >= 0 )
        for(int i=0; i<3; i++)
            _rf[i] = p_frame_rt_unitscale[i+0] * SCALE_ROTATION_FRAME;
    else
        for(int i=0; i<3; i++)
            _rf[i] = 0.0;

    CvMat tf = cvMat(3,1, CV_64FC1, &_tf);
    CvMat rf = cvMat(3,1, CV_64FC1, &_rf);

    union point3_t pt_ref = i_pt >= 0 ? get_refobject_point(i_pt) : (union point3_t){};

    if(!camera_at_identity)
    {
        double _rc[3];
        double _tc[3];
        for(int i=0; i<3; i++)
        {
            _rc[i] = p_camera_rt_unitscale[i+0] * SCALE_ROTATION_CAMERA;
            _tc[i] = p_camera_rt_unitscale[i+3] * SCALE_TRANSLATION_CAMERA;
        }
        CvMat rc = cvMat(3,1, CV_64FC1, &_rc);
        CvMat tc = cvMat(3,1, CV_64FC1, &_tc);

        cvComposeRT( &rf,      &tf,
                     &rc,      &tc,
                     &rj,      &tj,
                     &d_rj_rf, &d_rj_tf,
                     &d_rj_rc, &d_rj_tc,
                     &d_tj_rf, &d_tj_tf,
                     &d_tj_rc, &d_tj_tc );
        p_rj = &rj;
        p_tj = &tj;
    }
    else
    {
        // We're looking at camera0, so this camera transform is fixed at the
        // identity. We don't need to compose anything, nor propagate gradients
        // for the camera extrinsics, since those don't exist in the parameter
        // vector

        // Here the "joint" transform is just the "frame" transform
        p_rj = &rf;
        p_tj = &tf;
    }


    const struct intrinsics_t* intrinsic_core        = (const struct intrinsics_t*)p_intrinsic_core_unitscale;
    const double*              intrinsic_distortions = (const double             *)p_intrinsic_distortions_unitscale;

    if( distortion_model == DISTORTION_OPENCV4 ||
        distortion_model == DISTORTION_OPENCV5 ||
        distortion_model == DISTORTION_OPENCV8 )
    {
        // OpenCV does the projection AND the gradient propagation for me, so I
        // implement a separate code path for it
        union point2_t pt_out;

        CvMat object_points  = cvMat(3,1, CV_64FC1, pt_ref.xyz);
        CvMat image_points   = cvMat(2,1, CV_64FC1, pt_out.xy);

        double _dxy_drj[6];
        double _dxy_dtj[6];
        CvMat  dxy_drj = cvMat(2,3, CV_64FC1, _dxy_drj);
        CvMat  dxy_dtj = cvMat(2,3, CV_64FC1, _dxy_dtj);

        double fx = intrinsic_core->focal_xy [0] * SCALE_INTRINSICS_FOCAL_LENGTH;
        double fy = intrinsic_core->focal_xy [1] * SCALE_INTRINSICS_FOCAL_LENGTH;
        double cx = intrinsic_core->center_xy[0] * SCALE_INTRINSICS_CENTER_PIXEL;
        double cy = intrinsic_core->center_xy[1] * SCALE_INTRINSICS_CENTER_PIXEL;

        double _camera_matrix[] = { fx,  0, cx,
                                     0, fy, cy,
                                     0,  0,  1 };
        CvMat camera_matrix = cvMat(3,3, CV_64FC1, _camera_matrix);
        double _distortions[NdistortionParams];
        for(int i=0; i<NdistortionParams; i++)
            _distortions[i] = intrinsic_distortions[i] * SCALE_DISTORTION;
        CvMat distortions = cvMat( NdistortionParams, 1, CV_64FC1, _distortions);

        // dpdf should be views into dxy_dintrinsic_core[],
        // but it doesn't work. I suspect OpenCV has a bug, but debugging this
        // is taking too much time, so I just copy stuff instead. I wanted this:
        //
        // CvMat  dpdf           = cvMat( 2, 2,
        //                                CV_64FC1, ((struct intrinsics_t*)dxy_dintrinsic_core)->focal_xy);
        // dpdf.step = sizeof(double) * 4;

        double _dpdf[2*2];
        CvMat dpdf = cvMat(2,2, CV_64FC1, _dpdf);
        // instead I do this ^^^^^^^^^^^^^^^^

        CvMat dpddistortions = cvMat(2, NdistortionParams, CV_64FC1, dxy_dintrinsic_distortions);

        CvMat* p_dpdf;
        CvMat* p_dpddistortions;

        if( optimization_variable_choice.do_optimize_intrinsic_core )
            p_dpdf = &dpdf;
        else
            p_dpdf = NULL;
        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
            p_dpddistortions = &dpddistortions;
        else
            p_dpddistortions = NULL;

        cvProjectPoints2(&object_points,
                         p_rj, p_tj,
                         &camera_matrix,
                         &distortions,
                         &image_points,
                         &dxy_drj, &dxy_dtj,
                         p_dpdf,
                         NULL, // dp_dc is trivial: it's the identity
                         p_dpddistortions,
                         0 );


        if( optimization_variable_choice.do_optimize_intrinsic_core )
        {
            struct intrinsics_t* dxy_dintrinsics0 = (struct intrinsics_t*)dxy_dintrinsic_core;
            struct intrinsics_t* dxy_dintrinsics1 = (struct intrinsics_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

            dxy_dintrinsics0->focal_xy [0] = _dpdf[0] * SCALE_INTRINSICS_FOCAL_LENGTH;
            dxy_dintrinsics0->center_xy[0] = SCALE_INTRINSICS_CENTER_PIXEL;
            dxy_dintrinsics0->focal_xy [1] = 0.0;
            dxy_dintrinsics0->center_xy[1] = 0.0;
            dxy_dintrinsics1->focal_xy [0] = 0.0;
            dxy_dintrinsics1->center_xy[0] = 0.0;
            dxy_dintrinsics1->focal_xy [1] = _dpdf[3] * SCALE_INTRINSICS_FOCAL_LENGTH;
            dxy_dintrinsics1->center_xy[1] = SCALE_INTRINSICS_CENTER_PIXEL;
        }


        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
            for(int i=0; i<NdistortionParams; i++)
            {
                dxy_dintrinsic_distortions[i                    ] *= SCALE_DISTORTION;
                dxy_dintrinsic_distortions[i + NdistortionParams] *= SCALE_DISTORTION;
            }


        if(!camera_at_identity)
        {
            // I do this multiple times, one each for {r,t}{camera,frame}
            void propagate(// out
                           union point3_t* dxy_dparam,

                           // in
                           const double* _d_rj_dparam,
                           const double* _d_tj_dparam,

                           // I want the gradients in respect to the unit-scale params
                           double scale_param)
            {
                // I have dproj/drj and dproj/dtj
                // I want dproj/drc, dproj/dtc, dproj/drf, dprof/dtf
                //
                // dproj_drc = dproj/drj drj_drc + dproj/dtj dtj_drc

                mul_genN3_gen33_vout_scaled  (2, _dxy_drj, _d_rj_dparam, dxy_dparam[0].xyz, scale_param);
                mul_genN3_gen33_vaccum_scaled(2, _dxy_dtj, _d_tj_dparam, dxy_dparam[0].xyz, scale_param);
            }

            propagate( dxy_drcamera, _d_rj_rc, _d_tj_rc, SCALE_ROTATION_CAMERA    );
            propagate( dxy_dtcamera, _d_rj_tc, _d_tj_tc, SCALE_TRANSLATION_CAMERA );
            propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf, SCALE_TRANSLATION_FRAME  );
            if(dxy_drframe)
                propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf, SCALE_ROTATION_FRAME     );
        }
        else
        {
            // My gradient is already computed, I just need to scale it

            for(int i=0; i<3; i++)
            {
                dxy_dtframe[0].xyz[i] = _dxy_dtj[i + 3*0] * SCALE_TRANSLATION_FRAME;
                dxy_dtframe[1].xyz[i] = _dxy_dtj[i + 3*1] * SCALE_TRANSLATION_FRAME;
                if(dxy_drframe)
                {
                    dxy_drframe[0].xyz[i] = _dxy_drj[i + 3*0] * SCALE_ROTATION_FRAME;
                    dxy_drframe[1].xyz[i] = _dxy_drj[i + 3*1] * SCALE_ROTATION_FRAME;
                }
            }
        }

        return pt_out;
    }




    // Not using OpenCV distortions, the distortion and projection are not
    // coupled
    double _Rj[3*3];
    CvMat  Rj = cvMat(3,3,CV_64FC1, _Rj);
    double _d_Rj_rj[9*3];
    CvMat d_Rj_rj = cvMat(9,3,CV_64F, _d_Rj_rj);

    cvRodrigues2(p_rj, &Rj, &d_Rj_rj);

    // Rj * pt + tj -> pt
    union point3_t pt_cam;
    mul_vec3_gen33t_vout(pt_ref.xyz, _Rj, pt_cam.xyz);
    add_vec(3, pt_cam.xyz,  p_tj->data.db);

    double dxyz_ddistortion[3*NdistortionParams];
    double* d_distortion_xyz = NULL;
    double  _d_distortion_xyz[3*3] = {};

    // pt_cam is now in the camera coordinates. I can project
    if( distortion_model == DISTORTION_CAHVOR )
    {
        // I perturb pt_cam, and then apply the focal length, center pixel stuff
        // normally
        d_distortion_xyz = _d_distortion_xyz;

        // distortion parameter layout:
        //   theta
        //   phi
        //   r0
        //   r1
        //   r2
        double theta = intrinsic_distortions[0] * SCALE_DISTORTION;
        double phi   = intrinsic_distortions[1] * SCALE_DISTORTION;
        double r0    = intrinsic_distortions[2] * SCALE_DISTORTION;
        double r1    = intrinsic_distortions[3] * SCALE_DISTORTION;
        double r2    = intrinsic_distortions[4] * SCALE_DISTORTION;

        double sth, cth, sph, cph;
        sincos(theta, &sth, &cth);
        sincos(phi,   &sph, &cph);
        double o     [] = {  sph*cth, sph*sth,  cph };
        double do_dth[] = { -sph*sth, sph*cth,    0 };
        double do_dph[] = {  cph*cth, cph*sth, -sph };


        double norm2p = norm2_vec(3, pt_cam.xyz);
        double omega  = dot_vec(3, pt_cam.xyz, o);
        double domega_dth = dot_vec(3, pt_cam.xyz, do_dth);
        double domega_dph = dot_vec(3, pt_cam.xyz, do_dph);

        double omega_recip = 1.0 / omega;
        double tau    = norm2p * omega_recip*omega_recip - 1.0;
        double s__dtau_dthph__domega_dthph = -2.0*norm2p * omega_recip*omega_recip*omega_recip;
        double dmu_dtau = r1 + 2.0*tau*r2;
        double dmu_dxyz[3];
        for(int i=0; i<3; i++)
            dmu_dxyz[i] = dmu_dtau *
                (2.0 * pt_cam.xyz[i] * omega_recip*omega_recip + s__dtau_dthph__domega_dthph * o[i]);
        double mu     = r0 + tau*r1 + tau*tau*r2;
        double s__dmu_dthph__domega_dthph = dmu_dtau * s__dtau_dthph__domega_dthph;

        for(int i=0; i<3; i++)
        {
            double dmu_ddist[5] = { s__dmu_dthph__domega_dthph * domega_dth,
                                    s__dmu_dthph__domega_dthph * domega_dph,
                                    1.0,
                                    tau,
                                    tau * tau };

            dxyz_ddistortion[i*NdistortionParams + 0] = pt_cam.xyz[i] * dmu_ddist[0];
            dxyz_ddistortion[i*NdistortionParams + 1] = pt_cam.xyz[i] * dmu_ddist[1];
            dxyz_ddistortion[i*NdistortionParams + 2] = pt_cam.xyz[i] * dmu_ddist[2];
            dxyz_ddistortion[i*NdistortionParams + 3] = pt_cam.xyz[i] * dmu_ddist[3];
            dxyz_ddistortion[i*NdistortionParams + 4] = pt_cam.xyz[i] * dmu_ddist[4];

            dxyz_ddistortion[i*NdistortionParams + 0] -= dmu_ddist[0] * omega*o[i];
            dxyz_ddistortion[i*NdistortionParams + 1] -= dmu_ddist[1] * omega*o[i];
            dxyz_ddistortion[i*NdistortionParams + 2] -= dmu_ddist[2] * omega*o[i];
            dxyz_ddistortion[i*NdistortionParams + 3] -= dmu_ddist[3] * omega*o[i];
            dxyz_ddistortion[i*NdistortionParams + 4] -= dmu_ddist[4] * omega*o[i];

            dxyz_ddistortion[i*NdistortionParams + 0] -= mu * domega_dth*o[i];
            dxyz_ddistortion[i*NdistortionParams + 1] -= mu * domega_dph*o[i];

            dxyz_ddistortion[i*NdistortionParams + 0] -= mu * omega * do_dth[i];
            dxyz_ddistortion[i*NdistortionParams + 1] -= mu * omega * do_dph[i];


            _d_distortion_xyz[3*i + i] = mu+1.0;
            for(int j=0; j<3; j++)
            {
                _d_distortion_xyz[3*i + j] += (pt_cam.xyz[i] - omega*o[i]) * dmu_dxyz[j];
                _d_distortion_xyz[3*i + j] -= mu*o[i]*o[j];
            }

            pt_cam.xyz[i] += mu * (pt_cam.xyz[i] - omega*o[i]);
        }
    }
    else if( distortion_model == DISTORTION_CAHVORE )
    {
        // set ddistortion_dxyz

        fprintf(stderr, "CAHVORE not implemented yet\n");
        assert(0);
    }
    else if( distortion_model == DISTORTION_NONE )
    {
        d_distortion_xyz = NULL;
    }
    else
    {
        d_distortion_xyz = NULL;
        fprintf(stderr, "Unhandled distortion model: %d (%s)\n",
                distortion_model,
                mrcal_distortion_model_name(distortion_model));
        assert(0);
    }




    union point2_t pt_out;
    const double fx = intrinsic_core->focal_xy[0] * SCALE_INTRINSICS_FOCAL_LENGTH;
    const double fy = intrinsic_core->focal_xy[1] * SCALE_INTRINSICS_FOCAL_LENGTH;
    double z_recip = 1.0 / pt_cam.z;
    pt_out.x =
        pt_cam.x*z_recip * fx +
        intrinsic_core->center_xy[0] * SCALE_INTRINSICS_CENTER_PIXEL;
    pt_out.y =
        pt_cam.y*z_recip * fy +
        intrinsic_core->center_xy[1] * SCALE_INTRINSICS_CENTER_PIXEL;

    // I have the projection, and I now need to propagate the gradients
    if( optimization_variable_choice.do_optimize_intrinsic_core)
    {
        struct intrinsics_t* dxy_dintrinsics0 = (struct intrinsics_t*)dxy_dintrinsic_core;
        struct intrinsics_t* dxy_dintrinsics1 = (struct intrinsics_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

        // I have the projection, and I now need to propagate the gradients
        //
        // xy = fxy * distort(xy)/distort(z) + cxy
        dxy_dintrinsics0->focal_xy [0] = pt_cam.x*z_recip * SCALE_INTRINSICS_FOCAL_LENGTH;
        dxy_dintrinsics0->center_xy[0] = SCALE_INTRINSICS_CENTER_PIXEL;
        dxy_dintrinsics0->focal_xy [1] = 0.0;
        dxy_dintrinsics0->center_xy[1] = 0.0;
        dxy_dintrinsics1->focal_xy [0] = 0.0;
        dxy_dintrinsics1->center_xy[0] = 0.0;
        dxy_dintrinsics1->focal_xy [1] = pt_cam.y*z_recip * SCALE_INTRINSICS_FOCAL_LENGTH;
        dxy_dintrinsics1->center_xy[1] = SCALE_INTRINSICS_CENTER_PIXEL;
    }

    if( optimization_variable_choice.do_optimize_intrinsic_distortions)
        for(int i=0; i<NdistortionParams; i++)
        {
            const double dx = dxyz_ddistortion[i + 0*NdistortionParams];
            const double dy = dxyz_ddistortion[i + 1*NdistortionParams];
            const double dz = dxyz_ddistortion[i + 2*NdistortionParams];
            dxy_dintrinsic_distortions[i                    ] = SCALE_DISTORTION * fx * z_recip * (dx - pt_cam.x*z_recip*dz);
            dxy_dintrinsic_distortions[i + NdistortionParams] = SCALE_DISTORTION * fy * z_recip * (dy - pt_cam.y*z_recip*dz);
        }

    if(!camera_at_identity)
    {
        // I do this multiple times, one each for {r,t}{camera,frame}
        void propagate(union point3_t* dxy_dparam,
                       const double* _d_rj,
                       const double* _d_tj,

                       // I want the gradients in respect to the unit-scale params
                       double scale_param)
        {
            // d(proj_x) = d( fx x/z + cx ) = fx/z * (d(x) - x/z * d(z));
            // d(proj_y) = d( fy y/z + cy ) = fy/z * (d(y) - y/z * d(z));
            //
            // pt_cam.x    = Rj[row0]*pt_ref + tj.x
            // d(pt_cam.x) = d(Rj[row0])*pt_ref + d(tj.x);
            // dRj[row0]/drj is 3x3 matrix at &_d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc

            double d_undistorted_ptcam[3*3];
            double d_distorted_ptcam[3*3];
            double* d_ptcam;
            if(d_distortion_xyz) d_ptcam = d_undistorted_ptcam;
            else                 d_ptcam = d_distorted_ptcam;

            for(int i=0; i<3; i++)
            {
                mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*i], &d_ptcam[3*i] );
                mul_vec3_gen33     ( &d_ptcam[3*i],   _d_rj);
                add_vec(3, &d_ptcam[3*i], &_d_tj[3*i] );
            }

            if(d_distortion_xyz)
                // d_distorted_xyz__... = d_distorted_xyz__undistorted_xyz d_undistorted_xyz__...
                mul_genN3_gen33_vout(3, d_distortion_xyz, d_undistorted_ptcam, d_distorted_ptcam);

            for(int i=0; i<3; i++)
            {
                dxy_dparam[0].xyz[i] = scale_param *
                    fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                dxy_dparam[1].xyz[i] = scale_param *
                    fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
            }
        }

        propagate( dxy_drcamera, _d_rj_rc, _d_tj_rc, SCALE_ROTATION_CAMERA    );
        propagate( dxy_dtcamera, _d_rj_tc, _d_tj_tc, SCALE_TRANSLATION_CAMERA );
        propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf, SCALE_TRANSLATION_FRAME  );
        if( dxy_drframe )
            propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf, SCALE_ROTATION_FRAME     );
    }
    else
    {
        void propagate_r(union point3_t* dxy_dparam,

                         // I want the gradients in respect to the unit-scale params
                         double scale_param)
        {
            // d(proj_x) = d( fx x/z + cx ) = fx/z * (d(x) - x/z * d(z));
            // d(proj_y) = d( fy y/z + cy ) = fy/z * (d(y) - y/z * d(z));
            //
            // pt_cam.x       = Rj[row0]*pt_ref + ...
            // d(pt_cam.x)/dr = d(Rj[row0])*pt_ref
            // dRj[row0]/drj is 3x3 matrix at &_d_Rj_rj[0]

            double d_undistorted_ptcam[3*3];
            double d_distorted_ptcam[3*3];
            double* d_ptcam;
            if(d_distortion_xyz) d_ptcam = d_undistorted_ptcam;
            else                 d_ptcam = d_distorted_ptcam;

            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*0], &d_ptcam[3*0]);
            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*1], &d_ptcam[3*1]);
            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*2], &d_ptcam[3*2]);

            if(d_distortion_xyz)
                // d_distorted_xyz__... = d_distorted_xyz__undistorted_xyz d_undistorted_xyz__...
                mul_genN3_gen33_vout(3, d_distortion_xyz, d_undistorted_ptcam, d_distorted_ptcam);

            for(int i=0; i<3; i++)
            {
                dxy_dparam[0].xyz[i] = scale_param *
                    fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                dxy_dparam[1].xyz[i] = scale_param *
                    fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
            }
        }
        void propagate_t(union point3_t* dxy_dparam,

                         // I want the gradients in respect to the unit-scale params
                         double scale_param)
        {
            // d(proj_x) = d( fx x/z + cx ) = fx/z * (d(x) - x/z * d(z));
            // d(proj_y) = d( fy y/z + cy ) = fy/z * (d(y) - y/z * d(z));
            //
            // pt_cam.x    = ... + tj.x
            // d(pt_cam.x)/dt = identity
            if( d_distortion_xyz == NULL)
            {
                dxy_dparam[0].xyz[0] = scale_param * fx * z_recip;
                dxy_dparam[1].xyz[0] = 0.0;

                dxy_dparam[0].xyz[1] = 0.0;
                dxy_dparam[1].xyz[1] = scale_param * fy * z_recip;

                dxy_dparam[0].xyz[2] = -scale_param * fx * z_recip * pt_cam.x * z_recip;
                dxy_dparam[1].xyz[2] = -scale_param * fy * z_recip * pt_cam.y * z_recip;
            }
            else
            {
                double* d_distorted_ptcam = d_distortion_xyz;

                for(int i=0; i<3; i++)
                {
                    dxy_dparam[0].xyz[i] = scale_param *
                        fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                    dxy_dparam[1].xyz[i] = scale_param *
                        fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
                }
            }
        }

        if( dxy_drframe )
            propagate_r( dxy_drframe, SCALE_ROTATION_FRAME     );
        propagate_t( dxy_dtframe, SCALE_TRANSLATION_FRAME  );
    }
    return pt_out;
}


// The following functions define/use the layout of the state vector. In general
// I do:
//
//   intrinsics_cam0
//   intrinsics_cam1
//   intrinsics_cam2
//   ...
//   extrinsics_cam1
//   extrinsics_cam2
//   extrinsics_cam3
//   ...
//   frame0
//   frame1
//   frame2
//   ....


// From real values to unit-scale values. Optimizer sees unit-scale values
static int pack_solver_state_intrinsics( // out
                                         double* p,

                                         // in
                                         const struct intrinsics_t* intrinsics,
                                         const enum distortion_model_t distortion_model,
                                         struct mrcal_variable_select optimization_variable_choice,
                                         int Ncameras )
{
    int i_state      = 0;
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        if( optimization_variable_choice.do_optimize_intrinsic_core )
        {
            p[i_state++] = intrinsics->focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics->focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics->center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
            p[i_state++] = intrinsics->center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
            for(int i=0; i<Ndistortions; i++)
                p[i_state++] = intrinsics->distortions[i] / SCALE_DISTORTION;

        intrinsics = (struct intrinsics_t*)&((double*)intrinsics)[Nintrinsics];
    }
    return i_state;
}
static void pack_solver_state( // out
                              double* p,

                              // in
                              const struct intrinsics_t* intrinsics, // Ncameras of these
                              const enum distortion_model_t distortion_model,
                              const struct pose_t*       extrinsics, // Ncameras-1 of these
                              const struct pose_t*       frames,     // Nframes of these
                              const union  point3_t*     points,     // Npoints of these
                              struct mrcal_variable_select optimization_variable_choice,
                              int Ncameras, int Nframes, int Npoints,

                              int Nstate_ref)
{
    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, intrinsics,
                                             distortion_model, optimization_variable_choice,
                                             Ncameras );

    for(int i_camera=1; i_camera < Ncameras; i_camera++)
    {
        p[i_state++] = extrinsics[i_camera-1].r.xyz[0] / SCALE_ROTATION_CAMERA;
        p[i_state++] = extrinsics[i_camera-1].r.xyz[1] / SCALE_ROTATION_CAMERA;
        p[i_state++] = extrinsics[i_camera-1].r.xyz[2] / SCALE_ROTATION_CAMERA;

        p[i_state++] = extrinsics[i_camera-1].t.xyz[0] / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = extrinsics[i_camera-1].t.xyz[1] / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = extrinsics[i_camera-1].t.xyz[2] / SCALE_TRANSLATION_CAMERA;
    }

    for(int i_frame = 0; i_frame < Nframes; i_frame++)
    {
        p[i_state++] = frames[i_frame].r.xyz[0] / SCALE_ROTATION_FRAME;
        p[i_state++] = frames[i_frame].r.xyz[1] / SCALE_ROTATION_FRAME;
        p[i_state++] = frames[i_frame].r.xyz[2] / SCALE_ROTATION_FRAME;

        p[i_state++] = frames[i_frame].t.xyz[0] / SCALE_TRANSLATION_FRAME;
        p[i_state++] = frames[i_frame].t.xyz[1] / SCALE_TRANSLATION_FRAME;
        p[i_state++] = frames[i_frame].t.xyz[2] / SCALE_TRANSLATION_FRAME;
    }

    for(int i_point = 0; i_point < Npoints; i_point++)
    {
        p[i_state++] = points[i_point].xyz[0] / SCALE_POSITION_POINT;
        p[i_state++] = points[i_point].xyz[1] / SCALE_POSITION_POINT;
        p[i_state++] = points[i_point].xyz[2] / SCALE_POSITION_POINT;
    }
    assert(i_state == Nstate_ref);
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out
                                 struct intrinsics_t* intrinsics, // Ncameras of these
                                 struct pose_t*       extrinsics, // Ncameras-1 of these
                                 struct pose_t*       frames,     // Nframes of these
                                 union  point3_t*     points,     // Npoints of these

                                 // in
                                 const double* p,
                                 const enum distortion_model_t distortion_model,
                                 struct mrcal_variable_select optimization_variable_choice,
                                 int Ncameras, int Nframes, int Npoints,

                                 int Nstate_ref)

{
    int i_state = 0;

    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        if( optimization_variable_choice.do_optimize_intrinsic_core )
        {
            intrinsics->focal_xy [0] = p[i_state++] *  SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics->focal_xy [1] = p[i_state++] *  SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics->center_xy[0] = p[i_state++] *  SCALE_INTRINSICS_CENTER_PIXEL;
            intrinsics->center_xy[1] = p[i_state++] *  SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
            for(int i=0; i<Ndistortions; i++)
                intrinsics->distortions[i] = p[i_state++] * SCALE_DISTORTION;

        intrinsics = (struct intrinsics_t*)&((double*)intrinsics)[Nintrinsics];
    }

    for(int i_camera=1; i_camera < Ncameras; i_camera++)
    {
        // extrinsics first so that the intrinsics are evenly spaced and
        // state_index_rt_intrinsics() can be branchless
        extrinsics[i_camera-1].r.xyz[0] = p[i_state++] * SCALE_ROTATION_CAMERA;
        extrinsics[i_camera-1].r.xyz[1] = p[i_state++] * SCALE_ROTATION_CAMERA;
        extrinsics[i_camera-1].r.xyz[2] = p[i_state++] * SCALE_ROTATION_CAMERA;

        extrinsics[i_camera-1].t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        extrinsics[i_camera-1].t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        extrinsics[i_camera-1].t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    }

    for(int i_frame = 0; i_frame < Nframes; i_frame++)
    {
        frames[i_frame].r.xyz[0] = p[i_state++] * SCALE_ROTATION_FRAME;
        frames[i_frame].r.xyz[1] = p[i_state++] * SCALE_ROTATION_FRAME;
        frames[i_frame].r.xyz[2] = p[i_state++] * SCALE_ROTATION_FRAME;

        frames[i_frame].t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_FRAME;
        frames[i_frame].t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_FRAME;
        frames[i_frame].t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    }

    for(int i_point = 0; i_point < Npoints; i_point++)
    {
        points[i_point].xyz[0] = p[i_state++] * SCALE_POSITION_POINT;
        points[i_point].xyz[1] = p[i_state++] * SCALE_POSITION_POINT;
        points[i_point].xyz[2] = p[i_state++] * SCALE_POSITION_POINT;
    }

    assert(i_state == Nstate_ref);
}

static int state_index_intrinsic_core(int i_camera,
                                      struct mrcal_variable_select optimization_variable_choice,
                                      enum distortion_model_t distortion_model)
{
    return i_camera * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
}
static int state_index_intrinsic_distortions(int i_camera,
                                             struct mrcal_variable_select optimization_variable_choice,
                                             enum distortion_model_t distortion_model)
{
    int i =
        i_camera * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
    if( optimization_variable_choice.do_optimize_intrinsic_core )
        i += N_INTRINSICS_CORE;
    return i;
}
static int state_index_camera_rt(int i_camera, int Ncameras,
                                 struct mrcal_variable_select optimization_variable_choice,
                                 enum distortion_model_t distortion_model)
{
    // returns the variable index for the extrinsics-followed-by-intrinsics of a
    // given camera. Note that for i_camera==0 there are no extrinsics, so the
    // returned value will be negative, and the intrinsics live at index=0
    return
        getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model)*Ncameras +
        (i_camera-1)*6;
}
static int state_index_frame_rt(int i_frame, int Ncameras,
                                struct mrcal_variable_select optimization_variable_choice,
                                enum distortion_model_t distortion_model)
{
    return
        Ncameras * (getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model) + 6) - 6 +
        i_frame*6;
}
static int state_index_point(int i_point, int Nframes, int Ncameras,
                             struct mrcal_variable_select optimization_variable_choice,
                             enum distortion_model_t distortion_model)
{
    return
        Ncameras * (getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model) + 6) - 6 +
        Nframes * 6 +
        i_point*3;
}

double mrcal_optimize( // out, in (seed on input)

                      // These are the state. I don't have a state_t because Ncameras
                      // and Nframes aren't known at compile time
                      //
                      // intrinsics is struct intrinsics_t: a
                      // concatenation of the intrinsics core and the distortion
                      // params. The specific distortion parameters may vary,
                      // depending on distortion_model, so this is a
                      // variable-length structure
                      struct intrinsics_t* intrinsics, // Ncameras of these
                      struct pose_t*       extrinsics, // Ncameras-1 of these. Transform FROM camera0 frame
                      struct pose_t*       frames,     // Nframes of these.    Transform TO   camera0 frame
                      union  point3_t*     points,     // Npoints of these.    In the camera0 frame

                      // in
                      int Ncameras, int Nframes, int Npoints,

                      const struct observation_board_t* observations_board,
                      int NobservationsBoard,

                      const struct observation_point_t* observations_point,
                      int NobservationsPoint,

                      bool check_gradient,
                      enum distortion_model_t distortion_model,
                      struct mrcal_variable_select optimization_variable_choice )
{
#if defined VERBOSE && VERBOSE
    dogleg_setDebug(100);
#endif

#warning update these parameters
    // These were derived empirically, seeking high accuracy, fast convergence
    // and without serious concern for performance. I looked only at a single
    // frame. Tweak them please
    dogleg_setThresholds(0,1e-15,1e-15);
    dogleg_setMaxIterations(1000);
    dogleg_setTrustregionUpdateParameters(0.1, 0.15, 4.0, 0.75);


    const int Nstate        = get_Nstate(Ncameras, Nframes, Npoints,
                                         optimization_variable_choice,
                                         distortion_model);
    const int Nmeasurements = get_Nmeasurements(NobservationsBoard,
                                                observations_point, NobservationsPoint);
    const int N_j_nonzero   = get_N_j_nonzero(observations_board, NobservationsBoard,
                                              observations_point, NobservationsPoint,
                                              optimization_variable_choice,
                                              distortion_model);

    const int Ndistortions = mrcal_getNdistortionParams(distortion_model);


    // This is needed for the chunks of intrinsics we're not optimizing
    double intrinsics_unitscale[Ncameras * getNintrinsicParams(distortion_model)];
    pack_solver_state_intrinsics( intrinsics_unitscale,
                                  intrinsics,
                                  distortion_model,
                                  (struct mrcal_variable_select){.do_optimize_intrinsic_core        = true,
                                                                 .do_optimize_intrinsic_distortions = true},
                                  Ncameras );


    const char* reportFitMsg = NULL;

    void optimizerCallback(// input state
                           const double*   solver_state,

                           // output measurements
                           double*         x,

                           // Jacobian
                           cholmod_sparse* Jt,

                           void*           cookie __attribute__ ((unused)) )
    {
        // I could unpack_solver_state() here, but I don't want to waste the
        // memory, so I scale stuff as I need it

        int    iJacobian          = 0;
        int    iMeasurement       = 0;

        int*    Jrowptr = Jt ? (int*)   Jt->p : NULL;
        int*    Jcolidx = Jt ? (int*)   Jt->i : NULL;
        double* Jval    = Jt ? (double*)Jt->x : NULL;
#define STORE_JACOBIAN(col, g)                  \
        do                                      \
        {                                       \
            Jcolidx[ iJacobian ] = col;         \
            Jval   [ iJacobian ] = g;           \
            iJacobian++;                        \
        } while(0)
#define STORE_JACOBIAN3(col0, g0, g1, g2)       \
        do                                      \
        {                                       \
            Jcolidx[ iJacobian ] = col0+0;      \
            Jval   [ iJacobian ] = g0;          \
            iJacobian++;                        \
            Jcolidx[ iJacobian ] = col0+1;      \
            Jval   [ iJacobian ] = g1;          \
            iJacobian++;                        \
            Jcolidx[ iJacobian ] = col0+2;      \
            Jval   [ iJacobian ] = g2;          \
            iJacobian++;                        \
        } while(0)



        for(int i_observation_board = 0;
            i_observation_board < NobservationsBoard;
            i_observation_board++)
        {
            const struct observation_board_t* observation = &observations_board[i_observation_board];

            const int i_camera = observation->i_camera;
            const int i_frame  = observation->i_frame;

            const int     i_var_intrinsic_core        = state_index_intrinsic_core(i_camera, optimization_variable_choice, distortion_model);
            const int     i_var_intrinsic_distortions = state_index_intrinsic_distortions(i_camera, optimization_variable_choice, distortion_model);
            const int     i_var_camera_rt             = state_index_camera_rt (i_camera, Ncameras, optimization_variable_choice, distortion_model);
            const int     i_var_frame_rt              = state_index_frame_rt  (i_frame,  Ncameras, optimization_variable_choice, distortion_model);
            const double* p_intrinsic_core            = &solver_state[ i_var_intrinsic_core ];
            const double* p_intrinsic_distortions     = &solver_state[ i_var_intrinsic_distortions ];
            const double* p_camera_rt                 = &solver_state[ i_var_camera_rt ];
            const double* p_frame_rt                  = &solver_state[ i_var_frame_rt ];

            // Stuff we're not optimizing is still required to be known, but
            // it's not in my state variables, so I get it from elsewhere
            if( !optimization_variable_choice.do_optimize_intrinsic_core )
                p_intrinsic_core        = &intrinsics_unitscale[i_camera * getNintrinsicParams(distortion_model)];
            if( !optimization_variable_choice.do_optimize_intrinsic_distortions )
                p_intrinsic_distortions = &intrinsics_unitscale[i_camera * getNintrinsicParams(distortion_model) + N_INTRINSICS_CORE];

            for(int i_pt=0;
                i_pt < NUM_POINTS_IN_CALOBJECT;
                i_pt++)
            {
                // these are computed in respect to the unit-scale parameters
                // used by the optimizer
                double dxy_dintrinsic_core       [2 * N_INTRINSICS_CORE];
                double dxy_dintrinsic_distortions[2 * Ndistortions];
                union point3_t dxy_drcamera[2];
                union point3_t dxy_dtcamera[2];
                union point3_t dxy_drframe [2];
                union point3_t dxy_dtframe [2];

                union point2_t pt_hypothesis =
                    project(dxy_dintrinsic_core,
                            dxy_dintrinsic_distortions,
                            dxy_drcamera,
                            dxy_dtcamera,
                            dxy_drframe,
                            dxy_dtframe,
                            p_intrinsic_core,
                            p_intrinsic_distortions,
                            p_camera_rt, p_frame_rt,
                            i_camera == 0,
                            distortion_model, optimization_variable_choice,
                            i_pt);

                const union point2_t* pt_observed = &observation->px[i_pt];

                if(!observation->skip_observation)
                {
                    // I have my two measurements (dx, dy). I propagate their
                    // gradient and store them
                    for( int i_xy=0; i_xy<2; i_xy++ )
                    {
                        const double err = pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy];

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;

                        // I want these gradient values to be computed in
                        // monotonically-increasing order of variable index. I don't
                        // CHECK, so it's the developer's responsibility to make sure.
                        // This ordering is set in the intrinsics_t structure and in
                        // pack_solver_state(), unpack_solver_state()
                        if( optimization_variable_choice.do_optimize_intrinsic_core )
                            for(int i=0; i<N_INTRINSICS_CORE; i++)
                                STORE_JACOBIAN( i_var_intrinsic_core + i,
                                                dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + i] );

                        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                            for(int i=0; i<Ndistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                                dxy_dintrinsic_distortions[i_xy * Ndistortions + i] );

                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0,
                                             dxy_drcamera[i_xy].xyz[0],
                                             dxy_drcamera[i_xy].xyz[1],
                                             dxy_drcamera[i_xy].xyz[2]);
                            STORE_JACOBIAN3( i_var_camera_rt + 3,
                                             dxy_dtcamera[i_xy].xyz[0],
                                             dxy_dtcamera[i_xy].xyz[1],
                                             dxy_dtcamera[i_xy].xyz[2]);
                        }

                        STORE_JACOBIAN3( i_var_frame_rt + 0,
                                         dxy_drframe[i_xy].xyz[0],
                                         dxy_drframe[i_xy].xyz[1],
                                         dxy_drframe[i_xy].xyz[2]);
                        STORE_JACOBIAN3( i_var_frame_rt + 3,
                                         dxy_dtframe[i_xy].xyz[0],
                                         dxy_dtframe[i_xy].xyz[1],
                                         dxy_dtframe[i_xy].xyz[2]);

                        iMeasurement++;
                    }
                }
                else
                {
                    // This is arbitrary. I'm skipping this observation, so I
                    // don't touch the projection results. I need to have SOME
                    // dependency on the frame parameters to ensure a full-rank
                    // Hessian. So if we're skipping all observations for this
                    // frame, I replace this cost contribution with an L2 cost
                    // on the frame parameters.
                    for( int i_xy=0; i_xy<2; i_xy++ )
                    {
                        const double err = 0.0;

                        if( reportFitMsg )
                        {
                            fprintf(stderr, "%s: obs/frame/cam/dot: %d %d %d %d err: %f\n",
                                    reportFitMsg,
                                    i_observation_board, i_frame, i_camera, i_pt, err);
                            continue;
                        }

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;

                        if( optimization_variable_choice.do_optimize_intrinsic_core )
                            for(int i=0; i<N_INTRINSICS_CORE; i++)
                                STORE_JACOBIAN( i_var_intrinsic_core + i, 0.0 );

                        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                            for(int i=0; i<Ndistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i, 0.0 );

                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }

                        const double dframe = observation->skip_frame ? 1.0 : 0.0;
                        // Arbitrary differences between the dimensions to keep
                        // my Hessian non-singular. This is 100% arbitrary. I'm
                        // skipping these measurements so these variables
                        // actually don't affect the computation at all
                        STORE_JACOBIAN3( i_var_frame_rt + 0, dframe*1.1, dframe*1.2, dframe*1.3);
                        STORE_JACOBIAN3( i_var_frame_rt + 3, dframe*1.4, dframe*1.5, dframe*1.6);

                        iMeasurement++;
                    }
                }
            }
        }

        // Handle all the point observations. This is VERY similar to the
        // board-observation loop above. Please consolidate
        bool   have_invalid_point = false;
        for(int i_observation_point = 0;
            i_observation_point < NobservationsPoint;
            i_observation_point++)
        {
            const struct observation_point_t* observation = &observations_point[i_observation_point];

            const int i_camera = observation->i_camera;
            const int i_point  = observation->i_point;

            const int     i_var_intrinsic_core        = state_index_intrinsic_core(i_camera, optimization_variable_choice, distortion_model);
            const int     i_var_intrinsic_distortions = state_index_intrinsic_distortions(i_camera, optimization_variable_choice, distortion_model);
            const int     i_var_camera_rt             = state_index_camera_rt (i_camera, Ncameras, optimization_variable_choice, distortion_model);
            const int     i_var_point                 = state_index_point     (i_point,  Nframes, Ncameras, optimization_variable_choice, distortion_model);
            const double* p_intrinsic_core            = &solver_state[ i_var_intrinsic_core ];
            const double* p_intrinsic_distortions     = &solver_state[ i_var_intrinsic_distortions ];
            const double* p_camera_rt                 = &solver_state[ i_var_camera_rt ];
            const double* p_point                     = &solver_state[ i_var_point ];

            // Check for invalid points. I report a very poor cost if I see
            // this.
            if( p_point[2] <= 0.0 || p_point[2] >= POINT_MAXZ / SCALE_POSITION_POINT )
            {
                have_invalid_point = true;
#if defined VERBOSE && VERBOSE
                fprintf(stderr, "Saw invalid point distance: z = %f! obs/point/cam: %d %d %d\n",
                        p_point[2]*SCALE_POSITION_POINT,
                        i_observation_point, i_point, i_camera);
#endif
            }


            // Stuff we're not optimizing is still required to be known, but
            // it's not in my state variables, so I get it from elsewhere
            if( !optimization_variable_choice.do_optimize_intrinsic_core )
                p_intrinsic_core        = &intrinsics_unitscale[i_camera * getNintrinsicParams(distortion_model)];
            if( !optimization_variable_choice.do_optimize_intrinsic_distortions )
                p_intrinsic_distortions = &intrinsics_unitscale[i_camera * getNintrinsicParams(distortion_model) + N_INTRINSICS_CORE];

            // these are computed in respect to the unit-scale parameters
            // used by the optimizer
            double dxy_dintrinsic_core       [2 * N_INTRINSICS_CORE];
            double dxy_dintrinsic_distortions[2 * Ndistortions];
            union point3_t dxy_drcamera[2];
            union point3_t dxy_dtcamera[2];
            union point3_t dxy_dpoint  [2];

            union point2_t pt_hypothesis =
                project(dxy_dintrinsic_core,
                        dxy_dintrinsic_distortions,
                        dxy_drcamera,
                        dxy_dtcamera,
                        NULL, // frame rotation. I only have a point position
                        dxy_dpoint,
                        p_intrinsic_core,
                        p_intrinsic_distortions,
                        p_camera_rt,

                        // I only have the point position, so the 'rt' memory
                        // points 3 back. The fake "r" here will not be
                        // referenced
                        &p_point[-3],

                        i_camera == 0,
                        distortion_model, optimization_variable_choice,
                        -1);

            const union point2_t* pt_observed = &observation->px;

            if(!observation->skip_observation)
            {
                // I have my two measurements (dx, dy). I propagate their
                // gradient and store them
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy];

                    Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;

                    // I want these gradient values to be computed in
                    // monotonically-increasing order of variable index. I don't
                    // CHECK, so it's the developer's responsibility to make sure.
                    // This ordering is set in the intrinsics_t structure and in
                    // pack_solver_state(), unpack_solver_state()
                    if( optimization_variable_choice.do_optimize_intrinsic_core )
                        for(int i=0; i<N_INTRINSICS_CORE; i++)
                            STORE_JACOBIAN( i_var_intrinsic_core + i,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + i] );

                    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                        for(int i=0; i<Ndistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                            dxy_dintrinsic_distortions[i_xy * Ndistortions + i] );

                    if( i_camera != 0 )
                    {
                        STORE_JACOBIAN3( i_var_camera_rt + 0,
                                         dxy_drcamera[i_xy].xyz[0],
                                         dxy_drcamera[i_xy].xyz[1],
                                         dxy_drcamera[i_xy].xyz[2]);
                        STORE_JACOBIAN3( i_var_camera_rt + 3,
                                         dxy_dtcamera[i_xy].xyz[0],
                                         dxy_dtcamera[i_xy].xyz[1],
                                         dxy_dtcamera[i_xy].xyz[2]);
                    }

                    STORE_JACOBIAN3( i_var_point,
                                     dxy_dpoint[i_xy].xyz[0],
                                     dxy_dpoint[i_xy].xyz[1],
                                     dxy_dpoint[i_xy].xyz[2]);

                    iMeasurement++;
                }

                // Now handle the reference distance, if given
                if( observation->dist > 0.0)
                {
                    // I do this in the observing-camera coord system. The
                    // camera is at 0. The point is at
                    //
                    //   Rc*p_point + t

                    // This code is copied from project(). PLEASE consolidate
                    if(i_camera == 0)
                    {
                        double dist = sqrt( p_point[0]*p_point[0] +
                                            p_point[1]*p_point[1] +
                                            p_point[2]*p_point[2] );
                        const double err = dist*SCALE_POSITION_POINT - observation->dist;

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;

                        double scale = SCALE_POSITION_POINT/dist;
                        STORE_JACOBIAN3( i_var_point,
                                         scale*p_point[0],
                                         scale*p_point[1],
                                         scale*p_point[2] );
                        iMeasurement++;
                    }
                    else
                    {
                        // I need to transform the point. I already computed
                        // this stuff in project()...
                        double _rc[3];
                        double _tc[3];
                        for(int i=0; i<3; i++)
                        {
                            _rc[i] = p_camera_rt[i+0] * SCALE_ROTATION_CAMERA;
                            _tc[i] = p_camera_rt[i+3] * SCALE_TRANSLATION_CAMERA;
                        }
                        CvMat rc = cvMat(3,1, CV_64FC1, &_rc);

                        double _Rc[3*3];
                        CvMat  Rc = cvMat(3,3,CV_64FC1, _Rc);
                        double _d_Rc_rc[9*3];
                        CvMat d_Rc_rc = cvMat(9,3,CV_64F, _d_Rc_rc);
                        cvRodrigues2(&rc, &Rc, &d_Rc_rc);

                        union point3_t pt_cam;
                        mul_vec3_gen33t_vout_scaled(p_point, _Rc, pt_cam.xyz, SCALE_POSITION_POINT);
                        add_vec(3, pt_cam.xyz, _tc);

                        double dist = sqrt( pt_cam.x*pt_cam.x +
                                            pt_cam.y*pt_cam.y +
                                            pt_cam.z*pt_cam.z );
                        double dist_recip = 1.0/dist;
                        const double err = dist - observation->dist;

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;

                        // pt_cam.x       = Rc[row0]*point*SCALE + tc
                        // d(pt_cam.x)/dr = d(Rc[row0])/drc*point*SCALE
                        // d(Rc[row0])/drc is 3x3 matrix at &_d_Rc_rc[0]
                        double d_ptcamx_dr[3];
                        double d_ptcamy_dr[3];
                        double d_ptcamz_dr[3];
                        mul_vec3_gen33_vout_scaled( p_point, &_d_Rc_rc[9*0], d_ptcamx_dr, SCALE_POSITION_POINT );
                        mul_vec3_gen33_vout_scaled( p_point, &_d_Rc_rc[9*1], d_ptcamy_dr, SCALE_POSITION_POINT );
                        mul_vec3_gen33_vout_scaled( p_point, &_d_Rc_rc[9*2], d_ptcamz_dr, SCALE_POSITION_POINT );

                        STORE_JACOBIAN3( i_var_camera_rt + 0,
                                         SCALE_ROTATION_CAMERA*dist_recip*( pt_cam.x*d_ptcamx_dr[0] +
                                                                            pt_cam.y*d_ptcamy_dr[0] +
                                                                            pt_cam.z*d_ptcamz_dr[0] ),
                                         SCALE_ROTATION_CAMERA*dist_recip*( pt_cam.x*d_ptcamx_dr[1] +
                                                                            pt_cam.y*d_ptcamy_dr[1] +
                                                                            pt_cam.z*d_ptcamz_dr[1] ),
                                         SCALE_ROTATION_CAMERA*dist_recip*( pt_cam.x*d_ptcamx_dr[2] +
                                                                            pt_cam.y*d_ptcamy_dr[2] +
                                                                            pt_cam.z*d_ptcamz_dr[2] ) );
                        STORE_JACOBIAN3( i_var_camera_rt + 3,
                                         SCALE_TRANSLATION_CAMERA*dist_recip*pt_cam.x,
                                         SCALE_TRANSLATION_CAMERA*dist_recip*pt_cam.y,
                                         SCALE_TRANSLATION_CAMERA*dist_recip*pt_cam.z );
                        STORE_JACOBIAN3( i_var_point,
                                         SCALE_POSITION_POINT*dist_recip*(pt_cam.x*_Rc[0] + pt_cam.y*_Rc[3] + pt_cam.z*_Rc[6]),
                                         SCALE_POSITION_POINT*dist_recip*(pt_cam.x*_Rc[1] + pt_cam.y*_Rc[4] + pt_cam.z*_Rc[7]),
                                         SCALE_POSITION_POINT*dist_recip*(pt_cam.x*_Rc[2] + pt_cam.y*_Rc[5] + pt_cam.z*_Rc[8]) );
                        iMeasurement++;
                    }
                }
            }
            else
            {
                // This is arbitrary. I'm skipping this observation, so I
                // don't touch the projection results. I need to have SOME
                // dependency on the point parameters to ensure a full-rank
                // Hessian. So if we're skipping all observations for this
                // point, I replace this cost contribution with an L2 cost
                // on the point parameters.
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = 0.0;

                    Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;

                    if( optimization_variable_choice.do_optimize_intrinsic_core )
                        for(int i=0; i<N_INTRINSICS_CORE; i++)
                            STORE_JACOBIAN( i_var_intrinsic_core + i,
                                            0.0 );

                    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                        for(int i=0; i<Ndistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                            0.0 );

                    if( i_camera != 0 )
                    {
                        STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                        STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                    }

                    const double dpoint = observation->skip_point ? 1.0 : 0.0;
                    // Arbitrary differences between the dimensions to keep
                    // my Hessian non-singular. This is 100% arbitrary. I'm
                    // skipping these measurements so these variables
                    // actually don't affect the computation at all
                    STORE_JACOBIAN3( i_var_point + 0, dpoint*1.1, dpoint*1.2, dpoint*1.3);

                    iMeasurement++;
                }

                // Now handle the reference distance, if given
                if( observation->dist > 0.0)
                {
                    const double err = 0.0;

                    Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;

                    if(i_camera != 0)
                    {
                        STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                        STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                    }
                    STORE_JACOBIAN3( i_var_point, 0.0, 0.0, 0.0);
                    iMeasurement++;
                }
            }

        }

        // required to indicate the end of the jacobian matrix
        if( !reportFitMsg )
        {
            Jrowptr[iMeasurement] = iJacobian;
            assert(iMeasurement == Nmeasurements);
            assert(iJacobian    == N_j_nonzero  );

            // I have an invalid point. This is a VERY bad solution. The solver
            // needs to try again with a smaller step
            if( have_invalid_point )
                *x *= 1e6;
        }
    }








    double state[Nstate];
    pack_solver_state(state,
                      intrinsics,
                      distortion_model,
                      extrinsics,
                      frames,
                      points,
                      optimization_variable_choice,
                      Ncameras, Nframes, Npoints, Nstate);

    double norm2_error = -1.0;
    if( !check_gradient )
    {
#if defined VERBOSE && VERBOSE
        reportFitMsg = "Before";
#warning hook this up
        //        optimizerCallback(state, NULL, NULL, NULL);
#endif
        reportFitMsg = NULL;

        norm2_error = dogleg_optimize(state,
                                      Nstate, Nmeasurements, N_j_nonzero,
                                      &optimizerCallback, NULL, NULL);

        // Done. I have the final state. I spit it back out
        unpack_solver_state( intrinsics, // Ncameras of these
                             extrinsics, // Ncameras-1 of these
                             frames,     // Nframes of these
                             points,     // Npoints of these
                             state,
                             distortion_model,
                             optimization_variable_choice,
                             Ncameras, Nframes, Npoints, Nstate);


#if defined VERBOSE && VERBOSE
        reportFitMsg = "After";
#warning hook this up
        //        optimizerCallback(state, NULL, NULL, NULL);
#endif
    }
    else
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, state,
                                Nstate, Nmeasurements, N_j_nonzero,
                                &optimizerCallback, NULL);

    // Return RMS reprojection error

    // /2 because I have separate x and y measurements
    return sqrt(norm2_error / ((double)Nmeasurements / 2.0));
}
