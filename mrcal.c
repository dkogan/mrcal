#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

#include <dogleg.h>
#include <minimath.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include <opencv2/calib3d/calib3d_c.h>
#include <dogleg.h>

#include "mrcal.h"

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

#define DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M 1.0

// I need to constrain the point motion since it's not well-defined, and cal
// jump to clearly-incorrect values. This is the distance in front of camera0. I
// make sure this is positive and not unreasonably high
#define POINT_MAXZ                    50000

#warning make this not arbitrary
#define SCALE_DISTORTION              0.1

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define MSG_IF_VERBOSE(...) do { if(VERBOSE) MSG( __VA_ARGS__ ); } while(0)



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

const char* const* mrcal_getSupportedDistortionModels( void )
{
#define NAMESTRING(s,n) #s,
    static const char* names[] = { DISTORTION_LIST(NAMESTRING) NULL };
    return names;
}

int mrcal_getNintrinsicParams(enum distortion_model_t m)
{
    return
        N_INTRINSICS_CORE +
        mrcal_getNdistortionParams(m);
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
int mrcal_getNintrinsicOptimizationParams(struct mrcal_variable_select optimization_variable_choice,
                                          enum distortion_model_t distortion_model)
{
    return getNintrinsicOptimizationParams(optimization_variable_choice,
                                           distortion_model);
}

static union point3_t get_refobject_point(int i_pt,
                                          double calibration_object_spacing,
                                          int    calibration_object_width_n)
{
    int y = i_pt / calibration_object_width_n;
    int x = i_pt - y*calibration_object_width_n;

    union point3_t pt = {.x = (double)x* calibration_object_spacing,
                         .y = (double)y* calibration_object_spacing,
                         .z = 0.0 };
    return pt;
}

static int get_Nstate(int Ncameras, int Nframes, int Npoints,
                      struct mrcal_variable_select optimization_variable_choice,
                      enum distortion_model_t distortion_model)
{
    return
        // camera extrinsics
        (optimization_variable_choice.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +

        // frame poses
        (optimization_variable_choice.do_optimize_frames ? (Nframes * 6) : 0) +

        // individual observed points
        (optimization_variable_choice.do_optimize_frames ? (Npoints * 3) : 0) +

        // camera intrinsics
        (Ncameras * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model));
}

static int getNmeasurements_observationsonly(int NobservationsBoard,
                                             int NobservationsPoint,
                                             int calibration_object_width_n)
{
    // *2 because I have separate x and y measurements
    int Nmeas =
        NobservationsBoard *
        calibration_object_width_n*calibration_object_width_n *
        2;

    // *2 because I have separate x and y measurements
    Nmeas += NobservationsPoint * 2;
    return Nmeas;
}

int mrcal_getNmeasurements(int Ncameras, int NobservationsBoard,
                           const struct observation_point_t* observations_point,
                           int NobservationsPoint,
                           int calibration_object_width_n,
                           struct mrcal_variable_select optimization_variable_choice,
                           enum distortion_model_t distortion_model)
{
    int Nmeas = getNmeasurements_observationsonly(NobservationsBoard,
                                                  NobservationsPoint,
                                                  calibration_object_width_n);
    // known-distance measurements
    for(int i=0; i<NobservationsPoint; i++)
        if(observations_point[i].dist > 0.0) Nmeas++;

    // regularization
    if(optimization_variable_choice.do_optimize_intrinsic_distortions &&
       !optimization_variable_choice.do_skip_regularization)
    {
        Nmeas += mrcal_getNdistortionParams(distortion_model) * Ncameras;
    }

    return Nmeas;
}

static int get_N_j_nonzero( int Ncameras,
                            const struct observation_board_t* observations_board,
                            int NobservationsBoard,
                            const struct observation_point_t* observations_point,
                            int NobservationsPoint,
                            struct mrcal_variable_select optimization_variable_choice,
                            enum distortion_model_t distortion_model,
                            int calibration_object_width_n)
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // initial estimate counts extrinsics for camera0, which need to be
    // subtracted off
    int Nintrinsics = getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
    int N = NobservationsBoard * ( (optimization_variable_choice.do_optimize_frames     ? 6 : 0) +
                                   (optimization_variable_choice.do_optimize_extrinsics ? 6 : 0) +
                                   + Nintrinsics );
    if(optimization_variable_choice.do_optimize_extrinsics)
        for(int i=0; i<NobservationsBoard; i++)
            if(observations_board[i].i_camera == 0)
                N -= 6;
    N *= 2*calibration_object_width_n*calibration_object_width_n; // *2 because I have separate x and y measurements

    // Now the point observations
    for(int i=0; i<NobservationsPoint; i++)
    {
        N += 2*Nintrinsics;
        if(optimization_variable_choice.do_optimize_frames)
            N += 2*3;
        if( optimization_variable_choice.do_optimize_extrinsics &&
            observations_point[i].i_camera != 0 )
            N += 2*6;

        if(observations_point[i].dist > 0)
        {
            if(optimization_variable_choice.do_optimize_frames)
                N += 3;

            if( optimization_variable_choice.do_optimize_extrinsics &&
                observations_point[i].i_camera != 0 )
                N += 6;
        }
    }

    // regularization
    if(optimization_variable_choice.do_optimize_intrinsic_distortions &&
       !optimization_variable_choice.do_skip_regularization)
        N += mrcal_getNdistortionParams(distortion_model) * Ncameras;

    return N;
}

// internal function used by the optimizer
static union point2_t project( // out
                              double*         dxy_dintrinsic_core,
                              double*         dxy_dintrinsic_distortions,
                              union point3_t* dxy_drcamera,
                              union point3_t* dxy_dtcamera,
                              union point3_t* dxy_drframe,
                              union point3_t* dxy_dtframe,

                              // in
                              const struct intrinsics_core_t* intrinsics_core,
                              const double* distortions,
                              const struct pose_t* camera_rt,
                              const struct pose_t* frame_rt,
                              bool camera_at_identity, // if true, camera_rt is unused
                              enum distortion_model_t distortion_model,

                              // point index. If <0, a point at the origin is
                              // assumed, dxy_drframe is expected to be NULL and
                              // thus not filled-in, and frame_rt->r will not be
                              // referenced. And the calibration_object_...
                              // variables aren't used either
                              int i_pt,

                              double calibration_object_spacing,
                              int    calibration_object_width_n)
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

    const double zero3[3] = {};
    // removing const, but that's just because OpenCV's API is incomplete. It IS
    // const
    CvMat rf = cvMat(3,1, CV_64FC1, (double*)(i_pt <= 0 ? zero3 : frame_rt->r.xyz));
    CvMat tf = cvMat(3,1, CV_64FC1, (double*)frame_rt->t.xyz);

    union point3_t pt_ref =
        i_pt >= 0 ? get_refobject_point(i_pt,
                                        calibration_object_spacing,
                                        calibration_object_width_n)
        : (union point3_t){};

    if(!camera_at_identity)
    {
        // removing const here, but that's just because OpenCV's API is
        // incomplete. It IS const
        CvMat rc = cvMat(3,1, CV_64FC1, (double*)camera_rt->r.xyz);
        CvMat tc = cvMat(3,1, CV_64FC1, (double*)camera_rt->t.xyz);

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


    if( distortion_model == DISTORTION_OPENCV4 ||
        distortion_model == DISTORTION_OPENCV5 ||
        distortion_model == DISTORTION_OPENCV8 ||
        distortion_model == DISTORTION_OPENCV12 ||
        distortion_model == DISTORTION_OPENCV14 )
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

        double fx = intrinsics_core->focal_xy [0];
        double fy = intrinsics_core->focal_xy [1];
        double cx = intrinsics_core->center_xy[0];
        double cy = intrinsics_core->center_xy[1];

        double _camera_matrix[] = { fx,  0, cx,
                                     0, fy, cy,
                                     0,  0,  1 };
        CvMat camera_matrix = cvMat(3,3, CV_64FC1, _camera_matrix);
        CvMat _distortions = cvMat( NdistortionParams, 1, CV_64FC1,
                                   // removing const, but that's just because
                                   // OpenCV's API is incomplete. It IS const
                                   (double*)distortions);

        // dpdf should be views into dxy_dintrinsic_core[],
        // but it doesn't work. I suspect OpenCV has a bug, but debugging this
        // is taking too much time, so I just copy stuff instead. I wanted this:
        //
        // CvMat  dpdf           = cvMat( 2, 2,
        //                                CV_64FC1, ((struct intrinsics_core_t*)dxy_dintrinsic_core)->focal_xy);
        // dpdf.step = sizeof(double) * 4;

        double _dpdf[2*2];
        CvMat dpdf = cvMat(2,2, CV_64FC1, _dpdf);
        // instead I do this ^^^^^^^^^^^^^^^^

        CvMat dpddistortions;

        CvMat* p_dpdf;
        CvMat* p_dpddistortions;

        if( dxy_dintrinsic_core != NULL )
            p_dpdf = &dpdf;
        else
            p_dpdf = NULL;
        if( dxy_dintrinsic_distortions != NULL )
        {
            dpddistortions = cvMat(2, NdistortionParams, CV_64FC1, dxy_dintrinsic_distortions);
            p_dpddistortions = &dpddistortions;
        }
        else
            p_dpddistortions = NULL;

        cvProjectPoints2(&object_points,
                         p_rj, p_tj,
                         &camera_matrix,
                         &_distortions,
                         &image_points,
                         &dxy_drj, &dxy_dtj,
                         p_dpdf,
                         NULL, // dp_dc is trivial: it's the identity
                         p_dpddistortions,
                         0 );


        if( dxy_dintrinsic_core != NULL )
        {
            struct intrinsics_core_t* dxy_dintrinsics0 = (struct intrinsics_core_t*)dxy_dintrinsic_core;
            struct intrinsics_core_t* dxy_dintrinsics1 = (struct intrinsics_core_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

            dxy_dintrinsics0->focal_xy [0] = _dpdf[0];
            dxy_dintrinsics0->center_xy[0] = 1.0;
            dxy_dintrinsics0->focal_xy [1] = 0.0;
            dxy_dintrinsics0->center_xy[1] = 0.0;
            dxy_dintrinsics1->focal_xy [0] = 0.0;
            dxy_dintrinsics1->center_xy[0] = 0.0;
            dxy_dintrinsics1->focal_xy [1] = _dpdf[3];
            dxy_dintrinsics1->center_xy[1] = 1.0;
        }

        if( dxy_drcamera != NULL || dxy_drframe != NULL ||
            dxy_dtcamera != NULL || dxy_dtframe != NULL )
        {
            if(!camera_at_identity)
            {
                // I do this multiple times, one each for {r,t}{camera,frame}
                void propagate(// out
                               union point3_t* dxy_dparam,

                               // in
                               const double* _d_rj_dparam,
                               const double* _d_tj_dparam)
                {
                    if( dxy_dparam == NULL ) return;

                    // I have dproj/drj and dproj/dtj
                    // I want dproj/drc, dproj/dtc, dproj/drf, dprof/dtf
                    //
                    // dproj_drc = dproj/drj drj_drc + dproj/dtj dtj_drc

                    mul_genN3_gen33_vout  (2, _dxy_drj, _d_rj_dparam, dxy_dparam[0].xyz);
                    mul_genN3_gen33_vaccum(2, _dxy_dtj, _d_tj_dparam, dxy_dparam[0].xyz);
                }

                propagate( dxy_drcamera, _d_rj_rc, _d_tj_rc );
                propagate( dxy_dtcamera, _d_rj_tc, _d_tj_tc );
                propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf );
                propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf );
            }
            else
            {
                // My gradient is already computed. Copy it
                if(dxy_dtframe)
                {
                    memcpy(dxy_dtframe[0].xyz, &_dxy_dtj[3*0], 3*sizeof(double));
                    memcpy(dxy_dtframe[1].xyz, &_dxy_dtj[3*1], 3*sizeof(double));
                }
                if(dxy_drframe)
                {
                    memcpy(dxy_drframe[0].xyz, &_dxy_drj[3*0], 3*sizeof(double));
                    memcpy(dxy_drframe[1].xyz, &_dxy_drj[3*1], 3*sizeof(double));
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
        double theta = distortions[0];
        double phi   = distortions[1];
        double r0    = distortions[2];
        double r1    = distortions[3];
        double r2    = distortions[4];

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
    else if( distortion_model == DISTORTION_NONE )
    {
        d_distortion_xyz = NULL;
    }
    else
    {
        d_distortion_xyz = NULL;
        MSG("Unhandled distortion model: %d (%s)",
            distortion_model,
            mrcal_distortion_model_name(distortion_model));
        assert(0);
    }




    union point2_t pt_out;
    const double fx = intrinsics_core->focal_xy [0];
    const double fy = intrinsics_core->focal_xy [1];
    const double cx = intrinsics_core->center_xy[0];
    const double cy = intrinsics_core->center_xy[1];
    double z_recip = 1.0 / pt_cam.z;
    pt_out.x = pt_cam.x*z_recip * fx + cx;
    pt_out.y = pt_cam.y*z_recip * fy + cy;

    // I have the projection, and I now need to propagate the gradients
    if( dxy_dintrinsic_core != NULL )
    {
        struct intrinsics_core_t* dxy_dintrinsics0 = (struct intrinsics_core_t*)dxy_dintrinsic_core;
        struct intrinsics_core_t* dxy_dintrinsics1 = (struct intrinsics_core_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

        // I have the projection, and I now need to propagate the gradients
        //
        // xy = fxy * distort(xy)/distort(z) + cxy
        dxy_dintrinsics0->focal_xy [0] = pt_cam.x*z_recip;
        dxy_dintrinsics0->center_xy[0] = 1.0;
        dxy_dintrinsics0->focal_xy [1] = 0.0;
        dxy_dintrinsics0->center_xy[1] = 0.0;
        dxy_dintrinsics1->focal_xy [0] = 0.0;
        dxy_dintrinsics1->center_xy[0] = 0.0;
        dxy_dintrinsics1->focal_xy [1] = pt_cam.y*z_recip;
        dxy_dintrinsics1->center_xy[1] = 1.0;
    }

    if( dxy_dintrinsic_distortions != NULL )
        for(int i=0; i<NdistortionParams; i++)
        {
            const double dx = dxyz_ddistortion[i + 0*NdistortionParams];
            const double dy = dxyz_ddistortion[i + 1*NdistortionParams];
            const double dz = dxyz_ddistortion[i + 2*NdistortionParams];
            dxy_dintrinsic_distortions[i                    ] = fx * z_recip * (dx - pt_cam.x*z_recip*dz);
            dxy_dintrinsic_distortions[i + NdistortionParams] = fy * z_recip * (dy - pt_cam.y*z_recip*dz);
        }

    if(!camera_at_identity)
    {
        // I do this multiple times, one each for {r,t}{camera,frame}
        void propagate(union point3_t* dxy_dparam,
                       const double* _d_rj,
                       const double* _d_tj)
        {
            if( dxy_dparam == NULL ) return;

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
                dxy_dparam[0].xyz[i] =
                    fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                dxy_dparam[1].xyz[i] =
                    fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
            }
        }

        propagate( dxy_drcamera, _d_rj_rc, _d_tj_rc );
        propagate( dxy_dtcamera, _d_rj_tc, _d_tj_tc );
        propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf );
        propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf );
    }
    else
    {
        void propagate_r(union point3_t* dxy_dparam)
        {
            if( dxy_dparam == NULL ) return;

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
                dxy_dparam[0].xyz[i] =
                    fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                dxy_dparam[1].xyz[i] =
                    fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
            }
        }
        void propagate_t(union point3_t* dxy_dparam)
        {
            if( dxy_dparam == NULL ) return;

            // d(proj_x) = d( fx x/z + cx ) = fx/z * (d(x) - x/z * d(z));
            // d(proj_y) = d( fy y/z + cy ) = fy/z * (d(y) - y/z * d(z));
            //
            // pt_cam.x    = ... + tj.x
            // d(pt_cam.x)/dt = identity
            if( d_distortion_xyz == NULL)
            {
                dxy_dparam[0].xyz[0] = fx * z_recip;
                dxy_dparam[1].xyz[0] = 0.0;

                dxy_dparam[0].xyz[1] = 0.0;
                dxy_dparam[1].xyz[1] = fy * z_recip;

                dxy_dparam[0].xyz[2] = -fx * z_recip * pt_cam.x * z_recip;
                dxy_dparam[1].xyz[2] = -fy * z_recip * pt_cam.y * z_recip;
            }
            else
            {
                double* d_distorted_ptcam = d_distortion_xyz;

                for(int i=0; i<3; i++)
                {
                    dxy_dparam[0].xyz[i] =
                        fx * z_recip * (d_distorted_ptcam[3*0 + i] - pt_cam.x * z_recip * d_distorted_ptcam[3*2 + i]);
                    dxy_dparam[1].xyz[i] =
                        fy * z_recip * (d_distorted_ptcam[3*1 + i] - pt_cam.y * z_recip * d_distorted_ptcam[3*2 + i]);
                }
            }
        }

        propagate_r( dxy_drframe );
        propagate_t( dxy_dtframe );
    }
    return pt_out;
}


// external function. Mostly a wrapper around project()
void mrcal_project( // out
                   union point2_t* out,

                   // core, distortions concatenated. Stored as a row-first
                   // array of shape (N,2,Nintrinsics)
                   double*         dxy_dintrinsics,
                   // Stored as a row-first array of shape (N,2). Each element
                   // of this array is a point3_t
                   union point3_t* dxy_dp,

                   // in
                   const union point3_t* p,
                   int N,
                   enum distortion_model_t distortion_model,
                   // core, distortions concatenated
                   const double* intrinsics)
{
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + 4;

    for(int i=0; i<N; i++)
    {
        struct pose_t frame = {.r = {},
                               .t = p[i]};

        // The data is laid out differently in mrcal_project() and project(), so
        // I need to project() into these temporary variables, and then populate
        // my output array
        double dxy_dintrinsic_core       [2*4];
        double dxy_dintrinsic_distortions[2*Ndistortions];

        out[i] = project( dxy_dintrinsics != NULL ? dxy_dintrinsic_core        : NULL,
                          dxy_dintrinsics != NULL ? dxy_dintrinsic_distortions : NULL,
                          NULL, NULL, NULL,
                          dxy_dp,

                          // in
                          (const struct intrinsics_core_t*)(&intrinsics[0]),
                          &intrinsics[4],
                          NULL,
                          &frame,
                          true,
                          distortion_model,

                          -1, 0.0, 0);
        if(dxy_dintrinsics != NULL)
        {
            for(int j=0; j<4; j++)
            {
                dxy_dintrinsics[j + 0*Nintrinsics] = dxy_dintrinsic_core[j+0];
                dxy_dintrinsics[j + 1*Nintrinsics] = dxy_dintrinsic_core[j+4];
            }
            for(int j=0; j<Ndistortions; j++)
            {
                dxy_dintrinsics[j+4 + 0*Nintrinsics] = dxy_dintrinsic_distortions[j+0           ];
                dxy_dintrinsics[j+4 + 1*Nintrinsics] = dxy_dintrinsic_distortions[j+Ndistortions];
            }

            dxy_dintrinsics = &dxy_dintrinsics[2*Nintrinsics];
        }
        if(dxy_dp != NULL)
        {
            dxy_dp = &dxy_dp[2];
        }
    }
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
                                         const double* intrinsics, // each camera slice is (N_INTRINSICS_CORE, distortions)
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
            const struct intrinsics_core_t* intrinsics_core = (const struct intrinsics_core_t*)intrinsics;
            p[i_state++] = intrinsics_core->focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
            p[i_state++] = intrinsics_core->center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
            for(int i=0; i<Ndistortions; i++)
                p[i_state++] = intrinsics[N_INTRINSICS_CORE + i] / SCALE_DISTORTION;

        intrinsics = &intrinsics[Nintrinsics];
    }
    return i_state;
}
static void pack_solver_state( // out
                              double* p,

                              // in
                              const double* intrinsics, // Ncameras of these;
                                                        // each camera slice is
                                                        // (N_INTRINSICS_CORE,
                                                        // distortions)
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

    if( optimization_variable_choice.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
        {
            p[i_state++] = extrinsics[i_camera-1].r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics[i_camera-1].t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( optimization_variable_choice.do_optimize_frames )
    {
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
    }

    assert(i_state == Nstate_ref);
}

// Same as above, but packs/unpacks a vector instead of structures
static void pack_solver_state_vector( // out, in
                                     double* p, // unitless state on input,
                                                // scaled, meaningful state on
                                                // output

                                     // in
                                     const enum distortion_model_t distortion_model,
                                     struct mrcal_variable_select optimization_variable_choice,
                                     int Ncameras, int Nframes, int Npoints)
{
    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, p,
                                             distortion_model, optimization_variable_choice,
                                             Ncameras );

    static_assert( offsetof(struct pose_t, r) == 0,
                   "pose_t has expected structure");
    static_assert( offsetof(struct pose_t, t) == 3*sizeof(double),
                   "pose_t has expected structure");
    if( optimization_variable_choice.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
        {
            struct pose_t* extrinsics = (struct pose_t*)(&p[i_state]);

            p[i_state++] = extrinsics->r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics->r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics->r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics->t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics->t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics->t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( optimization_variable_choice.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
        {
            struct pose_t* frames = (struct pose_t*)(&p[i_state]);
            p[i_state++] = frames->r.xyz[0] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames->r.xyz[1] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames->r.xyz[2] / SCALE_ROTATION_FRAME;

            p[i_state++] = frames->t.xyz[0] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames->t.xyz[1] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames->t.xyz[2] / SCALE_TRANSLATION_FRAME;
        }

        for(int i_point = 0; i_point < Npoints; i_point++)
        {
            union point3_t* points = (union point3_t*)(&p[i_state]);
            p[i_state++] = points->xyz[0] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[1] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[2] / SCALE_POSITION_POINT;
        }
    }
}

static int unpack_solver_state_intrinsics_onecamera( // out
                                                    struct intrinsics_core_t* intrinsics_core,
                                                    double* distortions,

                                                    // in
                                                    const double* p,
                                                    int Ndistortions,
                                                    struct mrcal_variable_select optimization_variable_choice )
{
    int i_state = 0;
    if( optimization_variable_choice.do_optimize_intrinsic_core )
    {
        intrinsics_core->focal_xy [0] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        intrinsics_core->focal_xy [1] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        intrinsics_core->center_xy[0] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
        intrinsics_core->center_xy[1] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
    }

    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
        for(int i=0; i<Ndistortions; i++)
            distortions[i] = p[i_state++] * SCALE_DISTORTION;

    return i_state;
}

static int unpack_solver_state_intrinsics( // out
                                           double* intrinsics, // Ncameras of
                                                               // these; each
                                                               // camera slice
                                                               // is
                                                               // (N_INTRINSICS_CORE,
                                                               // distortions)

                                           // in
                                           const double* p,
                                           const enum distortion_model_t distortion_model,
                                           struct mrcal_variable_select optimization_variable_choice,
                                           int Ncameras )
{
    if( !optimization_variable_choice.do_optimize_intrinsic_core &&
        !optimization_variable_choice.do_optimize_intrinsic_distortions )
        return 0;

    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    int i_state = 0;
    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        i_state +=
            unpack_solver_state_intrinsics_onecamera( (struct intrinsics_core_t*)intrinsics,
                                                      &intrinsics[N_INTRINSICS_CORE],
                                                      &p[i_state], Ndistortions, optimization_variable_choice );
        intrinsics = &intrinsics[Nintrinsics];
    }
    return i_state;
}

static int unpack_solver_state_extrinsics_one(// out
                                              struct pose_t* extrinsic,

                                              // in
                                              const double* p)
{
    int i_state = 0;
    extrinsic->r.xyz[0] = p[i_state++] * SCALE_ROTATION_CAMERA;
    extrinsic->r.xyz[1] = p[i_state++] * SCALE_ROTATION_CAMERA;
    extrinsic->r.xyz[2] = p[i_state++] * SCALE_ROTATION_CAMERA;

    extrinsic->t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    extrinsic->t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    extrinsic->t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    return i_state;
}

static int unpack_solver_state_framert_one(// out
                                           struct pose_t* frame,

                                           // in
                                           const double* p)
{
    int i_state = 0;
    frame->r.xyz[0] = p[i_state++] * SCALE_ROTATION_FRAME;
    frame->r.xyz[1] = p[i_state++] * SCALE_ROTATION_FRAME;
    frame->r.xyz[2] = p[i_state++] * SCALE_ROTATION_FRAME;

    frame->t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    frame->t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    frame->t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    return i_state;

}

static int unpack_solver_state_point_one(// out
                                         union point3_t* point,

                                         // in
                                         const double* p)
{
    int i_state = 0;
    point->xyz[0] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[1] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[2] = p[i_state++] * SCALE_POSITION_POINT;
    return i_state;
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out
                                 double* intrinsics, // Ncameras of these; each
                                                     // camera slice is
                                                     // (N_INTRINSICS_CORE,
                                                     // distortions)

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
    int i_state = unpack_solver_state_intrinsics(intrinsics,
                                                 p, distortion_model, optimization_variable_choice, Ncameras);

    if( optimization_variable_choice.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics[i_camera-1], &p[i_state] );

    if( optimization_variable_choice.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames[i_frame], &p[i_state] );
        for(int i_point = 0; i_point < Npoints; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }

    assert(i_state == Nstate_ref);
}
// Same as above, but packs/unpacks a vector instead of structures
static void unpack_solver_state_vector( // out, in
                                       double* p, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       const enum distortion_model_t distortion_model,
                                       struct mrcal_variable_select optimization_variable_choice,
                                       int Ncameras, int Nframes, int Npoints)
{
    int i_state = unpack_solver_state_intrinsics(p,
                                                 p, distortion_model, optimization_variable_choice, Ncameras);

    if( optimization_variable_choice.do_optimize_extrinsics )
    {
        static_assert( offsetof(struct pose_t, r) == 0,
                       "pose_t has expected structure");
        static_assert( offsetof(struct pose_t, t) == 3*sizeof(double),
                       "pose_t has expected structure");

        struct pose_t* extrinsics = (struct pose_t*)(&p[i_state]);
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics[i_camera-1], &p[i_state] );
    }

    if( optimization_variable_choice.do_optimize_frames )
    {
        struct pose_t* frames = (struct pose_t*)(&p[i_state]);
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames[i_frame], &p[i_state] );
        union point3_t* points = (union point3_t*)(&p[i_state]);
        for(int i_point = 0; i_point < Npoints; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }
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
    // returns a bogus value if i_camera==0. This camera has no state, and is
    // assumed to be at identity. The caller must know to not use the return
    // value in that case
    int i = getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model)*Ncameras;
    return i + (i_camera-1)*6;
}
static int state_index_frame_rt(int i_frame, int Ncameras,
                                struct mrcal_variable_select optimization_variable_choice,
                                enum distortion_model_t distortion_model)
{
    return
        Ncameras * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model) +
        (optimization_variable_choice.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +
        i_frame * 6;
}
static int state_index_point(int i_point, int Nframes, int Ncameras,
                             struct mrcal_variable_select optimization_variable_choice,
                             enum distortion_model_t distortion_model)
{
    return
        Ncameras * getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model) +
        (optimization_variable_choice.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +
        (Nframes * 6) +
        i_point*3;
}

// This function is part of sensitivity analysis to quantify how much errors in
// the input pixel observations affect our solution. A "good" solution will not
// be very sensitive: measurement noise doesn't affect the solution very much.
//
// I minimize a cost function E = norm2(x) where x is the measurements. Some
// elements of x depend on inputs, and some don't (regularization for instance).
// I perturb the inputs, reoptimize (assuming everything is linear) and look
// what happens to the state p. I'm at an optimum p*:
//
//   dE/dp (p=p*) = 2 Jt x (p=p*) = 0
//
// I perturb the inputs:
//
//   E(x(p+dp, m+dm)) = norm2( x + J dp + dx/dm dm)
//
// And I reoptimize:
//
//   dE/ddp ~ ( x + J dp + dx/dm dm)t J = 0
//
// I'm at an optimum, so Jtx = 0, so
//
//   -Jt dx/dm dm = JtJ dp
//
// So if I perturb my input observation vector m by dm, the resulting effect on
// the parameters is dp = M dm
//
//   where M = -inv(JtJ) Jt dx/dm
//
// In order to be useful I need to do something with M. Let's say I want to
// quantify how precise our optimal intrinsics are. Ultimately these are always
// used in a projection operation. So given a 3d observation vector v, I project
// it onto our image plane:
//
//   q = project(v, intrinsics)
//
// I assume an independent, gaussian noise on my input observations, and for a
// set of given observation vectors v, I compute the effect on the projection.
//
//   dq = dprojection/dintrinsics dintrinsics
//
// dprojection/dintrinsics comes from cvProjectPoints2()
// dintrinsics is the shift in our optimal state: M dm
//
// If dm represents noise of the zero-mean, independent, gaussian variety, then
// dp is also zero-mean gaussian, but no longer independent.
//
//   Var(dp) = M Var(dm) Mt = M Mt s^2
//
// where s is the standard deviation of the noise of each parameter in dm.
//
// The intrinsics of each camera have 3 components:
//
// - f: focal lengths
// - c: center pixel coord
// - d: distortion parameters
//
// Let me define dprojection/df = F, dprojection/dc = C, dprojection/dd = D.
// These all come from cvProjectPoints2().
//
// Rewriting the projection equation I get
//
//   q = project(v,  f,c,d)
//   dq = F df + C dc + D dd
//
// df,dc,dd are random variables that come from dp.
//
//   Var(dq) = F Covar(df,df) Ft +
//             C Covar(dc,dc) Ct +
//             D Covar(dd,dd) Dt +
//             F Covar(df,dc) Ct +
//             F Covar(df,dd) Dt +
//             C Covar(dc,df) Ft +
//             C Covar(dc,dd) Dt +
//             D Covar(dd,df) Ft +
//             D Covar(dd,dc) Ct
//
// Covar(dx,dy) are all submatrices of the larger Var(dp) matrix we computed
// above: M Mt s^2.
//
// Here I look ONLY at the interactions of intrinsic parameters for a particular
// camera with OTHER intrinsic parameters of the same camera. I ignore
// cross-camera interactions and interactions with other parameters, such as the
// frame poses and extrinsics.
//
// For mrcal, the measurements are
//
// 1. reprojection errors of chessboard grid observations
// 2. reprojection errors of individual point observations
// 3. range errors for points with known range
// 4. regularization terms
//
// The observed pixel measurements come into play directly into 1 and 2 above,
// but NOT 3 and 4. Let's say I'm doing ordinary least squares, so x = f(p) - m
//
//   dx/dm = [ -I ]
//           [  0 ]
//
// I thus ignore measurements past the observation set.
//
// My matrices are large and sparse. Thus I compute the blocks of M Mt that I
// need here, and return these densely to the upper levels (python). These
// callers will then use these dense matrices to finish the computation
//
//   M Mt = sum(outer(col(M), col(M)))
//   col(M) = solve(JtJ, row(J))
//
// Note that libdogleg sees everything in the unitless space of scaled
// parameters, and I want this scaling business to be contained in the C code,
// and to not leak out to python. Let's say I have parameters p and their
// unitless scaled versions p*. dp = D dp*. So Var(dp) = D Var(dp*) D. From
// above I have Var(dp*) = M* M*t s^2. So Var(dp) = D M* M*t D s^2. So when
// talking to the upper level, I need to report M = DM*.
static bool computeConfidence_MMt(// out
                                  // dimensions (Ncameras,Nintrinsics_per_camera,Nintrinsics_per_camera)
                                  double* MMt_intrinsics,

                                  // in
                                  enum distortion_model_t distortion_model,
                                  struct mrcal_variable_select optimization_variable_choice,
                                  int Ncameras,
                                  int NobservationsBoard,
                                  int NobservationsPoint,
                                  int Nframes, int Npoints,
                                  int calibration_object_width_n,

                                  dogleg_solverContext_t* solverCtx)
{
    // for reading cholmod_sparse
#define P(A, index) ((unsigned int*)((A)->p))[index]
#define I(A, index) ((unsigned int*)((A)->i))[index]
#define X(A, index) ((double*      )((A)->x))[index]

    if(NobservationsBoard <= 0)
        return false;

    cholmod_sparse* Jt     = solverCtx->beforeStep->Jt;
    int             Nstate = Jt->nrow;
    int             Nmeas  = Jt->ncol;

    // I will repeatedly solve the system JtJ x = v. CHOLMOD can do this for me
    // quickly, if I pre-analyze and pre-factorize JtJ. I do this here, and then
    // just do the cholmod_solve() in the loop. I just ran the solver, so the
    // analysis and factorization are almost certainly already done. But just in
    // case...
    {
        if(solverCtx->factorization == NULL)
        {
            solverCtx->factorization = cholmod_analyze(Jt, &solverCtx->common);
            MSG("Couldn't factor JtJ");
            return false;
        }

        assert( cholmod_factorize(Jt, solverCtx->factorization, &solverCtx->common) );
        if(solverCtx->factorization->minor != solverCtx->factorization->n)
        {
            MSG("Got singular JtJ!");
            return false;
        }
    }

    // cholmod_spsolve works in chunks of 4, so I do this in chunks of 4 too. I
    // pass it rows of J, 4 at a time. I don't actually allocate anything, rather
    // using views into Jt. So I copy the Jt structure and use that
    const unsigned int chunk_size = 4;

    cholmod_dense* Jt_slice =
        cholmod_allocate_dense( Jt->nrow,
                                chunk_size,
                                Jt->nrow,
                                CHOLMOD_REAL,
                                &solverCtx->common );



    // As described above, I'm looking at what input noise does, so I only look at
    // the measurements that pertain to the input observations directly. In mrcal,
    // this is the leading ones, before the range errors and the regularization
    int Nintrinsics_per_camera =
        getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
#warning assumes the point range errors sit AFTER all the reprojection errors
    int Nmeas_observations = getNmeasurements_observationsonly(NobservationsBoard,
                                                               NobservationsPoint,
                                                               calibration_object_width_n);

    memset(MMt_intrinsics, 0,
           Ncameras*Nintrinsics_per_camera* Nintrinsics_per_camera*sizeof(double));






    // make sure the linearized expressions that look at the effects on E
    // and p from perturbations on m are correct
#define DEBUG_CHECK_LINEAR_EXPRESSIONS 0

    // if those expressions are correct, I use the linearized expressions to
    // look at the distribution of E, as dm is perturbed. I have an analytic
    // expression that I want to double-check with the sampling
#define DEBUG_CHECK_E_DISTRIBUTION     0

#if defined DEBUG_CHECK_LINEAR_EXPRESSIONS && DEBUG_CHECK_LINEAR_EXPRESSIONS && defined DEBUG_CHECK_E_DISTRIBUTION && DEBUG_CHECK_E_DISTRIBUTION
#error "No. Check one thing at a time. Do DEBUG_CHECK_LINEAR_EXPRESSIONS and then DEBUG_CHECK_E_DISTRIBUTION"
#endif


#if defined DEBUG_CHECK_LINEAR_EXPRESSIONS && DEBUG_CHECK_LINEAR_EXPRESSIONS
    // A test that this function works correctly. I dump a DENSE representation
    // of J into a file. Then I read it in python, compute MMt from it, and make
    // sure that the MMt I obtained from this function matches. Doing this
    // densely is very slow, but easy, and good-enough for verification
    static int count_Joutput = 0;
    if(count_Joutput == 1)
    {
        char logfilename[128];
        sprintf(logfilename, "/tmp/J%d_%d_%d.dat",count_Joutput,(int)Jt->ncol,(int)Jt->nrow);
        FILE* fp = fopen(logfilename, "w");
        double* Jrow;
        Jrow = malloc(Jt->nrow*sizeof(double));
        for(unsigned int icol=0; icol<Jt->ncol; icol++)
        {
            memset(Jrow, 0, Jt->nrow*sizeof(double));
            for(unsigned int i=P(Jt, icol); i<P(Jt, icol+1); i++)
            {
                int irow = I(Jt,i);
                double x = X(Jt,i);
                Jrow[irow] = x;
            }
            // J has units 1/p, so to UNPACK p I PACK 1/p
            pack_solver_state_vector( Jrow,
                                      distortion_model,
                                      optimization_variable_choice,
                                      Ncameras, Nframes, Npoints);
            // I write binary data. numpy is WAY too slow if I do it in ascii
            fwrite(Jrow,sizeof(double),Jt->nrow,fp);
        }
        fclose(fp);
        free(Jrow);
    }
    count_Joutput++;
    // On the python end, I validate thusly:
    // J     = np.fromfile("/tmp/J1_37008_760.dat").reshape(37008,760)
    // pinvJ = np.linalg.pinv(J)
    // pinvJ = pinvJ[:,:-8] # dx/dm ignores regularization measurements
    // MMt_dense   = nps.matmult(pinvJ, nps.transpose(pinvJ))
    // MMt0 = MMt_dense[0:8,0:8]
    // MMt1 = MMt_dense[8:16,8:16]
    //
    // In [25]: np.linalg.norm(MMt0 - stats['intrinsic_covariances'][0,:,:])
    // Out[25]: 1.4947344824339893e-12
    //
    // In [26]: np.linalg.norm(MMt1 - stats['intrinsic_covariances'][1,:,:])
    // Out[26]: 4.223914927650401e-12
#endif


    for(int i_meas=0; i_meas < Nmeas_observations; i_meas += chunk_size)
    {
        // sparse to dense for a chunk of Jt
        memset( Jt_slice->x, 0, Jt_slice->nrow*chunk_size*sizeof(double) );
        for(unsigned int icol=0; icol<chunk_size; icol++)
        {
            if( (int)(i_meas + icol) >= Nmeas_observations )
            {
                // at the end, we could have one chunk with less that chunk_size
                // columns
                Jt_slice->ncol = icol;
                break;
            }

            for(unsigned int i0=P(Jt, icol+i_meas); i0<P(Jt, icol+i_meas+1); i0++)
            {
                int irow = I(Jt,i0);
                double x0 = X(Jt,i0);
                ((double*)Jt_slice->x)[irow + icol*Jt_slice->nrow] = x0;
            }
        }

        // I'm solving JtJ x = b where J is sparse, b is sparse, but x ends up
        // dense. cholmod doesn't have functions for this exact case. so I use
        // the dense-sparse-dense function, and densify the input. Instead of
        // sparse-sparse-sparse and the densifying the output. This feels like
        // it'd be more efficient
        cholmod_dense* M = cholmod_solve( CHOLMOD_A, solverCtx->factorization,
                                          Jt_slice,
                                          &solverCtx->common);

        // I now have chunk_size columns of M. I accumulate sum of the outer
        // products. This is symmetric, but I store both halves; for now
        for(unsigned int icol=0; icol<M->ncol; icol++)
        {
            // The M I have here is a unitless, scaled M*. I need to scale it to get
            // M. See comment above.
            unpack_solver_state_vector( &((double*)(M->x))[icol*M->nrow],
                                        distortion_model,
                                        optimization_variable_choice,
                                        Ncameras, Nframes, Npoints);



            for(unsigned int irow0=0; irow0<M->nrow; irow0++)
            {
                double x0 = ((double*)(M->x))[irow0 + icol*M->nrow];

                int icam0 = irow0 / Nintrinsics_per_camera;
                if( icam0 >= Ncameras )
                    // not a camera intrinsic parameter
                    continue;

                int i_intrinsics0 = irow0 - icam0*Nintrinsics_per_camera;

                // special-case process the diagonal param
                MMt_intrinsics[icam0*Nintrinsics_per_camera*Nintrinsics_per_camera +
                               (Nintrinsics_per_camera+1)*i_intrinsics0] += x0*x0;

                // Now the off-diagonal
                for(unsigned int irow1=irow0+1; irow1<M->nrow; irow1++)
                {
                    int icam1 = irow1 / Nintrinsics_per_camera;

                    // I want to look at each camera individually, so I ignore the
                    // interactions between the parameters across cameras
                    if( icam0 != icam1 )
                        continue;

                    double x1 = ((double*)(M->x))[irow1 + icol*M->nrow];
                    double x0x1 = x0*x1;
                    int i_intrinsics1 = irow1 - icam1*Nintrinsics_per_camera;

                    double* MMt_thiscam = &MMt_intrinsics[icam0*Nintrinsics_per_camera*Nintrinsics_per_camera];
                    MMt_thiscam[Nintrinsics_per_camera*i_intrinsics0 + i_intrinsics1] += x0x1;
                    MMt_thiscam[Nintrinsics_per_camera*i_intrinsics1 + i_intrinsics0] += x0x1;
                }
            }
        }

        cholmod_free_dense (&M, &solverCtx->common);
    }

    Jt_slice->ncol = chunk_size; // I manually reset this earlier; put it back
    cholmod_free_dense(&Jt_slice, &solverCtx->common);





#if (defined DEBUG_CHECK_LINEAR_EXPRESSIONS && DEBUG_CHECK_LINEAR_EXPRESSIONS) || (defined DEBUG_CHECK_E_DISTRIBUTION && DEBUG_CHECK_E_DISTRIBUTION)
    static int count = 0;
    if(count == 1)
    {
        double rms(const double* x, const double* dx, int N)
        {
            double E = 0;
            for(int i=0;i<N;i++)
            {
                double _x = x[i];
                if(dx) _x += dx[i];
                E += _x*_x;
            }
            return sqrt(E/(double)N*2.0);
        }
        double randn(void)
        {
            // mostly from wikipedia
            static bool generate = false;
            static double z1;
            double z0;
            generate = !generate;

            if (!generate)
                return z1;

            double u1 = (double)rand() / (double)RAND_MAX;
            double u2 = (double)rand() / (double)RAND_MAX;

            double sq = sqrt(-2.0 * log(u1));
            double s,c;
            sincos(2.0*M_PI * u2, &s,&c);
            z0 = sq * c;
            z1 = sq * s;
            return z0;
        }
        // from libdogleg
        void mul_spmatrix_densevector(double* dest,
                                      const cholmod_sparse* A, const double* x)
        {
            memset(dest, 0, sizeof(double) * A->nrow);
            for(unsigned int i=0; i<A->ncol; i++)
            {
                for(unsigned int j=P(A, i); j<P(A, i+1); j++)
                {
                    int row = I(A, j);
                    dest[row] += x[i] * X(A, j);
                }
            }
        }
        void mul_densevector_spmatrix(double* dest,
                                      const double* x, const cholmod_sparse* A)
        {
            memset(dest, 0, sizeof(double) * A->ncol);
            for(unsigned int i=0; i<A->ncol; i++)
            {
                for(unsigned int j=P(A, i); j<P(A, i+1); j++)
                {
                    int row = I(A, j);
                    dest[i] += X(A, j) * x[row];
                }
            }
        }
        __attribute__((unused))
        void writevector( const char* filename, double* x, int N )
        {
            FILE* fp = fopen(filename, "w");
            assert(fp);
            for(int i=0; i<N; i++)
                fprintf(fp, "%g\n", x[i]);
            fclose(fp);
            MSG("wrote '%s'", filename);
        }


        double* x  = solverCtx->beforeStep->x;
        double  E0 = rms(x,NULL,Nmeas);

        // apply noise to get dm
        // dx                 = (JM-I)dm
        // JM                 = J inv(JtJ) Jt
        double* p             = calloc(Nstate,sizeof(double));
        double* dm            = calloc(Nmeas,sizeof(double));
        double* dx            = calloc(Nmeas,sizeof(double));
        double* Jtdm          = calloc(Nstate,sizeof(double));
        double* dx_hypothesis = calloc(Nmeas,sizeof(double));
        cholmod_dense _Jtdm = { .nrow = Nstate,
                                .ncol = 1,
                                .nzmax = Nstate,
                                .d = Nstate,
                                .x = Jtdm,
                                .xtype = CHOLMOD_REAL,
                                .dtype = CHOLMOD_DOUBLE };
        FILE* fp_E = fopen("/tmp/E", "w");
        fprintf(fp_E, "# observed linear quadratic both\n");

        // verification. Jt_x should be 0
        // double *Jt_x = malloc(Nstate*sizeof(double));
        // mul_spmatrix_densevector(Jt_x, Jt, x);
        // writevector("/tmp/Jt_x", Jt_x, Nstate);
        // fclose(fp_Jt_x);


        // I write the unperturbed E0 first
        fprintf(fp_E, "%g %g %g %g\n",E0,E0,E0,E0);

        // I run a whole lotta randomized trials to make sure the observed
        // distributions match the analytical expressions. Those tests are in
        // calibrate-cameras
        for(int i=0; i<10000; i++)
        {
            // leave the last Nmeas-Nmeas_observations always at 0
            for(int j=0; j<Nmeas_observations; j++)
                dm[j] = randn()*1;
            mul_spmatrix_densevector(Jtdm, Jt, dm);
            cholmod_dense* dp = cholmod_solve( CHOLMOD_A, solverCtx->factorization,
                                               &_Jtdm,
                                               &solverCtx->common);
            mul_densevector_spmatrix(dx_hypothesis, (double*)(dp->x), Jt);
            for(int j=0; j<Nmeas; j++)
                dx[j] = (dx_hypothesis[j] - dm[j]);

#if defined DEBUG_CHECK_E_DISTRIBUTION && DEBUG_CHECK_E_DISTRIBUTION
            double E1 = rms(x,dx,Nmeas);

            // dx = (JM-I)dm
            // E1 = norm2(x+dx) = norm2(x) + norm2(dx) + 2*inner(x,dx) =
            //      E0 + norm2(dx) + 2*inner(x,dx)
            //
            // I separate this into the terms linear and quadratic in dx to see
            // if I can ignore the quadratic. I would assume that I can just
            // look at the linear terms to get dE (which will then be gaussian),
            // but the numbers don't match
            double dlinear    = 0;
            double dquadratic = 0;
            double norm2x     = 0.0;
            for(int i=0; i<Nmeas; i++)
            {
                dlinear    += x [i]*dx[i];
                dquadratic += dx[i]*dx[i];
                norm2x     += x [i]*x [i];
            }
            dlinear *= 2.0;

            fprintf(fp_E, "%g %g %g %g\n",
                    E1,
                    sqrt((norm2x + dlinear)/((double)Nmeas/2.0)),
                    sqrt((norm2x + dquadratic)/((double)Nmeas/2.0)),
                    sqrt((norm2x + dlinear + dquadratic)/((double)Nmeas/2.0)));
            cholmod_free_dense(&dp, &solverCtx->common);
#else
            writevector("/tmp/dm",dm,Nmeas);
            writevector("/tmp/Jtdm", Jtdm, Nstate);

            unpack_solver_state_vector( (double*)(dp->x),
                                        distortion_model,
                                        optimization_variable_choice,
                                        Ncameras, Nframes, Npoints);
            writevector("/tmp/dp", dp->x, Nstate); // unpacked

            writevector("/tmp/dx_hypothesis", dx_hypothesis, Nmeas);
            writevector("/tmp/dx",            dx,            Nmeas);

            // Now I project before and after the perturbation
            memcpy(p, solverCtx->beforeStep->p, Nstate*sizeof(double));
            unpack_solver_state_vector( p,
                                        distortion_model,
                                        optimization_variable_choice,
                                        Ncameras, Nframes, Npoints);

            struct pose_t frame = {.t = {.xyz={-0.81691696, -0.02852554,  0.57604945}}};
            union point2_t v0 =  project( NULL,NULL,NULL,NULL,NULL,NULL,
                                          // in
                                          (const struct intrinsics_core_t*)(&p[0]),
                                          &p[4],
                                          NULL,
                                          &frame,
                                          true,
                                          distortion_model,
                                          -1,
                                          1.0, 10);
            for(int i=0; i<Nstate; i++)
                p[i] += ((double*)dp->x)[i];
            union point2_t v1 =  project( NULL,NULL,NULL,NULL,NULL,NULL,
                                          // in
                                          (const struct intrinsics_core_t*)(&p[0]),
                                          &p[4],
                                          NULL,
                                          &frame,
                                          true,
                                          distortion_model,
                                          -1,
                                          1.0, 10);
            fprintf(stderr, "proj before: %.10f %10f\n", v0.x, v0.y);
            fprintf(stderr, "proj after:  %.10f %10f\n", v1.x, v1.y);
            fprintf(stderr, "proj diff:   %.10f %10f\n", v1.x-v0.x, v1.y-v0.y);
            cholmod_free_dense(&dp, &solverCtx->common);
            break;
#endif
        }
        fclose(fp_E);
        free(p);
        free(Jtdm);
        free(dm);
        free(dx);
        free(dx_hypothesis);
    }
    count++;
#endif



    return true;

#undef P
#undef I
#undef X
}

struct mrcal_stats_t
mrcal_optimize( // out
                // These may be NULL. They're for diagnostic reporting to the
                // caller
                double* x_final,
                double* intrinsic_covariances,
                // Buffer should be at least Npoints long. stats->Noutliers
                // elements will be filled in
                int*    outlier_indices_final,

                // out, in

                // if(_solver_context != NULL) then this is a persistent solver
                // context. The context is NOT freed on exit.
                // mrcal_free_context() should be called to release it
                //
                // if(*_solver_context != NULL), the given context is reused
                // if(*_solver_context == NULL), a context is created, and
                // returned here on exit
                void** _solver_context,

                // These are a seed on input, solution on output
                // These are the state. I don't have a state_t because Ncameras
                // and Nframes aren't known at compile time
                //
                // intrinsics is a concatenation of the intrinsics core
                // and the distortion params. The specific distortion
                // parameters may vary, depending on distortion_model, so
                // this is a variable-length structure
                double*              intrinsics, // Ncameras * (N_INTRINSICS_CORE + Ndistortions)
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
                bool VERBOSE,
                const bool skip_outlier_rejection,

                enum distortion_model_t distortion_model,
                struct mrcal_variable_select optimization_variable_choice,

                double calibration_object_spacing,
                int calibration_object_width_n)
{
    if( IS_OPTIMIZE_NONE(optimization_variable_choice) )
        MSG("Warning: Not optimizing any of our variables");

    if(VERBOSE)
        dogleg_setDebug(100);

#warning update these parameters
    // These were derived empirically, seeking high accuracy, fast convergence
    // and without serious concern for performance. I looked only at a single
    // frame. Tweak them please
    dogleg_setThresholds(0, 1e-6, 0);
    dogleg_setMaxIterations(300);
    //dogleg_setTrustregionUpdateParameters(0.1, 0.15, 4.0, 0.75);


    const int Nstate        = get_Nstate(Ncameras, Nframes, Npoints,
                                         optimization_variable_choice,
                                         distortion_model);
    const int Nmeasurements = mrcal_getNmeasurements(Ncameras, NobservationsBoard,
                                                     observations_point, NobservationsPoint,
                                                     calibration_object_width_n,
                                                     optimization_variable_choice,
                                                     distortion_model);
    const int N_j_nonzero   = get_N_j_nonzero(Ncameras,
                                              observations_board, NobservationsBoard,
                                              observations_point, NobservationsPoint,
                                              optimization_variable_choice,
                                              distortion_model,
                                              calibration_object_width_n);

    const int Ndistortions = mrcal_getNdistortionParams(distortion_model);

    const int Npoints_fromBoards =
        NobservationsBoard *
        calibration_object_width_n*calibration_object_width_n;

#warning "outliers only work with board observations for now. I assume consecutive xy measurements, but points can have xyr sprinkled in there. I should make the range-full points always follow the range-less points. Then this will work"
    struct dogleg_outliers_t* markedOutliers = malloc(Npoints_fromBoards*sizeof(struct dogleg_outliers_t));
    if(markedOutliers == NULL)
    {
        MSG("Failed to allocate markedOutliers!");
        return (struct mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }
    memset(markedOutliers, 0, Npoints_fromBoards*sizeof(markedOutliers[0]));

    const char* reportFitMsg = NULL;

    void optimizerCallback(// input state
                           const double*   packed_state,

                           // output measurements
                           double*         x,

                           // Jacobian
                           cholmod_sparse* Jt,

                           void*           cookie __attribute__ ((unused)) )
    {
        double norm2_error = 0.0;

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





        // unpack the state for this observation as a combination of the
        // state and the seed, depending on what we're optimizing, exactly
        struct intrinsics_core_t intrinsic_cores[Ncameras];
        double distortionss[Ncameras][Ndistortions];
        for(int i_camera=0; i_camera<Ncameras; i_camera++)
        {
            const int i_var_intrinsic_core =
                state_index_intrinsic_core(i_camera, optimization_variable_choice, distortion_model);
            unpack_solver_state_intrinsics_onecamera(&intrinsic_cores[i_camera], distortionss[i_camera],
                                                     &packed_state[ i_var_intrinsic_core ],
                                                     Ndistortions,
                                                     optimization_variable_choice );
        }

        for(int i_observation_board = 0;
            i_observation_board < NobservationsBoard;
            i_observation_board++)
        {
            const struct observation_board_t* observation = &observations_board[i_observation_board];

            const int i_camera = observation->i_camera;
            const int i_frame  = observation->i_frame;

            // I could unpack_solver_state() at the top of this function, but I
            // don't want to waste memory, so I scale stuff as I need it; i.e:
            // here


            // Some of these are bogus if optimization_variable_choice says they're inactive
            const int i_var_intrinsic_core         = state_index_intrinsic_core(i_camera, optimization_variable_choice, distortion_model);
            const int i_var_intrinsic_distortions  = state_index_intrinsic_distortions(i_camera, optimization_variable_choice, distortion_model);
            const int i_var_camera_rt              = state_index_camera_rt (i_camera, Ncameras, optimization_variable_choice, distortion_model);
            const int i_var_frame_rt               = state_index_frame_rt  (i_frame,  Ncameras, optimization_variable_choice, distortion_model);

            // unpack the state for this observation as a combination of the
            // state and the seed, depending on what we're optimizing, exactly
            struct intrinsics_core_t* intrinsic_core = &intrinsic_cores[i_camera];
            double* distortions = distortionss[i_camera];
            struct pose_t camera_rt;
            struct pose_t frame_rt;
            if(!optimization_variable_choice.do_optimize_intrinsic_core)
                memcpy( intrinsic_core,
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera],
                        N_INTRINSICS_CORE*sizeof(double) );
            if(!optimization_variable_choice.do_optimize_intrinsic_distortions)
                memcpy( distortions,
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera + N_INTRINSICS_CORE],
                        Ndistortions*sizeof(double) );
            if( i_camera != 0 )
            {
                if(optimization_variable_choice.do_optimize_extrinsics)
                    unpack_solver_state_extrinsics_one(&camera_rt, &packed_state[i_var_camera_rt]);
                else
                    memcpy(&camera_rt, &extrinsics[i_camera-1], sizeof(struct pose_t));
            }
            if(optimization_variable_choice.do_optimize_frames)
                unpack_solver_state_framert_one(&frame_rt, &packed_state[i_var_frame_rt]);
            else
                memcpy(&frame_rt, &frames[i_frame], sizeof(struct pose_t));

            for(int i_pt=0;
                i_pt < calibration_object_width_n*calibration_object_width_n;
                i_pt++)
            {
                // these are computed in respect to the real-unit parameters,
                // NOT the unit-scale parameters used by the optimizer
                double dxy_dintrinsic_core       [2 * N_INTRINSICS_CORE];
                double dxy_dintrinsic_distortions[2 * Ndistortions];
                union point3_t dxy_drcamera[2];
                union point3_t dxy_dtcamera[2];
                union point3_t dxy_drframe [2];
                union point3_t dxy_dtframe [2];

                union point2_t pt_hypothesis =
                    project(optimization_variable_choice.do_optimize_intrinsic_core ?
                              dxy_dintrinsic_core : NULL,
                            optimization_variable_choice.do_optimize_intrinsic_distortions ?
                              dxy_dintrinsic_distortions : NULL,
                            optimization_variable_choice.do_optimize_extrinsics ?
                              dxy_drcamera : NULL,
                            optimization_variable_choice.do_optimize_extrinsics ?
                              dxy_dtcamera : NULL,
                            optimization_variable_choice.do_optimize_frames ?
                              dxy_drframe : NULL,
                            optimization_variable_choice.do_optimize_frames ?
                              dxy_dtframe : NULL,
                            intrinsic_core, distortions,
                            &camera_rt, &frame_rt,
                            i_camera == 0,
                            distortion_model,
                            i_pt,
                            calibration_object_spacing,
                            calibration_object_width_n);

                const union point2_t* pt_observed = &observation->px[i_pt];

                if(!observation->skip_observation &&

                   // /2 because I look at FEATURES here, not discrete
                   // measurements
                   !markedOutliers[iMeasurement/2].marked)
                {
                    // I have my two measurements (dx, dy). I propagate their
                    // gradient and store them
                    for( int i_xy=0; i_xy<2; i_xy++ )
                    {
                        const double err = pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy];

                        if( reportFitMsg )
                        {
                            MSG("%s: obs/frame/cam/dot: %d %d %d %d err: %g",
                                reportFitMsg,
                                i_observation_board, i_frame, i_camera, i_pt, err);
                            continue;
                        }

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        // I want these gradient values to be computed in
                        // monotonically-increasing order of variable index. I
                        // don't CHECK, so it's the developer's responsibility
                        // to make sure. This ordering is set in
                        // pack_solver_state(), unpack_solver_state()
                        if( optimization_variable_choice.do_optimize_intrinsic_core )
                        {
                            STORE_JACOBIAN( i_var_intrinsic_core + 0,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 0] * SCALE_INTRINSICS_FOCAL_LENGTH );
                            STORE_JACOBIAN( i_var_intrinsic_core + 1,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 1] * SCALE_INTRINSICS_FOCAL_LENGTH );
                            STORE_JACOBIAN( i_var_intrinsic_core + 2,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 2] * SCALE_INTRINSICS_CENTER_PIXEL );
                            STORE_JACOBIAN( i_var_intrinsic_core + 3,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 3] * SCALE_INTRINSICS_CENTER_PIXEL );
                        }

                        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                            for(int i=0; i<Ndistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                                dxy_dintrinsic_distortions[i_xy * Ndistortions + i] * SCALE_DISTORTION );

                        if( optimization_variable_choice.do_optimize_extrinsics )
                            if( i_camera != 0 )
                            {
                                STORE_JACOBIAN3( i_var_camera_rt + 0,
                                                 dxy_drcamera[i_xy].xyz[0] * SCALE_ROTATION_CAMERA,
                                                 dxy_drcamera[i_xy].xyz[1] * SCALE_ROTATION_CAMERA,
                                                 dxy_drcamera[i_xy].xyz[2] * SCALE_ROTATION_CAMERA);
                                STORE_JACOBIAN3( i_var_camera_rt + 3,
                                                 dxy_dtcamera[i_xy].xyz[0] * SCALE_TRANSLATION_CAMERA,
                                                 dxy_dtcamera[i_xy].xyz[1] * SCALE_TRANSLATION_CAMERA,
                                                 dxy_dtcamera[i_xy].xyz[2] * SCALE_TRANSLATION_CAMERA);
                            }

                        if( optimization_variable_choice.do_optimize_frames )
                        {
                            STORE_JACOBIAN3( i_var_frame_rt + 0,
                                             dxy_drframe[i_xy].xyz[0] * SCALE_ROTATION_FRAME,
                                             dxy_drframe[i_xy].xyz[1] * SCALE_ROTATION_FRAME,
                                             dxy_drframe[i_xy].xyz[2] * SCALE_ROTATION_FRAME);
                            STORE_JACOBIAN3( i_var_frame_rt + 3,
                                             dxy_dtframe[i_xy].xyz[0] * SCALE_TRANSLATION_FRAME,
                                             dxy_dtframe[i_xy].xyz[1] * SCALE_TRANSLATION_FRAME,
                                             dxy_dtframe[i_xy].xyz[2] * SCALE_TRANSLATION_FRAME);
                        }

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
                            MSG( "%s: obs/frame/cam/dot: %d %d %d %d err: %g",
                                 reportFitMsg,
                                 i_observation_board, i_frame, i_camera, i_pt, err);
                            continue;
                        }

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( optimization_variable_choice.do_optimize_intrinsic_core )
                            for(int i=0; i<N_INTRINSICS_CORE; i++)
                                STORE_JACOBIAN( i_var_intrinsic_core + i, 0.0 );

                        if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                            for(int i=0; i<Ndistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i, 0.0 );

                        if( optimization_variable_choice.do_optimize_extrinsics )
                            if( i_camera != 0 )
                            {
                                STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                                STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                            }

                        if( optimization_variable_choice.do_optimize_frames )
                        {
                            const double dframe = observation->skip_frame ? 1.0 : 0.0;
                            // Arbitrary differences between the dimensions to keep
                            // my Hessian non-singular. This is 100% arbitrary. I'm
                            // skipping these measurements so these variables
                            // actually don't affect the computation at all
                            STORE_JACOBIAN3( i_var_frame_rt + 0, dframe*1.1, dframe*1.2, dframe*1.3);
                            STORE_JACOBIAN3( i_var_frame_rt + 3, dframe*1.4, dframe*1.5, dframe*1.6);
                        }

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
            // unpack the state for this observation as a combination of the
            // state and the seed, depending on what we're optimizing, exactly
            struct intrinsics_core_t* intrinsic_core = &intrinsic_cores[i_camera];
            double* distortions = distortionss[i_camera];
            struct pose_t camera_rt;
            union  point3_t point;

            if(!optimization_variable_choice.do_optimize_intrinsic_core)
                memcpy( intrinsic_core,
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera],
                        N_INTRINSICS_CORE*sizeof(double) );
            if(!optimization_variable_choice.do_optimize_intrinsic_distortions)
                memcpy( distortions,
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera + N_INTRINSICS_CORE],
                        Ndistortions*sizeof(double) );
            if( i_camera != 0 )
            {
                if(optimization_variable_choice.do_optimize_extrinsics)
                    unpack_solver_state_extrinsics_one(&camera_rt, &packed_state[i_var_camera_rt]);
                else
                    memcpy(&camera_rt, &extrinsics[i_camera-1], sizeof(struct pose_t));
            }
            if(optimization_variable_choice.do_optimize_frames)
                unpack_solver_state_point_one(&point, &packed_state[i_var_point]);
            else
                memcpy(&point, &points[i_point], sizeof(union point3_t));


            // Check for invalid points. I report a very poor cost if I see
            // this.
            if( point.z <= 0.0 || point.z >= POINT_MAXZ )
            {
                have_invalid_point = true;
                if(VERBOSE)
                    MSG( "Saw invalid point distance: z = %g! obs/point/cam: %d %d %d",
                         point.z,
                         i_observation_point, i_point, i_camera);
            }

            // these are computed in respect to the unit-scale parameters
            // used by the optimizer
            double dxy_dintrinsic_core       [2 * N_INTRINSICS_CORE];
            double dxy_dintrinsic_distortions[2 * Ndistortions];
            union point3_t dxy_drcamera[2];
            union point3_t dxy_dtcamera[2];
            union point3_t dxy_dpoint  [2];


            // The array reference [-3] is intented, but the compiler throws a
            // warning. I silence it here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            union point2_t pt_hypothesis =
                project(optimization_variable_choice.do_optimize_intrinsic_core ?
                          dxy_dintrinsic_core        : NULL,
                        optimization_variable_choice.do_optimize_intrinsic_distortions ?
                          dxy_dintrinsic_distortions : NULL,

                        optimization_variable_choice.do_optimize_extrinsics ?
                          dxy_drcamera : NULL,
                        optimization_variable_choice.do_optimize_extrinsics ?
                          dxy_dtcamera : NULL,
                        NULL, // frame rotation. I only have a point position
                        optimization_variable_choice.do_optimize_frames ?
                          dxy_dpoint : NULL,
                        intrinsic_core, distortions,
                        &camera_rt,

                        // I only have the point position, so the 'rt' memory
                        // points 3 back. The fake "r" here will not be
                        // referenced
                        (struct pose_t*)(&point.xyz[-3]),

                        i_camera == 0,
                        distortion_model,
                        -1,
                        calibration_object_spacing,
                        calibration_object_width_n);
#pragma GCC diagnostic pop

            const union point2_t* pt_observed = &observation->px;

            if(!observation->skip_observation
#warning "no outlier rejection on points yet; see warning above"
               )
            {
                // I have my two measurements (dx, dy). I propagate their
                // gradient and store them
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy];

                    Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    // I want these gradient values to be computed in
                    // monotonically-increasing order of variable index. I don't
                    // CHECK, so it's the developer's responsibility to make
                    // sure. This ordering is set in pack_solver_state(),
                    // unpack_solver_state()
                    if( optimization_variable_choice.do_optimize_intrinsic_core )
                    {
                        STORE_JACOBIAN( i_var_intrinsic_core + 0,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 0] * SCALE_INTRINSICS_FOCAL_LENGTH );
                        STORE_JACOBIAN( i_var_intrinsic_core + 1,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 1] * SCALE_INTRINSICS_FOCAL_LENGTH );
                        STORE_JACOBIAN( i_var_intrinsic_core + 2,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 2] * SCALE_INTRINSICS_CENTER_PIXEL );
                        STORE_JACOBIAN( i_var_intrinsic_core + 3,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 3] * SCALE_INTRINSICS_CENTER_PIXEL );
                    }

                    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                        for(int i=0; i<Ndistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                            dxy_dintrinsic_distortions[i_xy * Ndistortions + i] * SCALE_DISTORTION );

                    if( optimization_variable_choice.do_optimize_extrinsics )
                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0,
                                             dxy_drcamera[i_xy].xyz[0] * SCALE_ROTATION_CAMERA,
                                             dxy_drcamera[i_xy].xyz[1] * SCALE_ROTATION_CAMERA,
                                             dxy_drcamera[i_xy].xyz[2] * SCALE_ROTATION_CAMERA);
                            STORE_JACOBIAN3( i_var_camera_rt + 3,
                                             dxy_dtcamera[i_xy].xyz[0] * SCALE_TRANSLATION_CAMERA,
                                             dxy_dtcamera[i_xy].xyz[1] * SCALE_TRANSLATION_CAMERA,
                                             dxy_dtcamera[i_xy].xyz[2] * SCALE_TRANSLATION_CAMERA);
                        }

                    if( optimization_variable_choice.do_optimize_frames )
                        STORE_JACOBIAN3( i_var_point,
                                         dxy_dpoint[i_xy].xyz[0] * SCALE_POSITION_POINT,
                                         dxy_dpoint[i_xy].xyz[1] * SCALE_POSITION_POINT,
                                         dxy_dpoint[i_xy].xyz[2] * SCALE_POSITION_POINT);

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
                        double dist = sqrt( point.x*point.x +
                                            point.y*point.y +
                                            point.z*point.z );
                        double err = dist - observation->dist;
                        err *= DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M;

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( optimization_variable_choice.do_optimize_frames )
                        {
                            double scale = DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M / dist * SCALE_POSITION_POINT;
                            STORE_JACOBIAN3( i_var_point,
                                             scale*point.x,
                                             scale*point.y,
                                             scale*point.z );
                        }

                        iMeasurement++;
                    }
                    else
                    {
                        // I need to transform the point. I already computed
                        // this stuff in project()...
                        CvMat rc = cvMat(3,1, CV_64FC1, camera_rt.r.xyz);

                        double _Rc[3*3];
                        CvMat  Rc = cvMat(3,3,CV_64FC1, _Rc);
                        double _d_Rc_rc[9*3];
                        CvMat d_Rc_rc = cvMat(9,3,CV_64F, _d_Rc_rc);
                        cvRodrigues2(&rc, &Rc, &d_Rc_rc);

                        union point3_t pt_cam;
                        mul_vec3_gen33t_vout(point.xyz, _Rc, pt_cam.xyz);
                        add_vec(3, pt_cam.xyz, camera_rt.t.xyz);

                        double dist = sqrt( pt_cam.x*pt_cam.x +
                                            pt_cam.y*pt_cam.y +
                                            pt_cam.z*pt_cam.z );
                        double dist_recip = 1.0/dist;
                        double err = dist - observation->dist;
                        err *= DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M;

                        Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( optimization_variable_choice.do_optimize_extrinsics )
                        {
                            // pt_cam.x       = Rc[row0]*point*SCALE + tc
                            // d(pt_cam.x)/dr = d(Rc[row0])/drc*point*SCALE
                            // d(Rc[row0])/drc is 3x3 matrix at &_d_Rc_rc[0]
                            double d_ptcamx_dr[3];
                            double d_ptcamy_dr[3];
                            double d_ptcamz_dr[3];
                            mul_vec3_gen33_vout( point.xyz, &_d_Rc_rc[9*0], d_ptcamx_dr );
                            mul_vec3_gen33_vout( point.xyz, &_d_Rc_rc[9*1], d_ptcamy_dr );
                            mul_vec3_gen33_vout( point.xyz, &_d_Rc_rc[9*2], d_ptcamz_dr );

                            STORE_JACOBIAN3( i_var_camera_rt + 0,
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M *
                                             SCALE_ROTATION_CAMERA*
                                             dist_recip*( pt_cam.x*d_ptcamx_dr[0] +
                                                          pt_cam.y*d_ptcamy_dr[0] +
                                                          pt_cam.z*d_ptcamz_dr[0] ),
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M *
                                             SCALE_ROTATION_CAMERA*
                                             dist_recip*( pt_cam.x*d_ptcamx_dr[1] +
                                                          pt_cam.y*d_ptcamy_dr[1] +
                                                          pt_cam.z*d_ptcamz_dr[1] ),
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M *
                                             SCALE_ROTATION_CAMERA*
                                             dist_recip*( pt_cam.x*d_ptcamx_dr[2] +
                                                          pt_cam.y*d_ptcamy_dr[2] +
                                                          pt_cam.z*d_ptcamz_dr[2] ) );
                            STORE_JACOBIAN3( i_var_camera_rt + 3,
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_TRANSLATION_CAMERA*
                                             dist_recip*pt_cam.x,
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_TRANSLATION_CAMERA*
                                             dist_recip*pt_cam.y,
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_TRANSLATION_CAMERA*
                                             dist_recip*pt_cam.z );
                        }

                        if( optimization_variable_choice.do_optimize_frames )
                            STORE_JACOBIAN3( i_var_point,
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_POSITION_POINT*
                                             dist_recip*(pt_cam.x*_Rc[0] + pt_cam.y*_Rc[3] + pt_cam.z*_Rc[6]),
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_POSITION_POINT*
                                             dist_recip*(pt_cam.x*_Rc[1] + pt_cam.y*_Rc[4] + pt_cam.z*_Rc[7]),
                                             DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M*
                                             SCALE_POSITION_POINT*
                                             dist_recip*(pt_cam.x*_Rc[2] + pt_cam.y*_Rc[5] + pt_cam.z*_Rc[8]) );
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
                    norm2_error += err*err;

                    if( optimization_variable_choice.do_optimize_intrinsic_core )
                        for(int i=0; i<N_INTRINSICS_CORE; i++)
                            STORE_JACOBIAN( i_var_intrinsic_core + i,
                                            0.0 );

                    if( optimization_variable_choice.do_optimize_intrinsic_distortions )
                        for(int i=0; i<Ndistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                            0.0 );

                    if( optimization_variable_choice.do_optimize_extrinsics )
                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }

                    if( optimization_variable_choice.do_optimize_frames )
                    {
                        const double dpoint = observation->skip_point ? 1.0 : 0.0;
                        // Arbitrary differences between the dimensions to keep
                        // my Hessian non-singular. This is 100% arbitrary. I'm
                        // skipping these measurements so these variables
                        // actually don't affect the computation at all
                        STORE_JACOBIAN3( i_var_point + 0, dpoint*1.1, dpoint*1.2, dpoint*1.3);
                    }

                    iMeasurement++;
                }

                // Now handle the reference distance, if given
                if( observation->dist > 0.0)
                {
                    const double err = 0.0;

                    Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    if( optimization_variable_choice.do_optimize_extrinsics )
                        if(i_camera != 0)
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }
                    if( optimization_variable_choice.do_optimize_frames )
                        STORE_JACOBIAN3( i_var_point, 0.0, 0.0, 0.0);
                    iMeasurement++;
                }
            }
        }

        // regularization terms. I favor smaller distortion parameters
        if(optimization_variable_choice.do_optimize_intrinsic_distortions &&
           !optimization_variable_choice.do_skip_regularization)
        {
            double scale_distortion_regularization =
                ({
                    // I want a "low" value relative to the rest of the cost
                    // function:
                    //
                    //   Nmeasurements_rest*normal_pixel_error =
                    //   Nmeasurements_regularization*normal_regularization_error*scale*LARGE
                    int    Nmeasurements_regularization    = Ncameras*Ndistortions;
                    int    Nmeasurements_nonregularization = Nmeasurements - Nmeasurements_regularization;
                    double normal_pixel_error              = 1.0;

                    // completely made up. Probably should be different for each
                    // distortion term and for each distortion model
                    double normal_regularization_error = 10.0;

                    (double)Nmeasurements_nonregularization*normal_pixel_error/
                        ( (double)Nmeasurements_regularization * normal_regularization_error * 1000.0);
                });

            for(int i_camera=0; i_camera<Ncameras; i_camera++)
            {
                const int i_var_intrinsic_distortions =
                    state_index_intrinsic_distortions(i_camera, optimization_variable_choice, distortion_model);

                for(int j=0; j<Ndistortions; j++)
                {
                    Jrowptr[iMeasurement] = iJacobian;

                    // This is very hoaky. distortion-parameter-0 of a CAHVOR
                    // model is a direction not a "strength" so it shouldn't be
                    // regularized. I really shouldn't be special-casing that
                    // one parameter.
                    if( distortion_model == DISTORTION_CAHVOR && j == 0 )
                    {
                        x[iMeasurement] = 0;
                        STORE_JACOBIAN( i_var_intrinsic_distortions + j, 0 );
                    }
                    else
                    {
                        double err = distortionss[i_camera][j] * scale_distortion_regularization;
                        x[iMeasurement]  = err;
                        norm2_error     += err*err;
                        STORE_JACOBIAN( i_var_intrinsic_distortions + j,
                                        scale_distortion_regularization * SCALE_DISTORTION );
                    }
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

            // this is just for diagnostics. Should probably do this only in a
            // #if of some sort. This sqrt() does no useful work
            MSG_IF_VERBOSE("RMS: %g", sqrt(norm2_error / ((double)Nmeasurements / 2.0)));
        }
    }








    dogleg_solverContext_t*  solver_context = NULL;
    // If I have a context already, I free it and create it anew later. Ideally
    // I'd reuse it, but then I'd need to make sure it's valid and such. Too
    // much work for now
    if(_solver_context != NULL && *_solver_context != NULL)
        dogleg_freeContext((dogleg_solverContext_t**)_solver_context);

    double packed_state[Nstate];
    pack_solver_state(packed_state,
                      intrinsics,
                      distortion_model,
                      extrinsics,
                      frames,
                      points,
                      optimization_variable_choice,
                      Ncameras, Nframes, Npoints, Nstate);

    double norm2_error = -1.0;
    struct mrcal_stats_t stats = {.rms_reproj_error__pixels = -1.0 };

    if( !check_gradient )
    {
        if(VERBOSE)
        {
            reportFitMsg = "Before";
#warning hook this up
            //        optimizerCallback(packed_state, NULL, NULL, NULL);
        }
        reportFitMsg = NULL;

        double getConfidence(int i_exclude_feature)
        {
            return 1.0;
        }


        bool firstpass = true;
        do
        {
            norm2_error = dogleg_optimize(packed_state,
                                          Nstate, Nmeasurements, N_j_nonzero,
                                          &optimizerCallback, NULL, &solver_context);
            if(_solver_context != NULL)
                *_solver_context = solver_context;

            if(firstpass && VERBOSE)
                // These are for debug reporting
                dogleg_reportOutliers(getConfidence,
                                      2, Npoints_fromBoards,
                                      solver_context->beforeStep, solver_context);
            firstpass = false;

        } while( !skip_outlier_rejection &&
                 dogleg_markOutliers(markedOutliers,
                                     &stats.Noutliers,
                                     &stats.mean_outliers, &stats.stdev_outliers,
                                     getConfidence,
                                     2, Npoints_fromBoards,
                                     solver_context->beforeStep, solver_context) );

        // Done. I have the final state. I spit it back out
        unpack_solver_state( intrinsics, // Ncameras of these
                             extrinsics, // Ncameras-1 of these
                             frames,     // Nframes of these
                             points,     // Npoints of these
                             packed_state,
                             distortion_model,
                             optimization_variable_choice,
                             Ncameras, Nframes, Npoints, Nstate);


        if(VERBOSE)
        {
            // These are for debug reporting
            dogleg_reportOutliers(getConfidence,
                                  2, Npoints_fromBoards,
                                  solver_context->beforeStep, solver_context);

            reportFitMsg = "After";
#warning hook this up
            //        optimizerCallback(packed_state, NULL, NULL, NULL);
        }
    }
    else
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, packed_state,
                                Nstate, Nmeasurements, N_j_nonzero,
                                &optimizerCallback, NULL);

    stats.rms_reproj_error__pixels =
        // /2 because I have separate x and y measurements
        sqrt(norm2_error / ((double)Nmeasurements / 2.0));

    if(x_final)
        memcpy(x_final, solver_context->beforeStep->x, Nmeasurements*sizeof(double));

    if( intrinsic_covariances )
    {
        bool result =
            computeConfidence_MMt(// out
                                  // dimensions (Ncameras,Nintrinsics_per_camera,Nintrinsics_per_camera)
                                  intrinsic_covariances,

                                  // in
                                  distortion_model,
                                  optimization_variable_choice,
                                  Ncameras,
                                  NobservationsBoard,
                                  NobservationsPoint,
                                  Nframes, Npoints,
                                  calibration_object_width_n,

                                  solver_context);
        if(!result)
        {
            MSG("Failed to compute MMt.");
            double nan = strtod("NAN", NULL);
            int Nintrinsics_per_camera =
                getNintrinsicOptimizationParams(optimization_variable_choice, distortion_model);
            for(int i=0; i<Ncameras*Nintrinsics_per_camera*Nintrinsics_per_camera; i++)
                intrinsic_covariances[i] = nan;
        }
    }
    if(outlier_indices_final)
    {
        int ioutlier = 0;
        for(int iFeature=0; iFeature<Npoints_fromBoards; iFeature++)
            if( markedOutliers[iFeature].marked )
                outlier_indices_final[ioutlier++] = iFeature;

        assert(ioutlier == stats.Noutliers);
    }

    if(_solver_context == NULL && solver_context)
        dogleg_freeContext(&solver_context);

    free(markedOutliers);
    return stats;
}

bool mrcal_queryIntrinsicOutliernessAt( // output
                                       double* traces,

                                       // input
                                       enum distortion_model_t distortion_model,
                                       bool do_optimize_intrinsic_core,
                                       bool do_optimize_intrinsic_distortions,
                                       int i_camera,

                                       // query vectors (and a count) in the
                                       // camera coord system. We're
                                       // projecting these
                                       const union point3_t* v,
                                       int N,

                                       // context from the solve we just ran.
                                       // I need this for the factorized JtJ
                                       void* _solver_context)
{
    // I add a hypothetical new measurement, projecting a 3d vector v in the
    // coord system of the camera
    //
    //   x = project(v) - observation
    //
    // I compute the projection now, so I know what the observation should be,
    // and I can set it such that x=0 here. If I do that, x fits the existing
    // data perfectly, and is very un-outliery looking.
    //
    // But everything is noisy, so observation will move around, and thus x
    // moves around. I'm assuming the observations are mean-0 gaussian, so I let
    // my x correspondingly also be mean-0 gaussian.
    //
    // I then have a quadratic form outlierness_factor = xt B/Nmeasurements x
    // for some known constant N and known symmetric matrix B. I compute the
    // expected value of this quadratic form: E = tr(B/Nmeasurements * Var(x))
    //
    // I get B from libdogleg. See
    // dogleg_getOutliernessTrace_newFeature_sparse() for a derivation.
    //
    // I'm assuming the noise on the x is independent, so
    //
    //   Var(x) = observed-pixel-uncertainty^2 I
    //
    // And thus E = tr(B) * observed-pixel-uncertainty^2/Nmeasurements
    //
    // I let the caller scale stuff by observed-pixel-uncertainty. I assume it
    // is 1 here.
    if(!do_optimize_intrinsic_core ||
       !do_optimize_intrinsic_distortions)
    {
        MSG("Not implemented unless we're optimizing all the intrinsics; it might work, I just need to think about it");
        return false;
    }


    struct mrcal_variable_select
        optimization_variable_choice = {.do_optimize_intrinsic_core        = do_optimize_intrinsic_core,
                                        .do_optimize_intrinsic_distortions = do_optimize_intrinsic_distortions};
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics = Ndistortions + 4;



    dogleg_solverContext_t*  solver_context = (dogleg_solverContext_t*)_solver_context;
    dogleg_operatingPoint_t* op             = solver_context->beforeStep;
    const double*            p_packed       = op->p;
    int                      Nmeasurements  = solver_context->Nmeasurements;

    int i_intrinsics = state_index_intrinsic_core(i_camera,
                                                  optimization_variable_choice,
                                                  distortion_model);
    double p[Nintrinsics];
    unpack_solver_state_intrinsics_onecamera((struct intrinsics_core_t*)p, &p[4],
                                             &p_packed[i_intrinsics],
                                             Ndistortions, optimization_variable_choice);

    union point2_t* q = malloc(N*sizeof(union point2_t));
    assert(q);

    double* dxy_dintrinsics_all = malloc(2*Nintrinsics*N*sizeof(double));
    assert(dxy_dintrinsics_all);

    mrcal_project(q,
                  dxy_dintrinsics_all, NULL,
                  v, N,
                  distortion_model, p);

    double* dxy_dintrinsics = dxy_dintrinsics_all;
    for(int i=0; i<N; i++)
    {
        // These are the full, unpacked gradients. But the JtJ libdogleg has
        // is in respect to the scaled, packed ones. Here the state is in
        // the denominator, so I call the "unpack" function even though I
        // want to pack the gradients
        unpack_solver_state_intrinsics_onecamera( (struct intrinsics_core_t*)(&dxy_dintrinsics[0]),
                                                  &dxy_dintrinsics[4],
                                                  dxy_dintrinsics, Ndistortions,
                                                  optimization_variable_choice);
        dxy_dintrinsics = &dxy_dintrinsics[Nintrinsics];

        unpack_solver_state_intrinsics_onecamera( (struct intrinsics_core_t*)(&dxy_dintrinsics[0]),
                                                  &dxy_dintrinsics[4],
                                                  dxy_dintrinsics, Ndistortions,
                                                  optimization_variable_choice);
        dxy_dintrinsics = &dxy_dintrinsics[Nintrinsics];


        traces[i] =
            dogleg_getOutliernessTrace_newFeature_sparse(&dxy_dintrinsics[-2*Nintrinsics],
                                                         i_intrinsics, Nintrinsics, 2,
                                                         op, solver_context) / (double)Nmeasurements;

    }

    free(q);
    free(dxy_dintrinsics_all);

    return true;
}

void mrcal_free_context(void** ctx)
{
    if( *ctx == NULL )
        return;

    dogleg_freeContext((dogleg_solverContext_t**)ctx);
}
