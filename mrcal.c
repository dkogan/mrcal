#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>

#include <dogleg.h>
#include <minimath.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>


// This is a workaround for OpenCV's stupidity: they decided to break their C
// API. Without this you get undefined references to cvRound() if you build
// without optimizations.
static inline int cvRound(float value)
{
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
}

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
#define SCALE_INTRINSICS_FOCAL_LENGTH 500.0
#define SCALE_INTRINSICS_CENTER_PIXEL 20.0
#define SCALE_ROTATION_CAMERA         (0.1 * M_PI/180.0)
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          (15.0 * M_PI/180.0)
#define SCALE_TRANSLATION_FRAME       100.0
#define SCALE_POSITION_POINT          SCALE_TRANSLATION_FRAME
#define SCALE_CALOBJECT_WARP          0.01

#define DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M 1.0

// I need to constrain the point motion since it's not well-defined, and can
// jump to clearly-incorrect values. This is the distance in front of camera0. I
// make sure this is positive and not unreasonably high
#define POINT_MAXZ                    50000

// This is hard-coded to 1.0; the computation of scale_distortion_regularization
// below assumes it
#define SCALE_DISTORTION              1.0

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define MSG_IF_VERBOSE(...) do { if(VERBOSE) MSG( __VA_ARGS__ ); } while(0)



const char* mrcal_distortion_model_name( distortion_model_t model )
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
distortion_model_t mrcal_distortion_model_from_name( const char* name )
{
#define CHECK_AND_RETURN(s,n) if( 0 == strcmp( name, #s) ) return s;
    DISTORTION_LIST( CHECK_AND_RETURN );

    return DISTORTION_INVALID;
}

int mrcal_getNdistortionParams(const distortion_model_t m)
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

// Returns the 'next' distortion model in a family
//
// In a family of distortion models we have a sequence of models with increasing
// complexity. Subsequent models add distortion parameters to the end of the
// vector. Ealier models are identical, but with the extra paramaters set to 0.
// This function returns the next model in a sequence.
//
// If this is the last model in the sequence, returns the current model. This
// function takes in both the current model, and the last model we're aiming
// for. The second part is required because all familie begin at
// DISTORTION_NONE, so the next model from DISTORTION_NONE is not well-defined
// without more information
distortion_model_t mrcal_getNextDistortionModel( distortion_model_t distortion_model_now,
                                                 distortion_model_t distortion_model_final )
{
    // if we're at the start of a sequence...
    if(distortion_model_now == DISTORTION_NONE)
    {
        if(DISTORTION_IS_OPENCV(distortion_model_final)) return DISTORTION_OPENCV4;
        if(DISTORTION_IS_CAHVOR(distortion_model_final)) return DISTORTION_CAHVOR;
        return DISTORTION_INVALID;
    }

    // if we're at the end of a sequence...
    if(distortion_model_now == distortion_model_final)
        return distortion_model_now;

    // I guess we're in the middle of a sequence
    return distortion_model_now+1;
}

static
int getNdistortionOptimizationParams(mrcal_problem_details_t problem_details,
                                     distortion_model_t distortion_model)
{
    if( !problem_details.do_optimize_intrinsic_distortions )
        return 0;

    int N = mrcal_getNdistortionParams(distortion_model);
    if( !problem_details.do_optimize_cahvor_optical_axis &&
        ( distortion_model == DISTORTION_CAHVOR ||
          distortion_model == DISTORTION_CAHVORE ))
    {
        // no optical axis parameters
        N -= 2;
    }
    return N;
}

int mrcal_getNintrinsicParams(distortion_model_t m)
{
    return
        N_INTRINSICS_CORE +
        mrcal_getNdistortionParams(m);
}
static int getNintrinsicOptimizationParams(mrcal_problem_details_t problem_details,
                                           distortion_model_t distortion_model)
{
    int N = getNdistortionOptimizationParams(problem_details, distortion_model);

    if( problem_details.do_optimize_intrinsic_core )
        N += N_INTRINSICS_CORE;
    return N;
}
int mrcal_getNintrinsicOptimizationParams(mrcal_problem_details_t problem_details,
                                          distortion_model_t distortion_model)
{
    return getNintrinsicOptimizationParams(problem_details,
                                           distortion_model);
}

static int get_Nstate(int Ncameras, int Nframes, int Npoints,
                      mrcal_problem_details_t problem_details,
                      distortion_model_t distortion_model)
{
    return
        // camera extrinsics
        (problem_details.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +

        // frame poses, individual observed points
        (problem_details.do_optimize_frames ? (Nframes * 6 + Npoints * 3) : 0) +

        // camera intrinsics
        (Ncameras * getNintrinsicOptimizationParams(problem_details, distortion_model)) +

        // warp
        (problem_details.do_optimize_calobject_warp ? 2 : 0);
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

static int getNregularizationTerms_percamera(mrcal_problem_details_t problem_details,
                                             distortion_model_t distortion_model)
{
    if(problem_details.do_skip_regularization)
        return 0;

    // distortions
    int N = getNdistortionOptimizationParams(problem_details, distortion_model);
    // optical center
    if(problem_details.do_optimize_intrinsic_core)
        N += 2;
    return N;
}

int mrcal_getNmeasurements_boards(int NobservationsBoard,
                                  int calibration_object_width_n)
{
    // *2 because I have separate x and y measurements
    return
        NobservationsBoard *
        calibration_object_width_n*calibration_object_width_n *
        2;
}

int mrcal_getNmeasurements_points(const observation_point_t* observations_point,
                                  int NobservationsPoint)
{
    // *2 because I have separate x and y measurements
    int Nmeas = NobservationsPoint * 2;

    // known-distance measurements
    for(int i=0; i<NobservationsPoint; i++)
        if(observations_point[i].dist > 0.0) Nmeas++;
    return Nmeas;
}

int mrcal_getNmeasurements_regularization(int Ncameras,
                                          mrcal_problem_details_t problem_details,
                                          distortion_model_t distortion_model)
{
    return
        Ncameras *
        getNregularizationTerms_percamera(problem_details, distortion_model);
}

int mrcal_getNmeasurements_all(int Ncameras, int NobservationsBoard,
                               const observation_point_t* observations_point,
                               int NobservationsPoint,
                               int calibration_object_width_n,
                               mrcal_problem_details_t problem_details,
                               distortion_model_t distortion_model)
{
    return
        mrcal_getNmeasurements_boards( NobservationsBoard, calibration_object_width_n) +
        mrcal_getNmeasurements_points( observations_point, NobservationsPoint) +
        mrcal_getNmeasurements_regularization( Ncameras, problem_details, distortion_model);
}

static int get_N_j_nonzero( int Ncameras,
                            const observation_board_t* observations_board,
                            int NobservationsBoard,
                            const observation_point_t* observations_point,
                            int NobservationsPoint,
                            mrcal_problem_details_t problem_details,
                            distortion_model_t distortion_model,
                            int calibration_object_width_n)
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // initial estimate counts extrinsics for camera0, which need to be
    // subtracted off
    int Nintrinsics = getNintrinsicOptimizationParams(problem_details, distortion_model);
    int N = NobservationsBoard * ( (problem_details.do_optimize_frames         ? 6 : 0) +
                                   (problem_details.do_optimize_extrinsics     ? 6 : 0) +
                                   (problem_details.do_optimize_calobject_warp ? 2 : 0) +
                                   + Nintrinsics );
    if(problem_details.do_optimize_extrinsics)
        for(int i=0; i<NobservationsBoard; i++)
            if(observations_board[i].i_camera == 0)
                N -= 6;
    N *= 2*calibration_object_width_n*calibration_object_width_n; // *2 because I have separate x and y measurements

    // Now the point observations
    for(int i=0; i<NobservationsPoint; i++)
    {
        N += 2*Nintrinsics;
        if(problem_details.do_optimize_frames)
            N += 2*3;
        if( problem_details.do_optimize_extrinsics &&
            observations_point[i].i_camera != 0 )
            N += 2*6;

        if(observations_point[i].dist > 0)
        {
            if(problem_details.do_optimize_frames)
                N += 3;

            if( problem_details.do_optimize_extrinsics &&
                observations_point[i].i_camera != 0 )
                N += 6;
        }
    }

    N +=
        Ncameras *
        getNregularizationTerms_percamera(problem_details,
                                          distortion_model);

    return N;
}

// Projects ONE 3D point, and reports the projection, and all the gradients.
// This is the main internal callback in the optimizer. If i_pt>0, then we're
// looking at the i_pt-th point in the calibration target, with the target pose
// in frame_rt. Otherwise we're looking at a single point at frame_rt->t;
// frame_rt->r is then not referenced
static point2_t project( // out
                         double*   restrict dxy_dintrinsic_core,
                         double*   restrict dxy_dintrinsic_distortions,
                         point3_t* restrict dxy_drcamera,
                         point3_t* restrict dxy_dtcamera,
                         point3_t* restrict dxy_drframe,
                         point3_t* restrict dxy_dtframe,
                         point2_t* restrict dxy_dcalobject_warp,

                         // in
                         const intrinsics_core_t* intrinsics_core,
                         const double* restrict distortions,
                         const pose_t* restrict camera_rt,
                         const pose_t* restrict frame_rt,
                         const point2_t* restrict calobject_warp,

                         bool camera_at_identity, // if true, camera_rt is unused
                         distortion_model_t distortion_model,

                         // point index. If <0, a point at frame_rt->t is
                         // assumed; frame_rt->r isn't referenced, and
                         // dxy_drframe is expected to be NULL. And the
                         // calibration_object_... variables aren't used either
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

    point3_t pt_ref = {};
    point2_t dpt_ref2_dwarp = {};

    double r = 1./(double)(calibration_object_width_n-1);
    if(i_pt >= 0)
    {
        // The calibration object has a simple grid geometry
        int y = i_pt / calibration_object_width_n;
        int x = i_pt - y*calibration_object_width_n;
        pt_ref.x = (double)x * calibration_object_spacing;
        pt_ref.y = (double)y * calibration_object_spacing;
        // pt_ref.z = 0.0; This is already done

        if(calobject_warp != NULL)
        {
            // Add a board warp here. I have two parameters, and they describe
            // additive flex along the x axis and along the y axis, in that
            // order. In each direction the flex is a parabola, with the
            // parameter k describing the max deflection at the center. If the
            // ends are at +- 1 I have d = k*(1 - x^2). If the ends are at
            // (0,N-1) the equivalent expression is: d = k*( 1 - 4*x^2/(N-1)^2 +
            // 4*x/(N-1) - 1 ) = d = 4*k*(x/(N-1) - x^2/(N-1)^2) = d =
            // 4.*k*x*r(1. - x*r)
            double xr = (double)x * r;
            double yr = (double)y * r;
            double dx = 4. * xr * (1. - xr);
            double dy = 4. * yr * (1. - yr);
            pt_ref.z += calobject_warp->x * dx;
            pt_ref.z += calobject_warp->y * dy;
            dpt_ref2_dwarp.x = dx;
            dpt_ref2_dwarp.y = dy;
        }
    }
    else
    {
        // We're not looking at a calibration board point, but rather a
        // standalone point. I leave pt_ref at the origin, and take the
        // coordinate from frame_rt->t
    }

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


    if( DISTORTION_IS_OPENCV(distortion_model) )
    {
        // OpenCV does the projection AND the gradient propagation for me, so I
        // implement a separate code path for it
        point2_t pt_out;

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
        //                                CV_64FC1, ((intrinsics_core_t*)dxy_dintrinsic_core)->focal_xy);
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
            intrinsics_core_t* dxy_dintrinsics0 = (intrinsics_core_t*)dxy_dintrinsic_core;
            intrinsics_core_t* dxy_dintrinsics1 = (intrinsics_core_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

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
                               point3_t* dxy_dparam,

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
        if( dxy_dcalobject_warp != NULL && i_pt >= 0)
        {
            // p = proj(R( warp(x) ) + t);
            // dp/dw = dp/dR(warp(x)) dR(warp(x))/dwarp(x) dwarp/dw =
            //       = dp/dt R dwarp/dw
            // dp/dt is _dxy_dtj
            // R is rodrigues(rj)
            // dwarp/dw = [0 0]
            //            [0 0]
            //            [a b]
            // Let R = [r0 r1 r2]
            // dp/dw = dp/dt [ar2 br2] = [a dp/dt r2    b dp/dt r2]

            double _Rj[3*3];
            CvMat  Rj = cvMat(3,3,CV_64FC1, _Rj);
            cvRodrigues2(p_rj, &Rj, NULL);

            double d[] =
                { _dxy_dtj[3*0 + 0] * _Rj[0*3 + 2] +
                  _dxy_dtj[3*0 + 1] * _Rj[1*3 + 2] +
                  _dxy_dtj[3*0 + 2] * _Rj[2*3 + 2],
                  _dxy_dtj[3*1 + 0] * _Rj[0*3 + 2] +
                  _dxy_dtj[3*1 + 1] * _Rj[1*3 + 2] +
                  _dxy_dtj[3*1 + 2] * _Rj[2*3 + 2]};

            dxy_dcalobject_warp[0].x = d[0]*dpt_ref2_dwarp.x;
            dxy_dcalobject_warp[0].y = d[0]*dpt_ref2_dwarp.y;
            dxy_dcalobject_warp[1].x = d[1]*dpt_ref2_dwarp.x;
            dxy_dcalobject_warp[1].y = d[1]*dpt_ref2_dwarp.y;
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
    point3_t pt_cam;
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
        //   alpha
        //   beta
        //   r0
        //   r1
        //   r2
        double alpha = distortions[0];
        double beta  = distortions[1];
        double r0    = distortions[2];
        double r1    = distortions[3];
        double r2    = distortions[4];

        double s_al, c_al, s_be, c_be;
        sincos(alpha, &s_al, &c_al);
        sincos(beta,  &s_be, &c_be);

        // I parametrize the optical axis such that
        // - o(alpha=0, beta=0) = (0,0,1) i.e. the optical axis is at the center
        //   if both parameters are 0
        // - The gradients are cartesian. I.e. do/dalpha and do/dbeta are both
        //   NOT 0 at (alpha=0,beta=0). This would happen at the poles (gimbal
        //   lock), and that would make my solver unhappy
        double o     []         = {  s_al*c_be, s_be,  c_al*c_be };
        double do_dalpha[]      = {  c_al*c_be,    0, -s_al*c_be };
        double do_dbeta[]       = { -s_al*s_be, c_be, -c_al*s_be };

        double norm2p        = norm2_vec(3, pt_cam.xyz);
        double omega         = dot_vec(3, pt_cam.xyz, o);
        double domega_dalpha = dot_vec(3, pt_cam.xyz, do_dalpha);
        double domega_dbeta  = dot_vec(3, pt_cam.xyz, do_dbeta);

        double omega_recip = 1.0 / omega;
        double tau         = norm2p * omega_recip*omega_recip - 1.0;
        double s__dtau_dalphabeta__domega_dalphabeta = -2.0*norm2p * omega_recip*omega_recip*omega_recip;
        double dmu_dtau = r1 + 2.0*tau*r2;
        double dmu_dxyz[3];
        for(int i=0; i<3; i++)
            dmu_dxyz[i] = dmu_dtau *
                (2.0 * pt_cam.xyz[i] * omega_recip*omega_recip + s__dtau_dalphabeta__domega_dalphabeta * o[i]);
        double mu = r0 + tau*r1 + tau*tau*r2;
        double s__dmu_dalphabeta__domega_dalphabeta = dmu_dtau * s__dtau_dalphabeta__domega_dalphabeta;

        for(int i=0; i<3; i++)
        {
            double dmu_ddist[5] = { s__dmu_dalphabeta__domega_dalphabeta * domega_dalpha,
                                    s__dmu_dalphabeta__domega_dalphabeta * domega_dbeta,
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

            dxyz_ddistortion[i*NdistortionParams + 0] -= mu * domega_dalpha*o[i];
            dxyz_ddistortion[i*NdistortionParams + 1] -= mu * domega_dbeta *o[i];

            dxyz_ddistortion[i*NdistortionParams + 0] -= mu * omega * do_dalpha[i];
            dxyz_ddistortion[i*NdistortionParams + 1] -= mu * omega * do_dbeta [i];


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




    point2_t pt_out;
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
        intrinsics_core_t* dxy_dintrinsics0 = (intrinsics_core_t*)dxy_dintrinsic_core;
        intrinsics_core_t* dxy_dintrinsics1 = (intrinsics_core_t*)&dxy_dintrinsic_core[N_INTRINSICS_CORE];

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
        void propagate(point3_t* dxy_dparam,
                       const double* _d_rj_dparam,
                       const double* _d_tj_dparam)
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
                mul_vec3_gen33     ( &d_ptcam[3*i],   _d_rj_dparam);
                add_vec(3, &d_ptcam[3*i], &_d_tj_dparam[3*i] );
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
        void propagate_r(point3_t* dxy_dparam)
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
        void propagate_t(point3_t* dxy_dparam)
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

    if( dxy_dcalobject_warp != NULL && i_pt >= 0)
    {
        assert(0);

        // not yet implemented

#if 0
        // p = proj(R( warp(x) ) + t);
        // dp/dw = dp/dR(warp(x)) dR(warp(x))/dwarp(x) dwarp/dw =
        //       = dp/dt R dwarp/dw
        // dp/dt is _dxy_dtj
        // R is rodrigues(rj)
        // dwarp/dw = [0 0]
        //            [0 0]
        //            [a b]
        // Let R = [r0 r1 r2]
        // dp/dw = dp/dt [ar2 br2] = [a dp/dt r2    b dp/dt r2]
        double d[] =
            { _dxy_dtj[3*0 + 0] * _Rj[0*3 + 2] +
              _dxy_dtj[3*0 + 1] * _Rj[1*3 + 2] +
              _dxy_dtj[3*0 + 2] * _Rj[2*3 + 2],
              _dxy_dtj[3*1 + 0] * _Rj[0*3 + 2] +
              _dxy_dtj[3*1 + 1] * _Rj[1*3 + 2] +
              _dxy_dtj[3*1 + 2] * _Rj[2*3 + 2]};

        dxy_dcalobject_warp[0].x = d[0]*dpt_ref2_dwarp.x;
        dxy_dcalobject_warp[0].y = d[0]*dpt_ref2_dwarp.y;
        dxy_dcalobject_warp[1].x = d[1]*dpt_ref2_dwarp.x;
        dxy_dcalobject_warp[1].y = d[1]*dpt_ref2_dwarp.y;
#endif
    }

    return pt_out;
}

// Compute the region-of-interest weight. The region I care about is in r=[0,1];
// here the weight is ~ 1. Past that, the weight falls off. I don't attenuate
// all the way to 0 to preserve the constraints of the problem. Letting these go
// to 0 could make the problem indeterminate
static double region_of_interest_weight_from_unitless_rad(double rsq)
{
    if( rsq < 1.0 ) return 1.0;
    return 1e-3;
}
static double region_of_interest_weight(const point2_t* pt,
                                        const double* roi, int i_camera)
{
    if(roi == NULL) return 1.0;

    roi = &roi[4*i_camera];
    double dx = (pt->x - roi[0]) / roi[2];
    double dy = (pt->y - roi[1]) / roi[3];

    return region_of_interest_weight_from_unitless_rad(dx*dx + dy*dy);
}

// external function. Mostly a wrapper around project()
void mrcal_project( // out
                   point2_t* out,

                   // core, distortions concatenated. Stored as a row-first
                   // array of shape (N,2,Nintrinsics)
                   double*         dxy_dintrinsics,
                   // Stored as a row-first array of shape (N,2,3). Each
                   // trailing ,3 dimension element is a point3_t
                   point3_t* dxy_dp,

                   // in
                   const point3_t* p,
                   int N,
                   distortion_model_t distortion_model,
                   // core, distortions concatenated
                   const double* intrinsics)
{
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + 4;

    // Special-case for opencv/pinhole and projection-only. cvProjectPoints2 and
    // project() have a lot of overhead apparently, and calling either in a loop
    // is very slow. I can call it once, and use its fast internal loop,
    // however. This special case does the same thing, but much faster.

    if(dxy_dintrinsics == NULL && dxy_dp == NULL &&
       (DISTORTION_IS_OPENCV(distortion_model) ||
        distortion_model == DISTORTION_NONE))
    {
        const intrinsics_core_t* intrinsics_core = (const intrinsics_core_t*)intrinsics;
        double fx = intrinsics_core->focal_xy [0];
        double fy = intrinsics_core->focal_xy [1];
        double cx = intrinsics_core->center_xy[0];
        double cy = intrinsics_core->center_xy[1];
        double _camera_matrix[] =
            { fx,  0, cx,
              0, fy, cy,
              0,  0,  1 };
        CvMat camera_matrix = cvMat(3,3, CV_64FC1, _camera_matrix);

        int NdistortionParams = mrcal_getNdistortionParams(distortion_model);
        CvMat _distortions;
        if(NdistortionParams > 0)
            _distortions = cvMat( NdistortionParams, 1, CV_64FC1,
                                  // removing const, but that's just because
                                  // OpenCV's API is incomplete. It IS const
                                  (double*)&intrinsics[4]);

        CvMat object_points  = cvMat(3,N, CV_64FC1, (double*)p  ->xyz);
        CvMat image_points   = cvMat(2,N, CV_64FC1, (double*)out->xy);
        double _zero3[3] = {};
        CvMat zero3 = cvMat(3,1,CV_64FC1, _zero3);
        cvProjectPoints2(&object_points,
                         &zero3, &zero3,
                         &camera_matrix,
                         NdistortionParams > 0 ? &_distortions : NULL,
                         &image_points,
                         NULL, NULL, NULL, NULL, NULL, 0 );
        return;
    }


    for(int i=0; i<N; i++)
    {
        pose_t frame = {.r = {},
                        .t = p[i]};

        // The data is laid out differently in mrcal_project() and project(), so
        // I need to project() into these temporary variables, and then populate
        // my output array
        double dxy_dintrinsic_core       [2*4];
        double dxy_dintrinsic_distortions[2*Ndistortions];

        out[i] = project( dxy_dintrinsics != NULL ? dxy_dintrinsic_core        : NULL,
                          dxy_dintrinsics != NULL ? dxy_dintrinsic_distortions : NULL,
                          NULL, NULL, NULL,
                          dxy_dp, NULL,

                          // in
                          (const intrinsics_core_t*)(&intrinsics[0]),
                          &intrinsics[4],
                          NULL,
                          &frame,
                          NULL,
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
            dxy_dp = &dxy_dp[2];
    }
}

// For the other ..._distort() functions I reuse project(). But since project()
// doesn't support CAHVORE, I need to special-case it here
static
bool cahvore_distort( // out
                      point2_t* out,

                      // in
                      const point2_t* q,
                      int N,

                      const intrinsics_core_t* core,
                      const double*            distortions,
                      const double             fx_pinhole,
                      const double             fy_pinhole,
                      const double             cx_pinhole,
                      const double             cy_pinhole)
{
    // Apply a CAHVORE warp to an un-distorted point

    //  Given intrinsic parameters of a CAHVORE model and a pinhole-projected
    //  point(s) numpy array of shape (..., 2), return the projected point(s) that
    //  we'd get with distortion. By default we assume the same fx,fy,cx,cy. A scale
    //  parameter allows us to scale the size of the output image by scaling the
    //  focal lengths

    // This comes from cmod_cahvore_3d_to_2d_general() in
    // m-jplv/libcmod/cmod_cahvore.c
    //
    // The lack of documentation here comes directly from the lack of
    // documentation in that function.

    // I parametrize the optical axis such that
    // - o(alpha=0, beta=0) = (0,0,1) i.e. the optical axis is at the center
    //   if both parameters are 0
    // - The gradients are cartesian. I.e. do/dalpha and do/dbeta are both
    //   NOT 0 at (alpha=0,beta=0). This would happen at the poles (gimbal
    //   lock), and that would make my solver unhappy
    // So o = { s_al*c_be, s_be,  c_al*c_be }
    const double alpha     = distortions[0];
    const double beta      = distortions[1];
    const double r0        = distortions[2];
    const double r1        = distortions[3];
    const double r2        = distortions[4];
    const double e0        = distortions[5];
    const double e1        = distortions[6];
    const double e2        = distortions[7];
    const double linearity = distortions[8];

    for(int i=0; i<N; i++)
    {
        // q is a 2d point. Convert to a 3d point by unprojecting using a
        // pinhole model
        double v[] = { (q[i].x - cx_pinhole) / fx_pinhole,
                       (q[i].y - cy_pinhole) / fy_pinhole,
                       1.0};

        double sa,ca;
        sincos(alpha, &sa, &ca);
        double sb,cb;
        sincos(beta, &sb, &cb);

        double o[] ={ cb * sa, sb, cb * ca };

        // cos( angle between v and o ) = inner(v,o) / (norm(o) * norm(v)) =
        // omega/norm(v)
        double omega = v[0]*o[0] + v[1]*o[1] + o[2];


        // Basic Computations

        // Calculate initial terms
        double u[3];
        for(int i=0; i<3; i++) u[i] = omega*o[i];

        double ll[3];
        for(int i=0; i<3; i++) ll[i] = v[i]-u[i];
        double l  = sqrt(ll[0]*ll[0] + ll[1]*ll[1] + ll[2]*ll[2]);

        // Calculate theta using Newton's Method
        double theta = atan2(l, omega);

        int inewton;
        for( inewton = 100; inewton; inewton--)
        {
            // Compute terms from the current value of theta
            double sth,cth;
            sincos(theta, &sth, &cth);

            double theta2  = theta * theta;
            double theta3  = theta * theta2;
            double theta4  = theta * theta3;
            double upsilon =
                omega*cth + l*sth
                - (1.0   - cth) * (e0 +      e1*theta2 +     e2*theta4)
                - (theta - sth) * (      2.0*e1*theta  + 4.0*e2*theta3);

            // Update theta
            double dtheta =
                (
                 omega*sth - l*cth
                 - (theta - sth) * (e0 + e1*theta2 + e2*theta4)
                 ) / upsilon;

            theta -= dtheta;

            // Check exit criterion from last update
            if(fabs(dtheta) < 1e-8)
                break;
        }
        if(inewton == 0)
        {
            fprintf(stderr, "%s(): too many iterations\n", __func__);
            return false;
        }

        // got a theta

        // Check the value of theta
        if(theta * fabs(linearity) > M_PI/2.)
        {
            fprintf(stderr, "%s(): theta out of bounds\n", __func__);
            return false;
        }

        // If we aren't close enough to use the small-angle approximation ...
        if (theta > 1e-8)
        {
            // ... do more math!
            double linth = linearity * theta;
            double chi;
            if (linearity < -1e-15)
                chi = sin(linth) / linearity;
            else if (linearity > 1e-15)
                chi = tan(linth) / linearity;
            else
                chi = theta;

            double chi2 = chi * chi;
            double chi3 = chi * chi2;
            double chi4 = chi * chi3;

            double zetap = l / chi;

            double mu = r0 + r1*chi2 + r2*chi4;

            double uu[3];
            for(int i=0; i<3; i++) uu[i] = zetap*o[i];
            double vv[3];
            for(int i=0; i<3; i++) vv[i] = (1. + mu)*ll[i];

            for(int i=0; i<3; i++)
                v[i] = uu[i] + vv[i];
        }

        // now I apply a normal projection to the warped 3d point v
        out[i].x = core->focal_xy[0] * v[0]/v[2] + core->center_xy[0];
        out[i].y = core->focal_xy[1] * v[1]/v[2] + core->center_xy[1];
    }
    return true;
}


// Maps a set of undistorted 2D imager points q to a set of imager points that
// would result from observing the same vectors with a distorted model. Here the
// undistorted model is a pinhole camera with the given parameters. Any of these
// pinhole parameters can be given as <= 0, in which case the corresponding
// parameter from the distorted model will be used
bool mrcal_distort( // out
                   point2_t* out,

                   // in
                   const point2_t* q,
                   int N,
                   distortion_model_t distortion_model,
                   // core, distortions concatenated
                   const double* intrinsics,
                   double fx_pinhole,
                   double fy_pinhole,
                   double cx_pinhole,
                   double cy_pinhole)
{
    const intrinsics_core_t* core =
        (const intrinsics_core_t*)intrinsics;
    const double* distortions = &intrinsics[4];

    if(fx_pinhole <= 0) fx_pinhole = core->focal_xy [0];
    if(fy_pinhole <= 0) fy_pinhole = core->focal_xy [1];
    if(cx_pinhole <= 0) cx_pinhole = core->center_xy[0];
    if(cy_pinhole <= 0) cy_pinhole = core->center_xy[1];

    // project() doesn't handle cahvore, so I special-case it here
    if( distortion_model == DISTORTION_CAHVORE )
        return cahvore_distort( out, q, N, core, distortions,
                                fx_pinhole, fy_pinhole, cx_pinhole, cy_pinhole );

    pose_t frame = {.r = {},
                    .t = {.z = 1.0}};
    for(int i=0; i<N; i++)
    {
        // q is a 2d point. Convert to a 3d point by unprojecting using a
        // pinhole model
        frame.t.x = (q[i].x - cx_pinhole) / fx_pinhole;
        frame.t.y = (q[i].y - cy_pinhole) / fy_pinhole;
        // initializing this above: frame.t[2] = 1.0;

        out[i] = project( NULL, NULL,
                          NULL, NULL, NULL,
                          NULL, NULL,

                          // in
                          core, distortions,
                          NULL,
                          &frame,
                          NULL,
                          true,
                          distortion_model,

                          -1, 0.0, 0);
    }

    return true;
}


// Maps a set of distorted 2D imager points q to a set of imager points that
// would result from observing the same vectors with an undistorted model. Here the
// undistorted model is a pinhole camera with the given parameters. Any of these
// pinhole parameters can be given as <= 0, in which case the corresponding
// parameter from the distorted model will be used
//
// This is the "reverse" direction, so we need a nonlinear optimization to compute
// this result. OpenCV has cvUndistortPoints() (and cv2.undistortPoints()), but
// these are inaccurate: https://github.com/opencv/opencv/issues/8811
//
// This function does this precisely AND supports distortions other than OpenCV's
bool mrcal_undistort( // out
                     point2_t* out,

                     // in
                     const point2_t* q,
                     int N,
                     distortion_model_t distortion_model,
                     // core, distortions concatenated
                     const double* intrinsics,
                     double fx_pinhole,
                     double fy_pinhole,
                     double cx_pinhole,
                     double cy_pinhole)
{
    const intrinsics_core_t* core =
        (const intrinsics_core_t*)intrinsics;
    const double* distortions = &intrinsics[4];

    if(fx_pinhole <= 0) fx_pinhole = core->focal_xy [0];
    if(fy_pinhole <= 0) fy_pinhole = core->focal_xy [1];
    if(cx_pinhole <= 0) cx_pinhole = core->center_xy[0];
    if(cy_pinhole <= 0) cy_pinhole = core->center_xy[1];

    const double fx_recip_distort = 1.0 / core->focal_xy[0];
    const double fy_recip_distort = 1.0 / core->focal_xy[1];
    const double fx_recip_pinhole = 1.0 / fx_pinhole;
    const double fy_recip_pinhole = 1.0 / fy_pinhole;

    // easy special-case
    if( distortion_model == DISTORTION_NONE )
    {
        for(int i=0; i<N; i++)
        {
            double x = (q[i].x - core->focal_xy[0]) * fx_recip_distort;
            double y = (q[i].y - core->focal_xy[1]) * fy_recip_distort;
            out[i].x = x*fx_recip_pinhole + cx_pinhole;
            out[i].y = y*fy_recip_pinhole + cy_pinhole;
        }
        return true;
    }

    if( distortion_model == DISTORTION_CAHVORE )
    {
        fprintf(stderr, "mrcal_undistort(DISTORTION_CAHVORE) not yet implemented. No gradients available\n");
        return false;
    }


    pose_t frame = {.r = {},
                    .t = {.z = 1.0}};

    for(int i=0; i<N; i++)
    {
        void cb(const double*   xy,
                double*         x,
                double*         J,
                void*           cookie __attribute__((unused)))
        {
            // I want q[i] == distort(xy)

            frame.t.x = (xy[0] - core->center_xy[0]) * fx_recip_distort;
            frame.t.y = (xy[1] - core->center_xy[1]) * fy_recip_distort;
            // initializing this above: frame.t.z = 1.0;

            point3_t dxy_dtframe[2];
            point2_t q_hypothesis =
                project( NULL, NULL,
                         NULL, NULL, NULL, dxy_dtframe,
                         NULL,

                         // in
                         core, distortions,
                         NULL,
                         &frame,
                         NULL,
                         true,
                         distortion_model,

                         -1, 0.0, 0);
            x[0] = q_hypothesis.x - q[i].x;
            x[1] = q_hypothesis.y - q[i].y;
            J[0*2 + 0] = dxy_dtframe[0].x*fx_recip_distort;
            J[0*2 + 1] = dxy_dtframe[0].y*fy_recip_distort;
            J[1*2 + 0] = dxy_dtframe[1].x*fx_recip_distort;
            J[1*2 + 1] = dxy_dtframe[1].y*fy_recip_distort;
        }

        out[i]= q[i]; // seed from the distorted value
        double norm2x =
            dogleg_optimize_dense(out[i].xy, 2, 2, cb, NULL, NULL);
    }

    return true;
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
//   calobject_warp0
//   calobject_warp1

// From real values to unit-scale values. Optimizer sees unit-scale values
static int pack_solver_state_intrinsics( // out
                                         double* p,

                                         // in
                                         const double* intrinsics, // each camera slice is (N_INTRINSICS_CORE, distortions)
                                         const distortion_model_t distortion_model,
                                         mrcal_problem_details_t problem_details,
                                         int Ncameras )
{
    int i_state      = 0;
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        if( problem_details.do_optimize_intrinsic_core )
        {
            const intrinsics_core_t* intrinsics_core = (const intrinsics_core_t*)intrinsics;
            p[i_state++] = intrinsics_core->focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
            p[i_state++] = intrinsics_core->center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( problem_details.do_optimize_intrinsic_distortions )

            for(int i = ( !problem_details.do_optimize_cahvor_optical_axis &&
                          ( distortion_model == DISTORTION_CAHVOR ||
                            distortion_model == DISTORTION_CAHVORE )) ? 2 : 0;
                i<Ndistortions;
                i++)
            {
                p[i_state++] = intrinsics[N_INTRINSICS_CORE + i] / SCALE_DISTORTION;
            }

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
                              const distortion_model_t distortion_model,
                              const pose_t*            extrinsics, // Ncameras-1 of these
                              const pose_t*            frames,     // Nframes of these
                              const point3_t*          points,     // Npoints of these
                              const point2_t*          calobject_warp, // 1 of these
                              mrcal_problem_details_t problem_details,
                              int Ncameras, int Nframes, int Npoints,

                              int Nstate_ref)
{
    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, intrinsics,
                                             distortion_model, problem_details,
                                             Ncameras );

    if( problem_details.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
        {
            p[i_state++] = extrinsics[i_camera-1].r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics[i_camera-1].t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics[i_camera-1].t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( problem_details.do_optimize_frames )
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

    if( problem_details.do_optimize_calobject_warp )
    {
        p[i_state++] = calobject_warp->x / SCALE_CALOBJECT_WARP;
        p[i_state++] = calobject_warp->y / SCALE_CALOBJECT_WARP;
    }

    assert(i_state == Nstate_ref);
}

// Same as above, but packs/unpacks a vector instead of structures
void mrcal_pack_solver_state_vector( // out, in
                                     double* p, // unitless, FULL state on
                                                // input, scaled, decimated
                                                // (subject to problem_details),
                                                // meaningful state on output

                                     // in
                                     const distortion_model_t distortion_model,
                                     mrcal_problem_details_t problem_details,
                                     int Ncameras, int Nframes, int Npoints)
{
    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, p,
                                             distortion_model, problem_details,
                                             Ncameras );

    static_assert( offsetof(pose_t, r) == 0,
                   "pose_t has expected structure");
    static_assert( offsetof(pose_t, t) == 3*sizeof(double),
                   "pose_t has expected structure");
    if( problem_details.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
        {
            pose_t* extrinsics = (pose_t*)(&p[i_state]);

            p[i_state++] = extrinsics->r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics->r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics->r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics->t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics->t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics->t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( problem_details.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
        {
            pose_t* frames = (pose_t*)(&p[i_state]);
            p[i_state++] = frames->r.xyz[0] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames->r.xyz[1] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames->r.xyz[2] / SCALE_ROTATION_FRAME;

            p[i_state++] = frames->t.xyz[0] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames->t.xyz[1] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames->t.xyz[2] / SCALE_TRANSLATION_FRAME;
        }

        for(int i_point = 0; i_point < Npoints; i_point++)
        {
            point3_t* points = (point3_t*)(&p[i_state]);
            p[i_state++] = points->xyz[0] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[1] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[2] / SCALE_POSITION_POINT;
        }
    }

    if( problem_details.do_optimize_calobject_warp )
    {
        point2_t* calobject_warp = (point2_t*)(&p[i_state]);
        p[i_state++] = calobject_warp->x / SCALE_CALOBJECT_WARP;
        p[i_state++] = calobject_warp->y / SCALE_CALOBJECT_WARP;
    }
}

static int unpack_solver_state_intrinsics_onecamera( // out
                                                    intrinsics_core_t* intrinsics_core,
                                                    const distortion_model_t distortion_model,
                                                    double* distortions,

                                                    // in
                                                    const double* p,
                                                    int Ndistortions,
                                                    mrcal_problem_details_t problem_details )
{
    int i_state = 0;
    if( problem_details.do_optimize_intrinsic_core )
    {
        intrinsics_core->focal_xy [0] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        intrinsics_core->focal_xy [1] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        intrinsics_core->center_xy[0] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
        intrinsics_core->center_xy[1] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
    }

    if( problem_details.do_optimize_intrinsic_distortions )

        for(int i = ( !problem_details.do_optimize_cahvor_optical_axis &&
                      ( distortion_model == DISTORTION_CAHVOR ||
                        distortion_model == DISTORTION_CAHVORE )) ? 2 : 0;
            i<Ndistortions;
            i++)
        {
            distortions[i] = p[i_state++] * SCALE_DISTORTION;
        }

    return i_state;
}

static double get_scale_solver_state_intrinsics_onecamera( int i_state,
                                                           int Ndistortions,
                                                           mrcal_problem_details_t problem_details )
{
    if( problem_details.do_optimize_intrinsic_core )
    {
        if( i_state < 4)
        {
            if( i_state < 2 ) return SCALE_INTRINSICS_FOCAL_LENGTH;
            return SCALE_INTRINSICS_CENTER_PIXEL;
        }

        i_state -= 4;
    }

    if( problem_details.do_optimize_intrinsic_distortions )
        if( i_state < Ndistortions)
            return SCALE_DISTORTION;

    fprintf(stderr, "ERROR! %s() was asked about an out-of-bounds state\n", __func__);
    return -1.0;
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
                                           const distortion_model_t distortion_model,
                                           mrcal_problem_details_t problem_details,
                                           int Ncameras )
{
    if( !problem_details.do_optimize_intrinsic_core &&
        !problem_details.do_optimize_intrinsic_distortions )
        return 0;

    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    int i_state = 0;
    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        i_state +=
            unpack_solver_state_intrinsics_onecamera( (intrinsics_core_t*)intrinsics,
                                                      distortion_model,
                                                      &intrinsics[N_INTRINSICS_CORE],
                                                      &p[i_state], Ndistortions, problem_details );
        intrinsics = &intrinsics[Nintrinsics];
    }
    return i_state;
}

static int unpack_solver_state_extrinsics_one(// out
                                              pose_t* extrinsic,

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
                                           pose_t* frame,

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
                                         point3_t* point,

                                         // in
                                         const double* p)
{
    int i_state = 0;
    point->xyz[0] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[1] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[2] = p[i_state++] * SCALE_POSITION_POINT;
    return i_state;
}

static int unpack_solver_state_calobject_warp(// out
                                              point2_t* calobject_warp,

                                              // in
                                              const double* p)
{
    int i_state = 0;
    calobject_warp->xy[0] = p[i_state++] * SCALE_CALOBJECT_WARP;
    calobject_warp->xy[1] = p[i_state++] * SCALE_CALOBJECT_WARP;
    return i_state;
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out
                                 double* intrinsics, // Ncameras of these; each
                                                     // camera slice is
                                                     // (N_INTRINSICS_CORE,
                                                     // distortions)

                                 pose_t*       extrinsics, // Ncameras-1 of these
                                 pose_t*       frames,     // Nframes of these
                                 point3_t*     points,     // Npoints of these
                                 point2_t*     calobject_warp, // 1 of these

                                 // in
                                 const double* p,
                                 const distortion_model_t distortion_model,
                                 mrcal_problem_details_t problem_details,
                                 int Ncameras, int Nframes, int Npoints,

                                 int Nstate_ref)
{
    int i_state = unpack_solver_state_intrinsics(intrinsics,
                                                 p, distortion_model, problem_details, Ncameras);

    if( problem_details.do_optimize_extrinsics )
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics[i_camera-1], &p[i_state] );

    if( problem_details.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames[i_frame], &p[i_state] );
        for(int i_point = 0; i_point < Npoints; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }

    if( problem_details.do_optimize_calobject_warp )
        i_state += unpack_solver_state_calobject_warp(calobject_warp, &p[i_state]);

    assert(i_state == Nstate_ref);
}
// Same as above, but packs/unpacks a vector instead of structures
void mrcal_unpack_solver_state_vector( // out, in
                                       double* p, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       const distortion_model_t distortion_model,
                                       mrcal_problem_details_t problem_details,
                                       int Ncameras, int Nframes, int Npoints)
{
    int i_state = unpack_solver_state_intrinsics(p,
                                                 p, distortion_model, problem_details, Ncameras);

    if( problem_details.do_optimize_extrinsics )
    {
        static_assert( offsetof(pose_t, r) == 0,
                       "pose_t has expected structure");
        static_assert( offsetof(pose_t, t) == 3*sizeof(double),
                       "pose_t has expected structure");

        pose_t* extrinsics = (pose_t*)(&p[i_state]);
        for(int i_camera=1; i_camera < Ncameras; i_camera++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics[i_camera-1], &p[i_state] );
    }

    if( problem_details.do_optimize_frames )
    {
        pose_t* frames = (pose_t*)(&p[i_state]);
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames[i_frame], &p[i_state] );
        point3_t* points = (point3_t*)(&p[i_state]);
        for(int i_point = 0; i_point < Npoints; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }
    if( problem_details.do_optimize_calobject_warp )
    {
        point2_t* calobject_warp = (point2_t*)(&p[i_state]);
        i_state += unpack_solver_state_calobject_warp(calobject_warp, &p[i_state]);
    }
}

int mrcal_state_index_intrinsic_core(int i_camera,
                                     mrcal_problem_details_t problem_details,
                                     distortion_model_t distortion_model)
{
    return i_camera * getNintrinsicOptimizationParams(problem_details, distortion_model);
}
int mrcal_state_index_intrinsic_distortions(int i_camera,
                                            mrcal_problem_details_t problem_details,
                                            distortion_model_t distortion_model)
{
    int i =
        i_camera * getNintrinsicOptimizationParams(problem_details, distortion_model);
    if( problem_details.do_optimize_intrinsic_core )
        i += N_INTRINSICS_CORE;
    return i;
}
int mrcal_state_index_camera_rt(int i_camera, int Ncameras,
                                mrcal_problem_details_t problem_details,
                                distortion_model_t distortion_model)
{
    // returns a bogus value if i_camera==0. This camera has no state, and is
    // assumed to be at identity. The caller must know to not use the return
    // value in that case
    int i = getNintrinsicOptimizationParams(problem_details, distortion_model)*Ncameras;
    return i + (i_camera-1)*6;
}
int mrcal_state_index_frame_rt(int i_frame, int Ncameras,
                               mrcal_problem_details_t problem_details,
                               distortion_model_t distortion_model)
{
    return
        Ncameras * getNintrinsicOptimizationParams(problem_details, distortion_model) +
        (problem_details.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +
        i_frame * 6;
}
int mrcal_state_index_point(int i_point, int Nframes, int Ncameras,
                            mrcal_problem_details_t problem_details,
                            distortion_model_t distortion_model)
{
    return
        Ncameras * getNintrinsicOptimizationParams(problem_details, distortion_model) +
        (problem_details.do_optimize_extrinsics ? ((Ncameras-1) * 6) : 0) +
        (Nframes * 6) +
        i_point*3;
}
int mrcal_state_index_calobject_warp(int Npoints,
                                     int Nframes, int Ncameras,
                                     mrcal_problem_details_t problem_details,
                                     distortion_model_t distortion_model)
{
    return mrcal_state_index_point(Npoints, Nframes,  Ncameras,
                                   problem_details,
                                   distortion_model);
}

static int intrinsics_index_from_state_index( int i_state,
                                              const distortion_model_t distortion_model,
                                              mrcal_problem_details_t problem_details )
{
    // the caller of this thing assumes that the only difference between the
    // packed and unpacked vectors is the scaling. problem_details could make
    // the contents vary in other ways, and I make sure this doesn't happen.
    // It's possible to make this work in general, but it's tricky, and I don't
    // need to spent the time right now
    assert(problem_details.do_optimize_intrinsic_core &&
           problem_details.do_optimize_intrinsic_distortions &&
           ( !(!problem_details.do_optimize_cahvor_optical_axis &&
               ( distortion_model == DISTORTION_CAHVOR ||
                 distortion_model == DISTORTION_CAHVORE ))));
    return i_state;
}


//////////// duplicated in docstring for compute_intrinsics_uncertainty() in utils.py //////////
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
// In order to be useful I need to do something with M. I want to quantify
// how precise our optimal intrinsics are. Ultimately these are always used
// in a projection operation. So given a 3d observation vector v, I project
// it onto our image plane:
//
//   q = project(v, intrinsics)
//
// I assume an independent, gaussian noise on my input observations, and for a
// set of given observation vectors v, I compute the effect on the projection.
//
//   dq = dproj/dintrinsics dintrinsics
//      = dproj/dintrinsics Mi dm
//
// dprojection/dintrinsics comes from cvProjectPoints2(). I'm assuming
// everything is locally linear, so this is a constant matrix for each v.
// dintrinsics is the shift in the intrinsics of this camera. Mi
// is the subset of M that corresponds to these intrinsics
//
// If dm represents noise of the zero-mean, independent, gaussian variety,
// then dp and dq are also zero-mean gaussian, but no longer independent
//
//   Var(dq) = (dproj/dintrinsics Mi) Var(dm) (dproj/dintrinsics Mi)t =
//           = (dproj/dintrinsics Mi) (dproj/dintrinsics Mi)t s^2
//           = dproj/dintrinsics (Mi Mit) dproj/dintrinsicst s^2
//
// where s is the standard deviation of the noise of each parameter in dm.
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
//     [ ... ]           [ ...  ...    ... ]
// M = [  Mi ] and MMt = [ ...  MiMit  ... ]
//     [ ... ]           [ ...  ...    ... ]
//
// MMt = inv(JtJ) Jt dx/dm dx/dmt J inv(JtJ)
//
// For another uncertainty computation I need a similar measure, but without the
// dx/dm piece:
//
//   inv(JtJ) Jt J inv(JtJ) = inv(JtJ)
//
// As before, I return the subset of the matrix for each set of camera
// intrinsics
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
static bool computeUncertaintyMatrices(// out
                                       // dimensions (Ncameras,Nintrinsics_per_camera,Nintrinsics_per_camera)
                                       double* invJtJ_intrinsics_full,
                                       double* invJtJ_intrinsics_observations_only,

                                       // in
                                       distortion_model_t distortion_model,
                                       mrcal_problem_details_t problem_details,
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

    //Nintrinsics_per_camera_state can be < Nintrinsics_per_camera_all, if we're
    //locking down some variables with problem_details
    int Nintrinsics_per_camera_all = mrcal_getNintrinsicParams(distortion_model);
    int Nintrinsics_per_camera_state =
        getNintrinsicOptimizationParams(problem_details, distortion_model);
#warning "assumes the point range errors sit AFTER all the reprojection errors, which is WRONG"
    int Nmeas_observations = getNmeasurements_observationsonly(NobservationsBoard,
                                                               NobservationsPoint,
                                                               calibration_object_width_n);
    memset(invJtJ_intrinsics_observations_only, 0,
           Ncameras*Nintrinsics_per_camera_all* Nintrinsics_per_camera_all*sizeof(double));

    if( !problem_details.do_optimize_intrinsic_core        &&
        !problem_details.do_optimize_intrinsic_distortions )
    {
        // We're not optimizing any of the intrinsics. MMt is 0
        return true;
    }

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
    const int chunk_size = 4;

    cholmod_dense* Jt_slice =
        cholmod_allocate_dense( Jt->nrow,
                                chunk_size,
                                Jt->nrow,
                                CHOLMOD_REAL,
                                &solverCtx->common );



    // As described above, I'm looking at what input noise does, so I only look
    // at the measurements that pertain to the input observations directly. In
    // mrcal, this is the leading ones, before the range errors and the
    // regularization



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

            // the pack_solver_state_vector() call assumes that the only
            // difference between the packed and unpacked vectors is the
            // scaling. problem_details could make the contents vary in other
            // ways, and I make sure this doesn't happen. It's possible to make
            // this work in general, but it's tricky, and I don't need to spent
            // the time right now
            assert(Nintrinsics_per_camera_all == Nintrinsics_per_camera_state);

            // J has units 1/p, so to UNPACK p I PACK 1/p
            pack_solver_state_vector( Jrow,
                                      distortion_model,
                                      problem_details,
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
    // In [25]: np.linalg.norm(MMt0 - stats['invJtJ_intrinsics_observations_only'][0,:,:])
    // Out[25]: 1.4947344824339893e-12
    //
    // In [26]: np.linalg.norm(MMt1 - stats['invJtJ_intrinsics_observations_only'][1,:,:])
    // Out[26]: 4.223914927650401e-12
#endif


    // Compute invJtJ_intrinsics_full
    for(int icam = 0; icam < Ncameras; icam++)
    {
        // Here I want the diagonal blocks of inv(JtJ) for each camera's
        // intrinsics. I get them by doing solve(JtJ, [0; I; 0])
        void compute_invJtJ_block(double* JtJ, const int istate0, int N)
        {
            // I'm solving JtJ x = b where J is sparse, b is sparse, but x ends up
            // dense. cholmod doesn't have functions for this exact case. so I use
            // the dense-sparse-dense function, and densify the input. Instead of
            // sparse-sparse-sparse and the densifying the output. This feels like
            // it'd be more efficient

            int istate = istate0;

            // I can do chunk_size cols at a time
            while(1)
            {
                int Ncols = N < chunk_size ? N : chunk_size;
                Jt_slice->ncol = Ncols;
                memset( Jt_slice->x, 0, Jt_slice->nrow*Ncols*sizeof(double) );
                for(int icol=0; icol<Ncols; icol++)
                    // The J are unitless. I need to scale them to get real units
                    ((double*)Jt_slice->x)[ istate + icol + icol*Jt_slice->nrow] =
                        get_scale_solver_state_intrinsics_onecamera(istate + icol - istate0,
                                                                    Nintrinsics_per_camera_state - N_INTRINSICS_CORE,
                                                                    problem_details);

                cholmod_dense* M = cholmod_solve( CHOLMOD_A, solverCtx->factorization,
                                                  Jt_slice,
                                                  &solverCtx->common);

                // The cols/rows I want are in M. I pull them out, and apply
                // scaling (because my J are unitless, and I want full-unit
                // data)
                for(int icol=0; icol<Ncols; icol++)
                    unpack_solver_state_intrinsics_onecamera( (intrinsics_core_t*)&JtJ[icol*Nintrinsics_per_camera_state],
                                                              distortion_model,
                                                              &JtJ[icol*Nintrinsics_per_camera_state + N_INTRINSICS_CORE],

                                                              &((double*)(M->x))[icol*M->nrow + istate0],
                                                              Nintrinsics_per_camera_state - N_INTRINSICS_CORE,
                                                              problem_details );
                cholmod_free_dense (&M, &solverCtx->common);

                N -= Ncols;
                if(N <= 0) break;
                istate += Ncols;
                JtJ = &JtJ[Ncols*Nintrinsics_per_camera_state];
            }
        }




        const int istate0 = Nintrinsics_per_camera_state * icam;
        double* JtJ_thiscam = &invJtJ_intrinsics_full[icam*Nintrinsics_per_camera_all*Nintrinsics_per_camera_all];
        compute_invJtJ_block( JtJ_thiscam, istate0, Nintrinsics_per_camera_state );
    }

    // Compute invJtJ_intrinsics_observations_only
    for(int i_meas=0; i_meas < Nmeas_observations; i_meas += chunk_size)
    {
        // sparse to dense for a chunk of Jt
        memset( Jt_slice->x, 0, Jt_slice->nrow*chunk_size*sizeof(double) );
        for(unsigned int icol=0; icol<(unsigned)chunk_size; icol++)
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
            // the unpack_solver_state_vector() call assumes that the only
            // difference between the packed and unpacked vectors is the
            // scaling. problem_details could make the contents vary in other
            // ways, and I make sure this doesn't happen. It's possible to make
            // this work in general, but it's tricky, and I don't need to spent
            // the time right now
            assert(Nintrinsics_per_camera_all == Nintrinsics_per_camera_state);


            // The M I have here is a unitless, scaled M*. I need to scale it to get
            // M. See comment above.
            mrcal_unpack_solver_state_vector( &((double*)(M->x))[icol*M->nrow],
                                              distortion_model,
                                              problem_details,
                                              Ncameras, Nframes, Npoints);



            for(unsigned int irow0=0; irow0<M->nrow; irow0++)
            {
                double x0 = ((double*)(M->x))[irow0 + icol*M->nrow];

                int icam0 = irow0 / Nintrinsics_per_camera_state;
                if( icam0 >= Ncameras )
                    // not a camera intrinsic parameter
                    continue;

                double* MMt_thiscam = &invJtJ_intrinsics_observations_only[icam0*Nintrinsics_per_camera_all*Nintrinsics_per_camera_all];

                int i_intrinsics0 = intrinsics_index_from_state_index( irow0 - icam0*Nintrinsics_per_camera_state,
                                                                       distortion_model,
                                                                       problem_details );


                // special-case process the diagonal param
                MMt_thiscam[(Nintrinsics_per_camera_all+1)*i_intrinsics0] += x0*x0;

                // Now the off-diagonal
                for(unsigned int irow1=irow0+1; irow1<M->nrow; irow1++)
                {
                    int icam1 = irow1 / Nintrinsics_per_camera_state;

                    // I want to look at each camera individually, so I ignore the
                    // interactions between the parameters across cameras
                    if( icam0 != icam1 )
                        continue;

                    double x1 = ((double*)(M->x))[irow1 + icol*M->nrow];
                    double x0x1 = x0*x1;
                    int i_intrinsics1 = intrinsics_index_from_state_index( irow1 - icam1*Nintrinsics_per_camera_state,
                                                                           distortion_model,
                                                                           problem_details );
                    MMt_thiscam[Nintrinsics_per_camera_all*i_intrinsics0 + i_intrinsics1] += x0x1;
                    MMt_thiscam[Nintrinsics_per_camera_all*i_intrinsics1 + i_intrinsics0] += x0x1;
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
        // mrcal-calibrate-cameras
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

            // the unpack_solver_state_vector() call assumes that the only
            // difference between the packed and unpacked vectors is the
            // scaling. problem_details could make the contents vary in other
            // ways, and I make sure this doesn't happen. It's possible to make
            // this work in general, but it's tricky, and I don't need to spent
            // the time right now
            assert(Nintrinsics_per_camera_all == Nintrinsics_per_camera_state);

            unpack_solver_state_vector( (double*)(dp->x),
                                        distortion_model,
                                        problem_details,
                                        Ncameras, Nframes, Npoints);
            writevector("/tmp/dp", dp->x, Nstate); // unpacked

            writevector("/tmp/dx_hypothesis", dx_hypothesis, Nmeas);
            writevector("/tmp/dx",            dx,            Nmeas);

            // Now I project before and after the perturbation
            memcpy(p, solverCtx->beforeStep->p, Nstate*sizeof(double));
            unpack_solver_state_vector( p,
                                        distortion_model,
                                        problem_details,
                                        Ncameras, Nframes, Npoints);

            pose_t frame = {.t = {.xyz={-0.81691696, -0.02852554,  0.57604945}}};
            point2_t v0 =  project( NULL,NULL,NULL,NULL,NULL,NULL,NULL,
                                    // in
                                    (const intrinsics_core_t*)(&p[0]),
                                    &p[4],
                                    NULL,
                                    &frame,
                                    NULL,
                                    true,
                                    distortion_model,
                                    -1,
                                    1.0, 10);
            for(int i=0; i<Nstate; i++)
                p[i] += ((double*)dp->x)[i];
            point2_t v1 =  project( NULL,NULL,NULL,NULL,NULL,NULL,NULL,
                                    // in
                                    (const intrinsics_core_t*)(&p[0]),
                                    &p[4],
                                    NULL,
                                    &frame,
                                    NULL,
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

// Doing this myself instead of hooking into the logic in libdogleg. THIS
// implementation is simpler (looks just at the residuals), but also knows to
// ignore the outside-ROI data
static
bool markOutliers(// output, input
                  struct dogleg_outliers_t* markedOutliers,

                  // output
                  int* Noutliers,

                  // input
                  const observation_board_t* observations_board,
                  int NobservationsBoard,
                  int calibration_object_width_n,
                  const double* roi,

                  const double* x_measurements,
                  double expected_xy_stdev,
                  bool VERBOSE)
{
    // I define an outlier as a feature that's > k stdevs past the mean. The
    // threshold stdev is the worse of
    // - The stdev of my data set
    // - The expected stdev of my noise (calibrate-cameras
    //   --observed-pixel-uncertainty)
    //
    // The rationale:
    //
    // - If I have a really good solve, the stdev of my data set will be very
    //   low, so deviations off that already-very-good solve aren't important. I
    //   use the expected-noise stdev in this case
    //
    // - If the solve isn't great, my errors may exceed the expected-noise stdev
    //   (if my model doesn't fit very well, say). In that case I want to use
    //   the stdev from the data

    // threshold. +- 3sigma includes 99.7% of the data in a normal distribution
    const double k = 3.0;

#warning "think about this. here I'm looking at the deviations off mean error. That sounds wrong. Do I care about mean error? I want error to be 0, so maybe looking at absolute error is the thing to do instead"

    *Noutliers = 0;

    int i_pt,i_feature;


#define LOOP_FEATURE_BEGIN()                                            \
    i_feature = 0;                                                      \
    for(int i_observation_board=0;                                      \
        i_observation_board<NobservationsBoard;                         \
        i_observation_board++)                                          \
    {                                                                   \
        const observation_board_t* observation = &observations_board[i_observation_board]; \
        const int i_camera = observation->i_camera;                     \
        for(i_pt=0;                                                     \
            i_pt < calibration_object_width_n*calibration_object_width_n; \
            i_pt++, i_feature++)                                        \
        {                                                               \
            const point2_t* pt_observed = &observation->px[i_pt]; \
            double weight = region_of_interest_weight(pt_observed, roi, i_camera);


#define LOOP_FEATURE_END() \
    }}


    // I loop through my data 3 times: 2 times to compute the stdev, and then
    // once more to use that value to identify the outliers

    int Nfeatures_active = 0;
    double sum_mean = 0.0;
    LOOP_FEATURE_BEGIN()
    {
        if(markedOutliers[i_feature].marked)
        {
          (*Noutliers)++;
          continue;
        }
        if(weight != 1.0) continue;

        Nfeatures_active++;
        sum_mean +=
            x_measurements[2*i_feature + 0] +
            x_measurements[2*i_feature + 1];
    }
    LOOP_FEATURE_END();
    sum_mean /= (double)(2*Nfeatures_active);

    double var = 0.0;
    LOOP_FEATURE_BEGIN()
    {
        if(markedOutliers[i_feature].marked)
          continue;
        if(weight != 1.0) continue;

        double dx = (x_measurements[2*i_feature + 0] - sum_mean);
        double dy = (x_measurements[2*i_feature + 1] - sum_mean);

        var += dx*dx + dy*dy;
    }
    LOOP_FEATURE_END();
    var /= (double)(2*Nfeatures_active);

    if(var < expected_xy_stdev*expected_xy_stdev)
        var = expected_xy_stdev*expected_xy_stdev;

    bool markedAny = false;
    LOOP_FEATURE_BEGIN()
    {
        if(markedOutliers[i_feature].marked)
          continue;
        if(weight != 1.0) continue;

        double dx = (x_measurements[2*i_feature + 0] - sum_mean);
        double dy = (x_measurements[2*i_feature + 1] - sum_mean);
        if(dx*dx > k*k*var || dy*dy > k*k*var )
        {
            markedOutliers[i_feature].marked = true;
            markedAny                        = true;
            (*Noutliers)++;

            // MSG_IF_VERBOSE("Feature %d looks like an outlier. x/y are %f/%f stdevs off mean. Observed stdev: %f, limit: %f",
            //                i_feature, dx/sqrt(var), dy/sqrt(var), sqrt(var), k);

        }
    }
    LOOP_FEATURE_END();

    return markedAny;

#undef LOOP_FEATURE_BEGIN
#undef LOOP_FEATURE_END
}


mrcal_stats_t
mrcal_optimize( // out
                // These may be NULL. They're for diagnostic reporting to the
                // caller
                double* x_final,
                double* invJtJ_intrinsics_full,
                double* invJtJ_intrinsics_observations_only,
                // Buffer should be at least Npoints long. stats->Noutliers
                // elements will be filled in
                int*    outlier_indices_final,
                // Buffer should be at least Npoints long. stats->NoutsideROI
                // elements will be filled in
                int*    outside_ROI_indices_final,

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
                double*       intrinsics, // Ncameras * (N_INTRINSICS_CORE + Ndistortions)
                pose_t*       extrinsics, // Ncameras-1 of these. Transform FROM camera0 frame
                pose_t*       frames,     // Nframes of these.    Transform TO   camera0 frame
                point3_t*     points,     // Npoints of these.    In the camera0 frame
                point2_t*     calobject_warp, // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                // in
                int Ncameras, int Nframes, int Npoints,

                const observation_board_t* observations_board,
                int NobservationsBoard,

                const observation_point_t* observations_point,
                int NobservationsPoint,

                bool check_gradient,
                int Noutlier_indices_input,
                int* outlier_indices_input,
                const double* roi,
                bool VERBOSE,
                const bool skip_outlier_rejection,

                distortion_model_t distortion_model,
                double observed_pixel_uncertainty,
                const int* imagersizes, // Ncameras*2 of these

                mrcal_problem_details_t problem_details,

                double calibration_object_spacing,
                int calibration_object_width_n)
{
    if( ( invJtJ_intrinsics_full && !invJtJ_intrinsics_observations_only) ||
        (!invJtJ_intrinsics_full &&  invJtJ_intrinsics_observations_only) )
    {
        fprintf(stderr, "ERROR: either both or none of (invJtJ_intrinsics_full.invJtJ_intrinsics_observations_only) can be NULL\n");
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }
    if( calobject_warp == NULL && problem_details.do_optimize_calobject_warp )
    {
        fprintf(stderr, "ERROR: We're optimizing the calibration object warp, so a buffer with a seed MUST be passed in.");
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }

    if(!problem_details.do_optimize_intrinsic_core        &&
       !problem_details.do_optimize_intrinsic_distortions &&
       !problem_details.do_optimize_extrinsics            &&
       !problem_details.do_optimize_frames                &&
       !problem_details.do_optimize_calobject_warp)
    {
        MSG("Warning: Not optimizing any of our variables");
    }

    dogleg_setDebug( VERBOSE ? DOGLEG_DEBUG_VNLOG : 0 );

#warning update these parameters
    // These were derived empirically, seeking high accuracy, fast convergence
    // and without serious concern for performance. I looked only at a single
    // frame. Tweak them please
    dogleg_setThresholds(0, 1e-6, 0);
    dogleg_setMaxIterations(300);
    //dogleg_setTrustregionUpdateParameters(0.1, 0.15, 4.0, 0.75);


    const int Nstate        = get_Nstate(Ncameras, Nframes, Npoints,
                                         problem_details,
                                         distortion_model);
    const int Nmeasurements = mrcal_getNmeasurements_all(Ncameras, NobservationsBoard,
                                                         observations_point, NobservationsPoint,
                                                         calibration_object_width_n,
                                                         problem_details,
                                                         distortion_model);
    const int N_j_nonzero   = get_N_j_nonzero(Ncameras,
                                              observations_board, NobservationsBoard,
                                              observations_point, NobservationsPoint,
                                              problem_details,
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
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
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
            if(Jt) {                            \
                Jcolidx[ iJacobian ] = col;     \
                Jval   [ iJacobian ] = g;       \
            }                                   \
            iJacobian++;                        \
        } while(0)
#define STORE_JACOBIAN2(col0, g0, g1)                   \
        do                                              \
        {                                               \
            if(Jt) {                                    \
                Jcolidx[ iJacobian+0 ] = col0+0;        \
                Jval   [ iJacobian+0 ] = g0;            \
                Jcolidx[ iJacobian+1 ] = col0+1;        \
                Jval   [ iJacobian+1 ] = g1;            \
            }                                           \
            iJacobian += 2;                             \
        } while(0)
#define STORE_JACOBIAN3(col0, g0, g1, g2)               \
        do                                              \
        {                                               \
            if(Jt) {                                    \
                Jcolidx[ iJacobian+0 ] = col0+0;        \
                Jval   [ iJacobian+0 ] = g0;            \
                Jcolidx[ iJacobian+1 ] = col0+1;        \
                Jval   [ iJacobian+1 ] = g1;            \
                Jcolidx[ iJacobian+2 ] = col0+2;        \
                Jval   [ iJacobian+2 ] = g2;            \
            }                                           \
            iJacobian += 3;                             \
        } while(0)




        int NlockedLeadingDistortions =
            ( !problem_details.do_optimize_cahvor_optical_axis &&
              ( distortion_model == DISTORTION_CAHVOR ||
                distortion_model == DISTORTION_CAHVORE )) ? 2 : 0;

        // If I'm locking down some parameters, then the state vector contains a
        // subset of my data. I reconstitute the intrinsics and extrinsics here.
        // I do the frame poses later. This is a good way to do it if I have few
        // cameras. With many cameras (this will be slow)
        intrinsics_core_t intrinsic_core_all[Ncameras];
        double distortions_all[Ncameras][Ndistortions];
        pose_t camera_rt[Ncameras];

        point2_t calobject_warp_local = {};
        const int i_var_calobject_warp = mrcal_state_index_calobject_warp(Npoints, Nframes, Ncameras, problem_details, distortion_model);
        if(problem_details.do_optimize_calobject_warp)
            unpack_solver_state_calobject_warp(&calobject_warp_local, &packed_state[i_var_calobject_warp]);
        else if(calobject_warp != NULL)
            calobject_warp_local = *calobject_warp;

        for(int i_camera=0; i_camera<Ncameras; i_camera++)
        {
            // First I pull in the chunks from the optimization vector
            const int i_var_intrinsic_core         = mrcal_state_index_intrinsic_core        (i_camera,           problem_details, distortion_model);
            const int i_var_intrinsic_distortions  = mrcal_state_index_intrinsic_distortions (i_camera,           problem_details, distortion_model);
            const int i_var_camera_rt              = mrcal_state_index_camera_rt             (i_camera, Ncameras, problem_details, distortion_model);
            unpack_solver_state_intrinsics_onecamera(&intrinsic_core_all[i_camera],
                                                     distortion_model,
                                                     distortions_all[i_camera],
                                                     &packed_state[ i_var_intrinsic_core ],
                                                     Ndistortions,
                                                     problem_details );

            // And then I fill in the gaps using the seed values
            if(!problem_details.do_optimize_intrinsic_core)
                memcpy( &intrinsic_core_all[i_camera],
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera],
                        N_INTRINSICS_CORE*sizeof(double) );
            if(!problem_details.do_optimize_intrinsic_distortions)
                memcpy( distortions_all[i_camera],
                        &intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera + N_INTRINSICS_CORE],
                        Ndistortions*sizeof(double) );
            else if( !problem_details.do_optimize_cahvor_optical_axis &&
                     ( distortion_model == DISTORTION_CAHVOR ||
                       distortion_model == DISTORTION_CAHVORE ) )
            {
                // We're optimizing distortions, just not those particular
                // cahvor components
                distortions_all[i_camera][0] = intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera + N_INTRINSICS_CORE + 0];
                distortions_all[i_camera][1] = intrinsics[(N_INTRINSICS_CORE+Ndistortions)*i_camera + N_INTRINSICS_CORE + 1];
            }

            // extrinsics
            if( i_camera != 0 )
            {
                if(problem_details.do_optimize_extrinsics)
                    unpack_solver_state_extrinsics_one(&camera_rt[i_camera-1], &packed_state[i_var_camera_rt]);
                else
                    memcpy(&camera_rt[i_camera-1], &extrinsics[i_camera-1], sizeof(pose_t));
            }
        }

        for(int i_observation_board = 0;
            i_observation_board < NobservationsBoard;
            i_observation_board++)
        {
            const observation_board_t* observation = &observations_board[i_observation_board];

            const int i_camera = observation->i_camera;
            const int i_frame  = observation->i_frame;


            // Some of these are bogus if problem_details says they're inactive
            const int i_var_frame_rt = mrcal_state_index_frame_rt  (i_frame,  Ncameras, problem_details, distortion_model);

            pose_t frame_rt;
            if(problem_details.do_optimize_frames)
                unpack_solver_state_framert_one(&frame_rt, &packed_state[i_var_frame_rt]);
            else
                memcpy(&frame_rt, &frames[i_frame], sizeof(pose_t));

            const int i_var_intrinsic_core         = mrcal_state_index_intrinsic_core        (i_camera,           problem_details, distortion_model);
            const int i_var_intrinsic_distortions  = mrcal_state_index_intrinsic_distortions (i_camera,           problem_details, distortion_model);
            const int i_var_camera_rt              = mrcal_state_index_camera_rt             (i_camera, Ncameras, problem_details, distortion_model);

            for(int i_pt=0;
                i_pt < calibration_object_width_n*calibration_object_width_n;
                i_pt++)
            {
                const point2_t* pt_observed = &observation->px[i_pt];
                double weight = region_of_interest_weight(pt_observed, roi, i_camera);

                // these are computed in respect to the real-unit parameters,
                // NOT the unit-scale parameters used by the optimizer
                double dxy_dintrinsic_core       [2 * N_INTRINSICS_CORE];
                double dxy_dintrinsic_distortions[2 * Ndistortions];
                point3_t dxy_drcamera[2];
                point3_t dxy_dtcamera[2];
                point3_t dxy_drframe [2];
                point3_t dxy_dtframe [2];
                point2_t dxy_dcalobject_warp[2];
                point2_t pt_hypothesis =
                    project(problem_details.do_optimize_intrinsic_core ?
                              dxy_dintrinsic_core : NULL,
                            problem_details.do_optimize_intrinsic_distortions ?
                              dxy_dintrinsic_distortions : NULL,
                            problem_details.do_optimize_extrinsics ?
                              dxy_drcamera : NULL,
                            problem_details.do_optimize_extrinsics ?
                              dxy_dtcamera : NULL,
                            problem_details.do_optimize_frames ?
                              dxy_drframe : NULL,
                            problem_details.do_optimize_frames ?
                              dxy_dtframe : NULL,
                            problem_details.do_optimize_calobject_warp ?
                              dxy_dcalobject_warp : NULL,
                            &intrinsic_core_all[i_camera], distortions_all[i_camera],
                            &camera_rt[i_camera-1], &frame_rt,
                            calobject_warp == NULL ? NULL : &calobject_warp_local,
                            i_camera == 0,
                            distortion_model,
                            i_pt,
                            calibration_object_spacing,
                            calibration_object_width_n);

                if(!observation->skip_observation &&

                   // /2 because I look at FEATURES here, not discrete
                   // measurements
                   !markedOutliers[iMeasurement/2].marked)
                {
                    // I have my two measurements (dx, dy). I propagate their
                    // gradient and store them
                    for( int i_xy=0; i_xy<2; i_xy++ )
                    {
                        const double err = (pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy]) * weight;

                        if( reportFitMsg )
                        {
                            MSG("%s: obs/frame/cam/dot: %d %d %d %d err: %g",
                                reportFitMsg,
                                i_observation_board, i_frame, i_camera, i_pt, err);
                            continue;
                        }

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        // I want these gradient values to be computed in
                        // monotonically-increasing order of variable index. I
                        // don't CHECK, so it's the developer's responsibility
                        // to make sure. This ordering is set in
                        // pack_solver_state(), unpack_solver_state()
                        if( problem_details.do_optimize_intrinsic_core )
                        {
                            STORE_JACOBIAN( i_var_intrinsic_core + 0,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 0] *
                                            weight * SCALE_INTRINSICS_FOCAL_LENGTH );
                            STORE_JACOBIAN( i_var_intrinsic_core + 1,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 1] *
                                            weight * SCALE_INTRINSICS_FOCAL_LENGTH );
                            STORE_JACOBIAN( i_var_intrinsic_core + 2,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 2] *
                                            weight * SCALE_INTRINSICS_CENTER_PIXEL );
                            STORE_JACOBIAN( i_var_intrinsic_core + 3,
                                            dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 3] *
                                            weight * SCALE_INTRINSICS_CENTER_PIXEL );
                        }

                        if( problem_details.do_optimize_intrinsic_distortions )
                            for(int i = NlockedLeadingDistortions; i<Ndistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i - NlockedLeadingDistortions,
                                                dxy_dintrinsic_distortions[i_xy * Ndistortions + i] *
                                                weight * SCALE_DISTORTION );

                        if( problem_details.do_optimize_extrinsics )
                            if( i_camera != 0 )
                            {
                                STORE_JACOBIAN3( i_var_camera_rt + 0,
                                                 dxy_drcamera[i_xy].xyz[0] *
                                                 weight * SCALE_ROTATION_CAMERA,
                                                 dxy_drcamera[i_xy].xyz[1] *
                                                 weight * SCALE_ROTATION_CAMERA,
                                                 dxy_drcamera[i_xy].xyz[2] *
                                                 weight * SCALE_ROTATION_CAMERA);
                                STORE_JACOBIAN3( i_var_camera_rt + 3,
                                                 dxy_dtcamera[i_xy].xyz[0] *
                                                 weight * SCALE_TRANSLATION_CAMERA,
                                                 dxy_dtcamera[i_xy].xyz[1] *
                                                 weight * SCALE_TRANSLATION_CAMERA,
                                                 dxy_dtcamera[i_xy].xyz[2] *
                                                 weight * SCALE_TRANSLATION_CAMERA);
                            }

                        if( problem_details.do_optimize_frames )
                        {
                            STORE_JACOBIAN3( i_var_frame_rt + 0,
                                             dxy_drframe[i_xy].xyz[0] *
                                             weight * SCALE_ROTATION_FRAME,
                                             dxy_drframe[i_xy].xyz[1] *
                                             weight * SCALE_ROTATION_FRAME,
                                             dxy_drframe[i_xy].xyz[2] *
                                             weight * SCALE_ROTATION_FRAME);
                            STORE_JACOBIAN3( i_var_frame_rt + 3,
                                             dxy_dtframe[i_xy].xyz[0] *
                                             weight * SCALE_TRANSLATION_FRAME,
                                             dxy_dtframe[i_xy].xyz[1] *
                                             weight * SCALE_TRANSLATION_FRAME,
                                             dxy_dtframe[i_xy].xyz[2] *
                                             weight * SCALE_TRANSLATION_FRAME);
                        }

                        if( problem_details.do_optimize_calobject_warp )
                        {
                            STORE_JACOBIAN2( i_var_calobject_warp,
                                             dxy_dcalobject_warp[i_xy].x * weight * SCALE_CALOBJECT_WARP,
                                             dxy_dcalobject_warp[i_xy].y * weight * SCALE_CALOBJECT_WARP);
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

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( problem_details.do_optimize_intrinsic_core )
                            for(int i=0; i<N_INTRINSICS_CORE; i++)
                                STORE_JACOBIAN( i_var_intrinsic_core + i, 0.0 );

                        if( problem_details.do_optimize_intrinsic_distortions )
                            for(int i=0; i<Ndistortions-NlockedLeadingDistortions; i++)
                                STORE_JACOBIAN( i_var_intrinsic_distortions + i, 0.0 );

                        if( problem_details.do_optimize_extrinsics )
                            if( i_camera != 0 )
                            {
                                STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                                STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                            }

                        if( problem_details.do_optimize_frames )
                        {
                            const double dframe = observation->skip_frame ? 1.0 : 0.0;
                            // Arbitrary differences between the dimensions to keep
                            // my Hessian non-singular. This is 100% arbitrary. I'm
                            // skipping these measurements so these variables
                            // actually don't affect the computation at all
                            STORE_JACOBIAN3( i_var_frame_rt + 0, dframe*1.1, dframe*1.2, dframe*1.3);
                            STORE_JACOBIAN3( i_var_frame_rt + 3, dframe*1.4, dframe*1.5, dframe*1.6);
                        }

                        if( problem_details.do_optimize_calobject_warp )
                            STORE_JACOBIAN2( i_var_calobject_warp, 0.0, 0.0 );


                        iMeasurement++;
                    }
                }
            }
        }

        // Handle all the point observations. This is VERY similar to the
        // board-observation loop above. Please consolidate
        for(int i_observation_point = 0;
            i_observation_point < NobservationsPoint;
            i_observation_point++)
        {
            const observation_point_t* observation = &observations_point[i_observation_point];

            const int i_camera = observation->i_camera;
            const int i_point  = observation->i_point;

            const point2_t* pt_observed = &observation->px;
            double weight = region_of_interest_weight(pt_observed, roi, i_camera);

            const int i_var_intrinsic_core         = mrcal_state_index_intrinsic_core        (i_camera,           problem_details, distortion_model);
            const int i_var_intrinsic_distortions  = mrcal_state_index_intrinsic_distortions (i_camera,           problem_details, distortion_model);
            const int i_var_camera_rt              = mrcal_state_index_camera_rt             (i_camera, Ncameras, problem_details, distortion_model);

            const int i_var_point                  = mrcal_state_index_point                 (i_point,  Nframes, Ncameras, problem_details, distortion_model);
            point3_t point;

            if(problem_details.do_optimize_frames)
                unpack_solver_state_point_one(&point, &packed_state[i_var_point]);
            else
                point = points[i_point];


            // Check for invalid points. I report a very poor cost if I see
            // this.
            bool have_invalid_point = false;
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
            point3_t dxy_drcamera[2];
            point3_t dxy_dtcamera[2];
            point3_t dxy_dpoint  [2];

            // The array reference [-3] is intended, but the compiler throws a
            // warning. I silence it here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
            point2_t pt_hypothesis =
                project(problem_details.do_optimize_intrinsic_core ?
                          dxy_dintrinsic_core        : NULL,
                        problem_details.do_optimize_intrinsic_distortions ?
                          dxy_dintrinsic_distortions : NULL,

                        problem_details.do_optimize_extrinsics ?
                          dxy_drcamera : NULL,
                        problem_details.do_optimize_extrinsics ?
                          dxy_dtcamera : NULL,
                        NULL, // frame rotation. I only have a point position
                        problem_details.do_optimize_frames ?
                          dxy_dpoint : NULL,
                        NULL,
                        &intrinsic_core_all[i_camera], distortions_all[i_camera],
                        &camera_rt[i_camera-1],

                        // I only have the point position, so the 'rt' memory
                        // points 3 back. The fake "r" here will not be
                        // referenced
                        (pose_t*)(&point.xyz[-3]),
                        NULL,

                        i_camera == 0,
                        distortion_model,
                        -1,
                        calibration_object_spacing,
                        calibration_object_width_n);
#pragma GCC diagnostic pop

            if(!observation->skip_observation
#warning "no outlier rejection on points yet; see warning above"
               )
            {
                // I have my two measurements (dx, dy). I propagate their
                // gradient and store them
                double invalid_point_scale = 1.0;
                if(have_invalid_point)
                    // I have an invalid point. This is a VERY bad solution. The solver
                    // needs to try again with a smaller step
                    invalid_point_scale = 1e6;


                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = (pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy])*invalid_point_scale*weight;

                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    // I want these gradient values to be computed in
                    // monotonically-increasing order of variable index. I don't
                    // CHECK, so it's the developer's responsibility to make
                    // sure. This ordering is set in pack_solver_state(),
                    // unpack_solver_state()
                    if( problem_details.do_optimize_intrinsic_core )
                    {
                        STORE_JACOBIAN( i_var_intrinsic_core + 0,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 0] *
                                        invalid_point_scale *
                                        weight * SCALE_INTRINSICS_FOCAL_LENGTH );
                        STORE_JACOBIAN( i_var_intrinsic_core + 1,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 1] *
                                        invalid_point_scale *
                                        weight * SCALE_INTRINSICS_FOCAL_LENGTH );
                        STORE_JACOBIAN( i_var_intrinsic_core + 2,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 2] *
                                        invalid_point_scale *
                                        weight * SCALE_INTRINSICS_CENTER_PIXEL );
                        STORE_JACOBIAN( i_var_intrinsic_core + 3,
                                        dxy_dintrinsic_core[i_xy * N_INTRINSICS_CORE + 3] *
                                        invalid_point_scale *
                                        weight * SCALE_INTRINSICS_CENTER_PIXEL );
                    }

                    if( problem_details.do_optimize_intrinsic_distortions )
                        for(int i = NlockedLeadingDistortions; i<Ndistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i - NlockedLeadingDistortions,
                                            dxy_dintrinsic_distortions[i_xy * Ndistortions + i] *
                                            invalid_point_scale *
                                            weight * SCALE_DISTORTION );

                    if( problem_details.do_optimize_extrinsics )
                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0,
                                             dxy_drcamera[i_xy].xyz[0] *
                                             invalid_point_scale *
                                             weight * SCALE_ROTATION_CAMERA,
                                             dxy_drcamera[i_xy].xyz[1] *
                                             invalid_point_scale *
                                             weight * SCALE_ROTATION_CAMERA,
                                             dxy_drcamera[i_xy].xyz[2] *
                                             invalid_point_scale *
                                             weight * SCALE_ROTATION_CAMERA);
                            STORE_JACOBIAN3( i_var_camera_rt + 3,
                                             dxy_dtcamera[i_xy].xyz[0] *
                                             invalid_point_scale *
                                             weight * SCALE_TRANSLATION_CAMERA,
                                             dxy_dtcamera[i_xy].xyz[1] *
                                             invalid_point_scale *
                                             weight * SCALE_TRANSLATION_CAMERA,
                                             dxy_dtcamera[i_xy].xyz[2] *
                                             invalid_point_scale *
                                             weight * SCALE_TRANSLATION_CAMERA);
                        }

                    if( problem_details.do_optimize_frames )
                        STORE_JACOBIAN3( i_var_point,
                                         dxy_dpoint[i_xy].xyz[0] *
                                         invalid_point_scale *
                                         weight * SCALE_POSITION_POINT,
                                         dxy_dpoint[i_xy].xyz[1] *
                                         invalid_point_scale *
                                         weight * SCALE_POSITION_POINT,
                                         dxy_dpoint[i_xy].xyz[2] *
                                         invalid_point_scale *
                                         weight * SCALE_POSITION_POINT);

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

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( problem_details.do_optimize_frames )
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
                        CvMat rc = cvMat(3,1, CV_64FC1, camera_rt[i_camera-1].r.xyz);

                        double _Rc[3*3];
                        CvMat  Rc = cvMat(3,3,CV_64FC1, _Rc);
                        double _d_Rc_rc[9*3];
                        CvMat d_Rc_rc = cvMat(9,3,CV_64F, _d_Rc_rc);
                        cvRodrigues2(&rc, &Rc, &d_Rc_rc);

                        point3_t pt_cam;
                        mul_vec3_gen33t_vout(point.xyz, _Rc, pt_cam.xyz);
                        add_vec(3, pt_cam.xyz, camera_rt[i_camera-1].t.xyz);

                        double dist = sqrt( pt_cam.x*pt_cam.x +
                                            pt_cam.y*pt_cam.y +
                                            pt_cam.z*pt_cam.z );
                        double dist_recip = 1.0/dist;
                        double err = dist - observation->dist;
                        err *= DISTANCE_ERROR_EQUIVALENT__PIXELS_PER_M;

                        if(Jt) Jrowptr[iMeasurement] = iJacobian;
                        x[iMeasurement] = err;
                        norm2_error += err*err;

                        if( problem_details.do_optimize_extrinsics )
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

                        if( problem_details.do_optimize_frames )
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

                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    if( problem_details.do_optimize_intrinsic_core )
                        for(int i=0; i<N_INTRINSICS_CORE; i++)
                            STORE_JACOBIAN( i_var_intrinsic_core + i,
                                            0.0 );

                    if( problem_details.do_optimize_intrinsic_distortions )
                        for(int i=0; i<Ndistortions-NlockedLeadingDistortions; i++)
                            STORE_JACOBIAN( i_var_intrinsic_distortions + i,
                                            0.0 );

                    if( problem_details.do_optimize_extrinsics )
                        if( i_camera != 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }

                    if( problem_details.do_optimize_frames )
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

                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    if( problem_details.do_optimize_extrinsics )
                        if(i_camera != 0)
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }
                    if( problem_details.do_optimize_frames )
                        STORE_JACOBIAN3( i_var_point, 0.0, 0.0, 0.0);
                    iMeasurement++;
                }
            }
        }

        // regularization terms for the intrinsics. I favor smaller distortion
        // parameters
        if(!problem_details.do_skip_regularization &&
           ( problem_details.do_optimize_intrinsic_distortions ||
             problem_details.do_optimize_intrinsic_core
           ))
        {
            // I want the total regularization cost to be low relative to the
            // other contributions to the cost. And I want each set of
            // regularization terms to weigh roughly the same. Let's say I want
            // regularization to account for ~ .5% of the other error
            // contributions:
            //
            //   Nmeasurements_rest*normal_pixel_error_sq * 0.005/2. =
            //   Nmeasurements_regularization_distortion *normal_regularization_distortion_error_sq  =
            //   Nmeasurements_regularization_centerpixel*normal_regularization_centerpixel_error_sq =


            const bool dump_regularizaton_details = false;


            int    Nmeasurements_regularization_distortion  = Ncameras*(Ndistortions - NlockedLeadingDistortions);
            int    Nmeasurements_regularization_centerpixel = Ncameras*2;

            int    Nmeasurements_nonregularization =
                Nmeasurements -
                Nmeasurements_regularization_distortion -
                Nmeasurements_regularization_centerpixel;

            double normal_pixel_error = 1.0;
            double expected_total_pixel_error_sq =
                (double)Nmeasurements_nonregularization *
                normal_pixel_error *
                normal_pixel_error;
            if(dump_regularizaton_details)
                MSG("expected_total_pixel_error_sq: %f", expected_total_pixel_error_sq);

            double scale_regularization_distortion =
                ({
                    double normal_distortion_value = 0.2;

                    double expected_regularization_distortion_error_sq_noscale =
                        (double)Nmeasurements_regularization_distortion *
                        normal_distortion_value;

                    double scale_sq =
                        expected_total_pixel_error_sq * 0.005/2. / expected_regularization_distortion_error_sq_noscale;

                    if(dump_regularizaton_details)
                        MSG("expected_regularization_distortion_error_sq: %f", expected_regularization_distortion_error_sq_noscale*scale_sq);

                    sqrt(scale_sq);
                });
            double scale_regularization_centerpixel =
                ({

                    double normal_centerpixel_offset = 50.0;

                    double expected_regularization_centerpixel_error_sq_noscale =
                        (double)Nmeasurements_regularization_centerpixel *
                        normal_centerpixel_offset *
                        normal_centerpixel_offset;

                    double scale_sq =
                        expected_total_pixel_error_sq * 0.005/2. / expected_regularization_centerpixel_error_sq_noscale;

                    if(dump_regularizaton_details)
                        MSG("expected_regularization_centerpixel_error_sq: %f", expected_regularization_centerpixel_error_sq_noscale*scale_sq);

                    sqrt(scale_sq);
                });

            for(int i_camera=0; i_camera<Ncameras; i_camera++)
            {
                if( problem_details.do_optimize_intrinsic_distortions)
                {
                    const int i_var_intrinsic_distortions =
                        mrcal_state_index_intrinsic_distortions(i_camera, problem_details, distortion_model);

                    for(int j=NlockedLeadingDistortions; j<Ndistortions; j++)
                    {
                        if(Jt) Jrowptr[iMeasurement] = iJacobian;

                        // This maybe should live elsewhere, but I put it here
                        // for now. Various distortion coefficients have
                        // different meanings, and should be regularized in
                        // different ways. Specific logic follows
                        double scale = scale_regularization_distortion;

                        if( DISTORTION_IS_OPENCV(distortion_model) &&
                            distortion_model >= DISTORTION_OPENCV8 &&
                            5 <= j && j <= 7 )
                        {
                            // The radial distortion in opencv is x_distorted =
                            // x*scale where r2 = norm2(xy - xyc) and
                            //
                            // scale = (1 + k0 r2 + k1 r4 + k4 r6)/(1 + k5 r2 + k6 r4 + k7 r6)
                            //
                            // Note that k2,k3 are tangential (NOT radial)
                            // distortion components. Note that the r6 factor in
                            // the numerator is only present for
                            // >=DISTORTION_OPENCV5. Note that the denominator
                            // is only present for >= DISTORTION_OPENCV8. The
                            // danger with a rational model is that it's
                            // possible to get into a situation where scale ~
                            // 0/0 ~ 1. This would have very poorly behaved
                            // derivatives. If all the rational coefficients are
                            // ~0, then the denominator is always ~1, and this
                            // problematic case can't happen. I favor that by
                            // regularizing the coefficients in the denominator
                            // more strongly
                            scale *= 5.;
                        }

                        // This exists to avoid /0 in the gradient
                        const double eps = 1e-3;

                        double sign         = copysign(1.0, distortions_all[i_camera][j]);
                        double err_no_scale = sqrt(fabs(distortions_all[i_camera][j]) + eps);
                        double err          = err_no_scale * scale;

                        x[iMeasurement]  = err;
                        norm2_error     += err*err;
                        STORE_JACOBIAN( i_var_intrinsic_distortions + j - NlockedLeadingDistortions,
                                        scale * sign * SCALE_DISTORTION / (2. * err_no_scale) );
                        iMeasurement++;
                        if(dump_regularizaton_details)
                            MSG("regularization distortion: %g; norm2: %g", err, err*err);

                    }
                }

                if( problem_details.do_optimize_intrinsic_core)
                {
                    // And another regularization term: optical center should be
                    // near the middle. This breaks the symmetry between moving the
                    // center pixel coords and pitching/yawing the camera.
                    const int i_var_intrinsic_core =
                        mrcal_state_index_intrinsic_core(i_camera,
                                                         problem_details,
                                                         distortion_model);

                    double cx_target = 0.5 * (double)(imagersizes[i_camera*2 + 0] - 1);
                    double cy_target = 0.5 * (double)(imagersizes[i_camera*2 + 1] - 1);

                    double err = scale_regularization_centerpixel *
                        (intrinsic_core_all[i_camera].center_xy[0] - cx_target);
                    x[iMeasurement]  = err;
                    norm2_error     += err*err;
                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    STORE_JACOBIAN( i_var_intrinsic_core + 2,
                                    scale_regularization_centerpixel * SCALE_INTRINSICS_CENTER_PIXEL );
                    iMeasurement++;
                    if(dump_regularizaton_details)
                        MSG("regularization center pixel off-center: %g; norm2: %g", err, err*err);

                    err = scale_regularization_centerpixel *
                        (intrinsic_core_all[i_camera].center_xy[1] - cy_target);
                    x[iMeasurement]  = err;
                    norm2_error     += err*err;
                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    STORE_JACOBIAN( i_var_intrinsic_core + 3,
                                    scale_regularization_centerpixel * SCALE_INTRINSICS_CENTER_PIXEL );
                    iMeasurement++;
                    if(dump_regularizaton_details)
                        MSG("regularization center pixel off-center: %g; norm2: %g", err, err*err);
                }
            }
        }


        // required to indicate the end of the jacobian matrix
        if( !reportFitMsg )
        {
            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            assert(iMeasurement == Nmeasurements);
            assert(iJacobian    == N_j_nonzero  );

            // MSG_IF_VERBOSE("RMS: %g", sqrt(norm2_error / ((double)Nmeasurements / 2.0)));
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
                      calobject_warp,
                      problem_details,
                      Ncameras, Nframes, Npoints, Nstate);

    double norm2_error = -1.0;
    mrcal_stats_t stats = {.rms_reproj_error__pixels = -1.0 };

    if( !check_gradient )
    {
        stats.Noutliers = 0;
        for(int i=0; i<Noutlier_indices_input; i++)
        {
            markedOutliers[outlier_indices_input[i]].marked = true;
            stats.Noutliers++;
        }

        if(VERBOSE)
        {
            reportFitMsg = "Before";
#warning hook this up
            //        optimizerCallback(packed_state, NULL, NULL, NULL);
        }
        reportFitMsg = NULL;


        double outliernessScale = -1.0;
        do
        {
            norm2_error = dogleg_optimize(packed_state,
                                          Nstate, Nmeasurements, N_j_nonzero,
                                          &optimizerCallback, NULL, &solver_context);
            if(_solver_context != NULL)
                *_solver_context = solver_context;

            if(norm2_error < 0)
                // libdogleg barfed. I quit out
                goto done;

#if 0
            // Not using dogleg_markOutliers() (for now?)

            if(outliernessScale < 0.0 && VERBOSE)
                // These are for debug reporting
                dogleg_reportOutliers(getConfidence,
                                      &outliernessScale,
                                      2, Npoints_fromBoards,
                                      stats.Noutliers,
                                      solver_context->beforeStep, solver_context);
#endif

        } while( !skip_outlier_rejection &&
                 markOutliers(markedOutliers,
                              &stats.Noutliers,
                              observations_board,
                              NobservationsBoard,
                              calibration_object_width_n,
                              roi,
                              solver_context->beforeStep->x,
                              observed_pixel_uncertainty,
                              VERBOSE) );

        // Done. I have the final state. I spit it back out
        unpack_solver_state( intrinsics, // Ncameras of these
                             extrinsics, // Ncameras-1 of these
                             frames,     // Nframes of these
                             points,     // Npoints of these
                             calobject_warp,
                             packed_state,
                             distortion_model,
                             problem_details,
                             Ncameras, Nframes, Npoints, Nstate);
        if(VERBOSE)
        {
            // Not using dogleg_markOutliers() (for now?)
#if 0
            // These are for debug reporting
            dogleg_reportOutliers(getConfidence,
                                  &outliernessScale,
                                  2, Npoints_fromBoards,
                                  stats.Noutliers,
                                  solver_context->beforeStep, solver_context);
#endif

            reportFitMsg = "After";
#warning hook this up
            //        optimizerCallback(packed_state, NULL, NULL, NULL);
        }

        if(!problem_details.do_skip_regularization)
        {
            double norm2_err_regularization = 0;
            int    Nmeasurements_regularization =
                Ncameras*getNregularizationTerms_percamera(problem_details,
                                                           distortion_model);

            for(int i=0; i<Nmeasurements_regularization; i++)
            {
                double x = solver_context->beforeStep->x[Nmeasurements - Nmeasurements_regularization + i];
                norm2_err_regularization += x*x;
            }

            double norm2_err_nonregularization = norm2_error - norm2_err_regularization;
            double ratio_regularization_cost = norm2_err_regularization / norm2_err_nonregularization;

            if(VERBOSE)
            {
                for(int i=0; i<Nmeasurements_regularization; i++)
                {
                    double x = solver_context->beforeStep->x[Nmeasurements - Nmeasurements_regularization + i];
                    MSG("regularization %d: %f (squared: %f)", i, x, x*x);
                }
                MSG("norm2_error: %f", norm2_error);
                MSG("norm2_err_regularization: %f", norm2_err_regularization);
            }
            else
                MSG("regularization cost ratio: %g", ratio_regularization_cost);
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

    if( invJtJ_intrinsics_observations_only )
    {
        int Nintrinsics_per_camera = mrcal_getNintrinsicParams(distortion_model);
        bool result =
            computeUncertaintyMatrices(// out
                                       // dimensions (Ncameras,Nintrinsics_per_camera,Nintrinsics_per_camera)
                                       invJtJ_intrinsics_full,
                                       invJtJ_intrinsics_observations_only,

                                       // in
                                       distortion_model,
                                       problem_details,
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
            for(int i=0; i<Ncameras*Nintrinsics_per_camera*Nintrinsics_per_camera; i++)
            {
                invJtJ_intrinsics_full             [i] = nan;
                invJtJ_intrinsics_observations_only[i] = nan;
            }
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
    if(outside_ROI_indices_final)
    {
        stats.NoutsideROI = 0;
        if( roi != NULL )
        {
            for(int i_observation_board=0;
                i_observation_board<NobservationsBoard;
                i_observation_board++)
            {
                const observation_board_t* observation = &observations_board[i_observation_board];
                const int i_camera = observation->i_camera;
                for(int i_pt=0;
                    i_pt < calibration_object_width_n*calibration_object_width_n;
                    i_pt++)
                {
                    const point2_t* pt_observed = &observation->px[i_pt];
                    double weight = region_of_interest_weight(pt_observed, roi, i_camera);
                    if( weight != 1.0 )
                        outside_ROI_indices_final[stats.NoutsideROI++] =
                            i_observation_board*calibration_object_width_n*calibration_object_width_n +
                            i_pt;
                }
            }
        }
    }

 done:
    if(_solver_context == NULL && solver_context)
        dogleg_freeContext(&solver_context);

    free(markedOutliers);
    return stats;
}

// frees a dogleg_solverContext_t. I don't want to #include <dogleg.h> here, so
// this is void
void mrcal_free_context(void** ctx)
{
    if( *ctx == NULL )
        return;

    dogleg_freeContext((dogleg_solverContext_t**)ctx);
}
