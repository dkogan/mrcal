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


static union point3_t get_refobject_point(int i_pt)
{
    int y = i_pt / CALOBJECT_W;
    int x = i_pt - y*CALOBJECT_W;

    union point3_t pt = {.x = (double)x* CALIBRATION_OBJECT_DOT_SPACING,
                         .y = (double)y* CALIBRATION_OBJECT_DOT_SPACING,
                         .z = 0.0 };
    return pt;
}

static int get_Nstate(int Ncameras, int Nframes)
{
    return
        Ncameras * NUM_INTRINSIC_PARAMS + // camera intrinsics
        (Ncameras-1) * 6                + // camera extrinsics
        Nframes      * 6;                 // frame poses
}

static int get_N_j_nonzero( const struct observation_t* observations,
                            int Nobservations )
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // initial estimate counts extrinsics for camera0, which need to be
    // subtracted off
    int N = Nobservations * (NUM_INTRINSIC_PARAMS + 6 + 6);
    for(int i=0; i<Nobservations; i++)
        if(observations[i].i_camera == 0)
            N -= 6;
    return N*NUM_POINTS_IN_CALOBJECT*2; // *2 because I have separate x and y measurements
}


#warning maybe this should project multiple points at a time?
static union point2_t project( // out
                              struct intrinsics_t* dxy_dintrinsics,
                              union point3_t* dxy_drcamera,
                              union point3_t* dxy_dtcamera,
                              union point3_t* dxy_drframe,
                              union point3_t* dxy_dtframe,

                              // in
                              // these are the unit-scale parameter vectors touched by the solver
                              const double* p_camera_rt_intrinsics_unitscale,
                              const double* p_frame_rt_unitscale,
                              bool camera_at_identity,
                              int i_pt )
{
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

    CvMat d_rj_rf = cvMat(3,3, CV_64FC1, _d_rj_rf);
    CvMat d_rj_tf = cvMat(3,3, CV_64FC1, _d_rj_tf);
    CvMat d_rj_rc = cvMat(3,3, CV_64FC1, _d_rj_rc);
    CvMat d_rj_tc = cvMat(3,3, CV_64FC1, _d_rj_tc);
    CvMat d_tj_rf = cvMat(3,3, CV_64FC1, _d_tj_rf);
    CvMat d_tj_tf = cvMat(3,3, CV_64FC1, _d_tj_tf);
    CvMat d_tj_rc = cvMat(3,3, CV_64FC1, _d_tj_rc);
    CvMat d_tj_tc = cvMat(3,3, CV_64FC1, _d_tj_tc);

    double _tj[3];
    CvMat  tj = cvMat(3,1,CV_64FC1, _tj);

    double _rf[3];
    double _tf[3];
    for(int i=0; i<3; i++)
    {
        _rf[i] = p_frame_rt_unitscale[i+0] * SCALE_ROTATION_FRAME;
        _tf[i] = p_frame_rt_unitscale[i+3] * SCALE_TRANSLATION_FRAME;
    }
    CvMat rf = cvMat(3,1, CV_64FC1, &_rf);
    CvMat tf = cvMat(3,1, CV_64FC1, &_tf);

    double _Rj[3*3];
    CvMat  Rj = cvMat(3,3,CV_64FC1, _Rj);
    double _d_Rj_rj[9*3];
    CvMat d_Rj_rj = cvMat(9,3,CV_64F, _d_Rj_rj);

    union point3_t pt_ref = get_refobject_point(i_pt);
    union point3_t pt_cam;

    if(!camera_at_identity)
    {
        double _rc[3];
        double _tc[3];
        for(int i=0; i<3; i++)
        {
            _rc[i] = p_camera_rt_intrinsics_unitscale[i+0] * SCALE_ROTATION_CAMERA;
            _tc[i] = p_camera_rt_intrinsics_unitscale[i+3] * SCALE_TRANSLATION_CAMERA;
        }
        CvMat rc = cvMat(3,1, CV_64FC1, &_rc);
        CvMat tc = cvMat(3,1, CV_64FC1, &_tc);

        double _rj[3];
        CvMat  rj = cvMat(3,1,CV_64FC1, _rj);
        cvComposeRT( &rf,      &tf,
                     &rc,      &tc,
                     &rj,      &tj,
                     &d_rj_rf, &d_rj_tf,
                     &d_rj_rc, &d_rj_tc,
                     &d_tj_rf, &d_tj_tf,
                     &d_tj_rc, &d_tj_tc );
        cvRodrigues2(&rj, &Rj, &d_Rj_rj);

        // Rj * pt + tj -> pt
        mul_vec3_gen33t_vout(pt_ref.xyz, _Rj, pt_cam.xyz);
        add_vec(3, pt_cam.xyz,  _tj);
    }
    else
    {
        // We're looking at camera0, so this camera transform is fixed at the
        // identity. We don't need to compose anything, nor propagate gradients
        // for the camera extrinsics, since those don't exist in the parameter
        // vector

        // Here the "joint" transform is just the "frame" transform
        cvRodrigues2(&rf, &Rj, &d_Rj_rj);

        // Rj * pt + tj -> pt
        mul_vec3_gen33t_vout(pt_ref.xyz, _Rj, pt_cam.xyz);
        add_vec(3, pt_cam.xyz,  _tf);
    }

    // pt is now in the camera coordinates. I can project
    union point2_t pt_out;
    const struct intrinsics_t* intrinsics = (const struct intrinsics_t*)(&p_camera_rt_intrinsics_unitscale[6]);
    const double sx = intrinsics->focal_xy[0] * SCALE_INTRINSICS_FOCAL_LENGTH;
    const double sy = intrinsics->focal_xy[1] * SCALE_INTRINSICS_FOCAL_LENGTH;
    double z_recip = 1.0 / pt_cam.z;
    pt_out.x =
        pt_cam.x*z_recip * sx +
        intrinsics->center_xy[0] * SCALE_INTRINSICS_CENTER_PIXEL;
    pt_out.y =
        pt_cam.y*z_recip * sy +
        intrinsics->center_xy[1] * SCALE_INTRINSICS_CENTER_PIXEL;


    // I have the projection, and I now need to propagate the gradients
    memset(dxy_dintrinsics, 0, 2*sizeof(dxy_dintrinsics[0]));
    dxy_dintrinsics[0].focal_xy [0] = pt_cam.x*z_recip * SCALE_INTRINSICS_FOCAL_LENGTH;
    dxy_dintrinsics[0].center_xy[0] = SCALE_INTRINSICS_CENTER_PIXEL;
    dxy_dintrinsics[1].focal_xy [1] = pt_cam.y*z_recip * SCALE_INTRINSICS_FOCAL_LENGTH;
    dxy_dintrinsics[1].center_xy[1] = SCALE_INTRINSICS_CENTER_PIXEL;


    if(!camera_at_identity)
    {
        // I do this multiple times, one each for {r,t}{camera,frame}
        void propagate(union point3_t* dxy_dparam,
                       const double* _d_rj,
                       const double* _d_tj,

                       // I want the gradients in respect to the unit-scale params
                       double scale_param)
        {
            // dxy_drcamera[0] = sx/pt_cam.z * (d(pt_cam.x) - pt_cam.x/pt_cam.z * d(pt_cam.z));
            // dxy_drcamera[1] = sy/pt_cam.z * (d(pt_cam.y) - pt_cam.y/pt_cam.z * d(pt_cam.z));
            //
            // pt_cam.x    = Rj[row0]*pt_ref + tj.x
            // d(pt_cam.x) = d(Rj[row0])*pt_ref + d(tj.x);
            // dRj[row0]/drj is 3x3 matrix at &_d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc

            double d_ptcamx[3];
            double d_ptcamy[3];
            double d_ptcamz[3];

            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*0], d_ptcamx );
            mul_vec3_gen33     ( d_ptcamx,   _d_rj);
            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*1], d_ptcamy );
            mul_vec3_gen33     ( d_ptcamy,   _d_rj);
            mul_vec3_gen33_vout( pt_ref.xyz, &_d_Rj_rj[9*2], d_ptcamz );
            mul_vec3_gen33     ( d_ptcamz,   _d_rj);

            add_vec(3, d_ptcamx, &_d_tj[3*0] );
            add_vec(3, d_ptcamy, &_d_tj[3*1] );
            add_vec(3, d_ptcamz, &_d_tj[3*2] );

            for(int i=0; i<3; i++)
            {
                dxy_dparam[0].xyz[i] = scale_param *
                    sx * z_recip * (d_ptcamx[i] - pt_cam.x * z_recip * d_ptcamz[i]);
                dxy_dparam[1].xyz[i] = scale_param *
                    sy * z_recip * (d_ptcamy[i] - pt_cam.y * z_recip * d_ptcamz[i]);
            }
        }

        propagate( dxy_drcamera, _d_rj_rc, _d_tj_rc, SCALE_ROTATION_CAMERA    );
        propagate( dxy_dtcamera, _d_rj_tc, _d_tj_tc, SCALE_TRANSLATION_CAMERA );
        propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf, SCALE_ROTATION_FRAME     );
        propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf, SCALE_TRANSLATION_FRAME  );
    }
    else
    {
        void propagate_r(union point3_t* dxy_dparam,

                         // I want the gradients in respect to the unit-scale params
                         double scale_param)
        {
            // dxy_drcamera[0] = sx/pt_cam.z * (d(pt_cam.x) - pt_cam.x/pt_cam.z * d(pt_cam.z));
            // dxy_drcamera[1] = sy/pt_cam.z * (d(pt_cam.y) - pt_cam.y/pt_cam.z * d(pt_cam.z));
            //
            // pt_cam.x    = Rj[row0]*pt_ref + tj.x
            // d(pt_cam.x) = d(Rj[row0])*pt_ref + d(tj.x);
            // dRj[row0]/drj is 3x3 matrix at &_d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc

            double d_ptcamx[3];
            double d_ptcamy[3];
            double d_ptcamz[3];

            memcpy(d_ptcamx, pt_ref.xyz, sizeof(d_ptcamx));
            mul_vec3_gen33_vout( d_ptcamx,   &_d_Rj_rj[9*1], d_ptcamy);
            mul_vec3_gen33_vout( d_ptcamx,   &_d_Rj_rj[9*2], d_ptcamz);
            mul_vec3_gen33     ( d_ptcamx,   &_d_Rj_rj[9*0]);

            for(int i=0; i<3; i++)
            {
                dxy_dparam[0].xyz[i] = scale_param *
                    sx * z_recip * (d_ptcamx[i] - pt_cam.x * z_recip * d_ptcamz[i]);
                dxy_dparam[1].xyz[i] = scale_param *
                    sy * z_recip * (d_ptcamy[i] - pt_cam.y * z_recip * d_ptcamz[i]);
            }
        }
        void propagate_t(union point3_t* dxy_dparam,

                         // I want the gradients in respect to the unit-scale params
                         double scale_param)
        {
            // dxy_drcamera[0] = sx/pt_cam.z * (d(pt_cam.x) - pt_cam.x/pt_cam.z * d(pt_cam.z));
            // dxy_drcamera[1] = sy/pt_cam.z * (d(pt_cam.y) - pt_cam.y/pt_cam.z * d(pt_cam.z));
            //
            // pt_cam.x    = Rj[row0]*pt_ref + tj.x
            // d(pt_cam.x) = d(Rj[row0])*pt_ref + d(tj.x);
            // dRj[row0]/drj is 3x3 matrix at &_d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc


            dxy_dparam[0].xyz[0] = scale_param * sx * z_recip;
            dxy_dparam[1].xyz[0] = 0.0;

            dxy_dparam[0].xyz[1] = 0.0;
            dxy_dparam[1].xyz[1] = scale_param * sy * z_recip;

            dxy_dparam[0].xyz[2] = -scale_param * sx * z_recip * pt_cam.x * z_recip;
            dxy_dparam[1].xyz[2] = -scale_param * sy * z_recip * pt_cam.y * z_recip;
        }

        propagate_r( dxy_drframe, SCALE_ROTATION_FRAME     );
        propagate_t( dxy_dtframe, SCALE_TRANSLATION_FRAME  );
    }

    return pt_out;
}

// From real values to unit-scale values. Optimizer sees unit-scale values
static void pack_solver_state( // out
                              double* p,

                              // in
                              const struct intrinsics_t* camera_intrinsics, // Ncameras of these
                              const struct pose_t*       camera_extrinsics, // Ncameras-1 of these
                              const struct pose_t*       frames,            // Nframes of these
                              int Ncameras, int Nframes,

                              int Nstate_ref)

{
    int i_state = 0;

    p[i_state++] = camera_intrinsics[0].focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
    p[i_state++] = camera_intrinsics[0].focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
    p[i_state++] = camera_intrinsics[0].center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
    p[i_state++] = camera_intrinsics[0].center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
    for(int i_camera=1; i_camera < Ncameras; i_camera++)
    {
        // extrinsics first so that the intrinsics are evenly spaced and
        // state_index_camera_rt_intrinsics() can be branchless
        p[i_state++] = camera_extrinsics[i_camera-1].r.xyz[0]     / SCALE_ROTATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera-1].r.xyz[1]     / SCALE_ROTATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera-1].r.xyz[2]     / SCALE_ROTATION_CAMERA;

        p[i_state++] = camera_extrinsics[i_camera-1].t.xyz[0]     / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera-1].t.xyz[1]     / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera-1].t.xyz[2]     / SCALE_TRANSLATION_CAMERA;

        p[i_state++] = camera_intrinsics[i_camera].focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
        p[i_state++] = camera_intrinsics[i_camera].focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
        p[i_state++] = camera_intrinsics[i_camera].center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
        p[i_state++] = camera_intrinsics[i_camera].center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
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

    assert(i_state == Nstate_ref);
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out
                                 struct intrinsics_t* camera_intrinsics, // Ncameras of these
                                 struct pose_t*       camera_extrinsics, // Ncameras-1 of these
                                 struct pose_t*       frames,            // Nframes of these

                                 // in
                                 const double* p,
                                 int Ncameras, int Nframes,

                                 int Nstate_ref)

{
    int i_state = 0;

    camera_intrinsics[0].focal_xy [0] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
    camera_intrinsics[0].focal_xy [1] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
    camera_intrinsics[0].center_xy[0] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
    camera_intrinsics[0].center_xy[1] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
    for(int i_camera=1; i_camera < Ncameras; i_camera++)
    {
        // extrinsics first so that the intrinsics are evenly spaced and
        // state_index_camera_rt_intrinsics() can be branchless
        camera_extrinsics[i_camera-1].r.xyz[0]     = p[i_state++] * SCALE_ROTATION_CAMERA;
        camera_extrinsics[i_camera-1].r.xyz[1]     = p[i_state++] * SCALE_ROTATION_CAMERA;
        camera_extrinsics[i_camera-1].r.xyz[2]     = p[i_state++] * SCALE_ROTATION_CAMERA;

        camera_extrinsics[i_camera-1].t.xyz[0]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        camera_extrinsics[i_camera-1].t.xyz[1]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        camera_extrinsics[i_camera-1].t.xyz[2]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;

        camera_intrinsics[i_camera].focal_xy [0] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        camera_intrinsics[i_camera].focal_xy [1] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
        camera_intrinsics[i_camera].center_xy[0] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
        camera_intrinsics[i_camera].center_xy[1] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
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

    assert(i_state == Nstate_ref);
}

static int state_index_camera_rt_intrinsics(int i_camera)
{
    // returns the variable index for the extrinsics-followed-by-intrinsics of a
    // given camera. Note that for i_camera==0 there are no extrinsics, so the
    // returned value will be negative, and the intrinsics live at index=0
    return i_camera*(NUM_INTRINSIC_PARAMS + 6) - 6;
}
static int state_index_frame_rt(int i_frame, int Ncameras)
{
    return i_frame*6 + (NUM_INTRINSIC_PARAMS + 6)*Ncameras - 6;
}

double mrcal_optimize( // out, in (seed on input)

                      // These are the state. I don't have a state_t because Ncameras
                      // and Nframes aren't known at compile time
                      struct intrinsics_t* camera_intrinsics, // Ncameras of these
                      struct pose_t*       camera_extrinsics, // Ncameras-1 of these. Transform FROM camera0 frame
                      struct pose_t*       frames,            // Nframes of these.    Transform TO   camera0 frame

                      // in
                      int Ncameras, int Nframes,

                      const struct observation_t* observations,
                      int Nobservations,

                      bool check_gradient,
                      enum distortion_model_t distortion_model,
                      bool do_optimize_intrinsics)
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


    const int Nstate        = get_Nstate(Ncameras, Nframes);
    const int Nmeasurements = Nobservations * NUM_POINTS_IN_CALOBJECT * 2; // *2 because I have separate x and y measurements
    const int N_j_nonzero   = get_N_j_nonzero(observations, Nobservations);



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




        for(int i_observation = 0;
            i_observation < Nobservations;
            i_observation++)
        {
            const struct observation_t* observation = &observations[i_observation];

            const int i_camera = observation->i_camera;
            const int i_frame  = observation->i_frame;

            const double* p_camera_rt_intrinsics =
                &solver_state[ state_index_camera_rt_intrinsics(i_camera) ];
            const double* p_frame_rt =
                &solver_state[ state_index_frame_rt(i_frame, Ncameras) ];

            for(int i_pt=0;
                i_pt < NUM_POINTS_IN_CALOBJECT;
                i_pt++)
            {
                // these are computed in respect to the unit-scale parameters
                // used by the optimizer
                struct intrinsics_t dxy_dintrinsics[2];
                union point3_t dxy_drcamera[2];
                union point3_t dxy_dtcamera[2];
                union point3_t dxy_drframe [2];
                union point3_t dxy_dtframe [2];

                union point2_t pt_hypothesis =
                    project(dxy_dintrinsics,
                            dxy_drcamera,
                            dxy_dtcamera,
                            dxy_drframe,
                            dxy_dtframe,
                            p_camera_rt_intrinsics, p_frame_rt,
                            i_camera == 0,
                            i_pt);

                const union point2_t* pt_observed = &observation->px[i_pt];

                // I have my two measurements (dx, dy). I propagate their
                // gradient and store them
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    Jrowptr[iMeasurement] = iJacobian;

                    const double err = pt_hypothesis.xy[i_xy] - pt_observed->xy[i_xy];
                    x[iMeasurement] = err;

                    // I want these gradient values to be computed in
                    // monotonically-increasing order of variable index. I don't
                    // CHECK, so it's the developer's responsibility to make sure.
                    // This ordering is set in the intrinsics_t structure and in
                    // pack_solver_state(), unpack_solver_state()
                    int i_var = state_index_camera_rt_intrinsics(i_camera);
                    if( i_camera != 0 )
                    {
                        STORE_JACOBIAN3( i_var + 0,
                                         dxy_drcamera[i_xy].xyz[0],
                                         dxy_drcamera[i_xy].xyz[1],
                                         dxy_drcamera[i_xy].xyz[2]);
                        STORE_JACOBIAN3( i_var + 3,
                                         dxy_dtcamera[i_xy].xyz[0],
                                         dxy_dtcamera[i_xy].xyz[1],
                                         dxy_dtcamera[i_xy].xyz[2]);
                    }

                    for(int i=0; i<NUM_INTRINSIC_PARAMS; i++)
                        STORE_JACOBIAN( i_var + 6 + i,
                                        ((const double*)&dxy_dintrinsics[i_xy])[i] );

                    i_var = state_index_frame_rt(i_frame, Ncameras);
                    STORE_JACOBIAN3( i_var + 0,
                                     dxy_drframe[i_xy].xyz[0],
                                     dxy_drframe[i_xy].xyz[1],
                                     dxy_drframe[i_xy].xyz[2]);
                    STORE_JACOBIAN3( i_var + 3,
                                     dxy_dtframe[i_xy].xyz[0],
                                     dxy_dtframe[i_xy].xyz[1],
                                     dxy_dtframe[i_xy].xyz[2]);

                    iMeasurement++;
                }
            }
        }

        // required to indicate the end of the jacobian matrix
        Jrowptr[iMeasurement] = iJacobian;

        assert(iMeasurement == Nmeasurements);
        assert(iJacobian    == N_j_nonzero  );
    }








    double state[Nstate];
    pack_solver_state(state,
                      camera_intrinsics,
                      camera_extrinsics,
                      frames,
                      Ncameras, Nframes, Nstate);

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
        unpack_solver_state( camera_intrinsics, // Ncameras of these
                             camera_extrinsics, // Ncameras-1 of these
                             frames,            // Nframes of these
                             state,
                             Ncameras, Nframes, Nstate);


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

const char* mrcal_distortion_model_name( enum distortion_model_t model )
{
    switch(model)
    {
#define CASE_STRING(s) case s: return #s;
        DISTORTION_LIST( CASE_STRING )
    }
    return NULL;
}
