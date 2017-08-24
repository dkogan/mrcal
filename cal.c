#include <stdio.h>
#include <stdlib.h>

#include <dogleg.h>
#include <minimath.h>
#include <assert.h>
#include <stdbool.h>

#warning am I actually using opencv? do i need to include this?
#include <opencv2/calib3d/calib3d.hpp>

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
#define SCALE_INTRINSICS_FOCAL_LENGTH 1.0
#define SCALE_INTRINSICS_CENTER_PIXEL 1.0
#define SCALE_ROTATION_CAMERA         1.0
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          1.0
#define SCALE_TRANSLATION_FRAME       1.0


#warning generalize
#define CALOBJECT_W                    10
#define NUM_POINTS_IN_CALOBJECT        (CALOBJECT_W*CALOBJECT_W)
#define CALIBRATION_OBJECT_DOT_SPACING (4.0 * 2.54 / 100.0) /* 4 inches */
static void get_refobject_point(int i_pt)
{
    int x = i_pt / CALOBJECT_W;
    int y = i_pt - x*CALOBJECT_W;

    union point3_t pt = {.x = (double)x* CALIBRATION_OBJECT_DOT_SPACING,
                         .y = (double)y* CALIBRATION_OBJECT_DOT_SPACING,
                         .z = 0.0 };
    return pt;
}


struct intrinsics_t
{
    double focal_xy [2];
    double center_xy[2];
#warning fill
    // double distortion[];
};
#define NUM_INTRINSIC_PARAMS sizeof(intrinsics_t)/sizeof(double)

// unconstrained 6DOF pose containing a rodrigues rotation and a translation
struct pose_t
{
    union point3_t r,t;
};

struct observation_t
{
#warning I need i_camera, but maybe i_frame should live in a separate frame_start[] ?
    int i_camera, i_frame;

    union point2_t px[NUM_POINTS_IN_CALOBJECT];
};




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
    return N*NUM_INTRINSIC_PARAMS;
}


#warning maybe this should project multiple points at a time?
static void project( // out
                    union point2_t* pt,
                    union point3_t* dxy_drcamera,
                    union point3_t* dxy_dtcamera,
                    union point3_t* dxy_drframe,
                    union point3_t* dxy_dtframe,

                    // in
                    // these are raw SCALED parameter vectors
                    const double* p_camera,
                    const double* p_frame,
                    bool extrinsic_are_identity,
                    int i_pt )
{
    union point3_t pt_ref = get_refobject_point(i_pt);

    if( extrinsic_are_identity )
    {
    }
}

// From real values, to scaled values. Optimizer sees scaled values
static void pack_solver_state( // out
                              double* p,

                              // in
                              const struct intrinsics_t* camera_intrinsics; // Ncameras of these
                              const struct pose_t*       camera_extrinsics; // Ncameras-1 of these
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
        // state_index_camera_intrinsics() can be branchless
        p[i_state++] = camera_extrinsics[i_camera].r.xyz[0]     / SCALE_ROTATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera].r.xyz[1]     / SCALE_ROTATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera].r.xyz[2]     / SCALE_ROTATION_CAMERA;

        p[i_state++] = camera_extrinsics[i_camera].t.xyz[0]     / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera].t.xyz[1]     / SCALE_TRANSLATION_CAMERA;
        p[i_state++] = camera_extrinsics[i_camera].t.xyz[2]     / SCALE_TRANSLATION_CAMERA;

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

    assert(Nstate == Nstate_ref);
}

// From scaled values, to real values. Optimizer sees scaled values
static void unpack_solver_state( // out
                                 struct intrinsics_t* camera_intrinsics; // Ncameras of these
                                 struct pose_t*       camera_extrinsics; // Ncameras-1 of these
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
        // state_index_camera_intrinsics() can be branchless
        camera_extrinsics[i_camera].r.xyz[0]     = p[i_state++] * SCALE_ROTATION_CAMERA;
        camera_extrinsics[i_camera].r.xyz[1]     = p[i_state++] * SCALE_ROTATION_CAMERA;
        camera_extrinsics[i_camera].r.xyz[2]     = p[i_state++] * SCALE_ROTATION_CAMERA;

        camera_extrinsics[i_camera].t.xyz[0]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        camera_extrinsics[i_camera].t.xyz[1]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;
        camera_extrinsics[i_camera].t.xyz[2]     = p[i_state++] * SCALE_TRANSLATION_CAMERA;

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

    assert(Nstate == Nstate_ref);
}

static int state_index_camera_ex_in_trinsics(int i_camera)
{
    // returns the variable index for the extrinsics-followed-by-intrinsics of a
    // given camera. Note that for i_camera==0 there are no extrinsics, so the
    // returned value will be negative, and the intrinsics live at index=0
    return i_camera*(NUM_INTRINSIC_PARAMS + 6) - 6;
}
static int state_index_frame_pose(int i_frame, int Ncameras)
{
    return i_frame*6 + (NUM_INTRINSIC_PARAMS + 6)*Ncameras - 6;
}

bool optimize( // out, in (seed on input)

               // These are the state. I don't have a state_t because Ncameras
               // and Nframes aren't known at compile time
               struct intrinsics_t* camera_intrinsics; // Ncameras of these
               struct pose_t*       camera_extrinsics; // Ncameras-1 of these
               struct pose_t*       frames;            // Nframes of these

               // in
               int Ncameras, Nframes;

               const struct observation_t* observations,
               int Nobservations,

               bool check_gradient)
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
    const int Nmeasurements = Nobservations * NUM_POINTS_IN_CALOBJECT;
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
#define STORE_JACOBIAN3(_col0, g0, g1, g2)       \
        do                                      \
        {                                       \
            int col0 = _col0;                   \
            Jcolidx[ iJacobian ] = col0 + 0;    \
            Jval   [ iJacobian ] = g0;          \
            iJacobian++;                        \
            Jcolidx[ iJacobian ] = col0 + 1;    \
            Jval   [ iJacobian ] = g1;          \
            iJacobian++;                        \
            Jcolidx[ iJacobian ] = col0 + 2;    \
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

            const double* p_camera =
                &solver_state[ state_index_camera_ex_in_trinsics(i_camera) ];
            const double* p_frame =
                &solver_state[ state_index_frame_pose(i_frame, Ncameras) ];

            for(int i_pt=0;
                i_pt < NUM_POINTS_IN_CALOBJECT;
                i_pt++)
            {
                Jrowptr[iMeasurement] = iJacobian;

                union point2_t pt_hypothesis;
                union point3_t dxy_drcamera[2];
                union point3_t dxy_dtcamera[2];
                union point3_t dxy_drframe [2];
                union point3_t dxy_dtframe [2];
                double dintrinsics[...];

                project(&pt_hypothesis,
                        dxy_drcamera,
                        dxy_dtcamera,
                        dxy_drframe,
                        dxy_dtframe,
                        p_camera, p_frame,
                        i_camera == 0,
                        i_pt);

                const union point2_t* pt_observed = &observation->px[i_pt];
                const double dx = pt_hypothesis.x - pt_observed->x;
                const double dy = pt_hypothesis.y - pt_observed->y;
                const double err2 = dx*dx + dy*dy;
                x[iMeasurement] = err2;

                static_assert(variable order; must be monotonic);
#error gradients need to be in respect to the PACKED variables
                STORE_JACOBIAN(ivar_intrinsics(i_camera),
                               2.0 * ( dx * dintrinsics[...] +
                                       dy * dintrinsics[...] ));
                STORE_JACOBIAN3( ivar_extrinsics_r0(i_camera),
                                 2.0 * ( dx * dxy_drcamera[0].xyz[0] + dy * dxy_drcamera[1].xyz[0] ),
                                 2.0 * ( dx * dxy_drcamera[0].xyz[1] + dy * dxy_drcamera[1].xyz[1] ),
                                 2.0 * ( dx * dxy_drcamera[0].xyz[2] + dy * dxy_drcamera[1].xyz[2] ))
                STORE_JACOBIAN3( ivar_extrinsics_t0(i_camera),
                                 2.0 * ( dx * dxy_dtcamera[0].xyz[0] + dy * dxy_dtcamera[1].xyz[0] ),
                                 2.0 * ( dx * dxy_dtcamera[0].xyz[1] + dy * dxy_dtcamera[1].xyz[1] ),
                                 2.0 * ( dx * dxy_dtcamera[0].xyz[2] + dy * dxy_dtcamera[1].xyz[2] ))
                STORE_JACOBIAN3( ivar_frame_r0(i_frame),
                                 2.0 * ( dx * dxy_drframe[0].xyz[0] + dy * dxy_drframe[1].xyz[0] ),
                                 2.0 * ( dx * dxy_drframe[0].xyz[1] + dy * dxy_drframe[1].xyz[1] ),
                                 2.0 * ( dx * dxy_drframe[0].xyz[2] + dy * dxy_drframe[1].xyz[2] ));
                STORE_JACOBIAN3( ivar_frame_t0(i_frame),
                                 2.0 * ( dx * dxy_dtframe[0].xyz[0] + dy * dxy_dtframe[1].xyz[0] ),
                                 2.0 * ( dx * dxy_dtframe[0].xyz[1] + dy * dxy_dtframe[1].xyz[1] ),
                                 2.0 * ( dx * dxy_dtframe[0].xyz[2] + dy * dxy_dtframe[1].xyz[2] ));

                iMeasurement++;
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
        optimizerCallback(state, NULL, NULL, NULL);
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
        optimizerCallback(state, NULL, NULL, NULL);
#endif
    }
    else
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, state,
                                Nstate, Nmeasurements, N_j_nonzero,
                                &optimizerCallback, NULL);

    // Return RMS reprojection error
    return sqrt(norm2_error / (double)Nmeasurements);
}
