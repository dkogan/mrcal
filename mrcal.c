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
static int getNintrinsicOptimizationParams(bool do_optimize_intrinsics, enum distortion_model_t distortion_model)
{
    if( !do_optimize_intrinsics ) return 0;
    return getNintrinsicParams( distortion_model );
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

static int get_Nstate(int Ncameras, int Nframes,
                      bool do_optimize_intrinsics,
                      enum distortion_model_t distortion_model)
{
    return
        (Ncameras-1) * 6 + // camera extrinsics
        Nframes      * 6 + // frame poses
        Ncameras * getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model); // camera intrinsics
}

static int get_N_j_nonzero( const struct observation_t* observations,
                            int Nobservations,
                            bool do_optimize_intrinsics,
                            enum distortion_model_t distortion_model)
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // initial estimate counts extrinsics for camera0, which need to be
    // subtracted off
    int N = Nobservations * (6 + 6 + getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model));
    for(int i=0; i<Nobservations; i++)
        if(observations[i].i_camera == 0)
            N -= 6;
    return N*NUM_POINTS_IN_CALOBJECT*2; // *2 because I have separate x and y measurements
}



#warning maybe this should project multiple points at a time?
static union point2_t project( // out
                              double*         dxy_dintrinsics,
                              union point3_t* dxy_drcamera,
                              union point3_t* dxy_dtcamera,
                              union point3_t* dxy_drframe,
                              union point3_t* dxy_dtframe,

                              // in
                              // these are the unit-scale parameter vectors touched by the solver
                              const double* p_intrinsics_unitscale,
                              const double* p_camera_rt_unitscale,
                              const double* p_frame_rt_unitscale,
                              bool camera_at_identity,
                              enum distortion_model_t distortion_model,
                              int i_pt )
{
    int NdistortionParams = mrcal_getNdistortionParams(distortion_model);
    int NintrinsicParams  = getNintrinsicParams (distortion_model);

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

    double _rf[3];
    double _tf[3];
    for(int i=0; i<3; i++)
    {
        _rf[i] = p_frame_rt_unitscale[i+0] * SCALE_ROTATION_FRAME;
        _tf[i] = p_frame_rt_unitscale[i+3] * SCALE_TRANSLATION_FRAME;
    }
    CvMat rf = cvMat(3,1, CV_64FC1, &_rf);
    CvMat tf = cvMat(3,1, CV_64FC1, &_tf);

    union point3_t pt_ref = get_refobject_point(i_pt);

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


    const struct intrinsics_t* intrinsics = (const struct intrinsics_t*)p_intrinsics_unitscale;

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

        double fx = intrinsics->focal_xy [0] * SCALE_INTRINSICS_FOCAL_LENGTH;
        double fy = intrinsics->focal_xy [1] * SCALE_INTRINSICS_FOCAL_LENGTH;
        double cx = intrinsics->center_xy[0] * SCALE_INTRINSICS_CENTER_PIXEL;
        double cy = intrinsics->center_xy[1] * SCALE_INTRINSICS_CENTER_PIXEL;

        double _camera_matrix[] = { fx,  0, cx,
                                     0, fy, cy,
                                     0,  0,  1 };
        CvMat camera_matrix = cvMat(3,3, CV_64FC1, _camera_matrix);
        double _distortions[NdistortionParams];
        for(int i=0; i<NdistortionParams; i++)
            _distortions[i] = intrinsics->distortions[i] * SCALE_DISTORTION;
        CvMat distortions = cvMat( NdistortionParams, 1, CV_64FC1, _distortions);

        // dpdf, dpddistortions should be views into dxy_dintrinsics[],
        // but it doesn't work. I suspect OpenCV has a bug, but debugging this
        // is taking too much time, so I just copy stuff instead. I wanted this:
        //
        // CvMat  dpdf           = cvMat( 2, 2,
        //                                CV_64FC1, ((struct intrinsics_t*)dxy_dintrinsics)->focal_xy);
        // CvMat  dpddistortions = cvMat( 2, NdistortionParams,
        //                                CV_64FC1, ((struct intrinsics_t*)dxy_dintrinsics)->distortions);
        // dpdf.step = dpddistortions.step = sizeof(double) * NintrinsicParams;
        double _dpdf[2*2];
        CvMat dpdf = cvMat(2,2, CV_64FC1, _dpdf);
        double _dpddistortions[2*NdistortionParams];
        CvMat dpddistortions = cvMat(2, NdistortionParams, CV_64FC1, _dpddistortions);
        // instead I do this ^^^^^^^^^^^^^^^^


        CvMat* p_dpdf;
        CvMat* p_dpddistortions;

        if(dxy_dintrinsics == NULL)
            p_dpdf = p_dpddistortions = NULL;
        else
        {
            p_dpdf           = &dpdf;
            p_dpddistortions = &dpddistortions;
        }

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


        if(dxy_dintrinsics != NULL)
        {
            struct intrinsics_t* dxy_dintrinsics0 = (struct intrinsics_t*)dxy_dintrinsics;
            struct intrinsics_t* dxy_dintrinsics1 = (struct intrinsics_t*)&dxy_dintrinsics[NintrinsicParams];

            dxy_dintrinsics0->focal_xy [0] = _dpdf[0] * SCALE_INTRINSICS_FOCAL_LENGTH;
            dxy_dintrinsics0->center_xy[0] = SCALE_INTRINSICS_CENTER_PIXEL;
            dxy_dintrinsics0->focal_xy [1] = 0.0;
            dxy_dintrinsics0->center_xy[1] = 0.0;
            dxy_dintrinsics1->focal_xy [0] = 0.0;
            dxy_dintrinsics1->center_xy[0] = 0.0;
            dxy_dintrinsics1->focal_xy [1] = _dpdf[3] * SCALE_INTRINSICS_FOCAL_LENGTH;
            dxy_dintrinsics1->center_xy[1] = SCALE_INTRINSICS_CENTER_PIXEL;

            for(int i=0; i<NdistortionParams; i++)
            {
                dxy_dintrinsics0->distortions[i] = _dpddistortions[i + 0*NdistortionParams] * SCALE_DISTORTION;
                dxy_dintrinsics1->distortions[i] = _dpddistortions[i + 1*NdistortionParams] * SCALE_DISTORTION;
            }
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
            propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf, SCALE_ROTATION_FRAME     );
            propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf, SCALE_TRANSLATION_FRAME  );
        }
        else
        {
            // My gradient is already computed, I just need to scale it

            for(int i=0; i<3; i++)
            {
                dxy_drframe[0].xyz[i] = _dxy_drj[i + 3*0] * SCALE_ROTATION_FRAME;
                dxy_dtframe[0].xyz[i] = _dxy_dtj[i + 3*0] * SCALE_TRANSLATION_FRAME;
                dxy_drframe[1].xyz[i] = _dxy_drj[i + 3*1] * SCALE_ROTATION_FRAME;
                dxy_dtframe[1].xyz[i] = _dxy_dtj[i + 3*1] * SCALE_TRANSLATION_FRAME;
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
        double theta = intrinsics->distortions[0] * SCALE_DISTORTION;
        double phi   = intrinsics->distortions[1] * SCALE_DISTORTION;
        double r0    = intrinsics->distortions[2] * SCALE_DISTORTION;
        double r1    = intrinsics->distortions[3] * SCALE_DISTORTION;
        double r2    = intrinsics->distortions[4] * SCALE_DISTORTION;

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

            pt_cam.xyz[i] = pt_cam.xyz[i] * (mu+1.0) - mu*omega*o[i];
        }
    }
    else if( distortion_model == DISTORTION_CAHVORE )
    {
        // set ddistortion_dxyz

        fprintf(stderr, "CAHVORE not implmented yet\n");
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
    const double fx = intrinsics->focal_xy[0] * SCALE_INTRINSICS_FOCAL_LENGTH;
    const double fy = intrinsics->focal_xy[1] * SCALE_INTRINSICS_FOCAL_LENGTH;
    double z_recip = 1.0 / pt_cam.z;
    pt_out.x =
        pt_cam.x*z_recip * fx +
        intrinsics->center_xy[0] * SCALE_INTRINSICS_CENTER_PIXEL;
    pt_out.y =
        pt_cam.y*z_recip * fy +
        intrinsics->center_xy[1] * SCALE_INTRINSICS_CENTER_PIXEL;

    // I have the projection, and I now need to propagate the gradients
    if(dxy_dintrinsics != NULL)
    {
        struct intrinsics_t* dxy_dintrinsics0 = (struct intrinsics_t*)dxy_dintrinsics;
        struct intrinsics_t* dxy_dintrinsics1 = (struct intrinsics_t*)&dxy_dintrinsics[NintrinsicParams];

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

        for(int i=0; i<NdistortionParams; i++)
        {
            const double dx = dxyz_ddistortion[i + 0*NdistortionParams];
            const double dy = dxyz_ddistortion[i + 1*NdistortionParams];
            const double dz = dxyz_ddistortion[i + 2*NdistortionParams];
            dxy_dintrinsics0->distortions[i] = SCALE_DISTORTION * fx * z_recip * (dx - pt_cam.x*z_recip*dz);
            dxy_dintrinsics1->distortions[i] = SCALE_DISTORTION * fy * z_recip * (dy - pt_cam.y*z_recip*dz);
        }
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
        propagate( dxy_drframe,  _d_rj_rf, _d_tj_rf, SCALE_ROTATION_FRAME     );
        propagate( dxy_dtframe,  _d_rj_tf, _d_tj_tf, SCALE_TRANSLATION_FRAME  );
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
                                         int Ncameras )
{
    int i_state      = 0;
    int Ndistortions = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

    for(int i_camera=0; i_camera < Ncameras; i_camera++)
    {
        p[i_state++] = intrinsics->focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
        p[i_state++] = intrinsics->focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
        p[i_state++] = intrinsics->center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
        p[i_state++] = intrinsics->center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;

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
                              int Ncameras, int Nframes,

                              int Nstate_ref,
                              bool do_optimize_intrinsics)
{
    int i_state = 0;

    if( do_optimize_intrinsics )
        i_state += pack_solver_state_intrinsics( p, intrinsics,
                                                 distortion_model,
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

    assert(i_state == Nstate_ref);
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out
                                 struct intrinsics_t* intrinsics, // Ncameras of these
                                 struct pose_t*       extrinsics, // Ncameras-1 of these
                                 struct pose_t*       frames,     // Nframes of these

                                 // in
                                 const double* p,
                                 const enum distortion_model_t distortion_model,
                                 int Ncameras, int Nframes,

                                 int Nstate_ref,
                                 bool do_optimize_intrinsics)

{
    int i_state = 0;

    if(do_optimize_intrinsics)
    {
        int Ndistortions = mrcal_getNdistortionParams(distortion_model);
        int Nintrinsics  = Ndistortions + N_INTRINSICS_CORE;

        for(int i_camera=0; i_camera < Ncameras; i_camera++)
        {
            intrinsics->focal_xy [0] = p[i_state++] *  SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics->focal_xy [1] = p[i_state++] *  SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics->center_xy[0] = p[i_state++] *  SCALE_INTRINSICS_CENTER_PIXEL;
            intrinsics->center_xy[1] = p[i_state++] *  SCALE_INTRINSICS_CENTER_PIXEL;

            for(int i=0; i<Ndistortions; i++)
                intrinsics->distortions[i] = p[i_state++] * SCALE_DISTORTION;

            intrinsics = (struct intrinsics_t*)&((double*)intrinsics)[Nintrinsics];
        }
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

    assert(i_state == Nstate_ref);
}

static int state_index_intrinsics(int i_camera,
                                  bool do_optimize_intrinsics,
                                  enum distortion_model_t distortion_model)
{
    // returns the variable index for the extrinsics-followed-by-intrinsics of a
    // given camera. Note that for i_camera==0 there are no extrinsics, so the
    // returned value will be negative, and the intrinsics live at index=0
    return i_camera * getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model);
}
static int state_index_camera_rt(int i_camera, int Ncameras,
                                 bool do_optimize_intrinsics,
                                 enum distortion_model_t distortion_model)
{
    // returns the variable index for the extrinsics-followed-by-intrinsics of a
    // given camera. Note that for i_camera==0 there are no extrinsics, so the
    // returned value will be negative, and the intrinsics live at index=0
    return
        getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model)*Ncameras +
        (i_camera-1)*6;
}
static int state_index_frame_rt(int i_frame, int Ncameras,
                                bool do_optimize_intrinsics,
                                enum distortion_model_t distortion_model)
{
    return
        Ncameras * (getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model) + 6) - 6 +
        i_frame*6;
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


    const int Nstate        = get_Nstate(Ncameras, Nframes, do_optimize_intrinsics, distortion_model);
    const int Nmeasurements = Nobservations * NUM_POINTS_IN_CALOBJECT * 2; // *2 because I have separate x and y measurements
    const int N_j_nonzero   = get_N_j_nonzero(observations, Nobservations, do_optimize_intrinsics, distortion_model);

    double intrinsics_unitscale[Ncameras * getNintrinsicParams(distortion_model)];
    if( !do_optimize_intrinsics )
        pack_solver_state_intrinsics( intrinsics_unitscale,
                                      intrinsics, distortion_model,
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




        for(int i_observation = 0;
            i_observation < Nobservations;
            i_observation++)
        {
            const struct observation_t* observation = &observations[i_observation];

            const int i_camera = observation->i_camera;
            const int i_frame  = observation->i_frame;

            const int     i_var_intrinsics = state_index_intrinsics(i_camera, do_optimize_intrinsics, distortion_model);
            const int     i_var_camera_rt  = state_index_camera_rt (i_camera, Ncameras, do_optimize_intrinsics, distortion_model);
            const int     i_var_frame_rt   = state_index_frame_rt  (i_frame, Ncameras, do_optimize_intrinsics, distortion_model);
            const double* p_intrinsics     = &solver_state[ i_var_intrinsics ];
            const double* p_camera_rt      = &solver_state[ i_var_camera_rt ];
            const double* p_frame_rt       = &solver_state[ i_var_frame_rt ];

            if( !do_optimize_intrinsics )
                p_intrinsics = &intrinsics_unitscale[i_camera * getNintrinsicParams(distortion_model)];

            for(int i_pt=0;
                i_pt < NUM_POINTS_IN_CALOBJECT;
                i_pt++)
            {
                // these are computed in respect to the unit-scale parameters
                // used by the optimizer
                double dxy_dintrinsics[2 * getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model)];
                union point3_t dxy_drcamera[2];
                union point3_t dxy_dtcamera[2];
                union point3_t dxy_drframe [2];
                union point3_t dxy_dtframe [2];

                union point2_t pt_hypothesis =
                    project(do_optimize_intrinsics ? dxy_dintrinsics : NULL,
                            dxy_drcamera,
                            dxy_dtcamera,
                            dxy_drframe,
                            dxy_dtframe,
                            p_intrinsics, p_camera_rt, p_frame_rt,
                            i_camera == 0,
                            distortion_model,
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
                    if(do_optimize_intrinsics)
                        for(int i=0;
                            i<getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model);
                            i++)
                        {
                            STORE_JACOBIAN( i_var_intrinsics + i,
                                            dxy_dintrinsics[i_xy *
                                                            getNintrinsicOptimizationParams(do_optimize_intrinsics, distortion_model) +
                                                            i] );
                        }
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
        }

        // required to indicate the end of the jacobian matrix
        Jrowptr[iMeasurement] = iJacobian;

        assert(iMeasurement == Nmeasurements);
        assert(iJacobian    == N_j_nonzero  );
    }








    double state[Nstate];
    pack_solver_state(state,
                      intrinsics,
                      distortion_model,
                      extrinsics,
                      frames,
                      Ncameras, Nframes, Nstate,
                      do_optimize_intrinsics);

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
                             state,
                             distortion_model,
                             Ncameras, Nframes, Nstate,
                             do_optimize_intrinsics);


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
