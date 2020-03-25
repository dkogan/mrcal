#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mrcal.h"

int main(int argc, char* argv[] )
{
    const char* usage = "Usage: %s LENSMODEL_XXX [problem-details problem-details ...]\n"
        "\n"
        "problem-details are a list of parameters we're optimizing. This is some set of\n"
        "  intrinsic-core\n"
        "  intrinsic-distortions\n"
        "  cahvor-optical-axis\n"
        "  extrinsics\n"
        "  frames\n"
        "  calobject-warp\n"
        "\n"
        "If no details are given, we optimize everything. Otherwise, we start with an empty\n"
        "mrcal_problem_details_t, and each argument sets a bit\n";

    if( argc >= 2 && argv[1][0] == '-' )
    {
        printf(usage, argv[0]);
        return 0;
    }

    mrcal_problem_details_t problem_details = {};


    int iarg = 1;
    if( iarg >= argc )
    {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }

    lens_model_t lens_model = mrcal_lens_model_from_name(argv[iarg]);
    if( lens_model == LENSMODEL_INVALID )
    {
#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"
        fprintf(stderr, "Lens model name '%s' unknown. I only know about ("
                        LENSMODEL_LIST( QUOTED_LIST_WITH_COMMA )
                ")\n", argv[iarg]);
        return 1;
    }
    iarg++;


    if(iarg >= argc)
        problem_details = DO_OPTIMIZE_ALL;
    else
        for(; iarg < argc; iarg++)
        {

            if( 0 == strcmp(argv[iarg], "intrinsic-core") )
            {
                problem_details.do_optimize_intrinsic_core = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "intrinsic-distortions") )
            {
                problem_details.do_optimize_intrinsic_distortions = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "extrinsics") )
            {
                problem_details.do_optimize_extrinsics = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "frames") )
            {
                problem_details.do_optimize_frames = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "cahvor-optical-axis" ) )
            {
                problem_details.do_optimize_cahvor_optical_axis = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "calobject-warp" ) )
            {
                problem_details.do_optimize_calobject_warp = true;
                continue;
            }

            fprintf(stderr, "Unknown optimization variable '%s'. Giving up.\n\n", argv[iarg]);
            fprintf(stderr, usage, argv[0]);
            return 1;
        }


    pose_t extrinsics[] =
        { { .r = { .xyz = {  .01,   .1,    .02}},  .t = { .xyz = { 2.3, 0.2, 0.1}}}};

    pose_t frames[] =
        { { .r = { .xyz = { -.1,    .52,  -.13}},  .t = { .xyz = { 1.3, 0.1, 10.2}}},
          { .r = { .xyz = {  .90,   .24,   .135}}, .t = { .xyz = { 0.7, 0.1, 20.3}}},
          { .r = { .xyz = {  .80,   .52,   .335}}, .t = { .xyz = { 0.7, 0.6, 30.4}}},
          { .r = { .xyz = {  .20,  -.22,   .75}},  .t = { .xyz = { 3.1, 6.3, 10.4}}}};
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);

    point3_t points[] =
        { {.xyz = {-5.3,   2.3, 20.4}},
          {.xyz = {-15.3, -3.2, 200.4}}};
    point2_t calobject_warp = {.x = 0.001, .y = -0.005};

    int Npoints = sizeof(points)/sizeof(points[0]);

#define calibration_object_width_n 10 /* arbitrary */

    point3_t observations_px      [6][calibration_object_width_n*calibration_object_width_n] = {};
    point3_t observations_point_px[4] = {};

#define NobservationsBoard 6
#define NobservationsPoint 4

    // fill observations with arbitrary data
    for(int i=0; i<NobservationsBoard; i++)
        for(int j=0; j<calibration_object_width_n; j++)
            for(int k=0; k<calibration_object_width_n; k++)
            {
                observations_px[i][calibration_object_width_n*j + k].x =
                    1000.0 + (double)k - 10.0*(double)j + (double)(i*j*k);
                observations_px[i][calibration_object_width_n*j + k].y =
                    1000.0 - (double)k + 30.0*(double)j - (double)(i*j*k);
                observations_px[i][calibration_object_width_n*j + k].z =
                    1. / (double)(1 << ((i+j+k) % 3));
            }
    for(int i=0; i<NobservationsPoint; i++)
    {
        observations_point_px[i].x = 1100.0 + (double)i*20.0;
        observations_point_px[i].y = 800.0  - (double)i*12.0;
        observations_point_px[i].z = 1. / (double)(1 << (i % 3));
    }

    observation_board_t observations_board[NobservationsBoard] =
        { {.i_camera = 0, .i_frame = 0, .px = observations_px[0]},
          {.i_camera = 1, .i_frame = 0, .px = observations_px[1]},
          {.i_camera = 1, .i_frame = 1, .px = observations_px[2]},
          {.i_camera = 0, .i_frame = 2, .px = observations_px[3]},
          {.i_camera = 0, .i_frame = 3, .px = observations_px[4]},
          {.i_camera = 1, .i_frame = 3, .px = observations_px[5]} };

    observation_point_t observations_point[NobservationsPoint] =
        { {.i_camera = 0, .i_point = 0, .px = observations_point_px[0]},
          {.i_camera = 1, .i_point = 0, .px = observations_point_px[1]},
          {.i_camera = 0, .i_point = 1, .px = observations_point_px[2], .dist = 18.0},
          {.i_camera = 1, .i_point = 1, .px = observations_point_px[3], .dist = 180.0} };

    int Ncameras = sizeof(extrinsics)/sizeof(extrinsics[0]) + 1;

    int Nintrinsics = mrcal_getNlensParams(lens_model);
    int Ndistortion = Nintrinsics;
    if(mrcal_modelHasCore_fxfycxcy(lens_model))
        Ndistortion -= 4;
    double intrinsics[Ncameras * Nintrinsics];

    int imagersizes[Ncameras*2];
    for(int i=0; i<Ncameras*2; i++)
        imagersizes[i] = 1000 + 10*i;

    intrinsics_core_t* intrinsics_core = (intrinsics_core_t*)intrinsics;
    intrinsics_core->focal_xy [0] = 2000.3;
    intrinsics_core->focal_xy [1] = 1900.5;
    intrinsics_core->center_xy[0] = 1800.3;
    intrinsics_core->center_xy[1] = 1790.2;

    intrinsics_core = (intrinsics_core_t*)(&intrinsics[Nintrinsics]);
    intrinsics_core->focal_xy [0] = 2100.2;
    intrinsics_core->focal_xy [1] = 2130.4;
    intrinsics_core->center_xy[0] = 1830.3;
    intrinsics_core->center_xy[1] = 1810.2;

    for(int i=0; i<Ncameras; i++)
        for(int j=0; j<Ndistortion; j++)
            intrinsics[Nintrinsics * i + 4 + j] = 0.1 + 0.05 * (double)(i + Ncameras*j);


    if(problem_details.do_optimize_intrinsic_core || problem_details.do_optimize_intrinsic_core)
    {
        printf("## The intrinsics occupy %d variables per camera, starting at variable 0\n",
               (problem_details.do_optimize_intrinsic_core        ? 4           : 0) +
               (problem_details.do_optimize_intrinsic_distortions ? Ndistortion : 0));
        if(problem_details.do_optimize_intrinsic_core)
            printf("## The intrinsics core occupies 4 variables per camera; the first is at variable 0\n");
        if(problem_details.do_optimize_intrinsic_distortions)
            printf("## The intrinsics distortions occupy %d variables per camera; the first is at variable %d\n",
                   Ndistortion, mrcal_state_index_intrinsic_distortions(0, problem_details, lens_model));
    }
    if(problem_details.do_optimize_extrinsics)
        printf("## The extrinsics occupy 6 variables per camera for all cameras except camera 0; the first is at variable %d\n",
               mrcal_state_index_camera_rt(1, Ncameras, problem_details, lens_model));
    if(problem_details.do_optimize_frames)
    {
        printf("## The frames occupy 6 variables per frame; the first is at variable %d\n",
               mrcal_state_index_frame_rt(0, Ncameras, problem_details, lens_model));
        printf("## The discrete points occupy 3 variables per point; the first is at variable %d\n",
               mrcal_state_index_point(0, Nframes, Ncameras, problem_details, lens_model));
    }
    if(problem_details.do_optimize_calobject_warp)
        printf("## The calibration object warp occupies 2 variables; the first is at variable %d\n",
               mrcal_state_index_calobject_warp(Npoints, Nframes, Ncameras, problem_details, lens_model));


    const double roi[] = { 1000., 1000., 400., 400.,
                            900., 1200., 300., 800. };
    mrcal_optimize( NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    intrinsics,
                    extrinsics,
                    frames,
                    points,
                    &calobject_warp,
                    Ncameras, Nframes, Npoints,

                    observations_board,
                    NobservationsBoard,

                    observations_point,
                    NobservationsPoint,

                    true,
                    0, NULL,
                    roi,
                    false,
                    true,
                    lens_model,
                    1.0,
                    imagersizes,
                    problem_details,

                    1.0, calibration_object_width_n);

    return 0;
}
