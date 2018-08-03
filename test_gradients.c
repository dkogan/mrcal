#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mrcal.h"


int main(int argc, char* argv[] )
{
    const char* usage = "Usage: %s DISTORTION_XXX optimizing_list\n"
        "\n"
        "optimizing_list is a list of parameters we're optimizing. This is some of:\n"
        "  intrinsic-core\n"
        "  intrinsic-distortions\n"
        "  extrinsics\n"
        "  frames\n"
        "  all\n"
        "\n"
        "'all' is a shorthand that includes all the others.\n";

    if( argc < 3 )
    {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }

    enum distortion_model_t distortion_model = mrcal_distortion_model_from_name(argv[1]);
    if( distortion_model == DISTORTION_INVALID )
    {
#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"
        fprintf(stderr, "Distortion name '%s' unknown. I only know about ("
                        DISTORTION_LIST( QUOTED_LIST_WITH_COMMA )
                ")\n", argv[1]);
        return 1;
    }

    struct mrcal_variable_select optimization_variable_choice = {};

    for(int iarg = 2; iarg < argc; iarg++)
    {
        if( 0 == strcmp(argv[iarg], "all") )
        {
            optimization_variable_choice = DO_OPTIMIZE_ALL;
            break;
        }

        if( 0 == strcmp(argv[iarg], "intrinsic-core") )
        {
            optimization_variable_choice.do_optimize_intrinsic_core = true;
            continue;
        }
        if( 0 == strcmp(argv[iarg], "intrinsic-distortions") )
        {
            optimization_variable_choice.do_optimize_intrinsic_distortions = true;
            continue;
        }
        if( 0 == strcmp(argv[iarg], "extrinsics") )
        {
            optimization_variable_choice.do_optimize_extrinsics = true;
            continue;
        }
        if( 0 == strcmp(argv[iarg], "frames") )
        {
            optimization_variable_choice.do_optimize_frames = true;
            continue;
        }

        fprintf(stderr, "Unknown optimization variable '%s'. Giving up.\n\n", argv[iarg]);
        fprintf(stderr, usage, argv[0]);
        return 1;
    }


    struct pose_t extrinsics[] =
        { { .r = { .xyz = {  .01,   .1,    .02}},  .t = { .xyz = { 2.3, 0.2, 0.1}}}};

    struct pose_t frames[] =
        { { .r = { .xyz = { -.1,    .52,  -.13}},  .t = { .xyz = { 1.3, 0.1, 10.2}}},
          { .r = { .xyz = {  .90,   .24,   .135}}, .t = { .xyz = { 0.7, 0.1, 20.3}}},
          { .r = { .xyz = {  .80,   .52,   .335}}, .t = { .xyz = { 0.7, 0.6, 30.4}}},
          { .r = { .xyz = {  .20,  -.22,   .75}},  .t = { .xyz = { 3.1, 6.3, 10.4}}}};
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);

    union point3_t points[] =
        { {.xyz = {-5.3,   2.3, 20.4}},
          {.xyz = {-15.3, -3.2, 200.4}}};
    int Npoints = sizeof(points)/sizeof(points[0]);

    // Dummy point observations. All use the same array, but this doesn't matter
    // for this test anyway
#define calibration_object_width_n 10 /* arbitrary */
    union point2_t observations_px[calibration_object_width_n*calibration_object_width_n] = {};

    struct observation_board_t observations_board[] =
        { {.i_camera = 0, .i_frame = 0, .px = observations_px},
          {.i_camera = 1, .i_frame = 0, .px = observations_px},
          {.i_camera = 1, .i_frame = 1, .px = observations_px},
          {.i_camera = 0, .i_frame = 2, .px = observations_px},
          {.i_camera = 0, .i_frame = 3, .px = observations_px},
          {.i_camera = 1, .i_frame = 3, .px = observations_px} };
    int NobservationsBoard = sizeof(observations_board)/sizeof(observations_board[0]);

    struct observation_point_t observations_point[] =
        { {.i_camera = 0, .i_point = 0, .px = {}},
          {.i_camera = 1, .i_point = 0, .px = {}},
          {.i_camera = 0, .i_point = 1, .px = {}, .dist = 18.0},
          {.i_camera = 1, .i_point = 1, .px = {}, .dist = 180.0} };
    int NobservationsPoint = sizeof(observations_point)/sizeof(observations_point[0]);


    int Ncameras = sizeof(extrinsics)/sizeof(extrinsics[0]) + 1;

    int Ndistortion = mrcal_getNdistortionParams(distortion_model);
    int Nintrinsics = Ndistortion + N_INTRINSICS_CORE;
    double intrinsics[Ncameras * Nintrinsics];

    struct intrinsics_core_t* intrinsics_core = (struct intrinsics_core_t*)intrinsics;
    intrinsics_core->focal_xy [0] = 2000.3;
    intrinsics_core->focal_xy [1] = 1900.5;
    intrinsics_core->center_xy[0] = 1800.3;
    intrinsics_core->center_xy[1] = 1790.2;

    intrinsics_core = (struct intrinsics_core_t*)(&intrinsics[Nintrinsics]);
    intrinsics_core->focal_xy [0] = 2100.2;
    intrinsics_core->focal_xy [1] = 2130.4;
    intrinsics_core->center_xy[0] = 1830.3;
    intrinsics_core->center_xy[1] = 1810.2;

    for(int i=0; i<Ncameras; i++)
        for(int j=0; j<Ndistortion; j++)
            intrinsics[Nintrinsics * i + N_INTRINSICS_CORE + j] = 0.0005 * (double)(i + Ncameras*j);


    mrcal_optimize( NULL, NULL, NULL,
                    intrinsics,
                    extrinsics,
                    frames,
                    points,
                    Ncameras, Nframes, Npoints,

                    observations_board,
                    NobservationsBoard,

                    observations_point,
                    NobservationsPoint,

                    true,
                    false,
                    true,
                    distortion_model,
                    optimization_variable_choice,

                    1.0, calibration_object_width_n);

    return 0;
}
