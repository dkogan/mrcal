#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mrcal.h"


int main(int argc, char* argv[] )
{
    const char* usage = "%s DISTORTION_XXX [no-]optimize-intrinsics\n";

    if( argc != 3 )
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

    bool do_optimize_intrinsics;
    if     ( 0 == strcmp("optimize-intrinsics",    argv[2]) ) do_optimize_intrinsics = true;
    else if( 0 == strcmp("no-optimize-intrinsics", argv[2]) ) do_optimize_intrinsics = false;
    else
    {
        fprintf(stderr, "I must be passed either 'optimize-intrinsics' or 'no-optimize-intrinsics'\n");
        return 1;
    }


    // I use a large distortion array. Should be the largest one. I check with
    // an assertion
    struct intrinsics_DISTORTION_OPENCV8_t intrinsics[] =
        { {.focal_xy  = { 10.3, 10.5},
           .center_xy = { 49.3, 50.2} },
          {.focal_xy  = {  9.3,  9.5},
           .center_xy = { 51.3, 53.2} } };
    assert( mrcal_getNdistortionParams(distortion_model) + 4 <=
            (int)(sizeof(struct intrinsics_DISTORTION_OPENCV8_t)/sizeof(double)) );

    struct pose_t extrinsics[] =
        { {.r = {.xyz = {1.0, 2.1, 3.5}},
           .t = {.xyz = {0.3, 0.2, 0.9}}} };

    struct pose_t frames[] =
        { {.r = {.xyz = {-1.0, 2.5, -3.1}},
           .t = {.xyz = {1.3, 0.1, 0.2}}},
          {.r = {.xyz = {9.0, 2.4, 13.5}},
           .t = {.xyz = {0.7, 0.1, 0.3}}},
          {.r = {.xyz = {8.0, 5.2, 33.5}},
           .t = {.xyz = {0.7, 0.6, 0.4}}},
          {.r = {.xyz = {2.0, -2.2, 7.5}},
           .t = {.xyz = {3.1, 6.3, 10.4}}}};

    int Ncameras = sizeof(intrinsics)/sizeof(intrinsics[0]);
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);
    for(int i=0; i<Nframes; i++)
        for(int j=0; j<mrcal_getNdistortionParams(distortion_model); j++)
            intrinsics[i].distortions[j] = 0.05 * (double)(i + Nframes*j);


    // Dummy point observations. All use the same array, but this doesn't matter
    // for this test anyway
    union point2_t observations_px[NUM_POINTS_IN_CALOBJECT] = {};

    struct observation_t observations[] =
        { {.i_camera = 0, .i_frame = 0, .px = observations_px},
          {.i_camera = 1, .i_frame = 0, .px = observations_px},
          {.i_camera = 1, .i_frame = 1, .px = observations_px},
          {.i_camera = 0, .i_frame = 2, .px = observations_px},
          {.i_camera = 0, .i_frame = 3, .px = observations_px},
          {.i_camera = 1, .i_frame = 3, .px = observations_px} };
    int Nobservations = sizeof(observations)/sizeof(observations[0]);

    mrcal_optimize( (struct intrinsics_t*)intrinsics,
                    extrinsics,
                    frames,
                    Ncameras, Nframes,

                    observations,
                    Nobservations,

                    true,
                    distortion_model,
                    do_optimize_intrinsics);

    return 0;
}
