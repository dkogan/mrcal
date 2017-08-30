#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mrcal.h"


int main(int argc, char* argv[] )
{
    const char* usage = "Usage: %s DISTORTION_XXX [no-]optimize-intrinsics\n";

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

    // I define all possible intrinsics arrays, and use the proper one later on.
    // This is required because sizeof(intrinsic) varies depending on the type
#define DECLARE_SPECIFIC_INTRINSIC(s,n)                         \
    __attribute__((unused))                                     \
        struct intrinsics_ ## s ## _t intrinsics ## s[] =       \
        { {.focal_xy  = { 10.3, 10.5},                          \
           .center_xy = { 49.3, 50.2} },                        \
          {.focal_xy  = {  9.3,  9.5},                          \
           .center_xy = { 51.3, 53.2} } };
    DISTORTION_LIST(DECLARE_SPECIFIC_INTRINSIC)

    int Ncameras = sizeof(intrinsicsDISTORTION_NONE)/sizeof(intrinsicsDISTORTION_NONE[0]);

#define SET_SPECIFIC_DISTORTION(s,n)                                    \
    for(int j=0; j<n; j++)                                              \
        intrinsics ## s[i].distortions[j] = 0.0005 * (double)(i + Ncameras*j);
    for(int i=0; i<Ncameras; i++)
    {
        DISTORTION_LIST(SET_SPECIFIC_DISTORTION);
    }



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
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);


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

#define PICK_SPECIFIC_INTRINSICS(s,n) case s: intrinsics_specific = (struct intrinsics_t*)intrinsics ## s; break;
    struct intrinsics_t* intrinsics_specific;
    switch(distortion_model)
    {
        DISTORTION_LIST(PICK_SPECIFIC_INTRINSICS)

    default:
        assert(0);
    }

    mrcal_optimize( intrinsics_specific,
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
