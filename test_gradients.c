#include <stdio.h>
#include <stdlib.h>

#include "mrcal.h"


int main(int argc     __attribute__((unused)),
         char* argv[] __attribute__((unused)))
{

    struct intrinsics_t intrinsics[] =
        { {.focal_xy  = { 10.3, 10.5},
           .center_xy = { 49.3, 50.2} },
          {.focal_xy  = {  9.3,  9.5},
           .center_xy = { 51.3, 53.2} } };

    struct pose_t extrinsics[] =
        { {.r = {.xyz = {1.0, 2.1, 3.5}},
           .t = {.xyz = {0.3, 0.2, 0.9}}} };

    struct pose_t frames[] =
        { {.r = {.xyz = {-1.0, 2.5, -3.1}},
           .t = {.xyz = {1.3, 0.1, 0.2}}},
          {.r = {.xyz = {9.0, 2.4, 13.5}},
           .t = {.xyz = {0.7, 0.1, 0.3}}},
          {.r = {.xyz = {8.0, 5.2, 33.5}},
           .t = {.xyz = {0.7, 0.6, 0.4}}}};

    int Ncameras = sizeof(intrinsics)/sizeof(intrinsics[0]);
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

    mrcal_optimize( intrinsics,
                    extrinsics,
                    frames,
                    Ncameras, Nframes,

                    observations,
                    Nobservations,

                    true,
                    DISTORTION_NONE,
                    true);

    return 0;
}
