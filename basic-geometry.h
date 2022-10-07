#pragma once

// A 2D point or vector
//
// The individual elements can be accessed via .x and .y OR the vector can be
// accessed as an .xy[] array:
//
//   mrcal_point2_t p = f();
//
// Now p.x and p.xy[0] refer to the same value.
typedef union
{
    struct
    {
        double x,y;
    };

    double xy[2];
} mrcal_point2_t;

// A 3D point or vector
//
// The individual elements can be accessed via .x and .y and .z OR the vector
// can be accessed as an .xyz[] array:
//
//   mrcal_point3_t p = f();
//
// Now p.x and p.xy[0] refer to the same value.
typedef union
{
    struct
    {
        double x,y,z;
    };
    double xyz[3];
} mrcal_point3_t;

// Unconstrained 6DOF pose containing a Rodrigues rotation and a translation
typedef struct
{
    mrcal_point3_t r,t;
} mrcal_pose_t;
