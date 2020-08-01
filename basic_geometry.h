#pragma once

typedef union
{
    struct
    {
        double x,y;
    };
    double xy[2];
} mrcal_point2_t;

typedef union
{
    struct
    {
        double x,y,z;
    };
    double xyz[3];
} mrcal_point3_t;

// unconstrained 6DOF pose containing a rodrigues rotation and a translation
typedef struct
{
    mrcal_point3_t r,t;
} mrcal_pose_t;
