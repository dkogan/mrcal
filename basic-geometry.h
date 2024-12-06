// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

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


//////////// Easy convenience stuff
////// point2
__attribute__((unused))
static double mrcal_point2_inner(const mrcal_point2_t a, const mrcal_point2_t b)
{
    return
        a.x*b.x +
        a.y*b.y;
}
__attribute__((unused))
static double mrcal_point2_norm2(const mrcal_point2_t a)
{
    return mrcal_point2_inner(a,a);
}
#define mrcal_point2_mag(a) sqrt(norm2(a)) // macro to not require #include <math.h>

__attribute__((unused))
static mrcal_point2_t mrcal_point2_add(const mrcal_point2_t a, const mrcal_point2_t b)
{
    return (mrcal_point2_t){ .x = a.x + b.x,
                             .y = a.y + b.y };
}
__attribute__((unused))
static mrcal_point2_t mrcal_point2_sub(const mrcal_point2_t a, const mrcal_point2_t b)
{
    return (mrcal_point2_t){ .x = a.x - b.x,
                             .y = a.y - b.y };
}
__attribute__((unused))
static mrcal_point2_t mrcal_point2_scale(const mrcal_point2_t a, const double s)
{
    return (mrcal_point2_t){ .x = a.x * s,
                             .y = a.y * s };
}
////// point3
__attribute__((unused))
static double mrcal_point3_inner(const mrcal_point3_t a, const mrcal_point3_t b)
{
    return
        a.x*b.x +
        a.y*b.y +
        a.z*b.z;
}
__attribute__((unused))
static double mrcal_point3_norm2(const mrcal_point3_t a)
{
    return mrcal_point3_inner(a,a);
}
#define mrcal_point3_mag(a) sqrt(mrcal_point3_norm2(a)) // macro to not require #include <math.h>

__attribute__((unused))
static mrcal_point3_t mrcal_point3_add(const mrcal_point3_t a, const mrcal_point3_t b)
{
    return (mrcal_point3_t){ .x = a.x + b.x,
                             .y = a.y + b.y,
                             .z = a.z + b.z };
}
__attribute__((unused))
static mrcal_point3_t mrcal_point3_sub(const mrcal_point3_t a, const mrcal_point3_t b)
{
    return (mrcal_point3_t){ .x = a.x - b.x,
                             .y = a.y - b.y,
                             .z = a.z - b.z };
}
__attribute__((unused))
static mrcal_point3_t mrcal_point3_scale(const mrcal_point3_t a, const double s)
{
    return (mrcal_point3_t){ .x = a.x * s,
                             .y = a.y * s,
                             .z = a.z * s };
}
__attribute__((unused))
static mrcal_point3_t mrcal_point3_cross(const mrcal_point3_t a, const mrcal_point3_t b)
{
    return (mrcal_point3_t){ .x = a.y*b.z - a.z*b.y,
                             .y = a.z*b.x - a.x*b.z,
                             .z = a.x*b.y - a.y*b.x };
}

#define mrcal_point2_print(p) do { printf( #p " = (%f %f)\n",    (p).x, (p).y);        } while(0)
#define mrcal_point3_print(p) do { printf( #p " = (%f %f %f)\n", (p).x, (p).y, (p).z); } while(0)
#define mrcal_Rt_print(Rt)    do { printf( #Rt " = (\n  %f %f %f\n  %f %f %f\n  %f %f %f\n  ============\n  %f %f %f\n)\n", \
                                           (Rt)[0],(Rt)[1],(Rt)[2],(Rt)[3],(Rt)[4],(Rt)[5],(Rt)[6],(Rt)[7],(Rt)[8],(Rt)[9],(Rt)[10],(Rt)[11]); } while(0)
#define mrcal_rt_print(rt)    do { printf( #rt " = (%f %f %f;  %f %f %f\n", \
                                           (rt)[0],(rt)[1],(rt)[2],(rt)[3],(rt)[4],(rt)[5]); } while(0)
