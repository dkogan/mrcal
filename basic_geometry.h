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
