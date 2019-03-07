#pragma once

typedef union
{
    struct
    {
        double x,y;
    };
    double xy[2];
} point2_t;

typedef union
{
    struct
    {
        double x,y,z;
    };
    double xyz[3];
} point3_t;
