#pragma once

union point2_t
{
    struct
    {
        double x,y;
    };
    double xy[2];
};

union point3_t
{
    struct
    {
        double x,y,z;
    };
    double xyz[3];
};
