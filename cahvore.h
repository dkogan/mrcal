#pragma once

#include <stdbool.h>
#include "basic-geometry.h"

bool project_cahvore_internals( // outputs
                                mrcal_point3_t* __restrict pdistorted,
                                double*         __restrict dpdistorted_dintrinsics_nocore,
                                double*         __restrict dpdistorted_dp,

                                // inputs
                                const mrcal_point3_t* __restrict p,
                                const double* __restrict intrinsics_nocore,
                                const double cahvore_linearity);

