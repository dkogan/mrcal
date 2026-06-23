// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#pragma once

#ifdef __cplusplus
extern "C" {
#endif


#include <stdio.h>

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)

#ifdef _MSC_VER
#define MRCAL_HIDDEN
#define MRCAL_UNUSED
#else
#define MRCAL_HIDDEN __attribute__((visibility ("hidden")))
#define MRCAL_UNUSED __attribute__((unused))
#endif

#ifdef __cplusplus
}
#endif
