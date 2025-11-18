#pragma once

#ifdef __cplusplus
extern "C" {
#endif


// mrcal_cameramodel_converter is a "converter" function that can be used with
// "O&" conversions in PyArg_ParseTupleAndKeywords() calls. Can interpret either
// path strings or mrcal.cameramodel objects as mrcal_cameramodel_VOID_t structures

#include <Python.h>
#include "mrcal.h"

int mrcal_cameramodel_converter(PyObject*             py_model,
                                mrcal_cameramodel_VOID_t** model);

#ifdef __cplusplus
}
#endif
