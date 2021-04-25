#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mrcal.h"

#include "test-harness.h"


#define CAMERAMODEL_T(Nintrinsics)              \
struct                                          \
{                                               \
    mrcal_cameramodel_t cameramodel;            \
    double intrinsics_data[Nintrinsics];        \
}

#define Nintrinsics_CAHVORE (4+5+3)
typedef CAMERAMODEL_T(Nintrinsics_CAHVORE) cameramodel_cahvore_t;

#define check(string, ref) do {                                         \
    mrcal_cameramodel_t* m = mrcal_read_cameramodel_string(string);     \
    confirm(m != NULL);                                                 \
    if(m != NULL)                                                       \
    {                                                                   \
      confirm_eq_int(m->    lensmodel.type,                             \
                     (ref)->lensmodel.type);                            \
      confirm_eq_double_max_array(m->    intrinsics_data,               \
                                  (ref)->intrinsics_data,               \
                                  Nintrinsics_CAHVORE,                  \
                                  1e-6);                                \
      confirm_eq_double_max_array(m->    rt_cam_ref,                    \
                                  (ref)->rt_cam_ref,                    \
                                  6,                                    \
                                  1e-6);                                \
      confirm_eq_int_max_array((int*)m->    imagersize,                 \
                               (int*)(ref)->imagersize,                 \
                               2);                                      \
      mrcal_free_cameramodel(&m);                                       \
    }                                                                   \
} while(0)

#define check_fail(string) do{                                          \
    mrcal_cameramodel_t* m = mrcal_read_cameramodel_string(string);     \
    confirm(m == NULL);                                                 \
    if(m != NULL)                                                       \
      mrcal_free_cameramodel(&m);                                       \
} while(0)

int main(int argc, char* argv[])
{
    mrcal_cameramodel_t cameramodel;

    cameramodel_cahvore_t cameramodel_ref =
        (cameramodel_cahvore_t)
        { .cameramodel =
          {.lensmodel  = {.type = MRCAL_LENSMODEL_CAHVORE,
                          .LENSMODEL_CAHVORE__config.linearity = 0.34},
           .imagersize = {110,400},
           .rt_cam_ref = {0,1,2,33,44e4,-55.3e-3},
          },
          .intrinsics_data = {4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12}
        };

    // baseline
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_t*)&cameramodel_ref);
    // different spacing, different quote, paren
    check("{\n"
          "    'lensmodel' :  b'LENSMODEL_CAHVORE_linearity=0.34',\n"
          "    b'extrinsics' :[ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    \"intrinsics\": (4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ),    'imagersize': [110, 400],\n"
          "\n"
          "}\n",
          (mrcal_cameramodel_t*)&cameramodel_ref);
    // comments, weird spacing
    check(" # f {\n"
          "#{ 'lensmodel': 'rrr'\n"
          "{'lensmodel':  #\"LENSMODEL_CAHVOR\",\n"
          "\"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'extrinsics': [ 0., 1, 2, 33, # 44e4, -55.3E-3,\n"
          "44e4, -55.3E-3\n"
          "#,\n"
          ",\n"
          "#]\n"
          "],'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400]\n"
          "# }\n"
          "}  \n"
          " # }\n",
          (mrcal_cameramodel_t*)&cameramodel_ref);

    // extra keys
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\", 'f': 5,\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ], 'xxx':\n"
          " # fff\n"
          " b'rr','qq': b'asdf;lkj&*()DSFEWR]]{}}}',\n"
          "'vvvv': [ 1,2, [4,5],[3,[4,3,[]],444], ],"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_t*)&cameramodel_ref);

    // trailing garbage
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "} f\n");

    // double-defined key
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");

    // missing key
    check_fail("{\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "}\n");

    // Wrong number of intrinsics
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 99,88],\n"
               "    'imagersize': [110, 400],\n"
               "}\n");




    TEST_FOOTER();
}

