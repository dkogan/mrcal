#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mrcal.h"

#include "test-harness.h"


#define CAMERAMODEL_T(Nintrinsics)         \
struct                                     \
{                                          \
    mrcal_cameramodel_VOID_t cameramodel;       \
    double intrinsics[Nintrinsics];        \
}

#define Nintrinsics_CAHVORE (4+5+3)
typedef CAMERAMODEL_T(Nintrinsics_CAHVORE) cameramodel_cahvore_t;

#define check(string, ref, len) do {                                    \
    mrcal_cameramodel_VOID_t* m = mrcal_read_cameramodel_string(string, len);\
    confirm(m != NULL);                                                 \
    if(m != NULL)                                                       \
    {                                                                   \
      confirm_eq_int(m->    lensmodel.type,                             \
                     (ref)->lensmodel.type);                            \
      confirm_eq_double_max_array(m->    intrinsics,                    \
                                  (ref)->intrinsics,                    \
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

#define check_fail(string,len) do{                                      \
    mrcal_cameramodel_VOID_t* m = mrcal_read_cameramodel_string(string,len); \
    confirm(m == NULL);                                                 \
    if(m != NULL)                                                       \
      mrcal_free_cameramodel(&m);                                       \
} while(0)

int main(int argc, char* argv[])
{
    mrcal_cameramodel_VOID_t cameramodel;

    cameramodel_cahvore_t cameramodel_ref =
        (cameramodel_cahvore_t)
        { .cameramodel =
          {.lensmodel  = {.type = MRCAL_LENSMODEL_CAHVORE,
                          .LENSMODEL_CAHVORE__config.linearity = 0.34},
           .imagersize = {110,400},
           .rt_cam_ref = {0,1,2,33,44e4,-55.3e-3},
          },
          .intrinsics = {4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12}
        };

    // baseline
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
          0);
    // different spacing, different quote, paren
    check("{\n"
          "    'lensmodel' :  b'LENSMODEL_CAHVORE_linearity=0.34',\n"
          "    b'extrinsics' :[ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    \"intrinsics\": (4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ),    'imagersize': [110, 400],\n"
          "\n"
          "}\n",
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
          0);
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
          "],'intrinsics': [ 4, 3, 4,\n5,    0,  \n\n  1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400]\n"
          "# }\n"
          "}  \n"
          " # }\n",
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
          0);

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
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
          0);

    // trailing garbage
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "} f\n",
               0);

    // double-defined key
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);

    // missing key
    check_fail("{\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
               "}\n",
               0);

    // Wrong number of intrinsics
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, ],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 99,88],\n"
               "    'imagersize': [110, 400],\n"
               "}\n",
               0);


    // Roundtrip write/read
    bool write_cameramodel_succeeded =
        mrcal_write_cameramodel_file("/tmp/test-parser-cameramodel.cameramodel",
                                     (mrcal_cameramodel_VOID_t*)&cameramodel_ref);
    confirm(write_cameramodel_succeeded);
    if(write_cameramodel_succeeded)
    {
        FILE* fp = fopen("/tmp/test-parser-cameramodel.cameramodel", "r");
        char buf[1024];
        int Nbytes_read = fread(buf, 1, sizeof(buf), fp);
        fclose(fp);
        confirm(Nbytes_read > 0);
        confirm(Nbytes_read < (int)sizeof(buf));
        if(Nbytes_read > 0 && Nbytes_read < (int)sizeof(buf))
        {
            // Added byte to make sure we're not 0-terminated. Which the parser
            // must deal with
            buf[Nbytes_read] = '5';
            check(buf,
                  (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
                  Nbytes_read);
        }
    }

    TEST_FOOTER();
}

