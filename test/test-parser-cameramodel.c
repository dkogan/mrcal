#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <libgen.h>

#include "../mrcal.h"
#include "../_util.h"

#include "test-harness.h"


#define CAMERAMODEL_T(Nintrinsics)         \
struct                                     \
{                                          \
    mrcal_cameramodel_VOID_t cameramodel;       \
    double intrinsics[Nintrinsics];        \
}

static void confirm_models_equal(const mrcal_cameramodel_VOID_t* a,
                                 const mrcal_cameramodel_VOID_t* b)
{
    confirm_eq_double_max_array(a->rt_cam_ref,
                                b->rt_cam_ref,
                                6,
                                1e-6);
    confirm_eq_int_max_array((const int*)a->imagersize,
                             (const int*)b->imagersize,
                             2);

    if(!confirm_eq_int(a->lensmodel.type,
                       b->lensmodel.type))
        return;

    char modelname_a[1024];
    char modelname_b[1024];
    if(!( confirm(mrcal_lensmodel_name( modelname_a, sizeof(modelname_a),
                                        &a->lensmodel)) &&
          confirm(mrcal_lensmodel_name( modelname_b, sizeof(modelname_b),
                                        &b->lensmodel)) ))
        return;

    if(!confirm(0 == strcmp(modelname_a,modelname_b)))
        return;

    const int Nintrinsics_a = mrcal_lensmodel_num_params(&a->lensmodel);
    const int Nintrinsics_b = mrcal_lensmodel_num_params(&b->lensmodel);
    if(!confirm_eq_int(Nintrinsics_a, Nintrinsics_b))
        return;

    confirm_eq_double_max_array(a->intrinsics,
                                b->intrinsics,
                                Nintrinsics_a,
                                1e-6);
}

#define check(string, ref, len) do {                                    \
    mrcal_cameramodel_VOID_t* m = mrcal_read_cameramodel_string(string, len);\
    confirm(m != NULL);                                                 \
    if(m != NULL)                                                       \
    {                                                                   \
      confirm_models_equal(m,ref);                                      \
      mrcal_free_cameramodel(&m);                                       \
    }                                                                   \
} while(0)

#define check_fail(string,len) do{                                      \
    mrcal_cameramodel_VOID_t* m = mrcal_read_cameramodel_string(string,len); \
    confirm(m == NULL);                                                 \
    if(m != NULL)                                                       \
      mrcal_free_cameramodel(&m);                                       \
} while(0)


static void check_read_from_disk(void)
{
    char self_exe[1024];
    const size_t bufsize_self_exe = sizeof(self_exe);
    ssize_t len_readlink = readlink("/proc/self/exe", self_exe, bufsize_self_exe);
    if(!confirm(len_readlink > 0))
        return;
    if(!confirm((int)len_readlink < (int)bufsize_self_exe))
        return;

    self_exe[len_readlink] = '\0';
    const char* self_dir = dirname(self_exe);

    char path_model[1024];
    const int snprintf_bytes_would_write =
        snprintf(path_model, sizeof(path_model),
                 "%s/data/cam0.opencv8.cameramodel",
                self_dir);
    if(!confirm(snprintf_bytes_would_write < (int)sizeof(path_model)))
        return;

    const mrcal_cameramodel_LENSMODEL_OPENCV8_t cameramodel_ref =
        (mrcal_cameramodel_LENSMODEL_OPENCV8_t)
        {.lensmodel  = {.type = MRCAL_LENSMODEL_OPENCV8},
         .imagersize = {4000,2200},
         .rt_cam_ref = {2e-2, -3e-1, -1e-2,  1., 2, -3.},
         .intrinsics = {1761.181055, 1761.250444, 1965.706996, 1087.518797, -0.01266096516, 0.03590794372, -0.0002547045941, 0.0005275929652, 0.01968883397, 0.01482863541, -0.0562239888, 0.0500223357}
        };

    mrcal_cameramodel_LENSMODEL_OPENCV8_t m;
    int Nintrinsics_max = 8+4;
    if(!confirm(mrcal_read_cameramodel_file_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max, path_model)))
        return;
    confirm_models_equal((mrcal_cameramodel_VOID_t*)&m,
                         (mrcal_cameramodel_VOID_t*)&cameramodel_ref);
    int Nintrinsics_max_too_small = 8+4 - 1;
    if(!confirm(!mrcal_read_cameramodel_file_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max_too_small, path_model)))
        return;

    confirm_eq_int(Nintrinsics_max_too_small, 8+4);

    // and reading into a buffer from a string
    char buf[16384];
    FILE* fp = fopen(path_model, "r");
    if(!confirm(fp != NULL))
        return;
    int nbytes = fread(buf,1,sizeof(buf)-1,fp);
    confirm(nbytes>0);
    buf[nbytes] = '\0';
    fclose(fp);

    Nintrinsics_max = 8+4;
    if(!confirm(mrcal_read_cameramodel_string_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max,
                                                   buf, 0)))
        return;
    confirm_models_equal((mrcal_cameramodel_VOID_t*)&m,
                         (mrcal_cameramodel_VOID_t*)&cameramodel_ref);
    Nintrinsics_max_too_small = 8+4 - 1;
    if(!confirm(!mrcal_read_cameramodel_string_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max_too_small,
                                                    buf, 0)))
        return;

    // And again, giving it the buffer size instead of '\0' termination
    buf[nbytes] = 'x';
    Nintrinsics_max = 8+4;
    if(!confirm(mrcal_read_cameramodel_string_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max,
                                                   buf, nbytes)))
        return;
    confirm_models_equal((mrcal_cameramodel_VOID_t*)&m,
                         (mrcal_cameramodel_VOID_t*)&cameramodel_ref);
    Nintrinsics_max_too_small = 8+4 - 1;
    if(!confirm(!mrcal_read_cameramodel_string_into((mrcal_cameramodel_VOID_t*)&m, &Nintrinsics_max_too_small,
                                                    buf, nbytes)))
        return;
}

int main(int argc, char* argv[])
{
    mrcal_cameramodel_VOID_t cameramodel;

    mrcal_cameramodel_LENSMODEL_CAHVORE_t cameramodel_ref =
        (mrcal_cameramodel_LENSMODEL_CAHVORE_t)
        { .lensmodel  = {.type = MRCAL_LENSMODEL_CAHVORE,
                         .LENSMODEL_CAHVORE__config.linearity = 0.34},
          .imagersize = {110,400},
          .rt_cam_ref = {0,1,2,33,44e4,-55.3e-3},
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

    // new rt_cam_ref key
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'rt_cam_ref': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
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
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
          0);
    check_fail("{\n"
               "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
               "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.4E-3, ],\n"
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

    // If rt_cam_ref key and extrinsics are given, they must be the same
    check("{\n"
          "    'lensmodel':  \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "    'rt_cam_ref': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_VOID_t*)&cameramodel_ref,
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

    check_read_from_disk();

    TEST_FOOTER();
}
