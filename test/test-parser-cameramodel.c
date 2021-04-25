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

static void check(const char* string,
                  const mrcal_cameramodel_t* ref)
{
    mrcal_cameramodel_t* m = mrcal_read_cameramodel_string(string);
    confirm(m != NULL);
    if(m == NULL)
        return;

    confirm_eq_int(m->  lensmodel.type,
                   ref->lensmodel.type);
    confirm_eq_double_max_array(m->  intrinsics_data,
                                ref->intrinsics_data,
                                Nintrinsics_CAHVORE,
                                1e-6);
    confirm_eq_double_max_array(m->  rt_cam_ref,
                                ref->rt_cam_ref,
                                6,
                                1e-6);
    confirm_eq_int_max_array((int*)m->  imagersize,
                             (int*)ref->imagersize,
                             2);
    mrcal_free_cameramodel(&m);
}

static void check_fail(const char* string,
                       const mrcal_cameramodel_t* ref)
{
    mrcal_cameramodel_t* m = mrcal_read_cameramodel_string(string);
    confirm(m == NULL);
    if(m != NULL)
    mrcal_free_cameramodel(&m);
}

int main(int argc, char* argv[])
{
    mrcal_cameramodel_t cameramodel;

    check("#  fa # dsf\n"
          "{\n"
          "    'lensmodel'  :    \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "\n"
          "    # extrinsics are rt_fromref\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "\n"
          "    'f': 'g', 'jj': # fff # ads 'q',\n"
          "    'b', # ,\n"
          "\n"
          "    # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12 ],\n"
          "\n"
          "    # extrinsics are rt_fromref\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_t*)
          &(cameramodel_cahvore_t)
          { .cameramodel =
            {.lensmodel  = {.type = MRCAL_LENSMODEL_CAHVORE,
                            .LENSMODEL_CAHVORE__config.linearity = 0.34},
             .imagersize = {110,400},
             .rt_cam_ref = {0,1,2,33,44e4,-55.3e-3},
            },
            .intrinsics_data = {4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12}
          });

    check_fail("#  fa # dsf\n"
          "{\n"
          "    'lensmodel'  :    \"LENSMODEL_CAHVORE_linearity=0.34\",\n"
          "\n"
          "    # extrinsics are rt_fromref\n"
          "    'extrinsics': [ 0., 1, 2, 33, 44e4, -55.3E-3, ],\n"
          "\n"
          "    'f': 'g', 'jj': # fff # ads 'q',\n"
          "    'b', # ,\n"
          "\n"
          "    # intrinsics are fx,fy,cx,cy,distortion0,distortion1,....\n"
          "    'intrinsics': [ 4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12, 4 ],\n"
          "\n"
          "    # extrinsics are rt_fromref\n"
          "    'imagersize': [110, 400],\n"
          "}\n",
          (mrcal_cameramodel_t*)
          &(cameramodel_cahvore_t)
          { .cameramodel =
            {.lensmodel  = {.type = MRCAL_LENSMODEL_CAHVORE,
                            .LENSMODEL_CAHVORE__config.linearity = 0.34},
             .imagersize = {110,400},
             .rt_cam_ref = {0,1,2,33,44e4,-55.3e-3},
            },
            .intrinsics_data = {4, 3, 4, 5, 0, 1, 3, 5, 4, 10, 11, 12}
          });


    TEST_FOOTER();
}

