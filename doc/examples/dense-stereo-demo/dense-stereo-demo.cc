// Sample dense stereo pipeline, implemented using the C mrcal API. Needs at
// least mrcal 2.4. This is a C++ source file because the remapping and
// disparity search come from OpenCV, which is a C++ library. The mrcal calls
// are C functions
//
// On a Debian machine you can build like this:
// g++                      \
//   -I/usr/include/opencv4 \
//   -fpermissive           \
//   dense-stereo-demo.cc   \
//   -lopencv_core          \
//   -lopencv_calib3d       \
//   -lopencv_imgproc       \
//   -lmrcal                \
//   -o dense-stereo-demo
//
// The -fpermissive is required for the C++ compiler to accept C99 code.
//
// Note: this sample code does not bother to deallocate any memory.

#include <stdio.h>
#include <stdlib.h>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

extern "C"
{
#include <mrcal/mrcal.h>
}

static
cv::Mat remap(const float* map,
              const mrcal_image_uint8_t* image,
              const int remapped_width,
              const int remapped_height)
{
    cv::Mat cv_image(image->height,
                     image->width,
                     CV_8UC1,
                     image->data,
                     image->stride);
    cv::Mat cv_remapped(remapped_height,
                        remapped_width,
                        CV_8UC1);

    cv::Mat cv_map(remapped_height,
                   remapped_width,
                   CV_32FC2,
                   (void*)map);

    cv::remap( cv_image,
               cv_remapped,
               cv_map, cv::Mat(),
               cv::INTER_LINEAR );
    return cv_remapped;
}


int main(int argc, char* argv[])
{

    // https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/0.cameramodel
    // https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/1.cameramodel
    const char* model_filenames[] =
        {
            "/tmp/0.cameramodel",
            "/tmp/1.cameramodel"
        };

    // https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/0.jpg
    // https://mrcal.secretsauce.net/external/2022-11-05--dtla-overpass--samyang--alpha7/stereo/1.jpg
    const char* image_filenames[] =
        {
            "/tmp/0.jpg",
            "/tmp/1.jpg",
        };
    // Hard-coded rectified field-of-view and center-of-view parameters
    mrcal_point2_t azel_fov_deg = {170., 60.};
    mrcal_point2_t azel0_deg    = {};

    const int disparity_min = 0;
    const int disparity_max = 256;

    // hard-coded scale used by OpenCV SGBM
    const int disparity_scale = 16;

    // Use the same resolution in the rectified image as in the input image
    double pixels_per_deg_az = -1.;
    double pixels_per_deg_el = -1.;



    mrcal_cameramodel_VOID_t* models[2];
    mrcal_image_uint8_t images[2];


    //// Read the models from disk
    for(int i=0; i<2; i++)
    {
        models[i] = mrcal_read_cameramodel_file(model_filenames[i]);
        if(models[i] == NULL)
        {
            fprintf(stderr, "Error loading model '%s'\n", model_filenames[i]);
            return 1;
        }
        if(!mrcal_image_uint8_load(&images[i],
                                   image_filenames[i]))
        {
            fprintf(stderr, "Error loading image '%s'\n", image_filenames[i]);
            return 1;
        }
    }


    //// Compute the rectified system
    mrcal_cameramodel_LENSMODEL_LATLON_t model_rectified0 =
        { .lensmodel = { .type = MRCAL_LENSMODEL_LATLON } };
    double baseline;
    if(!mrcal_rectified_system(// output
                               model_rectified0.imagersize,
                               model_rectified0.intrinsics,
                               model_rectified0.rt_cam_ref,
                               &baseline,
                               // input, output
                               &pixels_per_deg_az,
                               &pixels_per_deg_el,
                               &azel_fov_deg,
                               &azel0_deg,
                               // input
                               &models[0]->lensmodel,
                               models[0]->intrinsics,
                               models[0]->rt_cam_ref,
                               models[1]->rt_cam_ref,
                               model_rectified0.lensmodel.type,
                               // autodetect nothing
                               false,false,false,false))
    {
        fprintf(stderr, "Error calling mrcal_rectified_system()\n");
        return 1;
    }


    //// Compute the rectification maps
    const int rectified_width  = model_rectified0.imagersize[0];
    const int rectified_height = model_rectified0.imagersize[1];
    float* rectification_maps =
        (float*)aligned_alloc(0x10,
                              rectified_width*
                              rectified_height*
                              2*2*sizeof(float));
    if(rectification_maps == NULL)
    {
        fprintf(stderr, "Error calling malloc()\n");
        return 1;
    }
    if(!mrcal_rectification_maps(// output
                                 rectification_maps,
                                 // input
                                 &models[0]->lensmodel,
                                 models [0]->intrinsics,
                                 models [0]->rt_cam_ref,
                                 &models[1]->lensmodel,
                                 models [1]->intrinsics,
                                 models [1]->rt_cam_ref,
                                 model_rectified0.lensmodel.type,
                                 model_rectified0.intrinsics,
                                 model_rectified0.imagersize,
                                 model_rectified0.rt_cam_ref) )
    {
        fprintf(stderr, "Error calling mrcal_rectification_maps()\n");
        return 1;
    }


    //// Use the rectification maps to compute the rectified images
    cv::Mat cv_left_rect =
        remap(&rectification_maps[0],
              &images[0],
              rectified_width,
              rectified_height);
    cv::Mat cv_right_rect =
        remap(&rectification_maps[rectified_width*
                                  rectified_height*
                                  2],
              &images[1],
              rectified_width,
              rectified_height);


    //// Write the rectified images to disk
    mrcal_image_uint8_save("/tmp/rect-left.png",
                           &(mrcal_image_uint8_t)
                           {
                               .width  = rectified_width,
                               .height = rectified_height,
                               .stride = rectified_width,
                               .data   = (uint8_t*)cv_left_rect.data
                           });
    fprintf(stderr, "Wrote '/tmp/rect-left.png'\n");
    mrcal_image_uint8_save("/tmp/rect-right.png",
                           &(mrcal_image_uint8_t)
                           {
                               .width  = rectified_width,
                               .height = rectified_height,
                               .stride = rectified_width,
                               .data   = (uint8_t*)cv_right_rect.data
                           });
    fprintf(stderr, "Wrote '/tmp/rect-right.png'\n");


    //// Disparity search
    cv::Mat cv_disparity;
    cv::Ptr<cv::StereoSGBM> sgbm =
        cv::StereoSGBM::create(disparity_min,
                               disparity_max-disparity_min,
                               3,
                               600,
                               2400,
                               1,
                               0,
                               5,
                               100,
                               2);
    sgbm->compute(cv_left_rect, cv_right_rect, cv_disparity);
    mrcal_image_uint16_t image_disparity =
        {
            .width  = rectified_width,
            .height = rectified_height,
            .stride = rectified_width*2,
            .data   = (uint16_t*)cv_disparity.data
        };
    mrcal_image_bgr_t image_color_disparity =
        { .width  = rectified_width,
          .height = rectified_height,
          .stride = (int)(rectified_width*sizeof(mrcal_bgr_t)),
          .data   = (mrcal_bgr_t*)malloc(rectified_height*
                                         rectified_width*sizeof(mrcal_bgr_t))
        };
    if(image_color_disparity.data == NULL)
    {
        fprintf(stderr, "Error: malloc() failed\n");
        return 1;
    }
    if(!mrcal_apply_color_map_uint16(&image_color_disparity,
                                     &image_disparity,
                                     // no auto range
                                     false,false,
                                     // auto function
                                     true,
                                     disparity_min*disparity_scale,
                                     disparity_max*disparity_scale,
                                     // ignored functions
                                     0,0,0))
    {
        fprintf(stderr, "Error: mrcal_apply_color_map_uint16() failed\n");
        return 1;
    }
    mrcal_image_bgr_save("/tmp/disparity.png", &image_color_disparity);
    fprintf(stderr, "Wrote '/tmp/disparity.png'\n");


    //// Convert disparities to ranges
    mrcal_image_double_t image_range =
        { .width  = rectified_width,
          .height = rectified_height,
          .stride = (int)(rectified_width*sizeof(double)),
          .data   = (double*)aligned_alloc(0x10,
                                           rectified_height*
                                           rectified_width*sizeof(double))
        };
    if(image_range.data == NULL)
    {
        fprintf(stderr, "Error: malloc() failed\n");
        return 1;
    }
    if(!mrcal_stereo_range_dense(&image_range,
                                 &image_disparity,
                                 disparity_scale,
                                 disparity_min * disparity_scale,
                                 disparity_max * disparity_scale,
                                 model_rectified0.lensmodel.type,
                                 model_rectified0.intrinsics,
                                 baseline))
    {
        fprintf(stderr, "Error: mrcal_stereo_range_dense() failed\n");
        return 1;
    }
    mrcal_image_bgr_t image_color_range =
        { .width  = rectified_width,
          .height = rectified_height,
          .stride = (int)(rectified_width*sizeof(mrcal_bgr_t)),
          .data   = (mrcal_bgr_t*)malloc(rectified_height*
                                         rectified_width*sizeof(mrcal_bgr_t))
        };
    if(image_color_range.data == NULL)
    {
        fprintf(stderr, "Error: malloc() failed\n");
        return 1;
    }
    const double range_min = 0;
    const double range_max = 1000;
    if(!mrcal_apply_color_map_double(&image_color_range,
                                     &image_range,
                                     // no auto range
                                     false,false,
                                     // auto function
                                     true,
                                     range_min,
                                     range_max,
                                     // ignored functions
                                     0,0,0))
    {
        fprintf(stderr, "Error: mrcal_apply_color_map_double() failed\n");
        return 1;
    }
    mrcal_image_bgr_save("/tmp/range.png", &image_color_range);
    fprintf(stderr, "Wrote '/tmp/range.png'\n");

    return 0;
}
