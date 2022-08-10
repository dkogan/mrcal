#pragma once

// mrcal images. These are completely uninteresting, and don't do anything
// better that other image read/write APIS. If you have image libraries running,
// use those. If not, the ones defined here should be light and painless

// I support several image types:
// - "uint8":  8-bit grayscale
// - "uint16": 16-bit grayscale (using the system endian-ness)
// - "bgr":    24-bit BGR color
//
// Each type defines several functions in the MRCAL_IMAGE_DECLARE() macro:
//
// - mrcal_image_TYPE_t container image
// - mrcal_image_TYPE_at(mrcal_image_TYPE_t* image, int x, int y)
// - mrcal_image_TYPE_at_const(const mrcal_image_TYPE_t* image, int x, int y)
// - mrcal_image_TYPE_t mrcal_image_TYPE_crop(mrcal_image_TYPE_t* image, in x0, int y0, int w, int h)
// - mrcal_image_TYPE_save (const char* filename, const mrcal_image_TYPE_t*  image);
// - mrcal_image_TYPE_load( mrcal_image_TYPE_t*  image, const char* filename);
//
// The image-loading functions require a few notes:
//
// An image structure to fill in is given. image->data will be allocated to the
// proper size. It is the caller's responsibility to free(image->data) when
// they're done. Usage sample:
//
//   mrcal_image_uint8_t image;
//   mrcal_image_uint8_load(&image, image_filename);
//   .... do stuff ...
//   free(image.data);
//
// mrcal_image_uint8_load() converts images to 8-bpp grayscale. Color and
// palettized images are accepted
//
// mrcal_image_uint16_load() does NOT convert images. The images being read must
// already be stored as 16bpp grayscale images
//
// mrcal_image_bgr_load() converts images to 24-bpp color

#include <stdint.h>
#include <stdbool.h>


typedef struct { uint8_t bgr[3]; } bgr_t;

#define MRCAL_IMAGE_DECLARE(T, Tname)                                   \
typedef struct                                                          \
{                                                                       \
    union                                                               \
    {                                                                   \
        /* in pixels */                                                 \
        struct {int w, h;};                                             \
        struct {int width, height;};                                    \
        struct {int cols, rows;};                                       \
    };                                                                  \
    int stride; /* in bytes  */                                         \
    T* data;                                                            \
} mrcal_image_ ## Tname ## _t;                                          \
                                                                        \
static inline                                                           \
T* mrcal_image_ ## Tname ## _at(mrcal_image_ ## Tname ## _t* image, int x, int y) \
{                                                                       \
    return &image->data[x + y*image->stride / sizeof(T)];               \
}                                                                       \
                                                                        \
static inline                                                           \
const T* mrcal_image_ ## Tname ## _at_const(const mrcal_image_ ## Tname ## _t* image, int x, int y) \
{                                                                       \
    return &image->data[x + y*image->stride / sizeof(T)];               \
}                                                                       \
                                                                        \
mrcal_image_ ## Tname ## _t                                             \
mrcal_image_ ## Tname ## _crop(mrcal_image_ ## Tname ## _t* image,      \
                              int x0, int y0,                           \
                              int w,  int h)                            \
{                                                                       \
    return (mrcal_image_ ## Tname ## _t){ .data   = mrcal_image_ ## Tname ## _at(image,x0,y0), \
                                          .stride = image->stride,      \
                                          .w      = w,                  \
                                          .h      = h };                \
}                                                                       \
                                                                        \
bool mrcal_image_ ## Tname ## _save (const char* filename, const mrcal_image_ ## Tname ## _t*  image); \
bool mrcal_image_ ## Tname ## _load( mrcal_image_ ## Tname ## _t*  image, const char* filename);


MRCAL_IMAGE_DECLARE(uint8_t,  uint8);
MRCAL_IMAGE_DECLARE(uint16_t, uint16);
MRCAL_IMAGE_DECLARE(bgr_t,    bgr);

// Load the image into whatever type is stored on disk
bool mrcal_image_anytype_load(// output
                              // This is ONE of the known types
                              mrcal_image_uint8_t* image,
                              int* bits_per_pixel,
                              int* channels,
                              // input
                              const char* filename);
