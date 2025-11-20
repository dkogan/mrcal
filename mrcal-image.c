// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <FreeImage.h>
#include <stdlib.h>
#include <string.h>
#include <stdlib.h>

#include "mrcal-image.h"
#include "util.h"

// for diagnostics
__attribute__((unused))
static void report_image_details(/* const; FreeImage doesn't support that */
                                 FIBITMAP* fib, const char* what)
{
    MSG("%s colortype = %d bpp = %d dimensions: (%d,%d), pitch = %d",
        what,
        (int)FreeImage_GetColorType(fib),
        (int)FreeImage_GetBPP      (fib),
        (int)FreeImage_GetWidth    (fib),
        (int)FreeImage_GetHeight   (fib),
        (int)FreeImage_GetPitch    (fib));
}


static
bool generic_save(const char* filename,
                  const void* _image,
                  int bits_per_pixel)
{
    bool result = false;

    FIBITMAP* fib = NULL;

    // This may actually be a different mrcal_image_xxx_t type, but all the
    // fields line up anyway
    const mrcal_image_uint8_t* image = (const mrcal_image_uint8_t*)_image;

    if(image->w == 0 || image->h == 0)
    {
        MSG("Asked to save an empty image: dimensions (%d,%d)!",
            image->w, image->h);
        goto done;
    }

#if defined HAVE_OLD_LIBFREEIMAGE && HAVE_OLD_LIBFREEIMAGE
    if(bits_per_pixel == 16)
        MSG("WARNING: you have an old build of libfreeimage. It has trouble writing 16bpp images, so '%s' will probably be written incorrectly. You should upgrade your libfreeimage and rebuild",
            filename);
    fib = FreeImage_ConvertFromRawBits( (BYTE*)image->data,
                                        image->width, image->height, image->stride,
                                        bits_per_pixel,
                                        0,0,0,
                                        // Top row is stored first
                                        true);
#else
    // I do NOT reuse the input data because this function actually changes the
    // input buffer to flip it upside-down instead of accessing the bits in the
    // correct order
    fib = FreeImage_ConvertFromRawBitsEx(true,
                                         (BYTE*)image->data,
                                         (bits_per_pixel == 16) ? FIT_UINT16 : FIT_BITMAP,
                                         image->width, image->height, image->stride,
                                         bits_per_pixel,
                                         0,0,0,
                                         // Top row is stored first
                                         true);
#endif

    FREE_IMAGE_FORMAT format = FreeImage_GetFIFFromFilename(filename);
    if(format == FIF_UNKNOWN)
    {
        MSG("FreeImage doesn't know how to save '%s'", filename);
        goto done;
    }

    int flags = format == FIF_JPEG ? 96 : 0;
    if(!FreeImage_Save(format, fib, filename, flags))
    {
        MSG("FreeImage couldn't save '%s'", filename);
        goto done;
    }
    result = true;

 done:
    if(fib != NULL)
        FreeImage_Unload(fib);

    return result;
}

bool mrcal_image_uint8_save (const char* filename, const mrcal_image_uint8_t* image)
{
    return generic_save(filename, image, 8);
}

bool mrcal_image_uint16_save(const char* filename, const mrcal_image_uint16_t* image)
{
    return generic_save(filename, image, 16);
}

bool mrcal_image_bgr_save(const char* filename, const mrcal_image_bgr_t* image)
{
    return generic_save(filename, image, 24);
}

static
bool generic_load(// output

                  // mrcal_image_uint8_t  if bits_per_pixel == 8
                  // mrcal_image_uint16_t if bits_per_pixel == 16
                  // mrcal_image_bgr_t    if bits_per_pixel == 24
                  void* _image,
                  // if >0: this is the requested bits_per_pixel. If == 0: we
                  // get this from the input image, and set the value on the
                  // return
                  int* bits_per_pixel,

                  // input
                  const char* filename)
{
    bool      result        = false;
    FIBITMAP* fib           = NULL;
    FIBITMAP* fib_converted = NULL;

    FREE_IMAGE_FORMAT format = FreeImage_GetFileType(filename,0);
    if(format == FIF_UNKNOWN)
    {
        MSG("Couldn't load '%s': FreeImage_GetFileType() failed", filename);
        goto done;
    }

    fib = FreeImage_Load(format, filename, 0);
    if(fib == NULL)
    {
        MSG("Couldn't load '%s': FreeImage_Load() failed", filename);
        goto done;
    }

    // FreeImage loads images upside-down, so I flip it around
    if(!FreeImage_FlipVertical(fib))
    {
        MSG("Couldn't flip the image");
        goto done;
    }

    // might not be "uint8_t" necessarily, but all the fields still line up
    mrcal_image_uint8_t* image = NULL;
    FREE_IMAGE_COLOR_TYPE color_type_expected;
    const char* what_expected;

    if(*bits_per_pixel == 0)
    {
        // autodetect
        FREE_IMAGE_COLOR_TYPE color_type_have = FreeImage_GetColorType(fib);
        if(color_type_have == FIC_RGB ||
           color_type_have == FIC_PALETTE)
        {
            *bits_per_pixel = 24;
        }
        else if(color_type_have == FIC_MINISBLACK)
        {
            unsigned int bpp_have = FreeImage_GetBPP(fib);
            if(bpp_have == 16)
                *bits_per_pixel = 16;
            else
                *bits_per_pixel = 8;
        }
        else
        {
            MSG("Couldn't auto-detect image type. I only know about FIC_RGB, FIC_PALETTE, FIC_MINISBLACK");
            goto done;
        }
    }

    if(*bits_per_pixel == 8)
    {
        color_type_expected = FIC_MINISBLACK;
        what_expected = "grayscale";

        fib_converted = FreeImage_ConvertToGreyscale(fib);
        if(fib_converted == NULL)
        {
            MSG("Couldn't FreeImage_ConvertToGreyscale()");
            goto done;
        }
    }
    else if(*bits_per_pixel == 16)
    {
        color_type_expected = FIC_MINISBLACK;
        what_expected = "16-bit grayscale";

        // At this time, 16bpp grayscale images can only be read directly from
        // the input. I cannot be given a different kind of input, and convert
        // the images to 16bpp grayscale
        fib_converted = fib;
    }
    else if(*bits_per_pixel == 24)
    {
        color_type_expected = FIC_RGB;
        what_expected = "bgr 24-bit";

        fib_converted = FreeImage_ConvertTo24Bits(fib);
        if(fib_converted == NULL)
        {
            MSG("Couldn't FreeImage_ConvertTo24Bits()");
            goto done;
        }
    }
    else
    {
        MSG("Input bits_per_pixel must be 8 or 16 or 24; got %d", *bits_per_pixel);
        goto done;
    }

    if(!(FreeImage_GetColorType(fib_converted) == color_type_expected &&
         FreeImage_GetBPP(fib_converted) == (unsigned)*bits_per_pixel))
    {
        MSG("Loaded and preprocessed image isn't %s",
            what_expected);
        goto done;
    }
    // This may actually be a different mrcal_image_xxx_t type, but all the
    // fields line up anyway
    image = (mrcal_image_uint8_t*)_image;

    image->width  = (int)FreeImage_GetWidth (fib_converted);
    image->height = (int)FreeImage_GetHeight(fib_converted);
    image->stride = (int)FreeImage_GetPitch (fib_converted);

    int size = image->stride*image->height;
    if(posix_memalign((void**)&image->data, 16UL, size) != 0)
    {
        MSG("%s('%s') couldn't allocate image: malloc(%d) failed",
            __func__, filename, size);
        goto done;
    }

    memcpy( image->data,
            FreeImage_GetBits(fib_converted),
            size );

    result = true;

 done:
    if(fib != NULL)
        FreeImage_Unload(fib);
    if(fib_converted != NULL &&
       fib_converted != fib)
        FreeImage_Unload(fib_converted);
    return result;
}


bool mrcal_image_uint8_load(// output
                           mrcal_image_uint8_t* image,

                           // input
                           const char* filename)
{
    int bits_per_pixel = 8;
    return generic_load(image, &bits_per_pixel, filename);
}

bool mrcal_image_uint16_load(// output
                            mrcal_image_uint16_t* image,

                            // input
                            const char* filename)
{
    int bits_per_pixel = 16;
    return generic_load(image, &bits_per_pixel, filename);
}

bool mrcal_image_bgr_load  (// output
                           mrcal_image_bgr_t* image,

                           // input
                           const char* filename)
{
    int bits_per_pixel = 24;
    return generic_load(image, &bits_per_pixel, filename);
}

bool mrcal_image_anytype_load(// output
                              // This is ONE of the known types
                              mrcal_image_uint8_t* image,
                              int* bits_per_pixel,
                              int* channels,
                              // input
                              const char* filename)
{
    *bits_per_pixel = 0;
    if(!generic_load(image, bits_per_pixel, filename))
        return false;

    switch(*bits_per_pixel)
    {
    case 8:
    case 16:
        *channels = 1;
        break;
    case 24:
        *channels = 3;
        break;

    default:
        MSG("Getting here is a bug");
        return false;
    }

    return true;
}
