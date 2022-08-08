#include <FreeImage.h>
#include <malloc.h>
#include <string.h>

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

    // This may actually be a different mrcal_image_xxx_t type, but all the
    // fields line up anyway
    const mrcal_image_uint8_t* image = (const mrcal_image_uint8_t*)_image;


    // I would like to avoid copying the image buffer by reusing the data, and
    // just make a new header. Like this:
    //
    //     FIBITMAP* fib = FreeImage_ConvertFromRawBitsEx(false,
    //                                                    (BYTE*)image->data,
    //                                                    FIT_BITMAP,
    //                                                    image->width, image->height, image->stride,
    //                                                    bits_per_pixel,
    //                                                    0,0,0,
    //                                                    // Top row is stored first
    //                                                    true);
    //
    // But apparently freeimage can't just do this like the user expects: they
    // actually move the data around in the input image to flip it upside-down.
    // This function should not be modifying its input, and fighting this isn't
    // worth my time. So I let freeimage make a copy of the image, and then muck
    // around with the new buffer however much it likes.
    FIBITMAP* fib = FreeImage_ConvertFromRawBits( (BYTE*)image->data,
                                                  image->width, image->height, image->stride,
                                                  bits_per_pixel,
                                                  0,0,0,
                                                  // Top row is stored first
                                                  true);

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
                  unsigned int bits_per_pixel,

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

    if(bits_per_pixel == 8)
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
    else if(bits_per_pixel == 16)
    {
        color_type_expected = FIC_MINISBLACK;
        what_expected = "16-bit grayscale";

        // At this time, 16bpp grayscale images can only be read directly from
        // the input. I cannot be given a different kind of input, and convert
        // the images to 16bpp grayscale
        fib_converted = fib;
    }
    else if(bits_per_pixel == 24)
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
        MSG("bits_per_pixel must be 8 or 16 or 24; got %d", bits_per_pixel);
        goto done;
    }

    if(!(FreeImage_GetColorType(fib_converted) == color_type_expected &&
         FreeImage_GetBPP(fib_converted) == bits_per_pixel))
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
    image->data = malloc(size);
    if(image->data == NULL)
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
    return generic_load(image, 8, filename);
}

bool mrcal_image_uint16_load(// output
                            mrcal_image_uint16_t* image,

                            // input
                            const char* filename)
{
    return generic_load(image, 16, filename);
}

bool mrcal_image_bgr_load  (// output
                           mrcal_image_bgr_t* image,

                           // input
                           const char* filename)
{
    return generic_load(image, 24, filename);
}
