// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#define STBI_NO_HDR 1
#include <stb/stb_image.h>
#include <stb/stb_image_write.h>

// stb_image_write() cannot write 16-bit png files. So I use stb for everything
// except for writing png files. Using libpng to write all .png
#include <png.h>


#include <string.h>
#include <strings.h>
#include <stdlib.h>

#include "image.h"
#include "util.h"



static
void bgr_tofrom_rgb(mrcal_image_bgr_t* image)
{
    for(int i=0; i<image->height; i++)
    {
        mrcal_bgr_t* row = mrcal_image_bgr_at(image, 0, i);
        for(int j=0; j<image->width; j++)
        {
            mrcal_bgr_t* p = &row[j];
            uint8_t t = p->bgr[0];
            p->bgr[0] = p->bgr[2];
            p->bgr[2] = t;
        }
    }
}

const char* get_extension(const char* filename)
{
    const int filename_len = strlen(filename);
    if(filename_len < 5)
    {
        MSG("Image must be xxx.png or xxx.jpg; the name is too short");
        return NULL;
    }

    return &filename[filename_len - 4];
}

static
bool generic_save_png(const char* filename,
                      const mrcal_image_void_t* image,
                      const int bits_per_pixel,
                      const int channels)
{
    bool result = false;

    png_image pimage = { .version = PNG_IMAGE_VERSION,
                         .width   = image->width,
                         .height  = image->height,
                         .format  =
                             bits_per_pixel == 24 ? PNG_FORMAT_RGB :
                                 (bits_per_pixel == 16 ? PNG_FORMAT_LINEAR_Y : PNG_FORMAT_GRAY) };
    if(!png_image_write_to_file(&pimage, filename, 0, image->data,
                                bits_per_pixel == 16 ? image->stride/2 : image->stride,
                                NULL))
    {
        MSG("png_image_write_to_file('%s') failed", filename);
        goto done;
    }
    result = true;

done:
    png_image_free(&pimage);
    return result;
}

static
bool generic_save(const char* filename,
                  /* This really is const */ mrcal_image_void_t* image,
                  const int bits_per_pixel)
{
    bool result = false;
    char* buf = NULL;

    if(image->w == 0 || image->h == 0)
    {
        MSG("Asked to save an empty image: dimensions (%d,%d)!",
            image->w, image->h);
        goto done;
    }

    int channels = -1;
    if(bits_per_pixel == 8 || bits_per_pixel == 16)
        channels = 1;
    else if(bits_per_pixel == 24)
        channels = 3;
    else
    {
        MSG("bits_per_pixel must be 8 or 16 or 24");
        goto done;
    }

    if(bits_per_pixel == 24)
    {
        // bgr/rgb
        buf = malloc(image->w*image->h*3);
        if(buf == NULL)
        {
            MSG("Could not malloc buffer for bgr<->rgb");
            goto done;
        }
        if(image->stride == image->width*3)
            memcpy(buf, image->data, image->w*image->h*3);
        else
            for(int i=0; i<image->h; i++)
                memcpy(&buf[i*image->width*3],
                       &((uint8_t*)image->data)[i*image->stride],
                       image->width*3);
        image->data = (uint8_t*)buf;
        image->stride = image->width*3;
        bgr_tofrom_rgb( (mrcal_image_bgr_t*)image);
    }

    // stb_image_write() cannot write 16-bit png files. So I use stb for
    // everything except for writing png files. Using libpng to write all .png
    const char* extension = get_extension(filename);
    if(extension == NULL) goto done;

    if(0 == strcasecmp(extension, ".png"))
    {
        result = generic_save_png(filename, image, bits_per_pixel, channels);
        goto done;
    }

    if(0 == strcasecmp(extension, ".jpg"))
    {
        if(image->stride != image->width*bits_per_pixel/8)
        {
            MSG("jpg writing requires a densely-stored image");
            goto done;
        }
        if(!stbi_write_jpg(filename,
                           image->w, image->h,
                           channels,
                           image->data,
                           96))
        {
            MSG("stbi_write_jpg(\"%s\") failed", filename);
            goto done;
        }
    }
    else
    {
        MSG("The path being written MUST be XXX.png or XXX.jpg");
        goto done;
    }
    result = true;

 done:
    free(buf);
    return result;
}

bool mrcal_image_uint8_save (const char* filename, const mrcal_image_uint8_t* image)
{
    return generic_save(filename, (mrcal_image_void_t*)image, 8);
}

bool mrcal_image_uint16_save(const char* filename, const mrcal_image_uint16_t* image)
{
    return generic_save(filename, (mrcal_image_void_t*)image, 16);
}

bool mrcal_image_bgr_save(const char* filename, const mrcal_image_bgr_t* image)
{
    return generic_save(filename, (mrcal_image_void_t*)image, 24);
}


static
void stretch_equalization_uint8_from_uint16(mrcal_image_uint8_t* out,
                                            const mrcal_image_uint16_t* in)
{
    uint16_t min = UINT16_MAX;
    uint16_t max = 0;

    for(int i=0; i<in->height; i++)
    {
        const uint16_t* row_in = mrcal_image_uint16_at_const(in, 0, i);
        for (int j=0; j<in->width; j++)
        {
            const uint16_t x = row_in[j];
            if      (x < min) min = x;
            else if (x > max) max = x;
        }
    }

    uint16_t max_min = max-min;

    for(int i=0; i<in->height; i++)
    {
        const uint16_t* row_in  = mrcal_image_uint16_at_const(in,  0, i);
        uint8_t*        row_out = mrcal_image_uint8_at       (out, 0, i);

        for (int j=0; j<in->width; j++)
        {
            const uint16_t x = row_in[j];
            row_out[j] = (uint8_t)(0.5f + ((float)(x - min) * 255.f / (float)max_min));
        }
    }
}








static
bool generic_load(// output

                  // mrcal_image_uint8_t  if bits_per_pixel == 8
                  // mrcal_image_uint16_t if bits_per_pixel == 16
                  // mrcal_image_bgr_t    if bits_per_pixel == 24
                  mrcal_image_void_t* image,
                  // if >0: this is the requested bits_per_pixel. If == 0: we
                  // get this from the input image, and set the value on the
                  // return
                  int* bits_per_pixel,

                  // input
                  const char* filename)
{
    bool result = false;

    unsigned char* image_buf = NULL;
    int width=0, height=0, channels=0;

    FILE* fp = fopen(filename, "r");
    if(fp == NULL)
    {
        MSG("Couldn't open image: fopen(\"%s\") failed", filename);
        goto done;
    }

    if(*bits_per_pixel == 0)
    {
        // autodetect *bits_per_pixel. This path is ONLY for *bits_per_pixel.
        // The actual work is done in the if() below
        int width,height,channels;

        if(!stbi_info_from_file(fp, &width, &height, &channels))
        {
            MSG("Couldn't load image: stbi_info_from_file(\"%s\") failed", filename);
            goto done;
        }
        bool is_16bit = (bool)stbi_is_16_bit_from_file(fp);
#warning rgba
        if(!is_16bit)
        {
            if(channels == 3)
                *bits_per_pixel = 24;
            else if(channels == 1)
                *bits_per_pixel = 8;
            else
            {
                MSG("Couldn't load image \"%s\" 8-bit image: I only support 1-channel and 3-channel images", filename);
                goto done;
            }
        }
        else
        {
            if(channels == 1)
                *bits_per_pixel = 16;
            else
            {
                MSG("Couldn't load image \"%s\" 16-bit image: I only support 1-channel", filename);
                goto done;
            }
        }
    }

    if(*bits_per_pixel == 8)
    {
        const bool is_16bit = (bool)stbi_is_16_bit_from_file(fp);

        if(is_16bit)
        {
            // special case: uint16 monochrome image. I apply stretch
            // equalization
            image_buf = (unsigned char*)stbi_load_from_file_16(fp, &width, &height, &channels, 1);
            if(image_buf == NULL)
            {
                MSG("Couldn't load image: stbi_load_from_file_16(\"%s\", desired_channels=1) failed", filename);
                goto done;
            }

            // allocate new image
            int size = width*height;
            unsigned char* image_buf_8bit;
            if(posix_memalign((void**)&image_buf_8bit, 16UL, size) != 0)
            {
                MSG("couldn't allocate image: malloc(%d) failed",
                    size);
                goto done;
            }

            mrcal_image_uint16_t in =
                { .width  = (int)      width,
                  .height = (int)      height,
                  .stride = (int)      width*2,
                  .data   = (uint16_t*)image_buf
                };

            mrcal_image_uint8_t out =
                { .width  = width,
                  .height = height,
                  .stride = width,
                  .data   = (uint8_t*)image_buf_8bit
                };
            stretch_equalization_uint8_from_uint16(&out, &in);
            free(image_buf);
            image_buf = image_buf_8bit;
        }
        else
        {
            image_buf = stbi_load_from_file(fp, &width, &height, &channels, 1);
            if(image_buf == NULL)
            {
                MSG("Couldn't load image: stbi_load_from_file(\"%s\", desired_channels=1) failed", filename);
                goto done;
            }
        }
    }
    else if(*bits_per_pixel == 16)
    {
        image_buf = (unsigned char*)stbi_load_from_file_16(fp, &width, &height, &channels, 1);
        if(image_buf == NULL)
        {
            MSG("Couldn't load image: stbi_load_from_file_16(\"%s\", desired_channels=1) failed", filename);
            goto done;
        }
    }
    else if(*bits_per_pixel == 24)
    {
        image_buf = stbi_load_from_file(fp, &width, &height, &channels, 3);
        if(image_buf == NULL)
        {
            MSG("Couldn't load image: stbi_load_from_file(\"%s\", desired_channels=3) failed", filename);
            goto done;
        }
    }
    else
    {
        MSG("Input bits_per_pixel must be 8 or 16 or 24; got %d", *bits_per_pixel);
        goto done;
    }

    image->width  = width;
    image->height = height;
    image->stride = width * (*bits_per_pixel)/8;
    image->data   = image_buf;

    if(*bits_per_pixel == 24)
    {
        // we loaded rgb, but I want bgr
        bgr_tofrom_rgb((mrcal_image_bgr_t*)image);
    }

    image_buf = NULL; // to not free

    result = true;

 done:
    if(fp != NULL)
        fclose(fp);
    stbi_image_free(image_buf);
    return result;
}


bool mrcal_image_uint8_load(// output
                           mrcal_image_uint8_t* image,

                           // input
                           const char* filename)
{
    int bits_per_pixel = 8;
    return generic_load((mrcal_image_void_t*)image, &bits_per_pixel, filename);
}

bool mrcal_image_uint16_load(// output
                            mrcal_image_uint16_t* image,

                            // input
                            const char* filename)
{
    int bits_per_pixel = 16;
    return generic_load((mrcal_image_void_t*)image, &bits_per_pixel, filename);
}

bool mrcal_image_bgr_load  (// output
                           mrcal_image_bgr_t* image,

                           // input
                           const char* filename)
{
    int bits_per_pixel = 24;
    return generic_load((mrcal_image_void_t*)image, &bits_per_pixel, filename);
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
    if(!generic_load((mrcal_image_void_t*)image, bits_per_pixel, filename))
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
