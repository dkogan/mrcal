// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#define STBI_NO_HDR 1
#include <stb/stb_image.h>

// stb_image_write() cannot write 16-bit png files and it cannot write 8-bit jpg
// files AS 8-BIT FILES. So I use stb for reading but other things for writing.
// No stb_image_write.
#include <png.h>
#include <jpeglib.h>
#include <setjmp.h>


#include <string.h>
#include <strings.h>
#include <stdlib.h>

#include "image.h"
#include "_util.h"



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

static
const char* get_extension(const char* filename)
{
    const int filename_len = strlen(filename);
    if(filename_len < 5)
    {
        MSG("Image must be XXX.png or XXX.jpg or XXX.pgm or XXX.ppm; the name is too short");
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
bool generic_save_pnm(const char* filename,
                      const mrcal_image_void_t* image,
                      const int bits_per_pixel,
                      const int channels)
{
    bool result = false;
    FILE* fp = NULL;

    if(!( (channels == 1 && (bits_per_pixel == 8 || bits_per_pixel == 16)) ||
          (channels == 3 && bits_per_pixel == 24)))
    {
        MSG("Unsupported image format. I know about 1-channel 8- or 16-bit images; or 3-channel 24-bit images. Got channels==%d bits_per_pixel==%d", channels, bits_per_pixel);
        goto done;
    }

    if(NULL == (fp = fopen(filename, "wb")))
    {
        MSG("Couldn't open '%s'", filename);
        return false;
    }

    fprintf(fp,
            "P%d\n%d %d\n%d\n",
            channels == 1 ? 5 : 6,
            image->width,
            image->height,
            bits_per_pixel == 16 ? 65535 : 255);

    // I write each row separately, in case the rows aren't stored densely
    for(int i=0; i<image->height; i++)
    {
        if( (size_t)image->width !=
            fwrite(&((const uint8_t*)image->data)[ image->stride*i ],
                   bits_per_pixel/8,
                   image->width,
                   fp) )
        {
            MSG("Didn't write the correct number of bytes into '%s'; giving up", filename);
            goto done;
        }
    }
    result = true;

done:
    if(fp != NULL) fclose(fp);
    return result;
}

static
bool generic_save_pgm(const char* filename,
                      const mrcal_image_void_t* image,
                      const int bits_per_pixel,
                      const int channels)
{
    if(channels != 1)
    {
        MSG(".pgm files are by definition 1-channel images only; requested channels=%d",
            channels);
        return false;
    }
    return generic_save_pnm(filename,
                            image,
                            bits_per_pixel,
                            channels);
}

static
bool generic_save_ppm(const char* filename,
                      const mrcal_image_void_t* image,
                      const int bits_per_pixel,
                      const int channels)
{
    if(channels != 3)
    {
        MSG(".ppm files are by definition 3-channel images only; requested channels=%d",
            channels);
        return false;
    }
    return generic_save_pnm(filename,
                            image,
                            bits_per_pixel,
                            channels);
}


// Error handling in libjpeg
struct my_error_mgr {
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};
static void my_error_exit(j_common_ptr cinfo) {
    struct my_error_mgr* myerr = (struct my_error_mgr *)cinfo->err;
    (*cinfo->err->output_message) (cinfo);
    longjmp(myerr->setjmp_buffer, 1);
}
static
bool generic_save_jpg(const char* filename,
                      const mrcal_image_void_t* image,
                      const int bits_per_pixel,
                      const int channels)
{
    if(bits_per_pixel == 16)
    {
        MSG("16bpp jpeg not supported");
        return false;
    }

    bool result = false;

    FILE* fp = NULL;

    if(NULL == (fp = fopen(filename, "wb")))
    {
        MSG("Couldn't open '%s'", filename);
        return false;
    }

    struct my_error_mgr jerr = {};
    struct jpeg_compress_struct cinfo = {.err = jpeg_std_error(&jerr.pub)};
    jerr.pub.error_exit = my_error_exit;

    if (setjmp(jerr.setjmp_buffer))
    {
        jpeg_destroy_compress(&cinfo);
        fclose(fp);
        return false;
    }

    jpeg_create_compress(&cinfo);
    jpeg_stdio_dest(&cinfo, fp);

    cinfo.image_width      = image->width;
    cinfo.image_height     = image->height;
    cinfo.input_components = channels;
    cinfo.in_color_space   = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, 96, TRUE);
    if(channels == 3)
        jpeg_set_colorspace(&cinfo, JCS_RGB);
    jpeg_start_compress(&cinfo, TRUE);

    JSAMPROW row_pointer[1];
    while (cinfo.next_scanline < cinfo.image_height) {
        row_pointer[0] = &((uint8_t*)image->data)[cinfo.next_scanline * image->stride];
        jpeg_write_scanlines(&cinfo, row_pointer, 1);
    }

    jpeg_finish_compress(&cinfo);
    fclose(fp);
    jpeg_destroy_compress(&cinfo);

    return true;
}


static
bool generic_save(const char* filename,
                  const mrcal_image_void_t* image,
                  const int bits_per_pixel)
{
    bool result = false;
    mrcal_image_bgr_t _image = {}; // used in case bgr <-> rgb

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

        // make a copy of the image structure, allocate a new data buffer, and
        // populate it from the original data
        _image = (mrcal_image_bgr_t){.width  = image->width,
                                     .height = image->height,
                                     .stride = image->width*3,
                                     .data   = malloc(image->w*image->h*3 )};
        if(_image.data == NULL)
        {
            MSG("Could not malloc buffer for bgr<->rgb");
            goto done;
        }
        if(_image.stride == image->stride)
            memcpy(_image.data, image->data, _image.w*_image.h*3);
        else
            for(int i=0; i<_image.h; i++)
                memcpy(&((uint8_t*)_image.data)[i*_image.stride],
                       &((uint8_t*)image->data)[i*image->stride],
                       _image.width*3);
        bgr_tofrom_rgb( (mrcal_image_bgr_t*)&_image);

        image = (const mrcal_image_void_t*)&_image;
    }

    const char* extension = get_extension(filename);
    if(extension == NULL) goto done;

    if(0 == strcasecmp(extension, ".png"))
    {
        result = generic_save_png(filename, image, bits_per_pixel, channels);
        goto done;
    }
    if(0 == strcasecmp(extension, ".jpg"))
    {
#if 0
        // Not using stbi_write_jpg() to do this because it always saves color
        // jpg files. Even if I give it grayscale input
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
        result = true;

#else
        result = generic_save_jpg(filename, image, bits_per_pixel, channels);
#endif
        goto done;
    }
    if(0 == strcasecmp(extension, ".pgm"))
    {
        result = generic_save_pgm(filename, image, bits_per_pixel, channels);
        goto done;
    }
    if(0 == strcasecmp(extension, ".ppm"))
    {
        result = generic_save_ppm(filename, image, bits_per_pixel, channels);
        goto done;
    }

    {
        MSG("The path being written MUST be XXX.png or XXX.jpg or XXX.pgm or XXX.ppm");
        goto done;
    }
    result = true;

 done:
    free(_image.data);
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








typedef struct {
    jmp_buf         jmpbuf;
    unsigned char*  image_buf;
    unsigned char** row_pointers;
    bool            result;
} load_png_ctx_t;

static void load_png_on_error(png_structp png_ptr, png_const_charp msg)
{
    load_png_ctx_t* ctx = png_get_error_ptr(png_ptr);
    free(ctx->row_pointers); ctx->row_pointers = NULL;
    free(ctx->image_buf);    ctx->image_buf    = NULL;
    longjmp(ctx->jmpbuf, 1);
}

static
bool load_png(// out
              mrcal_image_void_t* image,
              int* bits_per_pixel,
              // in
              const char* filename, // for error reporting
              FILE* fp)
{
    // load_png() and generic_load() have a very similar structure. Please
    // update them together

    // Written by Claude

    int         width    = 0;
    int         height   = 0;
    png_structp png_ptr  = NULL;
    png_infop   info_ptr = NULL;
    // image_buf and row_pointers live in ctx so their address is taken;
    // GCC won't warn about them in the presence of setjmp
    load_png_ctx_t ctx = {};

    if(NULL == (png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, &ctx, load_png_on_error, NULL)))
    {
        MSG("png_create_read_struct() failed");
        goto done;
    }
    if(NULL == (info_ptr = png_create_info_struct(png_ptr)))
    {
        MSG("png_create_info_struct() failed");
        goto done;
    }
    if(setjmp(ctx.jmpbuf))
    {
        MSG("libpng error while reading '%s'", filename);
        goto done;
    }
    png_init_io(png_ptr, fp);
    png_read_info(png_ptr, info_ptr);

    {
        int bit_depth  = png_get_bit_depth (png_ptr, info_ptr);
        int color_type = png_get_color_type(png_ptr, info_ptr);

        // Normalise: expand palette→RGB, low-depth gray→8-bit, strip alpha
        if(color_type == PNG_COLOR_TYPE_PALETTE)
            png_set_palette_to_rgb(png_ptr);
        if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
            png_set_expand_gray_1_2_4_to_8(png_ptr);
        if(color_type & PNG_COLOR_MASK_ALPHA)
            png_set_strip_alpha(png_ptr);
        // PNG stores 16-bit values big-endian; swap to host byte order
        if(bit_depth == 16)
            png_set_swap(png_ptr);
        // If caller requests 3-channel BGR from an 8-bit grayscale file, expand
        if(*bits_per_pixel == 24 &&
           (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) &&
           bit_depth <= 8)
            png_set_gray_to_rgb(png_ptr);

        // Re-read after registering transforms: e.g. palette→RGB changes
        // color_type, expand_gray_1_2_4_to_8 changes bit_depth
        png_read_update_info(png_ptr, info_ptr);
        bit_depth  = png_get_bit_depth (png_ptr, info_ptr);
        color_type = png_get_color_type(png_ptr, info_ptr);

        int bits_per_pixel_detected;
        if(color_type == PNG_COLOR_TYPE_GRAY)
        {
            if     (bit_depth == 8)  bits_per_pixel_detected = 8;
            else if(bit_depth == 16) bits_per_pixel_detected = 16;
            else
            {
                MSG("Unsupported PNG bit depth %d in '%s'", bit_depth, filename);
                goto done;
            }
        }
        else if(color_type == PNG_COLOR_TYPE_RGB)
            bits_per_pixel_detected = 24;
        else
        {
            MSG("Couldn't load image \"%s\": unsupported PNG color type %d", filename, color_type);
            goto done;
        }

        const int stride_raw = (int)png_get_rowbytes(png_ptr, info_ptr);
        width  = (int)png_get_image_width (png_ptr, info_ptr);
        height = (int)png_get_image_height(png_ptr, info_ptr);

        if(*bits_per_pixel == 0)
            *bits_per_pixel = bits_per_pixel_detected;

        if(*bits_per_pixel == 8)
        {
            if(posix_memalign((void**)&ctx.image_buf, 16UL, (size_t)stride_raw * height) != 0)
            { MSG("posix_memalign() failed for %zu bytes", (size_t)stride_raw * height); goto done; }
            if(NULL == (ctx.row_pointers = malloc(height * sizeof(unsigned char*))))
            { MSG("malloc() failed for row pointers"); goto done; }
            for(int i = 0; i < height; i++)
                ctx.row_pointers[i] = ctx.image_buf + (size_t)i * stride_raw;
            png_read_image(png_ptr, ctx.row_pointers);
            free(ctx.row_pointers); ctx.row_pointers = NULL;

            if(bits_per_pixel_detected == 16)
            {
                unsigned char* image_buf_8bit;
                if(posix_memalign((void**)&image_buf_8bit, 16UL, (size_t)width * height) != 0)
                { MSG("posix_memalign() failed for %zu bytes", (size_t)width * height); goto done; }
                mrcal_image_uint16_t in  = { .width = width, .height = height, .stride = stride_raw, .data = (uint16_t*)ctx.image_buf };
                mrcal_image_uint8_t  out = { .width = width, .height = height, .stride = width,       .data = (uint8_t*)image_buf_8bit };
                stretch_equalization_uint8_from_uint16(&out, &in);
                free(ctx.image_buf);
                ctx.image_buf = image_buf_8bit;
            }
        }
        else if(*bits_per_pixel == 16)
        {
            if(bits_per_pixel_detected != 16)
            {
                MSG("Loading %d-bit image as a 16-bit image not supported",
                    bits_per_pixel_detected);
                goto done;
            }

            if(posix_memalign((void**)&ctx.image_buf, 16UL, (size_t)stride_raw * height) != 0)
            { MSG("posix_memalign() failed for %zu bytes", (size_t)stride_raw * height); goto done; }
            if(NULL == (ctx.row_pointers = malloc(height * sizeof(unsigned char*))))
            { MSG("malloc() failed for row pointers"); goto done; }
            for(int i = 0; i < height; i++)
                ctx.row_pointers[i] = ctx.image_buf + (size_t)i * stride_raw;
            png_read_image(png_ptr, ctx.row_pointers);
            free(ctx.row_pointers); ctx.row_pointers = NULL;
        }
        else if(*bits_per_pixel == 24)
        {
            if(bits_per_pixel_detected == 16)
            {
                MSG("Loading 16-bit image as a 24-bit image not supported");
                goto done;
            }

            if(posix_memalign((void**)&ctx.image_buf, 16UL, (size_t)stride_raw * height) != 0)
            { MSG("posix_memalign() failed for %zu bytes", (size_t)stride_raw * height); goto done; }
            if(NULL == (ctx.row_pointers = malloc(height * sizeof(unsigned char*))))
            { MSG("malloc() failed for row pointers"); goto done; }
            for(int i = 0; i < height; i++)
                ctx.row_pointers[i] = ctx.image_buf + (size_t)i * stride_raw;
            png_read_image(png_ptr, ctx.row_pointers);
            free(ctx.row_pointers); ctx.row_pointers = NULL;
        }
        else
        {
            MSG("Input bits_per_pixel must be 8 or 16 or 24; got %d", *bits_per_pixel);
            goto done;
        }
    }

    image->width  = width;
    image->height = height;
    image->stride = width * (*bits_per_pixel) / 8;
    image->data   = ctx.image_buf;
    ctx.image_buf = NULL;

    if(*bits_per_pixel == 24)
        bgr_tofrom_rgb((mrcal_image_bgr_t*)image);

    ctx.result = true;

done:
    free(ctx.row_pointers);
    free(ctx.image_buf);
    if(png_ptr) png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
    return ctx.result;
}

static
bool detect_image_parameters(// out
                             int* width,
                             int* height,
                             int* bits_per_pixel,
                             int* channels,
                             // in
                             const char* filename, // for error reporting
                             FILE* fp)
{
    if(!stbi_info_from_file(fp, width, height, channels))
    {
        MSG("Couldn't load image: stbi_info_from_file(\"%s\") failed", filename);
        return false;
    }
    const bool is_16bit = (bool)stbi_is_16_bit_from_file(fp);
    if(!is_16bit)
    {
        if(*channels == 3 || *channels == 4)
            *bits_per_pixel = 24;
        else if(*channels == 1)
            *bits_per_pixel = 8;
        else
        {
            MSG("Couldn't load image \"%s\" 8-bit image: I only support 1- and 3- and 4-channel images (alpha ignored)", filename);
            return false;
        }
    }
    else
    {
        if(*channels == 1)
            *bits_per_pixel = 16;
        else
        {
            MSG("Couldn't load image \"%s\" 16-bit image: I only support 1-channel", filename);
            return false;
        }
    }
    return true;
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
    int            width     = 0;
    int            height    = 0;
    int            channels  = 0;

    FILE* fp = fopen(filename, "r");
    if(fp == NULL)
    {
        MSG("Couldn't open image: fopen(\"%s\") failed", filename);
        goto done;
    }

    const char* extension = get_extension(filename);
    if(extension == NULL) goto done;
    if(0 == strcasecmp(extension, ".png"))
    {
        result = load_png(image, bits_per_pixel,
                          filename, fp);
        goto done;
    }

    // load_png() and generic_load() have a very similar structure. Please
    // update them together

    int width_detected;
    int height_detected;
    int bits_per_pixel_detected;
    int channels_detected;


    if(!detect_image_parameters(// out
                                &width_detected,
                                &height_detected,
                                &bits_per_pixel_detected,
                                &channels_detected,

                                // in
                                filename,
                                fp))
        goto done;

    if(*bits_per_pixel == 0)
        *bits_per_pixel = bits_per_pixel_detected;

    if(*bits_per_pixel == 8)
    {
        if(bits_per_pixel_detected == 16)
        {
            // special case: uint16 monochrome image. I apply stretch
            // equalization
            image_buf = (unsigned char*)stbi_load_from_file_16(fp, &width, &height, &channels,
                                                               // I want 1 channel in the output
                                                               1);
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
            image_buf = stbi_load_from_file(fp, &width, &height, &channels,
                                            // I want 1 channel in the output
                                            1);
            if(image_buf == NULL)
            {
                MSG("Couldn't load image: stbi_load_from_file(\"%s\", desired_channels=1) failed", filename);
                goto done;
            }
        }
    }
    else if(*bits_per_pixel == 16)
    {
        if(bits_per_pixel_detected != 16)
        {
            MSG("Loading %d-bit image as a 16-bit image not supported",
                bits_per_pixel_detected);
            goto done;
        }

        image_buf = (unsigned char*)stbi_load_from_file_16(fp, &width, &height, &channels,
                                                           // I want 1 channels in the output
                                                           1);
        if(image_buf == NULL)
        {
            MSG("Couldn't load image: stbi_load_from_file_16(\"%s\", desired_channels=1) failed", filename);
            goto done;
        }
    }
    else if(*bits_per_pixel == 24)
    {
        if(bits_per_pixel_detected == 16)
        {
            MSG("Loading %d-bit image as a 24-bit image not supported",
                bits_per_pixel_detected);
            goto done;
        }

        image_buf = stbi_load_from_file(fp, &width, &height, &channels,
                                        // I want 3 channels in the output
                                        3);
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
                              mrcal_image_void_t* image,
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
