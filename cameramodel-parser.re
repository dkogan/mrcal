/* -*- c -*- */

// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0


// Apparently I need this in MSVC to get constants
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include "mrcal.h"
#include "util.h"

#define DEBUG 0

// string defined by an explicit length. Instead of being 0-terminated
typedef struct
{
    const char* s;
    int len;
} string_t;


/*!re2c
  re2c:define:YYCTYPE = char;

  // No filling. All the data is available at the start
  re2c:yyfill:enable = 0;

  // I use @ tags
  re2c:flags:tags = 1;

  re2c:sentinel = 0;

  SPACE        = [ \t\n\r]*;
  IGNORE       = (SPACE | "#" [^\x00\n]* "\n")*;
  // This FLOAT definition will erroneously accept "." and "" as a valid float,
  // but the strtod converter will then reject it
  FLOAT        = "-"?[0-9]*("."[0-9]*)?([eE][+-]?[0-9]+)?;
  UNSIGNED_INT = "0"|[1-9][0-9]*;
*/

/*!stags:re2c format = 'static const char *@@;'; */

static bool string_is(const char* ref, string_t key)
{
    int ref_strlen = strlen(ref);
    return
        ref_strlen == key.len &&
        0 == strncmp(key.s, ref, ref_strlen);
}

static bool read_string( // output stored here. If NULL, we try to read off
                         // a string, but do not store it; success in the
                              // return value, as usual
                         string_t* out,
                         const char** pYYCURSOR,
                         const char* start_file,
                         const char* what)
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    while(true)
    {
        const char* s;
        const char* e;
        /*!re2c
          IGNORE "b"? ["'] @s [^"'\x00]* @e ["']
          {
            if(out != NULL)
            {
              out->s   = s;
              out->len = (int)(e-s);
            }
            *pYYCURSOR = YYCURSOR;
            return true;
          }
          * { break; }
        */
    }
    if(out != NULL)
        MSG("Didn't see the %s string at position %ld. Giving up.",
            what, (long int)(*pYYCURSOR - start_file));
    return false;
}

static bool read_value( const char** pYYCURSOR,
                        const char* start_file)
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    const char* s, *e;
    while(true)
    {
        /*!re2c
          IGNORE @s FLOAT @e
          {
            // FLOAT regex erroneously matches empty strings and ".". I
            // explicitly check for those, and return a failure
            if( e == s )
              return false;
            if( e == &s[1] && *s == '.')
              return false;
            break;
          }
          *
          {
            return false;
          }
        */
    }
    *pYYCURSOR = YYCURSOR;
    return true;
}

typedef bool (ingest_generic_consume_ignorable_t)(void* out0, int i,
                                                  const char** pYYCURSOR,
                                                  const char* start_file,
                                                  const char* what);

static bool ingest_double_consume_ignorable(void* out0, int i,
                                            const char** pYYCURSOR,
                                            const char* start_file,
                                            const char* what)
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    const char* s;
    const char* e;

    while(true)
    {
      /*!re2c
      IGNORE @s FLOAT @e IGNORE
      {
        break;
      }
      *
      {
        MSG("Error parsing double-precision value for %s at %ld",
            what, (long int)(*pYYCURSOR-start_file));
        return false;
      }
      */
    }

    if(out0 != NULL)
    {
        int N = e-s;
        char tok[N+1];
        memcpy(tok, s, N);
        tok[N] = '\0';
        char* endptr = NULL;
        ((double*)out0)[i] = strtod(tok, &endptr);
        if( N == 0 || endptr == NULL || endptr != &tok[N] ||
            !isfinite(((double*)out0)[i]))
        {
            MSG("Error parsing double-precision value for %s at %ld. String: '%s'",
                what, (long int)(*pYYCURSOR-start_file), tok);
            return false;
        }
    }

    *pYYCURSOR = YYCURSOR;
    return true;
}

static bool ingest_uint_consume_ignorable(void* out0, int i,
                                          const char** pYYCURSOR,
                                          const char* start_file,
                                          const char* what)
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    const char* s;
    const char* e;

    while(true)
    {
      /*!re2c
      IGNORE @s UNSIGNED_INT @e IGNORE
      {
        break;
      }
      *
      {
        MSG("Error parsing unsigned integer for %s at %ld",
            what, (long int)(*pYYCURSOR-start_file));
        return false;
      }
      */
    }

    if(out0 != NULL)
    {
        int N = e-s;
        char tok[N+1];
        memcpy(tok, s, N);
        tok[N] = '\0';
        int si = atoi(tok);
        if( N == 0 || si < 0 )
        {
            MSG("Error parsing unsigned int for %s at %ld. String: '%s'",
                what, (long int)(*pYYCURSOR-start_file), tok);
            return false;
        }
        ((unsigned int*)out0)[i] = (unsigned int)si;
    }

    *pYYCURSOR = YYCURSOR;
    return true;
}

static bool read_list_values_generic( // output stored here. If NULL, we try to
                                      // read off the values, but do not store
                                      // them; success in the return value, as
                                      // usual
                                      void* out,
                                      ingest_generic_consume_ignorable_t* f,
                                      const char** pYYCURSOR, const char* start_file,
                                      const char* what,
                                      int Nvalues)
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    while(true)
    {
        /*!re2c
          IGNORE [[(] { break; }
          *
          {
            MSG("Didn't see the opening [/( for the %s at position %ld. Giving up.",
                what, (long int)(YYCURSOR - start_file));
            return false;
          }
        */
    }

    int i;
    for(i=0; i<Nvalues-1; i++)
    {
        if(!(*f)(out, i, &YYCURSOR, start_file, what))
            return false;
        if(*YYCURSOR == ',')
            YYCURSOR++;
        else
        {
            MSG("Didn't see expected ',' at %ld while parsing %s",
                (long int)(YYCURSOR-start_file), what);
            return false;
        }
    }

    // one more, but the trailing , is optional
    {
        if(!(*f)(out, i, &YYCURSOR, start_file, what))
            return false;
        if(*YYCURSOR == ',')
            YYCURSOR++;
    }

    while(true)
    {
        /*!re2c
          IGNORE [\])] { break; }
          *
          {
            MSG("Didn't see the closing )/] for the %s at position %ld. Expected %d values, but the given list has more. Giving up.",
                what, (long int)(YYCURSOR - start_file), Nvalues);
            return false;
          }
        */
    }

    *pYYCURSOR = YYCURSOR;
    return true;
}

static bool read_balanced_list( const char** pYYCURSOR, const char* start_file )
{
    const char* YYMARKER;
    const char* YYCURSOR = *pYYCURSOR;

    int level = 0;
    while(true)
    {
        /*!re2c
          ( IGNORE | [0-9eE.,-]*)* [[(]
          {
            level++;
            continue;
          }
          ( IGNORE | [0-9eE.,-]*)* [\])] IGNORE ","?
          {
            level--;
            if(level < 0)
            {
              MSG("Error reading a balanced list at %ld (list not balanced)",
                  (long int)(YYCURSOR-start_file));
              return false;
            }
            if(level==0)
            {
              // closed last ]. Leave trailing , unprocessed. Caller will deal
              // with it
              if(YYCURSOR[-1] == ',')
                YYCURSOR--;

              *pYYCURSOR = YYCURSOR;
              return true;
            }
            if(level > 0)
            {
              // closed inner ]. Trailing , afterwards is optional. I don't
              // bother checking for this thoroughly, so I end up erroneously
              // accepting expressions like [1,2,3][3,4,5]. But that's OK
              ;
            }
            continue;
          }
          *
          {
            MSG("Error reading a balanced list at %ld (unexpected value)",
                (long int)(YYCURSOR-start_file));
            return false;
          }
        */
    }

    MSG("Getting here is a bug");
    return false;
}

// if len>0, the string doesn't need to be 0-terminated. If len<=0, the end of
// the buffer IS indicated by a 0 byte
mrcal_cameramodel_VOID_t* mrcal_read_cameramodel_string(const char *string,
                                                        const int len)
{
    // This is lame. If the end of the buffer is indicated by the buffer length
    // only, I allocate a new padded buffer, and copy into it. Then this code
    // looks for a terminated 0 always. I should instead use the re2c logic for
    // fixed-length buffers (YYLIMIT), but that looks complicated
    const char* YYCURSOR     = NULL;
    char*       malloced_buf = NULL;
    if(len > 0)
    {
        malloced_buf = malloc(len+1);
        if(malloced_buf == NULL)
        {
            MSG("malloc() failed");
            return NULL;
        }
        memcpy(malloced_buf, string, len);
        malloced_buf[len] = '\0';
        YYCURSOR = malloced_buf;
    }
    else
        YYCURSOR = string;


    // Set the output structure to invalid values that I can check later
    mrcal_cameramodel_VOID_t cameramodel_core =
        {.rt_cam_ref[0]   = DBL_MAX,
         .imagersize      = {},
         .lensmodel.type  = MRCAL_LENSMODEL_INVALID };
    mrcal_cameramodel_VOID_t* cameramodel_full = NULL;
    bool finished = false;

    const char* YYMARKER;
    const char* start_file = YYCURSOR;

    ///////// leading {
    while(true)
    {
        /*!re2c
          IGNORE "{" IGNORE { break; }
          *
          {
            MSG("Didn't see leading '{'. Giving up.");
            goto done;
          }
        */
    }

    ///////// key: value
    while(true)
    {
        string_t key = {};

        ///////// key:
        if(!read_string(&key, &YYCURSOR, start_file, "key"))
            goto done;
        while(true)
        {
            /*!re2c
              IGNORE ":" { break; }
              *
              {
                MSG("Didn't see expected ':' at %ld. Giving up.",
                    (long int)(YYCURSOR-start_file));
                goto done;
              }
            */
        }

#if defined DEBUG && DEBUG
        MSG("key: '%.*s'", key.len, key.s);
#endif

        if( string_is("lensmodel", key) )
        {
            if(cameramodel_core.lensmodel.type >= 0)
            {
                MSG("lensmodel defined more than once");
                goto done;
            }

            // "lensmodel" has string values
            string_t lensmodel;
            if(!read_string(&lensmodel,
                            &YYCURSOR, start_file, "lensmodel"))
                goto done;

            char lensmodel_string[lensmodel.len+1];
            memcpy(lensmodel_string, lensmodel.s, lensmodel.len);
            lensmodel_string[lensmodel.len] = '\0';

            if( !mrcal_lensmodel_from_name(&cameramodel_core.lensmodel, lensmodel_string) )
            {
                MSG("Could not parse lensmodel '%s'", lensmodel_string);
                goto done;
            }
        }
        else if(string_is("intrinsics", key))
        {
            if( cameramodel_full != NULL )
            {
                MSG("intrinsics defined more than once");
                goto done;
            }

            if(cameramodel_core.lensmodel.type < 0)
            {
                MSG("Saw 'intrinsics' key, before a 'lensmodel' key. Make sure that a 'lensmodel' key exists, and that it appears in the file before the 'intrinsics'");
                goto done;
            }

            int Nparams = mrcal_lensmodel_num_params(&cameramodel_core.lensmodel);
            cameramodel_full = malloc(sizeof(mrcal_cameramodel_VOID_t) +
                                      Nparams*sizeof(double));
            if(NULL == cameramodel_full)
            {
                MSG("malloc() failed");
                goto done;
            }
            if( !read_list_values_generic(cameramodel_full->intrinsics,
                                          ingest_double_consume_ignorable,
                                          &YYCURSOR, start_file,
                                          "intrinsics", Nparams) )
                goto done;
        }
        else if(string_is("extrinsics", key))
        {
            if(cameramodel_core.rt_cam_ref[0] != DBL_MAX)
            {
                MSG("extrinsics defined more than once");
                goto done;
            }
            if( !read_list_values_generic(cameramodel_core.rt_cam_ref,
                                          ingest_double_consume_ignorable,
                                          &YYCURSOR, start_file, "extrinsics", 6) )
                goto done;
        }
        else if(string_is("imagersize", key))
        {
            if(cameramodel_core.imagersize[0] > 0)
            {
                MSG("imagersize defined more than once");
                goto done;
            }
            if( !read_list_values_generic(cameramodel_core.imagersize,
                                          ingest_uint_consume_ignorable,
                                          &YYCURSOR, start_file, "imagersize", 2) )
                goto done;
        }
        else
        {
            // Some unknown key. Read off the data and continue
            // try to read a string...
            if(!read_value(&YYCURSOR, start_file) &&
               !read_string(NULL, &YYCURSOR, start_file, "unknown") &&
               !read_balanced_list(&YYCURSOR, start_file))
            {
                MSG("Error parsing value for key '%.*s' at %ld",
                    key.len, key.s,
                    (long int)(YYCURSOR-start_file));
                goto done;
            }
        }


        // Done with key:value. Look for optional trailing , and/or a }. We must
        // see at least some of those things. k:v k:v without a , in-between is
        // illegal
        bool closing_brace = false;
        const char* f;
        while(true)
        {
            /*!re2c
            IGNORE ("," | "}" | ("," IGNORE "}")) @f
            {
                if(f[-1] == '}') closing_brace = true;
                break;
            }
            *
            {
                MSG("Didn't see trailing , at %ld",
                    (long int)(YYCURSOR-start_file));
                goto done;
            }
            */
        }
        if(closing_brace)
        {
            while(true)
            {
              /*!re2c
              IGNORE [\x00]
              {
                finished = true;
                goto done;
              }
              *
              {
                  MSG("Garbage after trailing } at %ld. Giving up",
                      (long int)(f-1 - start_file));
                  goto done;
              }
              */
            }
        }
    }

 done:

    if(malloced_buf)
        free(malloced_buf);

    if(!finished)
    {
        free(cameramodel_full);
        return NULL;
    }

    // Done parsing everything! Validate and finalize
    if(!(cameramodel_core.lensmodel.type >= 0 &&
         cameramodel_full != NULL &&
         cameramodel_core.rt_cam_ref[0] != DBL_MAX &&
         cameramodel_core.imagersize[0] > 0))
    {
        MSG("Incomplete cameramodel. Need keys: lensmodel, intrinsics, extrinsics, imagersize");
        free(cameramodel_full);
        return NULL;
    }

    memcpy(cameramodel_full, &cameramodel_core, sizeof(cameramodel_core));
    return cameramodel_full;
}

mrcal_cameramodel_VOID_t* mrcal_read_cameramodel_file(const char* filename)
{
    int   fd     = -1;
    char* string = NULL;
    mrcal_cameramodel_VOID_t* result = NULL;

    fd = open(filename, O_RDONLY);
    if(fd < 0)
    {
        MSG("Couldn't open(\"%s\")", filename);
        goto done;
    }

    struct stat st;
    if(0 != fstat(fd, &st))
    {
        MSG("Couldn't stat(\"%s\")", filename);
        goto done;
    }

    // I mmap twice:
    //
    // 1. anonymous mapping slightly larger than the file size. These are all 0
    // 2. mmap of the file. The trailing 0 are preserved, and the parser can use
    //    the trailing 0 to indicate the end of file
    //
    // This is only needed if the file size is exactly a multiple of the page
    // size. If it isn't, then the remains of the last page are 0 anyway.
    string = mmap(NULL,
                  st.st_size + 1, // one extra byte
                  PROT_READ,
                  MAP_ANONYMOUS | MAP_PRIVATE,
                  -1, 0);
    if(string == MAP_FAILED)
    {
        MSG("Couldn't mmap(anonymous) right before mmap(\"%s\")", filename);
        goto done;
    }

    string = mmap(string, st.st_size,
                  PROT_READ,
                  MAP_FIXED | MAP_PRIVATE,
                  fd, 0);
    if(string == MAP_FAILED)
    {
        MSG("Couldn't mmap(\"%s\")", filename);
        goto done;
    }

    result = mrcal_read_cameramodel_string(string,
                                           // 0 indicates EOF, not the file size
                                           0);

 done:
    if(string != NULL && string != MAP_FAILED)
        munmap(string, st.st_size+1);
    if(fd >= 0)
        close(fd);

    return result;
}

void mrcal_free_cameramodel(mrcal_cameramodel_VOID_t** cameramodel)
{
    free(*cameramodel);
    *cameramodel = NULL;
}
