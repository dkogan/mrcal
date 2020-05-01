#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mrcal.h"

#include "test-harness.h"

static
bool modelHasCore_fxfycxcy( const lensmodel_t m )
{
    mrcal_lensmodel_meta_t meta = mrcal_lensmodel_meta(m);
    return meta.has_core;
}

static bool eq_lensmodel(const lensmodel_t* a, const lensmodel_t* b)
{
    // for now this is a binary comparison
    return 0 == memcmp(a, b, sizeof(*a));
}

static void _confirm_lensmodel(lensmodel_t x, lensmodel_t xref,
                               const char* what_x, const char* what_xref, int where)
{
    Ntests++;

    if(eq_lensmodel(&x, &xref))
        printf(GREEN "OK: equal %s and %s" COLOR_RESET"\n",
               what_x, what_xref);
    else
    {
        printf(RED "FAIL on line %d: NOT equal %s and %s" COLOR_RESET"\n",
               where, what_x, what_xref);
        NtestsFailed++;
    }
}
#define confirm_lensmodel(x,xref) _confirm_lensmodel(x, xref, #x, #xref, __LINE__)

static void _confirm_lensmodel_name(const char* x, const char* xref, int where)
{
    Ntests++;

    if(0 == strcmp(x,xref))
        printf(GREEN "OK: equal %s and %s" COLOR_RESET"\n",
               x, xref);
    else
    {
        printf(RED "FAIL on line %d: NOT equal %s and %s" COLOR_RESET"\n",
               where, x, xref);
        NtestsFailed++;
    }
}
#define confirm_lensmodel_name(x,xref) _confirm_lensmodel_name(x, xref, __LINE__)



int main(int argc, char* argv[])
{
    lensmodel_t ref;


    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVOR"),
                       (lensmodel_t){.type = LENSMODEL_CAHVOR} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVORE"),
                       (lensmodel_t){.type = LENSMODEL_CAHVORE} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC"),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20_200_"),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20"),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20__"),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20_200 "),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC__3_30_20_200"),
                       (lensmodel_t){.type = LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC3_30_20_200"),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    ref =
        (lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC,
        .LENSMODEL_SPLINED_STEREOGRAPHIC__config =
        { .spline_order = 3,
          .Nx           = 30,
          .Ny           = 20,
          .fov_x_deg    = 200 }};
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20_200"),
                       ref );

    confirm_lensmodel_name( mrcal_lensmodel_name((lensmodel_t){.type = LENSMODEL_CAHVOR}),
                            "LENSMODEL_CAHVOR" );
    confirm_lensmodel_name( mrcal_lensmodel_name((lensmodel_t){.type = LENSMODEL_CAHVORE}),
                            "LENSMODEL_CAHVORE" );

    char buf[1024];
    char buf_small[2];
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), (lensmodel_t){.type = LENSMODEL_CAHVOR}));
    confirm_lensmodel_name( buf, "LENSMODEL_CAHVOR" );
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), (lensmodel_t){.type = LENSMODEL_CAHVORE}));
    confirm_lensmodel_name( buf, "LENSMODEL_CAHVORE" );
    confirm(!mrcal_lensmodel_name_full(buf_small, sizeof(buf_small), (lensmodel_t){.type = LENSMODEL_CAHVOR}));
    confirm(!mrcal_lensmodel_name_full(buf_small, sizeof(buf_small), (lensmodel_t){.type = LENSMODEL_CAHVORE}));

    confirm_lensmodel_name( mrcal_lensmodel_name((lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC}), "LENSMODEL_SPLINED_STEREOGRAPHIC_..." );
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), (lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC}));
    confirm_lensmodel_name( buf, "LENSMODEL_SPLINED_STEREOGRAPHIC_0_0_0_0" );

    ref =
        (lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC,
        .LENSMODEL_SPLINED_STEREOGRAPHIC__config =
        { .spline_order = 3,
          .Nx           = 30,
          .Ny           = 20,
          .fov_x_deg    = 200 }};
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), ref));
    confirm_lensmodel_name( buf, "LENSMODEL_SPLINED_STEREOGRAPHIC_3_30_20_200" );


    confirm( modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_CAHVOR}));
    confirm( modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_OPENCV8}));
    confirm( modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC}));


    confirm_eq_int(mrcal_getNlensParams((lensmodel_t){.type = LENSMODEL_CAHVOR}),  9);
    confirm_eq_int(mrcal_getNlensParams((lensmodel_t){.type = LENSMODEL_OPENCV8}), 12);

    #warning add test for spline parameter counts
    // ref =
    //     (lensmodel_t){.type = LENSMODEL_SPLINED_STEREOGRAPHIC,
    //     .LENSMODEL_SPLINED_STEREOGRAPHIC__config = {.a = 1, .b = 2}};
    // confirm_eq_int(mrcal_getNlensParams(ref), 3);


    TEST_FOOTER();
}
