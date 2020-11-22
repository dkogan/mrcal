#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mrcal.h"

#include "test-harness.h"

static
bool modelHasCore_fxfycxcy( const mrcal_lensmodel_t m )
{
    mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(m);
    return meta.has_core;
}

static bool eq_lensmodel(const mrcal_lensmodel_t* a, const mrcal_lensmodel_t* b)
{
    // for now this is a binary comparison
    return 0 == memcmp(a, b, sizeof(*a));
}

static void _confirm_lensmodel(mrcal_lensmodel_t x, mrcal_lensmodel_t xref,
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
    mrcal_lensmodel_t ref;


    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVOR"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVORE"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVORE} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20_fov_x_deg=200_"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20__"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20_fov_x_deg=200 "),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC__order=3_Nx=30_Ny=20_fov_x_deg=200"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHICorder=3_Nx=30_Ny=20_fov_x_deg=200"),
                       (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID} );
    ref =
        (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC,
        .LENSMODEL_SPLINED_STEREOGRAPHIC__config =
        { .order     = 3,
          .Nx        = 30,
          .Ny        = 20,
          .fov_x_deg = 200 }};
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20_fov_x_deg=200"),
                       ref );

    confirm_lensmodel_name( mrcal_lensmodel_name_unconfigured((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR}),
                            "LENSMODEL_CAHVOR" );
    confirm_lensmodel_name( mrcal_lensmodel_name_unconfigured((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVORE}),
                            "LENSMODEL_CAHVORE" );

    char buf[1024];
    char buf_small[2];
    confirm(mrcal_lensmodel_name(buf, sizeof(buf), (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR}));
    confirm_lensmodel_name( buf, "LENSMODEL_CAHVOR" );
    confirm(mrcal_lensmodel_name(buf, sizeof(buf), (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVORE}));
    confirm_lensmodel_name( buf, "LENSMODEL_CAHVORE" );
    confirm(!mrcal_lensmodel_name(buf_small, sizeof(buf_small), (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR}));
    confirm(!mrcal_lensmodel_name(buf_small, sizeof(buf_small), (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVORE}));

    confirm_lensmodel_name( mrcal_lensmodel_name_unconfigured((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC}), "LENSMODEL_SPLINED_STEREOGRAPHIC_order=..._Nx=..._Ny=..._fov_x_deg=..." );
    confirm(mrcal_lensmodel_name(buf, sizeof(buf), (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC}));
    confirm_lensmodel_name( buf, "LENSMODEL_SPLINED_STEREOGRAPHIC_order=0_Nx=0_Ny=0_fov_x_deg=0" );

    ref =
        (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC,
        .LENSMODEL_SPLINED_STEREOGRAPHIC__config =
        { .order     = 3,
          .Nx        = 30,
          .Ny        = 20,
          .fov_x_deg = 200 }};
    confirm(mrcal_lensmodel_name(buf, sizeof(buf), ref));
    confirm_lensmodel_name( buf, "LENSMODEL_SPLINED_STEREOGRAPHIC_order=3_Nx=30_Ny=20_fov_x_deg=200" );


    confirm( modelHasCore_fxfycxcy((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR}));
    confirm( modelHasCore_fxfycxcy((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_OPENCV8}));
    confirm( modelHasCore_fxfycxcy((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC}));


    confirm_eq_int(mrcal_lensmodel_num_params((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_CAHVOR}),  9);
    confirm_eq_int(mrcal_lensmodel_num_params((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_OPENCV8}), 12);

    confirm_eq_int(mrcal_lensmodel_num_params((mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC,
                                                      .LENSMODEL_SPLINED_STEREOGRAPHIC__config =
                                                      { .order     = 3,
                                                        .Nx        = 30,
                                                        .Ny        = 20,
                                                        .fov_x_deg = 200 }}),
                   4 + 30*20*2);


    TEST_FOOTER();
}
