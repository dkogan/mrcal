#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mrcal.h"

#define GREEN       "\x1b[32m"
#define RED         "\x1b[31m"
#define COLOR_RESET "\x1b[0m"


static bool eq_lensmodel(const lensmodel_t* a, const lensmodel_t* b)
{
    // for now this is a binary comparison
    return 0 == memcmp(a, b, sizeof(*a));
}

#define confirm(x) do {                                                 \
    Ntests++;                                                           \
                                                                        \
    if(x)                                                               \
        printf(GREEN "OK: "#x" is true" COLOR_RESET"\n");       \
    else                                                                \
        {                                                               \
            printf(RED "FAIL on line %d: "#x" is false" COLOR_RESET"\n", __LINE__); \
            NtestsFailed++;                                             \
        }                                                               \
} while(0)

#define confirm_eq_int(x,xref) do {                                     \
    Ntests++;                                                           \
                                                                        \
    int _x    = x;                                                      \
    int _xref = xref;                                                   \
    if(_x == xref)                                                      \
        printf(GREEN "OK: "#x" == %d, as it should" COLOR_RESET"\n", _xref); \
    else                                                                \
    {                                                                   \
        printf(RED "FAIL on line %d: "#x" != %d. Instead it is %d" COLOR_RESET"\n", __LINE__, _xref, _x); \
        NtestsFailed++;                                                 \
    }                                                                   \
} while(0)

#define confirm_lensmodel(x, xref) do {                                 \
    Ntests++;                                                           \
                                                                        \
    lensmodel_t a = x;                                                  \
    lensmodel_t b = xref;                                               \
    if(eq_lensmodel(&a, &b))                                            \
        printf(GREEN "OK: equal "#x" and "#xref COLOR_RESET"\n");       \
    else                                                                \
        {                                                               \
            printf(RED "FAIL on line %d: NOT equal "#x" and "#xref COLOR_RESET"\n", __LINE__); \
            NtestsFailed++;                                             \
        }                                                               \
} while(0)

#define confirm_lensmodel_name(x, xref) do {                            \
    Ntests++;                                                           \
                                                                        \
    if(0 == strcmp(x,xref))                                             \
        printf(GREEN "OK: equal "#x" and "#xref COLOR_RESET"\n");       \
    else                                                                \
        {                                                               \
            printf(RED "FAIL on line %d: NOT equal "#x" and "#xref COLOR_RESET"\n", __LINE__); \
            NtestsFailed++;                                             \
        }                                                               \
} while(0)

int main(int argc, char* argv[])
{
    int Ntests       = 0;
    int NtestsFailed = 0;

    lensmodel_t ref;


    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVOR"),
                       (lensmodel_t){.type = LENSMODEL_CAHVOR} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_CAHVORE"),
                       (lensmodel_t){.type = LENSMODEL_CAHVORE} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV"),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV_1_2_"),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV_1_2 "),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV__1_2"),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV1_2"),
                       (lensmodel_t){.type = LENSMODEL_INVALID} );
    ref =
        (lensmodel_t){.type = LENSMODEL_UV,
        .LENSMODEL_UV__config = {.a = 1, .b = 2}};
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV_1_2"),
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

    confirm_lensmodel_name( mrcal_lensmodel_name((lensmodel_t){.type = LENSMODEL_UV}), "LENSMODEL_UV_..." );
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), (lensmodel_t){.type = LENSMODEL_UV}));
    confirm_lensmodel_name( buf, "LENSMODEL_UV_0_0" );

    ref =
        (lensmodel_t){.type = LENSMODEL_UV,
        .LENSMODEL_UV__config = {.a = 1, .b = 2}};
    confirm(mrcal_lensmodel_name_full(buf, sizeof(buf), ref));
    confirm_lensmodel_name( buf, "LENSMODEL_UV_1_2" );


    confirm( mrcal_modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_CAHVOR}));
    confirm( mrcal_modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_OPENCV8}));
    confirm(!mrcal_modelHasCore_fxfycxcy((lensmodel_t){.type = LENSMODEL_UV}));


    confirm_eq_int(mrcal_getNlensParams((lensmodel_t){.type = LENSMODEL_CAHVOR}),  9);
    confirm_eq_int(mrcal_getNlensParams((lensmodel_t){.type = LENSMODEL_OPENCV8}), 12);
    ref =
        (lensmodel_t){.type = LENSMODEL_UV,
        .LENSMODEL_UV__config = {.a = 1, .b = 2}};
    confirm_eq_int(mrcal_getNlensParams(ref), 3);


    if(NtestsFailed == 0)
    {
        printf(GREEN "All %d tests passed\n", Ntests);
        return 0;
    }
    else
    {
        printf(RED "%d/%d tests failed\n", NtestsFailed, Ntests);
        return 1;
    }
}
