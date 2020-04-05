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

int main(int argc, char* argv[])
{
    int Ntests       = 0;
    int NtestsFailed = 0;

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
    lensmodel_t ref =
        (lensmodel_t){.type = LENSMODEL_UV,
        .LENSMODEL_UV__config = {.a = 1, .b = 2}};
    confirm_lensmodel( mrcal_lensmodel_from_name("LENSMODEL_UV_1_2"),
                       ref );


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
