#pragma once


#define GREEN       "\x1b[32m"
#define RED         "\x1b[31m"
#define COLOR_RESET "\x1b[0m"

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
    if(_x == _xref)                                                      \
        printf(GREEN "OK: "#x" == %d, as it should" COLOR_RESET"\n", _xref); \
    else                                                                \
    {                                                                   \
        printf(RED "FAIL on line %d: "#x" != %d. Instead it is %d" COLOR_RESET"\n", __LINE__, _xref, _x); \
        NtestsFailed++;                                                 \
    }                                                                   \
} while(0)


#define TEST_HEADER()                           \
    int Ntests       = 0;                       \
    int NtestsFailed = 0;

#define TEST_FOOTER()                                                     \
    if(NtestsFailed == 0)                                               \
    {                                                                   \
        printf(GREEN "%s: all %d tests passed\n", argv[0], Ntests);     \
        return 0;                                                       \
    }                                                                   \
    else                                                                \
    {                                                                   \
        printf(RED "%s: %d/%d tests failed\n", argv[0], NtestsFailed, Ntests); \
        return 1;                                                       \
    }
