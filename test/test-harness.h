#pragma once

#include <stdio.h>
#include <stdbool.h>

#define GREEN       "\x1b[32m"
#define RED         "\x1b[31m"
#define COLOR_RESET "\x1b[0m"


// global variable for the results tracking
int Ntests       = 0;
int NtestsFailed = 0;

__attribute__((unused))
static void _confirm(bool x, const char* what_x, int where)
{
    Ntests++;

    if(x)
        printf(GREEN "OK: %s is true" COLOR_RESET"\n",
               what_x);
    else
    {
        printf(RED "FAIL on line %d: %s is false" COLOR_RESET"\n", where, what_x);
        NtestsFailed++;
    }
}
#define confirm(x) _confirm(x, #x, __LINE__)

__attribute__((unused))
static void _confirm_eq_int(int x, int xref, const char* what_x, const char* what_xref, int where)
{
    Ntests++;
    if(x == xref)
        printf(GREEN "OK: %s == %s, as it should" COLOR_RESET"\n", what_x, what_xref);
    else
    {
        printf(RED "FAIL on line %d: %s != %s: %d != %d" COLOR_RESET"\n",
               where, what_x, what_xref, x, xref);
        NtestsFailed++;
    }
}
#define confirm_eq_int(x,xref) _confirm_eq_int(x, xref, #x, #xref, __LINE__)

__attribute__((unused))
static void _confirm_eq_double(double x, double xref, const char* what_x, const char* what_xref, double eps, int where)
{
    Ntests++;
    if(fabs(x - xref) < eps)
        printf(GREEN "OK: %s ~ %s, as it should" COLOR_RESET"\n", what_x, what_xref);
    else
    {
        printf(RED "FAIL on line %d: %s !~ %s: %f !~ %f" COLOR_RESET"\n",
               where, what_x, what_xref, x, xref);
        NtestsFailed++;
    }
}
#define confirm_eq_double(x,xref, eps) _confirm_eq_double(x, xref, #x, #xref, eps, __LINE__)




#define TEST_FOOTER()                                                   \
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
