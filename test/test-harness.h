#pragma once

// Apparently I need this in MSVC to get constants
#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#define GREEN       "\x1b[32m"
#define RED         "\x1b[31m"
#define COLOR_RESET "\x1b[0m"


// global variable for the results tracking
int Ntests       = 0;
int NtestsFailed = 0;

__attribute__((unused))
static bool _confirm(bool x, const char* what_x, int where)
{
    Ntests++;

    if(x)
    {
        printf(GREEN "OK: %s is true" COLOR_RESET"\n",
               what_x);
        return true;
    }
    else
    {
        printf(RED "FAIL on line %d: %s is false" COLOR_RESET"\n", where, what_x);
        NtestsFailed++;
        return false;
    }
}
#define confirm(x) _confirm(x, #x, __LINE__)

__attribute__((unused))
static bool _confirm_eq_int(int x, int xref, const char* what_x, const char* what_xref, int where)
{
    Ntests++;
    if(x == xref)
    {
        printf(GREEN "OK: %s == %s, as it should" COLOR_RESET"\n", what_x, what_xref);
        return true;
    }
    else
    {
        printf(RED "FAIL on line %d: %s != %s: %d != %d" COLOR_RESET"\n",
               where, what_x, what_xref, x, xref);
        NtestsFailed++;
        return false;
    }
}
#define confirm_eq_int(x,xref) _confirm_eq_int(x, xref, #x, #xref, __LINE__)

__attribute__((unused))
static bool _confirm_eq_int_max_array(const int* x, const int* xref, int N, const char* what_x, const char* what_xref, int where)
{
    Ntests++;

    for(int i=0; i<N; i++)
        if(x[i] != xref[i])
        {
            printf(RED "FAIL on line %d: %s != %s" COLOR_RESET"\n",
                   where, what_x, what_xref);
            NtestsFailed++;
            return false;
        }

    printf(GREEN "OK: %s = %s, as it should" COLOR_RESET"\n", what_x, what_xref);
    return true;
}
#define confirm_eq_int_max_array(x,xref, N)                      \
    _confirm_eq_int_max_array(x, xref, N, #x, #xref, __LINE__)

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
#define confirm_eq_double(x,xref, eps) \
    _confirm_eq_double(x, xref, #x, #xref, eps, __LINE__)

__attribute__((unused))
static bool _confirm_eq_double_max_array(const double* x, const double* xref, int N, const char* what_x, const char* what_xref, double eps, int where)
{
    double worst_err = 0.0;
    for(int i=0; i<N; i++)
    {
        double err = fabs(x[i] - xref[i]);
        if(err > worst_err) worst_err = err;
    }
    Ntests++;
    if(worst_err < eps)
    {
        printf(GREEN "OK: %s ~ %s, as it should" COLOR_RESET"\n", what_x, what_xref);
        return true;
    }
    else
    {
        printf(RED "FAIL on line %d: %s !~ %s" COLOR_RESET"\n",
               where, what_x, what_xref);
        NtestsFailed++;
        return false;
    }
}
#define confirm_eq_double_max_array(x,xref, N, eps)                      \
    _confirm_eq_double_max_array(x, xref, N, #x, #xref, eps, __LINE__)




#define TEST_FOOTER()                                                   \
    if(NtestsFailed == 0)                                               \
    {                                                                   \
        printf(GREEN "%s: all %d tests passed" COLOR_RESET "\n", argv[0], Ntests);     \
        return 0;                                                       \
    }                                                                   \
    else                                                                \
    {                                                                   \
        printf(RED "%s: %d/%d tests failed" COLOR_RESET "\n", argv[0], NtestsFailed, Ntests); \
        return 1;                                                       \
    }
