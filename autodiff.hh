#pragma once

/*
  Automatic differentiation routines. Used in poseutils-uses-autodiff.cc. See
  that file for usage examples
 */

#include <math.h>
#include <string.h>

template<int NGRAD, int NVEC> struct vec_withgrad_t;


template<int NGRAD>
struct val_withgrad_t
{
    double x;
    double j[NGRAD];

    val_withgrad_t() {}
    val_withgrad_t(double _x) : x(_x)
    {
        for(int i=0; i<NGRAD; i++) j[i] = 0.0;
    }

    val_withgrad_t<NGRAD> operator+( const val_withgrad_t<NGRAD>& b ) const
    {
        val_withgrad_t<NGRAD> y = *this;
        y.x += b.x;
        for(int i=0; i<NGRAD; i++)
            y.j[i] += b.j[i];
        return y;
    }
    val_withgrad_t<NGRAD> operator+( double b ) const
    {
        val_withgrad_t<NGRAD> y = *this;
        y.x += b;
        return y;
    }
    val_withgrad_t<NGRAD> operator-( const val_withgrad_t<NGRAD>& b ) const
    {
        val_withgrad_t<NGRAD> y = *this;
        y.x -= b.x;
        for(int i=0; i<NGRAD; i++)
            y.j[i] -= b.j[i];
        return y;
    }
    val_withgrad_t<NGRAD> operator-( double b ) const
    {
        val_withgrad_t<NGRAD> y = *this;
        y.x -= b;
        return y;
    }
    val_withgrad_t<NGRAD> operator*( const val_withgrad_t<NGRAD>& b ) const
    {
        val_withgrad_t<NGRAD> y;
        y.x = x * b.x;
        for(int i=0; i<NGRAD; i++)
            y.j[i] = j[i]*b.x + x*b.j[i];
        return y;
    }
    val_withgrad_t<NGRAD> operator*( double b ) const
    {
        val_withgrad_t<NGRAD> y;
        y.x = x * b;
        for(int i=0; i<NGRAD; i++)
            y.j[i] = j[i]*b;
        return y;
    }
    val_withgrad_t<NGRAD> operator/( const val_withgrad_t<NGRAD>& b ) const
    {
        val_withgrad_t<NGRAD> y;
        y.x = x / b.x;
        for(int i=0; i<NGRAD; i++)
            y.j[i] = (j[i]*b.x - x*b.j[i]) / (b.x*b.x);
        return y;
    }
    val_withgrad_t<NGRAD> operator/( double b ) const
    {
        return (*this) * (1./b);
    }

    val_withgrad_t<NGRAD> sqrt(void) const
    {
        val_withgrad_t<NGRAD> y;
        y.x = ::sqrt(x);
        for(int i=0; i<NGRAD; i++)
            y.j[i] = j[i] / (2. * y.x);
        return y;
    }

    val_withgrad_t<NGRAD> square(void) const
    {
        val_withgrad_t<NGRAD> s;
        s.x = x*x;
        for(int i=0; i<NGRAD; i++)
            s.j[i] = 2. * x * j[i];
        return s;
    }

    val_withgrad_t<NGRAD> sin(void) const
    {
        double s, c;
        ::sincos(x, &s, &c);
        val_withgrad_t<NGRAD> y;
        y.x = s;
        for(int i=0; i<NGRAD; i++)
            y.j[i] = c*j[i];
        return y;
    }

    val_withgrad_t<NGRAD> cos(void) const
    {
        double s, c;
        ::sincos(x, &s, &c);
        val_withgrad_t<NGRAD> y;
        y.x = c;
        for(int i=0; i<NGRAD; i++)
            y.j[i] = -s*j[i];
        return y;
    }

    vec_withgrad_t<NGRAD, 2> sincos(void) const
    {
        double s, c;
        ::sincos(x, &s, &c);
        vec_withgrad_t<NGRAD, 2> sc;
        sc.v[0].x = s;
        sc.v[1].x = c;
        for(int i=0; i<NGRAD; i++)
        {
            sc.v[0].j[i] =  c*j[i];
            sc.v[1].j[i] = -s*j[i];
        }
        return sc;
    }

    val_withgrad_t<NGRAD> acos(void) const
    {
        val_withgrad_t<NGRAD> th;
        th.x = ::acos(x);
        double dacos_dx = -1. / ::sqrt( 1. - x*x );
        for(int i=0; i<NGRAD; i++)
            th.j[i] = dacos_dx * j[i];
        return th;
    }
};


template<int NGRAD, int NVEC>
struct vec_withgrad_t
{
    val_withgrad_t<NGRAD> v[NVEC];

    vec_withgrad_t() {}
    vec_withgrad_t(const double* x, int i_gradvec0 = -1)
    {
        // x[] is a vector of length NVEC. It represents consecutive gradient
        // variables starting at i_gradvec0. It's very possible that NGRAD >
        // NVEC. Initially the subset of the gradient array corresponding to
        // variables i_gradvec0..i_gradvec0+NVEC-1 is an identity, with the rest
        // being 0
        memset(v, 0, sizeof(v));
        for(int i=0; i<NVEC; i++)
        {
            v[i].x = x[i];
            if(i_gradvec0 >= 0) v[i].j[i_gradvec0+i] = 1.0;
        }
    }

    void extract_value(double* out) const
    {
        for(int i=0; i<NVEC; i++)
            out[i] = v[i].x;
    }
    void extract_grad(double* J, int i_gradvec0, int N_gradout) const
    {
        for(int i=0; i<NVEC; i++)
            for(int j=0; j<N_gradout; j++)
                J[N_gradout*i + j] = v[i].j[i_gradvec0+j];
    }
};
