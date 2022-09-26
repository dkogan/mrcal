#define _GNU_SOURCE

#include <stdio.h>
#include <string.h>
#include <math.h>

#include "poseutils.h"
#include "strides.h"

// All arrays stored in row-major order
//
// I have two different representations of pose transformations:
//
// - Rt is a concatenated (4,3) array: Rt = nps.glue(R,t, axis=-2). The
//   transformation is R*x+t
//
// - rt is a concatenated (6) array: rt = nps.glue(r,t, axis=-1). The
//   transformation is R*x+t where R = R_from_r(r)


// row vectors: vout = matmult(v,Mt)
// equivalent col vector expression: vout = matmult(M,v)
#define mul_vec3_gen33t_vout_scaled_full(vout, vout_stride0,            \
                                         v,    v_stride0,               \
                                         Mt,   Mt_stride0, Mt_stride1,  \
                                         scale)                         \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,0,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,0,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,0,2)*_P1(v,v_stride0,2) ),    \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,1,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,1,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,1,2)*_P1(v,v_stride0,2) ),    \
            scale *                                                     \
            (_P2(Mt,Mt_stride0,Mt_stride1,2,0)*_P1(v,v_stride0,0) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,2,1)*_P1(v,v_stride0,1) +     \
             _P2(Mt,Mt_stride0,Mt_stride1,2,2)*_P1(v,v_stride0,2) ) };  \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#define mul_vec3_gen33t_vout_full(vout, vout_stride0,                   \
                                  v,    v_stride0,                      \
                                  Mt,   Mt_stride0, Mt_stride1)         \
    mul_vec3_gen33t_vout_scaled_full(vout, vout_stride0,                \
                                     v,    v_stride0,                   \
                                     Mt,   Mt_stride0, Mt_stride1, 1.0)
// row vectors: vout = scale*matmult(v,M)
#define mul_vec3_gen33_vout_scaled_full(vout, vout_stride0,             \
                                        v,    v_stride0,                \
                                        M,    M_stride0, M_stride1,     \
                                        scale)                          \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,0)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,0)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,0)*_P1(v,v_stride0,2)),        \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,1)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,1)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,1)*_P1(v,v_stride0,2)),        \
            scale *                                                     \
            (_P2(M,M_stride0,M_stride1,0,2)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,2)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,2)*_P1(v,v_stride0,2)) };      \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)
#define mul_vec3_gen33_vout_full(vout, vout_stride0,                   \
                                 v,    v_stride0,                       \
                                 Mt,   Mt_stride0, Mt_stride1)          \
    mul_vec3_gen33_vout_scaled_full(vout, vout_stride0,                 \
                                    v,    v_stride0,                    \
                                    Mt,   Mt_stride0, Mt_stride1, 1.0)

// row vectors: vout = matmult(v,Mt)
// equivalent col vector expression: vout = matmult(M,v)
#define mul_vec3_gen33t_vaccum_full(vout, vout_stride0,                 \
                                    v,    v_stride0,                    \
                                    Mt,   Mt_stride0, Mt_stride1)       \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            _P1(vout,vout_stride0,0) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,0,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,0,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,0,2)*_P1(v,v_stride0,2),       \
            _P1(vout,vout_stride0,1) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,1,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,1,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,1,2)*_P1(v,v_stride0,2),       \
            _P1(vout,vout_stride0,2) +                                  \
            _P2(Mt,Mt_stride0,Mt_stride1,2,0)*_P1(v,v_stride0,0) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,2,1)*_P1(v,v_stride0,1) +      \
            _P2(Mt,Mt_stride0,Mt_stride1,2,2)*_P1(v,v_stride0,2) };     \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)

// row vectors: vout = scale*matmult(v,M)
#define mul_vec3_gen33_vaccum_scaled_full(vout, vout_stride0,           \
                                          v,    v_stride0,              \
                                          M,    M_stride0, M_stride1,   \
                                          scale)                        \
    do {                                                                \
        /* needed for in-place operations */                            \
        double outcopy[3] = {                                           \
            _P1(vout,vout_stride0,0) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,0)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,0)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,0)*_P1(v,v_stride0,2)),        \
            _P1(vout,vout_stride0,1) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,1)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,1)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,1)*_P1(v,v_stride0,2)),        \
            _P1(vout,vout_stride0,2) + scale *                          \
            (_P2(M,M_stride0,M_stride1,0,2)*_P1(v,v_stride0,0) +        \
             _P2(M,M_stride0,M_stride1,1,2)*_P1(v,v_stride0,1) +        \
             _P2(M,M_stride0,M_stride1,2,2)*_P1(v,v_stride0,2)) };      \
        _P1(vout,vout_stride0,0) = outcopy[0];                          \
        _P1(vout,vout_stride0,1) = outcopy[1];                          \
        _P1(vout,vout_stride0,2) = outcopy[2];                          \
    } while(0)

// multiply two (3,3) matrices
static inline
void mul_gen33_gen33_vout_full(// output
                               double* restrict m0m1,
                               int m0m1_stride0, int m0m1_stride1,

                               // input
                               const double* restrict m0,
                               int m0_stride0, int m0_stride1,
                               const double* restrict m1,
                               int m1_stride0, int m1_stride1)
{
    for(int i=0; i<3; i++)
        // one row at a time
        mul_vec3_gen33_vout_scaled_full(
                                        &_P2(m0m1,m0m1_stride0,m0m1_stride1,  i,0),   m0m1_stride1,
                                        &_P2(m0  ,  m0_stride0,  m0_stride1,  i,0),   m0_stride1,
                                        m1, m1_stride0, m1_stride1,
                                        1.0);
}
// multiply two (3,3) matrices, but accumulate the result instead of setting
static inline
void mul_gen33_gen33_vaccum_full(// output
                                 double* restrict m0m1,
                                 int m0m1_stride0, int m0m1_stride1,

                                 // input
                                 const double* restrict m0,
                                 int m0_stride0, int m0_stride1,
                                 const double* restrict m1,
                                 int m1_stride0, int m1_stride1)
{
    for(int i=0; i<3; i++)
        // one row at a time
        mul_vec3_gen33_vaccum_scaled_full(
                                          &_P2(m0m1,m0m1_stride0,m0m1_stride1,  i,0),   m0m1_stride1,
                                          &_P2(m0  ,  m0_stride0,  m0_stride1,  i,0),   m0_stride1,
                                          m1, m1_stride0, m1_stride1,
                                          1.0);
}

static inline
double inner3(const double* restrict a,
              const double* restrict b)
{
    double s = 0.0;
    for (int i=0; i<3; i++) s += a[i]*b[i];
    return s;
}




// Make an identity rotation or transformation
void mrcal_identity_R_full(double* R,      // (3,3) array
                           int R_stride0,  // in bytes. <= 0 means "contiguous"
                           int R_stride1   // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_2D(R, 3,3);
    P2(R, 0,0) = 1.0; P2(R, 0,1) = 0.0; P2(R, 0,2) = 0.0;
    P2(R, 1,0) = 0.0; P2(R, 1,1) = 1.0; P2(R, 1,2) = 0.0;
    P2(R, 2,0) = 0.0; P2(R, 2,1) = 0.0; P2(R, 2,2) = 1.0;
}
void mrcal_identity_r_full(double* r,      // (3,) array
                           int r_stride0   // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_1D(r, 3);
    P1(r, 0) = 0.0; P1(r, 1) = 0.0; P1(r, 2) = 0.0;
}
void mrcal_identity_Rt_full(double* Rt,      // (4,3) array
                            int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                            int Rt_stride1   // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_2D(Rt, 4,3);
    mrcal_identity_R_full(Rt, Rt_stride0, Rt_stride1);
    for(int i=0; i<3; i++) P2(Rt, 3, i) = 0.0;
}
void mrcal_identity_rt_full(double* rt,      // (6,) array
                            int rt_stride0   // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_1D(rt, 6);
    mrcal_identity_r_full(rt, rt_stride0);
    for(int i=0; i<3; i++) P1(rt, i+3) = 0.0;
}

void mrcal_rotate_point_R_full( // output
                               double* x_out,      // (3,) array
                               int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                               double* J_R,        // (3,3,3) array. May be NULL
                               int J_R_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_R_stride1,    // in bytes. <= 0 means "contiguous"
                               int J_R_stride2,    // in bytes. <= 0 means "contiguous"
                               double* J_x,        // (3,3) array. May be NULL
                               int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                               int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                               // input
                               const double* R,    // (3,3) array. May be NULL
                               int R_stride0,      // in bytes. <= 0 means "contiguous"
                               int R_stride1,      // in bytes. <= 0 means "contiguous"
                               const double* x_in, // (3,) array. May be NULL
                               int x_in_stride0,   // in bytes. <= 0 means "contiguous"

                               bool inverted       // if true, I apply a
                                                   // rotation in the opposite
                                                   // direction. J_R corresponds
                                                   // to the input R
                                )
{
    init_stride_1D(x_out, 3);
    init_stride_3D(J_R,   3,3,3 );
    init_stride_2D(J_x,   3,3 );
    init_stride_2D(R,     3,3 );
    init_stride_1D(x_in,  3 );

    if(inverted)
    {
        // transpose R
        int tmp;

        tmp = R_stride0;
        R_stride0 = R_stride1;
        R_stride1 = tmp;

        tmp = J_R_stride1;
        J_R_stride1 = J_R_stride2;
        J_R_stride2 = tmp;
    }

    if(J_R)
    {
        // out[i] = inner(R[i,:],in)
        for(int i=0; i<3; i++)
        {
            int j=0;
            for(; j<i; j++)
                for(int k=0; k<3; k++)
                    P3(J_R, i,j,k) = 0.0;
            for(int k=0; k<3; k++)
                P3(J_R, i,j,k) = P1(x_in, k);
            for(j++; j<3; j++)
                for(int k=0; k<3; k++)
                    P3(J_R, i,j,k) = 0.0;
        }
    }
    if(J_x)
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                P2(J_x, i,j) = P2(R, i,j);

    // R*x
    mul_vec3_gen33t_vout_full(x_out, x_out_stride0,
                              x_in,  x_in_stride0,
                              R,     R_stride0, R_stride1);
}


// mrcal_rotate_point_r() uses auto-differentiation, so it's implemented in C++
// in poseutils-uses-autodiff.cc


// Apply a transformation to a point
void mrcal_transform_point_Rt_full( // output
                                   double* x_out,      // (3,) array
                                   int x_out_stride0,  // in bytes. <= 0 means "contiguous"
                                   double* J_Rt,       // (3,4,3) array. May be NULL
                                   int J_Rt_stride0,   // in bytes. <= 0 means "contiguous"
                                   int J_Rt_stride1,   // in bytes. <= 0 means "contiguous"
                                   int J_Rt_stride2,   // in bytes. <= 0 means "contiguous"
                                   double* J_x,        // (3,3) array. May be NULL
                                   int J_x_stride0,    // in bytes. <= 0 means "contiguous"
                                   int J_x_stride1,    // in bytes. <= 0 means "contiguous"

                                   // input
                                   const double* Rt,   // (4,3) array. May be NULL
                                   int Rt_stride0,     // in bytes. <= 0 means "contiguous"
                                   int Rt_stride1,     // in bytes. <= 0 means "contiguous"
                                   const double* x_in, // (3,) array. May be NULL
                                   int x_in_stride0,   // in bytes. <= 0 means "contiguous"

                                   bool inverted       // if true, I apply a
                                                       // transformation in the opposite
                                                       // direction. J_Rt corresponds
                                                       // to the input Rt
                                    )
{
    init_stride_1D(x_out, 3);
    init_stride_3D(J_Rt,  3,4,3 );
    // init_stride_2D(J_x,   3,3 );
    init_stride_2D(Rt,    4,3 );
    // init_stride_1D(x_in,  3 );

    if(!inverted)
    {
        // for in-place operation
        double t[] = { P2(Rt,3,0), P2(Rt,3,1), P2(Rt,3,2) };

        // I want R*x + t
        // First R*x
        mrcal_rotate_point_R_full(x_out, x_out_stride0,
                                  J_Rt,  J_Rt_stride0,  J_Rt_stride1, J_Rt_stride2,
                                  J_x,   J_x_stride0,   J_x_stride1,
                                  Rt,    Rt_stride0,    Rt_stride1,
                                  x_in,  x_in_stride0,
                                  false);

        // And now +t. The J_R, J_x gradients are unaffected. J_t is identity
        for(int i=0; i<3; i++)
            P1(x_out,i) += t[i];
        if(J_Rt)
            mrcal_identity_R_full(&P3(J_Rt,0,3,0), J_Rt_stride0, J_Rt_stride2);
    }
    else
    {
        // inverted operation means
        //   y = transpose(R) (x - t)

        double x_minus_t[] = { P1(x_in,0) - P2(Rt,3,0),
                               P1(x_in,1) - P2(Rt,3,1),
                               P1(x_in,2) - P2(Rt,3,2)};

        // Compute. After this:
        //   x_out is done
        //   J_R is done
        //   J_x is done
        mrcal_rotate_point_R_full(x_out, x_out_stride0,
                                  J_Rt,  J_Rt_stride0,  J_Rt_stride1, J_Rt_stride2,
                                  J_x,   J_x_stride0,   J_x_stride1,
                                  Rt,    Rt_stride0,    Rt_stride1,
                                  x_minus_t,  sizeof(double),
                                  true);

        // I want J_t = -transpose(R)
        if(J_Rt)
            for(int i=0; i<3; i++)
                for(int j=0; j<3; j++)
                    P3(J_Rt, i, 3, j) = -P2(Rt, j, i);
    }
}

// Invert a rotation matrix. This is a transpose
//
// The input is given in R_in in a (3,3) array
//
// The result is returned in a (3,3) array R_out. In-place operation is
// supported
void mrcal_invert_R_full( // output
                         double* R_out,      // (3,3) array
                         int R_out_stride0,  // in bytes. <= 0 means "contiguous"
                         int R_out_stride1,  // in bytes. <= 0 means "contiguous"

                         // input
                         const double* R_in, // (3,3) array
                         int R_in_stride0,   // in bytes. <= 0 means "contiguous"
                         int R_in_stride1    // in bytes. <= 0 means "contiguous"
                         )
{
    init_stride_2D(R_out, 3,3);
    init_stride_2D(R_in,  3,3);

    // transpose(R). Extra stuff to make in-place operations work
    for(int i=0; i<3; i++)
        P2(R_out,i,i) = P2(R_in,i,i);
    for(int i=0; i<3; i++)
        for(int j=i+1; j<3; j++)
        {
            double tmp = P2(R_in,i,j);
            P2(R_out,i,j) = P2(R_in,j,i);
            P2(R_out,j,i) = tmp;
        }
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_r_from_R().
void mrcal_rt_from_Rt_full(// output
                           double* rt,      // (6,) vector
                           int rt_stride0,  // in bytes. <= 0 means "contiguous"
                           double* J_R,     // (3,3,3) array. Gradient. May be NULL
                           // No J_t. It's always the identity
                           int J_R_stride0, // in bytes. <= 0 means "contiguous"
                           int J_R_stride1, // in bytes. <= 0 means "contiguous"
                           int J_R_stride2, // in bytes. <= 0 means "contiguous"

                           // input
                           const double* Rt,  // (4,3) array
                           int Rt_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_stride1     // in bytes. <= 0 means "contiguous"
                           )
{
    mrcal_r_from_R_full(rt,  rt_stride0,
                        J_R, J_R_stride0, J_R_stride1, J_R_stride2,
                        Rt,  Rt_stride0,  Rt_stride1);

    init_stride_1D(rt,  6);
    // init_stride_3D(J_R, 3,3,3);
    init_stride_2D(Rt,  4,3);

    for(int i=0; i<3; i++)
        P1(rt, i+3) = P2(Rt,3,i);
}

// Convert a transformation representation from Rt to rt. This is mostly a
// convenience functions since 99% of the work is done by mrcal_R_from_r().
void mrcal_Rt_from_rt_full(// output
                           double* Rt,      // (4,3) array
                           int Rt_stride0,  // in bytes. <= 0 means "contiguous"
                           int Rt_stride1,  // in bytes. <= 0 means "contiguous"
                           double* J_r,     // (3,3,3) array. Gradient. May be NULL
                           // No J_t. It's just the identity
                           int J_r_stride0, // in bytes. <= 0 means "contiguous"
                           int J_r_stride1, // in bytes. <= 0 means "contiguous"
                           int J_r_stride2, // in bytes. <= 0 means "contiguous"

                           // input
                           const double* rt, // (6,) vector
                           int rt_stride0    // in bytes. <= 0 means "contiguous"
                           )
{
    mrcal_R_from_r_full(Rt,  Rt_stride0,  Rt_stride1,
                        J_r, J_r_stride0, J_r_stride1, J_r_stride2,
                        rt,  rt_stride0);

    init_stride_1D(rt,  6);
    // init_stride_3D(J_r, 3,3,3);
    init_stride_2D(Rt,  4,3);

    for(int i=0; i<3; i++)
        P2(Rt,3,i) = P1(rt,i+3);
}

// Invert an Rt transformation
//
// b = Ra + t  -> a = R'b - R't
void mrcal_invert_Rt_full( // output
                          double* Rt_out,      // (4,3) array
                          int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                          int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                          // input
                          const double* Rt_in, // (4,3) array
                          int Rt_in_stride0,   // in bytes. <= 0 means "contiguous"
                          int Rt_in_stride1    // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_2D(Rt_out, 4,3);
    init_stride_2D(Rt_in,  4,3);

    // transpose(R). Extra stuff to make in-place operations work
    for(int i=0; i<3; i++)
        P2(Rt_out,i,i) = P2(Rt_in,i,i);
    for(int i=0; i<3; i++)
        for(int j=i+1; j<3; j++)
        {
            double tmp = P2(Rt_in,i,j);
            P2(Rt_out,i,j) = P2(Rt_in,j,i);
            P2(Rt_out,j,i) = tmp;
        }

    // -transpose(R)*t
    mul_vec3_gen33t_vout_scaled_full(&P2(Rt_out,3,0), Rt_out_stride1,
                                     &P2(Rt_in, 3,0), Rt_in_stride1,
                                     Rt_out, Rt_out_stride0, Rt_out_stride1,
                                     -1.0);
}

// Invert an rt transformation
//
// b = rotate(a) + t  -> a = invrotate(b) - invrotate(t)
//
// drout_drin is not returned: it is always -I
// drout_dtin is not returned: it is always 0
void mrcal_invert_rt_full( // output
                          double* rt_out,          // (6,) array
                          int rt_out_stride0,      // in bytes. <= 0 means "contiguous"
                          double* dtout_drin,      // (3,3) array
                          int dtout_drin_stride0,  // in bytes. <= 0 means "contiguous"
                          int dtout_drin_stride1,  // in bytes. <= 0 means "contiguous"
                          double* dtout_dtin,      // (3,3) array
                          int dtout_dtin_stride0,  // in bytes. <= 0 means "contiguous"
                          int dtout_dtin_stride1,  // in bytes. <= 0 means "contiguous"

                          // input
                          const double* rt_in,     // (6,) array
                          int rt_in_stride0        // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_1D(rt_out, 6);
    // init_stride_2D(dtout_drin, 3,3);
    init_stride_2D(dtout_dtin, 3,3);
    init_stride_1D(rt_in,  6);

    // r uses an angle-axis representation, so to undo a rotation r, I can apply
    // a rotation -r (same axis, equal and opposite angle)
    for(int i=0; i<3; i++)
        P1(rt_out,i) = -P1(rt_in,i);

    mrcal_rotate_point_r_full( &P1(rt_out,3), rt_out_stride0,
                               dtout_drin, dtout_drin_stride0, dtout_drin_stride1,
                               dtout_dtin, dtout_dtin_stride0, dtout_dtin_stride1,

                               // input
                               rt_out, rt_out_stride0,
                               &P1(rt_in,3), rt_in_stride0,
                               false);
    for(int i=0; i<3; i++)
        P1(rt_out,3+i) *= -1.;

    if(dtout_dtin)
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                P2(dtout_dtin,i,j) *= -1.;
}


// Compose two Rt transformations
//   R0*(R1*x + t1) + t0 =
//   (R0*R1)*x + R0*t1+t0
void mrcal_compose_Rt_full( // output
                           double* Rt_out,      // (4,3) array
                           int Rt_out_stride0,  // in bytes. <= 0 means "contiguous"
                           int Rt_out_stride1,  // in bytes. <= 0 means "contiguous"

                           // input
                           const double* Rt_0,  // (4,3) array
                           int Rt_0_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_0_stride1,    // in bytes. <= 0 means "contiguous"
                           const double* Rt_1,  // (4,3) array
                           int Rt_1_stride0,    // in bytes. <= 0 means "contiguous"
                           int Rt_1_stride1     // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_2D(Rt_out, 4,3);
    init_stride_2D(Rt_0,   4,3);
    init_stride_2D(Rt_1,   4,3);

    // for in-place operation
    double t0[] = { P2(Rt_0,3,0),
                    P2(Rt_0,3,1),
                    P2(Rt_0,3,2) };

    // t <- R0*t1
    mul_vec3_gen33t_vout_full(&P2(Rt_out,3,0), Rt_out_stride1,
                              &P2(Rt_1,  3,0), Rt_1_stride1,
                              Rt_0, Rt_0_stride0, Rt_0_stride1);

    // R <- R0*R1
    mul_gen33_gen33_vout_full( Rt_out, Rt_out_stride0, Rt_out_stride1,
                               Rt_0,   Rt_0_stride0,   Rt_0_stride1,
                               Rt_1,   Rt_1_stride0,   Rt_1_stride1 );

    // t <- R0*t1+t0
    for(int i=0; i<3; i++)
        P2(Rt_out,3,i) += t0[i];
}

// Compose two rt transformations. It is assumed that we're getting no gradients
// at all or we're getting ALL the gradients: only dr_r0 is checked for NULL
//
// dr_dt0 is not returned: it is always 0
// dr_dt1 is not returned: it is always 0
// dt_dr1 is not returned: it is always 0
// dt_dt0 is not returned: it is always the identity matrix
void mrcal_compose_rt_full( // output
                           double* rt_out,       // (6,) array
                           int rt_out_stride0,   // in bytes. <= 0 means "contiguous"
                           double* dr_r0,        // (3,3) array; may be NULL
                           int dr_r0_stride0,    // in bytes. <= 0 means "contiguous"
                           int dr_r0_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dr_r1,        // (3,3) array; may be NULL
                           int dr_r1_stride0,    // in bytes. <= 0 means "contiguous"
                           int dr_r1_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dt_r0,        // (3,3) array; may be NULL
                           int dt_r0_stride0,    // in bytes. <= 0 means "contiguous"
                           int dt_r0_stride1,    // in bytes. <= 0 means "contiguous"
                           double* dt_t1,        // (3,3) array; may be NULL
                           int dt_t1_stride0,    // in bytes. <= 0 means "contiguous"
                           int dt_t1_stride1,    // in bytes. <= 0 means "contiguous"

                           // input
                           const double* rt_0,   // (6,) array
                           int rt_0_stride0,     // in bytes. <= 0 means "contiguous"
                           const double* rt_1,   // (6,) array
                           int rt_1_stride0      // in bytes. <= 0 means "contiguous"
                            )
{
    init_stride_1D(rt_out, 6);
    init_stride_2D(dr_r0,  3,3);
    init_stride_2D(dr_r1,  3,3);
    init_stride_2D(dt_r0,  3,3);
    init_stride_2D(dt_t1,  3,3);
    init_stride_1D(rt_0,   6);
    init_stride_1D(rt_1,   6);

    // r0 (r1 x + t1) + t0 = r0 r1 x + r0 t1 + t0
    // -> I want (r0 r1, r0 t1 + t0)


    // to make in-place operation work
    double rt0[6];
    for(int i=0; i<6; i++)
        rt0[i] = P1(rt_0, i);

    // Compute r01
    mrcal_compose_r_full( rt_out, rt_out_stride0,
                          dr_r0, dr_r0_stride0, dr_r0_stride1,
                          dr_r1, dr_r1_stride0, dr_r1_stride1,

                          rt_0, rt_0_stride0,
                          rt_1, rt_1_stride0);


    // t01 <- r0 t1
    mrcal_rotate_point_r_full( &P1(rt_out,3), rt_out_stride0,
                               dt_r0, dt_r0_stride0, dt_r0_stride1,
                               dt_t1, dt_t1_stride0, dt_t1_stride1,

                               rt0, -1,
                               &P1(rt_1,3), rt_1_stride0,

                               false );
    // t01 <- r0 t1 + t0
    for(int i=0; i<3; i++)
        P1(rt_out,3+i) += rt0[3+i];
}

void mrcal_compose_r_tinyr0_gradientr0_full( // output
                           double* dr_dr0,      // (3,3) array; may be NULL
                           int dr_dr0_stride0,  // in bytes. <= 0 means "contiguous"
                           int dr_dr0_stride1,  // in bytes. <= 0 means "contiguous"

                           // input
                           const double* r_1,   // (3,) array
                           int r_1_stride0      // in bytes. <= 0 means "contiguous"
                           )
{
    init_stride_2D(dr_dr0, 3, 3);
    init_stride_1D(r_1, 3);

    // All the comments and logic appear in compose_r_core() in
    // poseutils-uses-autodiff.cc. This is a special-case function with
    // manually-computed gradients (because I want to make sure they're fast)
    double norm2_r1 = 0.0;
    for(int i=0; i<3; i++)
        norm2_r1 += P1(r_1,i)*P1(r_1,i);

    if(norm2_r1 < 2e-8*2e-8)
    {
        // Both vectors are tiny, so I have r01 = r0 + r1, and the gradient is
        // an identity matrix
        for(int i=0; i<3; i++)
            for(int j=0; j<3; j++)
                P2(dr_dr0,i,j) = i==j ? 1.0 : 0.0;
        return;
    }

    // I have
    // r01 = r1
    //     - inner(r0,r1) (B/tanB - 1) / 4B^2 r1
    //     + B/tanB r0
    //     + cross(r0,r1) / 2
    //
    // I differentiate:
    //
    //   dr01/dr0 =
    //     - outer(r1,r1) (B/tanB - 1) / 4B^2
    //     + B/tanB I
    //     - skew_symmetric(r1) / 2
    double B    = sqrt(norm2_r1) / 2.;
    double B_over_tanB = B / tan(B);

    for(int i=0; i<3; i++)
        for(int j=0; j<3; j++)
            P2(dr_dr0,i,j) =
                - P1(r_1,i)*P1(r_1,j) * (B_over_tanB - 1.) / (4.*B*B);
    for(int i=0; i<3; i++)
        P2(dr_dr0,i,i) +=
            B_over_tanB;

    P2(dr_dr0,0,1) -= -P1(r_1,2)/2.;
    P2(dr_dr0,0,2) -=  P1(r_1,1)/2.;
    P2(dr_dr0,1,0) -=  P1(r_1,2)/2.;
    P2(dr_dr0,1,2) -= -P1(r_1,0)/2.;
    P2(dr_dr0,2,0) -= -P1(r_1,1)/2.;
    P2(dr_dr0,2,1) -=  P1(r_1,0)/2.;
}
