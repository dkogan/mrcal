#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#include <dogleg.h>
#include <minimath.h>
#include <assert.h>
#include <stdbool.h>
#include <math.h>
#include <string.h>

#include "mrcal.h"

// These are parameter variable scales. They have the units of the parameters
// themselves, so the optimizer sees x/SCALE_X for each parameter. I.e. as far
// as the optimizer is concerned, the scale of each variable is 1. This doesn't
// need to be precise; just need to get all the variables to be within the same
// order of magnitute. This is important because the dogleg solve treats the
// trust region as a ball in state space, and this ball is isotrophic, and has a
// radius that applies in every direction
//
// Can be visualized like this:
//
//   p0,x0,J0 = mrcal.optimizer_callback(**optimization_inputs)[:3]
//   J0 = J0.toarray()
//   ss = np.sum(np.abs(J0), axis=-2)
//   gp.plot(ss, _set=mrcal.plotoptions_state_boundaries(**optimization_inputs))
//
// This visualizes the overall effect of each variable. If the scales aren't
// tuned properly, some variables will have orders of magnitude stronger
// response than others, and the optimization problem won't converge well.
//
// The scipy.optimize.least_squares() function claims to be able to estimate
// these automatically, without requiring these hard-coded values from the user.
// See the description of the "x_scale" argument:
//
//   https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
//
// Supposedly this paper describes the method:
//
//   J. J. More, "The Levenberg-Marquardt Algorithm: Implementation and Theory,"
//   Numerical Analysis, ed. G. A. Watson, Lecture Notes in Mathematics 630,
//   Springer Verlag, pp. 105-116, 1977.
//
// Please somebody look at this
#define SCALE_INTRINSICS_FOCAL_LENGTH 500.0
#define SCALE_INTRINSICS_CENTER_PIXEL 20.0
#define SCALE_ROTATION_CAMERA         (0.1 * M_PI/180.0)
#define SCALE_TRANSLATION_CAMERA      1.0
#define SCALE_ROTATION_FRAME          (15.0 * M_PI/180.0)
#define SCALE_TRANSLATION_FRAME       1.0
#define SCALE_POSITION_POINT          SCALE_TRANSLATION_FRAME
#define SCALE_CALOBJECT_WARP          0.01

#define SCALE_DISTORTION              1.0

#define MSG(fmt, ...) fprintf(stderr, "%s(%d): " fmt "\n", __FILE__, __LINE__, ##__VA_ARGS__)
#define MSG_IF_VERBOSE(...) do { if(verbose) MSG( __VA_ARGS__ ); } while(0)



#define CHECK_CONFIG_NPARAM_NOCONFIG(s,n) \
    static_assert(n > 0, "no-config implies known-at-compile-time param count");
#define CHECK_CONFIG_NPARAM_WITHCONFIG(s,n) \
    static_assert(n <= 0, "with-config implies unknown-at-compile-time param count");
MRCAL_LENSMODEL_NOCONFIG_LIST(  CHECK_CONFIG_NPARAM_NOCONFIG)
MRCAL_LENSMODEL_WITHCONFIG_LIST(CHECK_CONFIG_NPARAM_WITHCONFIG)


// Returns a static string, using "..." as a placeholder for any configuration
// values
#define LENSMODEL_PRINT_CFG_ELEMENT_TEMPLATE(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    "_" #name "=..."
#define LENSMODEL_PRINT_CFG_ELEMENT_FMT(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    "_" #name "=%" PRIcode
#define LENSMODEL_PRINT_CFG_ELEMENT_VAR(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    ,config->name
#define LENSMODEL_SCAN_CFG_ELEMENT_FMT(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    "_" #name "=%" SCNcode
#define LENSMODEL_SCAN_CFG_ELEMENT_VAR(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    ,&config->name
#define LENSMODEL_SCAN_CFG_ELEMENT_PLUS1(name, type, pybuildvaluecode, PRIcode,SCNcode, bitfield, cookie) \
    +1
const char* mrcal_lensmodel_name( mrcal_lensmodel_t model )
{
    switch(model.type)
    {
#define CASE_STRING_NOCONFIG(s,n) case MRCAL_##s: ;     \
        return #s;
#define _CASE_STRING_WITHCONFIG(s,n,s_CONFIG_LIST) case MRCAL_##s: ;    \
        return #s s_CONFIG_LIST(LENSMODEL_PRINT_CFG_ELEMENT_TEMPLATE, );
#define CASE_STRING_WITHCONFIG(s,n) _CASE_STRING_WITHCONFIG(s,n,MRCAL_ ## s ## _CONFIG_LIST)

        MRCAL_LENSMODEL_NOCONFIG_LIST(   CASE_STRING_NOCONFIG )
        MRCAL_LENSMODEL_WITHCONFIG_LIST( CASE_STRING_WITHCONFIG )

    default:
        assert(0);


#undef CASE_STRING_NOCONFIG
#undef CASE_STRING_WITHCONFIG

    }
    return NULL;
}

// Write the model name WITH the full config into the given buffer. Identical to
// mrcal_lensmodel_name() for configuration-free models
static int LENSMODEL_SPLINED_STEREOGRAPHIC__snprintf_model
  (char* out, int size,
   const mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config)
{
    return
        snprintf( out, size, "LENSMODEL_SPLINED_STEREOGRAPHIC"
                  MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(LENSMODEL_PRINT_CFG_ELEMENT_FMT, )
                  MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(LENSMODEL_PRINT_CFG_ELEMENT_VAR, ));
}
bool mrcal_lensmodel_name_full( char* out, int size, mrcal_lensmodel_t model )
{
    switch(model.type)
    {
#define CASE_STRING_NOCONFIG(s,n) case MRCAL_##s: \
        return size > snprintf(out,size, #s);

#define CASE_STRING_WITHCONFIG(s,n) case MRCAL_##s: \
        return size > s##__snprintf_model(out, size, &model.s##__config);

        MRCAL_LENSMODEL_NOCONFIG_LIST(   CASE_STRING_NOCONFIG )
        MRCAL_LENSMODEL_WITHCONFIG_LIST( CASE_STRING_WITHCONFIG )

    default:
        assert(0);

#undef CASE_STRING_NOCONFIG
#undef CASE_STRING_WITHCONFIG

    }
    return NULL;
}


static bool LENSMODEL_SPLINED_STEREOGRAPHIC__scan_model_config( mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config, const char* config_str)
{
    int pos;
    int Nelements = 0 MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(LENSMODEL_SCAN_CFG_ELEMENT_PLUS1, );
    return
        Nelements ==
        sscanf( config_str,
                MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(LENSMODEL_SCAN_CFG_ELEMENT_FMT, )"%n"
                MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC_CONFIG_LIST(LENSMODEL_SCAN_CFG_ELEMENT_VAR, ),
                &pos) &&
        config_str[pos] == '\0';
}

const char* const* mrcal_supported_lensmodel_names( void )
{
#define NAMESTRING_NOCONFIG(s,n)                  #s,
#define _NAMESTRING_WITHCONFIG(s,n,s_CONFIG_LIST) #s s_CONFIG_LIST(LENSMODEL_PRINT_CFG_ELEMENT_TEMPLATE, ),
#define NAMESTRING_WITHCONFIG(s,n) _NAMESTRING_WITHCONFIG(s,n,MRCAL_ ## s ## _CONFIG_LIST)

    static const char* names[] = {
        MRCAL_LENSMODEL_NOCONFIG_LIST(  NAMESTRING_NOCONFIG)
        MRCAL_LENSMODEL_WITHCONFIG_LIST(NAMESTRING_WITHCONFIG)
        NULL };
    return names;
}

#undef LENSMODEL_PRINT_CFG_ELEMENT_TEMPLATE
#undef LENSMODEL_PRINT_CFG_ELEMENT_FMT
#undef LENSMODEL_PRINT_CFG_ELEMENT_VAR
#undef LENSMODEL_SCAN_CFG_ELEMENT_FMT
#undef LENSMODEL_SCAN_CFG_ELEMENT_VAR
#undef LENSMODEL_SCAN_CFG_ELEMENT_PLUS1

// parses the model name AND the configuration into a mrcal_lensmodel_t structure.
// Strings with valid model names but missing or unparseable configuration
// return {.type = MRCAL_LENSMODEL_INVALID_BADCONFIG}. Unknown model names return
// {.type = MRCAL_LENSMODEL_INVALID}
mrcal_lensmodel_t mrcal_lensmodel_from_name( const char* name )
{
#define CHECK_AND_RETURN_NOCONFIG(s,n)                                  \
    if( 0 == strcmp( name, #s) )                                        \
        return (mrcal_lensmodel_t){.type = MRCAL_##s};

#define CHECK_AND_RETURN_WITHCONFIG(s,n)                                \
    /* Configured model. I need to extract the config from the string. */ \
    /* The string format is NAME_cfg1=var1_cfg2=var2... */              \
    if( 0 == strcmp( name, #s) )                                        \
        return (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG};      \
    if( 0 == strncmp( name, #s"_", strlen(#s)+1) )                      \
    {                                                                   \
        /* found name. Now extract the config */                        \
        mrcal_lensmodel_t model = {.type = MRCAL_##s};                  \
        mrcal_##s##__config_t* config = &model.s##__config;             \
                                                                        \
        const char* config_str = &name[strlen(#s)];                     \
                                                                        \
        if(s##__scan_model_config(config, config_str))                  \
            return model;                                               \
        else                                                            \
            return (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID_BADCONFIG};  \
    }

    MRCAL_LENSMODEL_NOCONFIG_LIST(   CHECK_AND_RETURN_NOCONFIG );
    MRCAL_LENSMODEL_WITHCONFIG_LIST( CHECK_AND_RETURN_WITHCONFIG );

    return (mrcal_lensmodel_t){.type = MRCAL_LENSMODEL_INVALID};

#undef CHECK_AND_RETURN_NOCONFIG
#undef CHECK_AND_RETURN_WITHCONFIG
}

// parses the model name only. The configuration is ignored. Even if it's
// missing or unparseable. Unknown model names return MRCAL_LENSMODEL_INVALID
mrcal_lensmodel_type_t mrcal_lensmodel_type_from_name( const char* name )
{
#define CHECK_AND_RETURN_NOCONFIG(s,n)                                  \
    if( 0 == strcmp( name, #s) ) return MRCAL_##s;

#define CHECK_AND_RETURN_WITHCONFIG(s,n)                                \
    /* Configured model. If the name is followed by _ or nothing, I */  \
    /* accept this model */                                             \
    if( 0 == strcmp( name, #s) ) return MRCAL_##s;                      \
    if( 0 == strncmp( name, #s"_", strlen(#s)+1) ) return MRCAL_##s;

    MRCAL_LENSMODEL_NOCONFIG_LIST(   CHECK_AND_RETURN_NOCONFIG );
    MRCAL_LENSMODEL_WITHCONFIG_LIST( CHECK_AND_RETURN_WITHCONFIG );

    return MRCAL_LENSMODEL_INVALID;

#undef CHECK_AND_RETURN_NOCONFIG
#undef CHECK_AND_RETURN_WITHCONFIG
}

mrcal_lensmodel_meta_t mrcal_lensmodel_meta( const mrcal_lensmodel_t m )
{
    switch(m.type)
    {
    case MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC:
    case MRCAL_LENSMODEL_STEREOGRAPHIC:
        return (mrcal_lensmodel_meta_t) { .has_core                  = true,
                                          .can_project_behind_camera = true };
    case MRCAL_LENSMODEL_PINHOLE:
    case MRCAL_LENSMODEL_OPENCV4:
    case MRCAL_LENSMODEL_OPENCV5:
    case MRCAL_LENSMODEL_OPENCV8:
    case MRCAL_LENSMODEL_OPENCV12:
    case MRCAL_LENSMODEL_CAHVOR:
    case MRCAL_LENSMODEL_CAHVORE:
        return (mrcal_lensmodel_meta_t) { .has_core                  = true,
                                          .can_project_behind_camera = false };

    default: ;
    }
    MSG("Unknown lens model %d. Barfing out", m.type);
    assert(0);
}

static
bool modelHasCore_fxfycxcy( const mrcal_lensmodel_t m )
{
    mrcal_lensmodel_meta_t meta = mrcal_lensmodel_meta(m);
    return meta.has_core;
}
static
bool model_supports_projection_behind_camera( const mrcal_lensmodel_t m )
{
    mrcal_lensmodel_meta_t meta = mrcal_lensmodel_meta(m);
    return meta.can_project_behind_camera;
}

static int LENSMODEL_SPLINED_STEREOGRAPHIC__lensmodel_num_params(const mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config)
{
    return
        // I have two surfaces: one for x and another for y
        (int)config->Nx * (int)config->Ny * 2 +

        // and I have a core
        4;
}
int mrcal_lensmodel_num_params(const mrcal_lensmodel_t m)
{
    switch(m.type)
    {
#define CASE_NUM_NOCONFIG(s,n)                                          \
        case MRCAL_##s: return n;

#define CASE_NUM_WITHCONFIG(s,n)                                        \
        case MRCAL_##s: return s##__lensmodel_num_params(&m.s##__config);

        MRCAL_LENSMODEL_NOCONFIG_LIST(   CASE_NUM_NOCONFIG )
        MRCAL_LENSMODEL_WITHCONFIG_LIST( CASE_NUM_WITHCONFIG )

    default: ;
    }
    return -1;

#undef CASE_NUM_NOCONFIG
#undef CASE_NUM_WITHCONFIG
}

static
int get_num_distortions_optimization_params(mrcal_problem_details_t problem_details,
                                            mrcal_lensmodel_t lensmodel)
{
    if( !problem_details.do_optimize_intrinsics_distortions )
        return 0;

    int N = mrcal_lensmodel_num_params(lensmodel);
    if(modelHasCore_fxfycxcy(lensmodel))
        N -= 4; // ignoring fx,fy,cx,cy
    return N;
}

int mrcal_num_intrinsics_optimization_params(mrcal_problem_details_t problem_details,
                                             mrcal_lensmodel_t lensmodel)
{
    int N = get_num_distortions_optimization_params(problem_details, lensmodel);

    if( problem_details.do_optimize_intrinsics_core &&
        modelHasCore_fxfycxcy(lensmodel) )
        N += 4; // fx,fy,cx,cy
    return N;
}

int mrcal_num_states(int Ncameras_intrinsics, int Ncameras_extrinsics,
                     int Nframes,
                     int Npoints, int Npoints_fixed,
                     mrcal_problem_details_t problem_details,
                     mrcal_lensmodel_t lensmodel)
{
    int Npoints_variable = Npoints - Npoints_fixed;

    return
        // camera extrinsics
        (problem_details.do_optimize_extrinsics ? (Ncameras_extrinsics * 6) : 0) +

        // frame poses, individual observed points
        (problem_details.do_optimize_frames ? (Nframes * 6 + Npoints_variable * 3) : 0) +

        // camera intrinsics
        (Ncameras_intrinsics * mrcal_num_intrinsics_optimization_params(problem_details, lensmodel)) +

        // warp
        (problem_details.do_optimize_calobject_warp ? 2 : 0);
}

static int num_regularization_terms_percamera(mrcal_problem_details_t problem_details,
                                              mrcal_lensmodel_t lensmodel)
{
    if(problem_details.do_skip_regularization)
        return 0;

    // distortions
    int N = get_num_distortions_optimization_params(problem_details, lensmodel);
    // optical center
    if(problem_details.do_optimize_intrinsics_core)
        N += 2;
    return N;
}

int mrcal_measurement_index_boards(int i_observation_board,
                                   int Nobservations_board,
                                   int Nobservations_point,
                                   int calibration_object_width_n,
                                   int calibration_object_height_n)
{
    // *2 because I have separate x and y measurements
    return
        0 +
        i_observation_board *
        calibration_object_width_n*calibration_object_height_n *
        2;
}

int mrcal_num_measurements_boards(int Nobservations_board,
                                  int calibration_object_width_n,
                                  int calibration_object_height_n)
{
    return mrcal_measurement_index_boards( Nobservations_board,
                                           0,0,
                                           calibration_object_width_n,
                                           calibration_object_height_n);
}

int mrcal_measurement_index_points(int i_observation_point,
                                   int Nobservations_board,
                                   int Nobservations_point,
                                   int calibration_object_width_n,
                                   int calibration_object_height_n)
{
    // 3: x,y measurements, range normalization
    return
        mrcal_num_measurements_boards(Nobservations_board,
                                      calibration_object_width_n,
                                      calibration_object_height_n) +
        i_observation_point * 3;
}

int mrcal_num_measurements_points(int Nobservations_point)
{
    // 3: x,y measurements, range normalization
    return Nobservations_point * 3;
}

int mrcal_measurement_index_regularization(int Nobservations_board,
                                           int Nobservations_point,
                                           int calibration_object_width_n,
                                           int calibration_object_height_n)
{
    return
        mrcal_num_measurements_boards(Nobservations_board,
                                      calibration_object_width_n,
                                      calibration_object_height_n) +
        mrcal_num_measurements_points(Nobservations_point);
}

int mrcal_num_measurements_regularization(int Ncameras_intrinsics, int Ncameras_extrinsics,
                                          int Nframes,
                                          int Npoints, int Npoints_fixed,
                                          mrcal_problem_details_t problem_details,
                                          mrcal_lensmodel_t lensmodel)
{
    return
        Ncameras_intrinsics *
        num_regularization_terms_percamera(problem_details, lensmodel);
}

int mrcal_num_measurements(int Nobservations_board,
                           int Nobservations_point,
                           int calibration_object_width_n,
                           int calibration_object_height_n,
                           int Ncameras_intrinsics, int Ncameras_extrinsics,
                           int Nframes,
                           int Npoints, int Npoints_fixed,
                           mrcal_problem_details_t problem_details,
                           mrcal_lensmodel_t lensmodel)
{
    return
        mrcal_num_measurements_boards( Nobservations_board,
                                       calibration_object_width_n,
                                       calibration_object_height_n) +
        mrcal_num_measurements_points(Nobservations_point) +
        mrcal_num_measurements_regularization(Ncameras_intrinsics, Ncameras_extrinsics,
                                              Nframes,
                                              Npoints, Npoints_fixed,
                                              problem_details,
                                              lensmodel);
}

int mrcal_num_j_nonzero(int Nobservations_board,
                        int Nobservations_point,
                        int calibration_object_width_n,
                        int calibration_object_height_n,
                        int Ncameras_intrinsics, int Ncameras_extrinsics,
                        int Nframes,
                        int Npoints, int Npoints_fixed,
                        const mrcal_observation_board_t* observations_board,
                        const mrcal_observation_point_t* observations_point,
                        mrcal_problem_details_t problem_details,
                        mrcal_lensmodel_t lensmodel)
{
    // each observation depends on all the parameters for THAT frame and for
    // THAT camera. Camera0 doesn't have extrinsics, so I need to loop through
    // all my observations

    // Each projected point has an x and y measurement, and each one depends on
    // some number of the intrinsic parameters. Parametric models are simple:
    // each one depends on ALL of the intrinsics. Splined models are sparse,
    // however, and there's only a partial dependence
    int Nintrinsics_per_measurement;
    if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        int run_len =
            lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.order + 1;
        Nintrinsics_per_measurement =
            (problem_details.do_optimize_intrinsics_core        ? 4                 : 0)  +
            (problem_details.do_optimize_intrinsics_distortions ? (run_len*run_len) : 0);
    }
    else
        Nintrinsics_per_measurement =
            mrcal_num_intrinsics_optimization_params(problem_details, lensmodel);

    // x depends on fx,cx but NOT on fy, cy. And similarly for y.
    if( problem_details.do_optimize_intrinsics_core &&
        modelHasCore_fxfycxcy(lensmodel) )
        Nintrinsics_per_measurement -= 2;

    int N = Nobservations_board * ( (problem_details.do_optimize_frames         ? 6 : 0) +
                                    (problem_details.do_optimize_extrinsics     ? 6 : 0) +
                                    (problem_details.do_optimize_calobject_warp ? 2 : 0) +
                                  Nintrinsics_per_measurement );

    // initial estimate counts extrinsics for the reference camera, which need
    // to be subtracted off
    if(problem_details.do_optimize_extrinsics)
        for(int i=0; i<Nobservations_board; i++)
            if(observations_board[i].i_cam_extrinsics < 0)
                N -= 6;
    // *2 because I have separate x and y measurements
    N *= 2*calibration_object_width_n*calibration_object_height_n;

    // Now the point observations
    for(int i=0; i<Nobservations_point; i++)
    {
        N += 2*Nintrinsics_per_measurement;
        if( problem_details.do_optimize_frames &&
            observations_point[i].i_point < Npoints-Npoints_fixed )
            N += 2*3;
        if( problem_details.do_optimize_extrinsics &&
            observations_point[i].i_cam_extrinsics >= 0 )
            N += 2*6;

        // range normalization
        if(problem_details.do_optimize_frames &&
            observations_point[i].i_point < Npoints-Npoints_fixed )
            N += 3;
        if( problem_details.do_optimize_extrinsics &&
            observations_point[i].i_cam_extrinsics >= 0 )
            N += 6;
    }

    N +=
        Ncameras_intrinsics *
        num_regularization_terms_percamera(problem_details,
                                           lensmodel);

    return N;
}

// Used in the spline-based projection function.
//
// See bsplines.py for the derivation of the spline expressions and for
// justification of the 2D scheme
//
// Here we sample two interpolated surfaces at once: one each for the x and y
// focal-length scales
static
void sample_bspline_surface_cubic(double* out,
                                  double* dout_dx, // may be NULL
                                  double* dout_dy, // may be NULL
                                  double* ABCDx_ABCDy,

                                  double x, double y,
                                  // control points
                                  const double* c,
                                  int stridey

                                  // stridex is 2: the control points from the
                                  // two surfaces are next to each other. Better
                                  // cache locality maybe
                                  )
{
    double* ABCDx = &ABCDx_ABCDy[0];
    double* ABCDy = &ABCDx_ABCDy[4];

    // The sampling function assumes evenly spaced knots.
    // a,b,c,d are sequential control points
    // x is in [0,1] between b and c. Function looks like this:
    //   double A = fA(x);
    //   double B = fB(x);
    //   double C = fC(x);
    //   double D = fD(x);
    //   return A*a + B*b + C*c + D*d;
    // I need to sample many such 1D segments, so I compute A,B,C,D separately,
    // and apply them together
    void get_sample_coeffs(double* ABCD, double* ABCDgrad, double x)
    {
        double x2 = x*x;
        double x3 = x2*x;
        ABCD[0] =  (-x3 + 3*x2 - 3*x + 1)/6;
        ABCD[1] = (3 * x3/2 - 3*x2 + 2)/3;
        ABCD[2] = (-3 * x3 + 3*x2 + 3*x + 1)/6;
        ABCD[3] = x3 / 6;

        ABCDgrad[0] =  -x2/2 + x - 1./2.;
        ABCDgrad[1] = 3*x2/2 - 2*x;
        ABCDgrad[2] = -3*x2/2 + x + 1./2.;
        ABCDgrad[3] = x2 / 2;
    }

    // 4 samples along one dimension, and then one sample along the other
    // dimension, using the 4 samples as the control points. Order doesn't
    // matter. See bsplines.py
    //
    // I do this twice: one for each focal length surface
    double ABCDgradx[4];
    double ABCDgrady[4];
    get_sample_coeffs(ABCDx, ABCDgradx, x);
    get_sample_coeffs(ABCDy, ABCDgrady, y);
    void interp(double* out, const double* ABCDx, const double* ABCDy)
    {
        double cinterp[4][2];
        const int stridex = 2;
        for(int iy=0; iy<4; iy++)
            for(int k=0;k<2;k++)
                cinterp[iy][k] =
                    ABCDx[0] * c[iy*stridey + 0*stridex + k] +
                    ABCDx[1] * c[iy*stridey + 1*stridex + k] +
                    ABCDx[2] * c[iy*stridey + 2*stridex + k] +
                    ABCDx[3] * c[iy*stridey + 3*stridex + k];
        for(int k=0;k<2;k++)
            out[k] =
                ABCDy[0] * cinterp[0][k] +
                ABCDy[1] * cinterp[1][k] +
                ABCDy[2] * cinterp[2][k] +
                ABCDy[3] * cinterp[3][k];
    }

    // the intrinsics gradient is flatten(ABCDx[0..3] * ABCDy[0..3]) for both x
    // and y. By returning ABCD[xy] and not the cartesian products, I make
    // smaller temporary data arrays
    interp(out,     ABCDx,     ABCDy);
    if(dout_dx)
        interp(dout_dx, ABCDgradx, ABCDy);
    if(dout_dy)
        interp(dout_dy, ABCDx,     ABCDgrady);
}
static
void sample_bspline_surface_quadratic(double* out,
                                      double* dout_dx, // may be NULL
                                      double* dout_dy, // may be NULL
                                      double* ABCx_ABCy,

                                      double x, double y,
                                      // control points
                                      const double* c,
                                      int stridey

                                      // stridex is 2: the control points from the
                                      // two surfaces are next to each other. Better
                                      // cache locality maybe
                                      )
{
    double* ABCx = &ABCx_ABCy[0];
    double* ABCy = &ABCx_ABCy[3];

    // The sampling function assumes evenly spaced knots.
    // a,b,c are sequential control points
    // x is in [-1/2,1/2] around b. Function looks like this:
    //   double A = fA(x);
    //   double B = fB(x);
    //   double C = fC(x);
    //   return A*a + B*b + C*c;
    // I need to sample many such 1D segments, so I compute A,B,C separately,
    // and apply them together
    void get_sample_coeffs(double* ABC, double* ABCgrad, double x)
    {
        double x2 = x*x;
        ABC[0] = (4*x2 - 4*x + 1)/8;
        ABC[1] = (3 - 4*x2)/4;
        ABC[2] = (4*x2 + 4*x + 1)/8;

        ABCgrad[0] = x - 1./2.;
        ABCgrad[1] = -2.*x;
        ABCgrad[2] = x + 1./2.;
    }

    // 3 samples along one dimension, and then one sample along the other
    // dimension, using the 3 samples as the control points. Order doesn't
    // matter. See bsplines.py
    //
    // I do this twice: one for each focal length surface
    double ABCgradx[3];
    double ABCgrady[3];
    get_sample_coeffs(ABCx, ABCgradx, x);
    get_sample_coeffs(ABCy, ABCgrady, y);
    void interp(double* out, const double* ABCx, const double* ABCy)
    {
        double cinterp[3][2];
        const int stridex = 2;
        for(int iy=0; iy<3; iy++)
            for(int k=0;k<2;k++)
                cinterp[iy][k] =
                    ABCx[0] * c[iy*stridey + 0*stridex + k] +
                    ABCx[1] * c[iy*stridey + 1*stridex + k] +
                    ABCx[2] * c[iy*stridey + 2*stridex + k];
        for(int k=0;k<2;k++)
            out[k] =
                ABCy[0] * cinterp[0][k] +
                ABCy[1] * cinterp[1][k] +
                ABCy[2] * cinterp[2][k];
    }

    // the intrinsics gradient is flatten(ABCx[0..3] * ABCy[0..3]) for both x
    // and y. By returning ABC[xy] and not the cartesian products, I make
    // smaller temporary data arrays
    interp(out,     ABCx,     ABCy);
    if(dout_dx)
        interp(dout_dx, ABCgradx, ABCy);
    if(dout_dy)
        interp(dout_dy, ABCx,     ABCgrady);
}

typedef struct
{
    double _d_rj_rf[3*3];
    double _d_rj_rc[3*3];
    double _d_tj_tf[3*3];
    double _d_tj_rc[3*3];

    // _d_tj_tc is always identity
    // _d_tj_rf is always 0
    // _d_rj_tf is always 0
    // _d_rj_tc is always 0

} geometric_gradients_t;

// The implementation of _mrcal_project_internal_opencv is based on opencv. The
// sources have been heavily modified, but the opencv logic remains. This
// function is a cut-down cvProjectPoints2Internal() to keep only the
// functionality I want and to use my interfaces. Putting this here allows me to
// drop the C dependency on opencv. Which is a good thing, since opencv dropped
// their C API
//
// from opencv-4.2.0+dfsg/modules/calib3d/src/calibration.cpp
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of

// NOT A PART OF THE EXTERNAL API. This is exported for the mrcal python wrapper
// only
void _mrcal_project_internal_opencv( // outputs
                                    mrcal_point2_t* q,
                                    mrcal_point3_t* dq_dp,               // may be NULL
                                    double* dq_dintrinsics_nocore, // may be NULL

                                    // inputs
                                    const mrcal_point3_t* p,
                                    int N,
                                    const double* intrinsics,
                                    int Nintrinsics)
{
    const double fx = intrinsics[0];
    const double fy = intrinsics[1];
    const double cx = intrinsics[2];
    const double cy = intrinsics[3];

    double k[12] = {};
    for(int i=0; i<Nintrinsics-4; i++)
        k[i] = intrinsics[i+4];

    for( int i = 0; i < N; i++ )
    {
        double z_recip = 1./p[i].z;
        double x = p[i].x * z_recip;
        double y = p[i].y * z_recip;

        double r2      = x*x + y*y;
        double r4      = r2*r2;
        double r6      = r4*r2;
        double a1      = 2*x*y;
        double a2      = r2 + 2*x*x;
        double a3      = r2 + 2*y*y;
        double cdist   = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
        double icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
        double xd      = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
        double yd      = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

        q[i].x = xd*fx + cx;
        q[i].y = yd*fy + cy;


        if( dq_dp )
        {
            double dx_dp[] = { z_recip, 0,       -x*z_recip };
            double dy_dp[] = { 0,       z_recip, -y*z_recip };
            for( int j = 0; j < 3; j++ )
            {
                double dr2_dp = 2*x*dx_dp[j] + 2*y*dy_dp[j];
                double dcdist_dp = k[0]*dr2_dp + 2*k[1]*r2*dr2_dp + 3*k[4]*r4*dr2_dp;
                double dicdist2_dp = -icdist2*icdist2*(k[5]*dr2_dp + 2*k[6]*r2*dr2_dp + 3*k[7]*r4*dr2_dp);
                double da1_dp = 2*(x*dy_dp[j] + y*dx_dp[j]);
                double dmx_dp = (dx_dp[j]*cdist*icdist2 + x*dcdist_dp*icdist2 + x*cdist*dicdist2_dp +
                                k[2]*da1_dp + k[3]*(dr2_dp + 4*x*dx_dp[j]) + k[8]*dr2_dp + 2*r2*k[9]*dr2_dp);
                double dmy_dp = (dy_dp[j]*cdist*icdist2 + y*dcdist_dp*icdist2 + y*cdist*dicdist2_dp +
                                k[2]*(dr2_dp + 4*y*dy_dp[j]) + k[3]*da1_dp + k[10]*dr2_dp + 2*r2*k[11]*dr2_dp);
                dq_dp[i*2 + 0].xyz[j] = fx*dmx_dp;
                dq_dp[i*2 + 1].xyz[j] = fy*dmy_dp;
            }
        }
        if( dq_dintrinsics_nocore )
        {
            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 0] = fx*x*icdist2*r2;
            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 0] = fy*(y*icdist2*r2);

            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 1] = fx*x*icdist2*r4;
            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 1] = fy*y*icdist2*r4;

            if( Nintrinsics-4 > 2 )
            {
                dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 2] = fx*a1;
                dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 2] = fy*a3;
                dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 3] = fx*a2;
                dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 3] = fy*a1;
                if( Nintrinsics-4 > 4 )
                {
                    dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 4] = fx*x*icdist2*r6;
                    dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 4] = fy*y*icdist2*r6;

                    if( Nintrinsics-4 > 5 )
                    {
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 5] = fx*x*cdist*(-icdist2)*icdist2*r2;
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 5] = fy*y*cdist*(-icdist2)*icdist2*r2;
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 6] = fx*x*cdist*(-icdist2)*icdist2*r4;
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 6] = fy*y*cdist*(-icdist2)*icdist2*r4;
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 7] = fx*x*cdist*(-icdist2)*icdist2*r6;
                        dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 7] = fy*y*cdist*(-icdist2)*icdist2*r6;
                        if( Nintrinsics-4 > 8 )
                        {
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 8] = fx*r2; //s1
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 8] = fy*0; //s1
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 9] = fx*r4; //s2
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 9] = fy*0; //s2
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 10] = fx*0;//s3
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 10] = fy*r2; //s3
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 0) + 11] = fx*0;//s4
                            dq_dintrinsics_nocore[(Nintrinsics-4)*(2*i + 1) + 11] = fy*r4; //s4
                        }
                    }
                }
            }
        }
    }
}

// These are all internals for project(). It was getting unwieldy otherwise
static
void _project_point_parametric( // outputs
                               mrcal_point2_t* q,
                               mrcal_point2_t* dq_dfxy, double* dq_dintrinsics_nocore,
                               mrcal_point3_t* restrict dq_drcamera,
                               mrcal_point3_t* restrict dq_dtcamera,
                               mrcal_point3_t* restrict dq_drframe,
                               mrcal_point3_t* restrict dq_dtframe,

                               // inputs
                               const mrcal_point3_t* p,
                               const mrcal_point3_t* dp_drc,
                               const mrcal_point3_t* dp_dtc,
                               const mrcal_point3_t* dp_drf,
                               const mrcal_point3_t* dp_dtf,

                               const double* restrict intrinsics,
                               bool camera_at_identity,
                               mrcal_lensmodel_t lensmodel)
{
    // u = distort(p, distortions)
    // q = uxy/uz * fxy + cxy
    if( lensmodel.type == MRCAL_LENSMODEL_PINHOLE ||
        lensmodel.type == MRCAL_LENSMODEL_STEREOGRAPHIC ||
        MRCAL_LENSMODEL_IS_OPENCV(lensmodel.type) )
    {
        // q = fxy pxy/pz + cxy
        // dqx/dp = d( fx px/pz + cx ) = fx/pz^2 (pz [1 0 0] - px [0 0 1])
        // dqy/dp = d( fy py/pz + cy ) = fy/pz^2 (pz [0 1 0] - py [0 0 1])
        const double fx = intrinsics[0];
        const double fy = intrinsics[1];
        const double cx = intrinsics[2];
        const double cy = intrinsics[3];
        mrcal_point3_t dq_dp[2];
        if( lensmodel.type == MRCAL_LENSMODEL_PINHOLE )
        {
            double pz_recip = 1. / p->z;
            q->x = p->x*pz_recip * fx + cx;
            q->y = p->y*pz_recip * fy + cy;

            dq_dp[0].x = fx * pz_recip;
            dq_dp[0].y = 0;
            dq_dp[0].z = -fx*p->x*pz_recip*pz_recip;

            dq_dp[1].x = 0;
            dq_dp[1].y = fy * pz_recip;
            dq_dp[1].z = -fy*p->y*pz_recip*pz_recip;
        }
        else if(lensmodel.type == MRCAL_LENSMODEL_STEREOGRAPHIC)
        {
            mrcal_project_stereographic(q, dq_dp,
                                        p, 1, fx,fy,cx,cy);
        }
        else
        {
            int Nintrinsics = mrcal_lensmodel_num_params(lensmodel);
            _mrcal_project_internal_opencv( q, dq_dp,
                                            dq_dintrinsics_nocore,
                                            p, 1, intrinsics, Nintrinsics);
        }

        // dq/deee = dq/dp dp/deee
        if(camera_at_identity)
        {
            if( dq_drcamera != NULL ) memset(dq_drcamera, 0, 6*sizeof(double));
            if( dq_dtcamera != NULL ) memset(dq_dtcamera, 0, 6*sizeof(double));
            if( dq_drframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dp, (double*)dp_drf, (double*)dq_drframe);
            if( dq_dtframe  != NULL ) memcpy(dq_dtframe, (double*)dq_dp, 6*sizeof(double));
        }
        else
        {
            if( dq_drcamera != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dp, (double*)dp_drc, (double*)dq_drcamera);
            if( dq_dtcamera != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dp, (double*)dp_dtc, (double*)dq_dtcamera);
            if( dq_drframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dp, (double*)dp_drf, (double*)dq_drframe );
            if( dq_dtframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dp, (double*)dp_dtf, (double*)dq_dtframe );
        }

        // I have the projection, and I now need to propagate the gradients
        if( dq_dfxy )
        {
            // I have the projection, and I now need to propagate the gradients
            // xy = fxy * distort(xy)/distort(z) + cxy
            dq_dfxy->x = (q->x - cx)/fx; // dqx/dfx
            dq_dfxy->y = (q->y - cy)/fy; // dqy/dfy
        }
    }
    else if( lensmodel.type == MRCAL_LENSMODEL_CAHVOR )
    {
        int NdistortionParams = mrcal_lensmodel_num_params(lensmodel) - 4;

        // I perturb p, and then apply the focal length, center pixel stuff
        // normally
        mrcal_point3_t p_distorted;

        // distortion parameter layout:
        //   alpha
        //   beta
        //   r0
        //   r1
        //   r2
        double alpha = intrinsics[4 + 0];
        double beta  = intrinsics[4 + 1];
        double r0    = intrinsics[4 + 2];
        double r1    = intrinsics[4 + 3];
        double r2    = intrinsics[4 + 4];

        double s_al, c_al, s_be, c_be;
        sincos(alpha, &s_al, &c_al);
        sincos(beta,  &s_be, &c_be);

        // I parametrize the optical axis such that
        // - o(alpha=0, beta=0) = (0,0,1) i.e. the optical axis is at the center
        //   if both parameters are 0
        // - The gradients are cartesian. I.e. do/dalpha and do/dbeta are both
        //   NOT 0 at (alpha=0,beta=0). This would happen at the poles (gimbal
        //   lock), and that would make my solver unhappy
        double o     []         = {  s_al*c_be, s_be,  c_al*c_be };
        double do_dalpha[]      = {  c_al*c_be,    0, -s_al*c_be };
        double do_dbeta[]       = { -s_al*s_be, c_be, -c_al*s_be };

        double norm2p        = norm2_vec(3, p->xyz);
        double omega         = dot_vec(3, p->xyz, o);
        double domega_dalpha = dot_vec(3, p->xyz, do_dalpha);
        double domega_dbeta  = dot_vec(3, p->xyz, do_dbeta);

        double omega_recip = 1.0 / omega;
        double tau         = norm2p * omega_recip*omega_recip - 1.0;
        double s__dtau_dalphabeta__domega_dalphabeta = -2.0*norm2p * omega_recip*omega_recip*omega_recip;
        double dmu_dtau = r1 + 2.0*tau*r2;
        double dmu_dxyz[3];
        for(int i=0; i<3; i++)
            dmu_dxyz[i] = dmu_dtau *
                (2.0 * p->xyz[i] * omega_recip*omega_recip + s__dtau_dalphabeta__domega_dalphabeta * o[i]);
        double mu = r0 + tau*r1 + tau*tau*r2;
        double s__dmu_dalphabeta__domega_dalphabeta = dmu_dtau * s__dtau_dalphabeta__domega_dalphabeta;

        double  dpdistorted_dpcam[3*3] = {};
        double  dpdistorted_ddistortion[3*NdistortionParams];

        for(int i=0; i<3; i++)
        {
            double dmu_ddist[5] = { s__dmu_dalphabeta__domega_dalphabeta * domega_dalpha,
                s__dmu_dalphabeta__domega_dalphabeta * domega_dbeta,
                1.0,
                tau,
                tau * tau };

            dpdistorted_ddistortion[i*NdistortionParams + 0] = p->xyz[i] * dmu_ddist[0];
            dpdistorted_ddistortion[i*NdistortionParams + 1] = p->xyz[i] * dmu_ddist[1];
            dpdistorted_ddistortion[i*NdistortionParams + 2] = p->xyz[i] * dmu_ddist[2];
            dpdistorted_ddistortion[i*NdistortionParams + 3] = p->xyz[i] * dmu_ddist[3];
            dpdistorted_ddistortion[i*NdistortionParams + 4] = p->xyz[i] * dmu_ddist[4];

            dpdistorted_ddistortion[i*NdistortionParams + 0] -= dmu_ddist[0] * omega*o[i];
            dpdistorted_ddistortion[i*NdistortionParams + 1] -= dmu_ddist[1] * omega*o[i];
            dpdistorted_ddistortion[i*NdistortionParams + 2] -= dmu_ddist[2] * omega*o[i];
            dpdistorted_ddistortion[i*NdistortionParams + 3] -= dmu_ddist[3] * omega*o[i];
            dpdistorted_ddistortion[i*NdistortionParams + 4] -= dmu_ddist[4] * omega*o[i];

            dpdistorted_ddistortion[i*NdistortionParams + 0] -= mu * domega_dalpha*o[i];
            dpdistorted_ddistortion[i*NdistortionParams + 1] -= mu * domega_dbeta *o[i];

            dpdistorted_ddistortion[i*NdistortionParams + 0] -= mu * omega * do_dalpha[i];
            dpdistorted_ddistortion[i*NdistortionParams + 1] -= mu * omega * do_dbeta [i];

            dpdistorted_dpcam[3*i + i] = mu+1.0;
            for(int j=0; j<3; j++)
            {
                dpdistorted_dpcam[3*i + j] += (p->xyz[i] - omega*o[i]) * dmu_dxyz[j];
                dpdistorted_dpcam[3*i + j] -= mu*o[i]*o[j];
            }

            p_distorted.xyz[i] = p->xyz[i] + mu * (p->xyz[i] - omega*o[i]);
        }

        // q = fxy pxy/pz + cxy
        // dqx/dp = d( fx px/pz + cx ) = fx/pz^2 (pz [1 0 0] - px [0 0 1])
        // dqy/dp = d( fy py/pz + cy ) = fy/pz^2 (pz [0 1 0] - py [0 0 1])
        const double fx = intrinsics[0];
        const double fy = intrinsics[1];
        const double cx = intrinsics[2];
        const double cy = intrinsics[3];
        double pz_recip = 1. / p_distorted.z;
        q->x = p_distorted.x*pz_recip * fx + cx;
        q->y = p_distorted.y*pz_recip * fy + cy;

        double dq_dp[2][3] =
            { { fx * pz_recip,             0, -fx*p_distorted.x*pz_recip*pz_recip},
              { 0,             fy * pz_recip, -fy*p_distorted.y*pz_recip*pz_recip} };
        // This is for the DISTORTED p.
        // dq/deee = dq/dpdistorted dpdistorted/dpundistorted dpundistorted/deee

        double dq_dpundistorted[6];
        mul_genN3_gen33_vout(2, (double*)dq_dp, dpdistorted_dpcam, dq_dpundistorted);

        // dq/deee = dq/dp dp/deee
        if(camera_at_identity)
        {
            if( dq_drcamera != NULL ) memset(dq_drcamera, 0, 6*sizeof(double));
            if( dq_dtcamera != NULL ) memset(dq_dtcamera, 0, 6*sizeof(double));
            if( dq_drframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dpundistorted, (double*)dp_drf, (double*)dq_drframe);
            if( dq_dtframe  != NULL ) memcpy(dq_dtframe, dq_dpundistorted, 6*sizeof(double));
        }
        else
        {
            if( dq_drcamera != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dpundistorted, (double*)dp_drc, (double*)dq_drcamera);
            if( dq_dtcamera != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dpundistorted, (double*)dp_dtc, (double*)dq_dtcamera);
            if( dq_drframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dpundistorted, (double*)dp_drf, (double*)dq_drframe );
            if( dq_dtframe  != NULL ) mul_genN3_gen33_vout(2, (double*)dq_dpundistorted, (double*)dp_dtf, (double*)dq_dtframe );
        }

        if( dq_dintrinsics_nocore != NULL )
        {
            for(int i=0; i<NdistortionParams; i++)
            {
                const double dx = dpdistorted_ddistortion[i + 0*NdistortionParams];
                const double dy = dpdistorted_ddistortion[i + 1*NdistortionParams];
                const double dz = dpdistorted_ddistortion[i + 2*NdistortionParams];
                dq_dintrinsics_nocore[0*NdistortionParams + i] = fx * pz_recip * (dx - p_distorted.x*pz_recip*dz);
                dq_dintrinsics_nocore[1*NdistortionParams + i] = fy * pz_recip * (dy - p_distorted.y*pz_recip*dz);
            }
        }

        if( dq_dfxy )
        {
            // I have the projection, and I now need to propagate the gradients
            // xy = fxy * distort(xy)/distort(z) + cxy
            dq_dfxy->x = p_distorted.x*pz_recip; // dx/dfx
            dq_dfxy->y = p_distorted.y*pz_recip; // dy/dfy
        }
    }
    else
    {
        MSG("Unhandled lens model: %d (%s)",
            lensmodel.type, mrcal_lensmodel_name(lensmodel));
        assert(0);
    }
}

// Compute a stereographic projection using a constant fxy, cxy. This is the
// same as the pinhole projection for long lenses, but supports views behind the
// camera
void mrcal_project_stereographic( // output
                                 mrcal_point2_t* q,
                                 mrcal_point3_t* dq_dv, // May be NULL. Each point
                                                  // gets a block of 2 mrcal_point3_t
                                                  // objects

                                  // input
                                 const mrcal_point3_t* v,
                                 int N,
                                 double fx, double fy,
                                 double cx, double cy)
{
    // stereographic projection:
    //   (from https://en.wikipedia.org/wiki/Fisheye_lens)
    //   u = xy_unit * tan(th/2) * 2
    //
    // I compute the normalized (focal-length = 1) projection, and
    // use that to look-up the x and y focal length scalings

    // th is the angle between the observation and the projection
    // center
    //
    // sin(th)   = mag_xy/mag_xyz
    // cos(th)   = z/mag_xyz
    // tan(th/2) = sin(th) / (1 + cos(th))

    // tan(th/2) = mag_xy/mag_xyz / (1 + z/mag_xyz) =
    //           = mag_xy / (mag_xyz + z)
    // u = xy_unit * tan(th/2) * 2 =
    //   = xy/mag_xy * mag_xy/(mag_xyz + z) * 2 =
    //   = xy / (mag_xyz + z) * 2
    for(int i=0; i<N; i++)
    {
        double mag_xyz = sqrt( v[i].x*v[i].x +
                               v[i].y*v[i].y +
                               v[i].z*v[i].z );
        double scale = 2.0 / (mag_xyz + v[i].z);

        if(dq_dv)
        {
            // this is more or less already derived in _project_point_splined()
            //
            // dqx/dv = fx ( scale dx + x dscale ) =
            //        = fx ( [1 0 0] scale - 2 x / ()^2 * ( [x y z]/(sqrt) + [0 0 1]) )
            //        = fx ( [scale 0 0] - x scale^2/2 * ( [x y z]/mag_xyz + [0 0 1]) )
            // Let A = -scale^2/2
            //     B = A/mag_xyz
            // dqx_dv = fx ( [scale 0 0] - x scale^2/2 * [x y z]/mag_xyz - x scale^2/2 [0 0 1] )
            //        = fx ( [scale 0 0] + B x * [x y z] + x A [0 0 1] )
            double A = -scale*scale / 2.;
            double B = A / mag_xyz;
            dq_dv[2*i + 0] = (mrcal_point3_t){.x = fx * (v[i].x * (B*v[i].x) + scale),
                                        .y = fx * (v[i].x * (B*v[i].y)),
                                        .z = fx * (v[i].x * (B*v[i].z + A))};
            dq_dv[2*i + 1] = (mrcal_point3_t){.x = fy * (v[i].y * (B*v[i].x)),
                                        .y = fy * (v[i].y * (B*v[i].y) + scale),
                                        .z = fy * (v[i].y * (B*v[i].z + A))};
        }
        q[i] = (mrcal_point2_t){.x = v[i].x * scale * fx + cx,
                          .y = v[i].y * scale * fy + cy};
    }
}

// Compute a stereographic unprojection using a constant fxy, cxy
void mrcal_unproject_stereographic( // output
                                   mrcal_point3_t* v,
                                   mrcal_point2_t* dv_dq, // May be NULL. Each point
                                                    // gets a block of 3
                                                    // mrcal_point2_t objects

                                   // input
                                   const mrcal_point2_t* q,
                                   int N,
                                   double fx, double fy,
                                   double cx, double cy)
{
    // stereographic projection:
    //   (from https://en.wikipedia.org/wiki/Fisheye_lens)
    //   u = xy_unit * tan(th/2) * 2
    //
    // I compute the normalized (focal-length = 1) projection, and
    // use that to look-up the x and y focal length scalings
    //
    // th is the angle between the observation and the projection
    // center
    //
    // sin(th)   = mag_xy/mag_xyz
    // cos(th)   = z/mag_xyz
    // tan(th/2) = sin(th) / (1 + cos(th))
    //
    // tan(th/2) = mag_xy/mag_xyz / (1 + z/mag_xyz) =
    //           = mag_xy / (mag_xyz + z)
    // u = xy_unit * tan(th/2) * 2 =
    //   = xy/mag_xy * mag_xy/(mag_xyz + z) * 2 =
    //   = xy / (mag_xyz + z) * 2
    //
    // How do I compute the inverse?
    //
    // So q = u f + c
    // -> u = (q-c)/f
    // mag(u) = tan(th/2)*2
    //
    // So I can compute th. az comes from the direction of u. This is enough to
    // compute everything. th is in [0,pi].
    //
    //     [ sin(th) cos(az) ]   [ cos(az)   ]
    // v = [ sin(th) sin(az) ] ~ [ sin(az)   ]
    //     [ cos(th)         ]   [ 1/tan(th) ]
    //
    // mag(u) = tan(th/2)*2 -> mag(u)/2 = tan(th/2) ->
    // tan(th) = mag(u) / (1 - (mag(u)/2)^2)
    // 1/tan(th) = (1 - 1/4*mag(u)^2) / mag(u)
    //
    // This has a singularity at u=0 (imager center). But I can scale v to avoid
    // this. So
    //
    //     [ cos(az) mag(u)   ]
    // v = [ sin(az) mag(u)   ]
    //     [ 1 - 1/4*mag(u)^2 ]
    //
    // I can simplify this even more. az = atan2(u.y,u.x). cos(az) = u.x/mag(u) ->
    //
    //     [ u.x              ]
    // v = [ u.y              ]
    //     [ 1 - 1/4 mag(u)^2 ]
    //
    // Test script to confirm that the project/unproject expressions are
    // correct. unproj(proj(v))/v should be a constant
    //
    //     import numpy      as np
    //     import numpysane  as nps
    //     f = 2000
    //     c = 1000
    //     def proj(v):
    //         m = nps.mag(v)
    //         scale = 2.0 / (m + v[..., 2])
    //         u = v[..., :2] * nps.dummy(scale, -1)
    //         return u * f + c
    //     def unproj(q):
    //         u = (q-c)/f
    //         muxy = nps.mag(u[..., :2])
    //         m    = nps.mag(u)
    //         return nps.mv(nps.cat( u[..., 0],
    //                                u[..., 1],
    //                                1 - 1./4.* m*m),
    //                       0, -1)
    //     v = np.array(((1., 2., 3.),
    //                   (3., -2., -4.)))
    //     print( unproj(proj(v)) / v)
    for(int i=0; i<N; i++)
    {
        mrcal_point2_t u = {.x = (q[i].x - cx) / fx,
                            .y = (q[i].y - cy) / fy};

        double norm2u = u.x*u.x + u.y*u.y;
        if(dv_dq)
        {
            dv_dq[3*i + 0] = (mrcal_point2_t){.x = 1.0/fx};
            dv_dq[3*i + 1] = (mrcal_point2_t){.y = 1.0/fy};
            dv_dq[3*i + 2] = (mrcal_point2_t){.x = -u.x/2.0/fx,
                                        .y = -u.y/2.0/fy};
        }
        v[i] = (mrcal_point3_t){ .x = u.x,
                                 .y = u.y,
                                 .z = 1. - 1./4. * norm2u };
    }
}

static void _mrcal_precompute_lensmodel_data_MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC
  ( // output
    mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed_t* precomputed,

    //input
    const mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config )
{
    // I have N control points describing a given field-of-view. I
    // want to space out the control points evenly. I'm using
    // B-splines, so I need extra control points out past my edge.
    // With cubic splines I need a whole extra interval past the
    // edge. With quadratic splines I need half an interval (see
    // stuff in analyses/splines/).
    //
    // (width + k*interval_size)/(N-1) = interval_size
    // ---> width/(N-1) = interval_size * (1 - k/(N-1))
    // ---> interval_size = width / (N - 1 - k)
    int NextraIntervals;
    if(config->order == 2)
    {
        NextraIntervals = 1;
        if(config->Nx < 3 || config->Ny < 3)
        {
            MSG("Quadratic splines: absolute minimum Nx, Ny is 3. Got Nx=%d Ny=%d. Barfing out",
                config->Nx, config->Ny);
            assert(0);
        }
    }
    else if(config->order == 3)
    {
        NextraIntervals = 2;
        if(config->Nx < 4 || config->Ny < 4)
        {
            MSG("Cubic splines: absolute minimum Nx, Ny is 4. Got Nx=%d Ny=%d. Barfing out",
                config->Nx, config->Ny);
            assert(0);
        }
    }
    else
    {
        MSG("I only support spline order 2 and 3");
        assert(0);
    }

    double th_fov_x_edge = (double)config->fov_x_deg/2. * M_PI / 180.;
    double u_edge_x      = tan(th_fov_x_edge / 2.) * 2;
    precomputed->segments_per_u = (config->Nx - 1 - NextraIntervals) / (u_edge_x*2.);
}

// NOT A PART OF THE EXTERNAL API. This is exported for the mrcal python wrapper
// only
void _mrcal_precompute_lensmodel_data(mrcal_projection_precomputed_t* precomputed,
                                      mrcal_lensmodel_t lensmodel)
{
    // currently only this model has anything
    if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
        _mrcal_precompute_lensmodel_data_MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC
            ( &precomputed->LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed,
              &lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config );
    precomputed->ready = true;
}

bool mrcal_knots_for_splined_models( // buffers must hold at least
                                     // config->Nx and config->Ny values
                                     // respectively
                                     double* ux, double* uy,
                                     mrcal_lensmodel_t lensmodel)
{
    if(lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        MSG("This function works only with the MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC model. '%s' passed in",
            mrcal_lensmodel_name(lensmodel));
        return false;
    }

    mrcal_projection_precomputed_t precomputed_all;
    _mrcal_precompute_lensmodel_data(&precomputed_all, lensmodel);

    mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config =
        &lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config;
    mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed_t* precomputed =
        &precomputed_all.LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed;

    // The logic I'm reversing is
    //     double ix = u.x*segments_per_u + (double)(Nx-1)/2.;
    for(int i=0; i<config->Nx; i++)
        ux[i] =
            ((double)i - (double)(config->Nx-1)/2.) /
            precomputed->segments_per_u;
    for(int i=0; i<config->Ny; i++)
        uy[i] =
            ((double)i - (double)(config->Ny-1)/2.) /
            precomputed->segments_per_u;
    return true;
}

static
void _project_point_splined( // outputs
                            mrcal_point2_t* q,
                            mrcal_point2_t* dq_dfxy,

                            double* grad_ABCDx_ABCDy,
                            int* ivar0,

                            // Gradient outputs. May be NULL
                            mrcal_point3_t* restrict dq_drcamera,
                            mrcal_point3_t* restrict dq_dtcamera,
                            mrcal_point3_t* restrict dq_drframe,
                            mrcal_point3_t* restrict dq_dtframe,

                            // inputs
                            const mrcal_point3_t* restrict p,
                            const mrcal_point3_t* restrict dp_drc,
                            const mrcal_point3_t* restrict dp_dtc,
                            const mrcal_point3_t* restrict dp_drf,
                            const mrcal_point3_t* restrict dp_dtf,

                            const double* restrict intrinsics,
                            bool camera_at_identity,
                            int spline_order,
                            uint16_t Nx, uint16_t Ny,
                            double segments_per_u)
{
    // projections out-of-bounds will yield SOME value (function remains
    // continuous as we move out-of-bounds), but it wont be particularly
    // meaningful


    // stereographic projection:
    //   (from https://en.wikipedia.org/wiki/Fisheye_lens)
    //   u = xy_unit * tan(th/2) * 2
    //
    // I compute the normalized (focal-length = 1) projection, and
    // use that to look-up deltau

    // th is the angle between the observation and the projection
    // center
    //
    // sin(th)   = mag_pxy/mag_p
    // cos(th)   = z/mag_p
    // tan(th/2) = sin(th) / (1 + cos(th))

    // tan(th/2) = mag_pxy/mag_p / (1 + z/mag_p) =
    //           = mag_pxy / (mag_p + z)
    // u = xy_unit * tan(th/2) * 2 =
    //   = xy/mag_pxy * mag_pxy/(mag_p + z) * 2 =
    //   = xy / (mag_p + z) * 2
    //
    // The stereographic projection is used to query the spline surface, and it
    // is also the projection baseline. I do
    //
    //   q = (u + deltau(u)) * f + c
    //
    // If the spline surface is at 0 (deltau == 0) then this is a pure
    // stereographic projection
    double mag_p = sqrt( p->x*p->x +
                         p->y*p->y +
                         p->z*p->z );
    double scale = 2.0 / (mag_p + p->z);

    mrcal_point2_t u = {.x = p->x * scale,
                        .y = p->y * scale};
    // du/dp = d/dp ( xy * scale )
    //       = pxy dscale/dp + [I; 0] scale
    // dscale/dp = d (2.0 / (mag_p + p->z))/dp =
    //           = -2/()^2 ( [0,0,1] + dmag/dp)
    //           = -2/()^2 ( [0,0,1] + 2pt/2mag)
    //           = -2 scale^2/4 ( [0,0,1] + pt/mag)
    //           = -scale^2/2 * ( [0,0,1] + pt/mag )
    //           = A*[0,0,1] + B*pt
    double A = -scale*scale / 2.;
    double B = A / mag_p;
    double du_dp[2][3] = { { p->x * (B * p->x)      + scale,
                             p->x * (B * p->y),
                             p->x * (B * p->z + A) },
                           { p->y * (B * p->x),
                             p->y * (B * p->y)      + scale,
                             p->y * (B * p->z + A) } };

    double ix = u.x*segments_per_u + (double)(Nx-1)/2.;
    double iy = u.y*segments_per_u + (double)(Ny-1)/2.;

    mrcal_point2_t deltau;
    double ddeltau_dux[2];
    double ddeltau_duy[2];
    const double fx = intrinsics[0];
    const double fy = intrinsics[1];
    const double cx = intrinsics[2];
    const double cy = intrinsics[3];

    bool need_any_dq_drt =
        !( dq_drcamera == NULL &&
           dq_dtcamera == NULL &&
           dq_drframe  == NULL &&
           dq_dtframe  == NULL );

    if( spline_order == 3 )
    {
        int ix0 = (int)ix;
        int iy0 = (int)iy;

        // If out-of-bounds, clamp to the nearest valid spline segment. The
        // projection will fly off to infinity quickly (we're extrapolating a
        // polynomial), but at least it'll stay continuous
        if(     ix0 < 1)    ix0 = 1;
        else if(ix0 > Nx-3) ix0 = Nx-3;
        if(     iy0 < 1)    iy0 = 1;
        else if(iy0 > Ny-3) iy0 = Ny-3;

        *ivar0 =
            4 + // skip the core
            2*( (iy0-1)*Nx +
                (ix0-1) );

        sample_bspline_surface_cubic(deltau.xy,
                                     need_any_dq_drt ? ddeltau_dux : NULL,
                                     need_any_dq_drt ? ddeltau_duy : NULL,
                                     grad_ABCDx_ABCDy,
                                     ix - ix0, iy - iy0,

                                     // control points
                                     &intrinsics[*ivar0],
                                     2*Nx);
    }
    else if( spline_order == 2 )
    {
        int ix0 = (int)(ix + 0.5);
        int iy0 = (int)(iy + 0.5);

        // If out-of-bounds, clamp to the nearest valid spline segment. The
        // projection will fly off to infinity quickly (we're extrapolating a
        // polynomial), but at least it'll stay continuous
        if(     ix0 < 1)    ix0 = 1;
        else if(ix0 > Nx-2) ix0 = Nx-2;
        if(     iy0 < 1)    iy0 = 1;
        else if(iy0 > Ny-2) iy0 = Ny-2;

        *ivar0 =
            4 + // skip the core
            2*( (iy0-1)*Nx +
                (ix0-1) );

        sample_bspline_surface_quadratic(deltau.xy,
                                         need_any_dq_drt ? ddeltau_dux : NULL,
                                         need_any_dq_drt ? ddeltau_duy : NULL,
                                         grad_ABCDx_ABCDy,
                                         ix - ix0, iy - iy0,

                                         // control points
                                         &intrinsics[*ivar0],
                                         2*Nx);
    }
    else
    {
        MSG("I only support spline order==2 or 3. Somehow got %d. This is a bug. Barfing",
            spline_order);
        assert(0);
    }

    // u = stereographic(p)
    // q = (u + deltau(u)) * f + c
    //
    // Extrinsics:
    //   dqx/deee = fx (dux/deee + ddeltaux/du du/deee)
    //            = fx ( dux/deee (1 + ddeltaux/dux) + ddeltaux/duy duy/deee)
    //   dqy/deee = fy ( duy/deee (1 + ddeltauy/duy) + ddeltauy/dux dux/deee)
    q->x = (u.x + deltau.x) * fx + cx;
    q->y = (u.y + deltau.y) * fy + cy;

    if( dq_dfxy )
    {
        // I have the projection, and I now need to propagate the gradients
        // xy = fxy * distort(xy)/distort(z) + cxy
        dq_dfxy->x = u.x + deltau.x;
        dq_dfxy->y = u.y + deltau.y;
    }

    if(!need_any_dq_drt) return;


    // convert ddeltau_dixy to ddeltau_duxy
    for(int i=0; i<2; i++)
    {
        ddeltau_dux[i] *= segments_per_u;
        ddeltau_duy[i] *= segments_per_u;
    }

    void propagate_extrinsics( mrcal_point3_t* dq_deee,
                               const mrcal_point3_t* dp_deee)
    {
        mrcal_point3_t du_deee[2];
        mul_genN3_gen33_vout(2, (double*)du_dp, (double*)dp_deee, (double*)du_deee);

        for(int i=0; i<3; i++)
        {
            dq_deee[0].xyz[i] =
                fx *
                ( du_deee[0].xyz[i] * (1. + ddeltau_dux[0]) +
                  ddeltau_duy[0] * du_deee[1].xyz[i]);
            dq_deee[1].xyz[i] =
                fy *
                ( du_deee[1].xyz[i] * (1. + ddeltau_duy[1]) +
                  ddeltau_dux[1] * du_deee[0].xyz[i]);
        }
    }
    void propagate_extrinsics_cam0( mrcal_point3_t* dq_deee)
    {
        for(int i=0; i<3; i++)
        {
            dq_deee[0].xyz[i] =
                fx *
                ( du_dp[0][i] * (1. + ddeltau_dux[0]) +
                  ddeltau_duy[0] * du_dp[1][i]);
            dq_deee[1].xyz[i] =
                fy *
                ( du_dp[1][i] * (1. + ddeltau_duy[1]) +
                  ddeltau_dux[1] * du_dp[0][i]);
        }
    }
    if(camera_at_identity)
    {
        if( dq_drcamera != NULL ) memset(dq_drcamera->xyz, 0, 6*sizeof(double));
        if( dq_dtcamera != NULL ) memset(dq_dtcamera->xyz, 0, 6*sizeof(double));
        if( dq_drframe  != NULL ) propagate_extrinsics( dq_drframe,  dp_drf );
        if( dq_dtframe  != NULL ) propagate_extrinsics_cam0( dq_dtframe );
    }
    else
    {
        if( dq_drcamera != NULL ) propagate_extrinsics( dq_drcamera, dp_drc );
        if( dq_dtcamera != NULL ) propagate_extrinsics( dq_dtcamera, dp_dtc );
        if( dq_drframe  != NULL ) propagate_extrinsics( dq_drframe,  dp_drf );
        if( dq_dtframe  != NULL ) propagate_extrinsics( dq_dtframe,  dp_dtf );
    }
}

typedef struct
{
    double* pool;
    uint16_t run_side_length;
    uint16_t ivar_stridey;
} gradient_sparse_meta_t;

// Projects 3D point(s), and reports the projection, and all the gradients. This
// is the main internal callback in the optimizer. This operates in one of two modes:
//
// if(calibration_object_width_n == 0) then we're projecting ONE point. In world
// coords this point is at frame_rt->t. frame_rt->r is not referenced. q and the
// gradients reference 2 values (x,y in the imager)
//
// if(calibration_object_width_n > 0) then we're projecting a whole calibration
// object. The pose of this object is given in frame_rt. We project ALL
// calibration_object_width_n*calibration_object_height_n points. q and the
// gradients reference ALL of these points
static
void project( // out
             mrcal_point2_t* restrict q,

             // The intrinsics gradients. These are split among several arrays.
             // High-parameter-count lens models can return their gradients
             // sparsely. All the actual gradient values live in
             // dq_dintrinsics_pool_double, a buffer supplied by the caller. If
             // dq_dintrinsics_pool_double is not NULL, the rest of the
             // variables are assumed non-NULL, and we compute all the
             // intrinsics gradients. Conversely, if dq_dintrinsics_pool_double
             // is NULL, no intrinsics gradients are computed
             double*  restrict dq_dintrinsics_pool_double,
             int*     restrict dq_dintrinsics_pool_int,
             double** restrict dq_dfxy,
             double** restrict dq_dintrinsics_nocore,
             gradient_sparse_meta_t* gradient_sparse_meta,
             mrcal_point3_t* restrict dq_drcamera,
             mrcal_point3_t* restrict dq_dtcamera,
             mrcal_point3_t* restrict dq_drframe,
             mrcal_point3_t* restrict dq_dtframe,
             mrcal_point2_t* restrict dq_dcalobject_warp,

             // in

             // everything; includes the core, if there is one
             const double* restrict intrinsics,
             const mrcal_pose_t* restrict camera_rt,
             const mrcal_pose_t* restrict frame_rt,
             const mrcal_point2_t* restrict calobject_warp,

             bool camera_at_identity, // if true, camera_rt is unused
             mrcal_lensmodel_t lensmodel,
             const mrcal_projection_precomputed_t* precomputed,

             double calibration_object_spacing,
             int    calibration_object_width_n,
             int    calibration_object_height_n)
{
    assert(precomputed->ready);

    // Parametric and non-parametric models do different things:
    //
    // parametric models:
    //   u = distort(p, distortions)
    //   q = uxy/uz * fxy + cxy
    //
    //   extrinsic gradients:
    //       dqx/deee = d( ux/uz * fx + cx)/deee =
    //                = fx d(ux/uz)/deee =
    //                = fx/uz^2 ( uz dux/deee - duz/deee ux )
    //
    // nonparametric (splined) models
    //   u = stereographic(p)
    //   q = (u + deltau(u)) * f + c
    //
    //   Extrinsics:
    //     dqx/deee = fx (dux/deee + ddeltaux/du du/deee)
    //              = fx ( dux/deee (1 + ddeltaux/dux) + ddeltaux/duy duy/deee)
    //     dqy/deee = fy ( duy/deee (1 + ddeltauy/duy) + ddeltauy/dux dux/deee)
    //
    //   Intrinsics:
    //     dq/diii = f ddeltau/diii
    //
    // So the two kinds of models have completely different expressions for
    // their gradients, and I implement them separately

    const int Npoints =
        calibration_object_width_n ?
        calibration_object_width_n*calibration_object_height_n : 1;
    const int Nintrinsics = mrcal_lensmodel_num_params(lensmodel);

    // I need to compose two transformations
    //
    // (object in reference frame) -> [frame transform] -> (object in the reference frame) ->
    // -> [camera rt] -> (camera frame)
    //
    // Note that here the frame transform transforms TO the reference frame and
    // the camera transform transforms FROM the reference frame. This is how my
    // data is expected to be set up
    //
    // [Rc tc] [Rf tf] = [Rc*Rf  Rc*tf + tc]
    // [0  1 ] [0  1 ]   [0      1         ]
    //
    // This transformation (and its gradients) is handled by mrcal_compose_rt()
    // I refer to the camera*frame transform as the "joint" transform, or the
    // letter j
    geometric_gradients_t gg;

    double _joint_rt[6];
    double* joint_rt;

    // The caller has an odd-looking array reference [-3]. This is intended, but
    // the compiler throws a warning. I silence it here. gcc-10 produces a very
    // strange-looking warning that was reported to the maintainers:
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=97261
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
    mrcal_pose_t frame_rt_validr = {.t = frame_rt->t};
#pragma GCC diagnostic pop
    if(calibration_object_width_n) frame_rt_validr.r = frame_rt->r;

    if(!camera_at_identity)
    {
        // make sure I can pass mrcal_pose_t.r as an rt[] transformation
        static_assert( offsetof(mrcal_pose_t, r) == 0,                   "mrcal_pose_t has expected structure");
        static_assert( offsetof(mrcal_pose_t, t) == 3*sizeof(double),    "mrcal_pose_t has expected structure");
        mrcal_compose_rt( _joint_rt,
                          gg._d_rj_rc, gg._d_rj_rf,
                          gg._d_tj_rc, gg._d_tj_tf,
                          camera_rt     ->r.xyz,
                          frame_rt_validr.r.xyz);
        joint_rt = _joint_rt;
    }
    else
    {
        // We're looking at the reference frame, so this camera transform is
        // fixed at the identity. We don't need to compose anything, nor
        // propagate gradients for the camera extrinsics, since those don't
        // exist in the parameter vector

        // Here the "joint" transform is just the "frame" transform
        joint_rt = frame_rt_validr.r.xyz;
    }

    // Not using OpenCV distortions, the distortion and projection are not
    // coupled
    double Rj[3*3];
    double d_Rj_rj[9*3];

    mrcal_R_from_r(Rj, d_Rj_rj, joint_rt);

    mrcal_point2_t* p_dq_dfxy                  = NULL;
    double*   p_dq_dintrinsics_nocore    = NULL;
    bool      has_core                   = modelHasCore_fxfycxcy(lensmodel);
    bool      has_dense_intrinsics_grad  = (lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC);
    bool      has_sparse_intrinsics_grad = (lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC);

    if(dq_dintrinsics_pool_double != NULL)
    {
        // nothing by default
        *dq_dfxy                   = NULL;
        *dq_dintrinsics_nocore     = NULL;
        gradient_sparse_meta->pool = NULL;
        int ivar_pool = 0;

        if(has_core)
        {
            // Each point produces 2 measurements. Each one depends on ONE fxy
            // element. So Npoints*2 of these
            *dq_dfxy  = &dq_dintrinsics_pool_double[ivar_pool];
            p_dq_dfxy = (mrcal_point2_t*)*dq_dfxy;
            ivar_pool += Npoints*2;
        }
        if(has_dense_intrinsics_grad)
        {
            *dq_dintrinsics_nocore = p_dq_dintrinsics_nocore = &dq_dintrinsics_pool_double[ivar_pool];
            ivar_pool += Npoints*2 * (Nintrinsics-4);
        }
        if(has_sparse_intrinsics_grad)
        {
            if(lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
            {
                MSG("Unhandled lens model: %d (%s)",
                    lensmodel.type,
                    mrcal_lensmodel_name(lensmodel));
                assert(0);
            }
            const mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config =
                &lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config;
            *gradient_sparse_meta =
                (gradient_sparse_meta_t)
                {
                    .run_side_length = config->order+1,
                    .ivar_stridey    = 2*config->Nx,
                    .pool            = &dq_dintrinsics_pool_double[ivar_pool]
                };
        }
    }

    // These are produced by propagate_extrinsics() and consumed by
    // project_point()
    mrcal_point3_t _dp_drc[3];
    mrcal_point3_t _dp_dtc[3];
    mrcal_point3_t _dp_drf[3];
    mrcal_point3_t _dp_dtf[3];
    mrcal_point3_t* dp_drc;
    mrcal_point3_t* dp_dtc;
    mrcal_point3_t* dp_drf;
    mrcal_point3_t* dp_dtf;

    mrcal_point3_t propagate_extrinsics( const mrcal_point3_t* pt_ref,
                                   const geometric_gradients_t* gg,
                                   const double* Rj, const double* d_Rj_rj,
                                   const double* _tj )
    {
        // Rj * pt + tj -> pt
        mrcal_point3_t p;
        mul_vec3_gen33t_vout(pt_ref->xyz, Rj, p.xyz);
        add_vec(3, p.xyz,  _tj);

        void propagate_extrinsics_one(mrcal_point3_t* dp_dparam,
                                      const double* drj_dparam,
                                      const double* dtj_dparam,
                                      const double* d_Rj_rj)
        {
            // dRj[row0]/drj is 3x3 matrix at &d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc
            for(int i=0; i<3; i++)
            {
                mul_vec3_gen33_vout( pt_ref->xyz, &d_Rj_rj[9*i], dp_dparam[i].xyz );
                mul_vec3_gen33     ( dp_dparam[i].xyz,   drj_dparam);
                add_vec(3, dp_dparam[i].xyz, &dtj_dparam[3*i] );
            }
        }
        void propagate_extrinsics_one_rzero(mrcal_point3_t* dp_dparam,
                                            const double* dtj_dparam,
                                            const double* d_Rj_rj)
        {
            // dRj[row0]/drj is 3x3 matrix at &d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc
            memcpy(dp_dparam->xyz, dtj_dparam, 9*sizeof(double));
        }
        void propagate_extrinsics_one_tzero(mrcal_point3_t* dp_dparam,
                                            const double* drj_dparam,
                                            const double* d_Rj_rj)
        {
            // dRj[row0]/drj is 3x3 matrix at &d_Rj_rj[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc
            for(int i=0; i<3; i++)
            {
                mul_vec3_gen33_vout( pt_ref->xyz, &d_Rj_rj[9*i], dp_dparam[i].xyz );
                mul_vec3_gen33     ( dp_dparam[i].xyz,   drj_dparam);
            }
        }
        void propagate_extrinsics_one_rzero_tidentity(mrcal_point3_t* dp_dparam,
                                                      const double* d_Rj_rj)
        {
            dp_dparam[0] = (mrcal_point3_t){.x = 1.0};
            dp_dparam[1] = (mrcal_point3_t){.y = 1.0};
            dp_dparam[2] = (mrcal_point3_t){.z = 1.0};
        }

        void propagate_extrinsics_one_cam0(mrcal_point3_t* dp_rf,
                                           const double* _d_Rf_rf)
        {
            // dRj[row0]/drj is 3x3 matrix at &_d_Rf_rf[0]
            // dRj[row0]/drc = dRj[row0]/drj * drj_drc
            for(int i=0; i<3; i++)
                mul_vec3_gen33_vout( pt_ref->xyz, &_d_Rf_rf[9*i], dp_rf[i].xyz );
        }
        if(gg != NULL)
        {
            propagate_extrinsics_one(                _dp_drc, gg->_d_rj_rc, gg->_d_tj_rc, d_Rj_rj);
            propagate_extrinsics_one_rzero_tidentity(_dp_dtc,                             d_Rj_rj);
            propagate_extrinsics_one_tzero(          _dp_drf, gg->_d_rj_rf,               d_Rj_rj);
            propagate_extrinsics_one_rzero(          _dp_dtf,               gg->_d_tj_tf, d_Rj_rj);
            dp_drc = _dp_drc;
            dp_dtc = _dp_dtc;
            dp_drf = _dp_drf;
            dp_dtf = _dp_dtf;
        }
        else
        {
            // camera is at the reference. The "joint" coord system is the "frame"
            // coord system
            //
            //   p_cam = Rf p_ref + tf
            //
            // dp/drc = 0
            // dp/dtc = 0
            // dp/drf = reshape(dRf_drf p_ref)
            // dp/dtf = I
            propagate_extrinsics_one_cam0(_dp_drf, d_Rj_rj);

            dp_drc = NULL;
            dp_dtc = NULL;
            dp_drf = _dp_drf;
            dp_dtf = NULL; // this is I. The user of this MUST know to interpret
            // it that way
        }
        return p;
    }

    void project_point( // outputs
                       mrcal_point2_t* q,
                       mrcal_point2_t* p_dq_dfxy,
                       double* p_dq_dintrinsics_nocore,
                       double* gradient_sparse_meta_pool,
                       int runlen,
                       mrcal_point3_t* restrict dq_drcamera,
                       mrcal_point3_t* restrict dq_dtcamera,
                       mrcal_point3_t* restrict dq_drframe,
                       mrcal_point3_t* restrict dq_dtframe,
                       mrcal_point2_t* restrict dq_dcalobject_warp,
                       // inputs
                       const mrcal_point3_t* p,
                       const double* restrict intrinsics,
                       mrcal_lensmodel_t lensmodel,
                       const mrcal_point2_t* dpt_ref2_dwarp,

                       // if NULL then the camera is at the reference
                       bool camera_at_identity,
                       const double* Rj)
    {
        if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
        {
            // only need 3+3 for quadratic splines
            double grad_ABCDx_ABCDy[4+4];
            int ivar0;

            _project_point_splined( // outputs
                                   q, p_dq_dfxy,
                                   grad_ABCDx_ABCDy,
                                   &ivar0,

                                   dq_drcamera,dq_dtcamera,dq_drframe,dq_dtframe,
                                   // inputs
                                   p,
                                   dp_drc, dp_dtc, dp_drf, dp_dtf,
                                   intrinsics,
                                   camera_at_identity,
                                   lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.order,
                                   lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx,
                                   lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny,
                                   precomputed->LENSMODEL_SPLINED_STEREOGRAPHIC__precomputed.segments_per_u);
            if(dq_dintrinsics_pool_int != NULL)
            {
                *(dq_dintrinsics_pool_int++) = ivar0;
                memcpy(gradient_sparse_meta_pool,
                       grad_ABCDx_ABCDy,
                       sizeof(double)*runlen*2);
            }
        }
        else
        {
            _project_point_parametric( // outputs
                                      q,p_dq_dfxy,
                                      p_dq_dintrinsics_nocore,
                                      dq_drcamera,dq_dtcamera,dq_drframe,dq_dtframe,
                                      // inputs
                                      p,
                                      dp_drc, dp_dtc, dp_drf, dp_dtf,
                                      intrinsics,
                                      camera_at_identity,
                                      lensmodel);
        }

        if( dq_dcalobject_warp != NULL && dpt_ref2_dwarp != NULL )
        {
            // p = proj(Rc Rf warp(x) + Rc tf + tc);
            // dp/dw = dp/dRcRf(warp(x)) dR(warp(x))/dwarp(x) dwarp/dw =
            //       = dp/dtc RcRf dwarp/dw
            // dp/dtc is dq_dtcamera
            // R is rodrigues(rj)
            // dwarp/dw = [0 0]
            //            [0 0]
            //            [a b]
            // Let R = [r0 r1 r2]
            // dp/dw = dp/dt [ar2 br2] = [a dp/dt r2    b dp/dt r2]
            mrcal_point3_t* p_dq_dt;
            if(!camera_at_identity) p_dq_dt = dq_dtcamera;
            else                    p_dq_dt = dq_dtframe;
            double d[] =
                { p_dq_dt[0].xyz[0] * Rj[0*3 + 2] +
                  p_dq_dt[0].xyz[1] * Rj[1*3 + 2] +
                  p_dq_dt[0].xyz[2] * Rj[2*3 + 2],
                  p_dq_dt[1].xyz[0] * Rj[0*3 + 2] +
                  p_dq_dt[1].xyz[1] * Rj[1*3 + 2] +
                  p_dq_dt[1].xyz[2] * Rj[2*3 + 2]};

            dq_dcalobject_warp[0].x = d[0]*dpt_ref2_dwarp->x;
            dq_dcalobject_warp[0].y = d[0]*dpt_ref2_dwarp->y;
            dq_dcalobject_warp[1].x = d[1]*dpt_ref2_dwarp->x;
            dq_dcalobject_warp[1].y = d[1]*dpt_ref2_dwarp->y;
        }
    }



    int runlen = (lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC) ?
        (lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.order + 1) :
        0;
    if( calibration_object_width_n == 0 )
    { // projecting discrete points
        mrcal_point3_t p =
            propagate_extrinsics( &(mrcal_point3_t){},
                                  camera_at_identity ? NULL : &gg,
                                  Rj, d_Rj_rj, &joint_rt[3]);
        project_point(  q,
                        p_dq_dfxy, p_dq_dintrinsics_nocore,
                        gradient_sparse_meta ? gradient_sparse_meta->pool : NULL,
                        runlen,
                        dq_drcamera, dq_dtcamera, dq_drframe, dq_dtframe, NULL,

                        &p,
                        intrinsics, lensmodel,
                        NULL,
                        camera_at_identity, Rj);
    }
    else
    { // projecting a chessboard
        int i_pt = 0;
         // The calibration object has a simple grid geometry
        for(int y = 0; y<calibration_object_height_n; y++)
            for(int x = 0; x<calibration_object_width_n; x++)
            {
                mrcal_point3_t pt_ref = {.x = (double)x * calibration_object_spacing,
                                   .y = (double)y * calibration_object_spacing};
                mrcal_point2_t dpt_ref2_dwarp = {};

                if(calobject_warp != NULL)
                {
                    // Add a board warp here. I have two parameters, and they describe
                    // additive flex along the x axis and along the y axis, in that
                    // order. In each direction the flex is a parabola, with the
                    // parameter k describing the max deflection at the center. If the
                    // ends are at +- 1 I have d = k*(1 - x^2). If the ends are at
                    // (0,N-1) the equivalent expression is: d = k*( 1 - 4*x^2/(N-1)^2 +
                    // 4*x/(N-1) - 1 ) = d = 4*k*(x/(N-1) - x^2/(N-1)^2) = d =
                    // 4.*k*x*r(1. - x*r)
                    double xr = (double)x / (double)(calibration_object_width_n -1);
                    double yr = (double)y / (double)(calibration_object_height_n-1);
                    double dx = 4. * xr * (1. - xr);
                    double dy = 4. * yr * (1. - yr);
                    pt_ref.z += calobject_warp->x * dx;
                    pt_ref.z += calobject_warp->y * dy;
                    dpt_ref2_dwarp.x = dx;
                    dpt_ref2_dwarp.y = dy;
                }

                mrcal_point3_t p =
                    propagate_extrinsics( &pt_ref,
                                          camera_at_identity ? NULL : &gg,
                                          Rj, d_Rj_rj, &joint_rt[3]);

                mrcal_point3_t* dq_drcamera_here        = dq_drcamera        ? &dq_drcamera        [2*i_pt] : NULL;
                mrcal_point3_t* dq_dtcamera_here        = dq_dtcamera        ? &dq_dtcamera        [2*i_pt] : NULL;
                mrcal_point3_t* dq_drframe_here         = dq_drframe         ? &dq_drframe         [2*i_pt] : NULL;
                mrcal_point3_t* dq_dtframe_here         = dq_dtframe         ? &dq_dtframe         [2*i_pt] : NULL;
                mrcal_point2_t* dq_dcalobject_warp_here = dq_dcalobject_warp ? &dq_dcalobject_warp [2*i_pt] : NULL;

                mrcal_point3_t dq_dtcamera_here_dummy[2];
                mrcal_point3_t dq_dtframe_here_dummy [2];
                if(dq_dcalobject_warp)
                {
                    // I need all translation gradients to be available to
                    // compute the calobject_warp gradients (see the end of the
                    // project_point() function above). So I compute those even
                    // if the caller didn't ask for them
                    if(!dq_dtcamera_here) dq_dtcamera_here = dq_dtcamera_here_dummy;
                    if(!dq_dtframe_here)  dq_dtframe_here  = dq_dtframe_here_dummy;
                }

                project_point(&q[i_pt],
                              p_dq_dfxy ? &p_dq_dfxy[i_pt] : NULL,
                              p_dq_dintrinsics_nocore ? &p_dq_dintrinsics_nocore[2*(Nintrinsics-4)*i_pt] : NULL,
                              gradient_sparse_meta ? &gradient_sparse_meta->pool[i_pt*runlen*2] : NULL,
                              runlen,
                              dq_drcamera_here, dq_dtcamera_here, dq_drframe_here, dq_dtframe_here, dq_dcalobject_warp_here,
                              &p,
                              intrinsics, lensmodel,
                              &dpt_ref2_dwarp,
                              camera_at_identity, Rj);
                i_pt++;
            }
    }
}

// NOT A PART OF THE EXTERNAL API. This is exported for the mrcal python wrapper
// only
bool _mrcal_project_internal_cahvore( // out
                                     mrcal_point2_t* out,

                                     // in
                                     const mrcal_point3_t* v,
                                     int N,

                                     // core, distortions concatenated
                                     const double* intrinsics)
{
    // Apply a CAHVORE warp to an un-distorted point

    //  Given intrinsic parameters of a CAHVORE model and a set of
    //  camera-coordinate points, return the projected point(s)

    // This comes from cmod_cahvore_3d_to_2d_general() in
    // m-jplv/libcmod/cmod_cahvore.c
    //
    // The lack of documentation here comes directly from the lack of
    // documentation in that function.

    // I parametrize the optical axis such that
    // - o(alpha=0, beta=0) = (0,0,1) i.e. the optical axis is at the center
    //   if both parameters are 0
    // - The gradients are cartesian. I.e. do/dalpha and do/dbeta are both
    //   NOT 0 at (alpha=0,beta=0). This would happen at the poles (gimbal
    //   lock), and that would make my solver unhappy
    // So o = { s_al*c_be, s_be,  c_al*c_be }
    const mrcal_intrinsics_core_t* core = (const mrcal_intrinsics_core_t*)intrinsics;
    const double alpha     = intrinsics[4 + 0];
    const double beta      = intrinsics[4 + 1];
    const double r0        = intrinsics[4 + 2];
    const double r1        = intrinsics[4 + 3];
    const double r2        = intrinsics[4 + 4];
    const double e0        = intrinsics[4 + 5];
    const double e1        = intrinsics[4 + 6];
    const double e2        = intrinsics[4 + 7];
    const double linearity = intrinsics[4 + 8];

    double sa,ca;
    sincos(alpha, &sa, &ca);
    double sb,cb;
    sincos(beta, &sb, &cb);

    const double o[] ={ cb * sa, sb, cb * ca };

    for(int i_pt=0; i_pt<N; i_pt++)
    {
        ///////////////// THIS IS MADE UP, AND PROBABLY WRONG I'm using jplv as
        // the reference implementation for this, but that implementation can't
        // work. In jplv project(v) and project(k*v) don't project to the sample
        // point. Look at the definition of upsilon below. omega and l have
        // units of m, while the other terms are unitless. I'm hypothesizing
        // that they meant to normalize v, but never did it. So I'm going that
        // here. mrcal supports cahvore only for compatibility, so nobody's
        // using this code. IF YOU ARE GOING TO USE THIS CODE, PLEASE CONFIRM
        // THAT THIS CAHVORE PROJECTION IS CORRECT
        double vhere[3] = {
            v[i_pt].x,
            v[i_pt].y,
            v[i_pt].z
        };
        double vnorm = sqrt(vhere[0]*vhere[0] +
                            vhere[1]*vhere[1] +
                            vhere[2]*vhere[2]);
        for(int i=0; i<3; i++) vhere[i] /= vnorm;

        // cos( angle between v and o ) = inner(v,o) / (norm(o) * norm(v)) =
        // omega/norm(v)
        double omega = vhere[0]*o[0] + vhere[1]*o[1] + vhere[2]*o[2];


        // Basic Computations

        // Calculate initial terms
        double u[3];
        for(int i=0; i<3; i++) u[i] = omega*o[i];

        double ll[3];
        for(int i=0; i<3; i++) ll[i] = vhere[i]-u[i];
        double l = sqrt(ll[0]*ll[0] + ll[1]*ll[1] + ll[2]*ll[2]);

        // Calculate theta using Newton's Method
        double theta = atan2(l, omega);

        int inewton;
        for( inewton = 100; inewton; inewton--)
        {
            // Compute terms from the current value of theta
            double sth,cth;
            sincos(theta, &sth, &cth);

            double theta2  = theta * theta;
            double theta3  = theta * theta2;
            double theta4  = theta * theta3;
            double upsilon =
                omega*cth + l*sth
                - (1.0   - cth) * (e0 +      e1*theta2 +     e2*theta4)
                - (theta - sth) * (      2.0*e1*theta  + 4.0*e2*theta3);

            // Update theta
            double dtheta =
                (
                 omega*sth - l*cth
                 - (theta - sth) * (e0 + e1*theta2 + e2*theta4)
                 ) / upsilon;

            theta -= dtheta;

            // Check exit criterion from last update
            if(fabs(dtheta) < 1e-8)
                break;
        }
        if(inewton == 0)
        {
            fprintf(stderr, "%s(): too many iterations\n", __func__);
            return false;
        }

        // got a theta

        // Check the value of theta
        if(theta * fabs(linearity) > M_PI/2.)
        {
            fprintf(stderr, "%s(): theta out of bounds\n", __func__);
            return false;
        }

        // If we aren't close enough to use the small-angle approximation ...
        if (theta > 1e-8)
        {
            // ... do more math!
            double linth = linearity * theta;
            double chi;
            if (linearity < -1e-15)
                chi = sin(linth) / linearity;
            else if (linearity > 1e-15)
                chi = tan(linth) / linearity;
            else
                chi = theta;

            double chi2 = chi * chi;
            double chi3 = chi * chi2;
            double chi4 = chi * chi3;

            double zetap = l / chi;

            double mu = r0 + r1*chi2 + r2*chi4;

            double uu[3];
            for(int i=0; i<3; i++) uu[i] = zetap*o[i];
            double vv[3];
            for(int i=0; i<3; i++) vv[i] = (1. + mu)*ll[i];

            for(int i=0; i<3; i++)
                u[i] = uu[i] + vv[i];
            // now I apply a normal projection to the warped 3d point v
            out[i_pt].x = core->focal_xy[0] * u[0]/u[2] + core->center_xy[0];
            out[i_pt].y = core->focal_xy[1] * u[1]/u[2] + core->center_xy[1];
        }
        else
        {
            // now I apply a normal projection to the warped 3d point v
            out[i_pt].x = core->focal_xy[0] * vhere[0]/vhere[2] + core->center_xy[0];
            out[i_pt].y = core->focal_xy[1] * vhere[1]/vhere[2] + core->center_xy[1];
        }
    }
    return true;
}


// NOT A PART OF THE EXTERNAL API. This is exported for the mrcal python wrapper
// only
bool _mrcal_project_internal( // out
                             mrcal_point2_t* q,

                             // Stored as a row-first array of shape (N,2,3). Each
                             // trailing ,3 dimension element is a mrcal_point3_t
                             mrcal_point3_t* dq_dp,
                             // core, distortions concatenated. Stored as a row-first
                             // array of shape (N,2,Nintrinsics). This is a DENSE array.
                             // High-parameter-count lens models have very sparse
                             // gradients here, and the internal project() function
                             // returns those sparsely. For now THIS function densifies
                             // all of these
                             double*   dq_dintrinsics,

                             // in
                             const mrcal_point3_t* p,
                             int N,
                             mrcal_lensmodel_t lensmodel,
                             // core, distortions concatenated
                             const double* intrinsics,

                             int Nintrinsics,
                             const mrcal_projection_precomputed_t* precomputed)
{
    if( dq_dintrinsics == NULL )
    {
        for(int i=0; i<N; i++)
        {
            mrcal_pose_t frame = {.r = {},
                            .t = p[i]};

            // simple non-intrinsics-gradient path. dp_dp is handled entirely in
            // project()
            project( &q[i],
                     NULL, NULL, NULL, NULL, NULL,
                     NULL, NULL, NULL, dq_dp, NULL,

                     // in
                     intrinsics, NULL, &frame, NULL, true,
                     lensmodel, precomputed,
                     0.0, 0,0);
        }
        return true;
    }

    for(int i=0; i<N; i++)
    {
        mrcal_pose_t frame = {.r = {},
                        .t = p[i]};

        // simple non-intrinsics-gradient path. dp_dp is handled entirely in
        // project()
        double dq_dintrinsics_pool_double[2*(1+Nintrinsics-4)];
        int    dq_dintrinsics_pool_int   [1];
        double* dq_dfxy               = NULL;
        double* dq_dintrinsics_nocore = NULL;
        gradient_sparse_meta_t gradient_sparse_meta = {}; // init to pacify compiler warning

        project( &q[i],

                 dq_dintrinsics_pool_double,
                 dq_dintrinsics_pool_int,
                 &dq_dfxy, &dq_dintrinsics_nocore, &gradient_sparse_meta,

                 NULL, NULL, NULL, dq_dp, NULL,

                 // in
                 intrinsics, NULL, &frame, NULL, true,
                 lensmodel, precomputed,
                 0.0, 0,0);

        int Ncore = 0;
        if(dq_dfxy != NULL)
        {
            Ncore = 4;

            // fxy. off-diagonal elements are 0
            dq_dintrinsics[0*Nintrinsics + 0] = dq_dfxy[0];
            dq_dintrinsics[0*Nintrinsics + 1] = 0.0;
            dq_dintrinsics[1*Nintrinsics + 0] = 0.0;
            dq_dintrinsics[1*Nintrinsics + 1] = dq_dfxy[1];

            // cxy. Identity
            dq_dintrinsics[0*Nintrinsics + 2] = 1.0;
            dq_dintrinsics[0*Nintrinsics + 3] = 0.0;
            dq_dintrinsics[1*Nintrinsics + 2] = 0.0;
            dq_dintrinsics[1*Nintrinsics + 3] = 1.0;
        }
        if( dq_dintrinsics_nocore != NULL )
        {
            for(int i_xy=0; i_xy<2; i_xy++)
                memcpy(&dq_dintrinsics[i_xy*Nintrinsics + Ncore],
                       &dq_dintrinsics_nocore[i_xy*(Nintrinsics-Ncore)],
                       (Nintrinsics-Ncore)*sizeof(double));
        }
        if(gradient_sparse_meta.pool != NULL)
        {
            // u = stereographic(p)
            // q = (u + deltau(u)) * f + c
            //
            // Intrinsics:
            //   dq/diii = f ddeltau/diii
            //
            // ddeltau/diii = flatten(ABCDx[0..3] * ABCDy[0..3])

            const int     ivar0 = dq_dintrinsics_pool_int[0];
            const int     len   = gradient_sparse_meta.run_side_length;

            const double* ABCDx = &gradient_sparse_meta.pool[0];
            const double* ABCDy = &gradient_sparse_meta.pool[len];

            const int ivar_stridey = gradient_sparse_meta.ivar_stridey;
            const double* fxy = &intrinsics[0];
            for(int i_xy=0; i_xy<2; i_xy++)
                for(int iy=0; iy<len; iy++)
                    for(int ix=0; ix<len; ix++)
                    {
                        int ivar = ivar0 + ivar_stridey*iy + ix*2 + i_xy;
                        dq_dintrinsics[ivar + i_xy*Nintrinsics] =
                            ABCDx[ix]*ABCDy[iy]*fxy[i_xy];
                    }
        }

        // advance
        dq_dintrinsics = &dq_dintrinsics[2*Nintrinsics];
        if(dq_dp != NULL)
            dq_dp = &dq_dp[2];
    }
    return true;
}

// External interface to the internal project() function. The internal function
// is more general (supports geometric transformations prior to projection, and
// supports chessboards). dq_dintrinsics and/or dq_dp are allowed to be NULL if
// we're not interested in gradients.
//
// This function supports CAHVORE distortions if we don't ask for gradients
//
// Projecting out-of-bounds points (beyond the field of view) returns undefined
// values. Generally things remain continuous even as we move off the imager
// domain. Pinhole-like projections will work normally if projecting a point
// behind the camera. Splined projections clamp to the nearest spline segment:
// the projection will fly off to infinity quickly since we're extrapolating a
// polynomial, but the function will remain continuous.
bool mrcal_project( // out
                   mrcal_point2_t* q,

                   // Stored as a row-first array of shape (N,2,3). Each
                   // trailing ,3 dimension element is a mrcal_point3_t
                   mrcal_point3_t* dq_dp,
                   // core, distortions concatenated. Stored as a row-first
                   // array of shape (N,2,Nintrinsics). This is a DENSE array.
                   // High-parameter-count lens models have very sparse
                   // gradients here, and the internal project() function
                   // returns those sparsely. For now THIS function densifies
                   // all of these
                   double*   dq_dintrinsics,

                   // in
                   const mrcal_point3_t* p,
                   int N,
                   mrcal_lensmodel_t lensmodel,
                   // core, distortions concatenated
                   const double* intrinsics)
{
    // The outer logic (outside the loop-over-N-points) is duplicated in
    // mrcal_project() and in the python wrapper definition in _project() and
    // _project_withgrad() in mrcal-genpywrap.py. Please keep them in sync

    // project() doesn't handle cahvore, so I special-case it here
    if( lensmodel.type == MRCAL_LENSMODEL_CAHVORE )
    {
        if(dq_dintrinsics != NULL || dq_dp != NULL)
        {
            fprintf(stderr, "mrcal_project(MRCAL_LENSMODEL_CAHVORE) is not yet implemented if we're asking for gradients\n");
            return false;
        }
        return _mrcal_project_internal_cahvore(q, p, N, intrinsics);
    }

    int Nintrinsics = mrcal_lensmodel_num_params(lensmodel);

    // Special-case for opencv/pinhole and projection-only. cvProjectPoints2 and
    // project() have a lot of overhead apparently, and calling either in a loop
    // is very slow. I can call it once, and use its fast internal loop,
    // however. This special case does the same thing, but much faster.
    if(dq_dintrinsics == NULL && dq_dp == NULL &&
       (MRCAL_LENSMODEL_IS_OPENCV(lensmodel.type) ||
        lensmodel.type == MRCAL_LENSMODEL_PINHOLE))
    {
        _mrcal_project_internal_opencv( q, NULL,NULL,
                                        p, N, intrinsics, Nintrinsics);
        return true;
    }

    // Some models have sparse gradients, but I'm returning a dense array here.
    // So I init everything at 0
    if(dq_dintrinsics != NULL)
        memset(dq_dintrinsics, 0, N*2*Nintrinsics*sizeof(double));

    mrcal_projection_precomputed_t precomputed;
    _mrcal_precompute_lensmodel_data(&precomputed, lensmodel);

    return
        _mrcal_project_internal(q, dq_dp, dq_dintrinsics,
                                p, N, lensmodel, intrinsics,
                                Nintrinsics, &precomputed);
}


// Maps a set of distorted 2D imager points q to a 3d vector in camera
// coordinates that produced these pixel observations. The 3d vector is defined
// up-to-length, so the vectors reported here will all have z = 1.
//
// This is the "reverse" direction, so an iterative nonlinear optimization is
// performed internally to compute this result. This is much slower than
// mrcal_project. For OpenCV distortions specifically, OpenCV has
// cvUndistortPoints() (and cv2.undistortPoints()), but these are inaccurate:
// https://github.com/opencv/opencv/issues/8811
//
// This function does NOT support CAHVORE
bool mrcal_unproject( // out
                     mrcal_point3_t* out,

                     // in
                     const mrcal_point2_t* q,
                     int N,
                     mrcal_lensmodel_t lensmodel,
                     // core, distortions concatenated
                     const double* intrinsics)
{
    if( lensmodel.type == MRCAL_LENSMODEL_CAHVORE )
    {
        fprintf(stderr, "mrcal_unproject(MRCAL_LENSMODEL_CAHVORE) not yet implemented. No gradients available\n");
        return false;
    }

    // easy special-cases
    if( lensmodel.type == MRCAL_LENSMODEL_PINHOLE )
    {
        double fx = intrinsics[0];
        double fy = intrinsics[1];
        double cx = intrinsics[2];
        double cy = intrinsics[3];

        for(int i=0; i<N; i++)
        {
            out->x = (q[i].x - cx) / fx;
            out->y = (q[i].y - cy) / fy;
            out->z = 1.0;

            // advance
            out++;
        }
        return true;
    }
    if( lensmodel.type == MRCAL_LENSMODEL_STEREOGRAPHIC )
    {
        double fx = intrinsics[0];
        double fy = intrinsics[1];
        double cx = intrinsics[2];
        double cy = intrinsics[3];

        mrcal_unproject_stereographic(out, NULL, q, N, fx,fy,cx,cy);
        return true;
    }


    mrcal_projection_precomputed_t precomputed;
    _mrcal_precompute_lensmodel_data(&precomputed, lensmodel);

    return _mrcal_unproject_internal(out, q, N, lensmodel, intrinsics, &precomputed);
}

// NOT A PART OF THE EXTERNAL API. This is exported for the mrcal python wrapper
// only
bool _mrcal_unproject_internal( // out
                               mrcal_point3_t* out,

                               // in
                               const mrcal_point2_t* q,
                               int N,
                               mrcal_lensmodel_t lensmodel,
                               // core, distortions concatenated
                               const double* intrinsics,
                               const mrcal_projection_precomputed_t* precomputed)
{
    double fx = intrinsics[0];
    double fy = intrinsics[1];
    double cx = intrinsics[2];
    double cy = intrinsics[3];

    // I optimize in the space of the stereographic projection. This is a 2D
    // space with a direct mapping to/from observation vectors with a single
    // singularity directly behind the camera. The allows me to run an
    // unconstrained optimization here
    for(int i=0; i<N; i++)
    {
        void cb(const double*   u,
                double*         x,
                double*         J,
                void*           cookie __attribute__((unused)))
        {
            // u is the constant-fxy-cxy 2D stereographic
            // projection of the hypothesis v. I unproject it stereographically,
            // and project it using the actual model
            mrcal_point2_t dv_du[3];
            mrcal_pose_t frame = {};
            mrcal_unproject_stereographic( &frame.t, dv_du,
                                           (mrcal_point2_t*)u, 1,
                                           fx,fy,cx,cy );

            mrcal_point3_t dq_dtframe[2];
            mrcal_point2_t q_hypothesis;
            project( &q_hypothesis,
                     NULL,NULL,NULL,NULL,NULL,
                     NULL, NULL, NULL, dq_dtframe,
                     NULL,

                     // in
                     intrinsics,
                     NULL,
                     &frame,
                     NULL,
                     true,
                     lensmodel, precomputed,
                     0.0, 0,0);
            x[0] = q_hypothesis.x - q[i].x;
            x[1] = q_hypothesis.y - q[i].y;
            J[0*2 + 0] =
                dq_dtframe[0].x*dv_du[0].x +
                dq_dtframe[0].y*dv_du[1].x +
                dq_dtframe[0].z*dv_du[2].x;
            J[0*2 + 1] =
                dq_dtframe[0].x*dv_du[0].y +
                dq_dtframe[0].y*dv_du[1].y +
                dq_dtframe[0].z*dv_du[2].y;
            J[1*2 + 0] =
                dq_dtframe[1].x*dv_du[0].x +
                dq_dtframe[1].y*dv_du[1].x +
                dq_dtframe[1].z*dv_du[2].x;
            J[1*2 + 1] =
                dq_dtframe[1].x*dv_du[0].y +
                dq_dtframe[1].y*dv_du[1].y +
                dq_dtframe[1].z*dv_du[2].y;
        }


#warning "This should go away. For some reason it makes unproject() converge better, and it makes the tests pass. But it's not even right!"
#if 0
        out->xyz[0] = (q[i].x-cx)/fx;
        out->xyz[1] = (q[i].y-cy)/fy;
#else
        // Seed from a perfect stereographic projection, pushed towards the
        // center a bit. Normally I'd set out[] to q[i], but for some models
        // (OPENCV8 for instance) this pushes us into a place where stuff
        // doesn't converge anymore. This produces a more stable solution, and
        // my tests pass
        out->xyz[0] = (q[i].x-cx)*0.7 + cx;
        out->xyz[1] = (q[i].y-cy)*0.7 + cy;

        // something like this makes more sense, but it doesn't work! The tests still fail
        // out->xyz[0] = (q[i].x-cx)/fx * 0.7;
        // out->xyz[1] = (q[i].y-cy)/fy * 0.7;
#endif





        dogleg_parameters2_t dogleg_parameters;
        dogleg_getDefaultParameters(&dogleg_parameters);
        dogleg_parameters.dogleg_debug = 0;
        double norm2x =
            dogleg_optimize_dense2(out->xyz, 2, 2, cb, NULL,
                                   &dogleg_parameters,
                                   NULL);
        //This needs to be precise; if it isn't, I barf. Shouldn't happen
        //very often

        static bool already_complained = false;
        if(norm2x/2.0 > 1e-4)
        {
            if(!already_complained)
            {
                MSG("WARNING: I wasn't able to precisely compute some points. norm2x=%f. Returning nan for those. Will complain just once",
                    norm2x);
                already_complained = true;
            }
            double nan = strtod("NAN", NULL);
            out->xyz[0] = nan;
            out->xyz[1] = nan;
        }
        else
        {
            // out[0,1] is the stereographic representation of the observation
            // vector using idealized fx,fy,cx,cy. This is already the right
            // thing if we're reporting in 2d. Otherwise I need to unproject

            // This is the normal no-error path
            mrcal_unproject_stereographic((mrcal_point3_t*)out, NULL,
                                          (mrcal_point2_t*)out, 1,
                                          fx,fy,cx,cy);
            if(!model_supports_projection_behind_camera(lensmodel) && out->xyz[2] < 0.0)
            {
                out->xyz[0] *= -1.0;
                out->xyz[1] *= -1.0;
                out->xyz[2] *= -1.0;
            }
        }

        // Advance to the next point. Error or not
        out++;
    }
    return true;
}

// The following functions define/use the layout of the state vector. In general
// I do:
//
//   intrinsics_cam0
//   intrinsics_cam1
//   intrinsics_cam2
//   ...
//   extrinsics_cam1
//   extrinsics_cam2
//   extrinsics_cam3
//   ...
//   frame0
//   frame1
//   frame2
//   ....
//   calobject_warp0
//   calobject_warp1

// From real values to unit-scale values. Optimizer sees unit-scale values
static int pack_solver_state_intrinsics( // out
                                         double* p,

                                         // in
                                         const double* intrinsics,
                                         const mrcal_lensmodel_t lensmodel,
                                         mrcal_problem_details_t problem_details,
                                         int Ncameras_intrinsics )
{
    int i_state = 0;

    int Nintrinsics  = mrcal_lensmodel_num_params(lensmodel);
    int Ncore        = modelHasCore_fxfycxcy(lensmodel) ? 4 : 0;
    int Ndistortions = Nintrinsics - Ncore;

    for(int i_cam_intrinsics=0; i_cam_intrinsics < Ncameras_intrinsics; i_cam_intrinsics++)
    {
        if( problem_details.do_optimize_intrinsics_core )
        {
            const mrcal_intrinsics_core_t* intrinsics_core = (const mrcal_intrinsics_core_t*)intrinsics;
            p[i_state++] = intrinsics_core->focal_xy [0] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->focal_xy [1] / SCALE_INTRINSICS_FOCAL_LENGTH;
            p[i_state++] = intrinsics_core->center_xy[0] / SCALE_INTRINSICS_CENTER_PIXEL;
            p[i_state++] = intrinsics_core->center_xy[1] / SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( problem_details.do_optimize_intrinsics_distortions )

            for(int i = 0; i<Ndistortions; i++)
                p[i_state++] = intrinsics[Ncore + i] / SCALE_DISTORTION;

        intrinsics = &intrinsics[Nintrinsics];
    }
    return i_state;
}
static void pack_solver_state( // out
                              double* p,

                              // in
                              const mrcal_lensmodel_t lensmodel,
                              const double* intrinsics, // Ncameras_intrinsics of these
                              const mrcal_pose_t*            extrinsics_fromref, // Ncameras_extrinsics of these
                              const mrcal_pose_t*            frames_toref,     // Nframes of these
                              const mrcal_point3_t*          points,     // Npoints of these
                              const mrcal_point2_t*          calobject_warp, // 1 of these
                              mrcal_problem_details_t problem_details,
                              int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                              int Npoints_variable,

                              int Nstate_ref)
{
    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, intrinsics,
                                             lensmodel, problem_details,
                                             Ncameras_intrinsics );

    if( problem_details.do_optimize_extrinsics )
        for(int i_cam_extrinsics=0; i_cam_extrinsics < Ncameras_extrinsics; i_cam_extrinsics++)
        {
            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics_fromref[i_cam_extrinsics].t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( problem_details.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
        {
            p[i_state++] = frames_toref[i_frame].r.xyz[0] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames_toref[i_frame].r.xyz[1] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames_toref[i_frame].r.xyz[2] / SCALE_ROTATION_FRAME;

            p[i_state++] = frames_toref[i_frame].t.xyz[0] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames_toref[i_frame].t.xyz[1] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames_toref[i_frame].t.xyz[2] / SCALE_TRANSLATION_FRAME;
        }

        for(int i_point = 0; i_point < Npoints_variable; i_point++)
        {
            p[i_state++] = points[i_point].xyz[0] / SCALE_POSITION_POINT;
            p[i_state++] = points[i_point].xyz[1] / SCALE_POSITION_POINT;
            p[i_state++] = points[i_point].xyz[2] / SCALE_POSITION_POINT;
        }
    }

    if( problem_details.do_optimize_calobject_warp )
    {
        p[i_state++] = calobject_warp->x / SCALE_CALOBJECT_WARP;
        p[i_state++] = calobject_warp->y / SCALE_CALOBJECT_WARP;
    }

    assert(i_state == Nstate_ref);
}

// Same as above, but packs/unpacks a vector instead of structures
void mrcal_pack_solver_state_vector( // out, in
                                     double* p, // unitless, FULL state on
                                                // input, scaled, decimated
                                                // (subject to problem_details),
                                                // meaningful state on output

                                     // in
                                     int Ncameras_intrinsics, int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints, int Npoints_fixed,
                                     mrcal_problem_details_t problem_details,
                                     const mrcal_lensmodel_t lensmodel)
{
    int Npoints_variable = Npoints - Npoints_fixed;

    int i_state = 0;

    i_state += pack_solver_state_intrinsics( p, p,
                                             lensmodel, problem_details,
                                             Ncameras_intrinsics );

    static_assert( offsetof(mrcal_pose_t, r) == 0,                   "mrcal_pose_t has expected structure");
    static_assert( offsetof(mrcal_pose_t, t) == 3*sizeof(double),    "mrcal_pose_t has expected structure");
    if( problem_details.do_optimize_extrinsics )
        for(int i_cam_extrinsics=0; i_cam_extrinsics < Ncameras_extrinsics; i_cam_extrinsics++)
        {
            mrcal_pose_t* extrinsics_fromref = (mrcal_pose_t*)(&p[i_state]);

            p[i_state++] = extrinsics_fromref->r.xyz[0] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics_fromref->r.xyz[1] / SCALE_ROTATION_CAMERA;
            p[i_state++] = extrinsics_fromref->r.xyz[2] / SCALE_ROTATION_CAMERA;

            p[i_state++] = extrinsics_fromref->t.xyz[0] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics_fromref->t.xyz[1] / SCALE_TRANSLATION_CAMERA;
            p[i_state++] = extrinsics_fromref->t.xyz[2] / SCALE_TRANSLATION_CAMERA;
        }

    if( problem_details.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
        {
            mrcal_pose_t* frames_toref = (mrcal_pose_t*)(&p[i_state]);
            p[i_state++] = frames_toref->r.xyz[0] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames_toref->r.xyz[1] / SCALE_ROTATION_FRAME;
            p[i_state++] = frames_toref->r.xyz[2] / SCALE_ROTATION_FRAME;

            p[i_state++] = frames_toref->t.xyz[0] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames_toref->t.xyz[1] / SCALE_TRANSLATION_FRAME;
            p[i_state++] = frames_toref->t.xyz[2] / SCALE_TRANSLATION_FRAME;
        }

        for(int i_point = 0; i_point < Npoints_variable; i_point++)
        {
            mrcal_point3_t* points = (mrcal_point3_t*)(&p[i_state]);
            p[i_state++] = points->xyz[0] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[1] / SCALE_POSITION_POINT;
            p[i_state++] = points->xyz[2] / SCALE_POSITION_POINT;
        }
    }

    if( problem_details.do_optimize_calobject_warp )
    {
        mrcal_point2_t* calobject_warp = (mrcal_point2_t*)(&p[i_state]);
        p[i_state++] = calobject_warp->x / SCALE_CALOBJECT_WARP;
        p[i_state++] = calobject_warp->y / SCALE_CALOBJECT_WARP;
    }
}

static int unpack_solver_state_intrinsics( // out

                                           // Ncameras_intrinsics of these
                                           double* intrinsics,

                                           // in
                                           const double* p,
                                           const mrcal_lensmodel_t lensmodel,
                                           mrcal_problem_details_t problem_details,
                                           int intrinsics_stride,
                                           int Ncameras_intrinsics )
{
    if( !problem_details.do_optimize_intrinsics_core &&
        !problem_details.do_optimize_intrinsics_distortions )
        return 0;

    const int Nintrinsics = mrcal_lensmodel_num_params(lensmodel);
    const int Ncore       = modelHasCore_fxfycxcy(lensmodel) ? 4 : 0;

    int i_state = 0;
    for(int i_cam_intrinsics=0; i_cam_intrinsics < Ncameras_intrinsics; i_cam_intrinsics++)
    {
        if( problem_details.do_optimize_intrinsics_core && Ncore )
        {
            intrinsics[i_cam_intrinsics*intrinsics_stride + 0] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics[i_cam_intrinsics*intrinsics_stride + 1] = p[i_state++] * SCALE_INTRINSICS_FOCAL_LENGTH;
            intrinsics[i_cam_intrinsics*intrinsics_stride + 2] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
            intrinsics[i_cam_intrinsics*intrinsics_stride + 3] = p[i_state++] * SCALE_INTRINSICS_CENTER_PIXEL;
        }

        if( problem_details.do_optimize_intrinsics_distortions )
        {
            for(int i = 0; i<Nintrinsics-Ncore; i++)
                intrinsics[i_cam_intrinsics*intrinsics_stride + Ncore + i] = p[i_state++] * SCALE_DISTORTION;
        }
    }
    return i_state;
}

static int unpack_solver_state_extrinsics_one(// out
                                              mrcal_pose_t* extrinsic,

                                              // in
                                              const double* p)
{
    int i_state = 0;
    extrinsic->r.xyz[0] = p[i_state++] * SCALE_ROTATION_CAMERA;
    extrinsic->r.xyz[1] = p[i_state++] * SCALE_ROTATION_CAMERA;
    extrinsic->r.xyz[2] = p[i_state++] * SCALE_ROTATION_CAMERA;

    extrinsic->t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    extrinsic->t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    extrinsic->t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_CAMERA;
    return i_state;
}

static int unpack_solver_state_framert_one(// out
                                           mrcal_pose_t* frame,

                                           // in
                                           const double* p)
{
    int i_state = 0;
    frame->r.xyz[0] = p[i_state++] * SCALE_ROTATION_FRAME;
    frame->r.xyz[1] = p[i_state++] * SCALE_ROTATION_FRAME;
    frame->r.xyz[2] = p[i_state++] * SCALE_ROTATION_FRAME;

    frame->t.xyz[0] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    frame->t.xyz[1] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    frame->t.xyz[2] = p[i_state++] * SCALE_TRANSLATION_FRAME;
    return i_state;

}

static int unpack_solver_state_point_one(// out
                                         mrcal_point3_t* point,

                                         // in
                                         const double* p)
{
    int i_state = 0;
    point->xyz[0] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[1] = p[i_state++] * SCALE_POSITION_POINT;
    point->xyz[2] = p[i_state++] * SCALE_POSITION_POINT;
    return i_state;
}

static int unpack_solver_state_calobject_warp(// out
                                              mrcal_point2_t* calobject_warp,

                                              // in
                                              const double* p)
{
    int i_state = 0;
    calobject_warp->xy[0] = p[i_state++] * SCALE_CALOBJECT_WARP;
    calobject_warp->xy[1] = p[i_state++] * SCALE_CALOBJECT_WARP;
    return i_state;
}

// From unit-scale values to real values. Optimizer sees unit-scale values
static void unpack_solver_state( // out

                                 // ALL intrinsics; whether we're optimizing
                                 // them or not. Don't touch the parts of this
                                 // array we're not optimizing
                                 double* intrinsics_all,                 // Ncameras_intrinsics of these

                                 mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these
                                 mrcal_pose_t*       frames_toref,       // Nframes of these
                                 mrcal_point3_t*     points,             // Npoints of these
                                 mrcal_point2_t*     calobject_warp,     // 1 of these

                                 // in
                                 const double* p,
                                 const mrcal_lensmodel_t lensmodel,
                                 mrcal_problem_details_t problem_details,
                                 int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes, int Npoints_variable,

                                 int Nstate_ref)
{
    int i_state = unpack_solver_state_intrinsics(intrinsics_all,
                                                 p, lensmodel, problem_details,
                                                 mrcal_lensmodel_num_params(lensmodel),
                                                 Ncameras_intrinsics);

    if( problem_details.do_optimize_extrinsics )
        for(int i_cam_extrinsics=0; i_cam_extrinsics < Ncameras_extrinsics; i_cam_extrinsics++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics_fromref[i_cam_extrinsics], &p[i_state] );

    if( problem_details.do_optimize_frames )
    {
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames_toref[i_frame], &p[i_state] );
        for(int i_point = 0; i_point < Npoints_variable; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }

    if( problem_details.do_optimize_calobject_warp )
        i_state += unpack_solver_state_calobject_warp(calobject_warp, &p[i_state]);

    assert(i_state == Nstate_ref);
}
// Same as above, but packs/unpacks a vector instead of structures
void mrcal_unpack_solver_state_vector( // out, in
                                       double* p, // unitless state on input,
                                                  // scaled, meaningful state on
                                                  // output

                                       // in
                                       int Ncameras_intrinsics, int Ncameras_extrinsics,
                                       int Nframes,
                                       int Npoints, int Npoints_fixed,
                                       mrcal_problem_details_t problem_details,
                                       const mrcal_lensmodel_t lensmodel)
{
    int Npoints_variable = Npoints - Npoints_fixed;

    int i_state = unpack_solver_state_intrinsics(&p[modelHasCore_fxfycxcy(lensmodel) &&
                                                    !problem_details.do_optimize_intrinsics_core ? -4 : 0],
                                                 p, lensmodel, problem_details,
                                                 mrcal_num_intrinsics_optimization_params(problem_details,
                                                                                          lensmodel),
                                                 Ncameras_intrinsics);

    if( problem_details.do_optimize_extrinsics )
    {
        static_assert( offsetof(mrcal_pose_t, r) == 0,
                       "mrcal_pose_t has expected structure");
        static_assert( offsetof(mrcal_pose_t, t) == 3*sizeof(double),
                       "mrcal_pose_t has expected structure");

        mrcal_pose_t* extrinsics_fromref = (mrcal_pose_t*)(&p[i_state]);
        for(int i_cam_extrinsics=0; i_cam_extrinsics < Ncameras_extrinsics; i_cam_extrinsics++)
            i_state += unpack_solver_state_extrinsics_one( &extrinsics_fromref[i_cam_extrinsics], &p[i_state] );
    }

    if( problem_details.do_optimize_frames )
    {
        mrcal_pose_t* frames_toref = (mrcal_pose_t*)(&p[i_state]);
        for(int i_frame = 0; i_frame < Nframes; i_frame++)
            i_state += unpack_solver_state_framert_one( &frames_toref[i_frame], &p[i_state] );
        mrcal_point3_t* points = (mrcal_point3_t*)(&p[i_state]);
        for(int i_point = 0; i_point < Npoints_variable; i_point++)
            i_state += unpack_solver_state_point_one( &points[i_point], &p[i_state] );
    }
    if( problem_details.do_optimize_calobject_warp )
    {
        mrcal_point2_t* calobject_warp = (mrcal_point2_t*)(&p[i_state]);
        i_state += unpack_solver_state_calobject_warp(calobject_warp, &p[i_state]);
    }
}

int mrcal_state_index_intrinsics(int i_cam_intrinsics,
                                 int Ncameras_intrinsics, int Ncameras_extrinsics,
                                 int Nframes,
                                 int Npoints, int Npoints_fixed,
                                 mrcal_problem_details_t problem_details,
                                 mrcal_lensmodel_t lensmodel)
{
    return i_cam_intrinsics * mrcal_num_intrinsics_optimization_params(problem_details, lensmodel);
}

int mrcal_num_states_intrinsics(int Ncameras_intrinsics,
                                mrcal_problem_details_t problem_details,
                                mrcal_lensmodel_t lensmodel)
{
    return
        Ncameras_intrinsics *
        mrcal_num_intrinsics_optimization_params(problem_details, lensmodel);
}

int mrcal_state_index_extrinsics(int i_cam_extrinsics,
                                 int Ncameras_intrinsics, int Ncameras_extrinsics,
                                 int Nframes,
                                 int Npoints, int Npoints_fixed,
                                 mrcal_problem_details_t problem_details,
                                 mrcal_lensmodel_t lensmodel)
{
    return
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_details,
                                    lensmodel) +
        (problem_details.do_optimize_extrinsics ? (i_cam_extrinsics*6) : 0);
}

int mrcal_num_states_extrinsics(int Ncameras_extrinsics,
                                mrcal_problem_details_t problem_details)
{
    return problem_details.do_optimize_extrinsics ? (6*Ncameras_extrinsics) : 0;
}

int mrcal_state_index_frames(int i_frame,
                             int Ncameras_intrinsics, int Ncameras_extrinsics,
                             int Nframes,
                             int Npoints, int Npoints_fixed,
                             mrcal_problem_details_t problem_details,
                             mrcal_lensmodel_t lensmodel)
{
    return
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_details,
                                    lensmodel) +
        mrcal_num_states_extrinsics(Ncameras_extrinsics,
                                    problem_details) +
        (problem_details.do_optimize_frames ? (i_frame*6) : 0);
}

int mrcal_num_states_frames(int Nframes,
                            mrcal_problem_details_t problem_details)
{
    return problem_details.do_optimize_frames ? (6*Nframes) : 0;
}

int mrcal_state_index_points(int i_point,
                             int Ncameras_intrinsics, int Ncameras_extrinsics,
                             int Nframes,
                             int Npoints, int Npoints_fixed,
                             mrcal_problem_details_t problem_details,
                             mrcal_lensmodel_t lensmodel)
{
    return
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_details,
                                    lensmodel) +
        mrcal_num_states_extrinsics(Ncameras_extrinsics,
                                    problem_details) +
        mrcal_num_states_frames    (Nframes,
                                    problem_details) +
        (problem_details.do_optimize_frames ? (i_point*3) : 0);
}

int mrcal_num_states_points(int Npoints, int Npoints_fixed,
                            mrcal_problem_details_t problem_details)
{
    int Npoints_variable = Npoints - Npoints_fixed;
    return problem_details.do_optimize_frames ? (Npoints_variable*3) : 0;
}

int mrcal_state_index_calobject_warp(int Ncameras_intrinsics, int Ncameras_extrinsics,
                                     int Nframes,
                                     int Npoints, int Npoints_fixed,
                                     mrcal_problem_details_t problem_details,
                                     mrcal_lensmodel_t lensmodel)
{
    return
        mrcal_num_states_intrinsics(Ncameras_intrinsics,
                                    problem_details,
                                    lensmodel) +
        mrcal_num_states_extrinsics(Ncameras_extrinsics,
                                    problem_details) +
        mrcal_num_states_frames    (Nframes,
                                    problem_details) +
        mrcal_num_states_points    (Npoints, Npoints_fixed,
                                    problem_details);
}

int mrcal_num_states_calobject_warp(mrcal_problem_details_t problem_details)
{
    return problem_details.do_optimize_calobject_warp ? 2 : 0;
}

// Reports the icam_extrinsics corresponding to a given icam_intrinsics. On
// success, the result is written to *icam_extrinsics, and we return true. If
// the given camera is at the reference coordinate system, it has no extrinsics,
// and we report -1. This query only makes sense for a calibration problem:
// we're observing a moving object with stationary cameras. If we have moving
// cameras, there won't be a single icam_extrinsics for a given icam_intrinsics,
// and we report an error by returning false
bool mrcal_corresponding_icam_extrinsics(// out
                                         int* icam_extrinsics,

                                         // in
                                         int icam_intrinsics,
                                         int Ncameras_intrinsics,
                                         int Ncameras_extrinsics,
                                         int Nobservations_board,
                                         const mrcal_observation_board_t* observations_board)
{
    if( !(Ncameras_intrinsics == Ncameras_extrinsics ||
          Ncameras_intrinsics == Ncameras_extrinsics+1 ) )
    {
        MSG("Cannot compute icam_extrinsics. I don't have a pure calibration problem");
        return false;
    }

    int icam_map_to_extrinsics[Ncameras_intrinsics];
    int icam_map_to_intrinsics[Ncameras_extrinsics+1];
    for(int i=0; i<Ncameras_intrinsics;   i++) icam_map_to_extrinsics[i] = -100;
    for(int i=0; i<Ncameras_extrinsics+1; i++) icam_map_to_intrinsics[i] = -100;

    for(int i=0; i<Nobservations_board; i++)
    {
        int i_cam_intrinsics = observations_board[i].i_cam_intrinsics;
        int i_cam_extrinsics = observations_board[i].i_cam_extrinsics;
        if(i_cam_extrinsics < 0) i_cam_extrinsics = -1;

        if(icam_map_to_intrinsics[i_cam_extrinsics+1] == -100)
            icam_map_to_intrinsics[i_cam_extrinsics+1] = i_cam_intrinsics;
        else if(icam_map_to_intrinsics[i_cam_extrinsics+1] != i_cam_intrinsics)
        {
            MSG("Cannot compute icam_extrinsics. I don't have a pure calibration problem: observation %d has i_cam_intrinsics,i_cam_extrinsics %d,%d while I saw %d,%d previously",
                i,
                icam_map_to_intrinsics[i_cam_extrinsics+1], i_cam_extrinsics,
                i_cam_intrinsics, i_cam_extrinsics);
            return false;
        }

        if(icam_map_to_extrinsics[i_cam_intrinsics] == -100)
            icam_map_to_extrinsics[i_cam_intrinsics] = i_cam_extrinsics;
        else if(icam_map_to_extrinsics[i_cam_intrinsics] != i_cam_extrinsics)
        {
            MSG("Cannot compute icam_extrinsics. I don't have a pure calibration problem: observation %d has i_cam_intrinsics,i_cam_extrinsics %d,%d while I saw %d,%d previously",
                i,
                i_cam_intrinsics, icam_map_to_extrinsics[i_cam_intrinsics],
                i_cam_intrinsics, i_cam_extrinsics);
            return false;
        }
    }

    *icam_extrinsics = icam_map_to_extrinsics[icam_intrinsics];

    return true;
}

// Doing this myself instead of hooking into the logic in libdogleg for now.
// Bring back the fancy libdogleg logic once everything stabilizes
static
bool markOutliers(// output, input

                  // the weight stored in each mrcal_point3_t.z indicates outlierness
                  // on entry AND on exit. Outliers have weight < 0.0
                  mrcal_point3_t* observations_board_pool,

                  // output
                  int* Noutliers,

                  // input
                  const mrcal_observation_board_t* observations_board,
                  int Nobservations_board,
                  int calibration_object_width_n,
                  int calibration_object_height_n,

                  const double* x_measurements,
                  double observed_pixel_uncertainty,
                  bool verbose)
{
    // I define an outlier as a feature that's > k stdevs past the mean. I make
    // a broad assumption that the error distribution is normally-distributed,
    // with mean 0. This is reasonable because this function is applied after
    // running the optimization.
    //
    // The threshold stdev is the larger of
    //
    // - The stdev of my observed residuals
    // - The expected stdev of my noise passed-in as the
    //   observed_pixel_uncertainty
    //
    // The rationale:
    //
    // - If I have a really good solve, the stdev of my data set will be very
    //   low, so deviations off that already-very-good solve aren't important. I
    //   use the expected-noise stdev in this case
    //
    // - If the solve isn't great, my errors may exceed the expected-noise stdev
    //   (if my model doesn't fit very well, say). In that case I want to use
    //   the stdev from the data
    //
    // This assumes that the given obsered_pixel_uncertainty isn't very
    // well-known
    //
    // I have two separate thresholds. If any measurements are worse than the
    // higher threshold, then I will need to reoptimize, so I throw out some
    // extra points: all points worse than the lower threshold. This serves to
    // reduce the required re-optimizations

    // threshold. +- 3sigma includes 99.7% of the data in a normal distribution
    const double k0 = 3.0;
    const double k1 = 3.5;
    *Noutliers = 0;

    int i_pt,i_feature;


#define LOOP_FEATURE_BEGIN()                                            \
    i_feature = 0;                                                      \
    for(int i_observation_board=0;                                      \
        i_observation_board<Nobservations_board;                         \
        i_observation_board++)                                          \
    {                                                                   \
        const mrcal_observation_board_t* observation = &observations_board[i_observation_board]; \
        const int i_cam_intrinsics = observation->i_cam_intrinsics;     \
        for(i_pt=0;                                                     \
            i_pt < calibration_object_width_n*calibration_object_height_n; \
            i_pt++, i_feature++)                                        \
        {                                                               \
            const mrcal_point3_t* pt_observed = &observations_board_pool[i_feature]; \
            double* weight = &observations_board_pool[i_feature].z;


#define LOOP_FEATURE_END() \
    }}


    double sum_weight = 0.0;
    double var = 0.0;
    LOOP_FEATURE_BEGIN()
    {
        if(*weight < 0.0)
        {
            (*Noutliers)++;
            continue;
        }

        double dx = x_measurements[2*i_feature + 0];
        double dy = x_measurements[2*i_feature + 1];

        var += *weight * (dx*dx + dy*dy);
        sum_weight += *weight;
    }
    LOOP_FEATURE_END();
    var /= (2.*sum_weight);

    var = fmax(var, observed_pixel_uncertainty*observed_pixel_uncertainty);

    bool markedAny = false;
    LOOP_FEATURE_BEGIN()
    {
        if(*weight < 0.0)
          continue;

        double dx = x_measurements[2*i_feature + 0];
        double dy = x_measurements[2*i_feature + 1];
        if(dx*dx > k1*k1*var ||
           dy*dy > k1*k1*var )
        {
            *weight   = -1.0;
            markedAny = true;
            (*Noutliers)++;

            // MSG_IF_VERBOSE("Feature %d looks like an outlier. x/y are %f/%f stdevs off mean. Observed stdev: %f, limit: %f",
            //                i_feature, dx/sqrt(var), dy/sqrt(var), sqrt(var), k);

        }
    }
    LOOP_FEATURE_END();

    if(!markedAny)
        return false;

    // Some measurements were past the worse threshold, so I throw out a bit
    // extra to leave some margin so that the next re-optimization would be the
    // last. Hopefully
    LOOP_FEATURE_BEGIN()
    {
        if(*weight < 0)
          continue;

        double dx = x_measurements[2*i_feature + 0];
        double dy = x_measurements[2*i_feature + 1];
        if(dx*dx > k0*k0*var ||
           dy*dy > k0*k0*var )
        {
            *weight *= -1.0;
            (*Noutliers)++;

            // MSG("Feature %d looks like an outlier. x/y are %f/%f stdevs off mean. Observed stdev: %f, limit: %f",
            //                i_feature, dx/sqrt(var), dy/sqrt(var), sqrt(var), k0);

        }
    }
    LOOP_FEATURE_END();
    return true;

#undef LOOP_FEATURE_BEGIN
#undef LOOP_FEATURE_END
}

typedef struct
{
    // these are all UNPACKED
    const double*         intrinsics;         // Ncameras_intrinsics * NlensParams of these
    const mrcal_pose_t*   extrinsics_fromref; // Ncameras_extrinsics of these. Transform FROM the reference frame
    const mrcal_pose_t*   frames_toref;       // Nframes of these.    Transform TO the reference frame
    const mrcal_point3_t* points;             // Npoints of these.    In the reference frame
    const mrcal_point2_t* calobject_warp;     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

    // in
    int Ncameras_intrinsics, Ncameras_extrinsics, Nframes;
    int Npoints, Npoints_fixed;

    const mrcal_observation_board_t* observations_board;
    const mrcal_point3_t* observations_board_pool;
    int Nobservations_board;

    const mrcal_observation_point_t* observations_point;
    int Nobservations_point;

    bool verbose;

    mrcal_lensmodel_t lensmodel;
    mrcal_projection_precomputed_t precomputed;
    const int* imagersizes; // Ncameras_intrinsics*2 of these

    mrcal_problem_details_t          problem_details;
    const mrcal_problem_constants_t* problem_constants;

    double calibration_object_spacing;
    int calibration_object_width_n;
    int calibration_object_height_n;

    const int Nmeasurements, N_j_nonzero, Nintrinsics;
    const char* reportFitMsg;
} callback_context_t;

static
void optimizer_callback(// input state
                       const double*   packed_state,

                       // output measurements
                       double*         x,

                       // Jacobian
                       cholmod_sparse* Jt,

                       const callback_context_t* ctx)
{
    double norm2_error = 0.0;

    int    iJacobian          = 0;
    int    iMeasurement       = 0;

    int*    Jrowptr = Jt ? (int*)   Jt->p : NULL;
    int*    Jcolidx = Jt ? (int*)   Jt->i : NULL;
    double* Jval    = Jt ? (double*)Jt->x : NULL;
#define STORE_JACOBIAN(col, g)                  \
    do                                          \
    {                                           \
        if(Jt) {                                \
            Jcolidx[ iJacobian ] = col;         \
            Jval   [ iJacobian ] = g;           \
        }                                       \
        iJacobian++;                            \
    } while(0)
#define STORE_JACOBIAN2(col0, g0, g1)           \
    do                                          \
    {                                           \
        if(Jt) {                                \
            Jcolidx[ iJacobian+0 ] = col0+0;    \
            Jval   [ iJacobian+0 ] = g0;        \
            Jcolidx[ iJacobian+1 ] = col0+1;    \
            Jval   [ iJacobian+1 ] = g1;        \
        }                                       \
        iJacobian += 2;                         \
    } while(0)
    #define STORE_JACOBIAN3(col0, g0, g1, g2)               \
        do                                              \
        {                                               \
            if(Jt) {                                    \
                Jcolidx[ iJacobian+0 ] = col0+0;        \
                Jval   [ iJacobian+0 ] = g0;            \
                Jcolidx[ iJacobian+1 ] = col0+1;        \
                Jval   [ iJacobian+1 ] = g1;            \
                Jcolidx[ iJacobian+2 ] = col0+2;        \
                Jval   [ iJacobian+2 ] = g2;            \
            }                                           \
            iJacobian += 3;                             \
        } while(0)


    int Ncore = modelHasCore_fxfycxcy(ctx->lensmodel) ? 4 : 0;
    int Ncore_state = (modelHasCore_fxfycxcy(ctx->lensmodel) &&
                       ctx->problem_details.do_optimize_intrinsics_core) ? 4 : 0;

    // If I'm locking down some parameters, then the state vector contains a
    // subset of my data. I reconstitute the intrinsics and extrinsics here.
    // I do the frame poses later. This is a good way to do it if I have few
    // cameras. With many cameras (this will be slow)
    double intrinsics_all[ctx->Ncameras_intrinsics][ctx->Nintrinsics];
    mrcal_pose_t camera_rt[ctx->Ncameras_extrinsics];

    mrcal_point2_t calobject_warp_local = {};
    const int i_var_calobject_warp =
        mrcal_state_index_calobject_warp(ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
    if(ctx->problem_details.do_optimize_calobject_warp)
        unpack_solver_state_calobject_warp(&calobject_warp_local, &packed_state[i_var_calobject_warp]);
    else if(ctx->calobject_warp != NULL)
        calobject_warp_local = *ctx->calobject_warp;

    for(int i_camera_intrinsics=0;
        i_camera_intrinsics<ctx->Ncameras_intrinsics;
        i_camera_intrinsics++)
    {
        // Construct the FULL intrinsics vector, based on either the
        // optimization vector or the inputs, depending on what we're optimizing
        double* intrinsics_here  = &intrinsics_all[i_camera_intrinsics][0];
        double* distortions_here = &intrinsics_all[i_camera_intrinsics][Ncore];

        int i_var_intrinsics =
            mrcal_state_index_intrinsics(i_camera_intrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
        if(Ncore)
        {
            if( ctx->problem_details.do_optimize_intrinsics_core )
            {
                intrinsics_here[0] = packed_state[i_var_intrinsics++] * SCALE_INTRINSICS_FOCAL_LENGTH;
                intrinsics_here[1] = packed_state[i_var_intrinsics++] * SCALE_INTRINSICS_FOCAL_LENGTH;
                intrinsics_here[2] = packed_state[i_var_intrinsics++] * SCALE_INTRINSICS_CENTER_PIXEL;
                intrinsics_here[3] = packed_state[i_var_intrinsics++] * SCALE_INTRINSICS_CENTER_PIXEL;
            }
            else
                memcpy( intrinsics_here,
                        &ctx->intrinsics[ctx->Nintrinsics*i_camera_intrinsics],
                        Ncore*sizeof(double) );
        }
        if( ctx->problem_details.do_optimize_intrinsics_distortions )
        {
            for(int i = 0; i<ctx->Nintrinsics-Ncore; i++)
                distortions_here[i] = packed_state[i_var_intrinsics++] * SCALE_DISTORTION;
        }
        else
            memcpy( distortions_here,
                    &ctx->intrinsics[ctx->Nintrinsics*i_camera_intrinsics + Ncore],
                    (ctx->Nintrinsics-Ncore)*sizeof(double) );
    }
    for(int i_camera_extrinsics=0;
        i_camera_extrinsics<ctx->Ncameras_extrinsics;
        i_camera_extrinsics++)
    {
        if( i_camera_extrinsics < 0 ) continue;

        const int i_var_camera_rt =
            mrcal_state_index_extrinsics(i_camera_extrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
        if(ctx->problem_details.do_optimize_extrinsics)
            unpack_solver_state_extrinsics_one(&camera_rt[i_camera_extrinsics], &packed_state[i_var_camera_rt]);
        else
            memcpy(&camera_rt[i_camera_extrinsics], &ctx->extrinsics_fromref[i_camera_extrinsics], sizeof(mrcal_pose_t));
    }

    int i_feature = 0;
    for(int i_observation_board = 0;
        i_observation_board < ctx->Nobservations_board;
        i_observation_board++)
    {
        const mrcal_observation_board_t* observation = &ctx->observations_board[i_observation_board];

        const int i_cam_intrinsics = observation->i_cam_intrinsics;
        const int i_cam_extrinsics = observation->i_cam_extrinsics;
        const int i_frame          = observation->i_frame;


        // Some of these are bogus if problem_details says they're inactive
        const int i_var_frame_rt =
            mrcal_state_index_frames(i_frame,
                                     ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                     ctx->Nframes,
                                     ctx->Npoints, ctx->Npoints_fixed,
                                     ctx->problem_details, ctx->lensmodel);

        mrcal_pose_t frame_rt;
        if(ctx->problem_details.do_optimize_frames)
            unpack_solver_state_framert_one(&frame_rt, &packed_state[i_var_frame_rt]);
        else
            memcpy(&frame_rt, &ctx->frames_toref[i_frame], sizeof(mrcal_pose_t));

        const int i_var_intrinsics =
            mrcal_state_index_intrinsics(i_cam_intrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
        // invalid if i_cam_extrinsics < 0, but unused in that case
        const int i_var_camera_rt  =
            mrcal_state_index_extrinsics(i_cam_extrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);

        // these are computed in respect to the real-unit parameters,
        // NOT the unit-scale parameters used by the optimizer
        mrcal_point3_t dq_drcamera       [ctx->calibration_object_width_n*ctx->calibration_object_height_n][2];
        mrcal_point3_t dq_dtcamera       [ctx->calibration_object_width_n*ctx->calibration_object_height_n][2];
        mrcal_point3_t dq_drframe        [ctx->calibration_object_width_n*ctx->calibration_object_height_n][2];
        mrcal_point3_t dq_dtframe        [ctx->calibration_object_width_n*ctx->calibration_object_height_n][2];
        mrcal_point2_t dq_dcalobject_warp[ctx->calibration_object_width_n*ctx->calibration_object_height_n][2];
        mrcal_point2_t q_hypothesis      [ctx->calibration_object_width_n*ctx->calibration_object_height_n];
        // I get the intrinsics gradients in separate arrays, possibly sparsely.
        // All the data lives in dq_dintrinsics_pool_double[], with the other data
        // indicating the meaning of the values in the pool.
        //
        // dq_dfxy serves a special-case for a perspective core. Such models
        // are very common, and they have x = fx vx/vz + cx and y = fy vy/vz +
        // cy. So x depends on fx and NOT on fy, and similarly for y. Similar
        // for cx,cy, except we know the gradient value beforehand. I support
        // this case explicitly here. I store dx/dfx and dy/dfy; no cross terms
        double dq_dintrinsics_pool_double[ctx->calibration_object_width_n*ctx->calibration_object_height_n*2*(1+ctx->Nintrinsics)];
        int    dq_dintrinsics_pool_int   [ctx->calibration_object_width_n*ctx->calibration_object_height_n];
        double* dq_dfxy = NULL;
        double* dq_dintrinsics_nocore = NULL;
        gradient_sparse_meta_t gradient_sparse_meta = {};

        int splined_intrinsics_grad_irun = 0;

        project(q_hypothesis,

                ctx->problem_details.do_optimize_intrinsics_core || ctx->problem_details.do_optimize_intrinsics_distortions ?
                  dq_dintrinsics_pool_double : NULL,
                ctx->problem_details.do_optimize_intrinsics_core || ctx->problem_details.do_optimize_intrinsics_distortions ?
                  dq_dintrinsics_pool_int : NULL,
                &dq_dfxy, &dq_dintrinsics_nocore, &gradient_sparse_meta,

                ctx->problem_details.do_optimize_extrinsics ?
                (mrcal_point3_t*)dq_drcamera : NULL,
                ctx->problem_details.do_optimize_extrinsics ?
                (mrcal_point3_t*)dq_dtcamera : NULL,
                ctx->problem_details.do_optimize_frames ?
                (mrcal_point3_t*)dq_drframe : NULL,
                ctx->problem_details.do_optimize_frames ?
                (mrcal_point3_t*)dq_dtframe : NULL,
                ctx->problem_details.do_optimize_calobject_warp ?
                (mrcal_point2_t*)dq_dcalobject_warp : NULL,

                // input
                intrinsics_all[i_cam_intrinsics],
                &camera_rt[i_cam_extrinsics], &frame_rt,
                ctx->calobject_warp == NULL ? NULL : &calobject_warp_local,
                i_cam_extrinsics < 0,
                ctx->lensmodel, &ctx->precomputed,
                ctx->calibration_object_spacing,
                ctx->calibration_object_width_n,
                ctx->calibration_object_height_n);

        for(int i_pt=0;
            i_pt < ctx->calibration_object_width_n*ctx->calibration_object_height_n;
            i_pt++, i_feature++)
        {
            const mrcal_point3_t* qx_qy_w__observed = &ctx->observations_board_pool[i_feature];
            double weight = qx_qy_w__observed->z;

            if(weight >= 0.0)
            {
                // I have my two measurements (dx, dy). I propagate their
                // gradient and store them
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = (q_hypothesis[i_pt].xy[i_xy] - qx_qy_w__observed->xyz[i_xy]) * weight;

                    if( ctx->reportFitMsg )
                    {
                        MSG("%s: obs/frame/cam_i/cam_e/dot: %d %d %d %d %d err: %g",
                            ctx->reportFitMsg,
                            i_observation_board, i_frame, i_cam_intrinsics, i_cam_extrinsics, i_pt, err);
                        continue;
                    }

                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    if( ctx->problem_details.do_optimize_intrinsics_core )
                    {
                        // fx,fy. x depends on fx only. y depends on fy only
                        STORE_JACOBIAN( i_var_intrinsics + i_xy,
                                        dq_dfxy[i_pt*2 + i_xy] *
                                        weight * SCALE_INTRINSICS_FOCAL_LENGTH );

                        // cx,cy. The gradients here are known to be 1. And x depends on cx only. And y depends on cy only
                        STORE_JACOBIAN( i_var_intrinsics + i_xy+2,
                                        weight * SCALE_INTRINSICS_CENTER_PIXEL );
                    }

                    if( ctx->problem_details.do_optimize_intrinsics_distortions )
                    {
                        if(gradient_sparse_meta.pool != NULL)
                        {
                            // u = stereographic(p)
                            // q = (u + deltau(u)) * f + c
                            //
                            // Intrinsics:
                            //   dq/diii = f ddeltau/diii
                            //
                            // ddeltau/diii = flatten(ABCDx[0..3] * ABCDy[0..3])
                            const int ivar0 = dq_dintrinsics_pool_int[splined_intrinsics_grad_irun] -
                                ( ctx->problem_details.do_optimize_intrinsics_core ? 0 : 4 );

                            const int     len   = gradient_sparse_meta.run_side_length;
                            const double* ABCDx = &gradient_sparse_meta.pool[len*2*splined_intrinsics_grad_irun + 0];
                            const double* ABCDy = &gradient_sparse_meta.pool[len*2*splined_intrinsics_grad_irun + len];

                            const int ivar_stridey = gradient_sparse_meta.ivar_stridey;
                            const double* fxy = &intrinsics_all[i_cam_intrinsics][0];

                            for(int iy=0; iy<len; iy++)
                                for(int ix=0; ix<len; ix++)
                                    STORE_JACOBIAN( i_var_intrinsics + ivar0 + iy*ivar_stridey + ix*2 + i_xy,
                                                    ABCDx[ix]*ABCDy[iy]*fxy[i_xy] *
                                                    weight * SCALE_DISTORTION );
                        }
                        else
                        {
                            for(int i=0; i<ctx->Nintrinsics-Ncore; i++)
                                STORE_JACOBIAN( i_var_intrinsics+Ncore_state + i,
                                                dq_dintrinsics_nocore[i_pt*2*(ctx->Nintrinsics-Ncore) +
                                                                       i_xy*(ctx->Nintrinsics-Ncore) +
                                                                       i] *
                                                weight * SCALE_DISTORTION );
                        }
                    }

                    if( ctx->problem_details.do_optimize_extrinsics )
                        if( i_cam_extrinsics >= 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0,
                                             dq_drcamera[i_pt][i_xy].xyz[0] *
                                             weight * SCALE_ROTATION_CAMERA,
                                             dq_drcamera[i_pt][i_xy].xyz[1] *
                                             weight * SCALE_ROTATION_CAMERA,
                                             dq_drcamera[i_pt][i_xy].xyz[2] *
                                             weight * SCALE_ROTATION_CAMERA);
                            STORE_JACOBIAN3( i_var_camera_rt + 3,
                                             dq_dtcamera[i_pt][i_xy].xyz[0] *
                                             weight * SCALE_TRANSLATION_CAMERA,
                                             dq_dtcamera[i_pt][i_xy].xyz[1] *
                                             weight * SCALE_TRANSLATION_CAMERA,
                                             dq_dtcamera[i_pt][i_xy].xyz[2] *
                                             weight * SCALE_TRANSLATION_CAMERA);
                        }

                    if( ctx->problem_details.do_optimize_frames )
                    {
                        STORE_JACOBIAN3( i_var_frame_rt + 0,
                                         dq_drframe[i_pt][i_xy].xyz[0] *
                                         weight * SCALE_ROTATION_FRAME,
                                         dq_drframe[i_pt][i_xy].xyz[1] *
                                         weight * SCALE_ROTATION_FRAME,
                                         dq_drframe[i_pt][i_xy].xyz[2] *
                                         weight * SCALE_ROTATION_FRAME);
                        STORE_JACOBIAN3( i_var_frame_rt + 3,
                                         dq_dtframe[i_pt][i_xy].xyz[0] *
                                         weight * SCALE_TRANSLATION_FRAME,
                                         dq_dtframe[i_pt][i_xy].xyz[1] *
                                         weight * SCALE_TRANSLATION_FRAME,
                                         dq_dtframe[i_pt][i_xy].xyz[2] *
                                         weight * SCALE_TRANSLATION_FRAME);
                    }

                    if( ctx->problem_details.do_optimize_calobject_warp )
                    {
                        STORE_JACOBIAN2( i_var_calobject_warp,
                                         dq_dcalobject_warp[i_pt][i_xy].x * weight * SCALE_CALOBJECT_WARP,
                                         dq_dcalobject_warp[i_pt][i_xy].y * weight * SCALE_CALOBJECT_WARP);
                    }

                    iMeasurement++;
                }
            }
            else
            {
                // Outlier.

                // This is arbitrary. I'm skipping this observation, so I don't
                // touch the projection results, and I set the measurement and
                // all its gradients to 0. I need to have SOME dependency on the
                // frame parameters to ensure a full-rank Hessian, so if we're
                // skipping all observations for this frame the system will
                // become singular. I don't currently handle this. libdogleg
                // will complain loudly, and add small diagonal L2
                // regularization terms
                for( int i_xy=0; i_xy<2; i_xy++ )
                {
                    const double err = 0.0;

                    if( ctx->reportFitMsg )
                    {
                        MSG( "%s: obs/frame/cam_i/cam_e/dot: %d %d %d %d %d err: %g",
                             ctx->reportFitMsg,
                             i_observation_board, i_frame, i_cam_intrinsics, i_cam_extrinsics, i_pt, err);
                        continue;
                    }

                    if(Jt) Jrowptr[iMeasurement] = iJacobian;
                    x[iMeasurement] = err;
                    norm2_error += err*err;

                    if( ctx->problem_details.do_optimize_intrinsics_core )
                    {
                        STORE_JACOBIAN( i_var_intrinsics + i_xy,   0.0 );
                        STORE_JACOBIAN( i_var_intrinsics + i_xy+2, 0.0 );
                    }

                    if( ctx->problem_details.do_optimize_intrinsics_distortions )
                    {
                        if(gradient_sparse_meta.pool != NULL)
                        {
                            const int ivar0 = dq_dintrinsics_pool_int[splined_intrinsics_grad_irun] -
                                ( ctx->problem_details.do_optimize_intrinsics_core ? 0 : 4 );
                            const int len          = gradient_sparse_meta.run_side_length;
                            const int ivar_stridey = gradient_sparse_meta.ivar_stridey;

                            for(int iy=0; iy<len; iy++)
                                for(int ix=0; ix<len; ix++)
                                    STORE_JACOBIAN( i_var_intrinsics + ivar0 + iy*ivar_stridey + ix*2 + i_xy, 0.0 );
                        }
                        else
                        {
                            for(int i=0; i<ctx->Nintrinsics-Ncore; i++)
                                STORE_JACOBIAN( i_var_intrinsics+Ncore_state + i, 0.0 );
                        }
                    }

                    if( ctx->problem_details.do_optimize_extrinsics )
                        if( i_cam_extrinsics >= 0 )
                        {
                            STORE_JACOBIAN3( i_var_camera_rt + 0, 0.0, 0.0, 0.0);
                            STORE_JACOBIAN3( i_var_camera_rt + 3, 0.0, 0.0, 0.0);
                        }

                    if( ctx->problem_details.do_optimize_frames )
                    {
                        // Arbitrary differences between the dimensions to keep
                        // my Hessian non-singular. This is 100% arbitrary. I'm
                        // skipping these measurements so these variables
                        // actually don't affect the computation at all
                        STORE_JACOBIAN3( i_var_frame_rt + 0, 0,0,0);
                        STORE_JACOBIAN3( i_var_frame_rt + 3, 0,0,0);
                    }

                    if( ctx->problem_details.do_optimize_calobject_warp )
                        STORE_JACOBIAN2( i_var_calobject_warp, 0.0, 0.0 );


                    iMeasurement++;
                }
            }
            if(gradient_sparse_meta.pool != NULL)
                splined_intrinsics_grad_irun++;
        }
    }

    // Handle all the point observations. This is VERY similar to the
    // board-observation loop above. Please consolidate
    for(int i_observation_point = 0;
        i_observation_point < ctx->Nobservations_point;
        i_observation_point++)
    {
        const mrcal_observation_point_t* observation = &ctx->observations_point[i_observation_point];

        const int i_cam_intrinsics = observation->i_cam_intrinsics;
        const int i_cam_extrinsics = observation->i_cam_extrinsics;
        const int i_point          = observation->i_point;
        const bool use_position_from_state =
            ctx->problem_details.do_optimize_frames &&
            i_point < ctx->Npoints - ctx->Npoints_fixed;

        const mrcal_point3_t* qx_qy_w__observed = &observation->px;
        double weight = qx_qy_w__observed->z;

        if(weight < 0.0)
        {
            // Outlier. Cost = 0. Jacobians are 0 too, but I must preserve the
            // structure
            const int i_var_intrinsics =
                mrcal_state_index_intrinsics(i_cam_intrinsics,
                                             ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                             ctx->Nframes,
                                             ctx->Npoints, ctx->Npoints_fixed,
                                             ctx->problem_details, ctx->lensmodel);
            // invalid if i_cam_extrinsics < 0, but unused in that case
            const int i_var_camera_rt  =
                mrcal_state_index_extrinsics(i_cam_extrinsics,
                                             ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                             ctx->Nframes,
                                             ctx->Npoints, ctx->Npoints_fixed,
                                             ctx->problem_details, ctx->lensmodel);
            const int i_var_point      =
                mrcal_state_index_points(i_point,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);

            // I have my two measurements (dx, dy). I propagate their
            // gradient and store them
            for( int i_xy=0; i_xy<2; i_xy++ )
            {
                if(Jt) Jrowptr[iMeasurement] = iJacobian;
                x[iMeasurement] = 0;

                if( ctx->problem_details.do_optimize_intrinsics_core )
                {
                    // fx,fy. x depends on fx only. y depends on fy only
                    STORE_JACOBIAN( i_var_intrinsics + i_xy, 0 );

                    // cx,cy. The gradients here are known to be 1. And x depends on cx only. And y depends on cy only
                    STORE_JACOBIAN( i_var_intrinsics + i_xy+2, 0);
                }

                if( ctx->problem_details.do_optimize_intrinsics_distortions )
                {
                    if( (ctx->problem_details.do_optimize_intrinsics_core || ctx->problem_details.do_optimize_intrinsics_distortions) &&
                        ctx->lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC )
                    {
                        // sparse gradient. This is an outlier, so it doesn't
                        // matter which points I say I depend on, as long as I
                        // pick the right number, and says that j=0. I pick the
                        // control points at the start because why not
                        const mrcal_LENSMODEL_SPLINED_STEREOGRAPHIC__config_t* config =
                            &ctx->lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config;
                        int len = config->order+1;
                        for(int i=0; i<len*len; i++)
                            STORE_JACOBIAN( i_var_intrinsics+Ncore_state + i, 0);
                    }
                    else
                        for(int i=0; i<ctx->Nintrinsics-Ncore; i++)
                            STORE_JACOBIAN( i_var_intrinsics+Ncore_state + i, 0);
                }

                if(i_cam_extrinsics >= 0 && ctx->problem_details.do_optimize_extrinsics )
                {
                    STORE_JACOBIAN3( i_var_camera_rt + 0, 0,0,0 );
                    STORE_JACOBIAN3( i_var_camera_rt + 3, 0,0,0 );
                }

                if( use_position_from_state )
                    STORE_JACOBIAN3( i_var_point, 0,0,0 );

                iMeasurement++;
            }

            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = 0;
            if(i_cam_extrinsics >= 0 && ctx->problem_details.do_optimize_extrinsics )
            {
                STORE_JACOBIAN3( i_var_camera_rt + 0, 0,0,0 );
                STORE_JACOBIAN3( i_var_camera_rt + 3, 0,0,0 );
            }
            if( use_position_from_state )
                STORE_JACOBIAN3( i_var_point, 0,0,0 );
            iMeasurement++;

            continue;
        }


        const int i_var_intrinsics =
            mrcal_state_index_intrinsics(i_cam_intrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
        // invalid if i_cam_extrinsics < 0, but unused in that case
        const int i_var_camera_rt  =
            mrcal_state_index_extrinsics(i_cam_extrinsics,
                                         ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                         ctx->Nframes,
                                         ctx->Npoints, ctx->Npoints_fixed,
                                         ctx->problem_details, ctx->lensmodel);
        const int i_var_point      =
            mrcal_state_index_points(i_point,
                                     ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                     ctx->Nframes,
                                     ctx->Npoints, ctx->Npoints_fixed,
                                     ctx->problem_details, ctx->lensmodel);
        mrcal_point3_t point_ref;
        if(use_position_from_state)
            unpack_solver_state_point_one(&point_ref, &packed_state[i_var_point]);
        else
            point_ref = ctx->points[i_point];


#warning "compute size(dq_dintrinsics_pool_double) correctly and maybe bounds-check"
        double dq_dintrinsics_pool_double[2*(1+ctx->Nintrinsics)];
        int    dq_dintrinsics_pool_int   [1];
        double* dq_dfxy                             = NULL;
        double* dq_dintrinsics_nocore               = NULL;
        gradient_sparse_meta_t gradient_sparse_meta = {};

        mrcal_point3_t dq_drcamera[2];
        mrcal_point3_t dq_dtcamera[2];
        mrcal_point3_t dq_dpoint  [2];

        // The array reference [-3] is intended, but the compiler throws a
        // warning. I silence it here
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
        mrcal_point2_t q_hypothesis;
        project(&q_hypothesis,

                ctx->problem_details.do_optimize_intrinsics_core || ctx->problem_details.do_optimize_intrinsics_distortions ?
                dq_dintrinsics_pool_double : NULL,
                ctx->problem_details.do_optimize_intrinsics_core || ctx->problem_details.do_optimize_intrinsics_distortions ?
                dq_dintrinsics_pool_int : NULL,
                &dq_dfxy, &dq_dintrinsics_nocore, &gradient_sparse_meta,

                ctx->problem_details.do_optimize_extrinsics ?
                dq_drcamera : NULL,
                ctx->problem_details.do_optimize_extrinsics ?
                dq_dtcamera : NULL,
                NULL, // frame rotation. I only have a point position
                use_position_from_state ? dq_dpoint : NULL,
                NULL,

                // input
                intrinsics_all[i_cam_intrinsics],
                &camera_rt[i_cam_extrinsics],

                // I only have the point position, so the 'rt' memory
                // points 3 back. The fake "r" here will not be
                // referenced
                (mrcal_pose_t*)(&point_ref.xyz[-3]),
                NULL,

                i_cam_extrinsics < 0,
                ctx->lensmodel, &ctx->precomputed,
                0,0,0);
#pragma GCC diagnostic pop

        // I have my two measurements (dx, dy). I propagate their
        // gradient and store them
        for( int i_xy=0; i_xy<2; i_xy++ )
        {
            const double err = (q_hypothesis.xy[i_xy] - qx_qy_w__observed->xyz[i_xy])*weight;

            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = err;
            norm2_error += err*err;

            if( ctx->problem_details.do_optimize_intrinsics_core )
            {
                // fx,fy. x depends on fx only. y depends on fy only
                STORE_JACOBIAN( i_var_intrinsics + i_xy,
                                dq_dfxy[i_xy] *
                                weight * SCALE_INTRINSICS_FOCAL_LENGTH );

                // cx,cy. The gradients here are known to be 1. And x depends on cx only. And y depends on cy only
                STORE_JACOBIAN( i_var_intrinsics + i_xy+2,
                                weight * SCALE_INTRINSICS_CENTER_PIXEL );
            }

            if( ctx->problem_details.do_optimize_intrinsics_distortions )
            {
                if(gradient_sparse_meta.pool != NULL)
                {
                    // u = stereographic(p)
                    // q = (u + deltau(u)) * f + c
                    //
                    // Intrinsics:
                    //   dq/diii = f ddeltau/diii
                    //
                    // ddeltau/diii = flatten(ABCDx[0..3] * ABCDy[0..3])
                    const int ivar0 = dq_dintrinsics_pool_int[0] -
                        ( ctx->problem_details.do_optimize_intrinsics_core ? 0 : 4 );

                    const int     len   = gradient_sparse_meta.run_side_length;
                    const double* ABCDx = &gradient_sparse_meta.pool[0];
                    const double* ABCDy = &gradient_sparse_meta.pool[len];

                    const int ivar_stridey = gradient_sparse_meta.ivar_stridey;
                    const double* fxy = &intrinsics_all[i_cam_intrinsics][0];

                    for(int iy=0; iy<len; iy++)
                        for(int ix=0; ix<len; ix++)
                        {
                            STORE_JACOBIAN( i_var_intrinsics + ivar0 + iy*ivar_stridey + ix*2 + i_xy,
                                            ABCDx[ix]*ABCDy[iy]*fxy[i_xy] *
                                            weight * SCALE_DISTORTION );
                        }
                }
                else
                {
                    for(int i=0; i<ctx->Nintrinsics-Ncore; i++)
                        STORE_JACOBIAN( i_var_intrinsics+Ncore_state + i,
                                        dq_dintrinsics_nocore[i_xy*(ctx->Nintrinsics-Ncore) +
                                                               i] *
                                        weight * SCALE_DISTORTION );
                }
            }

            if( ctx->problem_details.do_optimize_extrinsics )
                if( i_cam_extrinsics >= 0 )
                {
                    STORE_JACOBIAN3( i_var_camera_rt + 0,
                                     dq_drcamera[i_xy].xyz[0] *
                                     weight * SCALE_ROTATION_CAMERA,
                                     dq_drcamera[i_xy].xyz[1] *
                                     weight * SCALE_ROTATION_CAMERA,
                                     dq_drcamera[i_xy].xyz[2] *
                                     weight * SCALE_ROTATION_CAMERA);
                    STORE_JACOBIAN3( i_var_camera_rt + 3,
                                     dq_dtcamera[i_xy].xyz[0] *
                                     weight * SCALE_TRANSLATION_CAMERA,
                                     dq_dtcamera[i_xy].xyz[1] *
                                     weight * SCALE_TRANSLATION_CAMERA,
                                     dq_dtcamera[i_xy].xyz[2] *
                                     weight * SCALE_TRANSLATION_CAMERA);
                }

            if( use_position_from_state )
                STORE_JACOBIAN3( i_var_point,
                                 dq_dpoint[i_xy].xyz[0] *
                                 weight * SCALE_POSITION_POINT,
                                 dq_dpoint[i_xy].xyz[1] *
                                 weight * SCALE_POSITION_POINT,
                                 dq_dpoint[i_xy].xyz[2] *
                                 weight * SCALE_POSITION_POINT);

            iMeasurement++;
        }

        // Now the range normalization (make sure the range isn't
        // aphysically high or aphysically low). This code is copied from
        // project(). PLEASE consolidate
        void get_penalty(// out
                         double* penalty, double* dpenalty_ddistsq,

                         // in
                         // SIGNED distance. <0 means "behind the camera"
                         const double distsq)
        {
            const double maxsq = ctx->problem_constants->point_max_range*ctx->problem_constants->point_max_range;
            if(distsq > maxsq)
            {
                *penalty = weight * (distsq/maxsq - 1.0);
                *dpenalty_ddistsq = weight*(1. / maxsq);
                return;
            }

            const double minsq = ctx->problem_constants->point_min_range*ctx->problem_constants->point_min_range;
            if(distsq < minsq)
            {
                // too close OR behind the camera
                *penalty = weight*(1.0 - distsq/minsq);
                *dpenalty_ddistsq = weight*(-1. / minsq);
                return;
            }

            *penalty = *dpenalty_ddistsq = 0.0;
        }


        if(i_cam_extrinsics < 0)
        {
            double distsq =
                point_ref.x*point_ref.x +
                point_ref.y*point_ref.y +
                point_ref.z*point_ref.z;
            double penalty, dpenalty_ddistsq;
            if(model_supports_projection_behind_camera(ctx->lensmodel) ||
               point_ref.z > 0.0)
                get_penalty(&penalty, &dpenalty_ddistsq, distsq);
            else
            {
                get_penalty(&penalty, &dpenalty_ddistsq, -distsq);
                dpenalty_ddistsq *= -1.;
            }

            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = penalty;
            norm2_error += penalty*penalty;

            if( use_position_from_state )
            {
                double scale = 2.0 * dpenalty_ddistsq * SCALE_POSITION_POINT;
                STORE_JACOBIAN3( i_var_point,
                                 scale*point_ref.x,
                                 scale*point_ref.y,
                                 scale*point_ref.z );
            }

            iMeasurement++;
        }
        else
        {
            // I need to transform the point. I already computed
            // this stuff in project()...
            double Rc[3*3];
            double d_Rc_rc[9*3];

            mrcal_R_from_r(Rc,
                           d_Rc_rc,
                           camera_rt[i_cam_extrinsics].r.xyz);

            mrcal_point3_t pcam;
            mul_vec3_gen33t_vout(point_ref.xyz, Rc, pcam.xyz);
            add_vec(3, pcam.xyz, camera_rt[i_cam_extrinsics].t.xyz);

            double distsq =
                pcam.x*pcam.x +
                pcam.y*pcam.y +
                pcam.z*pcam.z;
            double penalty, dpenalty_ddistsq;
            if(model_supports_projection_behind_camera(ctx->lensmodel) ||
               pcam.z > 0.0)
                get_penalty(&penalty, &dpenalty_ddistsq, distsq);
            else
            {
                get_penalty(&penalty, &dpenalty_ddistsq, -distsq);
                dpenalty_ddistsq *= -1.;
            }

            if(Jt) Jrowptr[iMeasurement] = iJacobian;
            x[iMeasurement] = penalty;
            norm2_error += penalty*penalty;

            if( ctx->problem_details.do_optimize_extrinsics )
            {
                // pcam.x       = Rc[row0]*point*SCALE + tc
                // d(pcam.x)/dr = d(Rc[row0])/drc*point*SCALE
                // d(Rc[row0])/drc is 3x3 matrix at &d_Rc_rc[0]
                double d_ptcamx_dr[3];
                double d_ptcamy_dr[3];
                double d_ptcamz_dr[3];
                mul_vec3_gen33_vout( point_ref.xyz, &d_Rc_rc[9*0], d_ptcamx_dr );
                mul_vec3_gen33_vout( point_ref.xyz, &d_Rc_rc[9*1], d_ptcamy_dr );
                mul_vec3_gen33_vout( point_ref.xyz, &d_Rc_rc[9*2], d_ptcamz_dr );

                STORE_JACOBIAN3( i_var_camera_rt + 0,
                                 SCALE_ROTATION_CAMERA*
                                 2.0*dpenalty_ddistsq*( pcam.x*d_ptcamx_dr[0] +
                                                        pcam.y*d_ptcamy_dr[0] +
                                                        pcam.z*d_ptcamz_dr[0] ),
                                 SCALE_ROTATION_CAMERA*
                                 2.0*dpenalty_ddistsq*( pcam.x*d_ptcamx_dr[1] +
                                                        pcam.y*d_ptcamy_dr[1] +
                                                        pcam.z*d_ptcamz_dr[1] ),
                                 SCALE_ROTATION_CAMERA*
                                 2.0*dpenalty_ddistsq*( pcam.x*d_ptcamx_dr[2] +
                                                        pcam.y*d_ptcamy_dr[2] +
                                                        pcam.z*d_ptcamz_dr[2] ) );
                STORE_JACOBIAN3( i_var_camera_rt + 3,
                                 SCALE_TRANSLATION_CAMERA*
                                 2.0*dpenalty_ddistsq*pcam.x,
                                 SCALE_TRANSLATION_CAMERA*
                                 2.0*dpenalty_ddistsq*pcam.y,
                                 SCALE_TRANSLATION_CAMERA*
                                 2.0*dpenalty_ddistsq*pcam.z );
            }

            if( use_position_from_state )
                STORE_JACOBIAN3( i_var_point,
                                 SCALE_POSITION_POINT*
                                 2.0*dpenalty_ddistsq*(pcam.x*Rc[0] + pcam.y*Rc[3] + pcam.z*Rc[6]),
                                 SCALE_POSITION_POINT*
                                 2.0*dpenalty_ddistsq*(pcam.x*Rc[1] + pcam.y*Rc[4] + pcam.z*Rc[7]),
                                 SCALE_POSITION_POINT*
                                 2.0*dpenalty_ddistsq*(pcam.x*Rc[2] + pcam.y*Rc[5] + pcam.z*Rc[8]) );
            iMeasurement++;
        }
    }


    // regularization terms for the intrinsics. I favor smaller distortion
    // parameters
    if(!ctx->problem_details.do_skip_regularization &&
       modelHasCore_fxfycxcy(ctx->lensmodel) &&
       ( ctx->problem_details.do_optimize_intrinsics_distortions ||
         ctx->problem_details.do_optimize_intrinsics_core
         ))
    {
        // I want the total regularization cost to be low relative to the
        // other contributions to the cost. And I want each set of
        // regularization terms to weigh roughly the same. Let's say I want
        // regularization to account for ~ .5% of the other error
        // contributions:
        //
        //   Nmeasurements_rest*normal_pixel_error_sq * 0.005/2. =
        //   Nmeasurements_regularization_distortion *normal_regularization_distortion_error_sq  =
        //   Nmeasurements_regularization_centerpixel*normal_regularization_centerpixel_error_sq =
        //
        //   normal_regularization_distortion_error_sq  = (scale*normal_centerpixel_offset)^2
        //   normal_regularization_centerpixel_error_sq = (scale*normal_distortion_value  )^2
        //
        // Regularization introduces a bias to the solution. The
        // test-calibration-uncertainty test measures it, and barfs if it is too
        // high. The constants should be adjusted if that test fails.
        const bool dump_regularizaton_details = false;

        int    Nmeasurements_regularization_distortion  = ctx->Ncameras_intrinsics*(ctx->Nintrinsics-Ncore);
        int    Nmeasurements_regularization_centerpixel = ctx->Ncameras_intrinsics*2;

        int    Nmeasurements_nonregularization =
            ctx->Nmeasurements -
            Nmeasurements_regularization_distortion -
            Nmeasurements_regularization_centerpixel;

        double normal_pixel_error = 1.0;
        double expected_total_pixel_error_sq =
            (double)Nmeasurements_nonregularization *
            normal_pixel_error *
            normal_pixel_error;
        if(dump_regularizaton_details)
            MSG("expected_total_pixel_error_sq: %f", expected_total_pixel_error_sq);

        double scale_regularization_distortion =
            ({
                double normal_distortion_value = 2.0;

                double expected_regularization_distortion_error_sq_noscale =
                    (double)Nmeasurements_regularization_distortion *
                    normal_distortion_value *
                    normal_distortion_value;

                double scale_sq =
                    expected_total_pixel_error_sq * 0.005/2. / expected_regularization_distortion_error_sq_noscale;

                if(dump_regularizaton_details)
                    MSG("expected_regularization_distortion_error_sq: %f", expected_regularization_distortion_error_sq_noscale*scale_sq);

                sqrt(scale_sq);
            });
        double scale_regularization_centerpixel =
            ({
                double normal_centerpixel_offset = 500.0;

                double expected_regularization_centerpixel_error_sq_noscale =
                    (double)Nmeasurements_regularization_centerpixel *
                    normal_centerpixel_offset *
                    normal_centerpixel_offset;

                double scale_sq =
                    expected_total_pixel_error_sq * 0.005/2. / expected_regularization_centerpixel_error_sq_noscale;

                if(dump_regularizaton_details)
                    MSG("expected_regularization_centerpixel_error_sq: %f", expected_regularization_centerpixel_error_sq_noscale*scale_sq);

                sqrt(scale_sq);
            });

        for(int i_cam_intrinsics=0; i_cam_intrinsics<ctx->Ncameras_intrinsics; i_cam_intrinsics++)
        {
            const int i_var_intrinsics =
                mrcal_state_index_intrinsics(i_cam_intrinsics,
                                             ctx->Ncameras_intrinsics, ctx->Ncameras_extrinsics,
                                             ctx->Nframes,
                                             ctx->Npoints, ctx->Npoints_fixed,
                                             ctx->problem_details, ctx->lensmodel);

            if( ctx->problem_details.do_optimize_intrinsics_distortions)
            {
                for(int j=0; j<ctx->Nintrinsics-Ncore; j++)
                {
                    if(Jt) Jrowptr[iMeasurement] = iJacobian;

                    // This maybe should live elsewhere, but I put it here
                    // for now. Various distortion coefficients have
                    // different meanings, and should be regularized in
                    // different ways. Specific logic follows
                    double scale = scale_regularization_distortion;

                    if( MRCAL_LENSMODEL_IS_OPENCV(ctx->lensmodel.type) &&
                        ctx->lensmodel.type >= MRCAL_LENSMODEL_OPENCV8 &&
                        5 <= j && j <= 7 )
                    {
                        // The radial distortion in opencv is x_distorted =
                        // x*scale where r2 = norm2(xy - xyc) and
                        //
                        // scale = (1 + k0 r2 + k1 r4 + k4 r6)/(1 + k5 r2 + k6 r4 + k7 r6)
                        //
                        // Note that k2,k3 are tangential (NOT radial)
                        // distortion components. Note that the r6 factor in
                        // the numerator is only present for
                        // >=MRCAL_LENSMODEL_OPENCV5. Note that the denominator
                        // is only present for >= MRCAL_LENSMODEL_OPENCV8. The
                        // danger with a rational model is that it's
                        // possible to get into a situation where scale ~
                        // 0/0 ~ 1. This would have very poorly behaved
                        // derivatives. If all the rational coefficients are
                        // ~0, then the denominator is always ~1, and this
                        // problematic case can't happen. I favor that by
                        // regularizing the coefficients in the denominator
                        // more strongly
                        scale *= 5.;
                    }

                    double err       = scale*intrinsics_all[i_cam_intrinsics][j+Ncore];
                    x[iMeasurement]  = err;
                    norm2_error     += err*err;

                    STORE_JACOBIAN( i_var_intrinsics + Ncore_state + j,
                                    scale * SCALE_DISTORTION );

                    iMeasurement++;
                    if(dump_regularizaton_details)
                        MSG("regularization distortion: %g; norm2: %g", err, err*err);

                }
            }

            if( ctx->problem_details.do_optimize_intrinsics_core)
            {
                // And another regularization term: optical center should be
                // near the middle. This breaks the symmetry between moving the
                // center pixel coords and pitching/yawing the camera.
                double cx_target = 0.5 * (double)(ctx->imagersizes[i_cam_intrinsics*2 + 0] - 1);
                double cy_target = 0.5 * (double)(ctx->imagersizes[i_cam_intrinsics*2 + 1] - 1);

                double err = scale_regularization_centerpixel *
                    (intrinsics_all[i_cam_intrinsics][2] - cx_target);
                x[iMeasurement]  = err;
                norm2_error     += err*err;
                if(Jt) Jrowptr[iMeasurement] = iJacobian;
                STORE_JACOBIAN( i_var_intrinsics + 2,
                                scale_regularization_centerpixel * SCALE_INTRINSICS_CENTER_PIXEL );
                iMeasurement++;
                if(dump_regularizaton_details)
                    MSG("regularization center pixel off-center: %g; norm2: %g", err, err*err);

                err = scale_regularization_centerpixel *
                    (intrinsics_all[i_cam_intrinsics][3] - cy_target);
                x[iMeasurement]  = err;
                norm2_error     += err*err;
                if(Jt) Jrowptr[iMeasurement] = iJacobian;
                STORE_JACOBIAN( i_var_intrinsics + 3,
                                scale_regularization_centerpixel * SCALE_INTRINSICS_CENTER_PIXEL );
                iMeasurement++;
                if(dump_regularizaton_details)
                    MSG("regularization center pixel off-center: %g; norm2: %g", err, err*err);
            }
        }
    }


    // required to indicate the end of the jacobian matrix
    if( !ctx->reportFitMsg )
    {
        if(Jt) Jrowptr[iMeasurement] = iJacobian;
        if(iMeasurement != ctx->Nmeasurements)
        {
            MSG("Assertion (iMeasurement == ctx->Nmeasurements) failed: (%d != %d)",
                iMeasurement, ctx->Nmeasurements);
            assert(0);
        }
        if(iJacobian    != ctx->N_j_nonzero  )
        {
            MSG("Assertion (iJacobian    == ctx->N_j_nonzero  ) failed: (%d != %d)",
                iJacobian, ctx->N_j_nonzero);
            assert(0);
        }

        // MSG_IF_VERBOSE("RMS: %g", sqrt(norm2_error / ((double)ctx>Nmeasurements / 2.0)));
    }
}

bool mrcal_optimizer_callback(// out
                             // Each one of these output pointers may be NULL

                             // Shape (Nstate,)
                             double* p_packed,
                             // used only to confirm that the user passed-in the buffer they
                             // should have passed-in. The size must match exactly
                             int buffer_size_p_packed,

                             // Shape (Nmeasurements,)
                             double* x,
                             // used only to confirm that the user passed-in the buffer they
                             // should have passed-in. The size must match exactly
                             int buffer_size_x,

                             // output Jacobian. May be NULL if we don't need
                             // it. This is the unitless Jacobian, used by the
                             // internal optimization routines
                             cholmod_sparse* Jt,


                             // in

                             // intrinsics is a concatenation of the intrinsics core
                             // and the distortion params. The specific distortion
                             // parameters may vary, depending on lensmodel, so
                             // this is a variable-length structure
                             const double*             intrinsics,         // Ncameras_intrinsics * NlensParams
                             const mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM reference frame
                             const mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO reference frame
                             const mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                             const mrcal_point2_t*     calobject_warp,     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                             int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                             int Npoints, int Npoints_fixed, // at the end of points[]

                             const mrcal_observation_board_t* observations_board,

                             // All the board pixel observations, in order.
                             // .x, .y are the pixel observations
                             // .z is the weight of the observation. Most of the
                             // weights are expected to be 1.0, which implies
                             // that the noise on the observation is gaussian,
                             // independent on x,y, and has standard deviation
                             // of observed_pixel_uncertainty.
                             // observed_pixel_uncertainty scales inversely with
                             // the weight
                             const mrcal_point3_t* observations_board_pool,
                             int Nobservations_board,

                             const mrcal_observation_point_t* observations_point,
                             int Nobservations_point,
                             bool verbose,

                             mrcal_lensmodel_t lensmodel,
                             double observed_pixel_uncertainty,
                             const int* imagersizes, // Ncameras_intrinsics*2 of these

                             mrcal_problem_details_t          problem_details,
                             const mrcal_problem_constants_t* problem_constants,

                             double calibration_object_spacing,
                             int calibration_object_width_n,
                             int calibration_object_height_n)
{
    bool result = false;

    if(!modelHasCore_fxfycxcy(lensmodel))
        problem_details.do_optimize_intrinsics_core = false;

    if(!problem_details.do_optimize_intrinsics_core        &&
       !problem_details.do_optimize_intrinsics_distortions &&
       !problem_details.do_optimize_extrinsics            &&
       !problem_details.do_optimize_frames                &&
       !problem_details.do_optimize_calobject_warp)
    {
        MSG("Not optimizing any of our variables!");
        goto done;
    }

    if( calobject_warp == NULL && problem_details.do_optimize_calobject_warp )
    {
        MSG("ERROR: We're using the calibration object warp, so a value MUST be passed in.");
        goto done;
    }


    const int Nstate = mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                                        Nframes,
                                        Npoints, Npoints_fixed,
                                        problem_details,
                                        lensmodel);
    if( buffer_size_p_packed != Nstate*(int)sizeof(double) )
    {
        MSG("The buffer passed to fill-in p_packed has the wrong size. Needed exactly %d bytes, but got %d bytes",
            Nstate*(int)sizeof(double),buffer_size_p_packed);
        goto done;
    }

    int Nmeasurements = mrcal_num_measurements(Nobservations_board,
                                               Nobservations_point,
                                               calibration_object_width_n,
                                               calibration_object_height_n,
                                               Ncameras_intrinsics, Ncameras_extrinsics,
                                               Nframes,
                                               Npoints, Npoints_fixed,
                                               problem_details,
                                               lensmodel);
    int Nintrinsics = mrcal_lensmodel_num_params(lensmodel);
    int N_j_nonzero = mrcal_num_j_nonzero(Nobservations_board,
                                          Nobservations_point,
                                          calibration_object_width_n,
                                          calibration_object_height_n,
                                          Ncameras_intrinsics, Ncameras_extrinsics,
                                          Nframes,
                                          Npoints, Npoints_fixed,
                                          observations_board,
                                          observations_point,
                                          problem_details,
                                          lensmodel);

    if( buffer_size_x != Nmeasurements*(int)sizeof(double) )
    {
        MSG("The buffer passed to fill-in x has the wrong size. Needed exactly %d bytes, but got %d bytes",
            Nmeasurements*(int)sizeof(double),buffer_size_x);
        goto done;
    }

    const int Npoints_fromBoards =
        Nobservations_board *
        calibration_object_width_n*calibration_object_height_n;

#warning "Outliers only work with board observations for now. Point outliers could be implemented without a lot of trouble now"

    const callback_context_t ctx = {
        .intrinsics                 = intrinsics,
        .extrinsics_fromref         = extrinsics_fromref,
        .frames_toref               = frames_toref,
        .points                     = points,
        .calobject_warp             = calobject_warp,
        .Ncameras_intrinsics        = Ncameras_intrinsics,
        .Ncameras_extrinsics        = Ncameras_extrinsics,
        .Nframes                    = Nframes,
        .Npoints                    = Npoints,
        .Npoints_fixed              = Npoints_fixed,
        .observations_board         = observations_board,
        .observations_board_pool    = observations_board_pool,
        .Nobservations_board        = Nobservations_board,
        .observations_point         = observations_point,
        .Nobservations_point        = Nobservations_point,
        .verbose                    = verbose,
        .lensmodel                  = lensmodel,
        .imagersizes                = imagersizes,
        .problem_details            = problem_details,
        .problem_constants          = problem_constants,
        .calibration_object_spacing = calibration_object_spacing,
        .calibration_object_width_n = calibration_object_width_n  > 0 ? calibration_object_width_n  : 0,
        .calibration_object_height_n= calibration_object_height_n > 0 ? calibration_object_height_n : 0,
        .Nmeasurements              = Nmeasurements,
        .N_j_nonzero                = N_j_nonzero,
        .Nintrinsics                = Nintrinsics};
    _mrcal_precompute_lensmodel_data((mrcal_projection_precomputed_t*)&ctx.precomputed, lensmodel);

    pack_solver_state(p_packed,
                      lensmodel, intrinsics,
                      extrinsics_fromref,
                      frames_toref,
                      points,
                      calobject_warp,
                      problem_details,
                      Ncameras_intrinsics, Ncameras_extrinsics,
                      Nframes, Npoints-Npoints_fixed, Nstate);

    optimizer_callback(p_packed, x, Jt, &ctx);

    result = true;

done:
    return result;
}

mrcal_stats_t
mrcal_optimize( // out
                // Each one of these output pointers may be NULL

                // Shape (Nstate,)
                double* p_packed_final,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_p_packed_final,

                // Shape (Nmeasurements,)
                double* x_final,
                // used only to confirm that the user passed-in the buffer they
                // should have passed-in. The size must match exactly
                int buffer_size_x_final,

                // out, in
                //
                // if(_solver_context != NULL) then this is a persistent solver
                // context. The context is NOT freed on exit.
                // mrcal_free_context() should be called to release it
                //
                // if(*_solver_context != NULL), the given context is reused
                // if(*_solver_context == NULL), a context is created, and
                // returned here on exit
                void** _solver_context,

                // These are a seed on input, solution on output

                // intrinsics is a concatenation of the intrinsics core and the
                // distortion params. The specific distortion parameters may
                // vary, depending on lensmodel, so this is a variable-length
                // structure
                double*             intrinsics,         // Ncameras_intrinsics * NlensParams
                mrcal_pose_t*       extrinsics_fromref, // Ncameras_extrinsics of these. Transform FROM the reference frame
                mrcal_pose_t*       frames_toref,       // Nframes of these.    Transform TO the reference frame
                mrcal_point3_t*     points,             // Npoints of these.    In the reference frame
                mrcal_point2_t*     calobject_warp,     // 1 of these. May be NULL if !problem_details.do_optimize_calobject_warp

                // All the board pixel observations, in order.
                // .x, .y are the pixel observations
                // .z is the weight of the observation. Most of the weights are
                // expected to be 1.0, which implies that the noise on the
                // observation has standard deviation of
                // observed_pixel_uncertainty. observed_pixel_uncertainty scales
                // inversely with the weight.
                //
                // z<0 indicates that this is an outlier. This is respected on
                // input (even if skip_outlier_rejection). New outliers are
                // marked with z<0 on output, so this isn't const
                mrcal_point3_t* observations_board_pool,
                int Nobservations_board,

                // in
                int Ncameras_intrinsics, int Ncameras_extrinsics, int Nframes,
                int Npoints, int Npoints_fixed, // at the end of points[]

                const mrcal_observation_board_t* observations_board,
                const mrcal_observation_point_t* observations_point,
                int Nobservations_point,

                bool check_gradient,
                bool verbose,
                // Whether to try to find NEW outliers. The outliers given on
                // input are respected regardless
                const bool skip_outlier_rejection,

                mrcal_lensmodel_t lensmodel,
                double observed_pixel_uncertainty,
                const int* imagersizes, // Ncameras_intrinsics*2 of these
                mrcal_problem_details_t          problem_details,
                const mrcal_problem_constants_t* problem_constants,

                double calibration_object_spacing,
                int calibration_object_width_n,
                int calibration_object_height_n)
{
    if( calobject_warp == NULL && problem_details.do_optimize_calobject_warp )
    {
        MSG("ERROR: We're optimizing the calibration object warp, so a buffer with a seed MUST be passed in.");
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }

    if(!modelHasCore_fxfycxcy(lensmodel))
        problem_details.do_optimize_intrinsics_core = false;

    if(!problem_details.do_optimize_intrinsics_core        &&
       !problem_details.do_optimize_intrinsics_distortions &&
       !problem_details.do_optimize_extrinsics            &&
       !problem_details.do_optimize_frames                &&
       !problem_details.do_optimize_calobject_warp)
    {
        MSG("Warning: Not optimizing any of our variables");
    }

    dogleg_parameters2_t dogleg_parameters;
    dogleg_getDefaultParameters(&dogleg_parameters);
    dogleg_parameters.dogleg_debug = verbose ? DOGLEG_DEBUG_VNLOG : 0;

#warning update these parameters
    // These were derived empirically, seeking high accuracy, fast convergence
    // and without serious concern for performance. I looked only at a single
    // frame. Tweak them please
    dogleg_parameters.Jt_x_threshold = 0;
    dogleg_parameters.update_threshold = 1e-6;
    dogleg_parameters.trustregion_threshold = 0;
    dogleg_parameters.max_iterations = 300;
    // dogleg_parameters.trustregion_decrease_factor    = 0.1;
    // dogleg_parameters.trustregion_decrease_threshold = 0.15;
    // dogleg_parameters.trustregion_increase_factor    = 4.0
    // dogleg_parameters.trustregion_increase_threshold = 0.75;

    const int Npoints_fromBoards =
        Nobservations_board *
        calibration_object_width_n*calibration_object_height_n;

#warning "outliers only work with board observations for now"

    callback_context_t ctx = {
        .intrinsics                 = intrinsics,
        .extrinsics_fromref         = extrinsics_fromref,
        .frames_toref               = frames_toref,
        .points                     = points,
        .calobject_warp             = calobject_warp,
        .Ncameras_intrinsics        = Ncameras_intrinsics,
        .Ncameras_extrinsics        = Ncameras_extrinsics,
        .Nframes                    = Nframes,
        .Npoints                    = Npoints,
        .Npoints_fixed              = Npoints_fixed,
        .observations_board         = observations_board,
        .observations_board_pool    = observations_board_pool,
        .Nobservations_board        = Nobservations_board,
        .observations_point         = observations_point,
        .Nobservations_point        = Nobservations_point,
        .verbose                    = verbose,
        .lensmodel                  = lensmodel,
        .imagersizes                = imagersizes,
        .problem_details            = problem_details,
        .problem_constants          = problem_constants,
        .calibration_object_spacing = calibration_object_spacing,
        .calibration_object_width_n = calibration_object_width_n  > 0 ? calibration_object_width_n  : 0,
        .calibration_object_height_n= calibration_object_height_n > 0 ? calibration_object_height_n : 0,
        .Nmeasurements              = mrcal_num_measurements(Nobservations_board,
                                                             Nobservations_point,
                                                             calibration_object_width_n,
                                                             calibration_object_height_n,
                                                             Ncameras_intrinsics, Ncameras_extrinsics,
                                                             Nframes,
                                                             Npoints, Npoints_fixed,
                                                             problem_details,
                                                             lensmodel),
        .N_j_nonzero                = mrcal_num_j_nonzero(Nobservations_board,
                                                          Nobservations_point,
                                                          calibration_object_width_n,
                                                          calibration_object_height_n,
                                                          Ncameras_intrinsics, Ncameras_extrinsics,
                                                          Nframes,
                                                          Npoints, Npoints_fixed,
                                                          observations_board,
                                                          observations_point,
                                                          problem_details,
                                                          lensmodel),
        .Nintrinsics                = mrcal_lensmodel_num_params(lensmodel)};
    _mrcal_precompute_lensmodel_data((mrcal_projection_precomputed_t*)&ctx.precomputed, lensmodel);

    const int Nstate = mrcal_num_states(Ncameras_intrinsics, Ncameras_extrinsics,
                                        Nframes,
                                        Npoints, Npoints_fixed,
                                        problem_details,
                                        lensmodel);

    if( p_packed_final != NULL &&
        buffer_size_p_packed_final != Nstate*(int)sizeof(double) )
    {
        MSG("The buffer passed to fill-in p_packed_final has the wrong size. Needed exactly %d bytes, but got %d bytes",
            Nstate*(int)sizeof(double),buffer_size_p_packed_final);
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }
    if( x_final != NULL &&
        buffer_size_x_final != ctx.Nmeasurements*(int)sizeof(double) )
    {
        MSG("The buffer passed to fill-in x_final has the wrong size. Needed exactly %d bytes, but got %d bytes",
            ctx.Nmeasurements*(int)sizeof(double),buffer_size_x_final);
        return (mrcal_stats_t){.rms_reproj_error__pixels = -1.0};
    }


    dogleg_solverContext_t*  solver_context = NULL;
    // If I have a context already, I free it and create it anew later. Ideally
    // I'd reuse it, but then I'd need to make sure it's valid and such. Too
    // much work for now
    if(_solver_context != NULL && *_solver_context != NULL)
        dogleg_freeContext((dogleg_solverContext_t**)_solver_context);

    if(verbose)
        MSG("## Nmeasurements=%d, Nstate=%d", ctx.Nmeasurements, Nstate);
    if(ctx.Nmeasurements <= Nstate)
    {
        MSG("WARNING: problem isn't overdetermined: Nmeasurements=%d, Nstate=%d. Solver may not converge, and if it does, the results aren't reliable. Add more constraints and/or regularization",
            ctx.Nmeasurements, Nstate);
    }

    double packed_state[Nstate];
    pack_solver_state(packed_state,
                      lensmodel, intrinsics,
                      extrinsics_fromref,
                      frames_toref,
                      points,
                      calobject_warp,
                      problem_details,
                      Ncameras_intrinsics, Ncameras_extrinsics,
                      Nframes, Npoints-Npoints_fixed, Nstate);

    double norm2_error = -1.0;
    mrcal_stats_t stats = {.rms_reproj_error__pixels = -1.0 };

    if( !check_gradient )
    {
        stats.Noutliers = 0;

#warning "outliers for board observations only"
        int Nfeatures =
            Nobservations_board *
            calibration_object_width_n *
            calibration_object_height_n;
        for(int i=0; i<Nfeatures; i++)
            if(observations_board_pool[i].z < 0.0)
                stats.Noutliers++;

        if(verbose)
        {
            ctx.reportFitMsg = "Before";
#warning hook this up
            //        optimizer_callback(packed_state, NULL, NULL, &ctx);
        }
        ctx.reportFitMsg = NULL;


        double outliernessScale = -1.0;
        do
        {
            norm2_error = dogleg_optimize2(packed_state,
                                           Nstate, ctx.Nmeasurements, ctx.N_j_nonzero,
                                           (dogleg_callback_t*)&optimizer_callback, &ctx,
                                           &dogleg_parameters,
                                           &solver_context);
            if(_solver_context != NULL)
                *_solver_context = solver_context;

            if(norm2_error < 0)
                // libdogleg barfed. I quit out
                goto done;

#if 0
            // Not using dogleg_markOutliers() (for now?)

            if(outliernessScale < 0.0 && verbose)
                // These are for debug reporting
                dogleg_reportOutliers(getConfidence,
                                      &outliernessScale,
                                      2, Npoints_fromBoards,
                                      stats.Noutliers,
                                      solver_context->beforeStep, solver_context);
#endif

        } while( !skip_outlier_rejection &&
                 markOutliers(observations_board_pool,
                              &stats.Noutliers,
                              observations_board,
                              Nobservations_board,
                              calibration_object_width_n,
                              calibration_object_height_n,
                              solver_context->beforeStep->x,
                              observed_pixel_uncertainty,
                              verbose) &&
                 ({MSG("Threw out some outliers (have a total of %d now); going again", stats.Noutliers); true;}));

        // Done. I have the final state. I spit it back out
        unpack_solver_state( intrinsics,         // Ncameras_intrinsics of these
                             extrinsics_fromref, // Ncameras_extrinsics of these
                             frames_toref,       // Nframes of these
                             points,             // Npoints of these
                             calobject_warp,
                             packed_state,
                             lensmodel,
                             problem_details,
                             Ncameras_intrinsics, Ncameras_extrinsics,
                             Nframes, Npoints-Npoints_fixed, Nstate);
        if(verbose)
        {
            // Not using dogleg_markOutliers() (for now?)
#if 0
            // These are for debug reporting
            dogleg_reportOutliers(getConfidence,
                                  &outliernessScale,
                                  2, Npoints_fromBoards,
                                  stats.Noutliers,
                                  solver_context->beforeStep, solver_context);
#endif

            ctx.reportFitMsg = "After";
#warning hook this up
            //        optimizer_callback(packed_state, NULL, NULL, &ctx);
            if(!problem_details.do_skip_regularization)
            {
                double norm2_err_regularization = 0;
                int    Nmeasurements_regularization =
                    Ncameras_intrinsics *
                    num_regularization_terms_percamera(problem_details,
                                                       lensmodel);

                for(int i=0; i<Nmeasurements_regularization; i++)
                {
                    double x = solver_context->beforeStep->x[ctx.Nmeasurements-1 - i];
                    norm2_err_regularization += x*x;
                }

                double norm2_err_nonregularization = norm2_error - norm2_err_regularization;
                double ratio_regularization_cost   = norm2_err_regularization / norm2_error;

                for(int i=0; i<Nmeasurements_regularization; i++)
                {
                    double x = solver_context->beforeStep->x[ctx.Nmeasurements - Nmeasurements_regularization + i];
                    MSG("regularization %d: %f (squared: %f)", i, x, x*x);
                }
                MSG("norm2_error: %f",               norm2_error);
                MSG("norm2_err_regularization: %f",  norm2_err_regularization);
                MSG("regularization cost ratio: %g", ratio_regularization_cost);
            }
        }
    }
    else
        for(int ivar=0; ivar<Nstate; ivar++)
            dogleg_testGradient(ivar, packed_state,
                                Nstate, ctx.Nmeasurements, ctx.N_j_nonzero,
                                (dogleg_callback_t*)&optimizer_callback, &ctx);

    stats.rms_reproj_error__pixels =
        // /2 because I have separate x and y measurements
        sqrt(norm2_error / ((double)ctx.Nmeasurements / 2.0));

    if(p_packed_final)
        memcpy(p_packed_final, solver_context->beforeStep->p, Nstate*sizeof(double));
    if(x_final)
        memcpy(x_final, solver_context->beforeStep->x, ctx.Nmeasurements*sizeof(double));

 done:
    if(_solver_context == NULL && solver_context)
        dogleg_freeContext(&solver_context);

    return stats;
}

// frees a dogleg_solverContext_t. I don't want to #include <dogleg.h> here, so
// this is void
void mrcal_free_context(void** ctx)
{
    if( *ctx == NULL )
        return;

    dogleg_freeContext((dogleg_solverContext_t**)ctx);
}
