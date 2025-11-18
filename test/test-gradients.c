// Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
// Government sponsorship acknowledged. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "../mrcal.h"

#define SPLINED_NX 11
#define SPLINED_NY 8

static
bool modelHasCore_fxfycxcy( const mrcal_lensmodel_t* lensmodel )
{
    mrcal_lensmodel_metadata_t meta = mrcal_lensmodel_metadata(lensmodel);
    return meta.has_core;
}

int main(int argc, char* argv[] )
{
    const char* usage = "Usage: %s LENSMODEL_XXX [problem-selections problem-selections ...]\n"
        "\n"
        "The lensmodels are given as the expected strings. Splined stereographic models\n"
        "MUST be given as either of\n"
        "  LENSMODEL_SPLINED_STEREOGRAPHIC_2\n"
        "  LENSMODEL_SPLINED_STEREOGRAPHIC_3\n"
        "\n"
        "problem-selections are a list of parameters we're optimizing. This is some set of\n"
        "  intrinsic-core\n"
        "  intrinsic-distortions\n"
        "  extrinsics\n"
        "  frames\n"
        "  calobject-warp\n"
        "\n"
        "If no selections are given, we optimize everything. Otherwise, we start with an empty\n"
        "mrcal_problem_selections_t, and each argument sets a bit\n";

    if( argc >= 2 && argv[1][0] == '-' )
    {
        printf(usage, argv[0]);
        return 0;
    }

    int iarg = 1;
    if( iarg >= argc )
    {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }

    mrcal_lensmodel_t lensmodel = {.type = mrcal_lensmodel_type_from_name(argv[iarg]) };
    if( !mrcal_lensmodel_type_is_valid(lensmodel.type) )
    {
#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"
        fprintf(stderr, "Lens model name '%s' unknown. I only know about ("
                        MRCAL_LENSMODEL_LIST( QUOTED_LIST_WITH_COMMA )
                ")\n", argv[iarg]);
        return 1;
    }
    if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC)
    {
        if(0 == strcmp(argv[iarg], "LENSMODEL_SPLINED_STEREOGRAPHIC_2"))
            lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.order = 2;
        else if(0 == strcmp(argv[iarg], "LENSMODEL_SPLINED_STEREOGRAPHIC_3"))
            lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.order = 3;
        else
        {
            fprintf(stderr, "A splined stereographic model must be specified as exactly one of \"LENSMODEL_SPLINED_STEREOGRAPHIC_2\" or \"LENSMODEL_SPLINED_STEREOGRAPHIC_3\". Giving up\n");
            return 1;
        }
    }


    iarg++;

    mrcal_problem_selections_t problem_selections = {};
    if(iarg >= argc)
    {
        // Default. Turn on everything
        problem_selections = ((mrcal_problem_selections_t) { .do_optimize_intrinsics_core         = true,
                                                             .do_optimize_intrinsics_distortions  = true,
                                                             .do_optimize_extrinsics              = true,
                                                             .do_optimize_frames                  = true,
                                                             .do_optimize_calobject_warp          = true,
                                                             .do_apply_regularization             = true,
                                                             .do_apply_regularization_unity_cam01 = true});
    }
    else
    {
        problem_selections = ((mrcal_problem_selections_t){.do_apply_regularization             = true,
                                                           .do_apply_regularization_unity_cam01 = true});

        for(; iarg < argc; iarg++)
        {
            if( 0 == strcmp(argv[iarg], "intrinsic-core") )
            {
                problem_selections.do_optimize_intrinsics_core = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "intrinsic-distortions") )
            {
                problem_selections.do_optimize_intrinsics_distortions = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "extrinsics") )
            {
                problem_selections.do_optimize_extrinsics = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "frames") )
            {
                problem_selections.do_optimize_frames = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "calobject-warp" ) )
            {
                problem_selections.do_optimize_calobject_warp = true;
                continue;
            }

            fprintf(stderr, "Unknown optimization variable '%s'. Giving up.\n\n", argv[iarg]);
            fprintf(stderr, usage, argv[0]);
            return 1;
        }
    }

    mrcal_pose_t extrinsics[] =
        { { .r = { .xyz = {  .01,   .1,    .02}},  .t = { .xyz = { -2.3, 0.2, 0.1}}},
          { .r = { .xyz = { -.02,  .03,   .002}},  .t = { .xyz = { -4.0, 0.1,-0.3}}}};

    mrcal_pose_t frames[] =
        { { .r = { .xyz = { -.1,    .52,  -.13}},  .t = { .xyz = { 1.3, 0.1, 10.2}}},
          { .r = { .xyz = {  .90,   .24,   .135}}, .t = { .xyz = { 0.7, 0.1, 20.3}}},
          { .r = { .xyz = {  .80,   .52,   .335}}, .t = { .xyz = { 0.7, 0.6, 30.4}}},
          { .r = { .xyz = {  .20,  -.22,   .75}},  .t = { .xyz = { 3.1, 6.3, 10.4}}}};
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);

    mrcal_point3_t points[] =
        { {.xyz = {-5.3,   2.3, 20.4}},
          {.xyz = {-15.3, -3.2, 200.4}}};
    mrcal_calobject_warp_t calobject_warp = {.x2 = 0.001, .y2 = -0.005};

    int Npoints      = sizeof(points)/sizeof(points[0]);
    int Npoints_fixed = 1;

#define calibration_object_width_n  10
#define calibration_object_height_n 9

    // The observations of chessboards and of discrete points
    const mrcal_observation_board_t observations_board[] =
        { {.icam = { .intrinsics = 0, .extrinsics = -1 }, .iframe = 0},
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .iframe = 0},
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .iframe = 1},
          {.icam = { .intrinsics = 0, .extrinsics = -1 }, .iframe = 2},
          {.icam = { .intrinsics = 0, .extrinsics = -1 }, .iframe = 3},
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .iframe = 3},
          {.icam = { .intrinsics = 2, .extrinsics =  1 }, .iframe = 3} };
    mrcal_observation_point_t observations_point[] =
        { {.icam = { .intrinsics = 0, .extrinsics = -1 }, .i_point = 0},
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .i_point = 0},
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .i_point = 1},
          {.icam = { .intrinsics = 2, .extrinsics =  1 }, .i_point = 1} };

#define Nobservations_board ((int)(sizeof(observations_board)/sizeof(observations_board[0])))
#define Nobservations_point ((int)(sizeof(observations_point)/sizeof(observations_point[0])))

    mrcal_point3_t observations_board_pool[Nobservations_board][calibration_object_width_n*calibration_object_height_n] = {};
    mrcal_point3_t observations_point_pool[Nobservations_point] = {};

    // fill observations with arbitrary data
    for(int i=0; i<Nobservations_board; i++)
        for(int j=0; j<calibration_object_height_n; j++)
            for(int k=0; k<calibration_object_width_n; k++)
            {
                observations_board_pool[i][calibration_object_width_n*j + k].x =
                    1000.0 + (double)k - 10.0*(double)j + (double)(i*j*k);
                observations_board_pool[i][calibration_object_width_n*j + k].y =
                    1000.0 - (double)k + 30.0*(double)j - (double)(i*j*k);
                observations_board_pool[i][calibration_object_width_n*j + k].z =
                    1. / (double)(1 << ((i+j+k) % 3));
            }
    for(int i=0; i<Nobservations_point; i++)
    {
        observations_point_pool[i].x = 1100.0 + (double)i*20.0;
        observations_point_pool[i].y = 800.0  - (double)i*12.0;
        observations_point_pool[i].z = 1. / (double)(1 << (i % 3));
    }

    // Observations of triangulated points
    int Nobservations_point_triangulated;
    mrcal_observation_point_triangulated_t* observations_point_triangulated;
    mrcal_observation_point_triangulated_t _observations_point_triangulated[] =
        { // convergent
          {.icam = { .intrinsics = 0, .extrinsics = -1 }, .last_in_set = false, .px = {.xyz = {  0., 0., 1.}} },
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .last_in_set = true,  .px = {.xyz = {-0.1, 0., 1.}} },
          // divergent
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .last_in_set = false, .px = {.xyz = { 0.2, 0., 1.}} },
          {.icam = { .intrinsics = 0, .extrinsics = -1 }, .last_in_set = true,  .px = {.xyz = { 0.,  0., 1.}} },

          // convergent
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .last_in_set = false, .px = {.xyz = {  0., -0.02, 1.}} },
          {.icam = { .intrinsics = 2, .extrinsics =  1 }, .last_in_set = true,  .px = {.xyz = {-0.1,  0.01, 1.}} },
          // divergent
          {.icam = { .intrinsics = 2, .extrinsics =  1 }, .last_in_set = false, .px = {.xyz = { 0.21,  0.01, 1.}} },
          {.icam = { .intrinsics = 1, .extrinsics =  0 }, .last_in_set = true,  .px = {.xyz = { -0.1,  0.03, 1.}} } };

    if(!(problem_selections.do_optimize_intrinsics_core ||
         problem_selections.do_optimize_intrinsics_distortions) &&
       problem_selections.do_optimize_extrinsics)
    {
        Nobservations_point_triangulated = (int)(sizeof(_observations_point_triangulated)/sizeof(_observations_point_triangulated[0]));
        observations_point_triangulated = _observations_point_triangulated;
    }
    else
    {
        Nobservations_point_triangulated = 0;
        observations_point_triangulated = NULL;
    }

    // simple camera calibration case
    int Ncameras_extrinsics = sizeof(extrinsics)/sizeof(extrinsics[0]);
    int Ncameras_intrinsics = Ncameras_extrinsics + 1;
    int imagersizes[Ncameras_intrinsics*2];
    for(int i=0; i<Ncameras_intrinsics*2; i++)
        imagersizes[i] = 1000 + 10*i;

    if(lensmodel.type == MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC )
    {
        // the order was already set above
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx           = SPLINED_NX;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny           = SPLINED_NY;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.fov_x_deg    = 200;
    }

    if(!modelHasCore_fxfycxcy(&lensmodel))
        // There is no core
        problem_selections.do_optimize_intrinsics_core = false;

    int Nintrinsics = mrcal_lensmodel_num_params(&lensmodel);
    int Ndistortion = Nintrinsics;
    if(modelHasCore_fxfycxcy(&lensmodel))
        Ndistortion -= 4;
    double intrinsics[Ncameras_intrinsics][Nintrinsics];

    intrinsics[0][0] = 2000.3; // focal_xy [0]
    intrinsics[0][1] = 1900.5; // focal_xy [1]
    intrinsics[0][2] = 1800.3; // center_xy[0]
    intrinsics[0][3] = 1790.2; // center_xy[1]

    intrinsics[1][0] = 2100.2; // focal_xy [0]
    intrinsics[1][1] = 2130.4; // focal_xy [1]
    intrinsics[1][2] = 1830.3; // center_xy[0]
    intrinsics[1][3] = 1810.2; // center_xy[1]

    intrinsics[2][0] = 2503.8; // focal_xy [0]
    intrinsics[2][1] = 2730.4; // focal_xy [1]
    intrinsics[2][2] = 1730.3; // center_xy[0]
    intrinsics[2][3] = 1610.2; // center_xy[1]

    if(Ncameras_intrinsics != 3)
    {
        fprintf(stderr, "Unexpected Ncameras_intrinsics=%d; should be 3\n", Ncameras_intrinsics);
        return 1;
    }

    if(lensmodel.type != MRCAL_LENSMODEL_SPLINED_STEREOGRAPHIC )
        for(int i=0; i<Ncameras_intrinsics; i++)
            for(int j=0; j<Ndistortion; j++)
                intrinsics[i][4 + j] = 0.1 + 0.05 * (double)(i + Ncameras_intrinsics*j);
    else
    {
        const double intrinsics_cam[3][SPLINED_NX*SPLINED_NY*2] =
            { { 2.017284705,1.242204557,2.053514381,1.214368063,2.0379067,1.212609628,
              2.033278227,1.183689487,2.040018023,1.188554431,2.069146825,1.196304649,
              2.085708658,1.186478238,2.065787617,1.163377825,2.086372192,1.138856716,
              2.131609155,1.125678279,2.128812604,1.120525061,2.00841491,1.21864154,
              2.024522768,1.239588759,2.034947935,1.19814079,2.065474055,1.19897294,
              2.044562395,1.200557321,2.087714092,1.160440038,2.086478691,1.151822407,
              2.112862582,1.147567288,2.101575718,1.146312256,2.10056469,1.157015327,
              2.113488262,1.111679758,2.019837901,1.244168216,2.025847768,1.215633807,
              2.041980956,1.205751212,2.075077056,1.199787561,2.070877831,1.203261678,
              2.067244278,1.184705736,2.082225077,1.185558149,2.091519961,1.17501817,
              2.120258866,1.137775228,2.120020747,1.152409316,2.121870228,1.113069319,
              2.043650555,1.247757041,2.019661062,1.230723629,2.067917203,1.209753396,
              2.035034141,1.219514335,2.045350268,1.178474255,2.046346049,1.169372592,
              2.097839998,1.194836758,2.112724938,1.172186377,2.110996386,1.154899043,
              2.128456883,1.133228404,2.122513384,1.131717886,2.044279196,1.233288366,
              2.023197297,1.230118703,2.06707694,1.199998862,2.044147271,1.191607451,
              2.058590053,1.1677808,2.081593501,1.182074581,2.08663053,1.159156329,
              2.084329086,1.157727374,2.073666528,1.151261965,2.114290905,1.144710519,
              2.138600912,1.119405248,2.016299528,1.206147494,2.029434175,1.211507857,
              2.057936091,1.19801196,2.035691392,1.174035359,2.084718618,1.203604729,
              2.085910021,1.158385222,2.080800068,1.150199852,2.087991586,1.162019581,
              2.094754507,1.151061493,2.115144642,1.154299799,2.107014195,1.127608146,
              2.005632475,1.238607328,2.02033157,1.202101384,2.061021703,1.214868271,
              2.043015135,1.211903685,2.05291186,1.188092787,2.09486724,1.179277314,
              2.078230124,1.186273023,2.077743945,1.148028845,2.081634186,1.131207467,
              2.112936851,1.126412871,2.113220553,1.114991063,2.017901873,1.244588667,
              2.051238803,1.201855728,2.043256406,1.216674722,2.035286046,1.178380907,
              2.08028318,1.178783085,2.051214271,1.173560417,2.059298121,1.182414688,
              2.094607679,1.177960959,2.086998287,1.147371259,2.12029442,1.138197348,
              2.138994213, 1.114846113},
            { 1.06737740498, 1.6278468389,  1.04927724765, 1.51277313216, 1.140998713,
              1.64403651896, 1.13541712362, 1.58101290967, 1.01718000045, 1.64287620869,
              1.11361588299, 1.58562781834, 1.1249749456,  1.51358613443, 1.14154450187,
              1.53057822388, 1.07784995662, 1.52510019005, 1.07014506296, 1.55479755376,
              1.01529274549, 1.59920448979, 1.06143236749, 1.58319827983, 1.00436011058,
              1.51233755113, 1.01292179023, 1.58840896169, 1.00685687888, 1.52955719805,
              1.06465100058, 1.60818510619, 1.09886182159, 1.589774718,   1.14235849863,
              1.57829578195, 1.08806595959, 1.57143781124, 1.03380760733, 1.56756788589,
              1.15640149366, 1.51745865903, 1.11989565371, 1.61113757948, 1.03046275975,
              1.56617249759, 1.05250166776, 1.6046848029,  1.02155197629, 1.60360581214,
              1.14114822675, 1.59000691575, 1.04830519414, 1.55286555953, 1.14484836996,
              1.53666079324, 1.09093978875, 1.51118399783, 1.08816194872, 1.54831753256,
              1.14935426194, 1.58730066916, 1.07997492114, 1.61022385144, 1.02168586725,
              1.6117772929,  1.08348083495, 1.62870511085, 1.00356734056, 1.54009290088,
              1.14993318757, 1.51261651222, 1.04681750165, 1.5821295221,  1.02737545682,
              1.5161299378,  1.04819389723, 1.54544720811, 1.10734490646, 1.62497460764,
              1.15086980433, 1.61968157874, 1.1197854042,  1.61710573116, 1.06295192732,
              1.52076036635, 1.04537939713, 1.58124859431, 1.03873446216, 1.57593405562,
              1.08666730896, 1.50813429145, 1.08016666775, 1.60069580125, 1.09260626796,
              1.61490325951, 1.07178791414, 1.57549609414, 1.07653597225, 1.51997714073,
              1.08264827694, 1.54091416138, 1.10289533399, 1.57523523827, 1.14253235963,
              1.6400700236,  1.06727237613, 1.53048587923, 1.11388964445, 1.62188883074,
              1.00256325257, 1.60008158905, 1.0868287319,  1.56726486721, 1.02159583401,
              1.59430164822, 1.1082270496,  1.50320112003, 1.05669874149, 1.61587109293,
              1.01766782939, 1.50663833776, 1.01421326826, 1.59681074813, 1.12224656331,
              1.62739303356, 1.02349175353, 1.57542271719, 1.15116671623, 1.57859368309,
              1.07719890006, 1.50675727417, 1.05410949808, 1.57584705406, 1.10758561587,
              1.57769472021, 1.04490472053, 1.5749043095,  1.0117995382,  1.63631842987,
              1.11537231841, 1.57981033106, 1.11236104094, 1.57533523567, 1.10052323251,
              1.51012243485, 1.04703859869, 1.5553025274,  1.06739886162, 1.61897641967,
              1.11608029783, 1.60614287099, 1.13614591325, 1.62652898321, 1.05596273978,
              1.60860965489, 1.06814986136, 1.57216976471, 1.05831640024, 1.58265849957,
              1.13733800215, 1.63824382463, 1.07916158505, 1.57074686246, 1.14380502334,
              1.51865523259, 1.06073374227, 1.55309119673, 1.04795568165, 1.60213038148,
              1.032439757,   1.50002407206, 1.00882762874, 1.58048933908, 1.15745735978,
              1.60724228971},
            { 1.02850179, 1.39745551, 0.73726311, 1.69722052, 0.7550817 ,
              1.59747822, 1.29574076, 1.93030849, 1.14721218, 1.71520154,
              1.03597672, 1.51389362, 0.77656641, 1.48857032, 1.21088787,
              1.64292209, 0.72150304, 1.5549714 , 1.25738621, 1.5156016 ,
              1.22164819, 1.69169524, 1.07892812, 1.40699013, 1.09403431,
              1.27118646, 0.57935248, 1.63232869, 1.38704362, 1.44397763,
              1.12014003, 1.58924238, 0.99272589, 1.6468557 , 1.14345147,
              1.52400491, 0.97436734, 1.52800453, 1.01705856, 1.37970847,
              1.27534043, 1.5610139 , 1.00825577, 1.4588616 , 1.24044827,
              1.55950213, 1.01063005, 1.43452465, 1.00472963, 1.62935646,
              1.11358954, 1.62786706, 1.11417026, 1.82755386, 1.21462754,
              1.53690111, 1.27958452, 1.37424464, 0.87862022, 1.7159569 ,
              1.34872155, 1.59497626, 0.61519145, 1.83481903, 1.28818211,
              2.06814798, 0.82880808, 1.69984126, 1.02371652, 1.64751215,
              1.58595348, 1.44365259, 1.10533183, 1.32605266, 1.10303567,
              1.30063005, 1.04169354, 1.90731939, 1.27847707, 1.8069966 ,
              1.0023919 , 1.30377844, 1.40248959, 1.52355389, 1.12024891,
              1.43681461, 1.16362021, 1.40156562, 1.00476566, 1.54506525,
              1.03704937, 1.71558348, 1.36660295, 1.43759866, 1.02229004,
              1.70653186, 1.08695356, 1.5366513 , 1.15967137, 1.582927  ,
              0.62855878, 1.51186917, 0.98216024, 1.64983457, 0.83781813,
              1.52653551, 1.06465434, 1.89579514, 0.82618455, 1.95597482,
              0.69156817, 1.51465488, 0.88425426, 1.52009533, 1.19902578,
              1.73166606, 1.22977288, 1.32246789, 1.00377336, 1.64651133,
              1.13926565, 1.38873322, 1.09961382, 1.67929291, 1.20107604,
              1.92667805, 0.8847579 , 1.75747935, 1.10356075, 1.50637047,
              1.45764175, 1.78125187, 1.26161492, 1.48525964, 1.05666625,
              1.46772942, 1.48943231, 1.4739409 , 1.00178741, 1.9376378 ,
              1.20700385, 1.48035217, 1.10464828, 1.83603468, 1.38030157,
              1.31444212, 1.06362372, 1.31788033, 1.12020553, 1.78206147,
              1.25868741, 1.22631558, 1.02886757, 1.80444816, 0.9465482 ,
              1.4809315 , 1.35208411, 1.95717077, 1.42476196, 1.23148041,
              1.21363688, 1.32485208, 1.0595888 , 1.76879512, 1.13493746,
              1.08887589, 0.9687434 , 1.57496242, 0.94266393, 1.53375802,
              0.98972987, 1.86917315, 1.15851243, 1.8772311 , 0.99256034,
              1.50009109 } };
        for(int i=0; i<Ncameras_intrinsics; i++)
            memcpy(&intrinsics[i][4],
                   intrinsics_cam[i],
                   sizeof(intrinsics_cam[i]));
    }

    printf("## Ncameras_intrinsics = %d\n", Ncameras_intrinsics);
    printf("## Ncameras_extrinsics = %d\n", Ncameras_extrinsics);
    printf("## Intrinsics: %d variables per camera (%d for the core, %d for the rest; %d total). Starts at variable %d\n",
           (problem_selections.do_optimize_intrinsics_core        ? 4           : 0) +
           (problem_selections.do_optimize_intrinsics_distortions ? Ndistortion : 0),
           (problem_selections.do_optimize_intrinsics_core        ? 4           : 0),
           (problem_selections.do_optimize_intrinsics_distortions ? Ndistortion : 0),
           Ncameras_intrinsics*((problem_selections.do_optimize_intrinsics_core        ? 4           : 0) +
                                (problem_selections.do_optimize_intrinsics_distortions ? Ndistortion : 0)),
           mrcal_state_index_intrinsics(0, Ncameras_intrinsics, Ncameras_extrinsics,
                                        Nframes,
                                        Npoints, Npoints_fixed, Nobservations_board,
                                        problem_selections,
                                        &lensmodel));
    printf("## Extrinsics: %d variables per camera for all cameras except camera 0 (%d total). Starts at variable %d\n",
           (problem_selections.do_optimize_extrinsics ? 6                     : 0),
           (problem_selections.do_optimize_extrinsics ? 6*Ncameras_extrinsics : 0),
           mrcal_state_index_extrinsics(0, Ncameras_intrinsics, Ncameras_extrinsics,
                                        Nframes,
                                        Npoints, Npoints_fixed, Nobservations_board,
                                        problem_selections,
                                        &lensmodel));
    printf("## Frames: %d variables per frame (%d total). Starts at variable %d\n",
           (problem_selections.do_optimize_frames ? 6         : 0),
           (problem_selections.do_optimize_frames ? 6*Nframes : 0),
           mrcal_state_index_frames(0, Ncameras_intrinsics, Ncameras_extrinsics,
                                    Nframes,
                                    Npoints, Npoints_fixed, Nobservations_board,
                                    problem_selections,
                                    &lensmodel));
    printf("## Discrete points: %d variables per point (%d total). Starts at variable %d\n",
           (problem_selections.do_optimize_frames ? 3                        : 0),
           (problem_selections.do_optimize_frames ? 3*(Npoints-Npoints_fixed) : 0),
           mrcal_state_index_points(0, Ncameras_intrinsics, Ncameras_extrinsics,
                                    Nframes,
                                    Npoints, Npoints_fixed, Nobservations_board,
                                    problem_selections,
                                    &lensmodel));

    int istate0_calobject_warp =
        mrcal_state_index_calobject_warp(Ncameras_intrinsics, Ncameras_extrinsics,
                                         Nframes,
                                         Npoints, Npoints_fixed, Nobservations_board,
                                         problem_selections,
                                         &lensmodel);
    printf("## calobject_warp: %d variables. Starts at variable %d\n",
           (istate0_calobject_warp >= 0 ? 2 : 0),
           istate0_calobject_warp);
    int Nmeasurements_boards         = mrcal_num_measurements_boards(Nobservations_board,
                                                                     calibration_object_width_n,
                                                                     calibration_object_height_n);
    int Nmeasurements_points         = mrcal_num_measurements_points(Nobservations_point);
    int Nmeasurements_points_triangulated = mrcal_num_measurements_points_triangulated(observations_point_triangulated,
                                                                                       Nobservations_point_triangulated);
    int Nmeasurements_regularization = mrcal_num_measurements_regularization(Ncameras_intrinsics, Ncameras_extrinsics,
                                                                             Nframes,
                                                                             Npoints, Npoints_fixed, Nobservations_board,
                                                                             problem_selections,
                                                                             &lensmodel);

    int imeasurement = 0;
    printf("## Measurement calobjects: %d measurements. Starts at measurement %d\n",
           Nmeasurements_boards, imeasurement);
    imeasurement += Nmeasurements_boards;

    printf("## Measurement points: %d measurements. Starts at measurement %d\n",
           Nmeasurements_points, imeasurement);
    imeasurement += Nmeasurements_points;

    printf("## Measurement points-triangulated: %d measurements. Starts at measurement %d\n",
           Nmeasurements_points_triangulated, imeasurement);
    imeasurement += Nmeasurements_points_triangulated;

    printf("## Measurement regularization: %d measurements. Starts at measurement %d\n",
           Nmeasurements_regularization, imeasurement);
    imeasurement += Nmeasurements_regularization;

    mrcal_problem_constants_t problem_constants =
        { .point_min_range =  30.0,
          .point_max_range = 180.0};

    mrcal_stats_t stats =
        mrcal_optimize( NULL,0, NULL,0,
                        (double*)intrinsics,
                        extrinsics,
                        frames,
                        points,
                        &calobject_warp,

                        Ncameras_intrinsics,Ncameras_extrinsics,
                        Nframes, Npoints, Npoints_fixed,
                        observations_board,
                        observations_point,
                        Nobservations_board,
                        Nobservations_point,

                        observations_point_triangulated,
                        Nobservations_point_triangulated,
                        (mrcal_point3_t*)observations_board_pool,
                        (mrcal_point3_t*)observations_point_pool,

                        &lensmodel,
                        imagersizes,
                        problem_selections,
                        &problem_constants,

                        1.2,
                        calibration_object_width_n,
                        calibration_object_height_n,

                        false,
                        true);

    if(stats.rms_reproj_error__pixels < 0)
        return 1;
    return 0;
}
