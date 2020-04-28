#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "mrcal.h"

int main(int argc, char* argv[] )
{
    const char* usage = "Usage: %s LENSMODEL_XXX [problem-details problem-details ...]\n"
        "\n"
        "problem-details are a list of parameters we're optimizing. This is some set of\n"
        "  intrinsic-core\n"
        "  intrinsic-distortions\n"
        "  extrinsics\n"
        "  frames\n"
        "  calobject-warp\n"
        "\n"
        "If no details are given, we optimize everything. Otherwise, we start with an empty\n"
        "mrcal_problem_details_t, and each argument sets a bit\n";

    if( argc >= 2 && argv[1][0] == '-' )
    {
        printf(usage, argv[0]);
        return 0;
    }

    mrcal_problem_details_t problem_details = {};


    int iarg = 1;
    if( iarg >= argc )
    {
        fprintf(stderr, usage, argv[0]);
        return 1;
    }

    lensmodel_t lensmodel = {.type = mrcal_lensmodel_type_from_name(argv[iarg]) };
    if( !mrcal_lensmodel_type_is_valid(lensmodel.type) )
    {
#define QUOTED_LIST_WITH_COMMA(s,n) "'" #s "',"
        fprintf(stderr, "Lens model name '%s' unknown. I only know about ("
                        LENSMODEL_LIST( QUOTED_LIST_WITH_COMMA )
                ")\n", argv[iarg]);
        return 1;
    }
    iarg++;

    if(iarg >= argc)
        problem_details = DO_OPTIMIZE_ALL;
    else
        for(; iarg < argc; iarg++)
        {

            if( 0 == strcmp(argv[iarg], "intrinsic-core") )
            {
                problem_details.do_optimize_intrinsic_core = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "intrinsic-distortions") )
            {
                problem_details.do_optimize_intrinsic_distortions = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "extrinsics") )
            {
                problem_details.do_optimize_extrinsics = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "frames") )
            {
                problem_details.do_optimize_frames = true;
                continue;
            }
            if( 0 == strcmp(argv[iarg], "calobject-warp" ) )
            {
                problem_details.do_optimize_calobject_warp = true;
                continue;
            }

            fprintf(stderr, "Unknown optimization variable '%s'. Giving up.\n\n", argv[iarg]);
            fprintf(stderr, usage, argv[0]);
            return 1;
        }


    pose_t extrinsics[] =
        { { .r = { .xyz = {  .01,   .1,    .02}},  .t = { .xyz = { 2.3, 0.2, 0.1}}}};

    pose_t frames[] =
        { { .r = { .xyz = { -.1,    .52,  -.13}},  .t = { .xyz = { 1.3, 0.1, 10.2}}},
          { .r = { .xyz = {  .90,   .24,   .135}}, .t = { .xyz = { 0.7, 0.1, 20.3}}},
          { .r = { .xyz = {  .80,   .52,   .335}}, .t = { .xyz = { 0.7, 0.6, 30.4}}},
          { .r = { .xyz = {  .20,  -.22,   .75}},  .t = { .xyz = { 3.1, 6.3, 10.4}}}};
    int Nframes  = sizeof(frames)    /sizeof(frames[0]);

    point3_t points[] =
        { {.xyz = {-5.3,   2.3, 20.4}},
          {.xyz = {-15.3, -3.2, 200.4}}};
    point2_t calobject_warp = {.x = 0.001, .y = -0.005};

    int Npoints = sizeof(points)/sizeof(points[0]);

#define calibration_object_width_n 10 /* arbitrary */

    point3_t observations_px      [6][calibration_object_width_n*calibration_object_width_n] = {};
    point3_t observations_point_px[4] = {};
    // How many of the observations we want to actually use. Can be fewer than
    // defined in the above arrays if we're testing something
#define NobservationsBoard 6
#define NobservationsPoint 4

    // fill observations with arbitrary data
    for(int i=0; i<NobservationsBoard; i++)
        for(int j=0; j<calibration_object_width_n; j++)
            for(int k=0; k<calibration_object_width_n; k++)
            {
                observations_px[i][calibration_object_width_n*j + k].x =
                    1000.0 + (double)k - 10.0*(double)j + (double)(i*j*k);
                observations_px[i][calibration_object_width_n*j + k].y =
                    1000.0 - (double)k + 30.0*(double)j - (double)(i*j*k);
                observations_px[i][calibration_object_width_n*j + k].z =
                    1. / (double)(1 << ((i+j+k) % 3));
            }
    for(int i=0; i<NobservationsPoint; i++)
    {
        observations_point_px[i].x = 1100.0 + (double)i*20.0;
        observations_point_px[i].y = 800.0  - (double)i*12.0;
        observations_point_px[i].z = 1. / (double)(1 << (i % 3));
    }

    // The observations of chessboards and of discrete points
    observation_board_t observations_board[] =
        { {.i_camera = 0, .i_frame = 0, .px = observations_px[0]},
          {.i_camera = 1, .i_frame = 0, .px = observations_px[1]},
          {.i_camera = 1, .i_frame = 1, .px = observations_px[2]},
          {.i_camera = 0, .i_frame = 2, .px = observations_px[3]},
          {.i_camera = 0, .i_frame = 3, .px = observations_px[4]},
          {.i_camera = 1, .i_frame = 3, .px = observations_px[5]} };
    observation_point_t observations_point[] =
        { {.i_camera = 0, .i_point = 0, .px = observations_point_px[0]},
          {.i_camera = 1, .i_point = 0, .px = observations_point_px[1]},
          {.i_camera = 0, .i_point = 1, .px = observations_point_px[2], .dist = 18.0},
          {.i_camera = 1, .i_point = 1, .px = observations_point_px[3], .dist = 180.0} };

    int Ncameras = sizeof(extrinsics)/sizeof(extrinsics[0]) + 1;
    int imagersizes[Ncameras*2];
    for(int i=0; i<Ncameras*2; i++)
        imagersizes[i] = 1000 + 10*i;

    if(lensmodel.type == LENSMODEL_SPLINED_STEREOGRAPHIC )
    {
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.spline_order = 3;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Nx           = 11;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.Ny           = 8;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.fov_x_deg    = 200;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.cx           = 1499.5;
        lensmodel.LENSMODEL_SPLINED_STEREOGRAPHIC__config.cy           = 999.5;
    }

    if(!mrcal_modelHasCore_fxfycxcy(lensmodel))
        // There is no core
        problem_details.do_optimize_intrinsic_core = false;

    int Nintrinsics = mrcal_getNlensParams(lensmodel);
    int Ndistortion = Nintrinsics;
    if(mrcal_modelHasCore_fxfycxcy(lensmodel))
        Ndistortion -= 4;
    double intrinsics[Ncameras * Nintrinsics];

    if(lensmodel.type != LENSMODEL_SPLINED_STEREOGRAPHIC )
    {

        intrinsics_core_t* intrinsics_core = (intrinsics_core_t*)intrinsics;
        intrinsics_core->focal_xy [0] = 2000.3;
        intrinsics_core->focal_xy [1] = 1900.5;
        intrinsics_core->center_xy[0] = 1800.3;
        intrinsics_core->center_xy[1] = 1790.2;

        intrinsics_core = (intrinsics_core_t*)(&intrinsics[Nintrinsics]);
        intrinsics_core->focal_xy [0] = 2100.2;
        intrinsics_core->focal_xy [1] = 2130.4;
        intrinsics_core->center_xy[0] = 1830.3;
        intrinsics_core->center_xy[1] = 1810.2;

        for(int i=0; i<Ncameras; i++)
            for(int j=0; j<Ndistortion; j++)
                intrinsics[Nintrinsics * i + 4 + j] = 0.1 + 0.05 * (double)(i + Ncameras*j);
    }
    else
    {
        const double intrinsics_cam0[] =
            { 2017.284705,1242.204557,2053.514381,1214.368063,2037.9067,1212.609628,
              2033.278227,1183.689487,2040.018023,1188.554431,2069.146825,1196.304649,
              2085.708658,1186.478238,2065.787617,1163.377825,2086.372192,1138.856716,
              2131.609155,1125.678279,2128.812604,1120.525061,2008.41491,1218.64154,
              2024.522768,1239.588759,2034.947935,1198.14079,2065.474055,1198.97294,
              2044.562395,1200.557321,2087.714092,1160.440038,2086.478691,1151.822407,
              2112.862582,1147.567288,2101.575718,1146.312256,2100.56469,1157.015327,
              2113.488262,1111.679758,2019.837901,1244.168216,2025.847768,1215.633807,
              2041.980956,1205.751212,2075.077056,1199.787561,2070.877831,1203.261678,
              2067.244278,1184.705736,2082.225077,1185.558149,2091.519961,1175.01817,
              2120.258866,1137.775228,2120.020747,1152.409316,2121.870228,1113.069319,
              2043.650555,1247.757041,2019.661062,1230.723629,2067.917203,1209.753396,
              2035.034141,1219.514335,2045.350268,1178.474255,2046.346049,1169.372592,
              2097.839998,1194.836758,2112.724938,1172.186377,2110.996386,1154.899043,
              2128.456883,1133.228404,2122.513384,1131.717886,2044.279196,1233.288366,
              2023.197297,1230.118703,2067.07694,1199.998862,2044.147271,1191.607451,
              2058.590053,1167.7808,2081.593501,1182.074581,2086.63053,1159.156329,
              2084.329086,1157.727374,2073.666528,1151.261965,2114.290905,1144.710519,
              2138.600912,1119.405248,2016.299528,1206.147494,2029.434175,1211.507857,
              2057.936091,1198.01196,2035.691392,1174.035359,2084.718618,1203.604729,
              2085.910021,1158.385222,2080.800068,1150.199852,2087.991586,1162.019581,
              2094.754507,1151.061493,2115.144642,1154.299799,2107.014195,1127.608146,
              2005.632475,1238.607328,2020.33157,1202.101384,2061.021703,1214.868271,
              2043.015135,1211.903685,2052.91186,1188.092787,2094.86724,1179.277314,
              2078.230124,1186.273023,2077.743945,1148.028845,2081.634186,1131.207467,
              2112.936851,1126.412871,2113.220553,1114.991063,2017.901873,1244.588667,
              2051.238803,1201.855728,2043.256406,1216.674722,2035.286046,1178.380907,
              2080.28318,1178.783085,2051.214271,1173.560417,2059.298121,1182.414688,
              2094.607679,1177.960959,2086.998287,1147.371259,2120.29442,1138.197348,
              2138.994213, 1114.846113};
        const double intrinsics_cam1[] =
            { 1067.37740498, 1627.8468389,  1049.27724765, 1512.77313216, 1140.998713,
              1644.03651896, 1135.41712362, 1581.01290967, 1017.18000045, 1642.87620869,
              1113.61588299, 1585.62781834, 1124.9749456,  1513.58613443, 1141.54450187,
              1530.57822388, 1077.84995662, 1525.10019005, 1070.14506296, 1554.79755376,
              1015.29274549, 1599.20448979, 1061.43236749, 1583.19827983, 1004.36011058,
              1512.33755113, 1012.92179023, 1588.40896169, 1006.85687888, 1529.55719805,
              1064.65100058, 1608.18510619, 1098.86182159, 1589.774718,   1142.35849863,
              1578.29578195, 1088.06595959, 1571.43781124, 1033.80760733, 1567.56788589,
              1156.40149366, 1517.45865903, 1119.89565371, 1611.13757948, 1030.46275975,
              1566.17249759, 1052.50166776, 1604.6848029,  1021.55197629, 1603.60581214,
              1141.14822675, 1590.00691575, 1048.30519414, 1552.86555953, 1144.84836996,
              1536.66079324, 1090.93978875, 1511.18399783, 1088.16194872, 1548.31753256,
              1149.35426194, 1587.30066916, 1079.97492114, 1610.22385144, 1021.68586725,
              1611.7772929,  1083.48083495, 1628.70511085, 1003.56734056, 1540.09290088,
              1149.93318757, 1512.61651222, 1046.81750165, 1582.1295221,  1027.37545682,
              1516.1299378,  1048.19389723, 1545.44720811, 1107.34490646, 1624.97460764,
              1150.86980433, 1619.68157874, 1119.7854042,  1617.10573116, 1062.95192732,
              1520.76036635, 1045.37939713, 1581.24859431, 1038.73446216, 1575.93405562,
              1086.66730896, 1508.13429145, 1080.16666775, 1600.69580125, 1092.60626796,
              1614.90325951, 1071.78791414, 1575.49609414, 1076.53597225, 1519.97714073,
              1082.64827694, 1540.91416138, 1102.89533399, 1575.23523827, 1142.53235963,
              1640.0700236,  1067.27237613, 1530.48587923, 1113.88964445, 1621.88883074,
              1002.56325257, 1600.08158905, 1086.8287319,  1567.26486721, 1021.59583401,
              1594.30164822, 1108.2270496,  1503.20112003, 1056.69874149, 1615.87109293,
              1017.66782939, 1506.63833776, 1014.21326826, 1596.81074813, 1122.24656331,
              1627.39303356, 1023.49175353, 1575.42271719, 1151.16671623, 1578.59368309,
              1077.19890006, 1506.75727417, 1054.10949808, 1575.84705406, 1107.58561587,
              1577.69472021, 1044.90472053, 1574.9043095,  1011.7995382,  1636.31842987,
              1115.37231841, 1579.81033106, 1112.36104094, 1575.33523567, 1100.52323251,
              1510.12243485, 1047.03859869, 1555.3025274,  1067.39886162, 1618.97641967,
              1116.08029783, 1606.14287099, 1136.14591325, 1626.52898321, 1055.96273978,
              1608.60965489, 1068.14986136, 1572.16976471, 1058.31640024, 1582.65849957,
              1137.33800215, 1638.24382463, 1079.16158505, 1570.74686246, 1143.80502334,
              1518.65523259, 1060.73374227, 1553.09119673, 1047.95568165, 1602.13038148,
              1032.439757,   1500.02407206, 1008.82762874, 1580.48933908, 1157.45735978,
              1607.24228971};
        memcpy(&intrinsics[Nintrinsics*0], intrinsics_cam0, sizeof(intrinsics_cam0));
        memcpy(&intrinsics[Nintrinsics*1], intrinsics_cam1, sizeof(intrinsics_cam1));
    }


    printf("## Ncameras = %d\n", Ncameras);
    printf("## Intrinsics: %d variables per camera (%d for the core, %d for the rest; %d total). Starts at variable %d\n",
           (problem_details.do_optimize_intrinsic_core        ? 4           : 0) +
           (problem_details.do_optimize_intrinsic_distortions ? Ndistortion : 0),
           (problem_details.do_optimize_intrinsic_core        ? 4           : 0),
           (problem_details.do_optimize_intrinsic_distortions ? Ndistortion : 0),
           Ncameras*((problem_details.do_optimize_intrinsic_core        ? 4           : 0) +
                     (problem_details.do_optimize_intrinsic_distortions ? Ndistortion : 0)),
           mrcal_state_index_intrinsics(0, problem_details, lensmodel));
    printf("## Extrinsics: %d variables per camera for all cameras except camera 0 (%d total). Starts at variable %d\n",
           (problem_details.do_optimize_extrinsics ? 6              : 0),
           (problem_details.do_optimize_extrinsics ? 6*(Ncameras-1) : 0),
           mrcal_state_index_camera_rt(1, Ncameras, problem_details, lensmodel));
    printf("## Frames: %d variables per frame (%d total). Starts at variable %d\n",
           (problem_details.do_optimize_frames ? 6         : 0),
           (problem_details.do_optimize_frames ? 6*Nframes : 0),
           mrcal_state_index_frame_rt(0, Ncameras, problem_details, lensmodel));
    printf("## Discrete points: %d variables per point (%d total). Starts at variable %d\n",
           (problem_details.do_optimize_frames ? 3         : 0),
           (problem_details.do_optimize_frames ? 3*Npoints : 0),
           mrcal_state_index_point(0, Nframes, Ncameras, problem_details, lensmodel));
    printf("## calobject_warp: %d variables. Starts at variable %d\n",
           (problem_details.do_optimize_calobject_warp ? 2 : 0),
           mrcal_state_index_calobject_warp(Npoints, Nframes, Ncameras, problem_details, lensmodel));
    int Nmeasurements_boards         = mrcal_getNmeasurements_boards(NobservationsBoard, calibration_object_width_n);
    int Nmeasurements_points         = mrcal_getNmeasurements_points(observations_point, NobservationsPoint);
    int Nmeasurements_regularization = mrcal_getNmeasurements_regularization(Ncameras, problem_details, lensmodel);
    printf("## Measurement calobjects: %d measurements. Starts at measurement %d\n",
           Nmeasurements_boards, 0);
    printf("## Measurement points: %d measurements. Starts at measurement %d\n",
           Nmeasurements_points, Nmeasurements_boards);
    printf("## Measurement regularization: %d measurements. Starts at measurement %d\n",
           Nmeasurements_regularization, Nmeasurements_boards+Nmeasurements_points);

    const double roi[] = { 1000., 1000., 400., 400.,
                            900., 1200., 300., 800. };
    mrcal_optimize( NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                    intrinsics,
                    extrinsics,
                    frames,
                    points,
                    &calobject_warp,
                    Ncameras, Nframes, Npoints,

                    observations_board,
                    NobservationsBoard,

                    observations_point,
                    NobservationsPoint,

                    true,
                    0, NULL,
                    roi,
                    false,
                    true,
                    lensmodel,
                    1.0,
                    imagersizes,
                    problem_details,

                    1.0, calibration_object_width_n);

    return 0;
}
