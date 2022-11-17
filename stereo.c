#include <math.h>

#include "mrcal.h"
#include "util.h"

// The equivalent function in Python is _rectified_resolution_python() in
// stereo.py
//
// Documentation is in the docstring of mrcal.rectified_resolution()
bool mrcal_rectified_resolution( // output and input
                                 // > 0: use given value
                                 // < 0: autodetect and scale
                                 double* pixels_per_deg_az,
                                 double* pixels_per_deg_el,

                                 // input
                                 const mrcal_lensmodel_t*     lensmodel,
                                 const double*                intrinsics,
                                 double                       az_fov_deg,
                                 double                       el_fov_deg,
                                 const mrcal_point2_t*        azel0_deg,
                                 const double*                R_cam0_rect0,
                                 const mrcal_lensmodel_type_t rectification_model)
{
    // Get the rectified image resolution
    if( *pixels_per_deg_az < 0 ||
        *pixels_per_deg_el < 0)
    {
        const mrcal_point2_t azel0 = {.x = azel0_deg->x * M_PI/180.,
                                      .y = azel0_deg->y * M_PI/180. };

        // I need to compute the resolution of the rectified images. I try to
        // match the resolution of the cameras. I just look at camera0. If your
        // two cameras are different, pass in the pixels_per_deg yourself
        //
        // I look at the center of the stereo field of view. There I have q =
        // project(v) where v is a unit projection vector. I compute dq/dth where
        // th is an angular perturbation applied to v.
        double v[3];
        double dv_dazel[3*2];
        if(rectification_model == MRCAL_LENSMODEL_LATLON)
            mrcal_unproject_latlon((mrcal_point3_t*)v, (mrcal_point2_t*)dv_dazel,
                                   &azel0,
                                   1,
                                   (double[]){1.,1.,0.,0.});
        else if(rectification_model == MRCAL_LENSMODEL_LONLAT)
            mrcal_unproject_lonlat((mrcal_point3_t*)v, (mrcal_point2_t*)dv_dazel,
                                   &azel0,
                                   1,
                                   (double[]){1.,1.,0.,0.});
        else if(rectification_model == MRCAL_LENSMODEL_PINHOLE)
        {

            mrcal_point2_t q0_normalized = {.x = tan(azel0.x),
                                            .y = tan(azel0.y)};
            mrcal_unproject_pinhole((mrcal_point3_t*)v, (mrcal_point2_t*)dv_dazel,
                                    &q0_normalized,
                                    1,
                                    (double[]){1.,1.,0.,0.});
            // dq/dth = dtanth/dth = 1/cos^2(th)
            double cos_az0 = cos(azel0.x);
            double cos_el0 = cos(azel0.y);

            for(int i=0; i<3; i++)
            {
                dv_dazel[2*i + 0] /= cos_az0*cos_az0;
                dv_dazel[2*i + 1] /= cos_el0*cos_el0;
            }
        }
        else
        {
            MSG("Unsupported rectification model");
            return false;

        }

        double v0[3];
        mrcal_rotate_point_R(v0, NULL,NULL,
                             R_cam0_rect0,
                             v);

        // dv0_dazel  = nps.matmult(R_cam0_rect0, dv_dazel)
        double dv0_daz[3] = {};
        double dv0_del[3] = {};
        for(int j=0; j<3; j++)
            for(int k=0; k<3; k++)
            {
                dv0_daz[j] += R_cam0_rect0[j*3+k]*dv_dazel[2*k + 0];
                dv0_del[j] += R_cam0_rect0[j*3+k]*dv_dazel[2*k + 1];
            }

        mrcal_point2_t qdummy;
        mrcal_point3_t dq_dv0[2];
        // _,dq_dv0,_ = mrcal.project(v0, *model.intrinsics(), get_gradients = True)
        mrcal_project(&qdummy,dq_dv0,NULL,
                      (const mrcal_point3_t*)v0, 1, lensmodel, intrinsics);

        // More complex method that's probably not any better
        //
        // if False:
        //     // I rotate my v to a coordinate system where u = rotate(v) is [0,0,1].
        //     // Then u = [a,b,0] are all orthogonal to v. So du/dth = [cos, sin, 0].
        //     // I then have dq/dth = dq/dv dv/du [cos, sin, 0]t
        //     // ---> dq/dth = dq/dv dv/du[:,:2] [cos, sin]t = M [cos,sin]t
        //     //
        //     // norm2(dq/dth) = [cos,sin] MtM [cos,sin]t is then an ellipse with the
        //     // eigenvalues of MtM giving me the best and worst sensitivities. I can
        //     // use mrcal.worst_direction_stdev() to find the densest direction. But I
        //     // actually know the directions I care about, so I evaluate them
        //     // independently for the az and el directions
        //     def rotation_any_v_to_z(v):
        //         r'''Return any rotation matrix that maps the given unit vector v to [0,0,1]'''
        //         z = v
        //         if np.abs(v[0]) < .9:
        //             x = np.array((1,0,0));
        //         else:
        //             x = np.array((0,1,0));
        //         x -= nps.inner(x,v)*v
        //         x /= nps.mag(x);
        //         y = np.cross(z,x);
        //         return nps.cat(x,y,z);
        //     Ruv = rotation_any_v_to_z(v0);
        //     M = nps.matmult(dq_dv0, nps.transpose(Ruv[:2,:]));
        //     // I pick the densest direction: highest |dq/dth|
        //     pixels_per_rad = mrcal.worst_direction_stdev( nps.matmult( nps.transpose(M),M) );

        // dq_dazel = nps.matmult(dq_dv0, dv0_dazel)
        double dq_daz[2] =
            { dq_dv0[0].x*dv0_daz[0] + dq_dv0[0].y*dv0_daz[1] + dq_dv0[0].z*dv0_daz[2],
              dq_dv0[1].x*dv0_daz[0] + dq_dv0[1].y*dv0_daz[1] + dq_dv0[1].z*dv0_daz[2] };
        double dq_del[2] =
            { dq_dv0[0].x*dv0_del[0] + dq_dv0[0].y*dv0_del[1] + dq_dv0[0].z*dv0_del[2],
              dq_dv0[1].x*dv0_del[0] + dq_dv0[1].y*dv0_del[1] + dq_dv0[1].z*dv0_del[2] };

        if(*pixels_per_deg_az < 0)
        {
            double dq_daz_norm2 = 0.;
            for(int i=0; i<2; i++) dq_daz_norm2 += dq_daz[i]*dq_daz[i];
            double pixels_per_deg_az_have = sqrt(dq_daz_norm2)*M_PI/180.;
            *pixels_per_deg_az *= -pixels_per_deg_az_have;
        }

        if(*pixels_per_deg_el < 0)
        {
            double dq_del_norm2 = 0.;
            for(int i=0; i<2; i++) dq_del_norm2 += dq_del[i]*dq_del[i];
            double pixels_per_deg_el_have = sqrt(dq_del_norm2)*M_PI/180.;
            *pixels_per_deg_el *= -pixels_per_deg_el_have;
        }
    }

    // I now have the desired pixels_per_deg
    //
    // With LENSMODEL_LATLON or LENSMODEL_LONLAT we have even angular spacing, so
    // q = f th + c -> dq/dth = f everywhere. I can thus compute the rectified
    // image size and adjust the resolution accordingly
    //
    // With LENSMODEL_PINHOLE this is much more complex, so this function just
    // leaves the desired pixels_per_deg as it is
    if(rectification_model == MRCAL_LENSMODEL_LATLON ||
       rectification_model == MRCAL_LENSMODEL_LONLAT)
    {
        int Naz = (int)round(az_fov_deg * (*pixels_per_deg_az));
        int Nel = (int)round(el_fov_deg * (*pixels_per_deg_el));

        *pixels_per_deg_az = (double)Naz/az_fov_deg;
        *pixels_per_deg_el = (double)Nel/el_fov_deg;
    }

    return true;
}
