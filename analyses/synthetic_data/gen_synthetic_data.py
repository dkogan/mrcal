#!/usr/bin/python2

import numpy as np
import numpysane as nps
import gnuplotlib as gp
import sys

sys.path[:0] = '../..',
import mrcal


# I move the board, and keep the camera stationary.
#
# Camera coords: x,y with pixels, z forward
# Board coords:  x,y along. z forward (i.e. back towards the camera)
#                rotation around (x,y,z) is (pitch,yaw,roll)

# board geometry
Nw                  = 10
width_square        = 0.01

# I try to get this many, but incomplete observations are thrown out, so in
# reality I get fewer. In practice I observe getting about 1/4 of this number
Nobservations = 50000

pitch_halfrange_deg = 20
yaw_halfrange_deg   = 20
roll_halfrange_deg  = 20
center              = (0,0,0.15) # camera coords
x_halfrange         = .3
y_halfrange         = .2
z_halfrange         = .05

m = mrcal.cameramodel('reference.cameramodel')
distortion_model = m.intrinsics()[0]
intrinsics       = m.intrinsics()[1]
cam_width,cam_height = m.imagersize()

pixel_noise_xy_1stdev = 0.5




xyz_halfrange = np.array((x_halfrange, y_halfrange, z_halfrange))




def rand_minus1_plus1(*args, **kwargs):
    x = np.random.random(*args, **kwargs)
    return x*2.0 - 1.0


# origin, in camera coords
xyz = center + rand_minus1_plus1((Nobservations,3)) * xyz_halfrange

pitch = pitch_halfrange_deg * rand_minus1_plus1((Nobservations,)) * np.pi/180.0
yaw   = yaw_halfrange_deg   * rand_minus1_plus1((Nobservations,)) * np.pi/180.0
roll  = roll_halfrange_deg  * rand_minus1_plus1((Nobservations,)) * np.pi/180.0

sp,cp = np.sin(pitch),np.cos(pitch)
sy,cy = np.sin(yaw),  np.cos(yaw)
sr,cr = np.sin(roll), np.cos(roll)

Rp = np.zeros((Nobservations,3,3), dtype=float)
Ry = np.zeros((Nobservations,3,3), dtype=float)
Rr = np.zeros((Nobservations,3,3), dtype=float)

Rp[:,0,0] =   1
Rp[:,1,1] =  cp
Rp[:,2,1] =  sp
Rp[:,1,2] = -sp
Rp[:,2,2] =  cp

Ry[:,1,1] =   1
Ry[:,0,0] =  cy
Ry[:,2,0] =  sy
Ry[:,0,2] = -sy
Ry[:,2,2] =  cy

Rr[:,2,2] =   1
Rr[:,0,0] =  cr
Rr[:,1,0] =  sr
Rr[:,0,1] = -sr
Rr[:,1,1] =  cr

# I didn't think about the order too hard; it might be backwards. It also
# probably doesn't really matter
R = nps.matmult(Rr, Ry, Rp)

board_reference = mrcal.get_ref_calibration_object(Nw, Nw, width_square)
# I shift the board so that the center in at the origin
board_reference -= (board_reference[0,0,:]+board_reference[-1,-1,:]) / 2.0

# shape = (Nobservations, Nw, Nw, 3)
boards = nps.matmult( nps.dummy(board_reference, -2), nps.mv(R, -3, -5))[..., 0, :] + nps.mv(xyz, -2, -4)

p = mrcal.project(boards, distortion_model, intrinsics)


def cull_out_of_bounds_observations(p, W, H):
    '''cuts out all observations that aren't entirely within the imager'''
    i = np.all(nps.clump(p,         n=-3) >= 0,   axis=-1) * \
        np.all(nps.clump(p[..., 0], n=-2) <= W-1, axis=-1) * \
        np.all(nps.clump(p[..., 1], n=-2) <= H-1, axis=-1)
    return p[i, ...]

p = cull_out_of_bounds_observations(p, cam_width, cam_height)


def write_data(filename, p, W,H):
    with open(filename, "w") as f:
        f.write("# filename x y\n")
        for i in xrange(len(p)):
            np.savetxt(f, nps.clump(p[i,...], n=2), fmt='{:06d}.xxx %.3f %.3f'.format(i))

write_data("../../studies/syntheticdata/synthetic-no-noise.vnl", p, cam_width, cam_height)
p += np.random.randn(*p.shape) * pixel_noise_xy_1stdev
write_data("../../studies/syntheticdata/synthetic.vnl",          p, cam_width, cam_height)


# gp.plot(nps.clump(p[...,0], n=-2), nps.clump(p[...,1], n=-2), _with='points', square=1,_xrange=[0,cam_width],_yrange=[0,cam_height])
