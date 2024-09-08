#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Remaps a captured image into another camera model

SYNOPSIS

  ### To "undistort" images to reproject to a pinhole projection
  $ mrcal-reproject-image --to-pinhole
      camera0.cameramodel
      image*.jpg
  Wrote image0-pinhole.jpg
  Wrote image1-pinhole.jpg
  ...

  ### To reproject images from one lens model to another
  $ mrcal-reproject-image
      camera0.cameramodel camera1.cameramodel
      image*.jpg
  Wrote image0-reprojected.jpg
  Wrote image1-reprojected.jpg
  Wrote image2-reprojected.jpg
  ...

  ### To reproject two sets of images to a common pinhole projection
  $ mrcal-reproject-image --to-pinhole
      camera0.cameramodel camera1.cameramodel
      'image*-cam0.jpg' 'image*-cam1.jpg'
  Wrote image0-reprojected.jpg
  Wrote image1-reprojected.jpg
  Wrote image2-reprojected.jpg
  ...

  ### To "manually" stereo-rectify a pair of images
  $ mrcal-stereo          \
      --az-fov-deg 80     \
      --el-fov-deg 50     \
      --outdir /tmp       \
      left.cameramodel    \
      right.cameramodel
  Wrote '/tmp/rectified0.cameramodel'
  Wrote '/tmp/rectified1.cameramodel'

  $ mrcal-reproject-image         \
      --outdir /tmp               \
      /tmp/left.cameramodel       \
      /tmp/rectified0.cameramodel \
      left.jpg
  Wrote /tmp/left-reprojected.jpg

  $ mrcal-reproject-image          \
      --outdir /tmp                \
      /tmp/right.cameramodel       \
      /tmp/rectified1.cameramodel  \
      right.jpg
  Wrote /tmp/right-reprojected.jpg

  $ mrcal-stereo                       \
      --already-rectified              \
      --outdir /tmp                    \
      /tmp/rectified[01].cameramodel   \
      /tmp/left-reprojected.jpg        \
      /tmp/right-reprojected.jpg

  # This is the same as using mrcal-stereo to do all the work:
  $ mrcal-stereo          \
      --az-fov-deg 80     \
      --el-fov-deg 50     \
      --outdir /tmp       \
      left.cameramodel    \
      right.cameramodel   \
      left.jpg            \
      right.jpg


This tool takes image(s) of a scene captured by one camera model, and produces
image(s) of the same scene, as it would appear if captured by a different model,
taking into account both the different lens parameters and geometries. This is
similar to mrcal-reproject-points, but acts on a full image, rather than a
discrete set of points.

There are several modes of operation, depending on how many camera models are
given, and whether --to-pinhole is given, and whether --plane-n,--plane-d are
given.

To "undistort" (remap to a pinhole projection) a set of images captured using a
particular camera model, invoke this tool like this:

  mrcal-reproject-image
    --to-pinhole
    model0.cameramodel image*.jpg

Each of the given images will be reprojected, and written to disk as
"image....-reprojected.jpg". The pinhole model used for the reprojection will be
written to standard output.

To remap images of a scene captured by model0 to images of the same scene
captured by model1, do this:

  mrcal-reproject-image
    model0.cameramodel model1.cameramodel image*.jpg

Each of the given images will be reprojected, and written to disk as
"image....-reprojected.jpg". Nothing will be written to standard output. By
default, full relative extrinsics between the two models are used in the
reprojection. The unprojection distance (given with --distance) is infinity by
default, so only the relative rotation is used by default. To ignore the
extrinsics entirely, pass --intrinsics-only.

A common use case is to validate the relative intrinsics and extrinsics in two
models. If you have a pair of models and a pair of observed images, you can
compute the reprojection, and compare the reprojection-to-model1 to images that
were actually captured by model1. If the intrinsics and extrinsics were correct,
then the two images would line up exactly for relevant objects (far-away observations with the default --distance, ground plane with --plane-n, etc).

Computing this reprojection map is often very slow. But if the use case is
comparing two sets of captured images, the next, much faster invocation method
can be used.

To remap images of a scene captured by model0 and images of the same scene
captured by model1 to a common pinhole projection, do this:

  mrcal-reproject-image
    --to-pinhole
    model0.cameramodel model1.cameramodel 'image*-cam0.jpg' 'image*-cam1.jpg'

A pinhole model is constructed that has the same extrinsics as model1, and both
sets of images are reprojected to this model. This is similar to the previous
mode, but since we're projecting to a pinhole model, this computes much faster.
The generated pinhole model is written to standard output.

Finally instead of reprojecting to match up images of objects at infinity, it is
possible to reproject to match up images of arbitrary planes. This can be done
by a command like this:

  mrcal-reproject-image
    --to-pinhole
    --plane-n 1.1 2.2 3.3
    --plane-d 4.4
    model0.cameramodel model1.cameramodel 'image*-cam0.jpg' 'image*-cam1.jpg'

If the models were already pinhole-projected, this does the same thing as

  mrcal-reproject-image
    --plane-n 1.1 2.2 3.3
    --plane-d 4.4
    model0.cameramodel model1.cameramodel 'image*-cam0.jpg'

This maps observations of a given plane in camera0 coordinates to where this
plane would be observed in camera1 coordinates. This requires both models to be
passed-in. And ALL the intrinsics, extrinsics and the plane representation are
used. If all of these are correct, the observations of this plane would line up
exactly in the remapped-camera0 image and the camera1 image. The plane is
represented in camera0 coordinates by a normal vector given by --plane-n, and
the distance to the normal given by plane-d. The plane is all points p such that
inner(p,planen) = planed. planen does not need to be normalized. This mode does
not require --to-pinhole, but it makes the computations run much faster, as
before.

If --to-pinhole, then we generate a pinhole model, that is written to standard
output. By default, the focal length of this pinhole model is the same as that
of the input model. The "zoom" level of this pinhole model can be adjusted by
passing --scale-focal SCALE, or more precisely by passing --fit. --fit takes an
argument that is one of

- "corners": make sure all of the corners of the original image remain in-bounds
  of the pinhole projection

- "centers-horizontal": make sure the extreme left-center and right-center
  points in the original image remain in-bounds of the pinhole projection

- "centers-vertical": make sure the extreme top-center and bottom-center points
  in the original image remain in-bounds of the pinhole projection

- A list of pixel coordinates x0,y0,x1,y1,x2,y2,.... The focal-length will be
  chosen to fit all of the given points

By default, the resolution of the generated pinhole model is the same as the
resolution of the input model. This can be adjusted by passing --scale-image.
For instance, passing "--scale-image 0.5" will generate a pinhole model and
images that are half the size of the input images, in both the width and height.

The output image(s) are written into the same directory as the input image(s),
with annotations in the filename. This tool will refuse to overwrite any
existing files unless --force is given.

It is often desired to apply transformations to lots of images in bulk. To make
this go faster, this tool supports the -j JOBS option. This works just like in
Make: the work will be parallelized among JOBS simultaneous processes. Unlike
make, the JOBS value must be specified.

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--to-pinhole',
                        action="store_true",
                        help='''If given, we reproject the images to a pinhole model that's generated off the
                        MODEL-FROM and --fit, --scale-focal, --scale-image. The
                        generated pinhole model is written to the standard
                        output''')

    parser.add_argument('--intrinsics-only',
                        action='store_true',
                        help='''If two camera models are given, then by default
                        the full relative transformation is used in the
                        reprojection. If we want to use the intrinsics ONLY,
                        pass this option''')

    parser.add_argument('--distance',
                        type=float,
                        help='''The fundamental operation of this tool is to
                        unproject points from one camera, and to reproject them
                        into the other. The distance used for the unprojection
                        is set by this argument. If omitted, infinity is used;
                        this is equivalent to only using the rotation component
                        of the relative transformation between the cameras. This
                        option only makes sense without --intrinsics-only and
                        without --plane-n/--plane-d''')

    parser.add_argument('--fit',
                        type=str,
                        required=False,
                        help='''If we generate a target pinhole model (if --to-pinhole is given) then we can
                        choose the focal length of the target model. This is a
                        "zoom" operation. By default just use whatever value
                        model-from has. Or we scale it by the value given in
                        --scale-focal. Or we use --fit to scale the focal length
                        intelligently. The --fit argument could be one of
                        ("corners", "centers-horizontal", "centers-vertical"),
                        or the argument could be given as a list of points
                        x0,y0,x1,y1,x2,y2,.... The focal length scale would then
                        be chosen to zoom in as far as possible, while fitting
                        all of these points''')

    parser.add_argument('--scale-focal',
                        type=float,
                        help='''If we generate a target pinhole model (if --to-pinhole is given) then we can
                        choose the focal length of the target model. This is a
                        "zoom" operation. By default just use whatever value
                        model-from has. Or we scale it by the value given in
                        --scale-focal. Or we use --fit to scale the focal length
                        intelligently.''')

    parser.add_argument('--scale-image',
                        type=float,
                        help='''If we generate a target pinhole model (if --to-pinhole is given) then we can
                        choose the dimensions of the output image. By default we
                        use the dimensions of model-from. If --scale-image is
                        given, we use this value to scale the imager dimensions
                        of model-from. This parameter changes the RESOLUTION of
                        the output, unlike --scale-focal, which ZOOMS the
                        output''')

    parser.add_argument('--plane-n',
                        type=float,
                        nargs=3,
                        help='''We're reprojecting a plane. The normal vector to this plane is given here, in
                        from-camera coordinates. The normal does not need to be
                        normalized; any scaling is compensated in planed. The
                        plane is all points p such that inner(p,planen) =
                        planed''')
    parser.add_argument('--plane-d',
                        type=float,
                        help='''We're reprojecting a plane. The distance-along-the-normal to the plane, in
                        from-camera coordinates is given here. The plane is all
                        points p such that inner(p,planen) = planed''')

    parser.add_argument('--outdir',
                        required=False,
                        type=lambda d: d if os.path.isdir(d) else \
                                parser.error("--outdir requires an existing directory as the arg, but got '{}'".format(d)),
                        help='''Directory to write the output images into. If omitted, we write the output
                        images to the same directory as the input images''')

    parser.add_argument('--valid-intrinsics-region',
                        action='store_true',
                        help='''If given, we annotate the images with the FROM model's valid-intrinsics
                        region''')

    parser.add_argument('--mask-valid-intrinsics-region',
                        action='store_true',
                        help='''If given, we draw everything outside the FROM model's valid-intrinsics region
                        as black. So the unreliable regions aren't even drawn''')

    parser.add_argument('--force', '-f',
                        action='store_true',
                        default=False,
                        help='''By default existing files are not overwritten. Pass --force to overwrite them
                        without complaint''')

    parser.add_argument('--jobs', '-j',
                        type=int,
                        required=False,
                        default=1,
                        help='''parallelize the processing JOBS-ways. This is like Make, except you're
                        required to explicitly specify a job count.''')

    parser.add_argument('model-from',
                        type=str,
                        help='''Camera model for the FROM image(s). If "-' is given, we read standard
                        input''')

    parser.add_argument('model-to-and-image-globs',
                        type=str,
                        nargs='+',
                        help='''Optionally, the camera model for the TO image. Followed, by the from/to image
                        globs. See the mrcal-reproject-image documentation for
                        the details.''')

    args = parser.parse_args()

    # use _ instead of - in attribute names so that I can access them easier
    args.model_to_and_image_globs = getattr(args, 'model-to-and-image-globs')
    args.model_from               = getattr(args, 'model-from')
    delattr(args, 'model-to-and-image-globs')
    delattr(args, 'model-from')

    return args


args = parse_args()

import mrcal

# I have to manually process this because the first model-to-and-image-globs
# element's identity is ambiguous in a way I can't communicate to argparse.
# It can be model-to or it can be the first image glob
def load_model_or_keep_filename(filename):
    try:
        m = mrcal.cameramodel(filename)
    except:
        # Couldn't load this file as a model. Are we pretty sure it WAS a model?
        if re.search(r"\.(cameramodel|cahv|cahvor|cahvore)$",
                     filename,
                     flags = re.I):
            # Filename tells us that this WAS a model. So I give up
            print(f"Couldn't read camera model '{filename}'", file=sys.stderr)
            sys.exit(1)

        # Let's try to interpret this as an image
        return filename

    return m
mi = [load_model_or_keep_filename(f) for f in args.model_to_and_image_globs]

args.model_to   = [ m for m in mi if     isinstance(m,mrcal.cameramodel) ]
args.imageglobs = [ m for m in mi if not isinstance(m,mrcal.cameramodel) ]
delattr(args, 'model_to_and_image_globs')



if   len(args.model_to) == 0: args.model_to = None
elif len(args.model_to) == 1: args.model_to = args.model_to[0]
else:
    print(f"At most one model-to can be given. Instead got {len(args.model_to)} of them. Giving up.", file=sys.stderr)
    sys.exit(1)

if args.model_from == '-' and \
   args.model_to   == '-':
    print("At most one model can be given at '-' to read standard input. Giving up.", file=sys.stderr)
    sys.exit(1)

if not args.to_pinhole:
    if args.fit         is not None or \
       args.scale_focal is not None or \
       args.scale_image is not None:
        print("--fit, --scale-focal, --scale-image make sense ONLY with --to-pinhole",
              file = sys.stderr)
        sys.exit(1)
else:
    if args.fit         is not None and \
       args.scale_focal is not None:
        print("--fit and --scale-focal are mutually exclusive", file=sys.stderr)
        sys.exit(1)

if args.model_to is None and \
   args.intrinsics_only:
    print("--intrinsics-only makes sense ONLY when both the FROM and TO camera models are given",
          file=sys.stderr)
    sys.exit(1)

if args.scale_image is not None and args.scale_image <= 1e-6:
    print("--scale-image should be given a reasonable value > 0", file=sys.stderr)
    sys.exit(1)

if (args.plane_n is     None and args.plane_d is not None) or \
   (args.plane_n is not None and args.plane_d is     None):
    print("--plane-n and --plane-d should both be given or neither should be", file=sys.stderr)
    sys.exit(1)

if args.plane_n is not None and \
   args.intrinsics_only:
    print("We're looking at remapping a plane (--plane-d, --plane-n are given), so --intrinsics-only doesn't make sense",
          file=sys.stderr)
    sys.exit(1)

if args.distance is not None and \
   (args.plane_n is not None or args.intrinsics_only):
    print("--distance makes sense only without --plane-n/--plane-d and without --intrinsics-only", file=sys.stderr)
    sys.exit(1)





import numpy as np
import numpysane as nps

if args.fit is not None:
    if re.match(r"^[0-9\.e-]+(,[0-9\.e-]+)*$", args.fit):
        xy = np.array([int(x) for x in args.fit.split(',')], dtype=float)
        Nxy = len(xy)
        if Nxy % 2 or Nxy < 4:
            print(f"If passing pixel coordinates to --fit, I need at least 2 x,y pairs. Instead got {Nxy} values",
                  file=sys.stderr)
            sys.exit(1)
        args.fit = xy.reshape(Nxy//2, 2)
    elif re.match("^(corners|centers-horizontal|centers-vertical)$", args.fit):
        # this is valid. nothing to do
        pass
    else:
        print("--fit must be a comma-separated list of numbers or one of ('corners','centers-horizontal','centers-vertical')",
              file=sys.stderr)
        sys.exit(1)




import glob
import multiprocessing
import signal

import time

try:
    model_from = mrcal.cameramodel(args.model_from)
except Exception as e:
    print(f"Couldn't read '{args.model_from}' as a cameramodel: {e}", file=sys.stderr)
    sys.exit(1)

if not args.to_pinhole:
    if not args.model_to:
        print("Either --to-pinhole or the TO camera model MUST be given. Giving up", file=sys.stderr)
        sys.exit(1)
    if len(args.imageglobs) < 1:
        print("No --to-pinhole with both TO and FROM models given: must have at least one set of image globs. Giving up", file=sys.stderr)
        sys.exit(1)

    model_to = args.model_to

else:
    if not args.model_to:
        if len(args.imageglobs) < 1:
            print("--to-pinhole with only the FROM models given: must have at least one set of image globs. Giving up", file=sys.stderr)
            sys.exit(1)

        model_to = mrcal.pinhole_model_for_reprojection(model_from, args.fit,
                                                        scale_focal = args.scale_focal,
                                                        scale_image = args.scale_image)

        print( "## generated on {} with   {}".format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     ' '.join(mrcal.shellquote(s) for s in sys.argv)) )
        print("# Generated pinhole model:")
        model_to.write(sys.stdout)

    else:
        if len(args.imageglobs) != 2:
            print("--to-pinhole with both the TO and FROM models given: must have EXACTLY two image globs. Giving up", file=sys.stderr)
            sys.exit(1)

        model_to     = args.model_to
        model_target = mrcal.pinhole_model_for_reprojection(model_to, args.fit,
                                                            scale_focal = args.scale_focal,
                                                            scale_image = args.scale_image)
        print( "## generated on {} with   {}".format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                                     ' '.join(mrcal.shellquote(s) for s in sys.argv)) )
        print("# Generated pinhole model:")
        model_target.write(sys.stdout)

if args.plane_n is not None:
    if args.model_to is None:
       print("Plane remapping requires BOTH camera models to be given", file=sys.stderr)
       sys.exit(1)

    args.plane_n = np.array(args.plane_n, dtype=float)


# I do the same thing in mrcal-stereo. Please consolidate
#
# weird business to handle weird signal handling in multiprocessing. I want
# things like the user hitting C-c to work properly. So I ignore SIGINT for the
# children. And I want the parent's blocking wait for results to respond to
# signals. Which means map_async() instead of map(), and wait(big number)
# instead of wait()
signal_handler_sigint = signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGINT, signal_handler_sigint)

# This stuff needs to be global for the multiprocessing pool to pick it up. It
# really is quite terrible. All I REALLY want is some os.fork() calls...
model_valid_intrinsics_region = None
mapxy                         = None
model_imagersize              = None
def _transform_this(inout):
    try:
        image = mrcal.load_image(inout[0])
    except:
        print(f"Couldn't load '{inout[0]}'",
              file=sys.stderr)
        return

    if image.shape[0] != model_imagersize[1] or \
       image.shape[1] != model_imagersize[0]:
        print(f"Couldn't process {inout[0]}: image dimensions don't match the input model dimensions. Image size: [{image.shape[1]} {image.shape[0]}]. model.imagersize(): {model_imagersize}",
              file=sys.stderr)
        return

    if model_valid_intrinsics_region is not None:
        mrcal.annotate_image__valid_intrinsics_region(image, model_valid_intrinsics_region)
    image_transformed = mrcal.transform_image(image, mapxy)
    mrcal.save_image(inout[1], image_transformed)
    print(f"Wrote {inout[1]}", file=sys.stderr)

def process(model_from, model_to, image_globs, suffix,
            intrinsics_only, distance, plane_n, plane_d):

    def target_image_filename(filename_in, suffix):

        base,extension = os.path.splitext(filename_in)
        if len(extension) != 4:
            print(f"imagefile must end in .xxx where 'xxx' is some image extension. Instead got '{filename_in}'",
                  file=sys.stderr)
            sys.exit(1)

        if args.outdir is not None:
            base = args.outdir + '/' + os.path.split(base)[1]

        filename_out = f"{base}-{suffix}{extension}"
        if not args.force and os.path.exists(filename_out):
            print(f"Target image '{filename_out}' already exists. Doing nothing, and giving up. Pass -f to overwrite",
                  file=sys.stderr)
            sys.exit(1)
        return filename_out

    filenames_in  = [f for g in image_globs for f in glob.glob(g)]
    if len(filenames_in) == 0:
        print(f"Globs '{image_globs}' matched no files!", file=sys.stderr)
        sys.exit(1)
    filenames_out = [target_image_filename(f, suffix) for f in filenames_in]
    filenames_inout = zip(filenames_in, filenames_out)

    global mapxy
    global model_valid_intrinsics_region
    global model_imagersize
    if args.valid_intrinsics_region:
        model_valid_intrinsics_region = model_from
    model_imagersize = model_from.imagersize()

    mapxy = mrcal.image_transformation_map(model_from, model_to,
                                           intrinsics_only = intrinsics_only,
                                           distance        = distance,
                                           plane_n         = plane_n,
                                           plane_d         = plane_d,
                                           mask_valid_intrinsics_region_from = \
                                           args.mask_valid_intrinsics_region,)

    if args.jobs > 1:
        # Normal parallelized path
        pool = multiprocessing.Pool(args.jobs)
        try:
            mapresult = pool.map_async(_transform_this, filenames_inout)

            # like wait(), but will barf if something goes wrong. I don't actually care
            # about the results
            mapresult.get(1000000)
        except:
            pool.terminate()

        pool.close()
        pool.join()

    else:
        # Serial path. Useful for debugging
        for f in filenames_inout:
            _transform_this(f)




if args.to_pinhole and args.model_to:
    # I'm reprojecting each of my sets of images to a pinhole model (a DIFFERENT
    # model from TO and FROM)
    process(model_from, model_target, (args.imageglobs[0],), "pinhole-remapped",
            args.intrinsics_only, args.distance, args.plane_n, args.plane_d)
    process(model_to,   model_target, (args.imageglobs[1],), "pinhole",
            args.intrinsics_only, args.distance, None, None)
else:
    # Simple case. I have my two models, and I reproject all the images
    process(model_from, model_to, args.imageglobs, "reprojected",
            args.intrinsics_only, args.distance, args.plane_n, args.plane_d)
