#!/usr/bin/env python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

r'''Reports which input points lie within the valid-intrinsics region

SYNOPSIS

  $ < points-in.vnl
    mrcal-is-within-valid-intrinsics-region --cols-xy x y
      camera.cameramodel
    > points-annotated.vnl

mrcal camera models may have an estimate of the region of the imager where the
intrinsics are trustworthy (originally computed with a low-enough error and
uncertainty). When using a model, we may want to process points that fall
outside of this region differently from points that fall within this region.
This tool augments an incoming vnlog with a new column, indicating whether each
point does or does not fall within the region.

The input data comes in on standard input, and the output data is written to
standard output. Both are vnlog data: a human-readable table of ascii text. The
names of the x and y columns in the input are given in the required --cols-xy
argument. The output contains all the columns from the input, with an extra
column appended at the end, containing the results. The name of this column can
be specified with --col-output, but this can be omitted if the default
'is-within-valid-intrinsics-region' is acceptable.

'''

import sys
import argparse
import re
import os

def parse_args():

    parser = \
        argparse.ArgumentParser(description = __doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cols-xy',
                        required=True,
                        type=str,
                        nargs=2,
                        help='''The names of the columns in the input containing the x and y pixel
                        coordinates respectively. This is required''')
    parser.add_argument('--col-output',
                        required=False,
                        type=str,
                        default='is-within-valid-intrinsics-region',
                        help='''The name of the column to append in the output. This is optional; a
                        reasonable default will be used if omitted''')

    parser.add_argument('model',
                        type=str,
                        help='''Camera model''')

    return parser.parse_args()

args = parse_args()

# arg-parsing is done before the imports so that --help works without building
# stuff, so that I can generate the manpages and README





import numpy as np
import numpysane as nps
import subprocess
import mrcal
import io

try:
    model = mrcal.cameramodel(args.model)
except Exception as e:
    print(f"Couldn't load camera model '{args.model}': {e}", file=sys.stderr)
    sys.exit(1)

if model.valid_intrinsics_region() is None:
    print("The given model has no valid-intrinsics region defined!",
          file=sys.stderr)
    sys.exit(1)

# I want to be able to add to any arbitrary existing columns, so I slurp.
# Good-enough for now.
in_file = sys.stdin.read()

cmd = ('vnl-filter', '-p', '__linenumber=NR', '-p', f'^{args.cols_xy[0]}$', '-p', f'^{args.cols_xy[1]}$')
try:
    ixy_text = subprocess.check_output(cmd,
                                       shell    = False,
                                       encoding = 'ascii',
                                       input    = in_file)
except:
    print(f"Couldn't read columns {args.cols_xy} from the input",
          file = sys.stderr)
    sys.exit(1)

ixy = nps.atleast_dims(np.loadtxt(io.StringIO(ixy_text)), -2)
mask = mrcal.is_within_valid_intrinsics_region(ixy[..., 1:], model)


fd_read,fd_write = os.pipe()
os.set_inheritable(fd_read, True)

pid = os.fork()

if pid == 0:
    # Child. I write the mask back to the pipe. The parent will join it
    os.close(fd_read)

    with open(f"/dev/fd/{fd_write}", 'w') as f:
        np.savetxt(f,
                   nps.transpose(nps.cat(ixy[...,0], mask)),
                   fmt='%06d %d',
                   header=f'__linenumber {args.col_output}')
    os.close(fd_write)
    sys.exit(0)

# parent. join the input data, and the data coming in from the child
os.close(fd_write)


# I don't re-sort the input. NR was printed with %06d so it's already sorted.
cmd = f"vnl-filter --skipcomments -p __linenumber='sprintf(\"%06d\",NR)',. | vnl-join -j __linenumber - /dev/fd/{fd_read} | vnl-filter -p \\!__linenumber"
subprocess.run( cmd,
                shell = True,
                close_fds = False,
                encoding = 'ascii',
                input = in_file )

os.close(fd_read)
