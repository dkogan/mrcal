#!/usr/bin/python3

r'''Removes the last N '''

import sys
import ast

if len(sys.argv) != 2:
    print(f"Usage: < in.cameramodel {sys.argv[0]} Ncull > out.cameramodel", file=sys.stderr)
    sys.exit(1)

try:
    Ncull = int(sys.argv[1])
except:
    print(f"The commandline argument Ncull MUST be a positive integer", file=sys.stderr)
    sys.exit(1)
if Ncull <= 0:
    print(f"The commandline argument Ncull MUST be a positive integer", file=sys.stderr)
    sys.exit(1)



# I read in the model without using mrcal, since the Nintrinsics might be
# unexpected
s = sys.stdin.read()

try:
    model = ast.literal_eval(s)
except:
    print("Failed to parse cameramodel from standard input", file=sys.stderr)
    sys.exit(1)


# And I write it back, culled, WITHOUT the optimization inputs
sys.stdout.write("{\n")
sys.stdout.write("    'lensmodel':  '{}',\n".format(model['lensmodel']))

intrinsics = model['intrinsics'][:-Ncull]
N = len(intrinsics)
sys.stdout.write(("    'intrinsics': [" + (" {:.10g}," * N) + "],\n").format(*intrinsics))

N = len(model['extrinsics'])
sys.stdout.write(("    'extrinsics': [" + (" {:.10g}," * N) + "],\n").format(*model['extrinsics']))

N = 2
sys.stdout.write(("    'imagersize': [" + (" {:d}," * N) + "],\n").format(*(int(x) for x in model['imagersize'])))
sys.stdout.write("}\n")
