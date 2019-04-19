#!/usr/bin/python3

# I want to refer to some stuff as mrcal.X instead of mrcal.Y.X
# For instance:
#
# I want to be able to refer to everything as mrcal.x instead of mrcal.y.x.
# Makes things easier for the user. EXCEPT cahvor. This is separate, and
# hopefully can go away eventually

# I do this first to pull in _mrcal.project
from ._mrcal      import *
# I do this second, do overwrite that. I want the projections version of
# project() to be the one imported into mrcal. This is the only name conflict
from .projections import *

from .cameramodel import *
from .poseutils   import *
from .utils       import *
