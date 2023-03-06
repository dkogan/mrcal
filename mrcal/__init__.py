#!/usr/bin/python3

# Copyright (c) 2017-2023 California Institute of Technology ("Caltech"). U.S.
# Government sponsorship acknowledged. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

'''The main mrcal Python package

This package doesn't contain any code itself, but all the mrcal.mmm submodules
export their symbols here for convenience. So any function that can be called as
mrcal.mmm.fff() can be called as mrcal.fff() instead. The latter is preferred.

'''

# The C wrapper is written by us in mrcal-pywrap.c
from ._mrcal import *

# The C wrapper is generated from mrcal-genpywrap.py
from ._mrcal_npsp    import *

from .projections           import *
from .cameramodel           import *
from .poseutils             import *
# The C wrapper is generated from poseutils-genpywrap.py
from ._poseutils_npsp       import *
from .stereo                import *
from .visualization         import *
from .model_analysis        import *
from .synthetic_data        import *
from .calibration           import *
from .image_transforms      import *
from .utils                 import *
from .triangulation         import *


# libelas is optional. If we don't have it, I don't complain
try:
    from ._elas_npsp    import *
except ModuleNotFoundError:
    pass

