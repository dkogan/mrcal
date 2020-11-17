#!/usr/bin/python3

'''The main mrcal Python module

All the mrcal.mmm submodules export their symbols here for convenience. So any
function that can be called as mrcal.mmm.fff() can be called as mrcal.fff()
instead. The latter is preferred.

All Python functionality is available in this mrcal module, EXCEPT mrcal.cahvor.
That exists purely as a compatibility layer, so we keep it separate.

'''

# The C wrapper is written by us in mrcal_pywrap_nonbroadcasted.c
from ._mrcal_nonbroadcasted import *

# The C wrapper is generated from mrcal-genpywrap.py
from ._mrcal_broadcasted    import *

from .projections           import *
from .cameramodel           import *
from .poseutils             import *
from ._poseutils            import *
from .stereo                import *
from .visualization         import *
from .model_analysis        import *
from .synthetic_data        import *
from .calibration           import *
from .image_transforms      import *
from .utils                 import *
