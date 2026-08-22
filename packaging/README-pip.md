# SYNOPSIS

    $ mrcal-calibrate-cameras --focal 2000
          --outdir /tmp --object-spacing 0.01
          --object-width-n 10 '/tmp/left*.png' '/tmp/right*.png'

    ... lots of output as the solve runs ...
    Wrote /tmp/camera0-0.cameramodel
    Wrote /tmp/camera0-1.cameramodel

And now we have a calibration!

# SUMMARY

`mrcal` is a generic toolkit built to solve the calibration and SFM-like
problems we encounter at NASA/JPL. Functionality related to these problems is
exposed as a set of C and Python libraries and some commandline tools.

# DESCRIPTION

Extensive documentation is available at <https://mrcal.secretsauce.net/>

These pip wheels are available for Linux (amd64) and macos (arm64). Pip is a
giant hack. These were a huge pain to build, even with Claude's help. They're
massively inefficient, and probably are missing things.

Everything mrcal does should be there. Plotting should be there (gnuplot is
shipped, with x11 and qt terminals for osx); it should just work. Tell me if it
doesn't. pyfltk and the GL image widget are available (`mrcal-stereo` should
work). vnlog is available. mrgingham is *not* available: building opencv was an
endless timesuck; the pip-building tools suck.


# INSTALLATION

These pip wheels should work. If at all possible, do not use these, and install
from the Debian packages instead, as noted on the ["Building or installing"
page](https://mrcal.secretsauce.net/install.html). If you find issues with any
of it, please let me know.

# REPOSITORY

<https://www.github.com/dkogan/mrcal/>

# AUTHOR

Dima Kogan (`dima@secretsauce.net`)

# LICENSE AND COPYRIGHT

Copyright (c) 2017-2023 California Institute of Technology
("Caltech"). U.S. Government sponsorship acknowledged. All rights
reserved.

Licensed under the Apache License, Version 2.0 (the "License"); You
may obtain a copy of the License at

<http://www.apache.org/licenses/LICENSE-2.0>
