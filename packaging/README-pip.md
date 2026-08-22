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

# INSTALLATION

These are pip wheels. If at all possible, do not use these, and install from the
Debian packages instead, as noted on the ["Building or installing"
page](https://mrcal.secretsauce.net/install.html). Pip is a huge hack, and this
is all deeply inefficient, and might not work as well. If you find issues,
please let me know.

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
