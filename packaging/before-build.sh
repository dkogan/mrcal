#!/bin/bash

# Build GL_image_display and mrgingham for the current Python version.
# Called by cibuildwheel's before-build hook (once per Python version, both
# platforms). before-all already cloned the sources and installed C build deps.
set -ex

BUILD_DEPS="${HOME}/build-deps"

# Capture the cibuildwheel Python before PATH is modified (macOS adds brew to
# PATH which would mask the cibuildwheel Python with Homebrew's externally-
# managed one).
PYTHON3=$(command -v python3)
PY_PLATLIB=$("${PYTHON3}" -c "import sysconfig; print(sysconfig.get_path('platlib'))")

if [ "$(uname)" = "Darwin" ]; then
    NCPUS=$(sysctl -n hw.ncpu)
    BREW=$(brew --prefix)
    # Put cibuildwheel Python first so mrbuild's python3 calls use it, then
    # GNU getopt (keg-only; needed for mrgingham man-page generation), then
    # brew tools, then the rest of PATH.
    export PATH="$(dirname "${PYTHON3}"):${BREW}/opt/gnu-getopt/bin:${BREW}/bin:/usr/local/bin:$PATH"
    export CPATH="${BUILD_DEPS}/include:${BREW}/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${BUILD_DEPS}/lib:${BREW}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    export SWIG_FLAGS="-I${BREW}/include"
    install_c_lib() { cp -a "$1${BUILD_DEPS}/". "${BUILD_DEPS}/"; }
else
    NCPUS=$(nproc)
    export PATH="${BUILD_DEPS}/bin:${PATH}"
    export CPATH="${BUILD_DEPS}/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${BUILD_DEPS}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    install_c_lib() { cp -a "$1${BUILD_DEPS}/". "${BUILD_DEPS}/"; ldconfig; }
fi

INSTALL_ROOTS="INSTALL_ROOT_PY3_MODULES=${PY_PLATLIB}
               INSTALL_ROOT_LIB=${BUILD_DEPS}/lib
               INSTALL_ROOT_INCLUDE=${BUILD_DEPS}/include
               INSTALL_ROOT_BIN=${BUILD_DEPS}/bin
               INSTALL_ROOT_MAN=${BUILD_DEPS}/share/man"

# ---------------------------------------------------------------------------
# GL_image_display
# ---------------------------------------------------------------------------
"${PYTHON3}" -m pip install numpy setuptools --quiet
NUMPY_INC=$("${PYTHON3}" -c 'import numpy; print(numpy.get_include())')
ln -sf "${NUMPY_INC}/numpy" "${BUILD_DEPS}/include/numpy"

GL_STAGING=/tmp/gl-py-staging
rm -rf "$GL_STAGING"
make -C /tmp/GL_image_display -j"${NCPUS}"
make -C /tmp/GL_image_display install DESTDIR="$GL_STAGING" ${INSTALL_ROOTS}
install_c_lib "$GL_STAGING"

# ---------------------------------------------------------------------------
# mrgingham
# ---------------------------------------------------------------------------
MRG_STAGING=/tmp/mrgingham-staging
rm -rf "$MRG_STAGING"
make -C /tmp/mrgingham -j"${NCPUS}"
make -C /tmp/mrgingham install DESTDIR="$MRG_STAGING" ${INSTALL_ROOTS}
install_c_lib "$MRG_STAGING"
