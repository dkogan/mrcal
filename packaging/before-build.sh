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

strip_staging() {
    # Strip debug symbols from a staging tree before installing to BUILD_DEPS.
    # Safe to run on text files: strip exits non-zero but we ignore it.
    if [ "$(uname)" = "Darwin" ]; then
        find "$1" ! -type l -type f -exec strip -x {} \; 2>/dev/null || true
    else
        find "$1" ! -type l -type f -exec strip --strip-debug {} \; 2>/dev/null || true
    fi
}

# Use fixed paths for the Python extension staging so the build backend can
# find them reliably. (The cibuildwheel build venv's platlib differs from the
# isolated build-backend venv's platlib, so using sysconfig here would cause
# the build backend to look in the wrong place.)
GL_PYLIB=/tmp/gl-pylib
MRG_PYLIB=/tmp/mrg-pylib

INSTALL_ROOTS_GL="INSTALL_ROOT_PY3_MODULES=${GL_PYLIB}
                  INSTALL_ROOT_LIB=${BUILD_DEPS}/lib
                  INSTALL_ROOT_INCLUDE=${BUILD_DEPS}/include
                  INSTALL_ROOT_BIN=${BUILD_DEPS}/bin
                  INSTALL_ROOT_MAN=${BUILD_DEPS}/share/man"

INSTALL_ROOTS_MRG="INSTALL_ROOT_PY3_MODULES=${MRG_PYLIB}
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
rm -rf "$GL_STAGING" "$GL_PYLIB"
make -C /tmp/GL_image_display -j"${NCPUS}"
make -C /tmp/GL_image_display install DESTDIR="$GL_STAGING" ${INSTALL_ROOTS_GL}
strip_staging "$GL_STAGING"
install_c_lib "$GL_STAGING"
# Python extension files were installed directly to GL_PYLIB (no DESTDIR prefix)
cp -r "$GL_STAGING$GL_PYLIB/." "$GL_PYLIB/"

# ---------------------------------------------------------------------------
# mrgingham
# ---------------------------------------------------------------------------
MRG_STAGING=/tmp/mrgingham-staging
rm -rf "$MRG_STAGING" "$MRG_PYLIB"
make -C /tmp/mrgingham -j"${NCPUS}"
make -C /tmp/mrgingham install DESTDIR="$MRG_STAGING" ${INSTALL_ROOTS_MRG}
strip_staging "$MRG_STAGING"
install_c_lib "$MRG_STAGING"
cp -r "$MRG_STAGING$MRG_PYLIB/." "$MRG_PYLIB/"
