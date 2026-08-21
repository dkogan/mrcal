#!/bin/bash

# Build GL_image_display for the current Python version.
# Called by cibuildwheel's before-build hook (once per Python version, both
# platforms). before-all already cloned the sources and installed C build deps.
set -ex

source "$(dirname "$0")/build-deps-common.sh"

MRCAL_SRC="$(cd "$(dirname "$0")/.." && pwd)"
ln -sf /tmp/mrbuild "${MRCAL_SRC}/mrbuild"

# Capture the cibuildwheel Python before PATH is modified (macOS adds brew to
# PATH which would mask the cibuildwheel Python with Homebrew's externally-
# managed one).
PYTHON3=$(command -v python3)

if [ "$(uname)" = "Darwin" ]; then
    NCPUS=$(sysctl -n hw.ncpu)
    BREW=$(brew --prefix)
    # Put cibuildwheel Python first so mrbuild's python3 calls use it,
    # then brew tools, then the rest of PATH.
    export PATH="$(dirname "${PYTHON3}"):${BREW}/bin:/usr/local/bin:$PATH"
    export CPATH="${BUILD_DEPS}/include:${BREW}/include"
    export LIBRARY_PATH="${BUILD_DEPS}/lib:${BREW}/lib"
    export DYLD_LIBRARY_PATH="${BUILD_DEPS}/lib:${BREW}/lib"
    export PKG_CONFIG_PATH="${BUILD_DEPS}/lib/pkgconfig:${BREW}/lib/pkgconfig"
    export SWIG_FLAGS="-I${BREW}/include"
    export PERL5LIB="${BUILD_DEPS}/lib/perl5${PERL5LIB:+:$PERL5LIB}"
else
    NCPUS=$(nproc)
    export PATH="${BUILD_DEPS}/bin:${PATH}"
    export CPATH="${BUILD_DEPS}/include"
    export LIBRARY_PATH="${BUILD_DEPS}/lib:${BUILD_DEPS}/lib64"
    export LD_LIBRARY_PATH="${BUILD_DEPS}/lib:${BUILD_DEPS}/lib64"
    export PKG_CONFIG_PATH="${BUILD_DEPS}/lib/pkgconfig:${BUILD_DEPS}/lib64/pkgconfig"
    export SWIG_FLAGS="-I${BUILD_DEPS}/include"
    export PERL5LIB="${BUILD_DEPS}/lib/perl5${PERL5LIB:+:$PERL5LIB}"
fi

# mrbuild computes PY3_MODULE_PATH via $(shell python3 ...).  With the
# cibuildwheel Python first in PATH it resolves correctly.  Record it so the
# build backend (which runs in a different isolated venv) can find the
# installed extensions.
PY3_MODULES=$("${PYTHON3}" -c 'import site; print(site.getsitepackages()[0])')
printf '%s' "${PY3_MODULES}" > "${BUILD_DEPS}/py3-modules-path"

INSTALL_ROOTS="INSTALL_ROOT_LIB=/lib
               INSTALL_ROOT_INCLUDE=/include
               INSTALL_ROOT_BIN=/bin
               INSTALL_ROOT_MAN=/share/man
               INSTALL_ROOT_PY3_MODULES=${PY3_MODULES}"

"${PYTHON3}" -m pip install numpy setuptools --quiet

NUMPY_INC=$("${PYTHON3}" -c 'import numpy; print(numpy.get_include())')
ln -sf "${NUMPY_INC}/numpy" "${BUILD_DEPS}/include/numpy"

make -C /tmp/GL_image_display -j"${NCPUS}"
make -C /tmp/GL_image_display install DESTDIR="${BUILD_DEPS}" ${INSTALL_ROOTS}
if [ "$(uname)" != "Darwin" ]; then ldconfig; fi

# ---------------------------------------------------------------------------
# pyfltk — built against the same fltk that GL_image_display uses
# ---------------------------------------------------------------------------
"${PYTHON3}" -m pip install --no-binary pyfltk --no-deps \
    --target="${BUILD_DEPS}${PY3_MODULES}" pyfltk
if [ "$(uname)" != "Darwin" ]; then ldconfig; fi
