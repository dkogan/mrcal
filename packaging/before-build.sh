#!/bin/bash

# Build GL_image_display and mrgingham for the current Python version.
# Called by cibuildwheel's before-build hook (once per Python version, both
# platforms). before-all already cloned the sources and installed C build deps.
set -ex

PY_PLATLIB=$(python3 -c "import sysconfig; print(sysconfig.get_path('platlib'))")

if [ "$(uname)" = "Darwin" ]; then
    NCPUS=$(sysctl -n hw.ncpu)
    export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
    BREW=$(brew --prefix)
    export CPATH="${BREW}/include${CPATH:+:$CPATH}"
    export LIBRARY_PATH="${BREW}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    LIB_ROOT="${BREW}/lib"
    INCLUDE_ROOT="${BREW}/include"
    BIN_ROOT="${BREW}/bin"
    MAN_ROOT="${BREW}/share/man"
    install_c_lib() { cp -a "$1${BREW}/". "${BREW}/"; }
else
    NCPUS=$(nproc)
    LIB_ROOT=/usr/local/lib
    INCLUDE_ROOT=/usr/local/include
    BIN_ROOT=/usr/local/bin
    MAN_ROOT=/usr/local/share/man
    install_c_lib() { cp -a "$1/usr/local/". /usr/local/; ldconfig; }
fi

INSTALL_ROOTS="INSTALL_ROOT_PY3_MODULES=${PY_PLATLIB}
               INSTALL_ROOT_LIB=${LIB_ROOT}
               INSTALL_ROOT_INCLUDE=${INCLUDE_ROOT}
               INSTALL_ROOT_BIN=${BIN_ROOT}
               INSTALL_ROOT_MAN=${MAN_ROOT}"

# ---------------------------------------------------------------------------
# GL_image_display
# ---------------------------------------------------------------------------
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
