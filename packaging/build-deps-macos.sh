#!/bin/bash

# Experimental pip-building thing for mrcal.
#
# Written 100% by Claude. Pip is heinous and stupid, and nobody should be using
# it. This may or may not work for you. If it does not, let me know, and/or
# better yet, send me a patch!


# Install C build dependencies on macOS.
# Called by cibuildwheel's before-all hook on macOS.
set -ex

NCPUS=$(sysctl -n hw.ncpu)
source "$(dirname "$0")/build-deps-common.sh"

# Non-interactive SSH sessions don't source the shell profile; set PATH explicitly.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

brew install suite-sparse openblas libpng libjpeg re2c cpanminus \
    fltk libepoxy swig opencv
cpanm --notest List::MoreUtils

BREW=$(brew --prefix)

# ---------------------------------------------------------------------------
# mrbuild  (not in Homebrew)
# choose_mrbuild.mk checks for mrbuild/ (local) or /usr/include/mrbuild/;
# /usr/include is SIP-protected on macOS, so install under the Homebrew prefix.
# _mrcal_build_backend.py creates a local mrbuild/ symlink before calling make.
# ---------------------------------------------------------------------------
install_mrbuild "${BREW}/include" "${BREW}/bin"

# ---------------------------------------------------------------------------
# libdogleg  (not in Homebrew; build from source)
# ---------------------------------------------------------------------------
# Homebrew headers/libs are not in the default compiler search path on macOS.
export CPATH="${BREW}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${BREW}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

clone_and_build_libdogleg "${BREW}/include/mrbuild"

make -C libdogleg install DESTDIR=/tmp/libdogleg-staging \
    INSTALL_ROOT_LIB="${BREW}/lib"            \
    INSTALL_ROOT_INCLUDE="${BREW}/include"    \
    INSTALL_ROOT_BIN="${BREW}/bin"            \
    INSTALL_ROOT_MAN="${BREW}/share/man"
cp -a /tmp/libdogleg-staging${BREW}/. ${BREW}/

rm -rf libdogleg /tmp/libdogleg-staging

# ---------------------------------------------------------------------------
# stb single-header image library (not in Homebrew; headers only)
# ---------------------------------------------------------------------------
install_stb "${BREW}/include"

# ---------------------------------------------------------------------------
# gnuplot  (build from source without Qt/lua/readline to keep deps clean)
# ---------------------------------------------------------------------------
build_gnuplot "${BREW}" --without-qt

# ---------------------------------------------------------------------------
# GL_image_display  (not in Homebrew; build from source per Python version
# in before-build hook — clone the source here so before-build can just 'make')
# ---------------------------------------------------------------------------
clone_gl_image_display "${BREW}/include/mrbuild"
clone_mrgingham "${BREW}/include/mrbuild"
