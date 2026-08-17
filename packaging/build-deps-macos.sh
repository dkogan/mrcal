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
    fltk freeglut libepoxy swig boost mesa-glu gnu-getopt cmake \
    qt cairo pango
cpanm --notest List::MoreUtils

BREW=$(brew --prefix)

mkdir -p "${BUILD_DEPS}/include" "${BUILD_DEPS}/lib" "${BUILD_DEPS}/bin"

# Make BUILD_DEPS and Homebrew visible to everything that follows.
export CPATH="${BUILD_DEPS}/include:${BREW}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${BUILD_DEPS}/lib:${BREW}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

# ---------------------------------------------------------------------------
# mrbuild  (not in Homebrew)
# ---------------------------------------------------------------------------
install_mrbuild

# ---------------------------------------------------------------------------
# libdogleg  (not in Homebrew; build from source)
# ---------------------------------------------------------------------------
clone_and_build_libdogleg

make -C libdogleg install DESTDIR=/tmp/libdogleg-staging \
    INSTALL_ROOT_LIB="${BUILD_DEPS}/lib"         \
    INSTALL_ROOT_INCLUDE="${BUILD_DEPS}/include" \
    INSTALL_ROOT_BIN="${BUILD_DEPS}/bin"         \
    INSTALL_ROOT_MAN="${BUILD_DEPS}/share/man"
cp -a /tmp/libdogleg-staging"${BUILD_DEPS}"/. "${BUILD_DEPS}"/
rm -rf libdogleg /tmp/libdogleg-staging
strip_installed

# ---------------------------------------------------------------------------
# stb single-header image library (not in Homebrew; headers only)
# ---------------------------------------------------------------------------
install_stb

# ---------------------------------------------------------------------------
# OpenCV  (minimal build — only the modules mrgingham needs)
# ---------------------------------------------------------------------------
build_opencv

# ---------------------------------------------------------------------------
# gnuplot  (build from source with Qt and cairo terminals; without lua/readline)
# ---------------------------------------------------------------------------
export PATH="${BREW}/opt/qt/bin:${PATH}"
build_gnuplot "${BREW}" CXXFLAGS="-std=c++17"

# ---------------------------------------------------------------------------
# GL_image_display and mrgingham — cloned here; built per Python version in
# the before-build hook so the Python extension links against the right ABI.
# ---------------------------------------------------------------------------
clone_gl_image_display
clone_mrgingham
