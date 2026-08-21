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

mkdir -p "${BUILD_DEPS}/include" "${BUILD_DEPS}/lib" "${BUILD_DEPS}/bin"

export PATH="${BUILD_DEPS}/bin:/opt/homebrew/bin:/opt/homebrew/opt/qt/bin:$PATH"
export CPATH="${BUILD_DEPS}/include:/opt/homebrew/include"
export LIBRARY_PATH="${BUILD_DEPS}/lib:/opt/homebrew/lib"

brew install suite-sparse openblas libpng libjpeg re2c cpanminus \
    fltk freeglut libepoxy swig boost mesa-glu gnu-getopt cmake \
    qt cairo pango mawk
cpanm --notest List::MoreUtils

install_mrbuild
install_stb
build_libdogleg
build_opencv

build_gnuplot "/opt/homebrew" CXXFLAGS="-std=c++17"

install_vnlog
clone_gl_image_display
clone_mrgingham

strip_installed
