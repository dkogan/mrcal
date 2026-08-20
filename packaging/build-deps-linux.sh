#!/bin/bash

# Experimental pip-building thing for mrcal.
#
# Written 100% by Claude. Pip is heinous and stupid, and nobody should be using
# it. This may or may not work for you. If it does not, let me know, and/or
# better yet, send me a patch!

# Install C build dependencies inside the manylinux (AlmaLinux) container.
# Called by cibuildwheel's before-all hook on Linux.
set -ex

NCPUS=$(nproc)
source "$(dirname "$0")/build-deps-common.sh"

mkdir -p "${BUILD_DEPS}/include" "${BUILD_DEPS}/lib" "${BUILD_DEPS}/lib64" "${BUILD_DEPS}/bin"

export PATH="${BUILD_DEPS}/bin:$PATH"
export CPATH="${BUILD_DEPS}/include"
export LIBRARY_PATH="${BUILD_DEPS}/lib:${BUILD_DEPS}/lib64"

# Make BUILD_DEPS/lib visible to the dynamic linker (needed for auditwheel ldd).
echo "${BUILD_DEPS}/lib"    > /etc/ld.so.conf.d/mrcal-build-deps.conf
echo "${BUILD_DEPS}/lib64" >> /etc/ld.so.conf.d/mrcal-build-deps.conf

# EPEL provides SuiteSparse, re2c, etc.
dnf install -y --setopt=keepcache=1 epel-release
dnf install -y --setopt=keepcache=1 \
    suitesparse-devel \
    openblas-devel \
    libpng-devel \
    libjpeg-devel \
    cairo-devel \
    pango-devel \
    libX11-devel \
    libXt-devel \
    chrpath \
    pkgconf \
    binutils \
    perl \
    perl-List-MoreUtils \
    git \
    make \
    cmake \
    mesa-libGL-devel \
    mesa-libGLU-devel \
    libepoxy-devel \
    freeglut-devel \
    swig \
    boost-devel

# openblas-devel doesn't provide liblapack.so; create a symlink so -llapack resolves to openblas
ln -sf /usr/lib64/libopenblas.so ${BUILD_DEPS}/liblapack.so
ln -sf /usr/lib64/libopenblas.so ${BUILD_DEPS}/libblas.so
ldconfig

build_re2c
build_fltk
ldconfig

install_mrbuild
install_stb
build_libdogleg
build_opencv

build_gnuplot ${BUILD_DEPS} --without-qt


# ---------------------------------------------------------------------------
# GL_image_display and mrgingham — cloned here; built per Python version in
# the before-build hook so the Python extension links against the right ABI.
# ---------------------------------------------------------------------------
clone_gl_image_display
clone_mrgingham

strip_installed
