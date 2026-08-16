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
    cmake

# openblas-devel doesn't provide liblapack.so; create a symlink so -llapack resolves to openblas
ln -sf /usr/lib64/libopenblas.so /usr/local/lib/liblapack.so
ln -sf /usr/lib64/libopenblas.so /usr/local/lib/libblas.so
ldconfig

# re2c: EPEL 8 ships 0.14.3 (too old; needs >= 1.0 for flags:tags).  Build 3.1 from source.
RE2C_VER=3.1
curl -fsSL "https://github.com/skvadrik/re2c/releases/download/${RE2C_VER}/re2c-${RE2C_VER}.tar.xz" | tar xJ -C /tmp
cmake -S /tmp/re2c-${RE2C_VER} -B /tmp/re2c-build -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/re2c-build -j"${NCPUS}"
cmake --install /tmp/re2c-build
rm -rf /tmp/re2c-${RE2C_VER} /tmp/re2c-build

# stb single-header image library (not in EPEL; headers only needed —
# USE_LOCAL_STB_IMPLEMENTATION=1 compiles stb into libmrcal, no libstb.so needed)
install_stb /usr/local/include

# ---------------------------------------------------------------------------
# mrbuild  (Makefile library; not in EPEL)
# ---------------------------------------------------------------------------
install_mrbuild /usr/include /usr/bin

# ---------------------------------------------------------------------------
# libdogleg  (not in any RPM repo; build from source)
# ---------------------------------------------------------------------------
clone_and_build_libdogleg ""

# mrbuild requires a non-empty DESTDIR
make -C libdogleg install DESTDIR=/tmp/libdogleg-staging \
    INSTALL_ROOT_LIB=/usr/local/lib         \
    INSTALL_ROOT_INCLUDE=/usr/local/include \
    INSTALL_ROOT_BIN=/usr/local/bin
cp -a /tmp/libdogleg-staging/. /
ldconfig
rm -rf libdogleg /tmp/libdogleg-staging

# ---------------------------------------------------------------------------
# gnuplot  (not in EPEL; build from source without X11/Qt to keep deps clean)
# The build backend bundles the gnuplot binary + its shared lib deps into the
# wheel so pip users get a working gnuplot without a separate system install.
# ---------------------------------------------------------------------------
build_gnuplot /usr/local --without-qt
