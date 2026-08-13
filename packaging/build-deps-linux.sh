#!/bin/bash

# Experimental pip-building thing for mrcal.
#
# Written 100% by Claude. Pip is heinous and stupid, and nobody should be using
# it. This may or may not work for you. If it does not, let me know, and/or
# better yet, send me a patch!

# Install C build dependencies inside the manylinux (AlmaLinux) container.
# Called by cibuildwheel's before-all hook on Linux.
set -ex

# EPEL provides SuiteSparse, re2c, etc.
dnf install -y --setopt=keepcache=1 epel-release
dnf install -y --setopt=keepcache=1 \
    suitesparse-devel \
    openblas-devel \
    libpng-devel \
    libjpeg-devel \
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
cmake --build /tmp/re2c-build -j"$(nproc)"
cmake --install /tmp/re2c-build
rm -rf /tmp/re2c-${RE2C_VER} /tmp/re2c-build

# stb single-header image library (not in EPEL; headers only needed —
# USE_LOCAL_STB_IMPLEMENTATION=1 compiles stb into libmrcal, no libstb.so needed)
git clone --depth=1 https://github.com/nothings/stb /tmp/stb
mkdir -p /usr/local/include/stb
cp /tmp/stb/*.h /usr/local/include/stb/
rm -rf /tmp/stb

# ---------------------------------------------------------------------------
# mrbuild  (Makefile library; not in EPEL)
# ---------------------------------------------------------------------------
V=1.16
cd /tmp
curl -fsSL "https://github.com/dkogan/mrbuild/archive/refs/tags/v${V}.tar.gz" | tar xz
mkdir -p /usr/include/mrbuild
cp /tmp/mrbuild-${V}/Makefile.common.* /usr/include/mrbuild/
find /tmp/mrbuild-${V} -maxdepth 1 -name '*.mk' -exec cp {} /usr/include/mrbuild/ \;
if [ -d /tmp/mrbuild-${V}/bin ]; then
    cp /tmp/mrbuild-${V}/bin/* /usr/bin/
fi
rm -rf /tmp/mrbuild-${V}
cd -

# ---------------------------------------------------------------------------
# libdogleg  (not in any RPM repo; build from source)
# ---------------------------------------------------------------------------
LIBDOGLEG_VER=0.18

git clone --depth=1 --branch "v${LIBDOGLEG_VER}" https://github.com/dkogan/libdogleg
make -C libdogleg -j"$(nproc)"

# mrbuild requires a non-empty DESTDIR
make -C libdogleg install DESTDIR=/tmp/libdogleg-staging \
    INSTALL_ROOT_LIB=/usr/local/lib         \
    INSTALL_ROOT_INCLUDE=/usr/local/include \
    INSTALL_ROOT_BIN=/usr/local/bin
cp -a /tmp/libdogleg-staging/. /
# mrbuild installs dogleg.h into /usr/include/dogleg/; mrcal includes it as <dogleg.h>
cp /usr/include/dogleg/dogleg.h /usr/local/include/
ldconfig
rm -rf libdogleg /tmp/libdogleg-staging
