#!/bin/bash

# Experimental pip-building thing for mrcal.
#
# Written 100% by Claude. Pip is heinous and stupid, and nobody should be using
# it. This may or may not work for you. If it does not, let me know, and/or
# better yet, send me a patch!


# Install C build dependencies on macOS.
# Called by cibuildwheel's before-all hook on macOS.
set -ex

# Non-interactive SSH sessions don't source the shell profile; set PATH explicitly.
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

brew install suite-sparse openblas libpng libjpeg re2c

BREW=$(brew --prefix)

# ---------------------------------------------------------------------------
# mrbuild  (not in Homebrew)
# choose_mrbuild.mk checks for mrbuild/ (local) or /usr/include/mrbuild/;
# /usr/include is SIP-protected on macOS, so install under the Homebrew prefix.
# _mrcal_build_backend.py creates a local mrbuild/ symlink before calling make.
# ---------------------------------------------------------------------------
V=1.16
curl -fsSL "https://github.com/dkogan/mrbuild/archive/refs/tags/v${V}.tar.gz" | tar xz -C /tmp
mkdir -p "${BREW}/include/mrbuild"
cp /tmp/mrbuild-${V}/Makefile.common.* "${BREW}/include/mrbuild/"
find /tmp/mrbuild-${V} -maxdepth 1 -name '*.mk' -exec cp {} "${BREW}/include/mrbuild/" \;
if [ -d /tmp/mrbuild-${V}/bin ]; then
    cp /tmp/mrbuild-${V}/bin/* "${BREW}/bin/"
fi
rm -rf /tmp/mrbuild-${V}

# ---------------------------------------------------------------------------
# libdogleg  (not in Homebrew; build from source)
# ---------------------------------------------------------------------------
# Homebrew headers/libs are not in the default compiler search path on macOS.
export CPATH="${BREW}/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="${BREW}/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

LIBDOGLEG_VER=0.18

rm -rf libdogleg /tmp/libdogleg-staging
git clone --depth=1 --branch "v${LIBDOGLEG_VER}" https://github.com/dkogan/libdogleg
# mrbuild/ symlink must exist in libdogleg's dir so its Makefile can find mrbuild
ln -sf "${BREW}/include/mrbuild" libdogleg/mrbuild
make -C libdogleg -j"$(sysctl -n hw.ncpu)"

make -C libdogleg install DESTDIR=/tmp/libdogleg-staging \
    INSTALL_ROOT_LIB="${BREW}/lib"         \
    INSTALL_ROOT_INCLUDE="${BREW}/include" \
    INSTALL_ROOT_BIN="${BREW}/bin"

# mrbuild staging puts headers in usr/include/dogleg/ and libs in usr/lib64/
# regardless of INSTALL_ROOT_*; copy manually to the Homebrew prefix.
cp /tmp/libdogleg-staging/usr/include/dogleg/dogleg.h "${BREW}/include/"
cp -P /tmp/libdogleg-staging/usr/lib64/libdogleg.* "${BREW}/lib/" 2>/dev/null || \
    cp -P /tmp/libdogleg-staging/usr/lib/libdogleg.* "${BREW}/lib/"

rm -rf libdogleg /tmp/libdogleg-staging
