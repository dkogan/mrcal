#!/bin/bash

# Shared build logic sourced by build-deps-linux.sh and build-deps-macos.sh.
# Callers must set NCPUS before sourcing this file.

MRBUILD_VER=1.19
LIBDOGLEG_VER=0.18
GNUPLOT_VER=6.0.2
GL_IMAGE_DISPLAY_VER=0.24
MRGINGHAM_VER=1.26

install_mrbuild() {
    local include_dir=$1  # e.g. /usr/include or ${BREW}/include
    local bin_dir=$2      # e.g. /usr/bin or ${BREW}/bin
    curl -fsSL "https://github.com/dkogan/mrbuild/archive/refs/tags/v${MRBUILD_VER}.tar.gz" | tar xz -C /tmp
    mkdir -p "${include_dir}/mrbuild"
    cp /tmp/mrbuild-${MRBUILD_VER}/Makefile.common.* "${include_dir}/mrbuild/"
    find /tmp/mrbuild-${MRBUILD_VER} -maxdepth 1 -name '*.mk' -exec cp {} "${include_dir}/mrbuild/" \;
    [ -d /tmp/mrbuild-${MRBUILD_VER}/bin ] && cp /tmp/mrbuild-${MRBUILD_VER}/bin/* "${bin_dir}/"
    rm -rf /tmp/mrbuild-${MRBUILD_VER}
}

install_stb() {
    local include_dir=$1  # e.g. /usr/local/include or ${BREW}/include
    git clone --depth=1 https://github.com/nothings/stb /tmp/stb
    mkdir -p "${include_dir}/stb"
    cp /tmp/stb/*.h "${include_dir}/stb/"
    rm -rf /tmp/stb
}

# Clone and build libdogleg, leaving the staging tree at /tmp/libdogleg-staging.
# Caller is responsible for copying from staging and cleaning up.
clone_and_build_libdogleg() {
    local mrbuild_link=$1  # path to symlink as libdogleg/mrbuild (macOS); empty on Linux
    rm -rf libdogleg /tmp/libdogleg-staging
    git clone --depth=1 --branch "v${LIBDOGLEG_VER}" https://github.com/dkogan/libdogleg
    [ -n "${mrbuild_link}" ] && ln -sf "${mrbuild_link}" libdogleg/mrbuild
    make -C libdogleg -j"${NCPUS}"
}

# Clone GL_image_display source to /tmp/GL_image_display for per-Python builds
# in the before-build hook.  Does not build; callers do that.
clone_mrgingham() {
    local mrbuild_link=$1  # path to symlink as mrbuild (macOS); empty on Linux
    rm -rf /tmp/mrgingham
    git clone --depth=1 --branch "upstream/${MRGINGHAM_VER}" \
        https://salsa.debian.org/science-team/mrgingham /tmp/mrgingham
    [ -n "${mrbuild_link}" ] && ln -sf "${mrbuild_link}" /tmp/mrgingham/mrbuild
}

clone_gl_image_display() {
    local mrbuild_link=$1  # path to symlink as mrbuild (macOS); empty on Linux
    rm -rf /tmp/GL_image_display
    git clone --depth=1 --branch "v${GL_IMAGE_DISPLAY_VER}" \
        https://github.com/dkogan/GL_image_display /tmp/GL_image_display
    [ -n "${mrbuild_link}" ] && ln -sf "${mrbuild_link}" /tmp/GL_image_display/mrbuild
}

build_gnuplot() {
    local prefix=$1
    shift
    curl -fsSL "https://sourceforge.net/projects/gnuplot/files/gnuplot/${GNUPLOT_VER}/gnuplot-${GNUPLOT_VER}.tar.gz/download" \
        | tar xz -C /tmp
    cd /tmp/gnuplot-${GNUPLOT_VER}
    ./configure --prefix="${prefix}" \
        --without-lua \
        --without-readline \
        "$@"
    make ${LRELEASE:+LRELEASE="$LRELEASE"} -j"${NCPUS}"
    make install ${LRELEASE:+LRELEASE="$LRELEASE"}
    cd /
    rm -rf /tmp/gnuplot-${GNUPLOT_VER}
}
