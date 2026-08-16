#!/bin/bash

# Shared build logic sourced by build-deps-linux.sh and build-deps-macos.sh.
# Callers must set NCPUS before sourcing this file.

MRBUILD_VER=1.19
LIBDOGLEG_VER=0.18
GNUPLOT_VER=6.0.2
GL_IMAGE_DISPLAY_COMMIT=d1c2651
MRGINGHAM_COMMIT=a2ec919

# All custom-built C dependencies install here.  The build scripts, before-build
# hook, and Python build backend all reference this path so everything agrees.
BUILD_DEPS="${HOME}/build-deps"

strip_installed() {
    # Strip debug symbols from libs and executables installed to BUILD_DEPS.
    if [ "$(uname)" = "Darwin" ]; then
        find "${BUILD_DEPS}/lib" ! -type l -type f \
            -exec strip -x {} \; 2>/dev/null || true
        find "${BUILD_DEPS}/bin" -maxdepth 1 ! -type l -type f \
            -exec strip {} \; 2>/dev/null || true
    else
        find "${BUILD_DEPS}/lib" ! -type l -type f \
            -exec strip --strip-debug {} \; 2>/dev/null || true
        find "${BUILD_DEPS}/bin" -maxdepth 1 ! -type l -type f \
            -exec strip {} \; 2>/dev/null || true
    fi
}

install_mrbuild() {
    mkdir -p "${BUILD_DEPS}/include" "${BUILD_DEPS}/bin"
    curl -fsSL "https://github.com/dkogan/mrbuild/archive/refs/tags/v${MRBUILD_VER}.tar.gz" | tar xz -C /tmp
    mkdir -p "${BUILD_DEPS}/include/mrbuild"
    cp /tmp/mrbuild-${MRBUILD_VER}/Makefile.common.* "${BUILD_DEPS}/include/mrbuild/"
    find /tmp/mrbuild-${MRBUILD_VER} -maxdepth 1 -name '*.mk' -exec cp {} "${BUILD_DEPS}/include/mrbuild/" \;
    if [ -d /tmp/mrbuild-${MRBUILD_VER}/bin ]; then
        cp /tmp/mrbuild-${MRBUILD_VER}/bin/* "${BUILD_DEPS}/bin/"
        mkdir -p "${BUILD_DEPS}/include/mrbuild/bin"
        cp /tmp/mrbuild-${MRBUILD_VER}/bin/* "${BUILD_DEPS}/include/mrbuild/bin/"
    fi
    rm -rf /tmp/mrbuild-${MRBUILD_VER}
}

install_stb() {
    mkdir -p "${BUILD_DEPS}/include" "${BUILD_DEPS}/lib"
    git clone --depth=1 https://github.com/nothings/stb /tmp/stb
    mkdir -p "${BUILD_DEPS}/include/stb"
    cp /tmp/stb/*.h "${BUILD_DEPS}/include/stb/"
    # Compile a shared libstb so dependents (GL_image_display) can link -lstb
    cat > /tmp/stb_impl.c << 'EOF'
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"
EOF
    if [ "$(uname)" = "Darwin" ]; then
        cc -dynamiclib -fPIC -I"${BUILD_DEPS}/include" -o "${BUILD_DEPS}/lib/libstb.dylib" /tmp/stb_impl.c
    else
        cc -shared -fPIC -I"${BUILD_DEPS}/include" -o "${BUILD_DEPS}/lib/libstb.so" /tmp/stb_impl.c
    fi
    rm -rf /tmp/stb /tmp/stb_impl.c
}

# Clone and build libdogleg in the current directory.
# Caller installs from there and cleans up.
clone_and_build_libdogleg() {
    rm -rf libdogleg /tmp/libdogleg-staging
    git clone --depth=1 --branch "v${LIBDOGLEG_VER}" https://github.com/dkogan/libdogleg
    ln -sf "${BUILD_DEPS}/include/mrbuild" libdogleg/mrbuild
    make -C libdogleg -j"${NCPUS}"
}

clone_mrgingham() {
    rm -rf /tmp/mrgingham
    git clone https://github.com/dkogan/mrgingham /tmp/mrgingham
    git -C /tmp/mrgingham checkout "${MRGINGHAM_COMMIT}"
    ln -sf "${BUILD_DEPS}/include/mrbuild" /tmp/mrgingham/mrbuild
}

clone_gl_image_display() {
    rm -rf /tmp/GL_image_display
    git clone https://github.com/dkogan/GL_image_display /tmp/GL_image_display
    git -C /tmp/GL_image_display checkout "${GL_IMAGE_DISPLAY_COMMIT}"
    ln -sf "${BUILD_DEPS}/include/mrbuild" /tmp/GL_image_display/mrbuild
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
    strip "${prefix}/bin/gnuplot" 2>/dev/null || true
    cd /
    rm -rf /tmp/gnuplot-${GNUPLOT_VER}
}
