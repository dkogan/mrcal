#!/bin/bash

# Shared build logic sourced by build-deps-linux.sh and build-deps-macos.sh.
# Callers must set NCPUS before sourcing this file.

MRBUILD_VER=1.19
LIBDOGLEG_VER=0.18
FLTK_VER=1.4.5
GNUPLOT_VER=6.0.2
GL_IMAGE_DISPLAY_COMMIT=dbd3eb0
MRGINGHAM_VER=1.27
OPENCV_VER=4.11.0
RE2C_VER=3.1

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
        find "${BUILD_DEPS}/lib64" ! -type l -type f \
            -exec strip --strip-debug {} \; 2>/dev/null || true
        find "${BUILD_DEPS}/bin" -maxdepth 1 ! -type l -type f \
            -exec strip {} \; 2>/dev/null || true
    fi
}

install_mrbuild() {
    curl -fsSL "https://github.com/dkogan/mrbuild/archive/refs/tags/v${MRBUILD_VER}.tar.gz" | tar xz -C /tmp
    mv /tmp/mrbuild-${MRBUILD_VER} /tmp/mrbuild
    mkdir -p "${BUILD_DEPS}/include/mrbuild"
    cp /tmp/mrbuild/Makefile.common.* "${BUILD_DEPS}/include/mrbuild/"
    cp /tmp/mrbuild/bin/*             "${BUILD_DEPS}/bin/"
}

build_fltk() {
    curl -fsSL "https://github.com/fltk/fltk/releases/download/release-${FLTK_VER}/fltk-${FLTK_VER}-source.tar.gz" \
        | tar xz -C /tmp
    local build_dir=/tmp/fltk-build
    rm -rf "$build_dir"
    cmake -S /tmp/fltk-${FLTK_VER} -B "$build_dir" \
        -DCMAKE_INSTALL_PREFIX="${BUILD_DEPS}" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release \
        -DFLTK_BUILD_SHARED_LIBS=ON \
        -DFLTK_BUILD_TEST=OFF \
        -DFLTK_BUILD_EXAMPLES=OFF \
        -DOPTION_BUILD_GL=ON \
        -DOPTION_USE_SYSTEM_LIBJPEG=ON \
        -DOPTION_USE_SYSTEM_LIBPNG=ON \
        -DOPTION_USE_SYSTEM_ZLIB=ON
    cmake --build "$build_dir" -j"${NCPUS}"
    cmake --install "$build_dir"
    rm -rf /tmp/fltk-${FLTK_VER} "$build_dir"
}

install_stb() {
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

build_libdogleg() {
    git clone --depth=1 --branch "v${LIBDOGLEG_VER}" https://github.com/dkogan/libdogleg /tmp/libdogleg
    ln -sf /tmp/mrbuild /tmp/libdogleg/mrbuild
    make -C /tmp/libdogleg -j"${NCPUS}"

    DESTDIR=${BUILD_DEPS}           \
    INSTALL_ROOT_LIB="/lib"         \
    INSTALL_ROOT_INCLUDE="/include" \
    INSTALL_ROOT_BIN="/bin"         \
    INSTALL_ROOT_MAN="/share/man"   \
      make -C /tmp/libdogleg install
    rm -rf /tmp/libdogleg
    if [ "$(uname)" != "Darwin" ]; then ldconfig; fi
}

clone_mrgingham() {
    git clone --depth=1 --branch "v${MRGINGHAM_VER}" \
        https://github.com/dkogan/mrgingham /tmp/mrgingham
    ln -sf /tmp/mrbuild /tmp/mrgingham/mrbuild
}

clone_gl_image_display() {
    git clone https://github.com/dkogan/GL_image_display /tmp/GL_image_display
    git -C /tmp/GL_image_display checkout "${GL_IMAGE_DISPLAY_COMMIT}"
    ln -sf /tmp/mrbuild /tmp/GL_image_display/mrbuild
}

build_opencv() {
    curl -fsSL "https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VER}.tar.gz" \
        | tar xz -C /tmp
    local build_dir=/tmp/opencv-build
    rm -rf "$build_dir"
    local cmake_args=(
        -DCMAKE_INSTALL_PREFIX="${BUILD_DEPS}"
        -DCMAKE_BUILD_TYPE=Release
        # Only the modules mrgingham actually needs
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,features2d,flann,calib3d
        # No heavyweight optional backends
        -DWITH_VTK=OFF
        -DWITH_CERES=OFF
        -DWITH_OPENVINO=OFF
        -DWITH_CUDA=OFF
        -DWITH_OPENCL=OFF
        -DWITH_QT=OFF
        -DWITH_GTK=OFF
        -DWITH_FFMPEG=OFF
        -DWITH_GSTREAMER=OFF
        -DBUILD_TESTS=OFF
        -DBUILD_PERF_TESTS=OFF
        -DBUILD_EXAMPLES=OFF
        -DBUILD_opencv_python3=OFF
        -DBUILD_opencv_python2=OFF
        -DBUILD_SHARED_LIBS=ON
        # OpenCV 4 defaults to include/opencv4/; flatten to include/ so that
        # #include <opencv2/...> works with a plain -I${BUILD_DEPS}/include.
        -DOPENCV_INCLUDE_INSTALL_PATH=include
    )
    if [ "$(uname)" = "Darwin" ]; then
        cmake_args+=(
            -DCMAKE_PREFIX_PATH="${BREW}"
            -DCMAKE_OSX_ARCHITECTURES=arm64
            -DCMAKE_OSX_DEPLOYMENT_TARGET=14.0
        )
    fi
    cmake -S /tmp/opencv-${OPENCV_VER} -B "$build_dir" "${cmake_args[@]}"
    cmake --build "$build_dir" -j"${NCPUS}"
    cmake --install "$build_dir"
    rm -rf /tmp/opencv-${OPENCV_VER} "$build_dir"

    # OpenCV 4.11 + CMake 4.x have a compatibility bug in OpenCVGenPkgconfig.cmake,
    # so we generate opencv4.pc manually.
    mkdir -p "${BUILD_DEPS}/lib/pkgconfig"
    cat > "${BUILD_DEPS}/lib/pkgconfig/opencv4.pc" << EOF
prefix=${BUILD_DEPS}
exec_prefix=\${prefix}
libdir=\${exec_prefix}/lib
includedir=\${prefix}/include

Name: OpenCV
Description: Open Source Computer Vision Library
Version: ${OPENCV_VER}
Libs: -L\${libdir} -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
Cflags: -I\${includedir}
EOF

    if [ "$(uname)" != "Darwin" ]; then ldconfig; fi
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
    rm -rf /tmp/gnuplot-${GNUPLOT_VER}
}

build_re2c() {
    # re2c: EPEL 8 ships 0.14.3 (too old; needs >= 1.0 for flags:tags).  Build 3.1 from source.
    curl -fsSL "https://github.com/skvadrik/re2c/releases/download/${RE2C_VER}/re2c-${RE2C_VER}.tar.xz" | tar xJ -C /tmp
    cmake -S /tmp/re2c-${RE2C_VER} -B /tmp/re2c-build -DCMAKE_INSTALL_PREFIX=${BUILD_DEPS} -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/re2c-build -j"${NCPUS}"
    cmake --install /tmp/re2c-build
    rm -rf /tmp/re2c-${RE2C_VER} /tmp/re2c-build
}
