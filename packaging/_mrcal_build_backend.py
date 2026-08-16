"""Experimental pip-building thing for mrcal.

Written 100% by Claude. Pip is heinous and stupid, and nobody should be using
it. This may or may not work for you. If it does not, let me know, and/or better
yet, send me a patch!

Minimal PEP 517 build backend for mrcal.

Build pipeline:
  1. make -j<N>     — compile libmrcal + Python extensions via mrbuild/Makefile
  2. pack raw wheel — mrcal/ package files + CLI scripts (no lib bundling yet)
  3. repair wheel   — bundle libmrcal + all transitive C deps, rewrite RPATHs
       Linux : auditwheel repair  →  manylinux-tagged self-contained wheel
       macOS : delocate-wheel     →  macosx-tagged self-contained wheel
  4. return repaired wheel name

The platform-appropriate repair tool is declared as a build requirement so
pip installs it automatically in the isolated build environment.

numpysane is also a build requirement: the *-genpywrap.py code-generation
scripts (run by make) import it.

LD_LIBRARY_PATH / DYLD_LIBRARY_PATH is pointed at the mrcal source tree so
the repair tool's ldd/otool invocations can find libmrcal (built there by
make, not yet in any system library path).  On Linux the extensions use
RUNPATH (not RPATH), so LD_LIBRARY_PATH takes precedence.

"""

import base64
import glob
import hashlib
import os
import re
import shutil
import sys
import zipfile
import tempfile
import subprocess

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DEPS = os.path.expanduser('~/build-deps')


# ---------------------------------------------------------------------------
# Version / metadata
# ---------------------------------------------------------------------------

def _version():
    VERSION_WHEEL_BASE = os.environ.get('VERSION_WHEEL_BASE', '').strip()
    VERSION_WHEEL_POST = os.environ.get('VERSION_WHEEL_POST', '').strip()
    print(f"{VERSION_WHEEL_BASE=} {VERSION_WHEEL_POST=}")
    if not VERSION_WHEEL_BASE:
        raise RuntimeError("VERSION_WHEEL_BASE is not set")
    if not VERSION_WHEEL_POST:
        raise RuntimeError("VERSION_WHEEL_POST is not set")
    return f"{VERSION_WHEEL_BASE}.post{VERSION_WHEEL_POST}"


def _metadata_text(version):
    return (
        f"Metadata-Version: 2.1\n"
        f"Name: mrcal\n"
        f"Version: {version}\n"
        f"Summary: Calibration and SFM library\n"
        f"Home-page: http://mrcal.secretsauce.net\n"
        f"License: Apache-2.0\n"
        f"Requires-Python: >=3.8\n"
        f"Requires-Dist: numpy\n"
        f"Requires-Dist: numpysane>=0.35\n"
        f"Requires-Dist: scipy>=0.18\n"
        f"Requires-Dist: opencv-python-headless\n"
        f"Requires-Dist: gnuplotlib>=0.38\n"
        f"Requires-Dist: shapely\n"
        f"Requires-Dist: ipython\n"
        f"Requires-Dist: pyyaml\n"
        f"Requires-Dist: pyfltk\n"
    )


# ---------------------------------------------------------------------------
# Wheel helpers
# ---------------------------------------------------------------------------

def _raw_wheel_tag():
    import sysconfig
    ver    = "".join(str(v) for v in sys.version_info[:2])
    abbrev = {"cpython": "cp", "pypy": "pp", "ironpython": "ip", "jython": "jy"}
    py     = abbrev.get(sys.implementation.name, sys.implementation.name) + ver
    plat   = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return f"{py}-{py}-{plat}"


def _wheel_header(version):
    py, abi, plat = _raw_wheel_tag().split("-")
    return (
        f"Wheel-Version: 1.0\n"
        f"Generator: _mrcal_build_backend\n"
        f"Root-Is-Purelib: false\n"
        f"Tag: {py}-{abi}-{plat}\n"
    )


_VENDOR_SETUP = """\
def _mrcal_setup_vendor():
    import os, glob, shutil, sys
    vendor = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_vendor')
    # Set gnuplot env vars so the bundled data files (terminals, help, etc.)
    # are found regardless of system gnuplot installation.
    if shutil.which('gnuplot'):
        os.environ.setdefault('GNUTERM', 'qt' if sys.platform == 'darwin' else 'x11')
        vers = sorted(glob.glob(os.path.join(vendor, 'share', 'gnuplot', '*')))
        if vers:
            os.environ.setdefault('GNUPLOT_LIB', vers[-1])
            gih = os.path.join(vers[-1], 'gnuplot.gih')
            if os.path.exists(gih):
                os.environ.setdefault('GNUPLOT_HELP', gih)
            ps_dir = os.path.join(vers[-1], 'PostScript')
            if os.path.isdir(ps_dir):
                os.environ.setdefault('GNUPLOT_PS_DIR', ps_dir)
        libexec = os.path.join(vendor, 'libexec', 'gnuplot')
        if os.path.isdir(libexec):
            vers = sorted(glob.glob(os.path.join(libexec, '*')))
            if vers:
                os.environ.setdefault('GNUPLOT_DRIVER_DIR', vers[-1])
_mrcal_setup_vendor()
del _mrcal_setup_vendor
"""


def _build_raw_wheel(raw_wheel_path, version, brew=None):
    pkg_dir   = f"{SRC}/mrcal"
    impl      = sys.implementation.name
    ver       = "".join(str(v) for v in sys.version_info[:2])
    ext_tag   = f"{impl}-{ver}"
    dist_info = f"mrcal-{version}.dist-info"
    data_dir  = f"mrcal-{version}.data"
    records   = []

    gnuplot_bin = shutil.which("gnuplot") or \
        (os.path.join(brew, "bin", "gnuplot") if brew else None)

    def add(zf, data, arcname, mode=0o644):
        if isinstance(data, str):
            data = data.encode()
        info = zipfile.ZipInfo(arcname)
        info.external_attr = mode << 16
        zf.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED)
        digest = hashlib.sha256(data).digest()
        sha    = "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        records.append((arcname, sha, len(data)))

    with zipfile.ZipFile(raw_wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        # Package files from the build tree
        for path in sorted(glob.glob(f"{pkg_dir}/**/*", recursive=True)):
            if not os.path.isfile(path):
                continue
            if "__pycache__" in path or path.endswith(".pyc"):
                continue
            # Only include .so files for the current Python version
            stem = os.path.splitext(os.path.basename(path))[0]
            if path.endswith(".so") and ext_tag not in stem:
                continue
            arcname = "mrcal/" + os.path.relpath(path, pkg_dir)
            data = open(path, "rb").read()
            # Prepend gnuplot env-var setup to __init__.py so data files are found
            if gnuplot_bin and path == f"{pkg_dir}/__init__.py":
                data = _VENDOR_SETUP.encode() + data
            add(zf, data, arcname)

        # Bundled gnuplot — full installation tree so help, terminals, etc. work.
        # auditwheel treats ELF files here like .so deps and bundles their libs.
        if gnuplot_bin:
            gnuplot_prefix = os.path.dirname(os.path.dirname(gnuplot_bin))

            # Main binary — installed to {venv}/bin/ by pip via the data/scripts mechanism
            add(zf, open(gnuplot_bin, "rb").read(), f"{data_dir}/scripts/gnuplot", mode=0o755)

            # Data files: .gih help, terminal scripts, colour names, etc.
            for src_dir, arc_prefix in [
                (f"{gnuplot_prefix}/share/gnuplot",   "mrcal/_vendor/share/gnuplot"),
                (f"{gnuplot_prefix}/libexec/gnuplot",  "mrcal/_vendor/libexec/gnuplot"),
            ]:
                if not os.path.isdir(src_dir):
                    continue
                for path in sorted(glob.glob(f"{src_dir}/**/*", recursive=True)):
                    if not os.path.isfile(path):
                        continue
                    arcname = arc_prefix + "/" + os.path.relpath(path, src_dir)
                    mode = 0o755 if os.access(path, os.X_OK) else 0o644
                    add(zf, open(path, "rb").read(), arcname, mode=mode)

        # Bundled mrgingham — binaries to mrcal/_vendor/bin/ (PATH is set in
        # __init__.py); Python extension at wheel root for 'import mrgingham'.
        # auditwheel/delocate bundles OpenCV and other C lib deps.
        # Fixed paths set by before-build.sh — independent of any venv's platlib.
        _mrg_staging = '/tmp/mrgingham-staging'
        _mrg_bin_dir = _mrg_staging + BUILD_DEPS + '/bin'
        if os.path.isdir(_mrg_bin_dir):
            for path in sorted(glob.glob(f'{_mrg_bin_dir}/mrgingham*')):
                if os.path.isfile(path):
                    add(zf, open(path, 'rb').read(),
                        f'{data_dir}/scripts/{os.path.basename(path)}', mode=0o755)
        _mrg_py_dir = '/tmp/mrg-pylib'
        if os.path.isdir(_mrg_py_dir):
            for path in sorted(glob.glob(f'{_mrg_py_dir}/mrgingham*')):
                if os.path.isfile(path):
                    mode = 0o755 if path.endswith('.so') else 0o644
                    add(zf, open(path, 'rb').read(), os.path.basename(path), mode=mode)

        # Bundled GL_image_display — Fl_Gl_Image_Widget.py + _Fl_Gl_Image_Widget.so
        # placed at the wheel root so they land in site-packages alongside mrcal/
        # and are importable as standalone modules.  auditwheel/delocate bundles
        # their C library dependencies (libGL_image_display_fltk, libfltk, etc.)
        _gl_py_dir = '/tmp/gl-pylib'
        if os.path.isdir(_gl_py_dir):
            for path in sorted(glob.glob(f'{_gl_py_dir}/Fl_Gl_Image_Widget*')):
                if os.path.isfile(path):
                    mode = 0o755 if path.endswith('.so') else 0o644
                    add(zf, open(path, 'rb').read(), os.path.basename(path), mode=mode)

        # CLI scripts from the source root
        for script in sorted(glob.glob(f"{SRC}/mrcal-*")):
            if os.path.isfile(script) and os.access(script, os.X_OK):
                name = os.path.basename(script)
                data = open(script, "rb").read()
                # pip rewrites "#!python" to the venv interpreter path at install time
                if data.startswith(b"#!"):
                    data = b"#!python\n" + data[data.index(b"\n") + 1:]
                add(zf, data, f"{data_dir}/scripts/{name}", mode=0o755)

        # dist-info
        add(zf, _metadata_text(version), f"{dist_info}/METADATA")
        add(zf, _wheel_header(version),  f"{dist_info}/WHEEL")

        record_lines  = "\n".join(f"{n},{h},{s}" for n, h, s in records)
        record_lines += f"\n{dist_info}/RECORD,,"
        zf.writestr(f"{dist_info}/RECORD", record_lines)


# ---------------------------------------------------------------------------
# PEP 517 hooks
# ---------------------------------------------------------------------------

def get_requires_for_build_wheel(config_settings=None):
    # numpysane: imported by the *-genpywrap.py scripts that make runs
    # auditwheel / delocate: bundle C libraries into the wheel
    repair = "delocate" if sys.platform == "darwin" else "auditwheel"
    return ["numpysane", "numpy", repair]



def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    version   = _version()
    dist_info = f"{metadata_directory}/mrcal-{version}.dist-info"
    os.makedirs(dist_info, exist_ok=True)
    with open(f"{dist_info}/METADATA", "w") as f: f.write(_metadata_text(version))
    with open(f"{dist_info}/WHEEL",    "w") as f: f.write(_wheel_header(version))
    return f"mrcal-{version}.dist-info"


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    version = _version()
    ncpus   = os.cpu_count() or 4

    # numpy headers are installed into the isolated build venv but make runs
    # outside it; add numpy's include dir via C_INCLUDE_PATH so GCC finds
    # numpy/arrayobject.h without any Makefile changes.
    import numpy
    env = os.environ.copy()
    numpy_inc = numpy.get_include()
    env["C_INCLUDE_PATH"] = numpy_inc + (":" + env["C_INCLUDE_PATH"] if env.get("C_INCLUDE_PATH") else "")

    # All custom C deps live in BUILD_DEPS; add to compiler/linker search paths.
    env["CPATH"]        = BUILD_DEPS + "/include" + (":" + env["CPATH"]        if env.get("CPATH")        else "")
    env["LIBRARY_PATH"] = BUILD_DEPS + "/lib"     + (":" + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else "")

    mrbuild_symlink = None
    if sys.platform == "darwin":
        brew_bin = next((p for p in ["/opt/homebrew/bin/brew", "/usr/local/bin/brew"] if os.path.exists(p)), "brew")
        brew = subprocess.check_output([brew_bin, "--prefix"], text=True).strip()
        # Homebrew headers/libs (FLTK, OpenCV, etc.) are not in the default
        # compiler search path on macOS.
        env["CPATH"]        = f"{brew}/include:" + env["CPATH"]
        env["LIBRARY_PATH"] = f"{brew}/lib:"     + env["LIBRARY_PATH"]
        # mrbuild respects ARCHFLAGS to override the arch flags it gets from
        # Python's sysconfig (which is universal2 for Python.org builds).
        import platform
        env.setdefault("ARCHFLAGS", f"-arch {platform.machine()}")

    # choose_mrbuild.mk checks mrbuild/ (local) or /usr/include/mrbuild/.
    # BUILD_DEPS is neither, so create a temporary local symlink.
    local_link = f"{SRC}/mrbuild"
    if not os.path.exists(local_link):
        candidate = f"{BUILD_DEPS}/include/mrbuild"
        if os.path.isdir(candidate):
            os.symlink(candidate, local_link)
            mrbuild_symlink = local_link

    # USE_LOCAL_STB_IMPLEMENTATION: compile stb into libmrcal rather than
    # linking against an external libstb.so (not available on all platforms).
    make_cmd = ["make", f"-j{ncpus}"]
    if sys.platform != "darwin":
        make_cmd.append("USE_LOCAL_STB_IMPLEMENTATION=1")

    try:
        subprocess.check_call(make_cmd, cwd=SRC, env=env)
    finally:
        if mrbuild_symlink and os.path.islink(mrbuild_symlink):
            os.unlink(mrbuild_symlink)

    # Strip debug symbols from built shared libraries before packing the wheel.
    strip_cmd = ['strip', '-x'] if sys.platform == 'darwin' else ['strip', '--strip-debug']
    for pattern in ['*.so', '*.so.*', '*.dylib']:
        for f in glob.glob(f'{SRC}/**/{pattern}', recursive=True):
            if not os.path.islink(f):
                subprocess.run(strip_cmd + [f], check=False, stderr=subprocess.DEVNULL)

    with tempfile.TemporaryDirectory(prefix="mrcal-raw-wheel-") as tmp:
        tag            = _raw_wheel_tag()
        raw_wheel_path = f"{tmp}/mrcal-{version}-{tag}.whl"

        # 2. Pack raw wheel (extensions keep their $ORIGIN/.. RUNPATH for now;
        #    auditwheel will rewrite everything)
        _build_raw_wheel(raw_wheel_path, version,
                         brew=brew if sys.platform == "darwin" else None)

        # 3. Repair: bundle libmrcal + all transitive C deps into the wheel
        env = os.environ.copy()

        if sys.platform == "darwin":
            # Point DYLD_LIBRARY_PATH at the source tree so delocate's otool
            # resolution can find libmrcal (built there by make).
            lib_path_var = "DYLD_LIBRARY_PATH"
            cmd = ["delocate-wheel", "-w", wheel_directory, raw_wheel_path]
            # Ensure MACOSX_DEPLOYMENT_TARGET is set so delocate accepts the
            # bundled libs (which were built for the current OS if using the
            # local Homebrew).  On CI the runner sets this; locally we detect it.
            import platform
            mac_ver = ".".join(platform.mac_ver()[0].split(".")[:2])
            env["MACOSX_DEPLOYMENT_TARGET"] = mac_ver
        else:
            # Extensions use RUNPATH (not RPATH), so LD_LIBRARY_PATH takes
            # precedence and auditwheel's ldd finds libmrcal.so.5 in SRC.
            lib_path_var = "LD_LIBRARY_PATH"
            cmd = ["auditwheel", "repair", raw_wheel_path, "-w", wheel_directory]

        lib_dirs = [SRC, f"{BUILD_DEPS}/lib"] + ([f"{brew}/lib"] if sys.platform == "darwin" else [])
        env[lib_path_var] = ":".join(lib_dirs + ([env[lib_path_var]] if env.get(lib_path_var) else []))
        subprocess.check_call(cmd, env=env)

    # 4. Return the repaired wheel filename.
    #    auditwheel renames to manylinux_*; delocate keeps the original name.
    repaired = sorted(glob.glob(f"{wheel_directory}/mrcal-*.whl"))
    if not repaired:
        raise RuntimeError(f"wheel repair did not produce a wheel in {wheel_directory}")
    return os.path.basename(repaired[-1])


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "Source distribution building is not supported by this backend. "
        "Use 'dpkg-buildpackage' for Debian source packages."
    )
