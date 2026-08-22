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



def _build_raw_wheel(raw_wheel_path, version, brew=None):
    pkg_dir   = f"{SRC}/mrcal"
    impl      = sys.implementation.name
    ver       = "".join(str(v) for v in sys.version_info[:2])
    ext_tag   = f"{impl}-{ver}"
    dist_info = f"mrcal-{version}.dist-info"
    data_dir  = f"mrcal-{version}.data"
    records   = []

    def _find_tool(name):
        # shutil.which uses the current process PATH, which doesn't include
        # BUILD_DEPS/bin.  Check there explicitly before falling back to brew.
        if found := shutil.which(name):
            return found
        p = os.path.join(BUILD_DEPS, "bin", name)
        if os.path.isfile(p):
            return p
        return os.path.join(brew, "bin", name) if brew else None

    gnuplot_bin = _find_tool("gnuplot")
    mawk_bin    = _find_tool("mawk")

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
            add(zf, open(path, "rb").read(), arcname)

        # Bundled gnuplot + gnuplot_x11.
        # gnuplot wrapper sets GNUPLOT_DRIVER_DIR to its own directory (venv/bin/)
        # so gnuplot finds gnuplot_x11 there at runtime.
        if gnuplot_bin:
            gnuplot_prefix = os.path.dirname(os.path.dirname(gnuplot_bin))

            add(zf, open(gnuplot_bin, "rb").read(), "mrcal/_vendor/bin/gnuplot", mode=0o755)
            wrapper = (
                '#!python\n'
                'import os, sys, importlib.util\n'
                '_s = importlib.util.find_spec("mrcal")\n'
                '_real = os.path.join(os.path.dirname(_s.origin), "_vendor", "bin", "gnuplot")\n'
                'os.environ["GNUPLOT_DRIVER_DIR"] = os.path.dirname(os.path.abspath(sys.argv[0]))\n'
                'os.execv(_real, sys.argv)\n'
            )
            add(zf, wrapper, f"{data_dir}/scripts/gnuplot", mode=0o755)

            # gnuplot_x11: find in libexec and place alongside gnuplot in venv/bin/
            x11_candidates = glob.glob(f"{gnuplot_prefix}/libexec/gnuplot/*/gnuplot_x11")
            if x11_candidates:
                x11_bin = x11_candidates[0]
                add(zf, open(x11_bin, "rb").read(), "mrcal/_vendor/bin/gnuplot_x11", mode=0o755)
                wrapper_x11 = (
                    '#!python\n'
                    'import os, sys, importlib.util\n'
                    '_s = importlib.util.find_spec("mrcal")\n'
                    '_real = os.path.join(os.path.dirname(_s.origin), "_vendor", "bin", "gnuplot_x11")\n'
                    'os.execv(_real, sys.argv)\n'
                )
                add(zf, wrapper_x11, f"{data_dir}/scripts/gnuplot_x11", mode=0o755)

        # Bundled mawk — needed by vnl-* tools at runtime.
        if mawk_bin and os.path.isfile(mawk_bin):
            add(zf, open(mawk_bin, "rb").read(), "mrcal/_vendor/bin/mawk", mode=0o755)
            wrapper = (
                '#!python\n'
                'import os, sys, importlib.util\n'
                '_s = importlib.util.find_spec("mrcal")\n'
                '_real = os.path.join(os.path.dirname(_s.origin), "_vendor", "bin", "mawk")\n'
                'os.execv(_real, sys.argv)\n'
            )
            add(zf, wrapper, f"{data_dir}/scripts/mawk", mode=0o755)

        # Bundled vnlog — scripts (vnl-*) and Perl modules.
        # Scripts get exec-wrappers in data/scripts/ so pip installs them to
        # venv/bin/. Wrappers set PERL5LIB so the Perl scripts find their modules.
        _vnlog_bin_dir = BUILD_DEPS + '/bin'
        _vnlog_perl_dir = BUILD_DEPS + '/lib/perl5'
        if os.path.isdir(_vnlog_perl_dir):
            for path in sorted(glob.glob(f'{_vnlog_perl_dir}/**', recursive=True)):
                if not os.path.isfile(path): continue
                rel = os.path.relpath(path, _vnlog_perl_dir)
                add(zf, open(path, 'rb').read(),
                    f'mrcal/_vendor/lib/perl5/{rel}', mode=0o644)
        if os.path.isdir(_vnlog_bin_dir):
            for path in sorted(glob.glob(f'{_vnlog_bin_dir}/vnl-*')):
                if not os.path.isfile(path): continue
                name = os.path.basename(path)
                add(zf, open(path, 'rb').read(),
                    f'mrcal/_vendor/bin/{name}', mode=0o755)
                wrapper = (
                    f'#!python\n'
                    f'import os, sys, importlib.util\n'
                    f'_s = importlib.util.find_spec("mrcal")\n'
                    f'_v = os.path.join(os.path.dirname(_s.origin), "_vendor")\n'
                    f'_pl = os.path.join(_v, "lib", "perl5")\n'
                    f'if os.path.isdir(_pl):\n'
                    f'    os.environ["PERL5LIB"] = _pl + (":" + os.environ["PERL5LIB"] if os.environ.get("PERL5LIB") else "")\n'
                    f'os.execv(os.path.join(_v, "bin", "{name}"), sys.argv)\n'
                )
                add(zf, wrapper, f'{data_dir}/scripts/{name}', mode=0o755)

        # Python extensions from GL_image_display, installed by
        # before-build.sh into the cibuildwheel Python's site-packages under
        # BUILD_DEPS.  The path is recorded in py3-modules-path because the
        # build backend runs in a different isolated venv.
        _py3_path_file = BUILD_DEPS + '/py3-modules-path'
        if os.path.exists(_py3_path_file):
            _py_dir = BUILD_DEPS + open(_py3_path_file).read().strip()
            if os.path.isdir(_py_dir):
                for path in sorted(glob.glob(f'{_py_dir}/**', recursive=True)):
                    if not os.path.isfile(path): continue
                    if '.dist-info' in path: continue
                    rel = os.path.relpath(path, _py_dir)
                    parts = rel.split(os.sep)
                    if 'test' in parts: continue
                    if path.endswith(('.cpp', '.h', '.c')): continue
                    add(zf, open(path, 'rb').read(), rel,
                        mode=os.stat(path).st_mode & 0o777)

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
    env["PATH"]         = os.path.dirname(sys.executable) + ":" + BUILD_DEPS + "/bin" + (":" + env["PATH"] if env.get("PATH") else "")


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

    # USE_LOCAL_STB_IMPLEMENTATION: compile stb into libmrcal rather than
    # linking against an external libstb.so (not available on all platforms).
    make_cmd = ["make", f"-j{ncpus}"]
    if sys.platform != "darwin":
        make_cmd.append("USE_LOCAL_STB_IMPLEMENTATION=1")

    subprocess.check_call(["make", "clean"], cwd=SRC, env=env)
    subprocess.check_call(make_cmd, cwd=SRC, env=env)



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
