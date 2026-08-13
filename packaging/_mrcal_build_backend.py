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
import sys
import zipfile
import tempfile
import subprocess

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Version / metadata
# ---------------------------------------------------------------------------

def _version():
    """Upstream version from debian/changelog, e.g. '2.5.2'."""
    with open(f"{SRC}/debian/changelog") as f:
        first_line = f.readline()
    m = re.match(r"^\S+\s+\((\d+\.\d+(?:\.\d+)?)", first_line)
    return m.group(1) if m else "0.0.0"


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


def _build_raw_wheel(raw_wheel_path, version):
    pkg_dir   = f"{SRC}/mrcal"
    impl      = sys.implementation.name
    ver       = "".join(str(v) for v in sys.version_info[:2])
    ext_tag   = f"{impl}-{ver}"
    dist_info = f"mrcal-{version}.dist-info"
    data_dir  = f"mrcal-{version}.data"
    records   = []

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


def get_requires_for_build_sdist(config_settings=None):
    return []


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

    # 1. Build (clean first to avoid stale objects from a prior host build)
    subprocess.check_call(["make", "clean"], cwd=SRC)
    # USE_LOCAL_STB_IMPLEMENTATION: compile stb into libmrcal rather than
    # linking against an external libstb.so (not available on all platforms).
    make_cmd = ["make", f"-j{ncpus}"]
    if sys.platform != "darwin":
        make_cmd.append("USE_LOCAL_STB_IMPLEMENTATION=1")

    # numpy headers are installed into the isolated build venv but make runs
    # outside it; add numpy's include dir via C_INCLUDE_PATH so GCC finds
    # numpy/arrayobject.h without any Makefile changes.
    import numpy
    env = os.environ.copy()
    numpy_inc = numpy.get_include()
    env["C_INCLUDE_PATH"] = numpy_inc + (":" + env["C_INCLUDE_PATH"] if env.get("C_INCLUDE_PATH") else "")

    # On macOS, /usr/include is SIP-protected so mrbuild is installed under the
    # Homebrew prefix.  choose_mrbuild.mk only checks mrbuild/ (local) or
    # /usr/include/mrbuild/; create a temporary local symlink so make finds it.
    mrbuild_symlink = None
    if sys.platform == "darwin":
        local_link = f"{SRC}/mrbuild"
        if not os.path.exists(local_link):
            brew = subprocess.check_output(["brew", "--prefix"], text=True).strip()
            candidate = f"{brew}/include/mrbuild"
            if os.path.isdir(candidate):
                os.symlink(candidate, local_link)
                mrbuild_symlink = local_link

    try:
        subprocess.check_call(make_cmd, cwd=SRC, env=env)
    finally:
        if mrbuild_symlink and os.path.islink(mrbuild_symlink):
            os.unlink(mrbuild_symlink)

    with tempfile.TemporaryDirectory(prefix="mrcal-raw-wheel-") as tmp:
        tag            = _raw_wheel_tag()
        raw_wheel_path = f"{tmp}/mrcal-{version}-{tag}.whl"

        # 2. Pack raw wheel (extensions keep their $ORIGIN/.. RUNPATH for now;
        #    auditwheel will rewrite everything)
        _build_raw_wheel(raw_wheel_path, version)

        # 3. Repair: bundle libmrcal + all transitive C deps into the wheel
        env = os.environ.copy()

        if sys.platform == "darwin":
            # Point DYLD_LIBRARY_PATH at the source tree so delocate's otool
            # resolution can find libmrcal (built there by make).
            lib_path_var = "DYLD_LIBRARY_PATH"
            cmd = ["delocate-wheel", "-w", wheel_directory, raw_wheel_path]
        else:
            # Extensions use RUNPATH (not RPATH), so LD_LIBRARY_PATH takes
            # precedence and auditwheel's ldd finds libmrcal.so.5 in SRC.
            lib_path_var = "LD_LIBRARY_PATH"
            cmd = ["auditwheel", "repair", raw_wheel_path, "-w", wheel_directory]

        env[lib_path_var] = SRC + (":" + env[lib_path_var] if env.get(lib_path_var) else "")
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
