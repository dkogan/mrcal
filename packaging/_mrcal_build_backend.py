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
import hashlib
import os
import re
import sys
import zipfile
import tempfile
import subprocess
from pathlib import Path

SRC = Path(__file__).parent


# ---------------------------------------------------------------------------
# Version / metadata
# ---------------------------------------------------------------------------

def _version():
    """Upstream version from debian/changelog, e.g. '2.5.2'."""
    with open(SRC / "debian" / "changelog") as f:
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

def _platform_tag():
    import sysconfig
    return sysconfig.get_platform().replace("-", "_").replace(".", "_")


def _python_tag():
    ver = "".join(str(v) for v in sys.version_info[:2])
    # PEP 425 abbreviations: cpython→cp, pypy→pp
    abbrev = {"cpython": "cp", "pypy": "pp", "ironpython": "ip", "jython": "jy"}
    prefix = abbrev.get(sys.implementation.name, sys.implementation.name)
    return f"{prefix}{ver}"                  # e.g. "cp313"


def _ext_tag():
    """Tag embedded in .so filenames, e.g. 'cpython-313'."""
    impl = sys.implementation.name
    ver  = "".join(str(v) for v in sys.version_info[:2])
    return f"{impl}-{ver}"


def _raw_wheel_tag():
    py = _python_tag()
    return f"{py}-{py}-{_platform_tag()}"


def _raw_wheel_name(version):
    return f"mrcal-{version}-{_raw_wheel_tag()}.whl"


def _sha256_of(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _wheel_header(version):
    py, abi, plat = _raw_wheel_tag().split("-")
    return (
        f"Wheel-Version: 1.0\n"
        f"Generator: _mrcal_build_backend\n"
        f"Root-Is-Purelib: false\n"
        f"Tag: {py}-{abi}-{plat}\n"
    )


def _build_raw_wheel(raw_wheel_path: Path, version: str):
    pkg_dir  = SRC / "mrcal"
    ext_tag  = _ext_tag()
    dist_info = f"mrcal-{version}.dist-info"
    data_dir  = f"mrcal-{version}.data"
    records   = []

    def add(zf, data: bytes, arcname: str):
        if isinstance(data, str):
            data = data.encode()
        zf.writestr(arcname, data)
        records.append((arcname, _sha256_of(data), len(data)))

    with zipfile.ZipFile(raw_wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:

        # Package files from the build tree
        for path in sorted(pkg_dir.rglob("*")):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            # Only include .so files for the current Python version
            if path.suffix == ".so" and ext_tag not in path.stem:
                continue
            arcname = "mrcal/" + str(path.relative_to(pkg_dir))
            add(zf, path.read_bytes(), arcname)

        # CLI scripts from the source root
        for script in sorted(SRC.glob("mrcal-*")):
            if script.is_file() and os.access(script, os.X_OK):
                add(zf, script.read_bytes(), f"{data_dir}/scripts/{script.name}")

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
    dist_info = Path(metadata_directory) / f"mrcal-{version}.dist-info"
    dist_info.mkdir(parents=True, exist_ok=True)
    (dist_info / "METADATA").write_text(_metadata_text(version))
    (dist_info / "WHEEL").write_text(_wheel_header(version))
    return dist_info.name


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
    existing = env.get("C_INCLUDE_PATH", "")
    env["C_INCLUDE_PATH"] = numpy_inc + (":" + existing if existing else "")

    # On macOS, /usr/include is SIP-protected so mrbuild is installed under the
    # Homebrew prefix.  choose_mrbuild.mk only checks mrbuild/ (local) or
    # /usr/include/mrbuild/; create a temporary local symlink so make finds it.
    mrbuild_symlink = None
    if sys.platform == "darwin":
        local_link = SRC / "mrbuild"
        if not local_link.exists():
            brew = subprocess.check_output(["brew", "--prefix"], text=True).strip()
            candidate = Path(brew) / "include" / "mrbuild"
            if candidate.is_dir():
                local_link.symlink_to(candidate)
                mrbuild_symlink = local_link

    try:
        subprocess.check_call(make_cmd, cwd=SRC, env=env)
    finally:
        if mrbuild_symlink and mrbuild_symlink.is_symlink():
            mrbuild_symlink.unlink()

    with tempfile.TemporaryDirectory(prefix="mrcal-raw-wheel-") as tmp:
        raw_wheel_path = Path(tmp) / _raw_wheel_name(version)

        # 2. Pack raw wheel (extensions keep their $ORIGIN/.. RUNPATH for now;
        #    auditwheel will rewrite everything)
        _build_raw_wheel(raw_wheel_path, version)

        # 3. Repair: bundle libmrcal + all transitive C deps into the wheel
        env = os.environ.copy()

        if sys.platform == "darwin":
            # Point DYLD_LIBRARY_PATH at the source tree so delocate's otool
            # resolution can find libmrcal (built there by make).
            lib_path_var = "DYLD_LIBRARY_PATH"
            cmd = ["delocate-wheel", "-w", wheel_directory, str(raw_wheel_path)]
        else:
            # Extensions use RUNPATH (not RPATH), so LD_LIBRARY_PATH takes
            # precedence and auditwheel's ldd finds libmrcal.so.5 in SRC.
            lib_path_var = "LD_LIBRARY_PATH"
            cmd = ["auditwheel", "repair", str(raw_wheel_path), "-w", wheel_directory]

        env[lib_path_var] = (
            str(SRC) + (":" + env[lib_path_var] if env.get(lib_path_var) else "")
        )
        subprocess.check_call(cmd, env=env)

    # 4. Return the repaired wheel filename.
    #    auditwheel renames to manylinux_*; delocate keeps the original name.
    repaired = sorted(Path(wheel_directory).glob("mrcal-*.whl"))
    if not repaired:
        raise RuntimeError("wheel repair did not produce a wheel in " + wheel_directory)
    return repaired[-1].name


def build_sdist(sdist_directory, config_settings=None):
    raise NotImplementedError(
        "Source distribution building is not supported by this backend. "
        "Use 'dpkg-buildpackage' for Debian source packages."
    )
