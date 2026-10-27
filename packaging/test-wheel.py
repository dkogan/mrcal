#!/usr/bin/env python3

"""Smoke-test an installed mrcal wheel.

This runs from cibuildwheel after installation. It deliberately changes away
from the source checkout and verifies compiled APIs, serialization, image I/O,
CLI installation and the bundled gnuplot resources.
"""

import importlib.metadata
import os
import platform
import shutil
import subprocess
import tempfile

import numpy as np
import mrcal


def run(*args, input_text=None):
    return subprocess.run(
        args,
        input=input_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout


with tempfile.TemporaryDirectory() as workdir:
    os.chdir(workdir)

    distribution = importlib.metadata.distribution("mrcal")
    assert distribution.version
    assert "site-packages" in mrcal.__file__

    expected_arch = os.environ.get("CIBW_ARCHS_MACOS")
    if expected_arch:
        assert platform.machine() == expected_arch

    lensmodel = "LENSMODEL_PINHOLE"
    intrinsics = np.array((800.0, 810.0, 320.0, 240.0))
    points = np.array(
        (
            (0.1, -0.2, 2.0),
            (-0.3, 0.4, 3.0),
            (0.0, 0.0, 1.5),
        )
    )

    q = mrcal.project(points, lensmodel, intrinsics)
    rays = mrcal.unproject(q, lensmodel, intrinsics, normalize=True)
    expected = points / np.linalg.norm(points, axis=-1)[..., None]
    np.testing.assert_allclose(rays, expected, atol=1e-12, rtol=0.0)

    model0 = mrcal.cameramodel(
        intrinsics=(lensmodel, intrinsics),
        imagersize=(640, 480),
    )
    model1 = mrcal.cameramodel(
        intrinsics=(lensmodel, intrinsics),
        imagersize=(640, 480),
        rt_cam_ref=np.array((0.0, 0.0, 0.0, -0.25, 0.0, 0.0)),
    )

    point_ref = np.array((0.15, -0.10, 4.0))
    q0 = mrcal.project(point_ref, *model0.intrinsics())
    q1 = mrcal.project(
        mrcal.transform_point_rt(model1.rt_cam_ref(), point_ref),
        *model1.intrinsics(),
    )
    point_observed = mrcal.triangulate(
        np.array((q0, q1)),
        (model0, model1),
    )
    np.testing.assert_allclose(
        point_observed,
        point_ref,
        atol=1e-10,
        rtol=0.0,
    )

    model_path = os.path.join(workdir, "camera.cameramodel")
    model0.write(model_path)
    restored_model = mrcal.cameramodel(model_path)
    np.testing.assert_allclose(
        restored_model.intrinsics()[1],
        intrinsics,
    )

    image = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    image_path = os.path.join(workdir, "image.png")
    mrcal.save_image(image_path, image)
    np.testing.assert_array_equal(mrcal.load_image(image_path), image)

    for command in (
        "mrcal-calibrate-cameras",
        "mrcal-convert-lensmodel",
        "mrcal-triangulate",
        "mrcal-report-fov",
    ):
        assert shutil.which(command), command
        run(command, "--help")

    gnuplot = shutil.which("gnuplot")
    assert gnuplot

    plot = run(
        gnuplot,
        input_text="set terminal dumb; unset key; plot sin(x)\n",
    )
    assert len(plot.splitlines()) >= 10
    assert "*" in plot

    help_output = run(
        gnuplot,
        input_text="help plot\nexit\n",
    )
    assert "primary commands for drawing plots" in help_output

    version_output = run(gnuplot, "-e", "show version long")
    vendor_root = os.path.join(
        os.path.dirname(mrcal.__file__),
        "_vendor",
    )
    assert os.path.join(
        vendor_root,
        "share",
        "gnuplot",
        "gnuplot.gih",
    ) in version_output
    assert os.path.join(
        vendor_root,
        "share",
        "gnuplot",
        "PostScript",
    ) in version_output

print("Installed mrcal wheel smoke test passed")
