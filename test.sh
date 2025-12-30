#!/usr/bin/env zsh

GREEN="\x1b[32m"
RED="\x1b[31m"
COLOR_RESET="\x1b[0m"


# The test suite no longer runs in parallel, but it ALWAYS tries to run all the
# tests, even without 'make -k'
TESTS=("test/test-pywrap-functions.py"
       "test/test-pylib-projections.py"
       "test/test-cameramodel.py"
       "test/test-poseutils.py"
       "test/test-poseutils-lib.py"
       "test/test-poseutils-near-singularity.py"
       "test/test-projections.py"
       "test/test-projection--special-cases.py pinhole"
       "test/test-projection--special-cases.py stereographic"
       "test/test-projection--special-cases.py lonlat"
       "test/test-projection--special-cases.py latlon"
       "test/test-gradients.py"
       "test/test-py-gradients.py"
       "test/test-cahvor"
       "test/test-optimizer-callback.py"
       "test/test-sfm-fixed-points.py"
       "test/test-sfm-regularization-unity-cam01.py"
       "test/test-sfm-triangulated-points.py"
       "test/test-basic-calibration.py"
       "test/test-surveyed-calibration.py"
       "test/test-surveyed-calibration.py --distance 8"
       "test/test-surveyed-calibration.py --oversample 10"
       "test/test-surveyed-calibration.py --do-sample"
       "test/test-surveyed-calibration.py --do-sample --distance 8"
       "test/test-surveyed-calibration.py --do-sample --oversample 10"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --do-sample"
       "test/test-projection-uncertainty.py --fixed frames --model opencv4 --do-sample --Nsamples 200"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --do-sample --observed-pixel-uncertainty 0.3 --reproject-perturbed cross-reprojection-rrp-Jfp --Nsamples 200"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --do-sample --observed-pixel-uncertainty 0.03 --reproject-perturbed cross-reprojection-rrp-Jfp --points"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --do-sample --reproject-perturbed bestq --observed-pixel-uncertainty 0.3 --Nsamples 200"
       "test/test-projection-uncertainty.py --fixed cam0 --model splined"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --Ncameras 1 --compare-baseline-against-mrcal-2.4"
       "test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --Ncameras 4 --compare-baseline-against-mrcal-2.4"
       "test/test-linearizations.py"
       "test/test-lensmodel-string-manipulation"
       "test/test-CHOLMOD-factorization.py"
       "test/test-projection-diff.py"
       "test/test-graft-models.py"
       "test/test-convert-lensmodel.py"
       "test/test-stereo.py"
       "test/test-solvepnp.py"
       "test/test-match-feature.py"
       "test/test-triangulation.py"
       "test/test-uncertainty-broadcasting.py --q-calibration-stdev 0.3 --q-observation-stdev 0.3 --q-observation-stdev-correlation 0.5"
       "test/test-parser-cameramodel"
       "test/test-extrinsics-moved-since-calibration.py"
       "test/test-broadcasting.py"
       "test/test-compute-chessboard-corners-parsing.py"
       "test/test-rectification-maps.py"
       "test/test-save-load-image.py"
       "test/test-worst_direction_stdev.py"
       "test/test-propagate-calibration-uncertainty.py"
       "test/test-heap"
       "test/test-traverse-sensor-links.py"
       "test/test-sorted-eig.py"
       "test/test-python-cameramodel-converter.py")

# Check the non-canonical problem definitions
TESTS+=("test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --Ncameras 1 --reproject-perturbed cross-reprojection-rrp-Jfp"
	"test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --Ncameras 1 --range-to-boards 4 --moving-camera"
	"test/test-projection-uncertainty.py --fixed cam0 --model opencv4 --Ncameras 1 --range-to-boards 4 --moving-camera --reproject-perturbed bestq")

# triangulation-uncertainty tests. Lots and lots of tests to exhaustively try
# out different scenarios
BASE="test/test-triangulation-uncertainty.py --do-sample --model opencv4 --observed-point -40 0 200 --Ncameras 3 --cameras 2 1 --q-observation-stdev-correlation 0.6"
TESTS_TRIANGULATION_UNCERTAINTY=("$BASE "{--stabilize-coords,}" --fixed "{cam0,frames}" --q-"{observation,calibration}"-stdev 0.5")

# Both sets of noise. Looking at different camera
BASE="test/test-triangulation-uncertainty.py --do-sample --model opencv4 --Ncameras 2 --cameras 1 0 --q-observation-stdev 0.5 --q-calibration-stdev 0.5"
# Different amounts of correlation and near/far
TESTS_TRIANGULATION_UNCERTAINTY+=("$BASE "{--stabilize-coords,}" --fixed "{cam0,frames}" --q-observation-stdev-correlation "{0.1,0.9}" --observed-point "{"-2 0 10","-40 0 200"})

TESTS_external_data=("test/test-stereo-range.py")


# TESTS_TRIANGULATION_UNCERTAINTY not included in TESTS yet, so TESTS_NOSAMPLING
# includes none of those
TESTS_NOSAMPLING=(${TESTS:#*--do-sample*})
TESTS+=($TESTS_TRIANGULATION_UNCERTAINTY)


# TESTS_external_data is not included in any other set, and must be requested
# explicitly

# Define the set for each "make test-xxx" command
TESTS_all=($TESTS)
TESTS_nosampling=($TESTS_NOSAMPLING)
TESTS_triangulation_uncertainty=($TESTS_TRIANGULATION_UNCERTAINTY)


####### Everything defined. Parse the commandline and do the thing
# "test-all" is the full set of tests
# "test-nosampling" excludes the very time-consuming tests
# "test-external-data" is everything that needs data other than what's in the repo
if ((1 != $#)) {
  echo "Usage: $0 TEST-TYPE\n  Need exactly one argument" > /dev/stderr
  exit 1
}

# downcase the argument, convert - -> _, remove leading "test-"
test_type=${${${(L)1}#test-}//-/_}
TESTS_name="TESTS_$test_type"
TESTS_selected=("${(@P)TESTS_name}")

Ntests=$#TESTS_selected
echo "Running $TESTS_name: $Ntests tests"


FAILED=()
for i ({1..$#TESTS_selected}) {
  t=$TESTS_selected[i];

  echo "========== Test $i/$Ntests: $t"
  ${=t} || FAILED+=($t)
}

if (( $#FAILED )) {
  echo -n $RED
  echo "$#FAILED TEST SETS FAILED:"
  for t ($FAILED) { echo "  $t" }
  echo -n $COLOR_RESET
  exit 1

} else {
  echo -n $GREEN
  echo "ALL TEST SETS PASSED!"
  echo -n $COLOR_RESET
  exit 0
}
