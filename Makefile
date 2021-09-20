PYTHON_VERSION_FOR_EXTENSIONS := 3

include mrbuild/Makefile.common.header

PROJECT_NAME := mrcal
ABI_VERSION  := 2
TAIL_VERSION := 0

# Custom version from git (or from debian/changelog if no git repo available)
_VERSION = $(shell test -d .git && \
  git describe --tags | sed 's/^.*\///' || \
  < debian/changelog sed -n 's/.*(\([0-9\.]*[0-9]\).*).*/\1/; p; q;')
# Memoize. $(VERSION) will evaluate the result the first time, and use the
# cached result during subsequent calls
VERSION = $(if $(_VERSION_EXPANDED),,$(eval _VERSION_EXPANDED:=$$(_VERSION)))$(_VERSION_EXPANDED)

LIB_SOURCES +=			\
  mrcal.c			\
  mrcal-opencv.c		\
  poseutils.c			\
  poseutils-opencv.c		\
  poseutils-uses-autodiff.cc	\
  triangulation.cc

BIN_SOURCES +=					\
  test-gradients.c				\
  test/test-cahvor.c				\
  test/test-lensmodel-string-manipulation.c     \
  test/test-parser-cameramodel.c

LDLIBS    += -ldogleg

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter
CCXXFLAGS += -ffast-math -mtune=native

mrcal.o test/test-cahvor.o: minimath/minimath_generated.h
minimath/minimath_generated.h: minimath/minimath_generate.pl
	./$< > $@.tmp && mv $@.tmp $@
EXTRA_CLEAN += minimath/minimath_generated.h

DIST_INCLUDE += \
	mrcal.h \
	mrcal_internal.h \
	basic_geometry.h \
	poseutils.h \
	triangulation.h
DIST_BIN :=					\
	mrcal-calibrate-cameras			\
	mrcal-convert-lensmodel			\
	mrcal-show-distortion-off-pinhole	\
	mrcal-show-splined-model-surface	\
	mrcal-show-projection-uncertainty	\
	mrcal-show-projection-diff		\
	mrcal-reproject-points			\
	mrcal-reproject-image			\
	mrcal-graft-models			\
	mrcal-to-cahvor				\
	mrcal-to-cameramodel			\
	mrcal-show-geometry			\
	mrcal-show-valid-intrinsics-region	\
	mrcal-is-within-valid-intrinsics-region \
	mrcal-triangulate			\
	mrcal-cull-corners                      \
	mrcal-show-residuals-board-observation  \
	mrcal-show-residuals                    \
	mrcal-stereo

# generate manpages from distributed binaries, and ship them. This is a hoaky
# hack because apparenly manpages from python tools is a crazy thing to want to
# do
DIST_MAN := $(addsuffix .1,$(DIST_BIN))

# parser
cameramodel-parser_GENERATED.c: cameramodel-parser.re mrcal.h
	re2c $< > $@.tmp && mv $@.tmp $@
LIB_SOURCES += cameramodel-parser_GENERATED.c
EXTRA_CLEAN += cameramodel-parser_GENERATED.c
cameramodel-parser_GENERATED.o: CCXXFLAGS += -fno-fast-math

ALL_NPSP_EXTENSION_MODULES := $(patsubst %-genpywrap.py,%,$(wildcard *-genpywrap.py))
ALL_PY_EXTENSION_MODULES   := _mrcal $(patsubst %,_%_npsp,$(ALL_NPSP_EXTENSION_MODULES))
%/:
	mkdir -p $@

######### python stuff
%-npsp-pywrap-GENERATED.c: %-genpywrap.py
	python3 $< > $@.tmp && mv $@.tmp $@
mrcal/_%_npsp$(PY_EXT_SUFFIX): %-npsp-pywrap-GENERATED.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

ALL_NPSP_C  := $(patsubst %,%-npsp-pywrap-GENERATED.c,$(ALL_NPSP_EXTENSION_MODULES))
ALL_NPSP_O  := $(patsubst %,%-npsp-pywrap-GENERATED.o,$(ALL_NPSP_EXTENSION_MODULES))
ALL_NPSP_SO := $(patsubst %,mrcal/_%_npsp$(PY_EXT_SUFFIX),$(ALL_NPSP_EXTENSION_MODULES))

EXTRA_CLEAN += $(ALL_NPSP_C)

# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95635
$(ALL_NPSP_O): CFLAGS += -Wno-array-bounds

mrcal-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcal/_mrcal$(PY_EXT_SUFFIX): mrcal-pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

PYTHON_OBJECTS := mrcal-pywrap.o $(ALL_NPSP_O)

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
$(PYTHON_OBJECTS): CFLAGS += -Wno-cast-function-type
$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := mrcal

all: mrcal/_mrcal$(PY_EXT_SUFFIX) $(ALL_NPSP_SO)
EXTRA_CLEAN += mrcal/*.so

include Makefile.doc
include Makefile.tests

include mrbuild/Makefile.common.footer
