include choose_mrbuild.mk
include $(MRBUILD_MK)/Makefile.common.header


# "0" or undefined means "false"
# everything else means  "true"

# libelas stereo matcher. Available in Debian/non-free. I don't want to depend
# on anything in non-free, so I default to not using libelas
USE_LIBELAS ?= 0


# convert all USE_XXX:=0 to an empty string
$(foreach v,$(filter USE_%,$(.VARIABLES)),$(if $(filter 0,${$v}),$(eval undefine $v)))
# to print them all: $(foreach v,$(filter USE_%,$(.VARIABLES)),$(warning $v = '${$v}'))


PROJECT_NAME := mrcal
ABI_VERSION  := 5
TAIL_VERSION := 0

VERSION = $(VERSION_FROM_PROJECT)

LIB_SOURCES +=			\
  mrcal.c			\
  opencv.c			\
  uncertainty.c			\
  image.c			\
  stereo.c			\
  poseutils.c			\
  poseutils-opencv.c		\
  poseutils-uses-autodiff.cc	\
  triangulation.cc              \
  cahvore.cc                    \
  traverse-sensor-links.c \
  heap.cc

ifneq (${USE_LIBELAS},) # using libelas
LIB_SOURCES := $(LIB_SOURCES) stereo-matching-libelas.cc
endif


BIN_SOURCES +=					\
  test/test-gradients.c				\
  test/test-cahvor.c				\
  test/test-lensmodel-string-manipulation.c     \
  test/test-parser-cameramodel.c                \
  test/test-heap.c

LDLIBS += -ldogleg -lfreeimage -llapack

ifneq (${USE_LIBELAS},) # using libelas
LDLIBS += -lelas
endif

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter -Wno-missing-braces

$(patsubst %.c,%.o,$(shell grep -l '#include .*minimath\.h' *.c */*.c)): minimath/minimath_generated.h
minimath/minimath_generated.h: minimath/minimath_generate.pl
	./$< > $@.tmp && mv $@.tmp $@
EXTRA_CLEAN += minimath/minimath_generated.h

DIST_INCLUDE +=			\
	mrcal.h			\
	mrcal-image.h		\
	mrcal-internal.h	\
	basic-geometry.h	\
	poseutils.h		\
	triangulation.h		\
	mrcal-types.h		\
	stereo.h                \
	heap.h



DIST_BIN :=					\
	mrcal-calibrate-cameras			\
	mrcal-convert-lensmodel			\
	mrcal-show-distortion-off-pinhole	\
	mrcal-show-splined-model-correction	\
	mrcal-show-projection-uncertainty	\
	mrcal-show-projection-diff		\
	mrcal-reproject-points			\
	mrcal-reproject-image			\
	mrcal-graft-models			\
	mrcal-to-cahvor				\
	mrcal-from-cahvor			\
	mrcal-to-kalibr				\
	mrcal-from-kalibr			\
	mrcal-from-ros				\
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

# if using an older mrbuild SO won't be defined, and we need it
SO ?= so

# parser
cameramodel-parser_GENERATED.c: cameramodel-parser.re mrcal.h
	re2c $< > $@.tmp && mv $@.tmp $@
LIB_SOURCES += cameramodel-parser_GENERATED.c
EXTRA_CLEAN += cameramodel-parser_GENERATED.c
cameramodel-parser_GENERATED.o: CCXXFLAGS += -fno-fast-math

ALL_NPSP_EXTENSION_MODULES := $(patsubst %-genpywrap.py,%,$(wildcard *-genpywrap.py))
ifeq (${USE_LIBELAS},) # not using libelas
ALL_NPSP_EXTENSION_MODULES := $(filter-out elas,$(ALL_NPSP_EXTENSION_MODULES))
endif
ALL_PY_EXTENSION_MODULES   := _mrcal $(patsubst %,_%_npsp,$(ALL_NPSP_EXTENSION_MODULES))
%/:
	mkdir -p $@

######### python stuff
%-npsp-pywrap-GENERATED.c: %-genpywrap.py
	python3 $< > $@.tmp && mv $@.tmp $@
mrcal/_%_npsp$(PY_EXT_SUFFIX): %-npsp-pywrap-GENERATED.o libmrcal.$(SO) libmrcal.$(SO).${ABI_VERSION}
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $< -lmrcal -o $@

ALL_NPSP_C  := $(patsubst %,%-npsp-pywrap-GENERATED.c,$(ALL_NPSP_EXTENSION_MODULES))
ALL_NPSP_O  := $(patsubst %,%-npsp-pywrap-GENERATED.o,$(ALL_NPSP_EXTENSION_MODULES))
ALL_NPSP_SO := $(patsubst %,mrcal/_%_npsp$(PY_EXT_SUFFIX),$(ALL_NPSP_EXTENSION_MODULES))

EXTRA_CLEAN += $(ALL_NPSP_C)

# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95635
$(ALL_NPSP_O): CFLAGS += -Wno-array-bounds

mrcal-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcal/_mrcal$(PY_EXT_SUFFIX): mrcal-pywrap.o libmrcal.$(SO) libmrcal.$(SO).${ABI_VERSION}
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $(LDFLAGS) $< -lmrcal -lsuitesparseconfig -o $@

CFLAGS += -I/usr/include/suitesparse
PYTHON_OBJECTS := mrcal-pywrap.o $(ALL_NPSP_O)

$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := mrcal

all: mrcal/_mrcal$(PY_EXT_SUFFIX) $(ALL_NPSP_SO)
EXTRA_CLEAN += mrcal/*.$(SO)

include Makefile.doc
include Makefile.tests

include $(MRBUILD_MK)/Makefile.common.footer
