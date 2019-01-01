PROJECT_NAME := mrcal
ABI_VERSION  := 0
TAIL_VERSION := 0

LIB_SOURCES += mrcal.c
BIN_SOURCES += test_gradients.c test/test_cahvor.c


CXXFLAGS_CV := $(shell pkg-config --cflags opencv)
LDLIBS_CV   := $(shell pkg-config --libs   opencv)

CCXXFLAGS += $(CXXFLAGS_CV)
LDLIBS    += $(LDLIBS_CV)

LDLIBS    += -ldogleg

CCXXFLAGS += --std=gnu99 -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter

DIST_INCLUDE    += basic_points.h mrcal.h
DIST_BIN :=					\
	calibrate-cameras			\
	convert-distortion			\
	visualize-distortion			\
	visualize-intrinsics-uncertainty        \
	visualize-intrinsics-diff               \
	redistort-points			\
	undistort-image                         \
	graft-cameramodel

# generate manpages from distributed binaries, and ship them. This is a hoaky
# hack because apparenly manpages from python tools is a crazy thing to want to
# do
DIST_MAN := $(addsuffix .1,$(DIST_BIN))

# I parse the version from the changelog. This version is generally something
# like 0.04 .I strip leading 0s, so the above becomes 0.4
VERSION_FROM_CHANGELOG = $(shell sed -n 's/.*(\([0-9\.]*[0-9]\).*).*/\1/; s/\.0*/./g; p; q;' debian/changelog)
$(DIST_MAN): %.1: %.pod
	pod2man --center="mrcal: camera projection, calibration toolkit" --name=MRCAL --release="mrcal $(VERSION_FROM_CHANGELOG)" --section=1 $< $@
%.pod: %
	./make-pod.pl $< > $@
	cat footer.pod >> $@
EXTRA_CLEAN += $(DIST_MAN) $(patsubst %.1,%.pod,$(DIST_MAN))

# Python docstring rules. I construct these from plain ASCII files to handle
# line wrapping
%.docstring.h: %.docstring
	< $^ sed 's/"/\\"/g; s/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.docstring.h

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
mrcal_pywrap.o: CFLAGS += -Wno-cast-function-type

mrcal_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

mrcal/_mrcal.so: mrcal_pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY2_MODULES := mrcal

all: mrcal/_mrcal.so
EXTRA_CLEAN += mrcal/*.so


include /usr/include/mrbuild/Makefile.common
