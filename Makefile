PROJECT_NAME := mrcal
ABI_VERSION  := 0
TAIL_VERSION := 0

LIB_SOURCES += mrcal.c
BIN_SOURCES += test_gradients.c test/test_cahvor.c


CXXFLAGS_CV := $(shell pkg-config --cflags opencv)
LDLIBS_CV   := $(shell pkg-config --libs   opencv)

# This will become unnecessary in a soon-to-be-released libdogleg
CXXFLAGS_DOGLEG := -I/usr/include/suitesparse

CCXXFLAGS += $(CXXFLAGS_CV) $(CXXFLAGS_DOGLEG)
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
	undistort-image



# Python docstring rules. I construct these from plain ASCII files to handle
# line wrapping
%.docstring.h: %.docstring
	< $^ sed 's/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.docstring.h

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
mrcal_pywrap.o: CFLAGS += -Wno-cast-function-type

mrcal_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

mrcal/optimizer.so: mrcal_pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY2_MODULES := mrcal

all: mrcal/optimizer.so
EXTRA_CLEAN += mrcal/*.so


include /usr/include/mrbuild/Makefile.common
