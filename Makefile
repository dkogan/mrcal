PROJECT_NAME := mrcal
ABI_VERSION  := 0
TAIL_VERSION := 0

LIB_SOURCES += mrcal.c
BIN_SOURCES += test_gradients.c test_cahvor.c


CXXFLAGS_CV := $(shell pkg-config --cflags opencv)
LDLIBS_CV   := $(shell pkg-config --libs   opencv)

# This will become unnecessary in a soon-to-be-released libdogleg
CXXFLAGS_DOGLEG := -I/usr/include/suitesparse

CCXXFLAGS += $(CXXFLAGS_CV) $(CXXFLAGS_DOGLEG)
LDLIBS    += $(LDLIBS_CV)

LDLIBS    += -ldogleg

CCXXFLAGS += --std=gnu99 -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter

DIST_INCLUDE    += basic_points.h mrcal.h
DIST_BIN := visualize_extrinsics.py



# Python docstring rules. I construct these from plain ASCII files to handle
# line wrapping
%.docstring.h: %.docstring
	< $^ sed 's/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.docstring.h

# The python libraries (compiled ones and ones written in python all live in
# mrcal/).

# The compilation flags are all the stuff python told us about. Some of its
# flags live inside its CC variable, so I pull those out. I also pull out the
# optimization flag, since I want THIS build system to control it
mrcal_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

mrcal/optimizer.so: mrcal_pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

# mrcal/ is a python2 module
DIST_PY2_MODULES := mrcal

all: mrcal/optimizer.so
EXTRA_CLEAN += mrcal/*.so


include /usr/include/mrbuild/Makefile.common
