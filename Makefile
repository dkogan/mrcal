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

DIST_INCLUDE += basic_points.h



# Python docstring rules. I construct these from plain ASCII files to handle
# line wrapping
%.docstring.h: %.docstring
	< $^ sed 's/^/"/; s/$$/\\n"/;' > $@
EXTRA_CLEAN += *.docstring.h

# The python extension library is handled by its own little build system. This
# is stupid, but that's how these people did it. Oh, and since this is
# effectively a recursive build, the proper dependency information doesn't make
# it into the inner (python-specific) Makefile, so I "build -f" to forcefully
# rebuild everything. Like I said, this is stupid.
build/lib.%/_mrcal.so: mrcal_pywrap.c $(addsuffix .h,$(wildcard *.docstring)) mrcal.h libmrcal.so
	CFLAGS='$(CPPFLAGS)' python setup.py build -f

# The python libraries (compiled ones and ones written in python all live in
# mrcal/). So 'import mrcal' pulls in the writte-in-C library, and something
# like 'import mrcal.cahvor' imports a python library. The C library is actually
# called _mrcal, but mrcal/__init__.py pulls that into the mrcal namespace
mrcal/_mrcal.so: build/lib.linux-x86_64-2.7/_mrcal.so
	ln -fs ../$< $@

all: mrcal/_mrcal.so

EXTRA_CLEAN += build mrcal/_mrcal.so



include /usr/include/mrbuild/Makefile.common
