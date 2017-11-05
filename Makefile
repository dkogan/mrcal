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

CCXXFLAGS += --std=gnu99 -Wno-missing-field-initializers

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
build/lib.%/mrcal.so: mrcal_pywrap.c $(addsuffix .h,$(wildcard *.docstring)) mrcal.h libmrcal.so
	python setup.py build -f
EXTRA_CLEAN += build
all: build/lib.linux-x86_64-2.7/mrcal.so


test_cahvor.o: CFLAGS += -Wno-unused-variable -Wno-unused-parameter

include /usr/include/mrbuild/Makefile.common
