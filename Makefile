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

# I build the python extension module without any setuptools or anything.
# Instead I ask python about the build flags it likes, and build the DSO
# normally using those flags.
#
# There's some sillyness in Make I need to work around. First, I produce a
# python script to query the various build flags, but replacing all whitespace
# with __whitespace__. The string I get when running this script will have a
# number of whitespace-separated tokens, each one setting ONE variable
define PYVARS_SCRIPT
import sysconfig
import re

for v in ("CC","CFLAGS","CCSHARED","INCLUDEPY","BLDSHARED","LDFLAGS"):
    print re.sub("[\t ]+", "__whitespace__", "PY_{}:={}".format(v, sysconfig.get_config_var(v)))
endef
PYVARS := $(shell python -c '$(PYVARS_SCRIPT)')

# I then $(eval) these tokens one at a time, restoring the whitespace
$(foreach v,$(PYVARS),$(eval $(subst __whitespace__, ,$v)))

# The compilation flags are all the stuff python told us about. Some of its
# flags live inside its CC variable, so I pull those out. I also pull out the
# optimization flag, since I want THIS build system to control it
mrcal_pywrap.o: CFLAGS += $(filter-out -O%,$(wordlist 2,$(words $(PY_CC)),$(PY_CC)) $(PY_CFLAGS) $(PY_CCSHARED) -I$(PY_INCLUDEPY) --std=gnu99)
mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

# I add an RPATH to the python extension DSO so that it runs in-tree. mrbuild
# will pull it out at install time
mrcal/optimizer.so: mrcal_pywrap.o libmrcal.so
	$(PY_BLDSHARED) $(PY_LDFLAGS) $< -lmrcal -o $@ -L$(abspath .) -Wl,-rpath=$(abspath .)

# mrcal/ is a python2 module
DIST_PY2_MODULES := mrcal

all: mrcal/optimizer.so
EXTRA_CLEAN += mrcal/*.so


include /usr/include/mrbuild/Makefile.common
