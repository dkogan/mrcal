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



# I build the python extension module without any setuptools or anything.
# Instead I ask python about the build flags it likes, and build the DSO
# normally using those flags
#
# The python libraries (compiled ones and ones written in python all live in
# mrcal/). So 'import mrcal' pulls in the writte-in-C library, and something
# like 'import mrcal.cahvor' imports a python library. The C library is actually
# called _mrcal, but mrcal/__init__.py pulls that into the mrcal namespace
define PYVARS_SCRIPT :=
import sysconfig
import re

for v in ("CC","CFLAGS","CCSHARED","INCLUDEPY","BLDSHARED","LDFLAGS"):
    print re.sub("[\t ]+", "__whitespace__", "PY_{}:={}".format(v, sysconfig.get_config_var(v)))
endef
PYVARS := $(shell python -c '$(PYVARS_SCRIPT)')
$(foreach v,$(PYVARS),$(eval $(subst __whitespace__, ,$v)))


mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcal_pywrap.o: CFLAGS += $(wordlist 2,$(words $(PY_CC)),$(PY_CC)) $(PY_CFLAGS) $(PY_CCSHARED) -I$(PY_INCLUDEPY) --std=gnu99

mrcal/_mrcal.so: mrcal_pywrap.o libmrcal.so
	$(PY_BLDSHARED) $(PY_LDFLAGS) $< -lmrcal -o $@ -L$(abspath .) -Wl,-rpath=$(abspath .)

# The python libraries (compiled ones and ones written in python all live in
# mrcal/). So 'import mrcal' pulls in the writte-in-C library, and something
# like 'import mrcal.cahvor' imports a python library. The C library is actually
# called _mrcal, but mrcal/__init__.py pulls that into the mrcal namespace

DIST_PY2_MODULES := mrcal
all: mrcal/_mrcal.so






EXTRA_CLEAN += mrcal/*.so


include /usr/include/mrbuild/Makefile.common
