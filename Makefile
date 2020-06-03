PYTHON_VERSION_FOR_EXTENSIONS := 3

include /usr/include/mrbuild/Makefile.common.header

PROJECT_NAME := mrcal
ABI_VERSION  := 0
TAIL_VERSION := 0

LIB_SOURCES += mrcal.c
BIN_SOURCES += test-gradients.c test/test-cahvor.c test/test-lensmodel-string-manipulation.c

CXXFLAGS_CV := $(shell pkg-config --cflags opencv)
LDLIBS_CV   := $(shell pkg-config --libs   opencv)

CCXXFLAGS += $(CXXFLAGS_CV)
LDLIBS    += $(LDLIBS_CV)

LDLIBS    += -ldogleg

CFLAGS    += --std=gnu99
CCXXFLAGS += -Wno-missing-field-initializers -Wno-unused-variable -Wno-unused-parameter
CCXXFLAGS += -ffast-math -mtune=native

DIST_INCLUDE += basic_points.h mrcal.h
DIST_BIN :=					\
	mrcal-calibrate-cameras			\
	mrcal-convert-distortion		\
	mrcal-show-distortion			\
	mrcal-show-splined-model-surface	\
	mrcal-show-intrinsics-uncertainty	\
	mrcal-show-intrinsics-diff		\
	mrcal-reproject-points			\
	mrcal-reproject-image			\
	mrcal-graft-cameramodel			\
	mrcal-to-cahvor				\
	mrcal-to-cameramodel			\
	mrcal-show-calibration-geometry		\
	mrcal-show-valid-intrinsics-region	\
	mrcal-is-within-valid-intrinsics-region

# generate manpages from distributed binaries, and ship them. This is a hoaky
# hack because apparenly manpages from python tools is a crazy thing to want to
# do
DIST_MAN := $(addsuffix .1,$(DIST_BIN))

# I construct the README.org from the template. The only thing I do is to insert
# the manpages. Note that this is more complicated than it looks:
#
# 1. The documentation lives in a POD
# 2. This documentation is stripped out here with pod2text, and included in the
#    README. This README is an org-mode file, and the README.template.org
#    container included the manpage text inside a #+BEGIN_EXAMPLE/#+END_EXAMPLE.
#    So the manpages are treated as a verbatim, unformatted text blob
# 3. Further down, the same POD is converted to a manpage via pod2man
define MAKE_README =
BEGIN									\
{									\
  for $$a (@ARGV)							\
  {									\
    $$base = $$a =~ s/\.pod$$//r;                                       \
    $$c{$$base} = `pod2text $$a | mawk "/REPOSITORY/{exit} {print}"`;	\
  }									\
}									\
									\
while(<STDIN>)								\
{									\
  print s/xxx-manpage-(.*?)-xxx/$$c{$$1}/gr;				\
}
endef

README.org: README.template.org $(DIST_BIN:%=%.pod)
	< $(filter README%,$^) perl -e '$(MAKE_README)' $(filter-out README%,$^) > $@
all: README.org

# I parse the version from the changelog. This version is generally something
# like 0.04 .I strip leading 0s, so the above becomes 0.4
VERSION_FROM_CHANGELOG = $(shell sed -n 's/.*(\([0-9\.]*[0-9]\).*).*/\1/; s/\.0*/./g; p; q;' debian/changelog)
$(DIST_MAN): %.1: %.pod
	pod2man --center="mrcal: camera projection, calibration toolkit" --name=MRCAL --release="mrcal $(VERSION_FROM_CHANGELOG)" --section=1 $< $@
%.pod: %
	make-pod-from-help.pl $< > $@
	cat footer.pod >> $@
EXTRA_CLEAN += $(DIST_MAN) $(patsubst %.1,%.pod,$(DIST_MAN))

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
mrcal_pywrap.o: CFLAGS += -Wno-cast-function-type

mrcal_pywrap.o: CFLAGS += $(PY_MRBUILD_CFLAGS)
mrcal_pywrap.o: $(addsuffix .h,$(wildcard *.docstring))

mrcal/_mrcal$(PY_EXT_SUFFIX): mrcal_pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := mrcal

all: mrcal/_mrcal$(PY_EXT_SUFFIX)
EXTRA_CLEAN += mrcal/*.so

# Set up the test suite to be runnable in parallel
TESTS :=					\
  test/test-pywrap-functions.py			\
  test/test-pylib-projections.py		\
  test/test-poseutils.py			\
  test/test-cameramodel.py			\
  test/test-projections.py			\
  test/test-projections-stereographic.py	\
  test/test-gradients.py			\
  test/test-py-gradients.py			\
  test/test-cahvor                              \
  test/test-optimizer-callback.py		\
  test/test-basic-sfm.py			\
  test/test-basic-calibration.py		\
  test/test-lensmodel-string-manipulation

TESTS_RUN := $(addsuffix .RUN,$(TESTS))
test check: $(TESTS_RUN)
	@echo "All tests in the test suite passed!"
$(TESTS_RUN): all
.PHONY: test check

%.RUN: %
	@echo "Running test 'make $@'"
	@$<


include /usr/include/mrbuild/Makefile.common.footer
