PYTHON_VERSION_FOR_EXTENSIONS := 3

include mrbuild/Makefile.common.header

PROJECT_NAME := mrcal
ABI_VERSION  := 1
TAIL_VERSION := 0

# Custom version from git (or from debian/changelog if no git repo available)
_VERSION = $(shell test -d .git && \
  git describe --tags | sed 's/^.*\///' || \
  < debian/changelog sed -n 's/.*(\([0-9\.]*[0-9]\).*).*/\1/; p; q;')
# Memoize. $(VERSION) will evaluate the result the first time, and use the
# cached result during subsequent calls
VERSION = $(if $(_VERSION_EXPANDED),,$(eval _VERSION_EXPANDED:=$$(_VERSION)))$(_VERSION_EXPANDED)

LIB_SOURCES += mrcal.c poseutils.c poseutils-uses-autodiff.cc triangulation.cc

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
	mrcal-show-residuals

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

ALL_PY_EXTENSION_MODULES := _mrcal _mrcal_npsp _poseutils
%/:
	mkdir -p $@

## mrcal-python-api-reference.html contains everything. It is large
DOC_HTML += doc/out/mrcal-python-api-reference.html
doc/out/mrcal-python-api-reference.html: $(wildcard mrcal/*.py) $(patsubst %,mrcal/%$(PY_EXT_SUFFIX),$(ALL_PY_EXTENSION_MODULES)) libmrcal.so.$(ABI_VERSION) | doc/out/
	python3 doc/pydoc.py -w mrcal > $@.tmp && mv $@.tmp $@

DOC_ALL_FIG          := $(wildcard doc/*.fig)
DOC_ALL_SVG_FROM_FIG := $(patsubst doc/%.fig,doc/out/figures/%.svg,$(DOC_ALL_FIG))
DOC_ALL_PDF_FROM_FIG := $(patsubst doc/%.fig,doc/out/figures/%.pdf,$(DOC_ALL_FIG))
doc: $(DOC_ALL_SVG_FROM_FIG) $(DOC_ALL_PDF_FROM_FIG)
$(DOC_ALL_SVG_FROM_FIG): doc/out/figures/%.svg: doc/%.fig | doc/out/figures/
	fig2dev -L svg $< $@
$(DOC_ALL_PDF_FROM_FIG): doc/out/figures/%.pdf: doc/%.fig | doc/out/figures/
	fig2dev -L pdf $< $@

## Each submodule in a separate .html. This works, but needs more effort:
##
## - top level mrcal.html is confused about what it contains. It has all of
##   _mrcal and _poseutils for some reason
## - cross-submodule links don't work
#
# doc-reference: \
# 	$(patsubst mrcal/%.py,doc/mrcal.%.html,$(filter-out %/__init__.py,$(wildcard mrcal/*.py))) \
# 	$(patsubst %,doc/out/mrcal.%.html,$(ALL_PY_EXTENSION_MODULES)) \
# 	doc/out/mrcal.html
# doc/out/mrcal.%.html: \
# 	mrcal/%.py \
# 	$(patsubst %,mrcal/%$(PY_EXT_SUFFIX),$(ALL_PY_EXTENSION_MODULES)) \
# 	libmrcal.so.$(ABI_VERSION)
# 	doc/pydoc.py -w mrcal.$* > $@.tmp && mv $@.tmp $@
# doc/out/mrcal.%.html: mrcal/%$(PY_EXT_SUFFIX)
# 	doc/pydoc.py -w mrcal.$* > $@.tmp && mv $@.tmp $@
# doc/out/mrcal.html: \
# 	$(wildcard mrcal/*.py) \
# 	$(patsubst %,mrcal/%$(PY_EXT_SUFFIX),$(ALL_PY_EXTENSION_MODULES)) \
# 	libmrcal.so.$(ABI_VERSION)
# 	doc/pydoc.py -w mrcal > $@.tmp && mv $@.tmp $@
# .PHONY: doc-reference



DOC_ALL_CSS        := $(wildcard doc/*.css)
DOC_ALL_CSS_TARGET := $(patsubst doc/%,doc/out/%,$(DOC_ALL_CSS))
doc: $(DOC_ALL_CSS_TARGET)
$(DOC_ALL_CSS_TARGET): doc/out/%.css: doc/%.css | doc/out/
	cp $< doc/out

DOC_ALL_ORG         := $(wildcard doc/*.org)
DOC_ALL_HTML_TARGET := $(patsubst doc/%.org,doc/out/%.html,$(DOC_ALL_ORG))
DOC_HTML += $(DOC_ALL_HTML_TARGET)

# This ONE command creates ALL the html files, so I want a pattern rule to indicate
# that. I want to do:
#   %/out/a.html %/out/b.html %/out/c.html: %/a.org %/b.org %/c.org
$(addprefix %,$(patsubst doc/%,/%,$(DOC_ALL_HTML_TARGET))): $(addprefix %,$(patsubst doc/%,/%,$(DOC_ALL_ORG)))
	emacs --chdir=doc -l mrcal-docs-publish.el --batch --eval '(load-library "org")' --eval '(org-publish-all t nil)'
$(DOC_ALL_HTML_TARGET): doc/mrcal-docs-publish.el | doc/out/


$(DIST_MAN): %.1: %.pod
	pod2man --center="mrcal: camera projection, calibration toolkit" --name=MRCAL --release="mrcal $(VERSION)" --section=1 $< $@
%.pod: %
	mrbuild/make-pod-from-help.pl $< > $@.tmp && cat footer.pod >> $@.tmp && mv $@.tmp $@
EXTRA_CLEAN += $(DIST_MAN) $(patsubst %.1,%.pod,$(DIST_MAN))

# I generate a manpage. Some perl stuff to add the html preamble
MANPAGES_HTML := $(patsubst %,doc/out/%.html,$(DIST_BIN))
doc/out/%.html: %.pod | doc/out/
	pod2html --noindex --css=mrcal.css --infile=$< | \
	  perl -ne 'BEGIN {$$h = `cat doc/mrcal-preamble-GENERATED.html`;} if(!/(.*<body>)(.*)/s) { print; } else { print "$$1 $$h $$2"; }' > $@.tmp && mv $@.tmp $@

DOC_HTML += $(MANPAGES_HTML)

$(DOC_HTML): doc/mrcal-preamble-GENERATED.html
# If the git HEAD moves, I regenerate the preamble. It contains a version string
# that uses the git info. This still isn't complete. A new git tag SHOULD
# trigger this to be regenerated, but it doesn't. I'll do that later
doc/mrcal-preamble-GENERATED.html: doc/mrcal-preamble-TEMPLATE.html $(and $(wildcard .git),.git/$(shell cut -d' ' -f2 .git/HEAD))
	< $< sed s/@@VERSION@@/$(VERSION)/g > $@.tmp && mv $@.tmp $@
EXTRA_CLEAN += doc/mrcal-preamble-GENERATED.html

doc: $(DOC_HTML)

.PHONY: doc

# the whole output documentation directory
EXTRA_CLEAN += doc/out






######### python stuff
mrcal-npsp-pywrap-GENERATED.c: mrcal-genpywrap.py
	python3 $< > $@.tmp && mv $@.tmp $@
poseutils-pywrap-GENERATED.c: poseutils-genpywrap.py
	python3 $< > $@.tmp && mv $@.tmp $@
mrcal/_mrcal_npsp$(PY_EXT_SUFFIX): mrcal-npsp-pywrap-GENERATED.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@
mrcal/_poseutils$(PY_EXT_SUFFIX): poseutils-pywrap-GENERATED.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@
EXTRA_CLEAN += mrcal-npsp-pywrap-GENERATED.c poseutils-pywrap-GENERATED.c

# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95635
poseutils-pywrap-GENERATED.o:         CFLAGS += -Wno-array-bounds
mrcal-npsp-pywrap-GENERATED.o: CFLAGS += -Wno-array-bounds

mrcal-pywrap.o: $(addsuffix .h,$(wildcard *.docstring))
mrcal/_mrcal$(PY_EXT_SUFFIX): mrcal-pywrap.o libmrcal.so
	$(PY_MRBUILD_LINKER) $(PY_MRBUILD_LDFLAGS) $< -lmrcal -o $@

PYTHON_OBJECTS := mrcal-npsp-pywrap-GENERATED.o poseutils-pywrap-GENERATED.o mrcal-pywrap.o

# In the python api I have to cast a PyCFunctionWithKeywords to a PyCFunction,
# and the compiler complains. But that's how Python does it! So I tell the
# compiler to chill
$(PYTHON_OBJECTS): CFLAGS += -Wno-cast-function-type
$(PYTHON_OBJECTS): CFLAGS += $(PY_MRBUILD_CFLAGS)

# The python libraries (compiled ones and ones written in python) all live in
# mrcal/
DIST_PY3_MODULES := mrcal

all: mrcal/_mrcal$(PY_EXT_SUFFIX) mrcal/_mrcal_npsp$(PY_EXT_SUFFIX) mrcal/_poseutils$(PY_EXT_SUFFIX)
EXTRA_CLEAN += mrcal/*.so

include Makefile.tests

include mrbuild/Makefile.common.footer
