PROJECT_NAME := mrcal
ABI_VERSION  := 0
TAIL_VERSION := 0

LIB_SOURCES += cal.c

CXXFLAGS_CV := $(shell pkg-config --cflags opencv)
LDLIBS_CV   := $(shell pkg-config --libs   opencv)

# This will become unnecessary in a soon-to-be-released libdogleg
CXXFLAGS_DOGLEG := -I/usr/include/suitesparse

CCXXFLAGS += $(CXXFLAGS_CV) $(CXXFLAGS_DOGLEG)
LDLIBS    += $(LDLIBS_CV)

LDLIBS    += -ldogleg

CCXXFLAGS += --std=gnu99 -Wno-missing-field-initializers

include /usr/include/mrbuild/Makefile.common
