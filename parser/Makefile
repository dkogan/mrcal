LDLIBS +=

# if any -O... is requested, use that; otherwise, do -O3
FLAGS_OPTIMIZATION := $(if $(filter -O%,$(CFLAGS) $(CXXFLAGS) $(CPPFLAGS)),,-O3 -ffast-math)
CPPFLAGS := -MMD -g $(FLAGS_OPTIMIZATION) -Wall -Wextra -Wno-missing-field-initializers
CFLAGS += -std=gnu11 -fno-omit-frame-pointer
CXXFLAGS += -std=gnu++11

all: cameramodel

%.yy.c: %.l
	flex -o $@ $<

%.tab.c %.tab.h: %.y
	bison -d $<

cameramodel.yy.o cameramodel.tab.o: CFLAGS += -Wno-unused-function -Wno-unused-parameter

%: %.yy.o %.tab.o
	gcc -o $@ $^

cameramodel.yy.o: cameramodel.tab.h

clean:
	rm -rf *.o *.d *.tab.* *.yy.* cameramodel
.PHONY: clean

-include *.d
