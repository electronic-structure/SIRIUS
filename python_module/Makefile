include /Users/colinkalin/my_SIRIUS/make.inc #how to fix this, must not contain absolute path...
LIB_SIRIUS = /Users/colinkalin/my_SIRIUS/src/libsirius.a #dito


CXX = mpic++
CXXFLAGS = -std=c++11
CXXFLAGS += -Wall -shared
CXXFLAGS += -O3 -undefined dynamic_lookup
CXXFLAGS += `python3 -m pybind11 --includes`
CXXEND = `python3-config --extension-suffix`


.PHONY: all
all: sirius

sirius: py_sirius.cpp
	 $(CXX) $(CXXFLAGS) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@$(CXXEND)

.PHONY: clean
clean:
	rm -f sirius
