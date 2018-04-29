include ../make.inc
LIB_SIRIUS = ../src/libsirius.a

PYTHON = python3.6
CXXFLAGS += -shared -undefined dynamic_lookup
CXXFLAGS += `$(PYTHON) -m pybind11 --includes`
CXXEND = `$(PYTHON)-config --extension-suffix`

.PHONY: all
all: py_sirius

py_sirius: py_sirius.cpp
	 $(CXX) $(CXXFLAGS) $(CXX_OPT) -std=c++14 $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@$(CXXEND)

.PHONY: clean
clean:
	rm -f py_sirius$(CXXEND)
