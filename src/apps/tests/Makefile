include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

INCLUDE_PATH = -I./../../lib

CXX_OPT = -O0 -g

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) hydrogen.cpp -o hydrogen
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) read_atom.cpp -o read_atom

clean:
	rm *.o *.a
