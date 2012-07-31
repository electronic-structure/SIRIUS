include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

INCLUDE_PATH = -I./../../lib

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) hydrogen.cpp -o hydrogen
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) read_atom.cpp -o read_atom
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) fft.cpp -o fft

clean:
	rm *.o *.a
