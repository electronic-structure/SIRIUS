include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(LINK_OPT) $(LIB_SIRIUS) hydrogen.cpp -o hydrogen
	$(CXX) $(CXX_OPT) $(LINK_OPT) $(LIB_SIRIUS) read_atom.cpp -o read_atom

clean:
	rm *.o *.a
