include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

INCLUDE_PATH = -I./../../lib

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) hdf5.cpp -o hdf5
#	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) hydrogen.cpp -o hydrogen
#	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) read_atom.cpp -o read_atom
#	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) fft.cpp -o fft
#	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) fft1k.cpp -o fft1k
#	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) spline.cpp -o spline

clean:
	rm *.o *.a
