include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE) splindex.cpp  $(LIB_SIRIUS) $(LIBS) -o splindex
	$(CXX) $(CXX_OPT) $(INCLUDE) hdf5.cpp      $(LIB_SIRIUS) $(LIBS) -o hdf5
	$(CXX) $(CXX_OPT) $(INCLUDE) hydrogen.cpp  $(LIB_SIRIUS) $(LIBS) -o hydrogen
	$(CXX) $(CXX_OPT) $(INCLUDE) read_atom.cpp $(LIB_SIRIUS) $(LIBS) -o read_atom
	$(CXX) $(CXX_OPT) $(INCLUDE) fft.cpp       $(LIB_SIRIUS) $(LIBS) -o fft
	$(CXX) $(CXX_OPT) $(INCLUDE) fft1k.cpp     $(LIB_SIRIUS) $(LIBS) -o fft1k
	$(CXX) $(CXX_OPT) $(INCLUDE) spline.cpp    $(LIB_SIRIUS) $(LIBS) -o spline

clean:
	rm -f *.o *.h5 splindex hdf5 hydrogen read_atom fft fft1k spline
