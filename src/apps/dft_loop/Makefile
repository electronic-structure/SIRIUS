include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

INCLUDE_PATH = -I./../../lib

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) dft_loop.cpp $(LIB_SIRIUS) $(LIBS) -o dft_loop

clean:
	rm *.o *.a
