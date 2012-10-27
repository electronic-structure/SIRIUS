include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

INCLUDE_PATH = -I./../../lib

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE_PATH) $(LINK_OPT) $(LIB_SIRIUS) $(LIBS) dft_loop.cpp -o dft_loop

clean:
	rm *.o *.a
