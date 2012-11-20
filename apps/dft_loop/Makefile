include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE) dft_loop.cpp $(LIB_SIRIUS) $(LIBS) -o dft_loop

clean:
	rm -f *.o dft_loop *.h5 timers.json
