include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: dft_loop plot

dft_loop: 
	$(CXX) $(CXX_OPT) $(INCLUDE) dft_loop.cpp $(LIB_SIRIUS) $(LIBS) -o dft_loop

plot:
	$(CXX) $(CXX_OPT) $(INCLUDE) plot.cpp $(LIB_SIRIUS) $(LIBS) -o plot

clean:
	rm -f *.o dft_loop *.h5 timers.json
