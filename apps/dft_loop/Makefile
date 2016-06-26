include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

LIB_SIRIUS_DBG = ../../src/libsirius.dbg.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: dft_loop

dft_loop: dft_loop.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) dft_loop.cpp $(LIB_SIRIUS) $(LIBS) -o dft_loop

plot:
	$(CXX) $(CXX_OPT) $(INCLUDE) plot.cpp $(LIB_SIRIUS) $(LIBS) -o plot

debug: dft_loop.cpp $(LIB_SIRIUS_DBG)
	$(CXX) $(CXX_OPT_DBG) $(INCLUDE) dft_loop.cpp $(LIB_SIRIUS_DBG) $(LIBS) -o debug/dft_loop



debug:

clean:
	rm -f *.o dft_loop *.h5 timers.json
