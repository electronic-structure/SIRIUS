include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: sirius.scf

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

plot:
	$(CXX) $(CXX_OPT) $(INCLUDE) plot.cpp $(LIB_SIRIUS) $(LIBS) -o plot

clean:
	rm -f *.o dft_loop sirius.scf *.h5 timers.json
