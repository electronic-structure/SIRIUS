include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
#	$(CXX) $(CXX_OPT) $(INCLUDE) block_davidson.cpp $(LIB_SIRIUS) $(LIBS) -o block_davidson
	$(CXX) $(CXX_OPT) $(INCLUDE) -c block_davidson.cpp -o block_davidson.o
	$(CXX) $(CXX_OPT) $(INCLUDE) -fopenmp block_davidson.o $(LIB_SIRIUS) $(LIBS) -o block_davidson
#	$(CXX) $(CXX_OPT) $(INCLUDE) lobpcg.cpp $(LIB_SIRIUS) $(LIBS) -o lobpcg

clean:
	rm -f *.o davidson
