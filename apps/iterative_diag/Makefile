include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
#	$(CXX) $(CXX_OPT) $(INCLUDE) -c iterative_diag.cpp -o iterative_diag.o
#	$(CXX) $(CXX_OPT) $(INCLUDE) -fopenmp iterative_diag.o $(LIB_SIRIUS) $(LIBS) -o iterative_diag
	$(CXX) $(CXX_OPT) $(INCLUDE) lobpcg.cpp $(LIB_SIRIUS) $(LIBS) -o lobpcg
	

clean:
	rm -f *.o iterative_diag
