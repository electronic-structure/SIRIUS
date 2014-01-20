include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE) -c iterative_diag.cpp -o iterative_diag.o
	$(CXX) $(CXX_OPT) $(INCLUDE) -fopenmp iterative_diag.o $(LIB_SIRIUS) $(LIBS) -o iterative_diag

clean:
	rm -f *.o iterative_diag
