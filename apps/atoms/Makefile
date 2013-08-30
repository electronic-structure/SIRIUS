include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE) atom.cpp $(LIB_SIRIUS) $(LIBS) -o atom

clean:
	rm -f *.o atom *.json run.x
