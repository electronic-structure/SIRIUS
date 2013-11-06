include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
#	$(CXX) $(CXX_OPT) $(INCLUDE) davidson.cpp $(LIB_SIRIUS) $(LIBS) -o davidson
	$(CXX) $(CXX_OPT) $(INCLUDE) lobpcg.cpp $(LIB_SIRIUS) $(LIBS) -o lobpcg

clean:
	rm -f *.o davidson
