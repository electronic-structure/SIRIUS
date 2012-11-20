include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	python atoms.py
	$(CXX) $(CXX_OPT) $(INCLUDE) atoms.cpp $(LIB_SIRIUS) $(LIBS) -o atoms

clean:
	rm -f *.o atoms atoms.cpp
