include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

.SUFFIXES: .cpp

all: 
	python atoms.py
	$(CXX) $(CXX_OPT) $(LINK_OPT) $(LIB_SIRIUS) atoms.cpp -o atoms

clean:
	rm *.o *.a
