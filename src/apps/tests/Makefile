include ../../../make.inc

LIB_SIRIUS = ../../lib/libsirius.a

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(LINK_OPT) $(LIB_SIRIUS) hydrogen.cpp -o hydrogen

clean:
	rm *.o *.a
