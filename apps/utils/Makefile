include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: 
	$(CXX) $(CXX_OPT) $(INCLUDE) unit_cell_tools.cpp $(LIB_SIRIUS) $(LIBS) -o unit_cell_tools

clean:
	rm -f *.o unit_cell_tools *.h5 timers.json
