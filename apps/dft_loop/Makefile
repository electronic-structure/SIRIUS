include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

LIB_SIRIUS_DBG = ../../src/libsirius.dbg.a

INCLUDE = -I./../../src

.SUFFIXES: .cpp

all: sirius.scf

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) $(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

debug: sirius.scf.cpp $(LIB_SIRIUS_DBG)
	$(CXX) $(CXX_OPT_DBG) $(INCLUDE) sirius.scf.cpp $(LIB_SIRIUS_DBG) $(LIBS) -o sirius.scf



clean:
	rm -rf *.o sirius.scf *.h5 *dSYM
