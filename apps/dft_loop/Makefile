include ../../make.inc

LIB_SIRIUS = ../../src/libsirius.a

LIB_SIRIUS_DBG = ../../src/libsirius.a

.SUFFIXES: .cpp

all: sirius.scf

%: %.cpp $(LIB_SIRIUS)
	$(CXX) $(CXX_OPT) -I$(INCLUDE) $< $(LIB_SIRIUS) $(LIBS) -o $@

debug: sirius.scf.cpp $(LIB_SIRIUS_DBG)
	$(CXX) $(CXX_OPT_DBG) -I$(INCLUDE) sirius.scf.cpp $(LIB_SIRIUS_DBG) $(LIBS) -o sirius.scf



clean:
	rm -rf *.o sirius.scf *.h5 *dSYM
