CXX = g++-mp-4.4

CPP_OPT = 

CXX_OPT = -O2 $(CPP_OPT)

LINK_OPT = -lstdc++ $(LIBS)

LIBS = $(LAPACK_LIB)

LAPACK_LIB = -llapack -lblas
