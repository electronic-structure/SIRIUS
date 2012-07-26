CXX = g++-mp-4.4

CPP_OPT = 

CXX_OPT = -O0 -g -Wall $(CPP_OPT)

LINK_OPT = -lstdc++ $(LIBS)

LIBS = $(LAPACK_LIB) $(XC_LIB) $(SPG_LIB)

LAPACK_LIB = -llapack -lblas

# libxc
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/libxc-1.2.0/src
XC_LIB = /Users/anton/src/libxc-1.2.0/src/.libs/libxc.a 

# spglib
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/sirius-project/spglib-1.2.2/src
SPG_LIB = /Users/anton/src/sirius-project/spglib-1.2.2/src/.libs/libsymspg.a
SPG_LIB_PATH = /Users/anton/src/sirius-project/spglib-1.2.2/src/.libs
