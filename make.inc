CXX = g++-mp-4.4

CPP_OPT = 

CXX_OPT = -O2 -Wall $(CPP_OPT)

LINK_OPT = -lstdc++ $(LIBS)

LIBS = $(LAPACK_LIB) $(XC_LIB)

LAPACK_LIB = -llapack -lblas

# LibXC is required
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/libxc-1.2.0/src
XC_LIB = /Users/anton/src/libxc-1.2.0/src/.libs/libxc.a 