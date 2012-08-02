CXX = g++-mp-4.4

CPP_OPT = 

CXX_OPT = -O3 -Wall $(CPP_OPT)

LINK_OPT = $(LIBS) -lstdc++

LIBS = $(LAPACK_LIB) $(XC_LIB) $(SPG_LIB) $(GSL_LIB) $(FFTW_LIB)

LAPACK_LIB = -llapack -lblas

# libxc
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/libxc-1.2.0/src
XC_LIB = /Users/anton/src/libxc-1.2.0/src/.libs/libxc.a 

# spglib
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/sirius-project/spglib-1.2.2/src
SPG_LIB = /Users/anton/src/sirius-project/spglib-1.2.2/src/.libs/libsymspg.a
SPG_LIB_PATH = /Users/anton/src/sirius-project/spglib-1.2.2/src/.libs

# GNU scientific library
CXX_OPT := $(CXX_OPT) -I/Users/anton/src/sirius-project/gsl-1.15
GSL_LIB = /Users/anton/src/sirius-project/gsl-1.15/.libs/libgsl.a

# FFTW
CXX_OPT := $(CXX_OPT) -D_FFTW_ 
FFTW_LIB = /opt/local/lib/libfftw3.a
