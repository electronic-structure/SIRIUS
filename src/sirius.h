// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file sirius.h
 *
 *  \brief "All-in-one" include file.
 */

#ifndef __SIRIUS_H__
#define __SIRIUS_H__

#if defined(__APEX)
#include <apex_api.hpp>
#endif

#include "utils/json.hpp"
using json = nlohmann::json;

#include "input.hpp"
#include "cmd_args.hpp"
#include "constants.h"
#include "spline.hpp"
#include "radial_solver.hpp"
#include "SHT/sht.hpp"
#include "SHT/gaunt.hpp"
#include "sddk.hpp"
#include "hdf5_tree.hpp"
#include "Band/band.hpp"
#include "dft_ground_state.hpp"

#if defined(__PLASMA)
extern "C" void plasma_init(int num_cores);
#endif

#if defined(__LIBSCI_ACC)
extern "C" void libsci_acc_init();
extern "C" void libsci_acc_finalize();
#endif

/// Namespace of the SIRIUS library.
namespace sirius {

/// Return the status of the library (initialized or not).
inline static bool& is_initialized()
{
    static bool b{false};
    return b;
}

/// Initialize the library.
inline void initialize(bool call_mpi_init__ = true)
{
    if (is_initialized()) {
        TERMINATE("SIRIUS library is already initialized");
    }
    if (call_mpi_init__) {
        Communicator::initialize(MPI_THREAD_MULTIPLE);
    }
    if (Communicator::world().rank() == 0) {
        printf("SIRIUS %i.%i, git hash: %s\n", major_version, minor_version, git_hash);
    }
#if defined(__APEX)
    apex::init("sirius", Communicator::world().rank(), Communicator::world().size());
#endif
    utils::start_global_timer();

#if defined(__GPU)
    if (acc::num_devices()) {
        if (acc::num_devices() > 1) {
            // TODO: this depends on the rank placement
            int dev_id = Communicator::world().rank() % acc::num_devices();
            acc::set_device_id(dev_id);
        }
        acc::create_streams(omp_get_max_threads() + 1);
        cublas::create_stream_handles();
    }
#endif
#if defined(__MAGMA)
    magma::init();
#endif
#if defined(__PLASMA)
    plasma_init(omp_get_max_threads());
#endif
#if defined(__LIBSCI_ACC)
    libsci_acc_init();
#endif
    /* for the fortran interface to blas/lapack */
    assert(sizeof(int) == 4);
    assert(sizeof(double) == 8);

    is_initialized() = true;
}

/// Shut down the library.
inline void finalize(bool call_mpi_fin__ = true, bool reset_device__ = true, bool fftw_cleanup__ = true)
{
    if (!is_initialized()) {
        TERMINATE("SIRIUS library was not initialized");
    }
#if defined(__MAGMA)
    magma::finalize();
#endif
#if defined(__LIBSCI_ACC)
    libsci_acc_finalize();
#endif

    Beta_projectors_base<1>::cleanup();
    Beta_projectors_base<3>::cleanup();
    Beta_projectors_base<9>::cleanup();

#if defined(__GPU)
    if (acc::num_devices()) {
        acc::set_device();
        cublas::destroy_stream_handles();
        acc::destroy_streams();
        if (reset_device__) {
            acc::reset();
        }
    }
#endif
    if (fftw_cleanup__) {
        fftw_cleanup();
    }

    utils::stop_global_timer();
#if defined(__APEX)
    apex::finalize();
#endif
    if (call_mpi_fin__) {
        Communicator::finalize();
    }

    is_initialized() = false;
}

}
#endif // __SIRIUS_H__

/**
\mainpage Welcome to SIRIUS
\section intro Introduction
SIRIUS is a domain-specific library for electronic structure calculations. It supports full-potential linearized
augmented plane wave (FP-LAPW) and pseudopotential plane wave (PP-PW) methods and is designed to work with codes
such as Exciting, Elk and Quantum ESPRESSO.
\section install Installation
First, you need to clone the source code:
\verbatim
git clone https://github.com/electronic-structure/SIRIUS.git
\endverbatim

Then you need to create a configuration \c json file where you specify your compiliers, compiler flags and libraries.
Examples of such configuration files can be found in the <tt>./platforms/</tt> folder. The following variables have to be
provided:
  - \c MPI_CXX -- the MPI wrapper for the C++11 compiler (this is the main compiler for the library)
  - \c MPI_CXX_OPT -- the C++ compiler options for the library
  - \c MPI_FC -- the MPI wrapper for the Fortran compiler (used to build ELPA and SIRIUS F90 interface)
  - \c MPI_FC_OPT -- Fortran compiler options
  - \c CC -- plain C compilers (used to build the external libraries)
  - \c CXX -- plain C++ compiler (used to build the external libraries)
  - \c FC -- plain Fortran compiler (used to build the external libraries)
  - \c FCCPP -- Fortran preprocessor (usually 'cpp', required by LibXC package)
  - \c SYSTEM_LIBS -- list of the libraries, necessary for the linking (typically BLAS/LAPACK/ScaLAPACK, libstdc++ and Fortran run-time)
  - \c install -- list of packages to download, configure and build

In addition, the following variables can also be specified:
  - \c CUDA_ROOT -- path to the CUDA toolkit (if you compile with GPU support)
  - \c NVCC -- name of the CUDA C++ compiler (usually \c nvcc)
  - \c NVCC_OPT -- CUDA compiler options
  - \c MAGMA_ROOT -- location of the compiled MAGMA library (if you compile with MAGMA support)

Below is an example of the configurtaion file for the Cray XC50 platform. Cray compiler wrappers for C/C++/Fortran are
ftn/cc/CC. The GNU compilers and MKL are used.
\verbatim
{
    "comment"     : "MPI C++ compiler and options",
    "MPI_CXX"     : "CC",
    "MPI_CXX_OPT" : "-std=c++11 -Wall -Wconversion -fopenmp -D__SCALAPACK -D__ELPA -D__GPU -D__MAGMA -I$(MKLROOT)/include/fftw/",

    "comment"     : "MPI Fortran compiler and oprions",
    "MPI_FC"      : "ftn",
    "MPI_FC_OPT"  : "-O3 -fopenmp -cpp",

    "comment"     : "plain C compler",
    "CC"          : "cc",

    "comment"     : "plain C++ compiler",
    "CXX"         : "CC",

    "comment"     : "plain Fortran compiler",
    "FC"          : "ftn",

    "comment"     : "Fortran preprocessor",
    "FCCPP"       : "cpp",

    "comment"     : "location of CUDA toolkit",
    "CUDA_ROOT"   : "$(CUDATOOLKIT_HOME)",

    "comment"     : "CUDA compiler and options",
    "NVCC"        : "nvcc",
    "NVCC_OPT"    : "-arch=sm_60 -m64 -DNDEBUG",

    "comment"     : "location of MAGMA library",
    "MAGMA_ROOT"  : "$(HOME)/src/daint/magma-2.2.0",

    "SYSTEM_LIBS" : "$(MKLROOT)/lib/intel64/libmkl_scalapack_lp64.a -Wl,--start-group  $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_gnu_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a $(MKLROOT)/lib/intel64/libmkl_blacs_intelmpi_lp64.a -Wl,--end-group -lpthread -lstdc++ -ldl",

    "install"     : ["spg", "gsl", "xc"]
}
\endverbatim
FFT library is part of the MKL. The corresponding C++ include directory is passed to the compiler: -I\$(MKLROOT)/include/fftw/.
The \c HDF5 library is installed as a module and handeled by the Cray wrappers. The remaining three libraries necessary for
SIRIUS (spglib, GSL, libXC) are not available and have to be installed.

Once the configuration \c json file is created, you can run
\verbatim
python configure.py path/to/config.json
\endverbatim

The Python script will download and configure external packages, specified in the <tt>"install"</tt> list and create
Makefile and make.inc files. That's it! The configuration is done and you can run
\verbatim
make
\endverbatim
*/

//! \page stdvarname Standard variable names
//!
//! Below is the list of standard names for some of the loop variables:
//!
//! l - index of orbital quantum number \n
//! m - index of azimutal quantum nuber \n
//! lm - combined index of (l,m) quantum numbers \n
//! ia - index of atom \n
//! ic - index of atom class \n
//! iat - index of atom type \n
//! ir - index of r-point \n
//! ig - index of G-vector \n
//! idxlo - index of local orbital \n
//! idxrf - index of radial function \n
//! xi - compbined index of lm and idxrf (product of angular and radial functions) \n
//! ik - index of k-point \n
//! itp - index of (theta, phi) spherical angles \n
//!
//! The _loc suffix is often added to the variables to indicate that they represent the local fraction of the elements
//! assigned to the given MPI rank.
//!

//! \page coding Coding style
//!
//! Below are some basic style rules that we follow:
//!     - Page width is approximately 120 characters. Screens are wide nowdays and 80 characters is an
//!       obsolete restriction. Going slightly over 120 characters is allowed if it is requird for the line continuity.
//!     - Indentation: 4 spaces (no tabs)
//!     - Comments are inserted before the code with slash-star style starting with the lower case:
//!       \code{.cpp}
//!           /* call a very important function */
//!           do_something();
//!       \endcode
//!     - Spaces between most operators:
//!       \code{.cpp}
//!           if (i < 5) {
//!               j = 5;
//!           }
//!
//!           for (int k = 0; k < 3; k++)
//!
//!           int lm = l * l + l + m;
//!
//!           double d = std::abs(e);
//!
//!           int k = idx[3];
//!       \endcode
//!     - Spaces between function arguments:
//!       \code{.cpp}
//!           double d = some_func(a, b, c);
//!       \endcode
//!       but not
//!       \code{.cpp}
//!           double d=some_func(a,b,c);
//!       \endcode
//!       or
//!       \code{.cpp}
//!           double d = some_func( a, b, c );
//!       \endcode
//!     - Spaces between template arguments, but not between <> brackets:
//!       \code{.cpp}
//!           std::vector<std::array<int, 2>> vec;
//!       \endcode
//!       but not
//!       \code{.cpp}
//!           std::vector< std::array< int, 2 > > vec;
//!       \endcode
//!     - Curly braces for classes and functions start form the new line:
//!       \code{.cpp}
//!           class A
//!           {
//!               ....
//!           };
//!
//!           inline int num_points()
//!           {
//!               return num_points_;
//!           }
//!       \endcode
//!     - Curly braces for if-statements, for-loops, switch-case statements, etc. start at the end of the line:
//!       \code{.cpp}
//!           for (int i: {0, 1, 2}) {
//!               some_func(i);
//!           }
//!
//!           if (a == 0) {
//!               printf("a is zero");
//!           } else {
//!               printf("a is not zero");
//!           }
//!
//!           switch (i) {
//!               case 1: {
//!                   do_something();
//!                   break;
//!               case 2: {
//!                   do_something_else();
//!                   break;
//!               }
//!           }
//!       \endcode
//!     - Even single line 'if' statements and 'for' loops must have the curly brackes:
//!       \code{.cpp}
//!           if (i == 4) {
//!               some_variable = 5;
//!           }
//!
//!           for (int k = 0; k < 10; k++) {
//!               do_something(k);
//!           }
//!       \endcode
//!     - Reference and pointer symbols are part of type:
//!       \code{.cpp}
//!           std::vector<double>& vec = make_vector();
//!
//!           double* ptr = &vec[0];
//!
//!           auto& atom = unit_cell().atom(ia);
//!       \endcode
//!     - Const modifier follows the type declaration:
//!       \code{.cpp}
//!           std::vector<int> const& idx() const
//!           {
//!               return idx_;
//!           }
//!       \endcode
//!     - Names of class members end with underscore:
//!       \code{.cpp}
//!           class A
//!           {
//!               private:
//!                   int lmax_;
//!           };
//!       \endcode
//!     - Setter method starts from set_, getter method is a variable name itself:
//!       \code{.cpp}
//!           class A
//!           {
//!               private:
//!                   int lmax_;
//!               public:
//!                   int lmax() const
//!                   {
//!                       return lmax_;
//!                   }
//!                   void set_lmax(int lmax__)
//!                   {
//!                       lmax_ = lmax__;
//!                   }
//!           };
//!       \endcode
//!       However, the new style for setter methods is preferable:
//!       \code{.cpp}
//!           class A
//!           {
//!               private:
//!                   int lmax_;
//!               public:
//!                   int lmax() const
//!                   {
//!                       return lmax_;
//!                   }
//!                   int lmax(int lmax__)
//!                   {
//!                       lmax_ = lmax__;
//!                       return lmax_;
//!                   }
//!           };
//!       \endcode
//!     - Single-line functions should not be flattened:
//!       \code{.cpp}
//!           struct A
//!           {
//!               int lmax() const
//!               {
//!                   return lmax_;
//!               }
//!           };
//!       \endcode
//!       but not
//!       \code{.cpp}
//!           struct A
//!           {
//!               int lmax() const { return lmax_; }
//!           };
//!       \endcode
//!     - Header guards have a standard name: double underscore + file name in capital letters + double underscore
//!       \code{.cpp}
//!           #ifndef __SIRIUS_INTERNAL_H__
//!           #define __SIRIUS_INTERNAL_H__
//!           ...
//!           #endif // __SIRIUS_INTERNAL_H__
//!       \endcode
//!     - Variable names are all in lowercase and underscore-separated (aka 'snake_case'):
//!       \code{.cpp}
//!           int num_bands;
//!           std::complex<double> beta_psi;
//!       \endcode
//!       but not
//!       \code{.cpp}
//!           int NumBands;
//!           /* or */
//!           std::complex<double> BetaPsi;
//!           /* or */
//!           std::complex<double> Another_BetaPsi;
//!       \endcode
//!     - Order of class members: private, protected, public
//!       \code{.cpp}
//!           class A
//!           {
//!               private:
//!                   int lmax_;
//!                   void bar();
//!               protected:
//!                   void foo();
//!               public:
//!                   int lmax() const
//!                   {
//!                       return lmax_;
//!                   }
//!           };
//!       \endcode
//!
//! We use clang-format utility to enforce the basic formatting style. Please have a look at .clang-format config file
//! in the source root folder for the definitions and use helper script 'clang_format.x'.
//!
//! Class naming convention.
//!
//! Problem: all 'standard' naming conventions are not satisfactory. For example, we have a class
//! which does a DFT ground state. Following the common naming conventions it could be named like this:
//! DFTGroundState, DftGroundState, dft_ground_state. Last two are bad, because DFT (and not Dft or dft)
//! is a well recognized abbreviation. First one is band because capital G adds to DFT and we automaticaly
//! read DFTG round state.
//!
//! Solution: we can propose the following: DFTgroundState or DFT_ground_state. The first variant still
//! doens't look very good because one of the words is captalized (State) and one (ground) - is not. So we pick
//! the second variant: DFT_ground_state (by the way, this is close to the Bjarne Stroustrup's naiming convention,
//! where he uses first capital letter and underscores, for example class Io_obj).
//!
//! Some other examples:
//!     - class Ground_state (composed of two words)
//!     - class FFT_interface (composed of an abbreviation and a word)
//!     - class Interface_XC (composed of a word and abbreviation)
//!     - class Spline (single word)
//!
//! Exceptions are allowed if it makes sense. For example, low level utility classes like 'mdarray' (multi-dimentional
//! array) or 'pstdout' (parallel standard output) are named with small letters.
//!

/** \page fderiv Functional derivatives

    Definition:
    \f[
      \frac{dF[f+\epsilon \eta ]}{d \epsilon}\Bigg\rvert_{\epsilon = 0} := \int \frac{\delta F[f]}{\delta f(x')} \eta(x') dx'
    \f]
    Alternative definition is:
    \f[
      \frac{\delta F[f(x)]}{\delta f(x')} = \lim_{\epsilon \to 0} \frac{F[f(x) + \epsilon \delta(x-x')] - F[f(x)]}{\epsilon}
    \f]


*/
