// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

#include "utils/cmd_args.hpp"
#include "utils/json.hpp"
#include "utils/profiler.hpp"
using json = nlohmann::json;

#include "input.hpp"
#include "simulation_context.hpp"
#include "Hamiltonian/local_operator.hpp"
#include "radial_solver.hpp"
#include "SHT/sht.hpp"
#include "SHT/gaunt.hpp"
#include "hdf5_tree.hpp"
#include "Band/band.hpp"
#include "dft_ground_state.hpp"
#include "sirius_version.hpp"

#if defined(__PLASMA)
extern "C" void plasma_init(int num_cores);
#endif

#if defined(__LIBSCI_ACC)
extern "C" void libsci_acc_init();
extern "C" void libsci_acc_finalize();
#endif

/// Namespace of the SIRIUS library.
namespace sirius {

json sirius_options_parser_;

/// Return the status of the library (initialized or not).
inline static bool& is_initialized()
{
    static bool b{false};
    return b;
}

/// Initialize the library.
inline void initialize(bool call_mpi_init__ = true)
{
    PROFILE_START("sirius");
    PROFILE("sirius::initialize");
    if (is_initialized()) {
        TERMINATE("SIRIUS library is already initialized");
    }
    if (call_mpi_init__) {
        Communicator::initialize(MPI_THREAD_MULTIPLE);
    }
#if defined(__APEX)
    apex::init("sirius", Communicator::world().rank(), Communicator::world().size());
#endif
    //utils::start_global_timer();

    if (Communicator::world().rank() == 0) {
        std::printf("SIRIUS %i.%i.%i, git hash: %s\n", sirius::major_version(), sirius::minor_version(),
               sirius::revision(), sirius::git_hash().c_str());
#if !defined(NDEBUG)
        std::printf("Warning! Compiled in 'debug' mode with assert statements enabled!\n");
#endif
    }
    /* get number of ranks per node during the global call to sirius::initialize() */
    sddk::num_ranks_per_node();
    if (acc::num_devices() > 0) {
        int devid = sddk::get_device_id(acc::num_devices());
        acc::set_device_id(devid);
        /* create extensive amount of streams */
        /* some parts of the code rely on the number of streams not related to the
           number of OMP threads */
        acc::create_streams(omp_get_max_threads() + 100);
#if defined(__GPU)
        gpublas::create_stream_handles();
#endif
#if defined(__CUDA)
        cublas::xt::create_handle();
        cusolver::create_handle();
#endif
    }
#if defined(__MAGMA)
    magma::init();
#endif
#if defined(__PLASMA)
    plasma_init(omp_get_max_threads());
#endif
#if defined(__LIBSCI_ACC)
    libsci_acc_init();
#endif
#if defined(__ELPA)
    if (elpa_init(20170403) != ELPA_OK) {
        TERMINATE("ELPA API version not supported");
    }
#endif
    /* for the fortran interface to blas/lapack */
    assert(sizeof(int) == 4);
    assert(sizeof(double) == 8);

    is_initialized() = true;
}

/// Shut down the library.
inline void finalize(bool call_mpi_fin__ = true, bool reset_device__ = true, bool fftw_cleanup__ = true)
{
    PROFILE_START("sirius::finalize");
    if (!is_initialized()) {
        TERMINATE("SIRIUS library was not initialized");
    }
#if defined(__MAGMA)
    magma::finalize();
#endif
#if defined(__LIBSCI_ACC)
    libsci_acc_finalize();
#endif

    if (acc::num_devices()) {
#if defined(__GPU)
        gpublas::destroy_stream_handles();
#endif
#if defined(__CUDA)
        cusolver::destroy_handle();
        cublas::xt::destroy_handle();
#endif
        acc::destroy_streams();
        if (reset_device__) {
            acc::reset();
        }
    }

    //utils::stop_global_timer();
#if defined(__APEX)
    apex::finalize();
#endif
    if (call_mpi_fin__) {
        Communicator::finalize();
    }
#if defined(__ELPA)
    int ierr;
    elpa_uninit(&ierr);
#endif

    is_initialized() = false;

    PROFILE_STOP("sirius::finalize");
    PROFILE_STOP("sirius");
}

}
#endif // __SIRIUS_H__

/** \mainpage Welcome to SIRIUS

SIRIUS is a domain-specific library for electronic structure calculations. It supports full-potential linearized
augmented plane wave (FP-LAPW) and pseudopotential plane wave (PP-PW) methods with ultrasoft, norm-conserving and PAW
flavors of pseudopotential and is designed to work with codes such as Exciting, Elk and Quantum ESPRESSO.

*/

/**
\page stdvarname Standard variable names

Below is the list of standard names for some of the loop variables:

l - index of orbital quantum number \n
m - index of azimutal quantum nuber \n
lm - combined index of (l,m) quantum numbers \n
ia - index of atom \n
ic - index of atom class \n
iat - index of atom type \n
ir - index of r-point \n
ig - index of G-vector \n
idxlo - index of local orbital \n
idxrf - index of radial function \n
xi - combined index of lm and idxrf (product of angular and radial functions) \n
ik - index of k-point \n
itp - index of (theta, phi) spherical angles \n

The _loc suffix is often added to the variables to indicate that they represent the local fraction of the elements
assigned to a given MPI rank.
*/

/**
\page coding Coding style

Below are some basic style rules that we follow:
  - Page width is approximately 120 characters. Screens are wide nowdays and 80 characters is an
    obsolete restriction. Going slightly over 120 characters is allowed if it is requird for the line continuity.
  - Indentation: 4 spaces (no tabs)

    Exception: class access modifiers are idented with 2 spaces
    \code{.cpp}
    class A
    {
      private:
        int n_;
      public:
        A();
    };
    \endcode

  - Comments are inserted before the code with slash-star style starting with the lower case:
    \code{.cpp}
    // call a very important function
    do_something();
    \endcode
  - Spaces between most operators:
    \code{.cpp}
    if (i < 5) {
        j = 5;
    }

    for (int k = 0; k < 3; k++)

    int lm = l * l + l + m;

    double d = std::abs(e);

    int k = idx[3];
    \endcode
  - Spaces between function arguments:
    \code{.cpp}
    double d = some_func(a, b, c);
    \endcode
    but not
    \code{.cpp}
    double d=some_func(a,b,c);
    \endcode
    and not
    \code{.cpp}
    double d = some_func( a, b, c );
    \endcode
  - Spaces between template arguments, but not between <,> brackets:
    \code{.cpp}
    std::vector<std::array<int, 2>> vec;
    \endcode
    but not
    \code{.cpp}
    std::vector< std::array< int, 2 > > vec;
    \endcode
  - Curly braces for classes and functions start form the new line:
    \code{.cpp}
    class A
    {
        ....
    };

    inline int num_points()
    {
        return num_points_;
    }
    \endcode
  - Curly braces for if-statements, for-loops, switch-case statements, etc. start at the end of the line:
    \code{.cpp}
    for (int i: {0, 1, 2}) {
        some_func(i);
    }

    if (a == 0) {
        std::printf("a is zero");
    } else {
        std::printf("a is not zero");
    }

    switch (i) {
        case 1: {
            do_something();
            break;
        case 2: {
            do_something_else();
            break;
        }
    }
    \endcode
  - Even single line 'if' statements and 'for' loops must have the curly brackes:
    \code{.cpp}
    if (i == 4) {
        some_variable = 5;
    }

    for (int k = 0; k < 10; k++) {
        do_something(k);
    }
    \endcode
  - Reference and pointer symbols are part of type:
    \code{.cpp}
    std::vector<double>& vec = make_vector();

    double* ptr = &vec[0];

    auto& atom = unit_cell().atom(ia);
    \endcode
  - Const modifier follows the type declaration:
    \code{.cpp}
    std::vector<int> const& idx() const
    {
        return idx_;
    }
    // or
    auto const& atom = unit_cell().atom(ia);
    \endcode
  - Names of class members end with underscore:
    \code{.cpp}
    class A
    {
      private:
        int lmax_;
    };
    \endcode
  - Setter method starts from set_, getter method is a variable name itself:
    \code{.cpp}
    class A
    {
      private:
        int lmax_;
      public:
        int lmax() const
        {
            return lmax_;
        }
        void set_lmax(int lmax__)
        {
            lmax_ = lmax__;
        }
    };
    \endcode
    However, the new style for setter methods is preferable:
    \code{.cpp}
    class A
    {
      private:
        int lmax_;
      public:
        int lmax() const
        {
            return lmax_;
        }
        int lmax(int lmax__)
        {
            lmax_ = lmax__;
            return lmax_;
        }
    };
    \endcode
  - Order of class members: private, protected, public
    \code{.cpp}
    class A
    {
      private:
        int lmax_;
        void bar();
      protected:
        void foo();
      public:
        int lmax() const
        {
            return lmax_;
        }
    };
    \endcode
  - Single-line functions should not be flattened:
    \code{.cpp}
    struct A
    {
        int lmax() const
        {
            return lmax_;
        }
    };
    \endcode
    but not
    \code{.cpp}
    struct A
    {
        int lmax() const { return lmax_; }
    };
    \endcode
  - Header guards have a standard name: double underscore + file name in capital letters + double underscore
    \code{.cpp}
    #ifndef __HEADER_HPP__
    #define __HEADER_HPP__
    ...
    #endif // __HEADER_HPP__
    \endcode
  - Variable names are all in lowercase and underscore-separated (aka 'snake_case'):
    \code{.cpp}
    int num_bands;
    std::complex<double> beta_psi;
    \endcode
    but not
    \code{.cpp}
    int NumBands;
    // or
    std::complex<double> BetaPsi;
    // or
    std::complex<double> Another_BetaPsi;
    \endcode

We use clang-format utility to enforce the basic formatting style. Please have a look at .clang-format config file
in the source root folder for the definitions and use helper script 'clang_format.x'.

<b>Class naming convention.</b>

Problem: all 'standard' naming conventions are not satisfactory. For example, we have a class
which does a DFT ground state. Following the common naming conventions it could be named like this:
DFTGroundState, DftGroundState, dft_ground_state. Last two are bad, because DFT (and not Dft or dft)
is a well recognized abbreviation. First one is band because capital G adds to DFT and we automaticaly
read DFTG round state.

Solution: we can propose the following: DFTgroundState or DFT_ground_state. The first variant still
doens't look very good because one of the words is captalized (State) and one (ground) - is not. So we pick
the second variant: DFT_ground_state (by the way, this is close to the Bjarne Stroustrup's naiming convention,
where he uses first capital letter and underscores, for example class Io_obj).

Some other examples:
    - class Ground_state (composed of two words)
    - class FFT_interface (composed of an abbreviation and a word)
    - class Interface_XC (composed of a word and abbreviation)
    - class Spline (single word)

Exceptions are allowed if it makes sense. For example, low level utility classes like 'mdarray' (multi-dimentional
array) or 'pstdout' (parallel standard output) are named with small letters.
*/

/**
\page fderiv Functional derivatives

Definition:
\f[
\frac{dF[f+\epsilon \eta ]}{d \epsilon}\Bigg\rvert_{\epsilon = 0} := \int \frac{\delta F[f]}{\delta f(x')} \eta(x') dx'
\f]
Alternative definition is:
\f[
\frac{\delta F[f(x)]}{\delta f(x')} = \lim_{\epsilon \to 0} \frac{F[f(x) + \epsilon \delta(x-x')] - F[f(x)]}{\epsilon}
\f]

*/
