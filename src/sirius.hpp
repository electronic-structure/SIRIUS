/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file sirius.hpp
 *
 *  \brief "All-in-one" include file.
 */

#ifndef __SIRIUS_HPP__
#define __SIRIUS_HPP__

#if defined(__APEX)
#include <apex_api.hpp>
#endif

#include "core/omp.hpp"
#if defined(SIRIUS_GPU) && defined(SIRIUS_CUDA)
#include "core/acc/cusolver.hpp"
#endif
#include "core/la/linalg_spla.hpp"
#include "core/cmd_args.hpp"
#include "core/json.hpp"
#include "core/profiler.hpp"
using json = nlohmann::json;
#if defined(SIRIUS_USE_POWER_COUNTER)
#include "core/power.hpp"
#endif

#include "core/sht/sht.hpp"
#include "core/sht/gaunt.hpp"
#include "core/hdf5_tree.hpp"
#include "core/sirius_version.hpp"
#include "core/la/eigenproblem.hpp"
#include "context/simulation_context.hpp"
#include "hamiltonian/local_operator.hpp"
#include "radial/radial_solver.hpp"
#include "dft/dft_ground_state.hpp"

/// Namespace of the SIRIUS library.
namespace sirius {

extern json sirius_options_parser_;

/// Return the status of the library (initialized or not).
inline static bool&
is_initialized()
{
    static bool b{false};
    return b;
}

#if defined(SIRIUS_USE_POWER_COUNTER)
inline static double&
energy()
{
    static double e__{0};
    return e__;
}

inline static double&
energy_acc()
{
    static double e__{0};
    return e__;
}
#endif

/// Initialize the library.
inline void
initialize(bool call_mpi_init__ = true)
{
    PROFILE_START("sirius");
    PROFILE("sirius::initialize");
    if (is_initialized()) {
        RTE_THROW("SIRIUS library is already initialized");
    }
#if defined(SIRIUS_USE_POWER_COUNTER)
    energy()     = -power::energy();
    energy_acc() = -power::device_energy();
#endif
    if (call_mpi_init__) {
        mpi::Communicator::initialize(MPI_THREAD_MULTIPLE);
    }
#if defined(__APEX)
    apex::init("sirius", Communicator::world().rank(), Communicator::world().size());
#endif

    if (mpi::Communicator::world().rank() == 0) {
        std::printf("SIRIUS %i.%i.%i, git hash: %s\n", major_version(), minor_version(), revision(),
                    git_hash().c_str());
#if !defined(NDEBUG)
        std::printf("Warning! Compiled in 'debug' mode with assert statements enabled!\n");
#endif
    }
    /* get number of ranks per node during the global call to sirius::initialize() */
    mpi::num_ranks_per_node();
    if (acc::num_devices() > 0) {
        int devid = mpi::get_device_id(acc::num_devices());
        acc::set_device_id(devid);
        /* create extensive amount of streams */
        /* some parts of the code rely on the number of streams not related to the
           number of OMP threads */
        acc::create_streams(std::max(omp_get_max_threads(), 6));
#if defined(SIRIUS_GPU)
        acc::blas::create_stream_handles();
#endif
#if defined(SIRIUS_CUDA)
        acc::blas::xt::create_handle();
        acc::cusolver::create_handle();
#endif
    }
    splablas::reset_handle();

#if defined(SIRIUS_MAGMA)
    magma::init();
#endif
#if defined(SIRIUS_ELPA)
    la::Eigensolver_elpa::initialize();
#endif
#if defined(SIRIUS_DLAF)
    la::Eigensolver_dlaf::initialize();
#endif
    /* for the fortran interface to blas/lapack */
    RTE_ASSERT(sizeof(int) == 4);
    RTE_ASSERT(sizeof(double) == 8);

    is_initialized() = true;
}

/// Shut down the library.
inline void
finalize(bool call_mpi_fin__ = true, bool reset_device__ = true, bool fftw_cleanup__ = true)
{
    PROFILE_START("sirius::finalize");
    if (!is_initialized()) {
        RTE_THROW("SIRIUS library was not initialized");
    }
#if defined(SIRIUS_MAGMA)
    magma::finalize();
#endif

    /* must be called before device is reset */
    splablas::reset_handle();

    get_memory_pool(memory_t::host).clear();

    if (acc::num_devices()) {
        get_memory_pool(memory_t::host_pinned).clear();
        get_memory_pool(memory_t::device).clear();
#if defined(SIRIUS_GPU)
        acc::blas::destroy_stream_handles();
#endif
#if defined(SIRIUS_CUDA)
        acc::cusolver::destroy_handle();
        acc::blas::xt::destroy_handle();
#endif
        acc::destroy_streams();
    }

#if defined(__APEX)
    apex::finalize();
#endif
#if defined(SIRIUS_USE_POWER_COUNTER)
    double e     = energy() + power::energy();
    double e_acc = energy_acc() + power::device_energy();
    if (mpi::Communicator::world().rank() == 0) {
        printf("=== Energy consumption (root MPI rank) ===\n");
        printf("energy     : %9.2f Joules\n", e);
        printf("energy_acc : %9.2f Joules\n", e_acc);
    }
    mpi::Communicator::world().allreduce(&e, 1);
    mpi::Communicator::world().allreduce(&e_acc, 1);
    int nn = power::num_nodes();
    if (Communicator::world().rank() == 0 && nn > 0) {
        printf("=== Energy consumption (all nodes) ===\n");
        printf("energy     : %9.2f Joules\n", e * nn / Communicator::world().size());
        printf("energy_acc : %9.2f Joules\n", e_acc * nn / Communicator::world().size());
    }
#endif
    auto rank = mpi::Communicator::world().rank();
    if (call_mpi_fin__) {
        mpi::Communicator::finalize();
    }
#if defined(SIRIUS_ELPA)
    la::Eigensolver_elpa::finalize();
#endif
#if defined(SIRIUS_DLAF)
    la::Eigensolver_dlaf::finalize();
#endif
    if (acc::num_devices() && reset_device__) {
        acc::reset();
    }

    is_initialized() = false;

    PROFILE_STOP("sirius::finalize");
    PROFILE_STOP("sirius");

    auto pt = env::print_timing();
    if (pt && rank == 0) {
        auto timing_result = global_rtgraph_timer.process();

        if (pt & 1) {
            std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                              rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median,
                                              rt_graph::Stat::Min, rt_graph::Stat::Max});
        }
        if (pt & 2) {
            timing_result = timing_result.flatten(1).sort_nodes();
            std::cout << timing_result.print({rt_graph::Stat::Count, rt_graph::Stat::Total, rt_graph::Stat::Percentage,
                                              rt_graph::Stat::SelfPercentage, rt_graph::Stat::Median,
                                              rt_graph::Stat::Min, rt_graph::Stat::Max});
        }
    }
}

} // namespace sirius
#endif // __SIRIUS_HPP__

/** \mainpage Welcome to SIRIUS

  SIRIUS is a domain specific library for electronic structure calculations. It implements pseudopotential plane
  wave (PP-PW) and full potential linearized augmented plane wave (FP-LAPW) methods and is designed for
  GPU acceleration of popular community codes such as Exciting, Elk and Quantum ESPRESSO.
  SIRIUS is written in C++11 with MPI, OpenMP and CUDA/ROCm programming models. SIRIUS is organised as a
  collection of classes that abstract away the different building blocks of DFT self-consistency cycle.

  For a quick start please refer to the main development page at
  <a href="https://github.com/electronic-structure/SIRIUS">GitHub</a>.

  The generated Fortran API is described in the file sirius.f90.

  The frequent variable names are listed on the page \ref stdvarname.

  We use the following \ref coding.

  The library files and directories are organised in the following way:
    - \b apps -
     - \b atoms - utility program to generate FP-LAPW atomic species files
     - \b bands - band plotting
     - \b cif_input - CIF parser
     - \b mini_app - DFT miniapp
     - \b hydrogen - solve hydrogen-like atom using Schrödinger equation
     - \b tests - tests of various functionality
     - \b timers - scripts to analyze timer outputs
     - \b unit_tests - unit tests
     - \b upf - scripts to parse and convert UPF files
     - \b utils - utilities to work with unit cell
    - \b ci - directory with Jenkins, Travis CI and GitHub action scripts
    - \b cmake - directory with CMake scripts
    - \b doc - this directory contains configuration file for Doxygen documentation and PNG images
    - \b examples - examples of input files for pseudopotential and full-potential calculations
    - \b python_module - Python interface module
    - \b reframe - ReFrame regression tests description
    - \b src - main directory with the source code
    - \b verification - verification tests
    - .clang-format - source code formatting rules
    - CMakeLists.txt - CMake file of the project
    - check_format.py, check_format.x - scripts to check source code formatting
    - clang_format.x - script to apply Clang format to a file

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
  - Page width is approximately 120 characters. Screens are wide nowadays and 80 characters is an
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

Exceptions are allowed if it makes sense. For example, low level utility classes like 'mdarray' (multi-dimensional
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
