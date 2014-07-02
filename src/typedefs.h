// Copyright (c) 2013-2014 Anton Kozhevnikov, Thomas Schulthess
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

/** \file typedefs.h
 *
 *  \brief Contains typedefs, enums and type_wrapper class.
 */

#ifndef __TYPEDEFS_H__
#define __TYPEDEFS_H__

#include <cstdlib>
#include <hdf5.h>
#include <mpi.h>
#include <assert.h>
#include <complex>

typedef std::complex<double> double_complex;

enum spin_block_t {nm, uu, ud, dd, du};

/// Type of the main processing unit
enum processing_unit_t 
{
    /// use CPU
    cpu, 

    /// use GPU (with CUDA programming model)
    gpu
};

enum lattice_t {direct, reciprocal};

enum coordinates_t {cartesian, fractional};

enum mpi_op_t {op_sum, op_max};

/// Type of the linear algebra package
enum linalg_t {lapack, scalapack};

/// Type of the solver to use for the standard or generalized eigen-value problem
enum ev_solver_t 
{
    /// use LAPACK
    ev_lapack, 

    /// use ScaLAPACK
    ev_scalapack,

    /// use ELPA1 solver
    ev_elpa1,

    /// use ELPA2 (2-stage) solver
    ev_elpa2,

    /// use MAGMA
    ev_magma,

    /// use PLASMA
    ev_plasma,

    /// 
    ev_rs_gpu,

    ev_rs_cpu
};

enum splindex_t {block, block_cyclic};

/// Type of electronic structure methods
enum electronic_structure_method_t 
{
    /// full potential linearized augmented plane waves with local orbitals
    full_potential_lapwlo, 

    /// full potential plane waves with local orbitals (heavily experimental and not completely implemented)
    full_potential_pwlo, 

    /// ultrasoft pseudopotential with plane wave basis (experimental)
    ultrasoft_pseudopotential
};

enum index_domain_t {global, local};

enum function_domain_t {spatial, spectral};

/// Types of radial grid.
enum radial_grid_t 
{
    linear_grid, 
    
    exponential_grid, 
    
    pow2_grid, 
    
    pow3_grid,

    scaled_pow_grid,
    
    hyperbolic_grid, 
    
    incremental_grid
};

/// type of local orbitals
/** lo_rs - local orbital, composed of radial solutions
 *  lo_cp - confined polynomial local orbital 
 */
enum local_orbital_t {lo_rs, lo_cp};

/// Wrapper for data types
template <typename T> 
class type_wrapper
{
    public:

        static bool is_primitive()
        {
            return false;
        }
};

template<> 
class type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_DOUBLE;
        }
        
        static inline double conjugate(double& v)
        {
            return v;
        }

        static inline double sift(std::complex<double> v)
        {
            return real(v);
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_DOUBLE;
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
        {
            return true;
        }

        static inline double abs(double val)
        {
            return fabs(val);
        }

        static bool is_primitive()
        {
            return true;
        }

        static inline double random()
        {
            return double(rand()) / RAND_MAX;
        }
};

template<> 
class type_wrapper<long double>
{
    public:
        typedef std::complex<long double> complex_t;
        typedef long double real_t;
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_LONG_DOUBLE;
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
        {
            return true;
        }

        static bool is_primitive()
        {
            return true;
        }

};

template<> 
class type_wrapper<float>
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};

template<> 
class type_wrapper< std::complex<double> >
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
        static inline std::complex<double> conjugate(std::complex<double>& v)
        {
            return conj(v);
        }
        
        static inline std::complex<double> sift(std::complex<double> v)
        {
            return v;
        }
        
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_LDOUBLE;
        }
        
        static MPI_Datatype mpi_type_id()
        {
            return MPI_COMPLEX16;
        }

        static bool is_complex()
        {
            return true;
        }
        
        static bool is_real()
        {
            return false;
        }
        
        static inline double abs(std::complex<double> val)
        {
            return std::abs(val);
        }

        static bool is_primitive()
        {
            return true;
        }

        static inline std::complex<double> random()
        {
            return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        }
};

/*template<> class type_wrapper< std::complex<float> >
{
    public:
        typedef std::complex<float> complex_t;
        typedef float real_t;
};*/

template<> 
class type_wrapper<int>
{
    public:
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_INT;
        }

        static MPI_Datatype mpi_type_id()
        {
            return MPI_INT;
        }

        static bool is_primitive()
        {
            return true;
        }

        static int abs(int val)
        {
            return abs(val);
        }

        /*static bool is_complex()
        {
            return false;
        }*/
};

template<> 
class type_wrapper<char>
{
    public:
        /*static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_INT;
        }*/

        static MPI_Datatype mpi_type_id()
        {
            return MPI_CHAR;
        }

        /*static bool is_complex()
        {
            return false;
        }*/

        static bool is_primitive()
        {
            return true;
        }
};

#endif // __TYPEDEFS_H__
