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

enum class spin_block_t {nm, uu, ud, dd, du};

//enum lattice_t {direct, reciprocal};

//enum coordinates_t {cartesian, fractional};

/// Type of electronic structure methods.
enum electronic_structure_method_t 
{
    /// Full potential linearized augmented plane waves with local orbitals.
    full_potential_lapwlo, 

    /// Full potential plane waves with local orbitals (heavily experimental and not completely implemented).
    full_potential_pwlo, 

    /// Ultrasoft pseudopotential with plane wave basis (experimental).
    ultrasoft_pseudopotential,

    /// PAW pseudopotential with plane wave basis (experimental).
    paw_pseudopotential,

    /// Norm-conserving pseudopotential with plane wave basis (experimental).
    norm_conserving_pseudopotential
};

enum index_domain_t {global, local};

enum function_domain_t {spatial, spectral};

/// Wrapper for data types
template <typename T> 
class type_wrapper;

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
        
        static inline double conjugate(double const& v)
        {
            return v;
        }

        static inline double sift(double_complex const& v)
        {
            return std::real(v);
        }

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
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

        static bool is_complex()
        {
            return false;
        }
        
        static bool is_real()
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
        
        static inline std::complex<double> conjugate(double_complex const& v)
        {
            return conj(v);
        }
        
        static inline std::complex<double> sift(double_complex const& v)
        {
            return v;
        }
        
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_LDOUBLE;
        }
        
        static bool is_complex()
        {
            return true;
        }
        
        static bool is_real()
        {
            return false;
        }
        
        static inline std::complex<double> random()
        {
            return std::complex<double>(double(rand()) / RAND_MAX, double(rand()) / RAND_MAX);
        }
};

template<> 
class type_wrapper<int>
{
    public:
        static hid_t hdf5_type_id()
        {
            return H5T_NATIVE_INT;
        }
};

template<> 
class type_wrapper<char>
{
    public:

        static inline char random()
        {
            return static_cast<char>(255 * (double(rand()) / RAND_MAX));
        }
};

enum class relativity_t
{
    none,

    koelling_harmon,

    zora,

    iora,

    dirac
};


#endif // __TYPEDEFS_H__
