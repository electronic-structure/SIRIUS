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
#include <limits>

using double_complex = std::complex<double>;

enum class spin_block_t {nm, uu, ud, dd, du};

/// Type of electronic structure methods.
enum class electronic_structure_method_t 
{
    /// Full potential linearized augmented plane waves with local orbitals.
    full_potential_lapwlo,

    /// Pseudopotential (ultrasoft, norm-conserving, PAW).
    pseudopotential
};

enum class index_domain_t {global, local};

enum function_domain_t {spatial, spectral};

inline uint32_t rnd()
{
    static uint32_t a = 123456;
    a = (a ^ 61) ^ (a >> 16);
    a = a + (a << 3);
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2d;
    a = a ^ (a >> 15);
    return a;
}

/// Wrapper for data types
template <typename T> 
class type_wrapper;

template<> 
class type_wrapper<double>
{
    public:
        typedef std::complex<double> complex_t;
        typedef double real_t;
        
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
            return static_cast<double>(rnd()) / std::numeric_limits<uint32_t>::max();
            //return double(std::rand()) / RAND_MAX;
        }

        static real_t bypass(complex_t val__)
        {
            return val__.real();
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
class type_wrapper<double_complex>
{
    public:
        typedef double_complex complex_t;
        typedef double real_t;
        
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
            //return std::complex<double>(double(std::rand()) / RAND_MAX, double(std::rand()) / RAND_MAX);
            double x = static_cast<double>(rnd()) / std::numeric_limits<uint32_t>::max();
            double y = static_cast<double>(rnd()) / std::numeric_limits<uint32_t>::max();
            return std::complex<double>(x, y);
        }

        static complex_t bypass(complex_t val__)
        {
            return val__;
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
