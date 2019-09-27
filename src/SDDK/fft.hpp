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

/** \file fft.hpp
 *
 *  \brief Contains helper functions for the interface with SpFFT library.
 */

#ifndef __FFT_HPP__
#define __FFT_HPP__

#include "splindex.hpp"
#include "communicator.hpp"
#include "spfft/spfft.hpp"

using double_complex = std::complex<double>;

const std::map<SpfftProcessingUnitType, sddk::memory_t> spfft_memory_t = {
    {SPFFT_PU_HOST, sddk::memory_t::host},
    {SPFFT_PU_GPU, sddk::memory_t::device}
};

template <typename F, typename T, typename ...Args>
using enable_return = typename std::enable_if<std::is_same<typename std::result_of<F(Args...)>::type, T>::value, void>::type;

/// Load data from real-valued lambda.
template <typename F>
inline enable_return<F, double, int>
spfft_input(spfft::Transform& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] = double_complex(fr__(i), 0.0);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<double*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] = fr__(i);
            }
            break;
        }
        default: {
            throw std::runtime_error("wrong spfft type");
        }
    }
}

/// Load data from complex-valued lambda.
template <typename F>
inline enable_return<F, std::complex<double>, int>
spfft_input(spfft::Transform& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] = fr__(i);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
        }
        default: {
            throw std::runtime_error("wrong spfft type");
        }
    }
}

/// Input CPU data to CPU buffer of SpFFT.
template <typename T>
inline void spfft_input(spfft::Transform& spfft__, T const* data__)
{
    spfft_input(spfft__, [&](int ir){return data__[ir];});
}

template <typename F>
inline void spfft_multiply(spfft::Transform& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] *= fr__(i);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<double*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] *= fr__(i);
            }
            break;
        }
        default: {
            throw std::runtime_error("wrong spfft type");
        }
    }
}

/// Output CPU data from the CPU buffer of SpFFT.
inline void spfft_output(spfft::Transform& spfft__, double* data__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                data__[i] = std::real(ptr[i]);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<double*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                data__[i] = ptr[i];
            }
            break;
        }
        default: {
            throw std::runtime_error("wrong spfft type");
        }
    }
}

inline void spfft_output(spfft::Transform& spfft__, double_complex* data__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<double_complex*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                data__[i] = ptr[i];
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            /* can't be a R2C transform and complex output data */
        }
        default: {
            throw std::runtime_error("wrong spfft type");
        }
    }
}

inline size_t spfft_grid_size(spfft::Transform const& spfft__)
{
    return spfft__.dim_x() * spfft__.dim_y() * spfft__.dim_z();
}

inline sddk::splindex<sddk::splindex_t::block> split_fft_z(int size_z__, sddk::Communicator const& comm_fft__)
{
    return sddk::splindex<sddk::splindex_t::block>(size_z__, comm_fft__.size(), comm_fft__.rank());
}

#endif // __FFT3D_H__

/** \page ft_pw Fourier transform and plane wave normalization
 *
 *  FFT convention:
 *  \f[
 *      f({\bf r}) = \sum_{{\bf G}} e^{i{\bf G}{\bf r}} f({\bf G})
 *  \f]
 *  is a \em backward transformation from a set of pw coefficients to a function.
 *
 *  \f[
 *      f({\bf G}) = \frac{1}{\Omega} \int e^{-i{\bf G}{\bf r}} f({\bf r}) d {\bf r} =
 *          \frac{1}{N} \sum_{{\bf r}_j} e^{-i{\bf G}{\bf r}_j} f({\bf r}_j)
 *  \f]
 *  is a \em forward transformation from a function to a set of coefficients.

 *  We use plane waves in two different cases: a) plane waves (or augmented plane waves in the case of APW+lo method)
 *  as a basis for expanding Kohn-Sham wave functions and b) plane waves are used to expand charge density and
 *  potential. When we are dealing with plane wave basis functions it is convenient to adopt the following
 *  normalization:
 *  \f[
 *      \langle {\bf r} |{\bf G+k} \rangle = \frac{1}{\sqrt \Omega} e^{i{\bf (G+k)r}}
 *  \f]
 *  such that
 *  \f[
 *      \langle {\bf G+k} |{\bf G'+k} \rangle_{\Omega} = \delta_{{\bf GG'}}
 *  \f]
 *  in the unit cell. However, for the expansion of periodic functions such as density or potential, the following
 *  convention is more appropriate:
 *  \f[
 *      \rho({\bf r}) = \sum_{\bf G} e^{i{\bf Gr}} \rho({\bf G})
 *  \f]
 *  where
 *  \f[
 *      \rho({\bf G}) = \frac{1}{\Omega} \int_{\Omega} e^{-i{\bf Gr}} \rho({\bf r}) d{\bf r} =
 *          \frac{1}{\Omega} \sum_{{\bf r}_i} e^{-i{\bf Gr}_i} \rho({\bf r}_i) \frac{\Omega}{N} =
 *          \frac{1}{N} \sum_{{\bf r}_i} e^{-i{\bf Gr}_i} \rho({\bf r}_i)
 *  \f]
 *  i.e. with such convention the plane-wave expansion coefficients are obtained with a normalized FFT.
 */
