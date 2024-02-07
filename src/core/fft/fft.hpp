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

#include <spfft/spfft.hpp>
#include "core/splindex.hpp"
#include "core/mpi/communicator.hpp"
#include "core/memory.hpp"

namespace sirius {

/// FFT-related functions and objects.
namespace fft {

/// Type traits to handle Spfft grid for different precision type.
template <typename T>
struct SpFFT_Grid
{
};

template <>
struct SpFFT_Grid<double>
{
    using type = spfft::Grid;
};

template <>
struct SpFFT_Grid<std::complex<double>>
{
    using type = spfft::Grid;
};

#ifdef SIRIUS_USE_FP32
template <>
struct SpFFT_Grid<std::complex<float>>
{
    using type = spfft::GridFloat;
};

template <>
struct SpFFT_Grid<float>
{
    using type = spfft::GridFloat;
};
#endif

template <typename T>
using spfft_grid_type = typename SpFFT_Grid<T>::type;

/// Type traits to handle Spfft driver for different precision type.
template <typename T>
struct SpFFT_Transform
{
};

template <>
struct SpFFT_Transform<double>
{
    using type = spfft::Transform;
};

template <>
struct SpFFT_Transform<std::complex<double>>
{
    using type = spfft::Transform;
};

#ifdef SIRIUS_USE_FP32
template <>
struct SpFFT_Transform<float>
{
    using type = spfft::TransformFloat;
};

template <>
struct SpFFT_Transform<std::complex<float>>
{
    using type = spfft::TransformFloat;
};
#endif

template <typename T>
using spfft_transform_type = typename SpFFT_Transform<T>::type;

const std::map<SpfftProcessingUnitType, memory_t> spfft_memory_t = {{SPFFT_PU_HOST, memory_t::host},
                                                                    {SPFFT_PU_GPU, memory_t::device}};

template <typename F, typename T, typename... Args>
using enable_return =
        typename std::enable_if<std::is_same<typename std::result_of<F(Args...)>::type, T>::value, void>::type;

/// Load data from real-valued lambda.
template <typename T, typename F>
inline enable_return<F, T, int>
spfft_input(spfft_transform_type<T>& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<std::complex<T>*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] = std::complex<T>(fr__(i), 0.0);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<T*>(spfft__.space_domain_data(SPFFT_PU_HOST));
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
template <typename T, typename F>
inline enable_return<F, std::complex<T>, int>
spfft_input(spfft_transform_type<T>& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<std::complex<T>*>(spfft__.space_domain_data(SPFFT_PU_HOST));
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
inline void
spfft_input(spfft_transform_type<T>& spfft__, T const* data__)
{
    spfft_input<T>(spfft__, [&](int ir) -> T { return data__[ir]; });
}

template <typename T>
inline void
spfft_input(spfft_transform_type<T>& spfft__, std::complex<T> const* data__)
{
    spfft_input<T>(spfft__, [&](int ir) -> std::complex<T> { return data__[ir]; });
}

template <typename T, typename F>
inline void
spfft_multiply(spfft_transform_type<T>& spfft__, F&& fr__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<std::complex<T>*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                ptr[i] *= fr__(i);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<T*>(spfft__.space_domain_data(SPFFT_PU_HOST));
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
template <typename T>
inline void
spfft_output(spfft_transform_type<T>& spfft__, T* data__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<std::complex<T>*>(spfft__.space_domain_data(SPFFT_PU_HOST));
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spfft__.local_slice_size(); i++) {
                data__[i] = std::real(ptr[i]);
            }
            break;
        }
        case SPFFT_TRANS_R2C: {
            auto ptr = reinterpret_cast<T*>(spfft__.space_domain_data(SPFFT_PU_HOST));
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

template <typename T>
inline void
spfft_output(spfft_transform_type<T>& spfft__, std::complex<T>* data__)
{
    switch (spfft__.type()) {
        case SPFFT_TRANS_C2C: {
            auto ptr = reinterpret_cast<std::complex<T>*>(spfft__.space_domain_data(SPFFT_PU_HOST));
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

/// Total size of the SpFFT transformation grid.
template <typename T>
inline size_t
spfft_grid_size(T const& spfft__)
{
    return spfft__.dim_x() * spfft__.dim_y() * spfft__.dim_z();
}

/// Local size of the SpFFT transformation grid.
template <typename T>
inline size_t
spfft_grid_size_local(T const& spfft__)
{
    return spfft__.local_slice_size();
}

/// Split z-dimenstion of size_z between MPI ranks of the FFT communicator.
/** SpFFT works with any z-distribution of the real-space FFT buffer. Here we split the z-dimenstion
 *  using block distribution. */
inline auto
split_z_dimension(int size_z__, mpi::Communicator const& comm_fft__)
{
    return splindex_block<>(size_z__, n_blocks(comm_fft__.size()), block_id(comm_fft__.rank()));
}

} // namespace fft

} // namespace sirius

#endif // __FFT_HPP__

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
