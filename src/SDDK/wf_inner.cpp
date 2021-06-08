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

/** \file wf_inner.cpp
 *
 *  \brief Definitions.
 *
 */
#include "wf_inner.hpp"
#include "utils/profiler.hpp"
#include "SDDK/omp.hpp"
#include <chrono>
#include <spla/spla.hpp>
#include "type_definition.hpp"

#if defined(SIRIUS_GPU)
#include "gpu/acc_blas.hpp"
#include "gpu/acc.hpp"
#endif

namespace sddk {

namespace {

// If scalar type, g-vector 0 contribution must be scaled before / after inner product to aboid counting twice
template <typename T>
void scale_gamma_wf(spin_range spins, int m, int i0, real_type<T> alpha, Wave_functions& bra) {
    for (auto s : spins) {
        const int incx = bra.pw_coeffs(s).prime().ld() * 2; // complex matrix is read as scalar
        if (bra.preferred_memory_t() == memory_t::device) {
#if defined(SIRIUS_GPU)
            if (std::is_same<T, double>::value) {
                accblas::dscal(m, &alpha,
                               reinterpret_cast<double*>(bra.pw_coeffs(s).prime().at(bra.preferred_memory_t(), 0, i0)),
                               incx);
            } else if (std::is_same<T, float>::value) {
                accblas::sscal(m, &alpha,
                               reinterpret_cast<float*>(bra.pw_coeffs(s).prime().at(bra.preferred_memory_t(), 0, i0)),
                               incx);
            }
#else
            throw std::runtime_error("not compiled with GPU support!");
#endif
        } else {
            if (std::is_same<T, double>::value) {
                FORTRAN(dscal)
                (&m, reinterpret_cast<double*>(&alpha),
                 reinterpret_cast<double*>(bra.pw_coeffs(s).prime().at(bra.preferred_memory_t(), 0, i0)), &incx);
            } else if (std::is_same<T, float>::value) {
                FORTRAN(sscal)
                (&m, reinterpret_cast<float*>(&alpha),
                 reinterpret_cast<float*>(bra.pw_coeffs(s).prime().at(bra.preferred_memory_t(), 0, i0)), &incx);
            }
        }
    }
}

// If complex type, no scaling required
template <>
void scale_gamma_wf<std::complex<float>>(spin_range spins, int m, int i0, float alpha, Wave_functions& bra) {}

template <>
void scale_gamma_wf<std::complex<double>>(spin_range spins, int m, int i0, double alpha, Wave_functions& bra) {}


// general real type
template <typename T>
void
inner_mt(::spla::Context& spla_ctx__, ::spla::MatrixDistribution& spla_mat_dist__, spin_range ispn__,
      Wave_functions& bra__, int i0__, int m__, Wave_functions& ket__,
      int j0__, int n__, dmatrix<T>& result__, int irow0__, int jcol0__)
{
}

// special for complex type
template <typename T>
void
inner_mt(::spla::Context& spla_ctx__, ::spla::MatrixDistribution& spla_mat_dist__, spin_range ispn__,
    Wave_functions& bra__, int i0__, int m__, Wave_functions& ket__,
    int j0__, int n__, dmatrix<std::complex<T>>& result__, int irow0__, int jcol0__)
{
    bool local_has_mt  = bra__.has_mt();
    bool global_has_mt = false;

    // Not all ranks may have mt, but all must call spla if at least one does
    MPI_Allreduce(&local_has_mt, &global_has_mt, 1, MPI_C_BOOL, MPI_LOR, bra__.comm().mpi_comm());
    if (global_has_mt) {
        std::complex<T>* result_ptr = result__.size_local() ? result__.at(memory_t::host, 0, 0) : nullptr;
        auto spins = spin_range(ispn__);
        for (auto s : spins) {
            PROFILE("sddk::wf_inner|mt");
            if (local_has_mt) {
                spla::pgemm_ssb(
                    m__, n__, bra__.mt_coeffs(s).num_rows_loc(), SPLA_OP_CONJ_TRANSPOSE, 1.0,
                    bra__.mt_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__), bra__.mt_coeffs(s).prime().ld(),
                    ket__.mt_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__), ket__.mt_coeffs(s).prime().ld(),
                    1.0, result_ptr, result__.ld(), irow0__, jcol0__, spla_mat_dist__, spla_ctx__);
            } else {
                spla::pgemm_ssb(m__, n__, 0, SPLA_OP_CONJ_TRANSPOSE, 1.0, nullptr, 0, nullptr, 0, 1.0,
                                result__.at(memory_t::host, 0, 0), result__.ld(), irow0__, jcol0__, spla_mat_dist__,
                                spla_ctx__);
            }
        }
    }
}

} // namespace

template <typename T>
void
inner(::spla::Context& spla_ctx__, ::sddk::spin_range spins__, Wave_functions& bra__, int i0__, int m__, Wave_functions& ket__,
      int j0__, int n__, dmatrix<T>& result__, int irow0__, int jcol0__)
{
    PROFILE("sddk::wf_inner");

    spla::MatrixDistribution spla_mat_dist = bra__.comm().size() > result__.comm().size()
                                                 ? spla::MatrixDistribution::create_mirror(bra__.comm().mpi_comm())
                                                 : result__.spla_distribution();

    using precision_type = real_type<T>;
    precision_type alpha = 1.0;
    int size_factor = 1;
    if(std::is_same<T, precision_type>::value) {
        alpha = 2.0;
        size_factor = 2;
    }

    // For gamma case, contribution of g = 0 vector must not be counted double -> multiply by 0.5
    if (bra__.comm().rank() == 0) {
        PROFILE("sddk::wf_inner|scale");
        scale_gamma_wf<T>(spins__, m__, i0__, 0.5, bra__);
    }

    precision_type beta = 0.0;

    T* result_ptr = reinterpret_cast<T*>(result__.size_local() ? result__.at(memory_t::host, 0, 0) : nullptr);

    for (auto s : spins__) {
        PROFILE("sddk::wf_inner|pw");
        spla::pgemm_ssb(m__, n__, size_factor * bra__.pw_coeffs(s).num_rows_loc(), SPLA_OP_CONJ_TRANSPOSE, alpha,
                        reinterpret_cast<const T*>(bra__.pw_coeffs(s).prime().at(bra__.preferred_memory_t(), 0, i0__)),
                        size_factor * bra__.pw_coeffs(s).prime().ld(),
                        reinterpret_cast<const T*>(ket__.pw_coeffs(s).prime().at(ket__.preferred_memory_t(), 0, j0__)),
                        size_factor * ket__.pw_coeffs(s).prime().ld(), beta, result_ptr, result__.ld(), irow0__,
                        jcol0__, spla_mat_dist, spla_ctx__);
        beta = 1.0;
    }

    // For gamma case, g = 0 vector is rescaled back
    if (bra__.comm().rank() == 0) {
        PROFILE("sddk::wf_inner|scale_back");
        scale_gamma_wf<T>(spins__, m__, i0__, 2.0, bra__);
    }

    // add mt contribution
    inner_mt(spla_ctx__, spla_mat_dist, spins__, bra__, i0__, m__, ket__, j0__, n__, result__, irow0__, jcol0__);

    // make sure result is updated on device as well
    if(result__.on_device()) {
        result__.copy_to(memory_t::device);
    }
}

// instantiate for required types
template void inner<double>(::spla::Context& ctx, ::sddk::spin_range ispn__, Wave_functions& bra__,
                            int i0__, int m__, Wave_functions& ket__, int j0__, int n__, dmatrix<double>& result__,
                            int irow0__, int jcol0__);

template void inner<double_complex>(::spla::Context& ctx, ::sddk::spin_range ispn__, Wave_functions& bra__, int i0__, int m__,
                                    Wave_functions& ket__, int j0__, int n__, dmatrix<double_complex>& result__,
                                    int irow0__, int jcol0__);

// TODO: test it after Wavefunction is templated
/*
template void inner<float>(::spla::Context& ctx, ::sddk::spin_range ispn__, Wave_functions& bra__,
                            int i0__, int m__, Wave_functions& ket__, int j0__, int n__, dmatrix<float>& result__,
                            int irow0__, int jcol0__);

template void inner<std::complex<float>>(::spla::Context& ctx, ::sddk::spin_range ispn__, Wave_functions& bra__, int i0__, int m__,
                                    Wave_functions& ket__, int j0__, int n__, dmatrix<std::complex<float>>& result__,
                                    int irow0__, int jcol0__);
*/
 } // namespace sddk
