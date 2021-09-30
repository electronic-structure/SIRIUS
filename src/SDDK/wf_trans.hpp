// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file wf_trans.hpp
 *
 *  \brief Wave-function linear transformation.
 */
#ifndef __WF_TRANS_HPP__
#define __WF_TRANS_HPP__

#include <spla/spla.hpp>
#include <spla/context.hpp>
#include <spla/matrix_distribution.hpp>
#include "wave_functions.hpp"
#include "type_definition.hpp"
#include "utils/profiler.hpp"
#include "SDDK/omp.hpp"
#if defined(SIRIUS_GPU)
#include "gpu/acc.hpp"
#endif

namespace sddk {

namespace {

template <typename T,  typename F, typename = std::enable_if_t<std::is_scalar<T>::value>>
inline std::enable_if_t<std::is_same<real_type<T>, real_type<F>>::value, void>
transform_mt(::spla::Context& spla_ctx__, int ispn__, real_type<F> alpha__,
             std::vector<Wave_functions<T>*> wf_in__, int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__,
             real_type<F> beta__, std::vector<Wave_functions<T>*> wf_out__, int j0__, int n__)
{
    if (wf_in__[0]->has_mt()) {
        TERMINATE("not implemented");
    }
}

// implemented only for complex type
template <typename T,  typename F, typename = std::enable_if_t<!std::is_scalar<T>::value>>
inline std::enable_if_t<std::is_same<real_type<T>, real_type<F>>::value, void>
transform_mt(::spla::Context& spla_ctx__, int ispn__, real_type<F> alpha__, std::vector<Wave_functions<real_type<T>>*> wf_in__,
             int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__, F beta__,
             std::vector<Wave_functions<real_type<T>>*> wf_out__, int j0__, int n__)
{
    spla::MatrixDistribution spla_mat_dist = mtrx__.spla_distribution();
    int nwf                                = static_cast<int>(wf_in__.size());
    const F* mtrx_ptr                      = mtrx__.size_local() ? mtrx__.at(memory_t::host, 0, 0) : nullptr;
    auto spins                             = spin_range(ispn__);
    for (int iv = 0; iv < nwf; iv++) {
        bool local_has_mt  = wf_in__[iv]->has_mt();
        bool global_has_mt = false;
        MPI_Allreduce(&local_has_mt, &global_has_mt, 1, MPI_C_BOOL, MPI_LOR, wf_in__[iv]->comm().mpi_comm());
        if (global_has_mt) {
            for (auto s : spins) {
                PROFILE("sddk::wf_trans|mt");
                /* input wave-functions may be scalar (this is the case of transformation of first-variational states
                   into spinor wave-functions or transforamtion of scalar auxiliary wave-functions into spin-dependent
                   wave-fucntions; in this case we set spin index of input wave-function to 0 */
                int in_s = (wf_in__[iv]->num_sc() == 1) ? 0 : s;

                if (local_has_mt) {
                    spla::pgemm_sbs(wf_in__[iv]->mt_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
                                    reinterpret_cast<const F*>(wf_in__[iv]->mt_coeffs(in_s).prime().at(wf_in__[iv]->preferred_memory_t(), 0, i0__)),
                                    wf_in__[iv]->mt_coeffs(in_s).prime().ld(), mtrx_ptr, mtrx__.ld(), irow0__, jcol0__,
                                    spla_mat_dist, beta__,
                                    reinterpret_cast<F*>(wf_out__[iv]->mt_coeffs(s).prime().at(wf_out__[iv]->preferred_memory_t(), 0, j0__)),
                                    wf_out__[iv]->mt_coeffs(s).prime().ld(), spla_ctx__);
                } else {
                    spla::pgemm_sbs(0, n__, m__, alpha__, nullptr, 0, mtrx__.at(memory_t::host, 0, 0), mtrx__.ld(),
                                    irow0__, jcol0__, spla_mat_dist, beta__, nullptr, 0, spla_ctx__);
                }
            }
        }
    }
}
} // namespace

/// Linear transformation of the wave-functions.
/** The transformation matrix is expected in the CPU memory. The following operation is performed:
 *  \f[
 *     \psi^{out}_{j} = \alpha \sum_{i} \psi^{in}_{i} Z_{ij} + \beta \psi^{out}_{j}
 *  \f]
 */
template <typename T, typename F>
inline std::enable_if_t<std::is_same<real_type<T>, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, int ispn__, real_type<F> alpha__,
          std::vector<Wave_functions<real_type<T>>*> wf_in__, int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__,
          real_type<F> beta__, std::vector<Wave_functions<real_type<T>>*> wf_out__, int j0__, int n__)
{
    PROFILE("sddk::wf_trans");
    int nwf = static_cast<int>(wf_in__.size());

    spla::MatrixDistribution spla_mat_dist = mtrx__.spla_distribution();

    int size_factor = std::is_same<F, real_type<F>>::value ? 2 : 1;

    auto spins = spin_range(ispn__);

    const F* mtrx_ptr = reinterpret_cast<const F*>(mtrx__.size_local() ? mtrx__.at(memory_t::host, 0, 0) : nullptr);

    for (int iv = 0; iv < nwf; iv++) {
        for (auto s : spins) {
            PROFILE("sddk::wf_trans|pw");
            /* input wave-functions may be scalar (this is the case of transformation of first-variational states
               into spinor wave-functions or transforamtion of scalar auxiliary wave-functions into spin-dependent
               wave-fucntions; in this case we set spin index of input wave-function to 0 */
            int in_s = (wf_in__[iv]->num_sc() == 1) ? 0 : s;

            spla::pgemm_sbs(size_factor * wf_in__[iv]->pw_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
                            reinterpret_cast<const F*>(
                                wf_in__[iv]->pw_coeffs(in_s).prime().at(wf_in__[iv]->preferred_memory_t(), 0, i0__)),
                            size_factor * wf_in__[iv]->pw_coeffs(in_s).prime().ld(), mtrx_ptr, mtrx__.ld(), irow0__,
                            jcol0__, spla_mat_dist, beta__,
                            reinterpret_cast<F*>(
                                wf_out__[iv]->pw_coeffs(s).prime().at(wf_out__[iv]->preferred_memory_t(), 0, j0__)),
                            size_factor * wf_out__[iv]->pw_coeffs(s).prime().ld(), spla_ctx__);
        }
    }

    transform_mt<T, F>(spla_ctx__, ispn__, alpha__, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, beta__, wf_out__, j0__,
                    n__);
}

template <typename T, typename F>
inline std::enable_if_t<!std::is_same<real_type<T>, real_type<F>>::value, void>
transform(::spla::Context& spla_ctx__, int ispn__, real_type<F> alpha__, std::vector<Wave_functions<real_type<T>>*>  wf_in__,
          int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__,
          real_type<F> beta__, std::vector<Wave_functions<real_type<T>>*>  wf_out__, int j0__, int n__)
{
    std::cout << "[wf_trans] slow implemntation" << std::endl;
    spin_range spins(ispn__);

    for (int idx = 0; idx < static_cast<int>(wf_in__.size()); idx++) {
        for (auto s : spins) {
            for (int j = 0; j < n__; j++) {
                for (int k = 0; k < wf_in__[idx]->pw_coeffs(s).num_rows_loc(); k++) {
                    complex_type<F> z(0, 0);;
                    for (int i = 0; i < m__; i++) {
                        z += static_cast<complex_type<F>>(wf_in__[idx]->pw_coeffs(s).prime(k, i + i0__)) *
                             mtrx__(irow0__ + i, jcol0__ + j);
                    }
                    wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__) = alpha__ * z +
                        static_cast<complex_type<F>>(wf_out__[idx]->pw_coeffs(s).prime(k, j + j0__)) * beta__;
                }
            }
        }
    }

}

template <typename T, typename F>
inline void
transform(::spla::Context& spla_ctx__, int ispn__, std::vector<Wave_functions<real_type<T>>*> wf_in__,
          int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__,
          std::vector<Wave_functions<real_type<T>>*> wf_out__, int j0__, int n__)
{
    transform<T, F>(spla_ctx__ , ispn__, static_cast<real_type<F>>(1.0), wf_in__, i0__, m__, mtrx__, irow0__, jcol0__,
                    static_cast<real_type<F>>(0.0), wf_out__, j0__, n__);
}

template <typename T, typename F>
inline void
transform(::spla::Context& spla_ctx__, int ispn__, Wave_functions<real_type<T>>& wf_in__,
          int i0__, int m__, dmatrix<F>& mtrx__, int irow0__, int jcol0__,
          Wave_functions<real_type<T>>& wf_out__, int j0__, int n__)
{
    transform<T, F>(spla_ctx__, ispn__, {&wf_in__}, i0__, m__, mtrx__, irow0__, jcol0__, {&wf_out__}, j0__, n__);
}

}
#endif
