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

/** \file wf_trans.cpp
 *
 *  \brief Definitions.
 *
 */

#include "wf_trans.hpp"
#include "spla/context.hpp"
#include "spla/matrix_distribution.hpp"
#include "utils/profiler.hpp"
#include "SDDK/omp.hpp"
#include <spla/spla.hpp>

#if defined(SIRIUS_GPU)
#include "gpu/acc.hpp"
#endif
namespace sddk {

namespace {
template <typename T>
void transform_mt(::spla::Context& spla_ctx__, int ispn__, typename real_type<T>::type alpha__,
                  std::vector<Wave_functions*> wf_in__, int i0__, int m__, dmatrix<T>& mtrx__, int irow0__, int jcol0__,
                  typename real_type<T>::type beta__, std::vector<Wave_functions*> wf_out__, int j0__, int n__)
{
    if (wf_in__[0]->has_mt()) {
        TERMINATE("not implemented");
    }
}

template <typename T>
void
transform_mt(::spla::Context& spla_ctx__, int ispn__, T alpha__, std::vector<Wave_functions*> wf_in__, int i0__,
             int m__, dmatrix<std::complex<T>>& mtrx__, int irow0__, int jcol0__, T beta__,
             std::vector<Wave_functions*> wf_out__, int j0__, int n__)
{
    spla::MatrixDistribution spla_mat_dist = mtrx__.spla_distribution();
    int nwf                                = static_cast<int>(wf_in__.size());
    const std::complex<T>* mtrx_ptr        = mtrx__.size_local() ? mtrx__.at(memory_t::host, 0, 0) : nullptr;
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
                                    wf_in__[iv]->mt_coeffs(in_s).prime().at(wf_in__[iv]->preferred_memory_t(), 0, i0__),
                                    wf_in__[iv]->mt_coeffs(in_s).prime().ld(), mtrx_ptr, mtrx__.ld(), irow0__, jcol0__,
                                    spla_mat_dist, beta__,
                                    wf_out__[iv]->mt_coeffs(s).prime().at(wf_out__[iv]->preferred_memory_t(), 0, j0__),
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

template <typename T>
void
transform(::spla::Context& spla_ctx__, int ispn__, typename real_type<T>::type alpha__,
          std::vector<Wave_functions*> wf_in__, int i0__, int m__, dmatrix<T>& mtrx__, int irow0__, int jcol0__,
          typename real_type<T>::type beta__, std::vector<Wave_functions*> wf_out__, int j0__, int n__)
{
    PROFILE("sddk::wf_trans");
    int nwf = static_cast<int>(wf_in__.size());

    spla::MatrixDistribution spla_mat_dist = mtrx__.spla_distribution();

    int size_factor = std::is_same<T, typename real_type<T>::type>::value ? 2 : 1;

    auto spins = spin_range(ispn__);

    const T* mtrx_ptr = reinterpret_cast<const T*>(mtrx__.size_local() ? mtrx__.at(memory_t::host, 0, 0) : nullptr);

    for (int iv = 0; iv < nwf; iv++) {
        for (auto s : spins) {
            PROFILE("sddk::wf_trans|pw");
            /* input wave-functions may be scalar (this is the case of transformation of first-variational states
               into spinor wave-functions or transforamtion of scalar auxiliary wave-functions into spin-dependent
               wave-fucntions; in this case we set spin index of input wave-function to 0 */
            int in_s = (wf_in__[iv]->num_sc() == 1) ? 0 : s;

            spla::pgemm_sbs(size_factor * wf_in__[iv]->pw_coeffs(in_s).num_rows_loc(), n__, m__, alpha__,
                            reinterpret_cast<const T*>(
                                wf_in__[iv]->pw_coeffs(in_s).prime().at(wf_in__[iv]->preferred_memory_t(), 0, i0__)),
                            size_factor * wf_in__[iv]->pw_coeffs(in_s).prime().ld(), mtrx_ptr, mtrx__.ld(), irow0__,
                            jcol0__, spla_mat_dist, beta__,
                            reinterpret_cast<T*>(
                                wf_out__[iv]->pw_coeffs(s).prime().at(wf_out__[iv]->preferred_memory_t(), 0, j0__)),
                            size_factor * wf_out__[iv]->pw_coeffs(s).prime().ld(), spla_ctx__);
        }
    }

    transform_mt<T>(spla_ctx__, ispn__, alpha__, wf_in__, i0__, m__, mtrx__, irow0__, jcol0__, beta__, wf_out__, j0__,
                    n__);
}

template void transform<double>(::spla::Context& spla_ctx__, int ispn__, double alpha__,
                                std::vector<Wave_functions*> wf_in__, int i0__, int m__, dmatrix<double>& mtrx__,
                                int irow0__, int jcol0__, double beta__, std::vector<Wave_functions*> wf_out__,
                                int j0__, int n__);

template void transform<double_complex>(::spla::Context& spla_ctx__, int ispn__, double alpha__,
                                        std::vector<Wave_functions*> wf_in__, int i0__, int m__,
                                        dmatrix<double_complex>& mtrx__, int irow0__, int jcol0__, double beta__,
                                        std::vector<Wave_functions*> wf_out__, int j0__, int n__);

// TODO: test it after Wavefunction is templated
/*
template void transform<float>(::spla::Context& spla_ctx__, int ispn__, float alpha__,
                               std::vector<Wave_functions*> wf_in__, int i0__, int m__, dmatrix<float>& mtrx__,
                               int irow0__, int jcol0__, float beta__, std::vector<Wave_functions*> wf_out__, int j0__,
                               int n__);

template void transform<std::complex<float>>(::spla::Context& spla_ctx__, int ispn__, float alpha__,
                                             std::vector<Wave_functions*> wf_in__, int i0__, int m__,
                                             dmatrix<std::complex<float>>& mtrx__, int irow0__, int jcol0__,
                                             float beta__, std::vector<Wave_functions*> wf_out__, int j0__, int n__);
*/
} // namespace sddk
