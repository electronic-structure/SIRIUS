// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file generate_fv_states.cpp
 *   
 *  \brief Contains implementation of sirius::K_point::generate_fv_states method.
 */

#include "k_point.h"

namespace sirius {

void K_point::generate_fv_states()
{
    PROFILE_WITH_TIMER("sirius::K_point::generate_fv_states");
    
    if (!ctx_.full_potential()) {
        return;
    }

    fv_eigen_vectors_->swap_forward(0, ctx_.num_fv_states());
    #ifdef __GPU
    auto& fv_ev_swp = fv_eigen_vectors_->coeffs_swapped();
    #endif
    /* local number of bands */
    int nbnd_loc = fv_eigen_vectors_->spl_num_col().local_size();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        fv_ev_swp.allocate(memory_t::device);
        fv_ev_swp.copy_to_device();
    }
    #endif

    fv_states<true>().set_num_swapped(ctx_.num_fv_states());

    assert(nbnd_loc == fv_states<true>().spl_num_col().local_size());

    //if (ctx_.processing_unit() == GPU)
    //{
    //    #ifdef __GPU
    //    STOP();
    //    ///* copy eigen-vectors to GPU */
    //    //fv_eigen_vectors_slice.allocate_on_device();
    //    //fv_eigen_vectors_slice.copy_to_device();
    //    ///* allocate GPU memory for fv_states */
    //    //fv_states_slice_.allocate_on_device();

    //    //double_complex alpha(1, 0);
    //    //double_complex beta(0, 0);
    //    //
    //    //Timer t1("sirius::K_point::generate_fv_states|zgemm_eff");
    //    //#pragma omp parallel
    //    //{
    //    //    int tid = Platform::thread_id();
    //    //    mdarray<double_complex, 2> alm(nullptr, num_gkvec(), unit_cell_.max_mt_aw_basis_size());
    //    //    alm.allocate_page_locked();
    //    //    alm.allocate_on_device();
    //    //    
    //    //    #pragma omp for
    //    //    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    //    //    {
    //    //        int mt_aw_size = unit_cell_.atom(ia)->mt_aw_basis_size();
    //    //        int offset_wf = unit_cell_.atom(ia)->offset_wf();
    //    //        alm_coeffs_->generate(ia, alm);
    //    //        alm.async_copy_to_device();
    //    //        linalg<GPU>::gemm(1, 0, mt_aw_size, fv_eigen_vectors_slice.num_cols_local(), num_gkvec(), &alpha,
    //    //                          alm.at<GPU>(), alm.ld(), fv_eigen_vectors_slice.at<GPU>(), fv_eigen_vectors_slice.ld(),
    //    //                          &beta, fv_states_slice_.at<GPU>(offset_wf, 0), fv_states_slice_.ld(), tid);

    //    //        /* copy block of local orbital coefficients */
    //    //        cuda_memcpy2D_device_to_device_async(fv_states_slice_.at<GPU>(offset_wf + mt_aw_size, 0),
    //    //                                             fv_states_slice_.ld(),
    //    //                                             fv_eigen_vectors_slice.at<GPU>(num_gkvec() + unit_cell_.atom(ia)->offset_lo(), 0),
    //    //                                             fv_eigen_vectors_slice.ld(),
    //    //                                             unit_cell_.atom(ia)->mt_lo_basis_size(), fv_states_slice_.num_cols_local(),
    //    //                                             sizeof(double_complex), tid);
    //    //        cuda_stream_synchronize(tid);
    //    //    }
    //    //}
    //    //double tval = t1.stop();
    //    //DUMP("effective zgemm performance: %f GFlops / rank",
    //    //     8e-9 * unit_cell_.mt_basis_size() * num_gkvec() * ctx_.num_fv_states() / tval / comm().size());
    //    ///* copy block of pw coefficients */
    //    //cuda_memcpy2D_device_to_device(fv_states_slice_.at<GPU>(unit_cell_.mt_basis_size(), 0),
    //    //                               fv_states_slice_.ld(),
    //    //                               fv_eigen_vectors_slice.at<GPU>(),
    //    //                               fv_eigen_vectors_slice.ld(),
    //    //                               num_gkvec(), fv_states_slice_.num_cols_local(), sizeof(double_complex));

    //    //fv_eigen_vectors_slice.deallocate_on_device();
    //    //fv_states_slice_.copy_to_host();
    //    //fv_states_slice_.deallocate_on_device();
    //    #else
    //    TERMINATE_NO_GPU
    //    #endif
    //}
    //if (ctx_.processing_unit() == GPU && num_ranks() == 1)
    //{
        //#ifdef __GPU
        ///* copy eigen-vectors to GPU */
        //fv_eigen_vectors_panel_.panel().allocate_on_device();
        //fv_eigen_vectors_panel_.panel().copy_to_device();

        ///* allocate GPU memory for fv_states */
        //fv_states_.allocate_on_device();

        //double_complex alpha(1, 0);
        //double_complex beta(0, 0);

        //int num_atoms_in_block = 2 * Platform::max_num_threads();
        //int nblk = unit_cell_.num_atoms() / num_atoms_in_block + std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);
        //DUMP("nblk: %i", nblk);

        //int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
        //DUMP("max_mt_aw: %i", max_mt_aw);

        //mdarray<double_complex, 3> alm_row(nullptr, num_gkvec_row(), max_mt_aw, 2);
        //alm_row.allocate(1);
        //alm_row.allocate_on_device();
        //
        //int mt_aw_blk_offset = 0;
        //for (int iblk = 0; iblk < nblk; iblk++)
        //{
        //    int num_mt_aw_blk = 0;
        //    std::vector<int> offsets(num_atoms_in_block);
        //    for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
        //    {
        //        auto atom = unit_cell_.atom(ia);
        //        auto type = atom->type();
        //        offsets[ia - iblk * num_atoms_in_block] = num_mt_aw_blk;
        //        num_mt_aw_blk += type->mt_aw_basis_size();
        //    }

        //    int s = iblk % 2;
        //        
        //    #pragma omp parallel
        //    {
        //        int tid = Platform::thread_id();
        //        for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
        //        {
        //            if (ia % Platform::num_threads() == tid)
        //            {
        //                int ialoc = ia - iblk * num_atoms_in_block;
        //                auto atom = unit_cell_.atom(ia);
        //                auto type = atom->type();

        //                mdarray<double_complex, 2> alm_row_tmp(alm_row.at<CPU>(0, offsets[ialoc], s),
        //                                                       alm_row.at<GPU>(0, offsets[ialoc], s),
        //                                                       num_gkvec_row(), type->mt_aw_basis_size());

        //                alm_coeffs_row()->generate(ia, alm_row_tmp);
        //                alm_row_tmp.async_copy_to_device(tid);
        //            }
        //        }
        //        cuda_stream_synchronize(tid);
        //    }
        //    cuda_stream_synchronize(Platform::max_num_threads());
        //    /* gnerate aw expansion coefficients */
        //    linalg<GPU>::gemm(1, 0, num_mt_aw_blk, ctx_.num_fv_states(), num_gkvec_row(), &alpha,
        //                      alm_row.at<GPU>(0, 0, s), alm_row.ld(),
        //                      fv_eigen_vectors_panel_.panel().at<GPU>(), fv_eigen_vectors_panel_.panel().ld(),
        //                      &beta, fv_states_.at<GPU>(mt_aw_blk_offset, 0), fv_states_.ld(), Platform::max_num_threads());
        //    mt_aw_blk_offset += num_mt_aw_blk;
        //}
        //cuda_stream_synchronize(Platform::max_num_threads());
        //alm_row.deallocate_on_device();

        //mdarray<double_complex, 2> tmp_buf(nullptr, unit_cell_.max_mt_aw_basis_size(), ctx_.num_fv_states());
        //tmp_buf.allocate_on_device();

        ///* copy aw coefficients starting from bottom */
        //for (int ia = unit_cell_.num_atoms() - 1; ia >= 0; ia--)
        //{
        //    int offset_wf = unit_cell_.atom(ia)->offset_wf();
        //    int offset_aw = unit_cell_.atom(ia)->offset_aw();
        //    int mt_aw_size = unit_cell_.atom(ia)->mt_aw_basis_size();
        //    
        //    /* copy to temporary array */
        //    cuda_memcpy2D_device_to_device(tmp_buf.at<GPU>(), tmp_buf.ld(),
        //                                   fv_states_.at<GPU>(offset_aw, 0), fv_states_.ld(),
        //                                   mt_aw_size, ctx_.num_fv_states(), sizeof(double_complex));

        //    /* copy to proper place in wave-function array */
        //    cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf, 0), fv_states_.ld(),
        //                                   tmp_buf.at<GPU>(), tmp_buf.ld(),
        //                                   mt_aw_size, ctx_.num_fv_states(), sizeof(double_complex));
        //    
        //    /* copy block of local orbital coefficients */
        //    cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(offset_wf + mt_aw_size, 0), fv_states_.ld(),
        //                                   fv_eigen_vectors_panel_.panel().at<GPU>(num_gkvec_row() + unit_cell_.atom(ia)->offset_lo(), 0),
        //                                   fv_eigen_vectors_panel_.panel().ld(),
        //                                   unit_cell_.atom(ia)->mt_lo_basis_size(), ctx_.num_fv_states(), sizeof(double_complex));
        //}
        ///* copy block of pw coefficients */
        //cuda_memcpy2D_device_to_device(fv_states_.at<GPU>(unit_cell_.mt_basis_size(), 0), fv_states_.ld(),
        //                               fv_eigen_vectors_panel_.panel().at<GPU>(),  fv_eigen_vectors_panel_.panel().ld(),
        //                               num_gkvec_row(), ctx_.num_fv_states(), sizeof(double_complex));

        //fv_eigen_vectors_panel_.panel().deallocate_on_device();
        //fv_states_.copy_to_host();
        ////fv_states_.deallocate_on_device();
        //#else
        //TERMINATE_NO_GPU
        //#endif
    //}
    //else
    //{
    #pragma omp parallel
    {
        /* get thread id */
        #ifdef __GPU
        int tid = omp_get_thread_num();
        #endif
        mdarray<double_complex, 2> alm(num_gkvec(), unit_cell_.max_mt_aw_basis_size(), memory_t::host_pinned);
        mdarray<double_complex, 2> tmp;

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.allocate(memory_t::device);
            tmp = mdarray<double_complex, 2>(unit_cell_.max_mt_aw_basis_size(), nbnd_loc, memory_t::device);

            // TODO: pin memory for fv_states (output buffer), otherwise async copy won't work
        }
        #endif
        
        #pragma omp for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            /* number of alm coefficients for atom */
            int mt_aw_size = unit_cell_.atom(ia).mt_aw_basis_size();
            /* offset in wave-function */
            int offset_wf = unit_cell_.atom(ia).offset_wf();
            /* generate matching coefficients for all G-vectors */
            alm_coeffs_->generate(ia, alm);
            
            /* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for a single atom */
            switch (ctx_.processing_unit()) {
                case CPU: {
                    /* multiply eigen-vectors and matching coefficients */
                    linalg<CPU>::gemm(1, 0, mt_aw_size, nbnd_loc, num_gkvec(),
                                      alm.at<CPU>(), alm.ld(),
                                      (*fv_eigen_vectors_)[0], gklo_basis_size(),
                                      &fv_states<true>()[0][offset_wf], wf_size());
                    break;
                }
                case GPU: {
                    #ifdef __GPU
                    alm.async_copy_to_device(tid);
                    linalg<GPU>::gemm(1, 0, mt_aw_size, nbnd_loc, num_gkvec(),
                                      alm.at<GPU>(), alm.ld(),
                                      fv_ev_swp.at<GPU>(), gklo_basis_size(),
                                      tmp.at<GPU>(), tmp.ld(), tid);
                    acc::copyout(&fv_states<true>()[0][offset_wf], wf_size(), tmp.at<GPU>(), tmp.ld(), mt_aw_size, nbnd_loc, tid);
                    #endif
                    break;
                }
            }

            for (int i = 0; i < nbnd_loc; i++) {
                /* lo block */
                std::memcpy(&fv_states<true>()[i][offset_wf + mt_aw_size],
                            &(*fv_eigen_vectors_)[i][num_gkvec() + unit_cell_.atom(ia).offset_lo()],
                            unit_cell_.atom(ia).mt_lo_basis_size() * sizeof(double_complex));
            }
        }
        #pragma omp for
        for (int i = 0; i < nbnd_loc; i++) {
            /* G+k block */
            std::memcpy(&fv_states<true>()[i][wf_pw_offset()],
                        (*fv_eigen_vectors_)[i], num_gkvec() * sizeof(double_complex));
        }
        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            acc::sync_stream(tid);
        }
        #endif
    }
    //}

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        fv_ev_swp.deallocate_on_device();
    }
    #endif

    fv_states<true>().swap_backward(0, ctx_.num_fv_states());
}

};
