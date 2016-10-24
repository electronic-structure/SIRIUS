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

/** \file generate_fv_states.hpp
 *   
 *  \brief Contains implementation of sirius::K_point::generate_fv_states method.
 */

inline void K_point::generate_fv_states()
{
    PROFILE_WITH_TIMER("sirius::K_point::generate_fv_states");
    
    if (!ctx_.full_potential()) {
        return;
    }

    //mdarray<double_complex, 2> pw_coeffs;
    //mdarray<double_complex, 2> mt_coeffs;

    //STOP();
    //
    //int nbnd_loc;
    ///* in both cases eigen-vectors are redistributed to the same "full column" storage */
    ////if (ctx_.iterative_solver_input_section().type_ == "exact") {
    ////    fv_eigen_vectors_->remap_forward(0, ctx_.num_fv_states());
    ////    /* local number of bands */
    ////    nbnd_loc = fv_eigen_vectors_->spl_num_col().local_size();
    ////    
    ////    if (nbnd_loc) {
    ////        pw_coeffs = mdarray<double_complex, 2>(fv_eigen_vectors_->extra().at<CPU>(), gklo_basis_size(), nbnd_loc);
    ////        mt_coeffs = mdarray<double_complex, 2>(fv_eigen_vectors_->extra().at<CPU>(num_gkvec(), 0), gklo_basis_size(), nbnd_loc);
    ////    }

    ////} else {
    //    fv_eigen_vectors_slab_->remap_to_full_column_distr(ctx_.num_fv_states());
    //    assert(fv_eigen_vectors_slab_->pw_coeffs().spl_num_col().local_size() ==
    //           fv_eigen_vectors_slab_->mt_coeffs().spl_num_col().local_size());
    //    /* local number of bands */
    //    nbnd_loc = fv_eigen_vectors_slab_->pw_coeffs().spl_num_col().local_size();
    //    if (nbnd_loc) {
    //        pw_coeffs = mdarray<double_complex, 2>(fv_eigen_vectors_slab_->pw_coeffs().extra().at<CPU>(), num_gkvec(), nbnd_loc);
    //        mt_coeffs = mdarray<double_complex, 2>(fv_eigen_vectors_slab_->mt_coeffs().extra().at<CPU>(), unit_cell_.mt_lo_basis_size(), nbnd_loc);
    //    }
    ////}

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        STOP();
        //pw_coeffs.allocate(memory_t::device);
        //pw_coeffs.copy_to_device();
    }
    #endif

    //fv_states().prepare_full_column_distr(ctx_.num_fv_states());

    //assert(nbnd_loc == fv_states().pw_coeffs().spl_num_col().local_size());
    //assert(nbnd_loc == fv_states().mt_coeffs().spl_num_col().local_size());

    //#pragma omp parallel
    //{
        /* get thread id */
        #ifdef __GPU
        int tid = omp_get_thread_num();
        #endif
        mdarray<double_complex, 2> alm(num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size(), memory_t::host_pinned);
        mdarray<double_complex, 2> tmp(unit_cell_.max_mt_aw_basis_size(), ctx_.num_fv_states());

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            alm.allocate(memory_t::device);
            tmp.allocate(memory_t::device); // = mdarray<double_complex, 2>(unit_cell_.max_mt_aw_basis_size(), nbnd_loc, memory_t::device);
        }
        #endif
        
        //#pragma omp for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            /* number of alm coefficients for atom */
            int mt_aw_size = unit_cell_.atom(ia).mt_aw_basis_size();
            int mt_lo_size = unit_cell_.atom(ia).mt_lo_basis_size();
            /* generate matching coefficients for all G-vectors */
            alm_coeffs_loc_->generate(ia, alm);

            mdarray<double_complex, 2> tmp1(tmp.at<CPU>(), mt_aw_size, ctx_.num_fv_states());
            linalg<CPU>::gemm(1, 0, mt_aw_size, ctx_.num_fv_states(), num_gkvec_loc(),
                              alm.at<CPU>(), alm.ld(),
                              fv_eigen_vectors_slab().pw_coeffs().prime().at<CPU>(), fv_eigen_vectors_slab().pw_coeffs().prime().ld(),
                              tmp1.at<CPU>(), tmp1.ld());
            auto location = fv_eigen_vectors_slab().spl_num_atoms().location(ia);
            comm_.reduce(tmp1.at<CPU>(), static_cast<int>(tmp1.size()), location.rank);

            if (location.rank == comm_.rank()) {
                int offset1 = fv_states().offset_mt_coeffs(location.local_index);
                int offset2 = fv_eigen_vectors_slab().offset_mt_coeffs(location.local_index);
                for (int i = 0; i < ctx_.num_fv_states(); i++) {
                    /* aw block */
                    std::memcpy(fv_states().mt_coeffs().prime().at<CPU>(offset1, i),
                                tmp1.at<CPU>(0, i),
                                mt_aw_size * sizeof(double_complex));
                    /* lo block */
                    std::memcpy(fv_states().mt_coeffs().prime().at<CPU>(offset1 + mt_aw_size, i),
                                fv_eigen_vectors_slab().mt_coeffs().prime().at<CPU>(offset2, i),
                                mt_lo_size * sizeof(double_complex));
                }
            }

            ///* compute F(lm, i) = A(lm, G)^{T} * evec(G, i) for a single atom */
            //if (nbnd_loc) {
            //    if (ctx_.processing_unit() == CPU) {
            //        /* multiply eigen-vectors and matching coefficients */
            //        linalg<CPU>::gemm(1, 0, mt_aw_size, nbnd_loc, num_gkvec(),
            //                          alm.at<CPU>(), alm.ld(),
            //                          pw_coeffs.at<CPU>(), pw_coeffs.ld(),
            //                          fv_states().mt_coeffs().extra().at<CPU>(offset_wf, 0), fv_states().mt_coeffs().extra().ld());
            //    }
            //    #ifdef __GPU
            //    if (ctx_.processing_unit() == GPU) {
            //        /* multiply eigen-vectors and matching coefficients */
            //        alm.async_copy_to_device(tid);
            //        linalg<GPU>::gemm(1, 0, mt_aw_size, nbnd_loc, num_gkvec(),
            //                          alm.at<GPU>(), alm.ld(),
            //                          pw_coeffs.at<GPU>(), pw_coeffs.ld(),
            //                          tmp.at<GPU>(), tmp.ld(),
            //                          tid);
            //        acc::copyout(fv_states().mt_coeffs().extra().at<CPU>(offset_wf, 0), fv_states().mt_coeffs().extra().ld(),
            //                     tmp.at<GPU>(), tmp.ld(),
            //                     mt_aw_size, nbnd_loc, tid);
            //        acc::sync_stream(tid);
            //    }
            //    #endif
            //}

            //for (int i = 0; i < nbnd_loc; i++) {
            //    /* lo block */
            //    std::memcpy(fv_states().mt_coeffs().extra().at<CPU>(offset_wf + mt_aw_size, i),
            //                mt_coeffs.at<CPU>(unit_cell_.atom(ia).offset_lo(), i),
            //                unit_cell_.atom(ia).mt_lo_basis_size() * sizeof(double_complex));
            //}
        }
        //#pragma omp for
        for (int i = 0; i < ctx_.num_fv_states(); i++) {
            /* G+k block */
            std::memcpy(fv_states().pw_coeffs().prime().at<CPU>(0, i),
                        fv_eigen_vectors_slab().pw_coeffs().prime().at<CPU>(0, i),
                        num_gkvec_loc() * sizeof(double_complex));
        }
    //}

    //fv_states().remap_to_prime_distr(ctx_.num_fv_states());
}
