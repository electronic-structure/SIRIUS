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

/** \file band_uspp.cpp
 *   
 *  \brief Contains ultrasoft pseudopotential related implementation of sirius::Band class.
 */

#include <thread>
#include <mutex>
#include <atomic>
#include "band.h"

namespace sirius {

void Band::uspp_residuals_cpu_parallel_simple(int N__,
                                              int num_bands__,
                                              K_point* kp__,
                                              std::vector<double>& eval__,
                                              dmatrix<double_complex>& evec__,
                                              dmatrix<double_complex>& hphi__,
                                              dmatrix<double_complex>& ophi__,
                                              dmatrix<double_complex>& hpsi__,
                                              dmatrix<double_complex>& opsi__,
                                              dmatrix<double_complex>& res__,
                                              std::vector<double_complex>& h_diag__,
                                              std::vector<double_complex>& o_diag__,
                                              std::vector<double>& res_norm__)

{
    Timer t("sirius::Band::uspp_residuals_cpu_parallel_simple");
    
    Timer t2("sirius::Band::uspp_residuals_cpu_parallel_simple|zgemm");
    /* compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
    /* compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
    linalg<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
    double tval = t2.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("pzgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double norm2 = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
        {
            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        }
        res_norm__[ires] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    
    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
    
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
            if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            res__(igk_row, i) /= z;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        norm2[ires] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}

//== void Band::uspp_cpu_residuals_parallel_v2(int N__,
//==                                           int num_bands__,
//==                                           K_point* kp__,
//==                                           std::vector<double>& eval__,
//==                                           dmatrix<double_complex>& evec__,
//==                                           dmatrix<double_complex>& hphi__,
//==                                           dmatrix<double_complex>& ophi__,
//==                                           dmatrix<double_complex>& hpsi__,
//==                                           dmatrix<double_complex>& opsi__,
//==                                           dmatrix<double_complex>& res__,
//==                                           std::vector<double_complex>& h_diag__,
//==                                           std::vector<double_complex>& o_diag__,
//==                                           std::vector<double>& res_norm__)
//== 
//== {
//==     Timer t("sirius::Band::uspp_cpu_residuals_parallel_v2");
//== 
//==     Timer t1("sirius::Band::uspp_cpu_residuals_parallel_v2|zgemm_eff");
//== 
//==     splindex<block> sub_spl_gkvec(kp__->num_gkvec_row(), kp__->num_ranks_col(), kp__->rank_col());
//== 
//==     mdarray<double_complex, 2> hphi_slice(N__, sub_spl_gkvec.local_size());
//==     hphi__.shuffle_horizontal<_panel_to_slice_>(N__, hphi_slice);
//== 
//==     mdarray<double_complex, 2> ophi_slice(N__, sub_spl_gkvec.local_size());
//==     ophi__.shuffle_horizontal<_panel_to_slice_>(N__, ophi_slice);
//== 
//==     mdarray<double_complex, 2> hpsi_slice(num_bands__, sub_spl_gkvec.local_size());
//==     mdarray<double_complex, 2> opsi_slice(num_bands__, sub_spl_gkvec.local_size());
//== 
//==     dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid());
//==     dmatrix<double_complex>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
//== 
//==     splindex<block_cyclic> spl_bands(num_bands__, kp__->num_ranks_row(), kp__->rank_row(), blacs_grid_.cyclic_block_size());
//== 
//==     mdarray<double_complex, 2> hpsi_slice_tmp(spl_bands.local_size(0), sub_spl_gkvec.local_size());
//==     mdarray<double_complex, 2> opsi_slice_tmp(spl_bands.local_size(0), sub_spl_gkvec.local_size());
//==     
//==     Timer t2("sirius::Band::uspp_cpu_residuals_parallel_v2|zgemm_loop");
//==     for (int irow = 0; irow < kp__->num_ranks_row(); irow++)
//==     {
//==         mdarray<double_complex, 2> evec_tmp(spl_bands.local_size(irow), N__);
//==         evec_tmp.zero();
//==         if (irow == kp__->rank_row())
//==         {
//==             for (int j = 0; j < evec_t.num_cols_local(); j++)
//==             {
//==                 for (int i = 0; i < (int)spl_bands.local_size(irow); i++)
//==                 {
//==                     evec_tmp(i, evec_t.icol(j)) = evec_t(i, j);
//==                 }
//==             }
//==             kp__->comm_col().allreduce(evec_tmp.ptr(), (int)evec_tmp.size());
//==         }
//==         kp__->comm_row().bcast(evec_tmp.ptr(), (int)evec_tmp.size(), irow);
//== 
//==         Timer t3("sirius::Band::uspp_cpu_residuals_parallel_v2|zgemm_loc");
//==         blas<CPU>::gemm(0, 0, (int)spl_bands.local_size(irow), (int)sub_spl_gkvec.local_size(), N__, 
//==                         evec_tmp.ptr(), evec_tmp.ld(), hphi_slice.ptr(), hphi_slice.ld(), 
//==                         hpsi_slice_tmp.ptr(), hpsi_slice_tmp.ld());
//==         
//==         blas<CPU>::gemm(0, 0, (int)spl_bands.local_size(irow), (int)sub_spl_gkvec.local_size(), N__, 
//==                         evec_tmp.ptr(), evec_tmp.ld(), ophi_slice.ptr(), ophi_slice.ld(), 
//==                         opsi_slice_tmp.ptr(), opsi_slice_tmp.ld());
//==         t3.stop();
//==         for (int j = 0; j < (int)sub_spl_gkvec.local_size(); j++)
//==         {
//==             for (int i = 0; i < (int)spl_bands.local_size(irow); i++)
//==             {
//==                 hpsi_slice(spl_bands.global_index(i, irow), j) = hpsi_slice_tmp(i, j);
//==                 opsi_slice(spl_bands.global_index(i, irow), j) = opsi_slice_tmp(i, j);
//==             }
//==         }
//==     }
//==     t2.stop();
//==     
//==     hpsi__.shuffle_horizontal<_slice_to_panel_>(num_bands__, hpsi_slice);
//==     opsi__.shuffle_horizontal<_slice_to_panel_>(num_bands__, opsi_slice);
//==         
//== 
//== 
//==     /* Compute H\Psi_{i} = H\phi_{mu} * Z_{mu, i} */
//==     //blas<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, hphi__, evec__, complex_zero, hpsi__);
//==     /* Compute O\Psi_{i} = O\phi_{mu} * Z_{mu, i} */
//==     //blas<CPU>::gemm(0, 0, kp__->num_gkvec(), num_bands__, N__, complex_one, ophi__, evec__, complex_zero, opsi__);
//==     double tval = t1.stop();
//== 
//==     if (verbosity_level >= 6 && kp__->comm().rank() == 0)
//==     {
//==         printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
//==                kp__->num_gkvec(), num_bands__, N__,
//==                tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
//==     }
//== 
//==     memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
//==     /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
//==     #pragma omp parallel for
//==     for (int i = 0; i < res__.num_cols_local(); i++)
//==     {
//==         int ires = res__.icol(i);
//==         double norm2 = 0;
//==         for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
//==         {
//==             res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
//==             norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
//==         }
//==         res_norm__[ires] = norm2;
//==     }
//==     kp__->comm().allreduce(res_norm__);
//==     
//==     /* compute norm */
//==     for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
//==     
//==     /* apply preconditioner */
//==     #pragma omp parallel for
//==     for (int i = 0; i < res__.num_cols_local(); i++)
//==     {
//==         int ires = res__.icol(i);
//==     
//==         for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
//==         {
//==             double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
//==             if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==             res__(igk_row, i) /= z;
//==         }
//==     }
//==     
//==     std::vector<double> norm2(num_bands__, 0);
//==     /* Normalize new basis functions */
//==     #pragma omp parallel for
//==     for (int i = 0; i < res__.num_cols_local(); i++)
//==     {
//==         int ires = res__.icol(i);
//==         double d = 0;
//==         for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
//==             d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
//==         norm2[ires] = d;
//==     }
//==     kp__->comm().allreduce(norm2);
//==     #pragma omp parallel for
//==     for (int i = 0; i < res__.num_cols_local(); i++)
//==     {
//==         int ires = res__.icol(i);
//==         double d = 1.0 / std::sqrt(norm2[ires]);
//==         for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
//==     }
//== }

#ifdef _SCALAPACK_
void Band::uspp_residuals_cpu_parallel_v3(int N__,
                                          int num_bands__,
                                          K_point* kp__,
                                          std::vector<double>& eval__,
                                          dmatrix<double_complex>& evec__,
                                          dmatrix<double_complex>& hphi__,
                                          dmatrix<double_complex>& ophi__,
                                          dmatrix<double_complex>& hpsi__,
                                          dmatrix<double_complex>& opsi__,
                                          dmatrix<double_complex>& res__,
                                          std::vector<double_complex>& h_diag__,
                                          std::vector<double_complex>& o_diag__,
                                          std::vector<double>& res_norm__,
                                          mdarray<double_complex, 2>& kappa__)
{
    Timer t("sirius::Band::uspp_residuals_cpu_parallel_v3");

    Timer t1("sirius::Band::uspp_residuals_cpu_parallel_v3|zgemm_eff");

    splindex<block_cyclic> spl_num_bands_col(num_bands__, kp__->num_ranks_col(), kp__->rank_col(),
                                             blacs_grid_.cyclic_block_size());
    splindex<block_cyclic> spl_num_bands_row(num_bands__, kp__->num_ranks_row(), kp__->rank_row(),
                                             blacs_grid_.cyclic_block_size());
    
    /* transpose matrix of eigen-vectors */
    dmatrix<double_complex> evec_t(num_bands__, N__, kp__->blacs_grid());
    linalg<CPU>::tranu(num_bands__, N__, evec__, 0, 0, evec_t, 0, 0);
    
    /* local number of basis function |phi> */
    int num_phi_loc = evec_t.num_cols_local();

    mdarray<double_complex, 3> evec_tmp(num_phi_loc, spl_num_bands_col.local_size(0), 2);

    std::array<std::atomic_bool, 2> lock_evec_tmp;
    std::atomic_bool lock_hpsi_tmp;
    std::atomic_bool lock_opsi_tmp;
    for (int i = 0; i < 2; i++) lock_evec_tmp[i].store(false);
    lock_hpsi_tmp.store(false);
    lock_opsi_tmp.store(false);

    mdarray<double_complex, 2> hpsi_tmp(&kappa__(0, 0), kp__->num_gkvec_row(), spl_num_bands_col.local_size(0));
    mdarray<double_complex, 2> opsi_tmp(&kappa__(0, spl_num_bands_col.local_size(0)), kp__->num_gkvec_row(), spl_num_bands_col.local_size(0));

    auto get_evec = [kp__, &spl_num_bands_col, &spl_num_bands_row, &evec_t, &evec_tmp, num_phi_loc](int icol) -> void 
    {
        int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
        memset(&evec_tmp(0, 0, icol % 2), 0, num_phi_loc * num_bands_of_col * sizeof(double_complex));
        for (int i = 0; i < num_bands_of_col; i++)
        {
            int iglob = (int)spl_num_bands_col.global_index(i, icol);
            auto p = spl_num_bands_row.location(iglob); 
            
            if (p.second == kp__->rank_row())
            {
                for (int j = 0; j < num_phi_loc; j++) evec_tmp(j, i, icol % 2) = evec_t(p.first, j);
            }
        }
        kp__->comm_row().allreduce(&evec_tmp(0, 0, icol % 2), num_phi_loc * num_bands_of_col);
    };

    /* get evec for first column */
    get_evec(0);
    lock_evec_tmp[0].store(true);
    
    /* communication thread */
    std::thread comm_thread([kp__, &lock_evec_tmp, &lock_hpsi_tmp, &hpsi_tmp, &hpsi__, 
                             &lock_opsi_tmp, &opsi_tmp, &opsi__, &spl_num_bands_col, get_evec]()
    {
        for (int icol = 0; icol < kp__->num_ranks_col(); icol++)
        {
            int num_bands_of_col = (int)spl_num_bands_col.local_size(icol);
            if (icol + 1 < kp__->num_ranks_col())
            {
                while (lock_evec_tmp[(icol + 1) % 2].load());
                get_evec(icol + 1);
                lock_evec_tmp[(icol + 1) % 2].store(true);
            }
            
            while (!lock_hpsi_tmp.load());
            kp__->comm_col().reduce(hpsi_tmp.ptr(), &hpsi__(0, 0), kp__->num_gkvec_row() * num_bands_of_col, icol);
            lock_hpsi_tmp.store(false);

            while (!lock_opsi_tmp.load());
            kp__->comm_col().reduce(opsi_tmp.ptr(), &opsi__(0, 0), kp__->num_gkvec_row() * num_bands_of_col, icol);
            lock_opsi_tmp.store(false);
        }
    });

    int nthread = omp_get_max_threads();
    if (nthread > 1) omp_set_num_threads(nthread - 1);

    for (int rank_col = 0; rank_col < kp__->num_ranks_col(); rank_col++)
    {
        int num_bands_of_rank = (int)spl_num_bands_col.local_size(rank_col);
        
        while (!lock_evec_tmp[rank_col % 2].load());
        
        while (lock_hpsi_tmp.load());
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                          &hphi__(0, 0), hphi__.ld(), &evec_tmp(0, 0, rank_col % 2), evec_tmp.ld(),
                          &hpsi_tmp(0, 0), hpsi_tmp.ld());
        lock_hpsi_tmp.store(true);
       
        while (lock_opsi_tmp.load());
        linalg<CPU>::gemm(0, 0, kp__->num_gkvec_row(), num_bands_of_rank, num_phi_loc, 
                          &ophi__(0, 0), ophi__.ld(), &evec_tmp(0, 0, rank_col % 2), evec_tmp.ld(),
                          &opsi_tmp(0, 0), opsi_tmp.ld());
        lock_opsi_tmp.store(true);

        lock_evec_tmp[rank_col % 2].store(false);
    }
    comm_thread.join();
    
    omp_set_num_threads(nthread);

    double tval = t1.stop();

    if (verbosity_level >= 6 && kp__->comm().rank() == 0)
    {
        printf("effective zgemm #6&7 with M, N, K: %6i %6i %6i,                        %12.4f sec, %12.4f GFlops/node\n",
               kp__->num_gkvec(), num_bands__, N__,
               tval, 2 * 8e-9 * kp__->num_gkvec() * num_bands__ * N__ / tval / kp__->num_ranks());
    }

    memset(&res_norm__[0], 0, num_bands__ * sizeof(double));
    /* compute residuals r_{i} = H\Psi_{i} - E_{i}O\Psi_{i} and norm squared */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double norm2 = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
        {
            res__(igk_row, i) = hpsi__(igk_row, i) - eval__[ires] * opsi__(igk_row, i);
            norm2 += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        }
        res_norm__[ires] = norm2;
    }
    kp__->comm().allreduce(res_norm__);
    
    /* compute norm */
    for (int i = 0; i < num_bands__; i++) res_norm__[i] = std::sqrt(res_norm__[i]);
    
    /* apply preconditioner */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
    
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++)
        {
            double_complex z = h_diag__[igk_row] - eval__[ires] * o_diag__[igk_row];
            if (std::abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
            res__(igk_row, i) /= z;
        }
    }
    
    std::vector<double> norm2(num_bands__, 0);
    /* Normalize new basis functions */
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 0;
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) 
            d += real(conj(res__(igk_row, i)) * res__(igk_row, i));
        norm2[ires] = d;
    }
    kp__->comm().allreduce(norm2);
    #pragma omp parallel for
    for (int i = 0; i < res__.num_cols_local(); i++)
    {
        int ires = res__.icol(i);
        double d = 1.0 / std::sqrt(norm2[ires]);
        for (int igk_row = 0; igk_row < kp__->num_gkvec_row(); igk_row++) res__(igk_row, i) *= d;
    }
}
#endif

#ifdef _GPU_
//== void Band::diag_fv_uspp_gpu(K_point* kp, Periodic_function<double>* effective_potential)
//== {
//==     //== Timer t("sirius::Band::diag_fv_uspp_gpu");
//== 
//==     //== // map effective potential to a corase grid
//==     //== std::vector<double> veff_it_coarse(parameters_.reciprocal_lattice()->fft_coarse()->size());
//==     //== std::vector<double_complex> veff_pw_coarse(parameters_.reciprocal_lattice()->num_gvec_coarse());
//== 
//==     //== // take only first num_gvec_coarse plane-wave harmonics; this is enough to apply V_eff to \Psi
//==     //== for (int igc = 0; igc < parameters_.reciprocal_lattice()->num_gvec_coarse(); igc++)
//==     //== {
//==     //==     int ig = parameters_.reciprocal_lattice()->gvec_index(igc);
//==     //==     veff_pw_coarse[igc] = effective_potential->f_pw(ig);
//==     //== }
//==     //== parameters_.reciprocal_lattice()->fft_coarse()->input(parameters_.reciprocal_lattice()->num_gvec_coarse(), 
//==     //==                                                       parameters_.reciprocal_lattice()->fft_index_coarse(),
//==     //==                                                       &veff_pw_coarse[0]);
//==     //== parameters_.reciprocal_lattice()->fft_coarse()->transform(1);
//==     //== parameters_.reciprocal_lattice()->fft_coarse()->output(&veff_it_coarse[0]);
//== 
//==     //== // short notation for target wave-functions
//==     //== mdarray<double_complex, 2>& psi = kp->fv_states();
//== 
//==     //== // short notation for number of target wave-functions
//==     //== int num_bands = parameters_.num_fv_states();     
//== 
//==     //== // cache kinetic energy,
//==     //== std::vector<double> pw_ekin = kp->get_pw_ekin();
//== 
//==     //== // get diagonal elements for preconditioning
//==     //== std::vector<double_complex> h_diag;
//==     //== std::vector<double_complex> o_diag;
//==     //== get_h_o_diag(kp, effective_potential, pw_ekin, h_diag, o_diag);
//==     //== 
//==     //== int max_iter = 10;
//==     //== int num_phi = std::min(4 * num_bands, kp->num_gkvec());
//== 
//==     //== // big array to store one of the beta, phi, hphi or ophi on the GPU
//==     //== mdarray<double_complex, 2> gamma(NULL, kp->num_gkvec(), std::max(num_phi, parameters_.unit_cell()->mt_lo_basis_size()));
//==     //== gamma.allocate_on_device();
//== 
//==     //== // small array to store residuals, wave-functions and updates to hphi and ophi
//==     //== mdarray<double_complex, 2> kappa(kp->num_gkvec(), 2 * num_bands);
//==     //== kappa.allocate_on_device();
//==     //== kappa.pin_memory();
//== 
//==     //== mdarray<double_complex, 2> phi(kp->num_gkvec(), num_phi);
//==     //== mdarray<double_complex, 2> hphi(kp->num_gkvec(), num_phi);
//==     //== mdarray<double_complex, 2> ophi(kp->num_gkvec(), num_phi);
//==     //== 
//==     //== mdarray<double_complex, 2> hmlt(num_phi, num_phi);
//==     //== mdarray<double_complex, 2> ovlp(num_phi, num_phi);
//==     //== mdarray<double_complex, 2> hmlt_old(num_phi, num_phi);
//==     //== mdarray<double_complex, 2> ovlp_old(num_phi, num_phi);
//==     //== 
//==     //== mdarray<double_complex, 2> evec(num_phi, num_phi);
//==     //== evec.allocate_on_device();
//==     //== evec.pin_memory();
//== 
//==     //== std::vector<double> eval(num_bands);
//==     //== std::vector<double> eval_old(num_bands, 1e100);
//==     //== 
//==     //== std::vector<double> res_norm(num_bands); // norm of residuals
//==     //== std::vector<double> res_rms(num_bands); // RMS of residual
//==     //== std::vector<double> res_e(num_bands);
//==     //== 
//==     //== if (parameters_.gen_evp_solver()->type() == ev_magma)
//==     //== {
//==     //==     hmlt.pin_memory();
//==     //==     ovlp.pin_memory();
//==     //== }
//== 
//==     //== bool convergence_by_energy = false;
//== 
//==     //== int N = 0; // current eigen-value problem size
//==     //== int n = num_bands; // number of added residuals
//== 
//==     //== // trial basis functions
//==     //== assert(phi.size(0) == psi.size(0));
//==     //== for (int i = 0; i < num_bands; i++) memcpy(&phi(0, i), &psi(0, i), kp->num_gkvec() * sizeof(double_complex));
//==     //== 
//==     //== // start iterative diagonalization
//==     //== for (int k = 0; k < max_iter; k++)
//==     //== {
//==     //==     Timer t1("sirius::Band::diag_fv_uspp_gpu|set_gevp");
//== 
//==     //==     // copy old Hamiltonian and overlap
//==     //==     for (int i = 0; i < N; i++)
//==     //==     {
//==     //==         memcpy(&hmlt(0, i), &hmlt_old(0, i), N * sizeof(double_complex));
//==     //==         memcpy(&ovlp(0, i), &ovlp_old(0, i), N * sizeof(double_complex));
//==     //==     }
//== 
//==     //==     // apply Hamiltonian and overlap operators
//==     //==     apply_h_o_uspp_gpu(kp, veff_it_coarse, pw_ekin, n, gamma, kappa, &phi(0, N), &hphi(0, N), &ophi(0, N));
//== 
//==     //==     // copy all phi to GPU
//==     //==     cublas_set_matrix(kp->num_gkvec(), n + N, sizeof(double_complex), phi.ptr(), phi.ld(), gamma.ptr_device(), gamma.ld());
//== 
//==     //==     // temporary storage for Hamiltonian and overlap 
//==     //==     mdarray<double_complex, 2> tmp(NULL, N + n, n);
//==     //==     tmp.allocate_on_device();
//== 
//==     //==     // compute the Hamiltonian matrix: <phi|H|phi>
//==     //==     blas<GPU>::gemm(2, 0, N + n, n, kp->num_gkvec(), gamma.ptr_device(), gamma.ld(), 
//==     //==                     kappa.ptr_device(), kappa.ld(), tmp.ptr_device(), tmp.ld());
//== 
//==     //==     cublas_get_matrix(N + n, n, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), &hmlt(0, N), hmlt.ld());
//== 
//==     //==     // compute overlap matrix <phi|O|phi>
//==     //==     blas<GPU>::gemm(2, 0, N + n, n, kp->num_gkvec(), gamma.ptr_device(), gamma.ld(), 
//==     //==                     kappa.ptr_device(0, n), kappa.ld(), tmp.ptr_device(), tmp.ld());
//== 
//==     //==     cublas_get_matrix(N + n, n, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), &ovlp(0, N), ovlp.ld());
//== 
//==     //==     tmp.deallocate_on_device();
//== 
//==     //==     // MAGMA works with lower triangular part
//==     //==     #ifdef _MAGMA_
//==     //==     for (int i = 0; i < N; i++)
//==     //==     {
//==     //==         for (int j = N; j < N + n; j++)
//==     //==         {
//==     //==             hmlt(j, i) = conj(hmlt(i, j));
//==     //==             ovlp(j, i) = conj(ovlp(i, j));
//==     //==         }
//==     //==     }
//==     //==     #endif
//== 
//==     //==     // increase the size of the variation space
//==     //==     N += n;
//== 
//==     //==     // save Hamiltonian and overlap
//==     //==     for (int i = 0; i < N; i++)
//==     //==     {
//==     //==         memcpy(&hmlt_old(0, i), &hmlt(0, i), N * sizeof(double_complex));
//==     //==         memcpy(&ovlp_old(0, i), &ovlp(0, i), N * sizeof(double_complex));
//==     //==     }
//==     //==     t1.stop();
//==     //==     
//==     //==     Timer t2("sirius::Band::diag_fv_uspp_gpu|solve_gevp");
//==     //==     parameters_.gen_evp_solver()->solve(N, num_bands, num_bands, num_bands, hmlt.ptr(), hmlt.ld(), ovlp.ptr(), ovlp.ld(), 
//==     //==                                         &eval[0], evec.ptr(), evec.ld());
//==     //==     t2.stop();
//== 
//==     //==     Timer t3("sirius::Band::diag_fv_uspp_gpu|residuals");
//==     //==     /* Quantum Espresso way of estimating basis update: residuals for which |e - e_old| > eps 
//==     //==        are accepted as the additional basis functions */
//==     //==     if (convergence_by_energy)
//==     //==     {
//==     //==         //== if (k == 0)
//==     //==         //== {
//==     //==         //==     n = num_bands;
//==     //==         //==     memcpy(&hmlt(0, 0), &evec(0, 0), N * n * sizeof(double_complex));
//==     //==         //== }
//==     //==         //== else
//==     //==         //== {
//==     //==         //==     n = 0;
//==     //==         //==     // check eigen-values for convergence
//==     //==         //==     for (int i = 0; i < num_bands; i++)
//==     //==         //==     {
//==     //==         //==         if (fabs(eval[i] - eval_old[i]) > parameters_.iterative_solver_tolerance())
//==     //==         //==         {
//==     //==         //==             res_e[n] = eval[i];
//==     //==         //==             
//==     //==         //==             // use hmlt as a temporary storage for evec
//==     //==         //==             memcpy(&hmlt(0, n), &evec(0, i), N * sizeof(double_complex));
//==  
//==     //==         //==             n++;
//==     //==         //==         }
//==     //==         //==         eval_old[i] = eval[i];
//==     //==         //==     }
//==     //==         //== }
//== 
//==     //==         //== // if we have unconverged eigen-states
//==     //==         //== if (n != 0)
//==     //==         //== {
//==     //==         //==     // compute residuals
//==     //==         //==     // 1. O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
//==     //==         //==     blas<CPU>::gemm(0, 0, kp->num_gkvec(), n, N, &ophi(0, 0), ophi.ld(), &hmlt(0, 0), hmlt.ld(), 
//==     //==         //==                     &res(0, 0), res.ld());
//==     //==         //==     // 2. multiply O\Psi_{i} with energy
//==     //==         //==     for (int i = 0; i < n; i++)
//==     //==         //==     {
//==     //==         //==         for (int igk = 0; igk < kp->num_gkvec(); igk++) res(igk, i) *= eval[i];
//==     //==         //==     }
//==     //==         //==     // 3. r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
//==     //==         //==     blas<CPU>::gemm(0, 0, kp->num_gkvec(), n, N, double_complex(1, 0), &hphi(0, 0), hphi.ld(), 
//==     //==         //==                     &hmlt(0, 0), hmlt.ld(), double_complex(-1, 0), &res(0, 0), res.ld());
//== 
//==     //==         //==     // apply preconditioner
//==     //==         //==     #pragma omp parallel for
//==     //==         //==     for (int i = 0; i < n; i++)
//==     //==         //==     {
//==     //==         //==         // apply preconditioner
//==     //==         //==         for (int igk = 0; igk < kp->num_gkvec(); igk++)
//==     //==         //==         {
//==     //==         //==             //double_complex z = h_diag[igk] - res_e[i] * o_diag[igk];
//==     //==         //==             //if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==     //==         //==             double d = real(h_diag[igk] - res_e[i] * o_diag[igk]);
//==     //==         //==             if (fabs(d) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==     //==         //==             res(igk, i) /= d;
//==     //==         //==         }
//==     //==         //==     }
//==     //==         //== }
//==     //==     }
//==     //==     /* Alternative way to estimate basis update: take residuals with norm > eps */
//==     //==     else
//==     //==     {
//==     //==         cublas_set_matrix(N, N, sizeof(double_complex), evec.ptr(), evec.ld(), evec.ptr_device(), evec.ld());
//== 
//==     //==         // copy all ophi to GPU
//==     //==         cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), ophi.ptr(), ophi.ld(), gamma.ptr_device(), gamma.ld());
//==     //==         
//==     //==         // O\Psi_{i} = O\phi_{mu} * Z_{mu, i}
//==     //==         blas<GPU>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
//==     //==                         evec.ptr_device(), evec.ld(), kappa.ptr_device(), kappa.ld());
//==     //==     
//==     //==         mdarray<double, 1> eval_gpu(&eval[0], num_bands);
//==     //==         eval_gpu.allocate_on_device();
//==     //==         eval_gpu.copy_to_device();
//==     //==         // multiply O\Psi_{i} with energy
//==     //==         scale_matrix_columns_gpu(kp->num_gkvec(), num_bands, kappa.ptr_device(), eval_gpu.ptr_device());
//==     //==         eval_gpu.deallocate_on_device();
//==     //==         
//==     //==         // copy all hphi to GPU
//==     //==         cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), hphi.ptr(), hphi.ld(), gamma.ptr_device(), gamma.ld());
//==     //==         
//==     //==         double_complex zone(1, 0);
//==     //==         double_complex mzone(-1, 0);
//==     //==         // r_{i} = H\Psi_{i} - E_{i}O\Psi_{i}
//==     //==         blas<GPU>::gemm(0, 0, kp->num_gkvec(), num_bands, N, &zone, gamma.ptr_device(), gamma.ld(), 
//==     //==                         evec.ptr_device(), evec.ld(), &mzone, kappa.ptr_device(), kappa.ld());
//==     //==        
//==     //==         // copy residuals to the host memory
//==     //==         cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex), kappa.ptr_device(), kappa.ld(), 
//==     //==                           kappa.ptr(), kappa.ld());
//== 
//==     //==         Timer t("sirius::Band::diag_fv_uspp_gpu|residuals|cpu_part");
//==     //==         // compute norm and apply preconditioner
//==     //==         #pragma omp parallel for
//==     //==         for (int i = 0; i < num_bands; i++)
//==     //==         {
//==     //==             double r = 0;
//==     //==             for (int igk = 0; igk < kp->num_gkvec(); igk++) r += real(conj(kappa(igk, i)) * kappa(igk, i));
//==     //==             res_norm[i] = r;
//==     //==             res_rms[i] = sqrt(r / kp->num_gkvec());
//==     //==             
//==     //==             // apply preconditioner
//==     //==             for (int igk = 0; igk < kp->num_gkvec(); igk++)
//==     //==             {
//==     //==                 double_complex z = h_diag[igk] - eval[i] * o_diag[igk];
//==     //==                 if (abs(z) < 1e-12) error_local(__FILE__, __LINE__, "problematic division");
//==     //==                 kappa(igk, i) /= z;
//==     //==             }
//==     //==         }
//== 
//==     //==         // check which residuals are converged
//==     //==         n = 0;
//==     //==         std::vector< std::pair<double, int> > res_rms_sorted;
//==     //==         for (int i = 0; i < num_bands; i++)
//==     //==         {
//==     //==             res_rms_sorted.push_back(std::pair<double, int>(res_rms[i], i));
//== 
//==     //==             // take the residual if it's norm is above the threshold
//==     //==             if (res_rms[i] > parameters_.iterative_solver_tolerance()) n++;
//==     //==         }
//==     //==         
//==     //==         if (n > 0 && n < num_bands)
//==     //==         {
//==     //==             n = std::max(n, (num_bands - 1) / (k + 1));
//== 
//==     //==             std::sort(res_rms_sorted.begin(), res_rms_sorted.end());
//== 
//==     //==             double tol = res_rms_sorted[num_bands - n].first;
//== 
//==     //==             n = 0;
//==     //==             for (int i = 0; i < num_bands; i++)
//==     //==             {
//==     //==                 // take the residual if it's norm is above the threshold
//==     //==                 if (res_rms[i] > tol) 
//==     //==                 {
//==     //==                     // shift unconverged residuals to the beginning of array
//==     //==                     if (n != i) memcpy(&kappa(0, n), &kappa(0, i), kp->num_gkvec() * sizeof(double_complex));
//==     //==                     n++;
//==     //==                 }
//==     //==             }
//==     //==         }
//==     //==         t.stop();
//==  
//==     //==         //== n = 0;
//==     //==         //== for (int i = 0; i < num_bands; i++)
//==     //==         //== {
//==     //==         //==     // take the residual if it's norm is above the threshold
//==     //==         //==     if (res_rms[i] > parameters_.iterative_solver_tolerance()) 
//==     //==         //==     {
//==     //==         //==         // shift unconverged residuals to the beginning of array
//==     //==         //==         if (n != i) memcpy(&res(0, n), &res(0, i), kp->num_gkvec() * sizeof(double_complex));
//==     //==         //==         n++;
//==     //==         //==     }
//==     //==         //== }
//==     //==     }
//==     //==     t3.stop();
//== 
//==     //==     //if (Platform::mpi_rank() == 0)
//==     //==     //{
//==     //==     //    std::cout << "iteration:" << k << ", current eigen-value size = " << N << ", number of added residuals = " << n << std::endl;
//==     //==     //    //printf("lower and upper eigen-values : %16.8f %16.8f\n", eval[0], eval[num_bands - 1]);
//==     //==     //}
//== 
//==     //==     // check if we run out of variational space or eigen-vectors are converged or it's a last iteration
//==     //==     if (N + n > num_phi || n == 0 || k == (max_iter - 1))
//==     //==     {   
//==     //==         Timer t3("sirius::Band::diag_fv_uspp_gpu|update_phi");
//==     //==         // copy all phi to GPU
//==     //==         cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), phi.ptr(), phi.ld(), 
//==     //==                           gamma.ptr_device(), gamma.ld());
//==     //==         // \Psi_{i} = \phi_{mu} * Z_{mu, i}
//==     //==         blas<GPU>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
//==     //==                         evec.ptr_device(), evec.ld(), kappa.ptr_device(), kappa.ld());
//== 
//==     //==         cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex), 
//==     //==                           kappa.ptr_device(), kappa.ld(), psi.ptr(), psi.ld());
//==     //==         t3.stop();
//== 
//==     //==         if (n == 0 || k == (max_iter - 1)) // exit the loop if the eigen-vectors are converged or it's a last iteration
//==     //==         {
//==     //==             std::cout << "converged in " << k << " iterations" << std::endl;
//==     //==             break;
//==     //==         }
//==     //==         else // otherwise set \Psi as a new trial basis and update related arrays
//==     //==         {
//==     //==             Timer t("sirius::Band::diag_fv_uspp_gpu|update_h_o");
//== 
//==     //==             // temporary storage for Hamiltonian and overlap 
//==     //==             mdarray<double_complex, 2> tmp(NULL, num_bands, num_bands);
//==     //==             tmp.allocate_on_device();
//== 
//==     //==             // compute H\Psi
//==     //==             cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), hphi.ptr(), hphi.ld(), 
//==     //==                               gamma.ptr_device(), gamma.ld());
//== 
//==     //==             blas<GPU>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
//==     //==                             evec.ptr_device(), evec.ld(), kappa.ptr_device(0, num_bands), kappa.ld());
//==     //==             
//==     //==             // copy H\Psi to host memory
//==     //==             cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex),
//==     //==                               kappa.ptr_device(0, num_bands), kappa.ld(), hphi.ptr(), hphi.ld());
//== 
//==     //==             // compute the Hamiltonian matrix: <Psi|H|Psi>
//==     //==             blas<GPU>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), kappa.ptr_device(), kappa.ld(), 
//==     //==                             kappa.ptr_device(0, num_bands), kappa.ld(), tmp.ptr_device(), tmp.ld());
//== 
//==     //==             // copy Hamiltonian to host
//==     //==             cublas_get_matrix(num_bands, num_bands, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), 
//==     //==                               hmlt_old.ptr(), hmlt_old.ld());
//==     //==             
//==     //==             // compute O\Psi
//==     //==             cublas_set_matrix(kp->num_gkvec(), N, sizeof(double_complex), ophi.ptr(), ophi.ld(), 
//==     //==                               gamma.ptr_device(), gamma.ld());
//== 
//==     //==             blas<GPU>::gemm(0, 0, kp->num_gkvec(), num_bands, N, gamma.ptr_device(), gamma.ld(), 
//==     //==                             evec.ptr_device(), evec.ld(), kappa.ptr_device(0, num_bands), kappa.ld());
//== 
//==     //==             // copy O\Psi to host memory
//==     //==             cublas_get_matrix(kp->num_gkvec(), num_bands, sizeof(double_complex),
//==     //==                               kappa.ptr_device(0, num_bands), kappa.ld(), ophi.ptr(), ophi.ld());
//== 
//==     //==             // compute the overlap matrix: <Psi|O|Psi>
//==     //==             blas<GPU>::gemm(2, 0, num_bands, num_bands, kp->num_gkvec(), kappa.ptr_device(), kappa.ld(), 
//==     //==                             kappa.ptr_device(0, num_bands), kappa.ld(), tmp.ptr_device(), tmp.ld());
//== 
//==     //==             // copy overlap matrix to host
//==     //==             cublas_get_matrix(num_bands, num_bands, sizeof(double_complex), tmp.ptr_device(), tmp.ld(), 
//==     //==                               ovlp_old.ptr(), ovlp_old.ld());
//==     //==          
//==     //==             // update phi with Psi
//==     //==             memcpy(phi.ptr(), psi.ptr(), num_bands * kp->num_gkvec() * sizeof(double_complex));
//== 
//==     //==             // new size of eigen-value problem 
//==     //==             N = num_bands;
//== 
//==     //==             tmp.deallocate_on_device();
//==     //==         }
//==     //==     }
//==     //==     // expand variational space with new preconditioned residuals
//==     //==     memcpy(&phi(0, N), &kappa(0, 0), n * kp->num_gkvec() * sizeof(double_complex));
//==     //== }
//== 
//==     //== kp->set_fv_eigen_values(&eval[0]);
//== }
#endif

}
