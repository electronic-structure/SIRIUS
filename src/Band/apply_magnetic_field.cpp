// Copyright (c) 2013-2015 Anton Kozhevnikov, Thomas Schulthess
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

/** \file apply_magnetic_field.cpp
 *
 *  \brief Contains the implementation of Band::apply_magnetic_field() function.
 */

#include <thread>
#include <mutex>
#include "band.h"

namespace sirius {

void Band::apply_magnetic_field(dmatrix<double_complex>& fv_states__,
                                int num_gkvec__,
                                int const* fft_index__, 
                                Periodic_function<double>* effective_magnetic_field__[3],
                                std::vector< dmatrix<double_complex> >& hpsi__)
{
    PROFILE();

    assert(hpsi__.size() >= 2);
    for (int i = 0; i < (int)hpsi__.size(); i++)
    {
        assert(fv_states__.num_rows_local() == hpsi__[i].num_rows_local());
        assert(fv_states__.num_cols_local() == hpsi__[i].num_cols_local());
    }

    int nfv = fv_states__.num_cols_local();

    Timer t("sirius::Band::apply_magnetic_field");

    mdarray<double_complex, 3> zm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                  parameters_.num_mag_dims());

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        int offset = unit_cell_.atom(ia)->offset_wf();
        int mt_basis_size = unit_cell_.atom(ia)->type()->mt_basis_size();
        Atom* atom = unit_cell_.atom(ia);
        
        zm.zero();
        
        /* only upper triangular part of zm is computed because it is a hermitian matrix */
        #pragma omp parallel for default(shared)
        for (int j2 = 0; j2 < mt_basis_size; j2++)
        {
            int lm2 = atom->type()->indexb(j2).lm;
            int idxrf2 = atom->type()->indexb(j2).idxrf;
            
            for (int i = 0; i < parameters_.num_mag_dims(); i++)
            {
                for (int j1 = 0; j1 <= j2; j1++)
                {
                    int lm1 = atom->type()->indexb(j1).lm;
                    int idxrf1 = atom->type()->indexb(j1).idxrf;

                    zm(j1, j2, i) = gaunt_coefs_->sum_L3_gaunt(lm1, lm2, atom->b_radial_integrals(idxrf1, idxrf2, i)); 
                }
            }
        }
        /* compute bwf = B_z*|wf_j> */
        linalg<CPU>::hemm(0, 0, mt_basis_size, nfv, complex_one, &zm(0, 0, 0), zm.ld(), 
                          &fv_states__(offset, 0), fv_states__.ld(), complex_zero, &hpsi__[0](offset, 0), hpsi__[0].ld());
        
        /* compute bwf = (B_x - iB_y)|wf_j> */
        if (hpsi__.size() >= 3)
        {
            /* reuse first (z) component of zm matrix to store (Bx - iBy) */
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) - complex_i * zm(j1, j2, 2);
                
                /* remember: zm is hermitian and we computed only the upper triangular part */
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) - complex_i * conj(zm(j2, j1, 2));
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                              &fv_states__(offset, 0), fv_states__.ld(), &hpsi__[2](offset, 0), hpsi__[2].ld());
        }
        
        /* compute bwf = (B_x + iB_y)|wf_j> */
        if (hpsi__.size() == 4 && std_evp_solver()->parallel())
        {
            /* reuse first (z) component of zm matrix to store (Bx + iBy) */
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) + complex_i * zm(j1, j2, 2);
                
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = std::conj(zm(j2, j1, 1)) + complex_i * std::conj(zm(j2, j1, 2));
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                              &fv_states__(offset, 0), fv_states__.ld(), &hpsi__[2](offset, 0), hpsi__[2].ld());
        }
    }
    
    //int num_fft_threads = -1;
    //switch (parameters_.processing_unit())
    //{
    //    case CPU:
    //    {
    //        num_fft_threads = parameters_.num_fft_threads();
    //        break;
    //    }
    //    case GPU:
    //    {
    //        num_fft_threads = std::min(parameters_.num_fft_threads() + 1, Platform::max_num_threads());
    //        break;
    //    }
    //}
    /* index of first-variational state */
    //int idx_psi = 0;
    //std::mutex idx_psi_mutex;

    Timer t1("sirius::Band::apply_magnetic_field|it");

    //int wf_pw_offset = unit_cell_.mt_basis_size();
    //auto fft = fft_;
    #ifdef __GPU
    auto fft_gpu = ctx_.fft_gpu();
    #endif
    //auto step_function = ctx_.step_function();

    STOP();
    
//    if (parameters_.processing_unit() == GPU)
//    {
//        #ifdef __GPU
//        fv_states__.allocate_on_device();
//        fv_states__.copy_to_device();
//        #endif
//    }
//
//    std::vector<std::thread> thread_workers;
//
//    for (int thread_id = 0; thread_id < num_fft_threads; thread_id++)
//    {
//        if (thread_id == (num_fft_threads - 1) && num_fft_threads > 1 && parameters_.processing_unit() == GPU)
//        {
//            #ifdef __GPU
//            thread_workers.push_back(std::thread([thread_id, &idx_psi, &idx_psi_mutex, nfv, num_gkvec__, wf_pw_offset,
//                                                  fft_gpu, fft_index__, &fv_states__, &hpsi__, step_function,
//                                                  effective_magnetic_field__]()
//            {
//                Timer t("sirius::Band::apply_magnetic_field|it_gpu");
//
//                /* move fft index to GPU */
//                mdarray<int, 1> fft_index_gpu(const_cast<int*>(fft_index__), num_gkvec__);
//                fft_index_gpu.allocate_on_device();
//                fft_index_gpu.copy_to_device();
//
//                int nfft_max = fft_gpu->num_fft();
// 
//                /* allocate work area array */
//                mdarray<char, 1> work_area(nullptr, fft_gpu->work_area_size());
//                work_area.allocate_on_device();
//                fft_gpu->set_work_area_ptr(work_area.at<GPU>());
//                
//                /* allocate space for plane-wave expansion coefficients */
//                mdarray<double_complex, 2> psi_pw_gpu(nullptr, num_gkvec__, nfft_max); 
//                psi_pw_gpu.allocate_on_device();
//                
//                /* allocate space for real-space grid */
//                mdarray<double_complex, 2> psi_it_gpu(nullptr, fft_gpu->size(), nfft_max);
//                psi_it_gpu.allocate_on_device();
//                
//                /* effecive field multiplied by step function */
//                mdarray<double, 1> beff_gpu(fft_gpu->size());
//                for (int ir = 0; ir < (int)fft_gpu->size(); ir++)
//                    beff_gpu(ir) = effective_magnetic_field__[0]->f_it(ir) * step_function->theta_r(ir);
//                beff_gpu.allocate_on_device();
//                beff_gpu.copy_to_device();
//                
//                bool done = false;
//
//                while (!done)
//                {
//                    idx_psi_mutex.lock();
//                    int i = idx_psi;
//                    if (idx_psi + nfft_max > nfv) 
//                    {
//                        done = true;
//                    }
//                    else
//                    {
//                        idx_psi += nfft_max;
//                    }
//                    idx_psi_mutex.unlock();
//
//                    if (!done)
//                    {
//                        if (hpsi__.size() >= 3) STOP(); // need to implement this
//
//                        /* copy pw coeffs to GPU */
//                        //mdarray<double_complex, 1>(&fv_states(wf_pw_offset, i), psi_pw_gpu.at<GPU>(), num_gkvec).copy_to_device();
//                        cuda_copy_device_to_device(psi_pw_gpu.at<GPU>(), fv_states__.at<GPU>(wf_pw_offset, i), num_gkvec__ * sizeof(double_complex));
//
//                        fft_gpu->batch_load(num_gkvec__, fft_index_gpu.at<GPU>(), psi_pw_gpu.at<GPU>(), psi_it_gpu.at<GPU>());
//
//                        fft_gpu->transform(1, psi_it_gpu.at<GPU>());
//
//                        scale_matrix_rows_gpu(fft_gpu->size(), nfft_max, psi_it_gpu.at<GPU>(), beff_gpu.at<GPU>());
//                        
//                        fft_gpu->transform(-1, psi_it_gpu.at<GPU>());
//
//                        fft_gpu->batch_unload(num_gkvec__, fft_index_gpu.at<GPU>(), psi_it_gpu.at<GPU>(), psi_pw_gpu.at<GPU>(), 0.0);
//
//                        mdarray<double_complex, 1>(&hpsi__[0](wf_pw_offset, i), psi_pw_gpu.at<GPU>(), num_gkvec__).copy_to_host();
//                    }
//                }
//            }));
//            #else
//            TERMINATE_NO_GPU
//            #endif
//        }
//        else
//        {
//            thread_workers.push_back(std::thread([thread_id, &idx_psi, &idx_psi_mutex, nfv, num_gkvec__, wf_pw_offset,
//                                                  fft, fft_index__, &fv_states__, &hpsi__, step_function,
//                                                  effective_magnetic_field__]()
//            {
//                bool done = false;
//
//                std::vector<double_complex> psi_it;
//                std::vector<double_complex> hpsi_it;
//
//                if (hpsi__.size() >= 3)
//                {
//                    psi_it.resize(fft->size());
//                    hpsi_it.resize(fft->size());
//                }
//
//                while (!done)
//                {
//                    /* increment the band index */
//                    idx_psi_mutex.lock();
//                    int i = idx_psi;
//                    if (idx_psi + 1 > nfv) 
//                    {
//                        done = true;
//                    }
//                    else
//                    {
//                        idx_psi++;
//                    }
//                    idx_psi_mutex.unlock();
//
//                    if (!done)
//                    {
//                        fft->input(num_gkvec__, fft_index__, &fv_states__(wf_pw_offset, i), thread_id);
//                        STOP();
//                        //fft->transform(1, thread_id);
//                                                    
//                        for (int ir = 0; ir < fft->size(); ir++)
//                        {
//                            /* hpsi(r) = psi(r) * Bz(r) * Theta(r) */
//                            fft->buffer(ir, thread_id) *= (effective_magnetic_field__[0]->f_it(ir) * step_function->theta_r(ir));
//                        }
//                        
//                        STOP();
//                        //fft->transform(-1, thread_id);
//                        fft->output(num_gkvec__, fft_index__, &hpsi__[0](wf_pw_offset, i), thread_id); 
//
//                        if (hpsi__.size() >= 3)
//                        {
//                            for (int ir = 0; ir < fft->size(); ir++)
//                            {
//                                /* hpsi(r) = psi(r) * (Bx(r) - iBy(r)) * Theta(r) */
//                                hpsi_it[ir] = psi_it[ir] * step_function->theta_r(ir) * 
//                                              (effective_magnetic_field__[1]->f_it(ir) - 
//                                               complex_i * effective_magnetic_field__[2]->f_it(ir));
//                            }
//                            
//                            fft->input(&hpsi_it[0], thread_id);
//                            STOP();
//                            //fft->transform(-1, thread_id);
//                            fft->output(num_gkvec__, fft_index__, &hpsi__[2](wf_pw_offset, i), thread_id); 
//                        }
//                        
//                        if (hpsi__.size() == 4)
//                        {
//                            for (int ir = 0; ir < fft->size(); ir++)
//                            {
//                                /* hpsi(r) = psi(r) * (Bx(r) + iBy(r)) * Theta(r) */
//                                hpsi_it[ir] = psi_it[ir] * step_function->theta_r(ir) *
//                                              (effective_magnetic_field__[1]->f_it(ir) + 
//                                               complex_i * effective_magnetic_field__[2]->f_it(ir));
//                            }
//                            
//                            fft->input(&hpsi_it[0], thread_id);
//                            STOP();
//                            //fft->transform(-1, thread_id);
//                            fft->output(num_gkvec__, fft_index__, &hpsi__[3](wf_pw_offset, i), thread_id); 
//                        }
//                    }
//                }
//            }));
//        }
//    }
//
//    for (auto& thread: thread_workers) thread.join();
//
//    /* copy Bz|\psi> to -Bz|\psi> */
//    for (int i = 0; i < nfv; i++)
//    {
//        for (int j = 0; j < (int)fv_states__.num_rows_local(); j++) hpsi__[1](j, i) = -hpsi__[0](j, i);
//    }
//
//    if (parameters_.processing_unit() == GPU)
//    {
//        #ifdef __GPU
//        fv_states__.deallocate_on_device();
//        #endif
//    }
}

};
