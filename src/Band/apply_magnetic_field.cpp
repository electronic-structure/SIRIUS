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

void Band::apply_magnetic_field(Wave_functions<true>& fv_states__,
                                Gvec const& gkvec__,
                                Periodic_function<double>* effective_magnetic_field__[3],
                                std::vector<Wave_functions<true>*>& hpsi__) const
{
    PROFILE_WITH_TIMER("sirius::Band::apply_magnetic_field");

    assert(hpsi__.size() >= 2);

    int nfv = fv_states__.spl_num_swapped().local_size();

    for (auto& e: hpsi__) e->set_num_swapped(ctx_.num_fv_states());

    mdarray<double_complex, 3> zm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), 
                                  ctx_.num_mag_dims());

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
    {
        auto& atom = unit_cell_.atom(ia);
        int offset = atom.offset_wf();
        int mt_basis_size = atom.type().mt_basis_size();
        
        zm.zero();
        
        /* only upper triangular part of zm is computed because it is a hermitian matrix */
        #pragma omp parallel for default(shared)
        for (int j2 = 0; j2 < mt_basis_size; j2++)
        {
            int lm2 = atom.type().indexb(j2).lm;
            int idxrf2 = atom.type().indexb(j2).idxrf;
            
            for (int i = 0; i < ctx_.num_mag_dims(); i++)
            {
                for (int j1 = 0; j1 <= j2; j1++)
                {
                    int lm1 = atom.type().indexb(j1).lm;
                    int idxrf1 = atom.type().indexb(j1).idxrf;

                    zm(j1, j2, i) = gaunt_coefs_->sum_L3_gaunt(lm1, lm2, atom.b_radial_integrals(idxrf1, idxrf2, i)); 
                }
            }
        }
        /* compute bwf = B_z*|wf_j> */
        linalg<CPU>::hemm(0, 0, mt_basis_size, nfv, complex_one, &zm(0, 0, 0), zm.ld(), 
                          &fv_states__[0][offset], fv_states__.wf_size(), complex_zero, &(*hpsi__[0])[0][offset], hpsi__[0]->wf_size());
        
        /* compute bwf = (B_x - iB_y)|wf_j> */
        if (hpsi__.size() >= 3)
        {
            /* reuse first (z) component of zm matrix to store (B_x - iB_y) */
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) - complex_i * zm(j1, j2, 2);
                
                /* remember: zm is hermitian and we computed only the upper triangular part */
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = std::conj(zm(j2, j1, 1)) - complex_i * std::conj(zm(j2, j1, 2));
            }
              
            linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                              &fv_states__[0][offset], fv_states__.wf_size(), &(*hpsi__[2])[0][offset], hpsi__[2]->wf_size());
        }
        
        //== /* compute bwf = (B_x + iB_y)|wf_j> */
        //== if (hpsi__.size() == 4 && std_evp_solver()->parallel())
        //== {
        //==     /* reuse first (z) component of zm matrix to store (Bx + iBy) */
        //==     for (int j2 = 0; j2 < mt_basis_size; j2++)
        //==     {
        //==         for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) + complex_i * zm(j1, j2, 2);
        //==         
        //==         for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = std::conj(zm(j2, j1, 1)) + complex_i * std::conj(zm(j2, j1, 2));
        //==     }
        //==       
        //==     linalg<CPU>::gemm(0, 0, mt_basis_size, nfv, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
        //==                       &fv_states__(offset, 0), fv_states__.ld(), &hpsi__[2](offset, 0), hpsi__[2].ld());
        //== }
    }
    std::vector<double_complex> psi_r;
    if (hpsi__.size() == 3) psi_r.resize(ctx_.fft().local_size());

    int wf_pw_offset = unit_cell_.mt_basis_size();
    for (int i = 0; i < fv_states__.spl_num_swapped().local_size(); i++)
    {
        /* transform first-variational state to real space */
        ctx_.fft().transform<1>(gkvec__, &fv_states__[i][wf_pw_offset]);
        /* save for a reuse */
        if (hpsi__.size() == 3) ctx_.fft().output(&psi_r[0]);

        for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
        {
            /* hpsi(r) = psi(r) * B_z(r) * Theta(r) */
            ctx_.fft().buffer(ir) *= (effective_magnetic_field__[0]->f_rg(ir) * ctx_.step_function().theta_r(ir));
        }
        ctx_.fft().transform<-1>(gkvec__, &(*hpsi__[0])[i][wf_pw_offset]);

        if (hpsi__.size() >= 3)
        {
            for (int ir = 0; ir < ctx_.fft().local_size(); ir++)
            {
                /* hpsi(r) = psi(r) * (B_x(r) - iB_y(r)) * Theta(r) */
                ctx_.fft().buffer(ir) = psi_r[ir] * ctx_.step_function().theta_r(ir) * 
                                         (effective_magnetic_field__[1]->f_rg(ir) - 
                                          complex_i * effective_magnetic_field__[2]->f_rg(ir));
            }
            ctx_.fft().transform<-1>(gkvec__, &(*hpsi__[2])[i][wf_pw_offset]);
        }
    }

   /* copy Bz|\psi> to -Bz|\psi> */
    for (int i = 0; i < nfv; i++)
    {
        for (int j = 0; j < fv_states__.wf_size(); j++) (*hpsi__[1])[i][j] = -(*hpsi__[0])[i][j];
    }
}

};
