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

/** \file get_h_o_diag.cpp
 *   
 *  \brief Contains implementation of sirius::Band::get_h_diag and sirius::Band::get_o_diag methods.
 */

#include "band.h"

namespace sirius {

template <typename T>
std::vector<double> Band::get_h_diag(K_point* kp__,
                                     int ispn__,
                                     double v0__,
                                     D_operator<T>& d_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::get_h_diag");

    std::vector<double> h_diag(kp__->num_gkvec_loc());

    /* local H contribution */
    for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++)
    {
        int ig = kp__->gklo_basis_descriptor_row(ig_loc).ig;
        auto vgk = kp__->gkvec().cart_shifted(ig);
        h_diag[ig_loc] = 0.5 * (vgk * vgk) + v0__;
    }

    /* non-local H contribution */
    auto& beta_gk_t = kp__->beta_projectors().beta_gk_t();
    matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type.mt_basis_size();
        matrix<double_complex> d_sum(nbf, nbf);
        d_sum.zero();

        for (int i = 0; i < atom_type.num_atoms(); i++)
        {
            int ia = atom_type.atom_id(i);
        
            for (int xi2 = 0; xi2 < nbf; xi2++)
                for (int xi1 = 0; xi1 < nbf; xi1++) 
                    d_sum(xi1, xi2) += d_op__(xi1, xi2, ispn__, ia);
        }

        int offs = unit_cell_.atom_type(iat).offset_lo();
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++)
            for (int xi = 0; xi < nbf; xi++)
                beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);

        #pragma omp parallel for schedule(static)
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++)
        {
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    /* compute <G+k|beta_xi1> D_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                    auto z = beta_gk_tmp(xi1, ig_loc) * d_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
                    h_diag[ig_loc] += z.real();
                }
            }
        }
    }
    return h_diag;
}

template <typename T>
std::vector<double> Band::get_o_diag(K_point* kp__,
                                     Q_operator<T>& q_op__) const
{
    PROFILE_WITH_TIMER("sirius::Band::get_o_diag");

    std::vector<double> o_diag(kp__->num_gkvec_loc(), 1.0);
    if (ctx_.esm_type() != ultrasoft_pseudopotential) STOP(); // decide what to do here

    /* non-local O contribution */
    auto& beta_gk_t = kp__->beta_projectors().beta_gk_t();
    matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type.mt_basis_size();

        matrix<double_complex> q_sum(nbf, nbf);
        q_sum.zero();
        
        for (int i = 0; i < atom_type.num_atoms(); i++)
        {
            int ia = atom_type.atom_id(i);
        
            for (int xi2 = 0; xi2 < nbf; xi2++)
                for (int xi1 = 0; xi1 < nbf; xi1++) 
                    q_sum(xi1, xi2) += q_op__(xi1, xi2, ia);
        }

        int offs = unit_cell_.atom_type(iat).offset_lo();
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++)
            for (int xi = 0; xi < nbf; xi++)
                beta_gk_tmp(xi, ig_loc) = beta_gk_t(ig_loc, offs + xi);

        #pragma omp parallel for
        for (int ig_loc = 0; ig_loc < kp__->num_gkvec_loc(); ig_loc++)
        {
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    /* compute <G+k|beta_xi1> Q_{xi1, xi2} <beta_xi2|G+k> contribution from all atoms */
                    auto z = beta_gk_tmp(xi1, ig_loc) * q_sum(xi1, xi2) * std::conj(beta_gk_tmp(xi2, ig_loc));
                    o_diag[ig_loc] += z.real();
                }
            }
        }
    }

    return o_diag;
}

template std::vector<double> Band::get_h_diag<double>(K_point* kp__,
                                                      int ispn__,
                                                      double v0__,
                                                      D_operator<double>& d_op__) const;

template std::vector<double> Band::get_h_diag<double_complex>(K_point* kp__,
                                                              int ispn__,
                                                              double v0__,
                                                              D_operator<double_complex>& d_op__) const;

template std::vector<double> Band::get_o_diag<double>(K_point* kp__,
                                                      Q_operator<double>& q_op__) const;

template std::vector<double> Band::get_o_diag<double_complex>(K_point* kp__,
                                                              Q_operator<double_complex>& q_op__) const;
};
