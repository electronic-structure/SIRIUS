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

/** \file band.hpp
 *   
 *   \brief Contains implementation of templated methods of sirius::Band class.
 */

// TODO: look at multithreading in apw_lo and lo_apw blocks 
// TODO: k-independent L3 sum

//== template <spin_block_t sblock>
//== void Band::apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
//== {
//==     Timer t("sirius::Band::apply_uj_correction");
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         if (unit_cell_.atom(ia)->apply_uj_correction())
//==         {
//==             Atom_type* type = unit_cell_.atom(ia)->type();
//== 
//==             int offset = unit_cell_.atom(ia)->offset_wf();
//== 
//==             int l = unit_cell_.atom(ia)->uj_correction_l();
//== 
//==             int nrf = type->indexr().num_rf(l);
//== 
//==             for (int order2 = 0; order2 < nrf; order2++)
//==             {
//==                 for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
//==                 {
//==                     int idx2 = type->indexb_by_lm_order(lm2, order2);
//==                     for (int order1 = 0; order1 < nrf; order1++)
//==                     {
//==                         double ori = unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
//==                         
//==                         for (int ist = 0; ist < parameters_.spl_fv_states().local_size(); ist++)
//==                         {
//==                             for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
//==                             {
//==                                 int idx1 = type->indexb_by_lm_order(lm1, order1);
//==                                 double_complex z1 = fv_states(offset + idx1, ist) * ori;
//== 
//==                                 if (sblock == uu)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 0) += z1 * 
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);
//==                                 }
//== 
//==                                 if (sblock == dd)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 1) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);
//==                                 }
//== 
//==                                 if (sblock == ud)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 2) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
//==                                 }
//==                                 
//==                                 if (sblock == du)
//==                                 {
//==                                     hpsi(offset + idx2, ist, 3) += z1 *
//==                                         unit_cell_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
//==                                 }
//==                             }
//==                         }
//==                     }
//==                 }
//==             }
//==         }
//==     }
//== }

template <spin_block_t sblock>
void Band::apply_hmt_to_apw(int num_gkvec__,
                            int ia__,
                            mdarray<double_complex, 2>& alm__,
                            mdarray<double_complex, 2>& halm__)
{
    Atom* atom = unit_cell_.atom(ia__);
    Atom_type* type = atom->type();

    // TODO: this is k-independent and can in principle be precomputed together with radial integrals if memory is available
    mdarray<double_complex, 2> hmt(type->mt_aw_basis_size(), type->mt_aw_basis_size());
    for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
    {
        int lm2 = type->indexb(j2).lm;
        int idxrf2 = type->indexb(j2).idxrf;
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
            hmt(j1, j2) = atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
        }
    }
    linalg<CPU>::gemm(0, 1, num_gkvec__, type->mt_aw_basis_size(), type->mt_aw_basis_size(), alm__, hmt, halm__);
}

//== template <spin_block_t sblock>
//== void Band::apply_hmt_to_apw(mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm)
//== {
//==     Timer t("sirius::Band::apply_hmt_to_apw", _global_timer_);
//== 
//==     int ngk_loc = (int)alm.size(1);
//== 
//==     mdarray<double_complex, 2> alm_tmp(ngk_loc, alm.size(0));
//==     for (int igk = 0; igk < ngk_loc; igk++)
//==     {
//==         for (int i0 = 0; i0 < (int)alm.size(0); i0++) alm_tmp(igk, i0) = alm(i0, igk);
//==     }
//==     
//==     #pragma omp parallel default(shared)
//==     {
//==         std::vector<double_complex> zv(ngk_loc);
//==         
//==         #pragma omp for
//==         for (int j = 0; j < unit_cell_.mt_aw_basis_size(); j++)
//==         {
//==             int ia = unit_cell_.mt_aw_basis_descriptor(j).ia;
//==             int xi = unit_cell_.mt_aw_basis_descriptor(j).xi;
//==             Atom* atom = unit_cell_.atom(ia);
//==             Atom_type* type = atom->type();
//==             int lm1 = type->indexb(xi).lm;
//==             int idxrf1 = type->indexb(xi).idxrf; 
//== 
//==             memset(&zv[0], 0, zv.size() * sizeof(double_complex));
//== 
//==             for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
//==             {
//==                 int lm2 = type->indexb(j2).lm;
//==                 int idxrf2 = type->indexb(j2).idxrf;
//==                 double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
//== 
//==                 if (abs(zsum) > 1e-14) 
//==                 {
//==                     for (int igk = 0; igk < ngk_loc; igk++) zv[igk] += zsum * alm_tmp(igk, atom->offset_aw() + j2); 
//==                 }
//==             }
//==             
//==             for (int igk = 0; igk < ngk_loc; igk++) halm(j, igk) = zv[igk];
//==         }
//==     }
//== }

//template <spin_block_t sblock>
//void Band::set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
//                        mdarray<double_complex, 2>& h)
//{
//    Timer t("sirius::Band::set_h_apw_lo");
//    
//    int apw_offset_col = kp->apw_offset_col();
//    
//    #pragma omp parallel default(shared)
//    {
//        // apw-lo block
//        #pragma omp for
//        for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
//        {
//            int icol = kp->lo_col(ia, i);
//
//            int lm = kp->gklo_basis_descriptor_col(icol).lm;
//            int idxrf = kp->gklo_basis_descriptor_col(icol).idxrf;
//            
//            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
//            {
//                int lm1 = type->indexb(j1).lm;
//                int idxrf1 = type->indexb(j1).idxrf;
//                        
//                double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm1, lm));
//                
//                if (abs(zsum) > 1e-14)
//                {
//                    for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
//                }
//            }
//        }
//    }
//    
//    #pragma omp parallel default(shared)
//    {
//        std::vector<double_complex> ztmp(kp->num_gkvec_col());
//        // lo-apw block
//        #pragma omp for
//        for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
//        {
//            int irow = kp->lo_row(ia, i);
//
//            int lm = kp->gklo_basis_descriptor_row(irow).lm;
//            int idxrf = kp->gklo_basis_descriptor_row(irow).idxrf;
//
//            memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(double_complex));
//        
//            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
//            {
//                int lm1 = type->indexb(j1).lm;
//                int idxrf1 = type->indexb(j1).idxrf;
//                        
//                double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm, lm1));
//
//                if (abs(zsum) > 1e-14)
//                {
//                    for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
//                        ztmp[igkloc] += zsum * conj(alm(apw_offset_col + igkloc, j1));
//                }
//            }
//
//            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 
//        }
//    }
//}

template <spin_block_t sblock>
void Band::set_h_it(K_point* kp, Periodic_function<double>* effective_potential, 
                    Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h)
{
    Timer t("sirius::Band::set_h_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = ctx_.gvec().index_g12(kp->gklo_basis_descriptor_row(igk_row).gvec,
                                             kp->gklo_basis_descriptor_col(igk_col).gvec);
            
            /* pw kinetic energy */
            double t1 = 0.5 * (kp->gklo_basis_descriptor_row(igk_row).gkvec_cart *
                               kp->gklo_basis_descriptor_col(igk_col).gkvec_cart);
                              
            switch (sblock)
            {
                case nm:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + t1 * ctx_.step_function()->theta_pw(ig12));
                    break;
                }
                case uu:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + effective_magnetic_field[0]->f_pw(ig12) +  
                                            t1 * ctx_.step_function()->theta_pw(ig12));
                    break;
                }
                case dd:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) - effective_magnetic_field[0]->f_pw(ig12) +  
                                            t1 * ctx_.step_function()->theta_pw(ig12));
                    break;
                }
                case ud:
                {
                    h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) - 
                                            double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
                case du:
                {
                    h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) + 
                                            double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
            }
        }
    }
}

template <spin_block_t sblock>
void Band::set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h)
{
    Timer t("sirius::Band::set_h_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = kp->num_gkvec_col(); icol < kp->gklo_basis_size_col(); icol++)
    {
        int ia = kp->gklo_basis_descriptor_col(icol).ia;
        int lm2 = kp->gklo_basis_descriptor_col(icol).lm; 
        int idxrf2 = kp->gklo_basis_descriptor_col(icol).idxrf; 

        for (int irow = kp->num_gkvec_row(); irow < kp->gklo_basis_size_row(); irow++)
        {
            if (ia == kp->gklo_basis_descriptor_row(irow).ia)
            {
                Atom* atom = unit_cell_.atom(ia);
                int lm1 = kp->gklo_basis_descriptor_row(irow).lm; 
                int idxrf1 = kp->gklo_basis_descriptor_row(irow).idxrf; 

                h(irow, icol) += atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
    }
}

//== template <spin_block_t sblock> 
//== void Band::set_h(K_point* kp, Periodic_function<double>* effective_potential, 
//==                  Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h)
//== {
//==     Timer t("sirius::Band::set_h");
//==    
//==     // index of column apw coefficients in apw array
//==     int apw_offset_col = kp->apw_offset_col();
//==     
//==     mdarray<double_complex, 2> alm(kp->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
//==     mdarray<double_complex, 2> halm(kp->num_gkvec_row(), unit_cell_.max_mt_aw_basis_size());
//== 
//==     h.zero();
//== 
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom* atom = unit_cell_.atom(ia);
//==         Atom_type* type = atom->type();
//==        
//==         // generate conjugated coefficients
//==         kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
//==         
//==         // apply muffin-tin part to <bra|
//==         apply_hmt_to_apw<sblock>(kp->num_gkvec_row(), ia, alm, halm);
//==         
//==         // generate <apw|H|apw> block; |ket> is conjugated, so it is "unconjugated" back
//==         blas<CPU>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), complex_one, 
//==                         &halm(0, 0), halm.ld(), &alm(apw_offset_col, 0), alm.ld(), complex_one, &h(0, 0), h.ld());
//==        
//==         // setup apw-lo blocks
//==         set_h_apw_lo<sblock>(kp, type, atom, ia, alm, h);
//==     } //ia
//== 
//==     set_h_it<sblock>(kp, effective_potential, effective_magnetic_field, h);
//== 
//==     set_h_lo_lo<sblock>(kp, h);
//== 
//==     alm.deallocate();
//==     halm.deallocate();
//== }

template <bool need_o_diag>
void Band::get_h_o_diag(K_point* kp__,
                  double v0__,
                  std::vector<double> const& pw_ekin__,
                  std::vector<double>& h_diag__,
                  std::vector<double>& o_diag__)
{
    Timer t("sirius::Band::get_h_o_diag");

    h_diag__.resize(kp__->num_gkvec_loc());
    o_diag__.resize(kp__->num_gkvec_loc());

    /* local H contribution */
    for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
    {
        int igk = kp__->gklo_basis_descriptor_row(igk_loc).ig;
        h_diag__[igk_loc] = pw_ekin__[igk] + v0__;
        o_diag__[igk_loc] = 1.0;
    }

    /* non-local H contribution */
    auto& beta_gk_t = kp__->beta_projectors().beta_gk_t();
    matrix<double_complex> beta_gk_tmp(unit_cell_.max_mt_basis_size(), kp__->num_gkvec_loc());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
    {
        auto atom_type = unit_cell_.atom_type(iat);
        int nbf = atom_type->mt_basis_size();
        matrix<double_complex> d_sum(nbf, nbf);
        d_sum.zero();

        matrix<double_complex> q_sum;
        if (need_o_diag)
        {
            q_sum = matrix<double_complex>(nbf, nbf);
            q_sum.zero();
        }

        for (int i = 0; i < atom_type->num_atoms(); i++)
        {
            int ia = atom_type->atom_id(i);
        
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 < nbf; xi1++)
                {
                    d_sum(xi1, xi2) += unit_cell_.atom(ia)->d_mtrx(xi1, xi2);
                    if (need_o_diag) q_sum(xi1, xi2) += unit_cell_.atom(ia)->type()->uspp().q_mtrx(xi1, xi2);
                }
            }
        }

        int ofs = unit_cell_.atom_type(iat)->offset_lo();
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
        {
            for (int xi = 0; xi < nbf; xi++) beta_gk_tmp(xi, igk_loc) = beta_gk_t(igk_loc, ofs + xi);
        }

        std::vector< std::pair<int, int> > idx(nbf * nbf);
        for (int xi2 = 0, n = 0; xi2 < nbf; xi2++)
        {
            for (int xi1 = 0; xi1 < nbf; xi1++) idx[n++] = std::pair<int, int>(xi1, xi2);
        }

        #pragma omp parallel for
        for (int igk_loc = 0; igk_loc < kp__->num_gkvec_loc(); igk_loc++)
        {
            for (auto& it: idx)
            {
                int xi1 = it.first;
                int xi2 = it.second;
                double_complex z = beta_gk_tmp(xi1, igk_loc) * conj(beta_gk_tmp(xi2, igk_loc));

                h_diag__[igk_loc] += real(z * d_sum(xi1, xi2));
                if (need_o_diag) o_diag__[igk_loc] += real(z * q_sum(xi1, xi2));
            }
        }
    }
}

