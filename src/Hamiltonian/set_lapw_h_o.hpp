//
// Created by mathieut on 7/23/19.
//

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

/** \file set_lapw_h_o.hpp
 *
 *  \brief Contains functions of LAPW Hamiltonian and overlap setup.
 */

//template <spin_block_t sblock>
//void Band::set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm,
//                        mdarray<double_complex, 2>& h)
//{
//    Timer t("sirius::Hamiltonian::set_h_apw_lo");
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
void Hamiltonian::set_h_it(K_point* kp, Periodic_function<double>* effective_potential,
                           Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h) const
{
    PROFILE("sirius::Hamiltonian::set_h_it");

    STOP(); // effective potential is now stored in the veff_pw_ auxiliary array. Fix this.

    //#pragma omp parallel for default(shared)
    //for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) {
    //    auto gkvec_col_cart = kp->gkvec().gkvec_cart(kp->igk_col(igk_col));
    //    auto gvec_col = kp->gkvec().gvec(kp->igk_col(igk_col));
    //    for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) {
    //        auto gkvec_row_cart = kp->gkvec().gkvec_cart(kp->igk_row(igk_row));
    //        auto gvec_row = kp->gkvec().gvec(kp->igk_row(igk_row));

    //        int ig12 = ctx_.gvec().index_g12(gvec_row, gvec_col);
    //
    //        /* pw kinetic energy */
    //        double t1 = 0.5 * (gkvec_row_cart * gkvec_col_cart);
    //
    //        switch (sblock) {
    //            case spin_block_t::nm: {
    //                h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + t1 * ctx_.step_function().theta_pw(ig12));
    //                break;
    //            }
    //            case spin_block_t::uu: {
    //                h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + effective_magnetic_field[0]->f_pw(ig12) +
    //                                        t1 * ctx_.step_function().theta_pw(ig12));
    //                break;
    //            }
    //            case spin_block_t::dd: {
    //                h(igk_row, igk_col) += (effective_potential->f_pw(ig12) - effective_magnetic_field[0]->f_pw(ig12) +
    //                                        t1 * ctx_.step_function().theta_pw(ig12));
    //                break;
    //            }
    //            case spin_block_t::ud: {
    //                h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) -
    //                                        double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
    //                break;
    //            }
    //            case spin_block_t::du: {
    //                h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) +
    //                                        double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
    //                break;
    //            }
    //        }
    //    }
    //}
}

template <spin_block_t sblock>
void Hamiltonian::set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h) const
{
    PROFILE("sirius::Hamiltonian::set_h_lo_lo");

    /* lo-lo block */
    #pragma omp parallel for default(shared)
    for (int icol = 0; icol < kp->num_lo_col(); icol++) {
        int ia     = kp->lo_basis_descriptor_col(icol).ia;
        int lm2    = kp->lo_basis_descriptor_col(icol).lm;
        int idxrf2 = kp->lo_basis_descriptor_col(icol).idxrf;

        for (int irow = 0; irow < kp->num_lo_row(); irow++) {
            if (ia == kp->lo_basis_descriptor_row(irow).ia) {
                auto& atom = unit_cell_.atom(ia);
                int lm1    = kp->lo_basis_descriptor_row(irow).lm;
                int idxrf1 = kp->lo_basis_descriptor_row(irow).idxrf;

                h(kp->num_gkvec_row() + irow, kp->num_gkvec_col() + icol) +=
                    atom.radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
    }
}

//== template <spin_block_t sblock>
//== void Band::set_h(K_point* kp, Periodic_function<double>* effective_potential,
//==                  Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h)
//== {
//==     Timer t("sirius::Hamiltonian::set_h");
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

//void Band::set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm,
//                        mdarray<double_complex, 2>& o)
//{
//    Timer t("sirius::Hamiltonian::set_o_apw_lo");
//
//    int apw_offset_col = kp->apw_offset_col();
//
//    // apw-lo block
//    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
//    {
//        int icol = kp->lo_col(ia, i);
//
//        int l = kp->gklo_basis_descriptor_col(icol).l;
//        int lm = kp->gklo_basis_descriptor_col(icol).lm;
//        int order = kp->gklo_basis_descriptor_col(icol).order;
//
//        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
//        {
//            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
//            {
//                o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) *
//                                   alm(igkloc, type->indexb_by_lm_order(lm, order1));
//            }
//        }
//    }
//
//    std::vector<double_complex> ztmp(kp->num_gkvec_col());
//    // lo-apw block
//    for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
//    {
//        int irow = kp->lo_row(ia, i);
//
//        int l = kp->gklo_basis_descriptor_row(irow).l;
//        int lm = kp->gklo_basis_descriptor_row(irow).lm;
//        int order = kp->gklo_basis_descriptor_row(irow).order;
//
//        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
//        {
//            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
//            {
//                o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) *
//                                   conj(alm(apw_offset_col + igkloc, type->indexb_by_lm_order(lm, order1)));
//            }
//        }
//    }
//}

//== void Band::set_o(K_point* kp, mdarray<double_complex, 2>& o)
//== {
//==     Timer t("sirius::Hamiltonian::set_o");
//==
//==     // index of column apw coefficients in apw array
//==     int apw_offset_col = kp->apw_offset_col();
//==
//==     mdarray<double_complex, 2> alm(kp->num_gkvec_loc(), unit_cell_.max_mt_aw_basis_size());
//==     o.zero();
//==
//==     double_complex zone(1, 0);
//==
//==     for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
//==     {
//==         Atom* atom = unit_cell_.atom(ia);
//==         Atom_type* type = atom->type();
//==
//==         // generate conjugated coefficients
//==         kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
//==
//==         // generate <apw|apw> block; |ket> is conjugated, so it is "unconjugated" back
//==         blas<CPU>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone,
//==                         &alm(0, 0), alm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &o(0, 0), o.ld());
//==
//==         // setup apw-lo blocks
//==         set_o_apw_lo(kp, type, atom, ia, alm, o);
//==     } //ia
//==
//==     set_o_it(kp, o);
//==
//==     set_o_lo_lo(kp, o);
//==
//==     alm.deallocate();
//== }
