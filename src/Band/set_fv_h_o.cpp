#include "band.h"

namespace sirius {

void Band::set_fv_h_o_apw_lo(K_point* kp, 
                             Atom_type const& type, 
                             Atom const& atom, 
                             int ia, 
                             mdarray<double_complex, 2>& alm_row, // alm_row comes conjugated 
                             mdarray<double_complex, 2>& alm_col, 
                             mdarray<double_complex, 2>& h, 
                             mdarray<double_complex, 2>& o) const
{
    /* apw-lo block */
    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++) {
        int icol = kp->lo_col(ia, i);
        /* local orbital indices */
        int l = kp->gklo_basis_descriptor_col(icol).l;
        int lm = kp->gklo_basis_descriptor_col(icol).lm;
        int idxrf = kp->gklo_basis_descriptor_col(icol).idxrf;
        int order = kp->gklo_basis_descriptor_col(icol).order;
        /* loop over apw components */ 
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;
                    
            double_complex zsum = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm1, lm));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) {
                    h(igkloc, icol) += zsum * alm_row(igkloc, j1);
                }
            }
        }

        for (int order1 = 0; order1 < (int)type.aw_descriptor(l).size(); order1++) {
            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) {
                o(igkloc, icol) += atom.symmetry_class().o_radial_integral(l, order1, order) * 
                                   alm_row(igkloc, type.indexb_by_lm_order(lm, order1));
            }
        }
    }

    std::vector<double_complex> ztmp(kp->num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp->num_atom_lo_rows(ia); i++) {
        int irow = kp->lo_row(ia, i);
        /* local orbital indices */
        int l = kp->gklo_basis_descriptor_row(irow).l;
        int lm = kp->gklo_basis_descriptor_row(irow).lm;
        int idxrf = kp->gklo_basis_descriptor_row(irow).idxrf;
        int order = kp->gklo_basis_descriptor_row(irow).order;

        std::memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(double_complex));
        /* loop over apw components */ 
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;
                    
            double_complex zsum = atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf, gaunt_coefs_->gaunt_vector(lm, lm1));

            if (std::abs(zsum) > 1e-14) {
                for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                    ztmp[igkloc] += zsum * alm_col(igkloc, j1);
                }
            }
        }

        for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
            h(irow, igkloc) += ztmp[igkloc]; 
        }

        for (int order1 = 0; order1 < (int)type.aw_descriptor(l).size(); order1++) {
            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) {
                o(irow, igkloc) += atom.symmetry_class().o_radial_integral(l, order, order1) * 
                                   alm_col(igkloc, type.indexb_by_lm_order(lm, order1));
            }
        }
    }
}

void Band::set_fv_h_o_it(K_point* kp, Periodic_function<double>* effective_potential, 
                         mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o) const
{
    runtime::Timer t("sirius::Band::set_fv_h_o_it");

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z1 = mdarray<double_complex, 1>(&effective_potential->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    DUMP("checksum(veff_pw): %18.10f %18.10f", std::real(z1), std::imag(z1));
    #endif

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        auto gkvec_col_cart = unit_cell_.reciprocal_lattice_vectors() * kp->gklo_basis_descriptor_col(igk_col).gkvec;
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            auto gkvec_row_cart = unit_cell_.reciprocal_lattice_vectors() * kp->gklo_basis_descriptor_row(igk_row).gkvec;
            int ig12 = ctx_.gvec().index_g12(kp->gklo_basis_descriptor_row(igk_row).gvec,
                                             kp->gklo_basis_descriptor_col(igk_col).gvec);
            
            /* pw kinetic energy */
            double t1 = 0.5 * (gkvec_row_cart * gkvec_col_cart);
                               
            h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + t1 * ctx_.step_function().theta_pw(ig12));
            o(igk_row, igk_col) += ctx_.step_function().theta_pw(ig12);
        }
    }
}

void Band::set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o) const
{
    runtime::Timer t("sirius::Band::set_fv_h_o_lo_lo");

    /* lo-lo block */
    #pragma omp parallel for default(shared)
    for (int icol = kp->num_gkvec_col(); icol < kp->gklo_basis_size_col(); icol++) {
        int ia = kp->gklo_basis_descriptor_col(icol).ia;
        int lm2 = kp->gklo_basis_descriptor_col(icol).lm; 
        int idxrf2 = kp->gklo_basis_descriptor_col(icol).idxrf; 

        for (int irow = kp->num_gkvec_row(); irow < kp->gklo_basis_size_row(); irow++) {
            /* lo-lo block is diagonal in atom index */ 
            if (ia == kp->gklo_basis_descriptor_row(irow).ia) {
                auto& atom = unit_cell_.atom(ia);
                int lm1 = kp->gklo_basis_descriptor_row(irow).lm; 
                int idxrf1 = kp->gklo_basis_descriptor_row(irow).idxrf; 

                h(irow, icol) += atom.radial_integrals_sum_L3<spin_block_t::nm>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));

                if (lm1 == lm2) {
                    int l = kp->gklo_basis_descriptor_row(irow).l;
                    int order1 = kp->gklo_basis_descriptor_row(irow).order; 
                    int order2 = kp->gklo_basis_descriptor_col(icol).order; 
                    o(irow, icol) += atom.symmetry_class().o_radial_integral(l, order1, order2);
                }
            }
        }
    }
}

};
