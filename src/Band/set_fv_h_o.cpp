#include "band.h"

namespace sirius {

template<> 
void Band::set_fv_h_o<CPU, full_potential_lapwlo>(K_point* kp__,
                                                  Periodic_function<double>* effective_potential__,
                                                  dmatrix<double_complex>& h__,
                                                  dmatrix<double_complex>& o__) const
{
    PROFILE_WITH_TIMER("sirius::Band::set_fv_h_o");
    
    h__.zero();
    o__.zero();

    double_complex zone(1, 0);
    
    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk = unit_cell_.num_atoms() / num_atoms_in_block +
               std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);
    DUMP("nblk: %i", nblk);

    int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
    DUMP("max_mt_aw: %i", max_mt_aw);

    mdarray<double_complex, 2> alm_row(kp__->num_gkvec_row(), max_mt_aw);
    mdarray<double_complex, 2> alm_col(kp__->num_gkvec_col(), max_mt_aw);
    mdarray<double_complex, 2> halm_col(kp__->num_gkvec_col(), max_mt_aw);

    runtime::Timer t1("sirius::Band::set_fv_h_o|zgemm");
    for (int iblk = 0; iblk < nblk; iblk++)
    {
        int num_mt_aw = 0;
        std::vector<int> offsets(num_atoms_in_block);
        for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
        {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
            offsets[ia - iblk * num_atoms_in_block] = num_mt_aw;
            num_mt_aw += type.mt_aw_basis_size();
        }
        
        #ifdef __PRINT_OBJECT_CHECKSUM
        alm_row.zero();
        alm_col.zero();
        halm_col.zero();
        #endif

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
            {
                if (ia % omp_get_num_threads() == tid)
                {
                    int ialoc = ia - iblk * num_atoms_in_block;
                    auto& atom = unit_cell_.atom(ia);
                    auto& type = atom.type();

                    mdarray<double_complex, 2> alm_row_tmp(alm_row.at<CPU>(0, offsets[ialoc]), kp__->num_gkvec_row(), type.mt_aw_basis_size());
                    mdarray<double_complex, 2> alm_col_tmp(alm_col.at<CPU>(0, offsets[ialoc]), kp__->num_gkvec_col(), type.mt_aw_basis_size());
                    mdarray<double_complex, 2> halm_col_tmp(halm_col.at<CPU>(0, offsets[ialoc]), kp__->num_gkvec_col(), type.mt_aw_basis_size());

                    kp__->alm_coeffs_row()->generate(ia, alm_row_tmp);
                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++)
                    {
                        for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) alm_row_tmp(igk, xi) = std::conj(alm_row_tmp(igk, xi));
                    }
                    kp__->alm_coeffs_col()->generate(ia, alm_col_tmp);
                    apply_hmt_to_apw<spin_block_t::nm>(kp__->num_gkvec_col(), ia, alm_col_tmp, halm_col_tmp);

                    /* setup apw-lo and lo-apw blocks */
                    set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row_tmp, alm_col_tmp, h__.panel(), o__.panel());
                }
            }
        }
        #ifdef __PRINT_OBJECT_CHECKSUM
        double_complex z1 = alm_row.checksum();
        double_complex z2 = alm_col.checksum();
        double_complex z3 = halm_col.checksum();
        DUMP("checksum(alm_row): %18.10f %18.10f", std::real(z1), std::imag(z1));
        DUMP("checksum(alm_col): %18.10f %18.10f", std::real(z2), std::imag(z2));
        DUMP("checksum(halm_col): %18.10f %18.10f", std::real(z3), std::imag(z3));
        #endif
        linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, zone,
                          alm_row.at<CPU>(), alm_row.ld(), alm_col.at<CPU>(), alm_col.ld(), zone, 
                          o__.at<CPU>(), o__.ld());

        linalg<CPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, zone, 
                          alm_row.at<CPU>(), alm_row.ld(), halm_col.at<CPU>(), halm_col.ld(), zone,
                          h__.at<CPU>(), h__.ld());
    }
    double tval = t1.stop();
    if (kp__->comm().rank() == 0)
    {
        DUMP("effective zgemm performance: %12.6f GFlops/rank",
             2 * 8e-9 * kp__->num_gkvec() * kp__->num_gkvec() * unit_cell_.mt_aw_basis_size() / tval);
    }

    /* add interstitial contributon */
    set_fv_h_o_it(kp__, effective_potential__, h__.panel(), o__.panel());

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(kp__, h__.panel(), o__.panel());
}

//=====================================================================================================================
// GPU code, (L)APW+lo basis
//=====================================================================================================================
#ifdef __GPU
template<> 
void Band::set_fv_h_o<GPU, full_potential_lapwlo>(K_point* kp__,
                                                  Periodic_function<double>* effective_potential__,
                                                  dmatrix<double_complex>& h__,
                                                  dmatrix<double_complex>& o__) const
{
    runtime::Timer t("sirius::Band::set_fv_h_o");
    
    h__.zero();
    h__.allocate_on_device();
    h__.zero_on_device();

    o__.zero();
    o__.allocate_on_device();
    o__.zero_on_device();

    double_complex zone(1, 0);

    int num_atoms_in_block = 2 * omp_get_max_threads();
    int nblk = unit_cell_.num_atoms() / num_atoms_in_block +
               std::min(1, unit_cell_.num_atoms() % num_atoms_in_block);
    DUMP("nblk: %i", nblk);

    int max_mt_aw = num_atoms_in_block * unit_cell_.max_mt_aw_basis_size();
    DUMP("max_mt_aw: %i", max_mt_aw);

    mdarray<double_complex, 3> alm_row(nullptr, kp__->num_gkvec_row(), max_mt_aw, 2);
    alm_row.allocate(1);
    alm_row.allocate_on_device();

    mdarray<double_complex, 3> alm_col(nullptr, kp__->num_gkvec_col(), max_mt_aw, 2);
    alm_col.allocate(1);
    alm_col.allocate_on_device();

    mdarray<double_complex, 3> halm_col(nullptr, kp__->num_gkvec_col(), max_mt_aw, 2);
    halm_col.allocate(1);
    halm_col.allocate_on_device();

    runtime::Timer t1("sirius::Band::set_fv_h_o|zgemm");
    for (int iblk = 0; iblk < nblk; iblk++)
    {
        int num_mt_aw = 0;
        std::vector<int> offsets(num_atoms_in_block);
        for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
        {
            auto& atom = unit_cell_.atom(ia);
            auto& type = atom.type();
            offsets[ia - iblk * num_atoms_in_block] = num_mt_aw;
            num_mt_aw += type.mt_aw_basis_size();
        }

        int s = iblk % 2;
            
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            for (int ia = iblk * num_atoms_in_block; ia < std::min(unit_cell_.num_atoms(), (iblk + 1) * num_atoms_in_block); ia++)
            {
                if (ia % omp_get_num_threads() == tid)
                {
                    int ialoc = ia - iblk * num_atoms_in_block;
                    auto& atom = unit_cell_.atom(ia);
                    auto& type = atom.type();

                    mdarray<double_complex, 2> alm_row_tmp(alm_row.at<CPU>(0, offsets[ialoc], s),
                                                           alm_row.at<GPU>(0, offsets[ialoc], s),
                                                           kp__->num_gkvec_row(), type.mt_aw_basis_size());

                    mdarray<double_complex, 2> alm_col_tmp(alm_col.at<CPU>(0, offsets[ialoc], s),
                                                           alm_col.at<GPU>(0, offsets[ialoc], s),
                                                           kp__->num_gkvec_col(), type.mt_aw_basis_size());
                    
                    mdarray<double_complex, 2> halm_col_tmp(halm_col.at<CPU>(0, offsets[ialoc], s),
                                                            halm_col.at<GPU>(0, offsets[ialoc], s),
                                                            kp__->num_gkvec_col(), type.mt_aw_basis_size());

                    kp__->alm_coeffs_row()->generate(ia, alm_row_tmp);
                    for (int xi = 0; xi < type.mt_aw_basis_size(); xi++)
                    {
                        for (int igk = 0; igk < kp__->num_gkvec_row(); igk++) alm_row_tmp(igk, xi) = std::conj(alm_row_tmp(igk, xi));
                    }
                    alm_row_tmp.async_copy_to_device(tid);

                    kp__->alm_coeffs_col()->generate(ia, alm_col_tmp);
                    alm_col_tmp.async_copy_to_device(tid);

                    apply_hmt_to_apw<nm>(kp__->num_gkvec_col(), ia, alm_col_tmp, halm_col_tmp);
                    halm_col_tmp.async_copy_to_device(tid);

                    /* setup apw-lo and lo-apw blocks */
                    set_fv_h_o_apw_lo(kp__, type, atom, ia, alm_row_tmp, alm_col_tmp, h__.panel(), o__.panel());
                }
            }
            cuda_stream_synchronize(tid);
        }
        cuda_stream_synchronize(omp_get_max_threads());
        linalg<GPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, &zone, 
                          alm_row.at<GPU>(0, 0, s), alm_row.ld(), alm_col.at<GPU>(0, 0, s), alm_col.ld(), &zone, 
                          o__.at<GPU>(), o__.ld(), omp_get_max_threads());

        linalg<GPU>::gemm(0, 1, kp__->num_gkvec_row(), kp__->num_gkvec_col(), num_mt_aw, &zone, 
                          alm_row.at<GPU>(0, 0, s), alm_row.ld(), halm_col.at<GPU>(0, 0, s), halm_col.ld(), &zone,
                          h__.at<GPU>(), h__.ld(), omp_get_max_threads());
    }

    acc::copyout(h__.at<CPU>(), h__.ld(), h__.at<GPU>(), h__.ld(), kp__->num_gkvec_row(), kp__->num_gkvec_col());
    acc::copyout(o__.at<CPU>(), o__.ld(), o__.at<GPU>(), o__.ld(), kp__->num_gkvec_row(), kp__->num_gkvec_col());
    
    double tval = t1.stop();
    if (kp__->comm().rank() == 0)
    {
        DUMP("effective zgemm performance: %12.6f GFlops/rank",
             2 * 8e-9 * kp__->num_gkvec() * kp__->num_gkvec() * unit_cell_.mt_aw_basis_size() / tval);
    }

    /* add interstitial contributon */
    set_fv_h_o_it(kp__, effective_potential__, h__.panel(), o__.panel());

    /* setup lo-lo block */
    set_fv_h_o_lo_lo(kp__, h__.panel(), o__.panel());

    h__.deallocate_on_device();
    o__.deallocate_on_device();
}
#endif

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
    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
    {
        int icol = kp->lo_col(ia, i);

        int l = kp->gklo_basis_descriptor_col(icol).l;
        int lm = kp->gklo_basis_descriptor_col(icol).lm;
        int idxrf = kp->gklo_basis_descriptor_col(icol).idxrf;
        int order = kp->gklo_basis_descriptor_col(icol).order;
        
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) 
        {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;
                    
            double_complex zsum = gaunt_coefs_->sum_L3_gaunt(lm1, lm, atom.h_radial_integrals(idxrf, idxrf1));

            if (std::abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm_row(igkloc, j1);
            }
        }

        for (int order1 = 0; order1 < (int)type.aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
            {
                o(igkloc, icol) += atom.symmetry_class().o_radial_integral(l, order1, order) * 
                                   alm_row(igkloc, type.indexb_by_lm_order(lm, order1));
            }
        }
    }

    std::vector<double_complex> ztmp(kp->num_gkvec_col());
    /* lo-apw block */
    for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
    {
        int irow = kp->lo_row(ia, i);

        int l = kp->gklo_basis_descriptor_row(irow).l;
        int lm = kp->gklo_basis_descriptor_row(irow).lm;
        int idxrf = kp->gklo_basis_descriptor_row(irow).idxrf;
        int order = kp->gklo_basis_descriptor_row(irow).order;

        memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(double_complex));
    
        for (int j1 = 0; j1 < type.mt_aw_basis_size(); j1++) 
        {
            int lm1 = type.indexb(j1).lm;
            int idxrf1 = type.indexb(j1).idxrf;
                    
            double_complex zsum = gaunt_coefs_->sum_L3_gaunt(lm, lm1, atom.h_radial_integrals(idxrf, idxrf1));

            if (std::abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
                    ztmp[igkloc] += zsum * alm_col(igkloc, j1);
            }
        }

        for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 

        for (int order1 = 0; order1 < (int)type.aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
            {
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
                auto& atom = unit_cell_.atom(ia);
                int lm1 = kp->gklo_basis_descriptor_row(irow).lm; 
                int idxrf1 = kp->gklo_basis_descriptor_row(irow).idxrf; 

                h(irow, icol) += gaunt_coefs_->sum_L3_gaunt(lm1, lm2, atom.h_radial_integrals(idxrf1, idxrf2));

                if (lm1 == lm2)
                {
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
