#include "band.h"

namespace sirius {

/** \param [in] phi Input wave-functions [storage: CPU && GPU].
 *  \param [in] hphi Hamiltonian, applied to wave-functions [storage: CPU || GPU].
 *  \param [in] ophi Overlap operator, applied to wave-functions [storage: CPU || GPU].
 */
void Band::set_fv_h_o(K_point* kp__,
                      int N__,
                      int n__,
                      Wave_functions<false>& phi__,
                      Wave_functions<false>& hphi__,
                      Wave_functions<false>& ophi__,
                      matrix<double_complex>& h__,
                      matrix<double_complex>& o__,
                      matrix<double_complex>& h_old__,
                      matrix<double_complex>& o_old__)
{
    PROFILE_WITH_TIMER("sirius::Band::set_fv_h_o");
    
    assert(n__ != 0);

    /* copy old Hamiltonian and overlap */
    for (int i = 0; i < N__; i++)
    {
        std::memcpy(&h__(0, i), &h_old__(0, i), N__ * sizeof(double_complex));
        std::memcpy(&o__(0, i), &o_old__(0, i), N__ * sizeof(double_complex));
    }

    /* <{phi,res}|H|res> */
    phi__.inner(0, N__ + n__, hphi__, N__, n__, h__, 0, N__);
    /* <{phi,res}|O|res> */
    phi__.inner(0, N__ + n__, ophi__, N__, n__, o__, 0, N__);

    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex cs1(0, 0);
    double_complex cs2(0, 0);
    for (int i = 0; i < N__ + n__; i++)
    {
        for (int j = 0; j <= i; j++) 
        {
            cs1 += h__(j, i);
            cs2 += o__(j, i);
        }
    }
    DUMP("checksum(h): %18.10f %18.10f", cs1.real(), cs1.imag());
    DUMP("checksum(o): %18.10f %18.10f", cs2.real(), cs2.imag());
    #endif

    for (int i = 0; i < N__ + n__; i++)
    {
        if (h__(i, i).imag() > 1e-12)
        {
            std::stringstream s;
            s << "wrong diagonal of H: " << h__(i, i);
            TERMINATE(s);
        }
        if (o__(i, i).imag() > 1e-12)
        {
            std::stringstream s;
            s << "wrong diagonal of O: " << o__(i, i);
            TERMINATE(s);
        }
        h__(i, i) = h__(i, i).real();
        o__(i, i) = o__(i, i).real();
    }

    #if (__VERIFICATION > 0)
    /* check n__ * n__ block */
    for (int i = N__; i < N__ + n__; i++)
    {
        for (int j = N__; j < N__ + n__; j++)
        {
            if (std::abs(h__(i, j) - std::conj(h__(j, i))) > 1e-10 ||
                std::abs(o__(i, j) - std::conj(o__(j, i))) > 1e-10)
            {
                double_complex z1, z2;
                z1 = h__(i, j);
                z2 = h__(j, i);

                std::cout << "h(" << i << "," << j << ")=" << z1 << " "
                          << "h(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
                z1 = o__(i, j);
                z2 = o__(j, i);

                std::cout << "o(" << i << "," << j << ")=" << z1 << " "
                          << "o(" << j << "," << i << ")=" << z2 << ", diff=" << std::abs(z1 - std::conj(z2)) << std::endl;
                
            }
        }
    }
    #endif

    /* restore the lower part */
    #pragma omp parallel for
    for (int i = 0; i < N__; i++)
    {
        for (int j = N__; j < N__ + n__; j++)
        {
            h__(j, i) = std::conj(h__(i, j));
            o__(j, i) = std::conj(o__(i, j));
        }
    }

    /* save Hamiltonian and overlap */
    //for (int i = N__; i < N__ + n__; i++)
    #pragma omp parallel for
    for (int i = 0; i < N__ + n__; i++)
    {
        std::memcpy(&h_old__(0, i), &h__(0, i), (N__ + n__) * sizeof(double_complex));
        std::memcpy(&o_old__(0, i), &o__(0, i), (N__ + n__) * sizeof(double_complex));
    }
}

template<> 
void Band::set_fv_h_o<CPU, full_potential_lapwlo>(K_point* kp__,
                                                  Periodic_function<double>* effective_potential__,
                                                  dmatrix<double_complex>& h__,
                                                  dmatrix<double_complex>& o__)
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
                    apply_hmt_to_apw<nm>(kp__->num_gkvec_col(), ia, alm_col_tmp, halm_col_tmp);

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
                                                  dmatrix<double_complex>& o__)
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

    cublas_get_matrix(kp__->num_gkvec_row(), kp__->num_gkvec_col(), sizeof(double_complex), h__.at<GPU>(0, 0), h__.ld(), 
                      h__.at<CPU>(), h__.ld());
    
    cublas_get_matrix(kp__->num_gkvec_row(), kp__->num_gkvec_col(), sizeof(double_complex), o__.at<GPU>(0, 0), o__.ld(), 
                      o__.at<CPU>(), o__.ld());
    
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
                             mdarray<double_complex, 2>& o)
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
                         mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
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

void Band::set_fv_h_o_lo_lo(K_point* kp, mdarray<double_complex, 2>& h, mdarray<double_complex, 2>& o)
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
