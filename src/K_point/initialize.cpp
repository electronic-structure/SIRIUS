#include "k_point.h"

namespace sirius {

void K_point::initialize()
{
    PROFILE_WITH_TIMER("sirius::K_point::initialize");
    
    zil_.resize(ctx_.lmax_apw() + 1);
    for (int l = 0; l <= ctx_.lmax_apw(); l++) zil_[l] = std::pow(double_complex(0, 1), l);
   
    l_by_lm_ = Utils::l_by_lm(ctx_.lmax_apw());

    int bs = ctx_.cyclic_block_size();

    /* In case of collinear magnetism we store only non-zero spinor components.
     *
     * non magnetic case: 
     * +---+
     * |   |
     * +---+
     *
     * collinear case:
     * +---+
     * |uu |
     * +---+---+
     *     |dd |
     *     +---+
     *
     * non collinear case:
     * +-------+
     * |       |
     * +-------+
     * |       |
     * +-------+
     */
    int nst = (ctx_.num_mag_dims() == 3) ? ctx_.num_bands() : ctx_.num_fv_states();

    if (use_second_variation && ctx_.need_sv())
    {
        /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix */
        sv_eigen_vectors_[0] = dmatrix<double_complex>(nst, nst, ctx_.blacs_grid(), bs, bs);
        if (ctx_.num_mag_dims() == 1)
            sv_eigen_vectors_[1] = dmatrix<double_complex>(nst, nst, ctx_.blacs_grid(), bs, bs);
    }

    if (use_second_variation) fv_eigen_values_.resize(ctx_.num_fv_states());

    /* Find the cutoff for G+k vectors. For pseudopotential calculations this comes 
     * form the input whereas for full-potential calculations this is derived 
     * from rgkmax (aw_cutoff here) and minimal MT radius. */
    double gk_cutoff = 0;
    switch (ctx_.esm_type())
    {
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            gk_cutoff = ctx_.gk_cutoff();
            break;
        }
        case full_potential_lapwlo:
        {
            gk_cutoff = ctx_.aw_cutoff() / unit_cell_.min_mt_radius();
            break;
        }
        default:
        {
            STOP();
        }
    }

    /* Build a full list of G+k vectors for all MPI ranks */
    generate_gkvec(gk_cutoff);
    /* build a list of basis functions */
    build_gklo_basis_descriptors();
    /* distribute basis functions */
    distribute_basis_index();
    
    if (ctx_.full_potential())
    {
        atom_lo_cols_.clear();
        atom_lo_cols_.resize(unit_cell_.num_atoms());

        atom_lo_rows_.clear();
        atom_lo_rows_.resize(unit_cell_.num_atoms());

        for (int icol = num_gkvec_col(); icol < gklo_basis_size_col(); icol++)
        {
            int ia = gklo_basis_descriptor_col(icol).ia;
            atom_lo_cols_[ia].push_back(icol);
        }
        
        for (int irow = num_gkvec_row(); irow < gklo_basis_size_row(); irow++)
        {
            int ia = gklo_basis_descriptor_row(irow).ia;
            atom_lo_rows_[ia].push_back(irow);
        }
    }
    if (ctx_.esm_type() == full_potential_pwlo)
    {
        /** \todo Correct the memory leak */
        STOP();
        //== sbessel_.resize(num_gkvec_loc()); 
        //== for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        //== {
        //==     sbessel_[igkloc] = new sbessel_pw<double>(ctx_.unit_cell(), ctx_.lmax_pw());
        //==     sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        //== }
    }

    if (ctx_.esm_type() == full_potential_lapwlo)
    {
        alm_coeffs_ = new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec(),
                                                gklo_basis_descriptors_);
        alm_coeffs_row_ = new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_row(),
                                                    gklo_basis_descriptors_row_);
        alm_coeffs_col_ = new Matching_coefficients(unit_cell_, ctx_.lmax_apw(), num_gkvec_col(),
                                                    gklo_basis_descriptors_col_);
    }

    /* compute |beta> projectors for atom types */
    if (!ctx_.full_potential())
    {
        beta_projectors_ = new Beta_projectors(comm_, unit_cell_, gkvec_, ctx_.processing_unit());
        
        if (false)
        {
            //p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), unit_cell_.num_atom_types());
            //p_mtrx_.zero();

            //for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
            //{
            //    auto atom_type = unit_cell_.atom_type(iat);
            //    int nbf = atom_type->mt_basis_size();
            //    int ofs = atom_type->offset_lo();

            //    matrix<double_complex> qinv(nbf, nbf);
            //    atom_type->uspp().q_mtrx >> qinv;
            //    linalg<CPU>::geinv(nbf, qinv);
            //    
            //    /* compute P^{+}*P */
            //    linalg<CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(), &beta_gk_t_(0, ofs), beta_gk_t_.ld(), 
            //                      &beta_gk_t_(0, ofs), beta_gk_t_.ld(), &p_mtrx_(0, 0, iat), p_mtrx_.ld());
            //    comm_row().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size());

            //    for (int xi1 = 0; xi1 < nbf; xi1++)
            //    {
            //        for (int xi2 = 0; xi2 < nbf; xi2++) qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
            //    }
            //    /* compute (Q^{-1} + P^{+}*P)^{-1} */
            //    linalg<CPU>::geinv(nbf, qinv);
            //    for (int xi1 = 0; xi1 < nbf; xi1++)
            //    {
            //        for (int xi2 = 0; xi2 < nbf; xi2++) p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
            //    }
            //}
        }
    }

    if (ctx_.full_potential())
    {
        if (use_second_variation)
        {
            fv_eigen_vectors_ = new Wave_functions<true>(gklo_basis_size(), ctx_.num_fv_states(), bs, ctx_.blacs_grid(), ctx_.blacs_grid_slice());

            fv_states_ = new Wave_functions<true>(wf_size(), ctx_.num_fv_states(), bs, ctx_.blacs_grid(), ctx_.blacs_grid_slice());

            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
            {
                spinor_wave_functions_[ispn] = new Wave_functions<true>(wf_size(), nst, bs, ctx_.blacs_grid(), ctx_.blacs_grid_slice());
            }
        }
        else
        {
            TERMINATE_NOT_IMPLEMENTED
        //    fd_eigen_vectors_ = mdarray<double_complex, 2>(gklo_basis_size_row(), spl_spinor_wf_.local_size());
        //    spinor_wave_functions_.allocate();
        }
    }
    else
    {
        assert(ctx_.num_fv_states() < num_gkvec());

        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
        {
            spinor_wave_functions_[ispn] = new Wave_functions<false>(nst, nst, gkvec_, ctx_.mpi_grid_fft(),
                                                                     ctx_.processing_unit());

            for (int i = 0; i < nst; i++) // TODO: init from atomic WFs
            {
                double norm = 1.0 / std::sqrt(gkvec_.num_gvec());
                for (int igk = 0; igk < num_gkvec_loc(); igk++) spinor_wave_functions<false>(ispn)(igk, i) = type_wrapper<double_complex>::random() * norm;
            }
        }
    }
}

};
