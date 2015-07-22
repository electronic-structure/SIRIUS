#include "k_point.h"

namespace sirius {

void K_point::initialize()
{
    PROFILE();

    Timer t("sirius::K_point::initialize");
    
    zil_.resize(parameters_.lmax_apw() + 1);
    for (int l = 0; l <= parameters_.lmax_apw(); l++) zil_[l] = std::pow(double_complex(0, 1), l);
   
    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_apw());

    if (use_second_variation) fv_eigen_values_.resize(parameters_.num_fv_states());

    if (use_second_variation && parameters_.need_sv())
    {
        /* in case of collinear magnetism store pure up and pure dn components, otherwise store the full matrix */
        if (parameters_.num_mag_dims() == 3)
        {
            sv_eigen_vectors_[0] = dmatrix<double_complex>(parameters_.num_bands(), parameters_.num_bands(), blacs_grid_,
                                                           parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
        }
        else
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                sv_eigen_vectors_[ispn] = dmatrix<double_complex>(parameters_.num_fv_states(), parameters_.num_fv_states(), 
                                                                  blacs_grid_, parameters_.cyclic_block_size(),
                                                                  parameters_.cyclic_block_size());
            }
        }
    }
    
    /* Find the cutoff for G+k vectors. For pseudopotential calculations this comes 
     * form the input whereas for full-potential calculations this is derived 
     * from rgkmax (aw_cutoff here) and minimal MT radius. */
    double gk_cutoff = 0;
    switch (parameters_.esm_type())
    {
        case ultrasoft_pseudopotential:
        case norm_conserving_pseudopotential:
        {
            gk_cutoff = parameters_.gk_cutoff();
            break;
        }
        case full_potential_lapwlo:
        {
            gk_cutoff = parameters_.aw_cutoff() / unit_cell_.min_mt_radius();
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
    /* initialize phase factors */
    init_gkvec_phase_factors(num_gkvec_row(), gklo_basis_descriptors_row_);
    
    if (parameters_.full_potential())
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
    if (parameters_.esm_type() == full_potential_pwlo)
    {
        /** \todo Correct the memory leak */
        STOP();
        //== sbessel_.resize(num_gkvec_loc()); 
        //== for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        //== {
        //==     sbessel_[igkloc] = new sbessel_pw<double>(parameters_.unit_cell(), parameters_.lmax_pw());
        //==     sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        //== }
    }

    if (parameters_.esm_type() == full_potential_lapwlo)
    {
        alm_coeffs_row_ = new Matching_coefficients(&unit_cell_, parameters_.lmax_apw(), num_gkvec_row(),
                                                    gklo_basis_descriptors_row_);
        alm_coeffs_col_ = new Matching_coefficients(&unit_cell_, parameters_.lmax_apw(), num_gkvec_col(),
                                                    gklo_basis_descriptors_col_);
    }

    /* compute |beta> projectors for atom types */
    if (!parameters_.full_potential())
    {
        generate_beta_gk_t();

        beta_gk_ = matrix<double_complex>(num_gkvec_loc(), unit_cell_.mt_basis_size());

        for (int i = 0; i < unit_cell_.mt_basis_size(); i++)
        {
            int ia = unit_cell_.mt_lo_basis_descriptor(i).ia;
            int xi = unit_cell_.mt_lo_basis_descriptor(i).xi;

            auto atom_type = unit_cell_.atom(ia)->type();

            for (int igk = 0; igk < num_gkvec_loc(); igk++)
            {
                beta_gk_(igk, i) = beta_gk_t_(igk, atom_type->offset_lo() + xi) * conj(gkvec_phase_factors_(igk, ia));
            }
        }

        p_mtrx_ = mdarray<double_complex, 3>(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(), unit_cell_.num_atom_types());
        p_mtrx_.zero();

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++)
        {
            auto atom_type = unit_cell_.atom_type(iat);
            int nbf = atom_type->mt_basis_size();
            int ofs = atom_type->offset_lo();

            matrix<double_complex> qinv(nbf, nbf);
            atom_type->uspp().q_mtrx >> qinv;
            linalg<CPU>::geinv(nbf, qinv);
            
            /* compute P^{+}*P */
            linalg<CPU>::gemm(2, 0, nbf, nbf, num_gkvec_loc(), &beta_gk_t_(0, ofs), beta_gk_t_.ld(), 
                              &beta_gk_t_(0, ofs), beta_gk_t_.ld(), &p_mtrx_(0, 0, iat), p_mtrx_.ld());
            comm_row().allreduce(&p_mtrx_(0, 0, iat), unit_cell_.max_mt_basis_size() * unit_cell_.max_mt_basis_size());

            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                for (int xi2 = 0; xi2 < nbf; xi2++) qinv(xi2, xi1) += p_mtrx_(xi2, xi1, iat);
            }
            /* compute (Q^{-1} + P^{+}*P)^{-1} */
            linalg<CPU>::geinv(nbf, qinv);
            for (int xi1 = 0; xi1 < nbf; xi1++)
            {
                for (int xi2 = 0; xi2 < nbf; xi2++) p_mtrx_(xi2, xi1, iat) = qinv(xi2, xi1);
            }
        }
        
        if (parameters_.processing_unit() == GPU)
        {
            #ifdef __GPU
            gkvec_row_ = mdarray<double, 2>(3, num_gkvec_row());
            /* copy G+k vectors */
            for (int igk_row = 0; igk_row < num_gkvec_row(); igk_row++)
            {
                for (int x = 0; x < 3; x++) gkvec_row_(x, igk_row) = gklo_basis_descriptor_row(igk_row).gkvec[x];
            }
            gkvec_row_.allocate_on_device();
            gkvec_row_.copy_to_device();

            beta_gk_t_.allocate_on_device();
            beta_gk_t_.copy_to_device();
            #endif
        }
    }

    splindex<block_cyclic> spl_bands(parameters_.num_fv_states(), blacs_grid_slice_.comm().size(), blacs_grid_slice_.comm().rank(), 1);
    
    if (parameters_.full_potential())
    {
        spinor_wave_functions_ = mdarray<double_complex, 3>(nullptr, wf_size(), sub_spl_spinor_wf_.local_size(), parameters_.num_spins());
    }
    else
    {
        spinor_wave_functions_ = mdarray<double_complex, 3>(nullptr, wf_size(), spl_bands.local_size(), parameters_.num_spins());
    }

    if (use_second_variation)
    {
        /* allocate memory for first-variational eigen vectors */
        if (parameters_.full_potential())
        {
            fv_eigen_vectors_panel_ = dmatrix<double_complex>(nullptr, gklo_basis_size(), parameters_.num_fv_states(),
                                                              blacs_grid_, parameters_.cyclic_block_size(),
                                                              parameters_.cyclic_block_size());
            fv_eigen_vectors_panel_.allocate(alloc_mode);

            fv_states_ = dmatrix<double_complex>(wf_size(), parameters_.num_fv_states(),
                                                 blacs_grid_,
                                                 parameters_.cyclic_block_size(), parameters_.cyclic_block_size());
        }

        if (!parameters_.full_potential())
        {
            fv_states_ = dmatrix<double_complex>(wf_size(), parameters_.num_fv_states(),
                                                 blacs_grid_slab_,
                                                 (int)splindex_base::block_size(wf_size(), num_ranks()), 1);

            assert(parameters_.num_fv_states() < num_gkvec());
            assert(fv_states_.num_rows_local() == num_gkvec_loc());
            assert(fv_states_.num_cols_local() == parameters_.num_fv_states());

            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                for (int igk = 0; igk < num_gkvec_loc(); igk++) fv_states_(igk, i) = type_wrapper<double_complex>::random();
            }
        }

        if (comm_.size() == 1)
        {
            fv_states_slice_ = dmatrix<double_complex>(fv_states_.at<CPU>(), wf_size(), parameters_.num_fv_states(),
                                                       blacs_grid_slice_, 1, 1);
        }
        else
        {
            fv_states_slice_ = dmatrix<double_complex>(wf_size(), parameters_.num_fv_states(),
                                                       blacs_grid_slice_, 1, 1);
        }

        if (parameters_.need_sv())
        {
            spinor_wave_functions_.allocate();
        }
        else
        {
            //spinor_wave_functions_ = mdarray<double_complex, 3>(fv_states_.at<CPU>(), wf_size(), sub_spl_spinor_wf_.local_size(), parameters_.num_spins());
            spinor_wave_functions_ = mdarray<double_complex, 3>(fv_states_slice_.at<CPU>(), wf_size(), spl_bands.local_size(), parameters_.num_spins());
        }
    }
    else  /* use full diagonalziation */
    {
        if (parameters_.full_potential())
        {
            fd_eigen_vectors_ = mdarray<double_complex, 2>(gklo_basis_size_row(), spl_spinor_wf_.local_size());
            spinor_wave_functions_.allocate();
        }
    }
}

};
