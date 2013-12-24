/** \file band_fd.hpp
    
    \brief Implementation of full-diagonalization methods for Band class.
*/

template <spin_block_t sblock>
void Band::set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<complex16, 2>& alm, 
                        mdarray<complex16, 2>& h)
{
    Timer t("sirius::Band::set_h_apw_lo");
    
    int apw_offset_col = kp->apw_offset_col();
    
    #pragma omp parallel default(shared)
    {
        // apw-lo block
        #pragma omp for
        for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
        {
            int icol = kp->lo_col(ia, i);

            int lm = kp->apwlo_basis_descriptors_col(icol).lm;
            int idxrf = kp->apwlo_basis_descriptors_col(icol).idxrf;
            
            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                        
                complex16 zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm1, lm));
                
                if (abs(zsum) > 1e-14)
                {
                    for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
                }
            }
        }
    }
    
    #pragma omp parallel default(shared)
    {
        std::vector<complex16> ztmp(kp->num_gkvec_col());
        // lo-apw block
        #pragma omp for
        for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
        {
            int irow = kp->lo_row(ia, i);

            int lm = kp->apwlo_basis_descriptors_row(irow).lm;
            int idxrf = kp->apwlo_basis_descriptors_row(irow).idxrf;

            memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(complex16));
        
            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                        
                complex16 zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm, lm1));

                if (abs(zsum) > 1e-14)
                {
                    for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
                        ztmp[igkloc] += zsum * conj(alm(apw_offset_col + igkloc, j1));
                }
            }

            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 
        }
    }
}

void Band::set_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<complex16, 2>& alm, 
                        mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_o_apw_lo");
    
    int apw_offset_col = kp->apw_offset_col();
    
    // apw-lo block
    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
    {
        int icol = kp->lo_col(ia, i);

        int l = kp->apwlo_basis_descriptors_col(icol).l;
        int lm = kp->apwlo_basis_descriptors_col(icol).lm;
        int order = kp->apwlo_basis_descriptors_col(icol).order;
        
        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++)
            {
                o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) * 
                                   alm(igkloc, type->indexb_by_lm_order(lm, order1));
            }
        }
    }

    std::vector<complex16> ztmp(kp->num_gkvec_col());
    // lo-apw block
    for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
    {
        int irow = kp->lo_row(ia, i);

        int l = kp->apwlo_basis_descriptors_row(irow).l;
        int lm = kp->apwlo_basis_descriptors_row(irow).lm;
        int order = kp->apwlo_basis_descriptors_row(irow).order;

        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
            {
                o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) * 
                                   conj(alm(apw_offset_col + igkloc, type->indexb_by_lm_order(lm, order1)));
            }
        }
    }
}

template <spin_block_t sblock>
void Band::set_h_it(K_point* kp, Periodic_function<double>* effective_potential, 
                    Periodic_function<double>* effective_magnetic_field[3], mdarray<complex16, 2>& h)
{
    Timer t("sirius::Band::set_h_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = parameters_.reciprocal_lattice()->index_g12(kp->apwlo_basis_descriptors_row(igk_row).ig,
                                             kp->apwlo_basis_descriptors_col(igk_col).ig);
            
            // pw kinetic energy
            double t1 = 0.5 * Utils::scalar_product(kp->apwlo_basis_descriptors_row(igk_row).gkvec_cart, 
                                                    kp->apwlo_basis_descriptors_col(igk_col).gkvec_cart);
                              
            switch (sblock)
            {
                case nm:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + t1 * parameters_.step_function()->theta_pw(ig12));
                    break;
                }
                case uu:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + effective_magnetic_field[0]->f_pw(ig12) +  
                                            t1 * parameters_.step_function()->theta_pw(ig12));
                    break;
                }
                case dd:
                {
                    h(igk_row, igk_col) += (effective_potential->f_pw(ig12) - effective_magnetic_field[0]->f_pw(ig12) +  
                                            t1 * parameters_.step_function()->theta_pw(ig12));
                    break;
                }
                case ud:
                {
                    h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) - 
                                            complex16(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
                case du:
                {
                    h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) + 
                                            complex16(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
            }
        }
    }
}

void Band::set_o_it(K_point* kp, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_o_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = parameters_.reciprocal_lattice()->index_g12(kp->apwlo_basis_descriptors_row(igk_row).ig,
                                             kp->apwlo_basis_descriptors_col(igk_col).ig);
            
            o(igk_row, igk_col) += parameters_.step_function()->theta_pw(ig12);
        }
    }
}

template <spin_block_t sblock>
void Band::set_h_lo_lo(K_point* kp, mdarray<complex16, 2>& h)
{
    Timer t("sirius::Band::set_h_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = kp->num_gkvec_col(); icol < kp->apwlo_basis_size_col(); icol++)
    {
        int ia = kp->apwlo_basis_descriptors_col(icol).ia;
        int lm2 = kp->apwlo_basis_descriptors_col(icol).lm; 
        int idxrf2 = kp->apwlo_basis_descriptors_col(icol).idxrf; 

        for (int irow = kp->num_gkvec_row(); irow < kp->apwlo_basis_size_row(); irow++)
        {
            if (ia == kp->apwlo_basis_descriptors_row(irow).ia)
            {
                Atom* atom = parameters_.unit_cell()->atom(ia);
                int lm1 = kp->apwlo_basis_descriptors_row(irow).lm; 
                int idxrf1 = kp->apwlo_basis_descriptors_row(irow).idxrf; 

                h(irow, icol) += atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
    }
}

void Band::set_o_lo_lo(K_point* kp, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_o_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = kp->num_gkvec_col(); icol < kp->apwlo_basis_size_col(); icol++)
    {
        int ia = kp->apwlo_basis_descriptors_col(icol).ia;
        int lm2 = kp->apwlo_basis_descriptors_col(icol).lm; 

        for (int irow = kp->num_gkvec_row(); irow < kp->apwlo_basis_size_row(); irow++)
        {
            if (ia == kp->apwlo_basis_descriptors_row(irow).ia)
            {
                Atom* atom = parameters_.unit_cell()->atom(ia);
                int lm1 = kp->apwlo_basis_descriptors_row(irow).lm; 

                if (lm1 == lm2)
                {
                    int l = kp->apwlo_basis_descriptors_row(irow).l;
                    int order1 = kp->apwlo_basis_descriptors_row(irow).order; 
                    int order2 = kp->apwlo_basis_descriptors_col(icol).order; 
                    o(irow, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order2);
                }
            }
        }
    }
}

template <spin_block_t sblock> 
void Band::set_h(K_point* kp, Periodic_function<double>* effective_potential, 
                 Periodic_function<double>* effective_magnetic_field[3], mdarray<complex16, 2>& h)
{
    Timer t("sirius::Band::set_h");
   
    // index of column apw coefficients in apw array
    int apw_offset_col = kp->apw_offset_col();
    
    mdarray<complex16, 2> alm(kp->num_gkvec_loc(), parameters_.unit_cell()->max_mt_aw_basis_size());
    mdarray<complex16, 2> halm(kp->num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());

    h.zero();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        Atom* atom = parameters_.unit_cell()->atom(ia);
        Atom_type* type = atom->type();
       
        // generate conjugated coefficients
        kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
        
        // apply muffin-tin part to <bra|
        apply_hmt_to_apw<sblock>(kp->num_gkvec_row(), ia, alm, halm);
        
        // generate <apw|H|apw> block; |ket> is conjugated, so it is "unconjugated" back
        blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone, 
                        &halm(0, 0), halm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &h(0, 0), h.ld());
       
        // setup apw-lo blocks
        set_h_apw_lo<sblock>(kp, type, atom, ia, alm, h);
    } //ia

    set_h_it<sblock>(kp, effective_potential, effective_magnetic_field, h);

    set_h_lo_lo<sblock>(kp, h);

    alm.deallocate();
    halm.deallocate();
}

void Band::set_o(K_point* kp, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_o");
   
    // index of column apw coefficients in apw array
    int apw_offset_col = kp->apw_offset_col();
    
    mdarray<complex16, 2> alm(kp->num_gkvec_loc(), parameters_.unit_cell()->max_mt_aw_basis_size());
    o.zero();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        Atom* atom = parameters_.unit_cell()->atom(ia);
        Atom_type* type = atom->type();
       
        // generate conjugated coefficients
        kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
        
        // generate <apw|apw> block; |ket> is conjugated, so it is "unconjugated" back
        blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone, 
                        &alm(0, 0), alm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &o(0, 0), o.ld()); 
            
        // setup apw-lo blocks
        set_o_apw_lo(kp, type, atom, ia, alm, o);
    } //ia

    set_o_it(kp, o);

    set_o_lo_lo(kp, o);

    alm.deallocate();
}

void Band::solve_fd(K_point* kp, Periodic_function<double>* effective_potential, 
                    Periodic_function<double>* effective_magnetic_field[3])
{
    Timer t("sirius::Band::solve_fd");

    if (kp->num_ranks() > 1 && (parameters_.eigen_value_solver() == lapack || parameters_.eigen_value_solver() == magma))
        error_local(__FILE__, __LINE__, "Can't use more than one MPI rank for LAPACK or MAGMA eigen-value solver");

    generalized_evp* solver = NULL;

    // create eigen-value solver
    switch (parameters_.eigen_value_solver())
    {
        case lapack:
        {
            solver = new generalized_evp_lapack(-1.0);
            break;
        }
        case scalapack:
        {
            solver = new generalized_evp_scalapack(parameters_.cyclic_block_size(), kp->num_ranks_row(), 
                                                   kp->num_ranks_col(), parameters_.blacs_context(), -1.0);
            break;
        }
        case elpa:
        {
            solver = new generalized_evp_elpa(parameters_.cyclic_block_size(), 
                                              kp->apwlo_basis_size_row(), kp->num_ranks_row(), kp->rank_row(),
                                              kp->apwlo_basis_size_col(), kp->num_ranks_col(), kp->rank_col(), 
                                              parameters_.blacs_context(), 
                                              parameters_.mpi_grid().communicator(1 << _dim_row_),
                                              parameters_.mpi_grid().communicator(1 << _dim_col_),
                                              parameters_.mpi_grid().communicator(1 << _dim_col_ | 1 << _dim_row_));
            break;
        }
        case magma:
        {
            solver = new generalized_evp_magma();
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "eigen value solver is not defined");
        }
    }

    mdarray<complex16, 2> h(kp->apwlo_basis_size_row(), kp->apwlo_basis_size_col());
    mdarray<complex16, 2> o(kp->apwlo_basis_size_row(), kp->apwlo_basis_size_col());
    
    set_o(kp, o);

    std::vector<double> eval(parameters_.num_bands());
    mdarray<complex16, 2>& fd_evec = kp->fd_eigen_vectors();

    Timer t2("sirius::Band::solve_fd|diag", false);

    if (parameters_.num_mag_dims() == 0)
    {
        assert(kp->apwlo_basis_size() >= parameters_.num_fv_states());
        set_h<nm>(kp, effective_potential, effective_magnetic_field, h);
       
        t2.start();
        solver->solve(kp->apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o.get_ptr(), o.ld(), 
                      &eval[0], fd_evec.get_ptr(), fd_evec.ld());
        t2.stop();
    }
    
    if (parameters_.num_mag_dims() == 1)
    {
        assert(kp->apwlo_basis_size() >= parameters_.num_fv_states());

        mdarray<complex16, 2> o1(kp->apwlo_basis_size_row(), kp->apwlo_basis_size_col());
        memcpy(&o1(0, 0), &o(0, 0), o.size() * sizeof(complex16));

        set_h<uu>(kp, effective_potential, effective_magnetic_field, h);
       
        t2.start();
        solver->solve(kp->apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o.get_ptr(), o.ld(), 
                      &eval[0], &fd_evec(0, 0), fd_evec.ld());
        t2.stop();

        set_h<dd>(kp, effective_potential, effective_magnetic_field, h);
        
        t2.start();
        solver->solve(kp->apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o1.get_ptr(), o1.ld(), 
                      &eval[parameters_.num_fv_states()], &fd_evec(0, parameters_.spl_fv_states_col().local_size()), fd_evec.ld());
        t2.stop();
    }

    kp->set_band_energies(&eval[0]);

    delete solver;
}

