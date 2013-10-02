/** \file band.hpp
    
    \brief Implementation of methods for Band class.
*/

void Band::apply_magnetic_field(mdarray<complex16, 2>& fv_states, int mtgk_size, int num_gkvec, int* fft_index, 
                                Periodic_function<double>* effective_magnetic_field[3], mdarray<complex16, 3>& hpsi)
{
    assert(hpsi.size(2) >= 2);
    assert(fv_states.size(0) == hpsi.size(0));
    assert(fv_states.size(1) == hpsi.size(1));

    Timer t("sirius::Band::apply_magnetic_field");

    complex16 zzero = complex16(0, 0);
    complex16 zone = complex16(1, 0);
    complex16 zi = complex16(0, 1);

    mdarray<complex16, 3> zm(parameters_.max_mt_basis_size(), parameters_.max_mt_basis_size(), 
                             parameters_.num_mag_dims());

    // column fv states are further distributed over rows to make use of all row processors
    int num_fv_local = parameters_.sub_spl_fv_states_col().local_size();
    int idx_fv_local = parameters_.sub_spl_fv_states_col().global_offset();
            
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        int offset = parameters_.atom(ia)->offset_wf();
        int mt_basis_size = parameters_.atom(ia)->type()->mt_basis_size();
        Atom* atom = parameters_.atom(ia);
        
        zm.zero();

        #pragma omp parallel for default(shared)
        for (int j2 = 0; j2 < mt_basis_size; j2++)
        {
            int lm2 = atom->type()->indexb(j2).lm;
            int idxrf2 = atom->type()->indexb(j2).idxrf;
            
            for (int i = 0; i < parameters_.num_mag_dims(); i++)
            {
                for (int j1 = 0; j1 <= j2; j1++)
                {
                    int lm1 = atom->type()->indexb(j1).lm;
                    int idxrf1 = atom->type()->indexb(j1).idxrf;

                    zm(j1, j2, i) = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2, atom->b_radial_integrals(idxrf1, idxrf2, i)); 
                }
            }
        }
        // compute bwf = B_z*|wf_j>
        blas<cpu>::hemm(0, 0, mt_basis_size, num_fv_local, zone, &zm(0, 0, 0), zm.ld(), 
                        &fv_states(offset, idx_fv_local), fv_states.ld(), zzero, 
                        &hpsi(offset, idx_fv_local, 0), hpsi.ld());
        
        // compute bwf = (B_x - iB_y)|wf_j>
        if (hpsi.size(2) >= 3)
        {
            // reuse first (z) component of zm matrix to store (Bx - iBy)
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) - zi * zm(j1, j2, 2);
                
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) - zi * conj(zm(j2, j1, 2));
            }
              
            blas<cpu>::gemm(0, 0, mt_basis_size, num_fv_local, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                            &fv_states(offset, idx_fv_local), fv_states.ld(), 
                            &hpsi(offset, idx_fv_local, 2), hpsi.ld());
        }
        
        // compute bwf = (B_x + iB_y)|wf_j>
        if ((hpsi.size(2) == 4) && 
            (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa))
        {
            // reuse first (z) component of zm matrix to store (Bx + iBy)
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) + zi * zm(j1, j2, 2);
                
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) + zi * conj(zm(j2, j1, 2));
            }
              
            blas<cpu>::gemm(0, 0, mt_basis_size, num_fv_local, mt_basis_size, &zm(0, 0, 0), zm.ld(), 
                            &fv_states(offset, idx_fv_local), fv_states.ld(), 
                            &hpsi(offset, idx_fv_local, 3), hpsi.ld());
        }
    }
    
    Timer *t1 = new Timer("sirius::Band::apply_magnetic_field:it");

    mdarray<complex16, 3> hpsi_pw(num_gkvec, parameters_.spl_fv_states_col().local_size(), hpsi.size(2));
    hpsi_pw.zero();

    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        
        std::vector<complex16> psi_it(parameters_.fft().size());
        std::vector<complex16> hpsi_it(parameters_.fft().size());
        
        #pragma omp for
        for (int iloc = 0; iloc < num_fv_local; iloc++)
        {
            int i = parameters_.sub_spl_fv_states_col(iloc);

            parameters_.fft().input(num_gkvec, fft_index, &fv_states(parameters_.mt_basis_size(), i), thread_id);
            parameters_.fft().transform(1, thread_id);
            parameters_.fft().output(&psi_it[0], thread_id);
                                        
            for (int ir = 0; ir < parameters_.fft().size(); ir++)
            {
                // hpsi(r) = psi(r) * Bz(r) * Theta(r)
                hpsi_it[ir] = psi_it[ir] * effective_magnetic_field[0]->f_it<global>(ir) * parameters_.step_function(ir);
            }
            
            parameters_.fft().input(&hpsi_it[0], thread_id);
            parameters_.fft().transform(-1, thread_id);
            parameters_.fft().output(num_gkvec, fft_index, &hpsi_pw(0, i, 0), thread_id); 

            if (hpsi.size(2) >= 3)
            {
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                {
                    // hpsi(r) = psi(r) * (Bx(r) - iBy(r)) * Theta(r)
                    hpsi_it[ir] = psi_it[ir] * parameters_.step_function(ir) * 
                                  (effective_magnetic_field[1]->f_it<global>(ir) - 
                                   zi * effective_magnetic_field[2]->f_it<global>(ir));
                }
                
                parameters_.fft().input(&hpsi_it[0], thread_id);
                parameters_.fft().transform(-1, thread_id);
                parameters_.fft().output(num_gkvec, fft_index, &hpsi_pw(0, i, 2), thread_id); 
            }
            
            if ((hpsi.size(2)) == 4 && 
                (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa))
            {
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                {
                    // hpsi(r) = psi(r) * (Bx(r) + iBy(r)) * Theta(r)
                    hpsi_it[ir] = psi_it[ir] * parameters_.step_function(ir) *
                                  (effective_magnetic_field[1]->f_it<global>(ir) + 
                                   zi * effective_magnetic_field[2]->f_it<global>(ir));
                }
                
                parameters_.fft().input(&hpsi_it[0], thread_id);
                parameters_.fft().transform(-1, thread_id);
                parameters_.fft().output(num_gkvec, fft_index, &hpsi_pw(0, i, 3), thread_id); 
            }
        }
    }
    delete t1;

    for (int n = 0; n < hpsi.size(2); n++)
    {
        if (n != 1)
        {
            for (int iloc = 0; iloc < num_fv_local; iloc++)
            {
                int i = parameters_.sub_spl_fv_states_col(iloc);
                memcpy(&hpsi(parameters_.mt_basis_size(), i, n), &hpsi_pw(0, i, n), num_gkvec * sizeof(complex16));
            }
        }
        else
        {
            // copy Bz|\psi> to -Bz|\psi>
            for (int iloc = 0; iloc < num_fv_local; iloc++)
            {
                int i = parameters_.sub_spl_fv_states_col(iloc);
                for (int j = 0; j < mtgk_size; j++) hpsi(j, i, 1) = -hpsi(j, i, 0);
            }
        }
    }

    for (int n = 0; n < hpsi.size(2); n++)
    {
        Platform::allgather(&hpsi(0, 0, n), hpsi.size(0) * idx_fv_local, hpsi.size(0) * num_fv_local, 
                            parameters_.mpi_grid().communicator(1 << _dim_row_));
    }
}

void Band::apply_so_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi)
{
    Timer t("sirius::Band::apply_so_correction");

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom_type* type = parameters_.atom(ia)->type();

        int offset = parameters_.atom(ia)->offset_wf();

        for (int l = 0; l <= parameters_.lmax_apw(); l++)
        {
            int nrf = type->indexr().num_rf(l);

            for (int order1 = 0; order1 < nrf; order1++)
            {
                for (int order2 = 0; order2 < nrf; order2++)
                {
                    double sori = parameters_.atom(ia)->symmetry_class()->so_radial_integral(l, order1, order2);
                    
                    for (int m = -l; m <= l; m++)
                    {
                        int idx1 = type->indexb_by_l_m_order(l, m, order1);
                        int idx2 = type->indexb_by_l_m_order(l, m, order2);
                        int idx3 = (m + l != 0) ? type->indexb_by_l_m_order(l, m - 1, order2) : 0;
                        int idx4 = (m - l != 0) ? type->indexb_by_l_m_order(l, m + 1, order2) : 0;

                        for (int ist = 0; ist < parameters_.spl_fv_states_col().local_size(); ist++)
                        {
                            complex16 z1 = fv_states(offset + idx2, ist) * double(m) * sori;
                            hpsi(offset + idx1, ist, 0) += z1;
                            hpsi(offset + idx1, ist, 1) -= z1;
                            // apply L_{-} operator
                            if (m + l) hpsi(offset + idx1, ist, 2) += fv_states(offset + idx3, ist) * sori * 
                                                                      sqrt(double(l * (l + 1) - m * (m - 1)));
                            // apply L_{+} operator
                            if (m - l) hpsi(offset + idx1, ist, 3) += fv_states(offset + idx4, ist) * sori * 
                                                                      sqrt(double(l * (l + 1) - m * (m + 1)));
                        }
                    }
                }
            }
        }
    }
}

template <spin_block_t sblock>
void Band::apply_uj_correction(mdarray<complex16, 2>& fv_states, mdarray<complex16, 3>& hpsi)
{
    Timer t("sirius::Band::apply_uj_correction");

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        if (parameters_.atom(ia)->apply_uj_correction())
        {
            Atom_type* type = parameters_.atom(ia)->type();

            int offset = parameters_.atom(ia)->offset_wf();

            int l = parameters_.atom(ia)->uj_correction_l();

            int nrf = type->indexr().num_rf(l);

            for (int order2 = 0; order2 < nrf; order2++)
            {
                for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                {
                    int idx2 = type->indexb_by_lm_order(lm2, order2);
                    for (int order1 = 0; order1 < nrf; order1++)
                    {
                        double ori = parameters_.atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
                        
                        for (int ist = 0; ist < parameters_.spl_fv_states_col().local_size(); ist++)
                        {
                            for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                            {
                                int idx1 = type->indexb_by_lm_order(lm1, order1);
                                complex16 z1 = fv_states(offset + idx1, ist) * ori;

                                if (sblock == uu)
                                {
                                    hpsi(offset + idx2, ist, 0) += z1 * 
                                        parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);
                                }

                                if (sblock == dd)
                                {
                                    hpsi(offset + idx2, ist, 1) += z1 *
                                        parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);
                                }

                                if (sblock == ud)
                                {
                                    hpsi(offset + idx2, ist, 2) += z1 *
                                        parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
                                }
                                
                                if (sblock == du)
                                {
                                    hpsi(offset + idx2, ist, 3) += z1 *
                                        parameters_.atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Band::solve_sv(K_point* kp, Periodic_function<double>* effective_magnetic_field[3])
{
    Timer t("sirius::Band::solve_sv");

    if (!parameters_.need_sv())
    {
        kp->bypass_sv();
        return;
    }
    
    if (kp->num_ranks() > 1 && (parameters_.eigen_value_solver() == lapack || parameters_.eigen_value_solver() == magma))
        error_local(__FILE__, __LINE__, "Can't use more than one MPI rank for LAPACK or MAGMA eigen-value solver");

    // number of h|\psi> components 
    int nhpsi = parameters_.num_mag_dims() + 1;

    int nrow = parameters_.spl_fv_states_row().local_size();
    int ncol = parameters_.spl_fv_states_col().local_size();
    int fvsz = kp->mtgk_size();

    mdarray<complex16, 2>& fv_states_row = kp->fv_states_row();
    mdarray<complex16, 2>& sv_eigen_vectors = kp->sv_eigen_vectors();
    std::vector<double> band_energies(parameters_.num_bands());

    // product of the second-variational Hamiltonian and a wave-function
    mdarray<complex16, 3> hpsi(fvsz, ncol, nhpsi);
    hpsi.zero();

    // compute product of magnetic field and wave-function 
    if (parameters_.num_spins() == 2)
        apply_magnetic_field(kp->fv_states_col(), kp->mtgk_size(), kp->num_gkvec(), kp->fft_index(), effective_magnetic_field, hpsi);

    if (parameters_.uj_correction())
    {
        apply_uj_correction<uu>(kp->fv_states_col(), hpsi);
        if (parameters_.num_mag_dims() != 0) apply_uj_correction<dd>(kp->fv_states_col(), hpsi);
        if (parameters_.num_mag_dims() == 3) 
        {
            apply_uj_correction<ud>(kp->fv_states_col(), hpsi);
            if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
                apply_uj_correction<du>(kp->fv_states_col(), hpsi);
        }
    }

    if (parameters_.so_correction()) apply_so_correction(kp->fv_states_col(), hpsi);

    Timer t1("sirius::Band::solve_sv:stdevp", false);

    standard_evp* solver = NULL;
    switch (parameters_.eigen_value_solver())
    {
        case lapack:
        {
            solver = new standard_evp_lapack();
            break;
        }
        case scalapack:
        {
            solver = new standard_evp_scalapack(parameters_.cyclic_block_size(), kp->num_ranks_row(), 
                                                kp->num_ranks_col(), parameters_.blacs_context());
            break;
        }
        case elpa:
        {
            solver = new standard_evp_scalapack(parameters_.cyclic_block_size(), kp->num_ranks_row(), 
                                                kp->num_ranks_col(), parameters_.blacs_context());
            break;
        }
        case magma:
        {
            solver = new standard_evp_lapack();
            break;
        }
        default:
        {
            error_local(__FILE__, __LINE__, "eigen value solver is not defined");
        }
    }
    
    if (parameters_.num_mag_dims() == 1)
    {
        mdarray<complex16, 2> h(nrow, ncol);
        
        //perform two consecutive diagonalizations
        for (int ispn = 0; ispn < 2; ispn++)
        {
            // compute <wf_i | (h * wf_j)> for up-up or dn-dn block
            blas<cpu>::gemm(2, 0, nrow, ncol, fvsz, &fv_states_row(0, 0), fv_states_row.ld(), 
                            &hpsi(0, 0, ispn), hpsi.ld(), &h(0, 0), h.ld());

            for (int icol = 0; icol < ncol; icol++)
            {
                int i = parameters_.spl_fv_states_col(icol);
                for (int irow = 0; irow < nrow; irow++)
                {
                    if (parameters_.spl_fv_states_row(irow) == i) h(irow, icol) += kp->fv_eigen_value(i);
                }
            }
        
            t1.start();
            solver->solve(parameters_.num_fv_states(), h.get_ptr(), h.ld(),
                          &band_energies[ispn * parameters_.num_fv_states()],
                          &sv_eigen_vectors(0, ispn * ncol), sv_eigen_vectors.ld());
            t1.stop();
        }
    }

    if (parameters_.num_mag_dims() == 3)
    {
        mdarray<complex16, 2> h(2 * nrow, 2 * ncol);
        h.zero();

        // compute <fv_i | (h * fv_j)> for up-up block
        blas<cpu>::gemm(2, 0, nrow, ncol, fvsz, &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 0), hpsi.ld(), 
                        &h(0, 0), h.ld());

        // compute <fv_i | (h * fv_j)> for up-dn block
        blas<cpu>::gemm(2, 0, nrow, ncol, fvsz, &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 2), hpsi.ld(), 
                        &h(0, ncol), h.ld());
       
        // compute <fv_i | (h * fv_j)> for dn-dn block
        blas<cpu>::gemm(2, 0, nrow, ncol, fvsz, &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 1), hpsi.ld(), 
                        &h(nrow, ncol), h.ld());

        if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
        {
            // compute <fv_i | (h * fv_j)> for dn-up block
            blas<cpu>::gemm(2, 0, nrow, ncol, fvsz, &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 3), hpsi.ld(), 
                            &h(nrow, 0), h.ld());
        }
      
        for (int ispn = 0; ispn < 2; ispn++)
        {
            for (int icol = 0; icol < ncol; icol++)
            {
                int i = parameters_.spl_fv_states_col(icol) + ispn * parameters_.num_fv_states();
                for (int irow = 0; irow < nrow; irow++)
                {
                    int j = parameters_.spl_fv_states_row(irow) + ispn * parameters_.num_fv_states();
                    if (j == i) 
                    {
                        h(irow + ispn * nrow, icol + ispn * ncol) += kp->fv_eigen_value(parameters_.spl_fv_states_col(icol));
                    }
                }
            }
        }
    
        t1.start();
        solver->solve(parameters_.num_bands(), h.get_ptr(), h.ld(), &band_energies[0], 
                      sv_eigen_vectors.get_ptr(), sv_eigen_vectors.ld());
        t1.stop();
    }
    delete solver;

    kp->set_band_energies(&band_energies[0]);
}

void Band::apply_hmt_to_apw(int num_gkvec, int ia, mdarray<complex16, 2>& alm, mdarray<complex16, 2>& halm)
{
    Timer t("sirius::Band::apply_hmt_to_apw");
    
    Atom* atom = parameters_.atom(ia);
    Atom_type* type = atom->type();
    
    #pragma omp parallel default(shared)
    {
        std::vector<complex16> zv(num_gkvec);
        
        #pragma omp for
        for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
        {
            memset(&zv[0], 0, num_gkvec * sizeof(complex16));

            int lm2 = type->indexb(j2).lm;
            int idxrf2 = type->indexb(j2).idxrf;

            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                
                complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2,  
                                                                          atom->h_radial_integrals(idxrf1, idxrf2));
                
                if (abs(zsum) > 1e-14) 
                {
                    for (int ig = 0; ig < num_gkvec; ig++) zv[ig] += zsum * alm(ig, j1); 
                }
            } // j1
             
            int l2 = type->indexb(j2).l;
            int order2 = type->indexb(j2).order;
            
            for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
            {
                double t1 = 0.5 * pow(type->mt_radius(), 2) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order1, 0) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order2, 1);
                
                for (int ig = 0; ig < num_gkvec; ig++) 
                    zv[ig] += t1 * alm(ig, type->indexb_by_lm_order(lm2, order1));
            }
            
            memcpy(&halm(0, j2), &zv[0], num_gkvec * sizeof(complex16));
        } // j2
    }
}

//=====================================================================================================================
// CPU code, plane-wave basis
//=====================================================================================================================
//** template<> void Band::set_fv_h_o<cpu, pwlo>(K_point* kp, Periodic_function<double>* effective_potential, 
//**                                             mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** {
//**     Timer t("sirius::Band::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < k->num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < kp->num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (kp->apwlo_basis_descriptors_row(igkloc1).idxglob == apwlo_basis_descriptors_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = complex16(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
//**                                              apwlo_basis_descriptors_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<cpu>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//**
//** template<> void K_point::set_fv_h_o_pw_lo<cpu>(Periodic_function<double>* effective_potential, int num_ranks, 
//**                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o_pw_lo");
//**     
//**     int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
//**     
//**     mdarray<Spline<complex16>*, 2> svlo(parameters_.lmmax_pw(), std::max(num_lo_col(), num_lo_row()));
//** 
//**     // first part: compute <G+k|H|lo> and <G+k|lo>
//** 
//**     Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
//**     Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
//**     Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);
//** 
//**     // compute V|lo>
//**     t1.start();
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
//**         int lm = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].lm;
//**         int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             svlo(lm1, icol) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
//**                                                     parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     (*svlo(lm1, icol))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
//**                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//** 
//**             svlo(lm1, icol)->interpolate();
//**         }
//**     }
//**     t1.stop();
//**     
//**     t2.start();
//**     // compute overlap and kinetic energy
//**     for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**     {
//**         int ia = apwlo_basis_descriptors_col_[icol].ia;
//**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int l = apwlo_basis_descriptors_col_[icol].l;
//**         int lm = apwlo_basis_descriptors_col_[icol].lm;
//**         int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;
//** 
//**         Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < slo.num_points(); ir++)
//**             slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         slo.interpolate();
//**         
//**         #pragma omp parallel for default(shared)
//**         for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**         {
//**             o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**                               Spline<double>::integrate(&slo, (*sbessel_[igkloc])(l, iat)) * 
//**                               conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//**         }
//**     }
//**     t2.stop();
//** 
//**     t3.start();
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**         {
//**             int ia = apwlo_basis_descriptors_col_[icol].ia;
//**             int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**             //int l = apwlo_basis_descriptors_col_[icol].l;
//**             //int lm = apwlo_basis_descriptors_col_[icol].lm;
//**             //int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;
//** 
//**             //*// compue overlap <G+k|lo>
//**             //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**             //*for (int ir = 0; ir < s.num_points(); ir++)
//**             //*{
//**             //*    s[ir] = (*sbessel_[igkloc])(ir, l, iat) * 
//**             //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**             //*}
//**             //*s.interpolate();
//**             //*    
//**             //*o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**             //*                  s.integrate(2) * conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             //*// kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             //*h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//** 
//**             // add <G+k|V|lo>
//**             complex16 zt1(0, 0);
//**             for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**             {
//**                 for (int m1 = -l1; m1 <= l1; m1++)
//**                 {
//**                     int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**                     zt1 += Spline<complex16>::integrate(svlo(lm1, icol - num_gkvec_col()), 
//**                                                         (*sbessel_[igkloc])(l1, iat)) * 
//**                            conj(zil_[l1]) * gkvec_ylm_(lm1, igkloc);
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
//**             h(igkloc, icol) += zt1;
//**         }
//**     }
//**     t3.stop();
//**    
//**     // deallocate V|lo>
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, icol);
//**     }
//** 
//**     // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
//**     if (num_ranks == 1)
//**     {
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**             {
//**                 h(irow, igkloc) = conj(h(igkloc, irow));
//**                 o(irow, igkloc) = conj(o(igkloc, irow));
//**             }
//**         }
//**         return;
//**     }
//** 
//**     // second part: compute <lo|H|G+k> and <lo|G+k>
//** 
//**     // compute V|lo>
//**     t1.start();
//**     for (int irow = 0; irow < num_lo_row(); irow++)
//**     {
//**         int ia = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].ia;
//**         int lm = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].lm;
//**         int idxrf = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             svlo(lm1, irow) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
//**                                                     parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
//**                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//** 
//**             svlo(lm1, irow)->interpolate();
//**         }
//**     }
//**     t1.stop();
//**    
//**     t2.start();
//**     for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     {
//**         int ia = apwlo_basis_descriptors_row_[irow].ia;
//**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int l = apwlo_basis_descriptors_row_[irow].l;
//**         int lm = apwlo_basis_descriptors_row_[irow].lm;
//**         int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
//** 
//**         // compue overlap <lo|G+k>
//**         Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < slo.num_points(); ir++)
//**             slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         slo.interpolate();
//**         
//**         #pragma omp parallel for default(shared)
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**                               conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
//**                               Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
//**                               gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**             // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**             h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//**         }
//**     }
//**     t2.stop();
//** 
//**     t3.start();
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     {
//**         for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**         {
//**             int ia = apwlo_basis_descriptors_row_[irow].ia;
//**             int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**             //int l = apwlo_basis_descriptors_row_[irow].l;
//**             //int lm = apwlo_basis_descriptors_row_[irow].lm;
//**             //int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
//** 
//**             //*// compue overlap <lo|G+k>
//**             //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**             //*for (int ir = 0; ir < s.num_points(); ir++)
//**             //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
//**             //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**             //*s.interpolate();
//**             //*    
//**             //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**             //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
//**             //*                  gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**             //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**             //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//** 
//**             // add <lo|V|G+k>
//**             complex16 zt1(0, 0);
//**             for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**             {
//**                 for (int m1 = -l1; m1 <= l1; m1++)
//**                 {
//**                     int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**                     zt1 += conj(Spline<complex16>::integrate(svlo(lm1, irow - num_gkvec_row()), 
//**                                                              (*sbessel_[offset_col + igkloc])(l1, iat))) * 
//**                            zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
//**             h(irow, igkloc) += zt1;
//**         }
//**     }
//**     t3.stop();
//**     
//**     for (int irow = 0; irow < num_lo_row(); irow++)
//**     {
//**         for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
//**     }
//** }
//** 
//** template<> void K_point::set_fv_h_o<cpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
//**                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (apwlo_basis_descriptors_row_[igkloc1].idxglob == apwlo_basis_descriptors_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = complex16(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
//**                                              apwlo_basis_descriptors_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<cpu>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//** 
//** 


//=====================================================================================================================
// GPU code, plane-wave basis
//=====================================================================================================================
//** template<> void K_point::set_fv_h_o<gpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
//**                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o");
//**     
//**     h.zero();
//**     o.zero();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
//**     {
//**         for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
//**         {
//**             if (apwlo_basis_descriptors_row_[igkloc1].idxglob == apwlo_basis_descriptors_col_[igkloc2].idxglob) 
//**             {
//**                 h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
//**                 o(igkloc1, igkloc2) = complex16(1, 0);
//**             }
//**                                
//**             int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
//**                                              apwlo_basis_descriptors_col_[igkloc2].ig);
//**             h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
//**         }
//**     }
//**     
//**     set_fv_h_o_pw_lo<gpu>(effective_potential, num_ranks, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** }
//** #endif
//** 
//** template<> void K_point::set_fv_h_o_pw_lo<gpu>(Periodic_function<double>* effective_potential, int num_ranks, 
//**                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** {
//**     Timer t("sirius::K_point::set_fv_h_o_pw_lo");
//**     
//**     // ===========================================
//**     // first part: compute <G+k|H|lo> and <G+k|lo>
//**     // ===========================================
//** 
//**     Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
//**     Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
//**     Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);
//** 
//**     mdarray<int, 1> kargs(4);
//**     kargs(0) = parameters_.num_atom_types();
//**     kargs(1) = parameters_.max_num_mt_points();
//**     kargs(2) = parameters_.lmax_pw();
//**     kargs(3) = parameters_.lmmax_pw();
//**     kargs.allocate_on_device();
//**     kargs.copy_to_device();
//** 
//**     // =========================
//**     // compute V|lo> for columns
//**     // =========================
//**     t1.start();
//**     mdarray<complex16, 3> vlo_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmmax_pw(), num_lo_col());
//**     #pragma omp parallel for default(shared)
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
//**         int lm = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].lm;
//**         int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;
//**         
//**         for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**         {
//**             Spline<complex16> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//** 
//**             for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**             {
//**                 int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**                 complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**                 for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**                 {
//**                     s[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
//**                               parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**                 }
//**             }
//**             s.interpolate();
//**             s.get_coefs(&vlo_coefs(0, lm1, icol), parameters_.max_num_mt_points());
//**         }
//**     }
//**     vlo_coefs.pin_memory();
//**     vlo_coefs.allocate_on_device();
//**     vlo_coefs.async_copy_to_device(-1);
//**     t1.stop();
//**     
//**     // ===========================================
//**     // pack Bessel function splines into one array
//**     // ===========================================
//**     mdarray<double, 4> sbessel_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmax_pw() + 1, 
//**                                      parameters_.num_atom_types(), num_gkvec_row());
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//**         {
//**             for (int l = 0; l <= parameters_.lmax_pw(); l++)
//**             {
//**                 (*sbessel_[igkloc])(l, iat)->get_coefs(&sbessel_coefs(0, l, iat, igkloc), 
//**                                                        parameters_.max_num_mt_points());
//**             }
//**         }
//**     }
//**     sbessel_coefs.pin_memory();
//**     sbessel_coefs.allocate_on_device();
//**     sbessel_coefs.async_copy_to_device(-1);
//** 
//**     // ==============================
//**     // pack lo splines into one array
//**     // ==============================
//**     mdarray<double, 2> lo_coefs(parameters_.max_num_mt_points() * 4, num_lo_col());
//**     mdarray<int, 1> l_by_ilo(num_lo_col());
//**     mdarray<int, 1> iat_by_ilo(num_lo_col());
//**     for (int icol = 0; icol < num_lo_col(); icol++)
//**     {
//**         int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
//**         l_by_ilo(icol) = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].l;
//**         iat_by_ilo(icol) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**         int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;
//** 
//**         Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**         for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**             s[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**         s.interpolate();
//**         s.get_coefs(&lo_coefs(0, icol), parameters_.max_num_mt_points());
//**     }
//**     lo_coefs.pin_memory();
//**     lo_coefs.allocate_on_device();
//**     lo_coefs.async_copy_to_device(-1);
//**     l_by_ilo.allocate_on_device();
//**     l_by_ilo.async_copy_to_device(-1);
//**     iat_by_ilo.allocate_on_device();
//**     iat_by_ilo.async_copy_to_device(-1);
//** 
//**     // ============
//**     // radial grids
//**     // ============
//**     mdarray<double, 2> r_dr(parameters_.max_num_mt_points() * 2, parameters_.num_atom_types());
//**     mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
//**     for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//**     {
//**         nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
//**         parameters_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), parameters_.max_num_mt_points());
//**     }
//**     r_dr.allocate_on_device();
//**     r_dr.async_copy_to_device(-1);
//**     nmtp_by_iat.allocate_on_device();
//**     nmtp_by_iat.async_copy_to_device(-1);
//** 
//**     mdarray<double, 2> jlo(num_gkvec_row(), num_lo_col());
//**     jlo.allocate_on_device();
//** 
//**     t2.start();
//**     sbessel_lo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), l_by_ilo.get_ptr_device(), 
//**                                  iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), r_dr.get_ptr_device(), 
//**                                  sbessel_coefs.get_ptr_device(), lo_coefs.get_ptr_device(), jlo.get_ptr_device());
//**     jlo.copy_to_host();
//**     // compute overlap and kinetic energy
//**     for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**     {
//**         int ia = apwlo_basis_descriptors_col_[icol].ia;
//**         int l = apwlo_basis_descriptors_col_[icol].l;
//**         int lm = apwlo_basis_descriptors_col_[icol].lm;
//** 
//**         for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**         {
//**             o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
//**                               jlo(igkloc, icol - num_gkvec_col()) * conj(gkvec_phase_factors_(igkloc, ia));
//** 
//**             // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
//**             h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
//**         }
//**     }
//**     t2.stop();
//** 
//**     l_by_lm_.allocate_on_device();
//**     l_by_lm_.copy_to_device();
//** 
//**     mdarray<complex16, 3> jvlo(parameters_.lmmax_pw(), num_gkvec_row(), num_lo_col());
//**     jvlo.allocate_on_device();
//**     
//**     t3.start();
//**     sbessel_vlo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), parameters_.lmmax_pw(), 
//**                                   l_by_lm_.get_ptr_device(), iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), 
//**                                   r_dr.get_ptr_device(), sbessel_coefs.get_ptr_device(), vlo_coefs.get_ptr_device(), 
//**                                   jvlo.get_ptr_device());
//**     jvlo.copy_to_host();
//** 
//**     l_by_lm_.deallocate_on_device();
//** 
//**     #pragma omp parallel for default(shared)
//**     for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//**     {
//**         for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
//**         {
//**             int ia = apwlo_basis_descriptors_col_[icol].ia;
//** 
//**             // add <G+k|V|lo>
//**             complex16 zt1(0, 0);
//**             for (int l = 0; l <= parameters_.lmax_pw(); l++)
//**             {
//**                 for (int m = -l; m <= l; m++)
//**                 {
//**                     int lm = Utils::lm_by_l_m(l, m);
//**                     zt1 += jvlo(lm, igkloc, icol - num_gkvec_col()) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc);
//**                 }
//**             }
//**             zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
//**             h(igkloc, icol) += zt1;
//**         }
//**     }
//**     t3.stop();
//**    
//**     // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
//**     if (num_ranks == 1)
//**     {
//**         for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**         {
//**             for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**             {
//**                 h(irow, igkloc) = conj(h(igkloc, irow));
//**                 o(irow, igkloc) = conj(o(igkloc, irow));
//**             }
//**         }
//**         return;
//**     }
//** 
//**     //** // second part: compute <lo|H|G+k> and <lo|G+k>
//** 
//**     //** int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
//**     //** // compute V|lo>
//**     //** t1.start();
//**     //** for (int irow = 0; irow < num_lo_row(); irow++)
//**     //** {
//**     //**     int ia = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].ia;
//**     //**     int lm = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].lm;
//**     //**     int idxrf = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].idxrf;
//**     //**     
//**     //**     for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
//**     //**     {
//**     //**         svlo(lm1, irow) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
//**     //**                                                 parameters_.atom(ia)->radial_grid());
//** 
//**     //**         for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
//**     //**         {
//**     //**             int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
//**     //**             complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;
//** 
//**     //**             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**     //**             {
//**     //**                 (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
//**     //**                                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
//**     //**             }
//**     //**         }
//** 
//**     //**         svlo(lm1, irow)->interpolate();
//**     //**     }
//**     //** }
//**     //** t1.stop();
//**    
//**     //** t2.start();
//**     //** for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     //** {
//**     //**     int ia = apwlo_basis_descriptors_row_[irow].ia;
//**     //**     int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**     //**     int l = apwlo_basis_descriptors_row_[irow].l;
//**     //**     int lm = apwlo_basis_descriptors_row_[irow].lm;
//**     //**     int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
//** 
//**     //**     // compue overlap <lo|G+k>
//**     //**     Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**     //**     for (int ir = 0; ir < slo.num_points(); ir++)
//**     //**         slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**     //**     slo.interpolate();
//**     //**     
//**     //**     #pragma omp parallel for default(shared)
//**     //**     for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     //**     {
//**     //**         o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**     //**                           conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
//**     //**                           Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
//**     //**                           gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**     //**         // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**     //**         h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//**     //**     }
//**     //** }
//**     //** t2.stop();
//** 
//**     //** t3.start();
//**     //** #pragma omp parallel for default(shared)
//**     //** for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
//**     //** {
//**     //**     for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
//**     //**     {
//**     //**         int ia = apwlo_basis_descriptors_row_[irow].ia;
//**     //**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//** 
//**     //**         //int l = apwlo_basis_descriptors_row_[irow].l;
//**     //**         //int lm = apwlo_basis_descriptors_row_[irow].lm;
//**     //**         //int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
//** 
//**     //**         //*// compue overlap <lo|G+k>
//**     //**         //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
//**     //**         //*for (int ir = 0; ir < s.num_points(); ir++)
//**     //**         //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
//**     //**         //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//**     //**         //*s.interpolate();
//**     //**         //*    
//**     //**         //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
//**     //**         //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
//**     //**         //*                  gkvec_phase_factors_(offset_col + igkloc, ia);
//** 
//**     //**         //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
//**     //**         //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
//** 
//**     //**         // add <lo|V|G+k>
//**     //**         complex16 zt1(0, 0);
//**     //**         for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
//**     //**         {
//**     //**             for (int m1 = -l1; m1 <= l1; m1++)
//**     //**             {
//**     //**                 int lm1 = Utils::lm_by_l_m(l1, m1);
//** 
//**     //**                 zt1 += conj(Spline<complex16>::integrate(svlo(lm1, irow - num_gkvec_row()), 
//**     //**                                                          (*sbessel_[offset_col + igkloc])(l1, iat))) * 
//**     //**                        zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
//**     //**             }
//**     //**         }
//**     //**         zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
//**     //**         h(irow, igkloc) += zt1;
//**     //**     }
//**     //** }
//**     //** t3.stop();
//**     //** 
//**     //** for (int irow = 0; irow < num_lo_row(); irow++)
//**     //** {
//**     //**     for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
//**     //** }
//** }
//** 
//**




//=====================================================================================================================
// CPU code, (L)APW+lo basis
//=====================================================================================================================
template<> void Band::set_fv_h_o<cpu, apwlo>(K_point* kp, Periodic_function<double>* effective_potential,
                                             mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_fv_h_o");
   
    // index of column apw coefficients in apw array
    int apw_offset_col = kp->apw_offset_col();
    
    mdarray<complex16, 2> alm(kp->num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm(kp->num_gkvec_row(), parameters_.max_mt_aw_basis_size());

    h.zero();
    o.zero();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        Atom_type* type = atom->type();
       
        // generate conjugated coefficients
        kp->generate_matching_coefficients<true>(kp->num_gkvec_loc(), ia, alm);
        
        // apply muffin-tin part to <bra|
        apply_hmt_to_apw(kp->num_gkvec_row(), ia, alm, halm);
        
        // generate <apw|apw> block; |ket> is conjugated, so it is "unconjugated" back
        blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone, 
                        &alm(0, 0), alm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &o(0, 0), o.ld()); 
            
        // generate <apw|H|apw> block; |ket> is conjugated, so it is "unconjugated" back
        blas<cpu>::gemm(0, 2, kp->num_gkvec_row(), kp->num_gkvec_col(), type->mt_aw_basis_size(), zone, 
                        &halm(0, 0), halm.ld(), &alm(apw_offset_col, 0), alm.ld(), zone, &h(0, 0), h.ld());
       
        // setup apw-lo blocks
        set_fv_h_o_apw_lo(kp, type, atom, ia, alm, h, o);
    } //ia

    set_fv_h_o_it(kp, effective_potential, h, o);

    set_fv_h_o_lo_lo(kp, h, o);

    alm.deallocate();
    halm.deallocate();
}

void Band::set_fv_h_o_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<complex16, 2>& alm, 
                             mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_fv_h_o_apw_lo");
    
    int apw_offset_col = kp->apw_offset_col();
    
    // apw-lo block
    for (int i = 0; i < kp->num_atom_lo_cols(ia); i++)
    {
        int icol = kp->lo_col(ia, i);

        int l = kp->apwlo_basis_descriptors_col(icol).l;
        int lm = kp->apwlo_basis_descriptors_col(icol).lm;
        int idxrf = kp->apwlo_basis_descriptors_col(icol).idxrf;
        int order = kp->apwlo_basis_descriptors_col(icol).order;
        
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
            }
        }

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
        int idxrf = kp->apwlo_basis_descriptors_row(irow).idxrf;
        int order = kp->apwlo_basis_descriptors_row(irow).order;

        memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(complex16));
    
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm, lm1, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++)
                    ztmp[igkloc] += zsum * conj(alm(apw_offset_col + igkloc, j1));
            }
        }

        for (int igkloc = 0; igkloc < kp->num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 

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

void Band::set_fv_h_o_it(K_point* kp, Periodic_function<double>* effective_potential, 
                         mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::Band::set_fv_h_o_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = parameters_.index_g12(kp->apwlo_basis_descriptors_row(igk_row).ig,
                                             kp->apwlo_basis_descriptors_col(igk_col).ig);
            
            // pw kinetic energy
            double t1 = 0.5 * Utils::scalar_product(kp->apwlo_basis_descriptors_row(igk_row).gkvec_cart, 
                                                    kp->apwlo_basis_descriptors_col(igk_col).gkvec_cart);
                               
            h(igk_row, igk_col) += (effective_potential->f_pw(ig12) + t1 * parameters_.step_function_pw(ig12));
            o(igk_row, igk_col) += parameters_.step_function_pw(ig12);
        }
    }
}

void Band::set_fv_h_o_lo_lo(K_point* kp, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_lo_lo");

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
                Atom* atom = parameters_.atom(ia);
                int lm1 = kp->apwlo_basis_descriptors_row(irow).lm; 
                int idxrf1 = kp->apwlo_basis_descriptors_row(irow).idxrf; 

                h(irow, icol) += parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2, 
                                                                          atom->h_radial_integrals(idxrf1, idxrf2));

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

void Band::solve_fv_evp_1stage(K_point* kp, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o, 
                               std::vector<double>& fv_eigen_values, mdarray<complex16, 2>& fv_eigen_vectors)
{
    Timer t("sirius::Band::solve_fv_evp");
    generalized_evp* solver = NULL;

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

    solver->solve(kp->apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o.get_ptr(), o.ld(), 
                  &fv_eigen_values[0], fv_eigen_vectors.get_ptr(), fv_eigen_vectors.ld());

    delete solver;
}

//== void K_point::solve_fv_evp_2stage(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//== {
//==     if (parameters_.eigen_value_solver() != lapack) error_local(__FILE__, __LINE__, "implemented for LAPACK only");
//==     
//==     standard_evp_lapack s;
//== 
//==     std::vector<double> o_eval(apwlo_basis_size());
//==     
//==     mdarray<complex16, 2> o_tmp(apwlo_basis_size(), apwlo_basis_size());
//==     memcpy(o_tmp.get_ptr(), o.get_ptr(), o.size() * sizeof(complex16));
//==     mdarray<complex16, 2> o_evec(apwlo_basis_size(), apwlo_basis_size());
//==  
//==     s.solve(apwlo_basis_size(), o_tmp.get_ptr(), o_tmp.ld(), &o_eval[0], o_evec.get_ptr(), o_evec.ld());
//== 
//==     int num_dependent_apwlo = 0;
//==     for (int i = 0; i < apwlo_basis_size(); i++) 
//==     {
//==         if (fabs(o_eval[i]) < 1e-4) 
//==         {
//==             num_dependent_apwlo++;
//==         }
//==         else
//==         {
//==             o_eval[i] = 1.0 / sqrt(o_eval[i]);
//==         }
//==     }
//== 
//==     //std::cout << "num_dependent_apwlo = " << num_dependent_apwlo << std::endl;
//== 
//==     mdarray<complex16, 2> h_tmp(apwlo_basis_size(), apwlo_basis_size());
//==     // compute h_tmp = Z^{h.c.} * H
//==     blas<cpu>::gemm(2, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), o_evec.get_ptr(), 
//==                     o_evec.ld(), h.get_ptr(), h.ld(), h_tmp.get_ptr(), h_tmp.ld());
//==     // compute \tilda H = Z^{h.c.} * H * Z = h_tmp * Z
//==     blas<cpu>::gemm(0, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), h_tmp.get_ptr(), 
//==                     h_tmp.ld(), o_evec.get_ptr(), o_evec.ld(), h.get_ptr(), h.ld());
//== 
//==     int reduced_apwlo_basis_size = apwlo_basis_size() - num_dependent_apwlo;
//==     
//==     for (int i = 0; i < reduced_apwlo_basis_size; i++)
//==     {
//==         for (int j = 0; j < reduced_apwlo_basis_size; j++)
//==         {
//==             double d = o_eval[num_dependent_apwlo + j] * o_eval[num_dependent_apwlo + i];
//==             h(num_dependent_apwlo + j, num_dependent_apwlo + i) *= d;
//==         }
//==     }
//== 
//==     std::vector<double> h_eval(reduced_apwlo_basis_size);
//==     s.solve(reduced_apwlo_basis_size, &h(num_dependent_apwlo, num_dependent_apwlo), h.ld(), &h_eval[0], 
//==             h_tmp.get_ptr(), h_tmp.ld());
//== 
//==     for (int i = 0; i < reduced_apwlo_basis_size; i++)
//==     {
//==         for (int j = 0; j < reduced_apwlo_basis_size; j++) h_tmp(j, i) *= o_eval[num_dependent_apwlo + j];
//==     }
//== 
//==     for (int i = 0; i < parameters_.num_fv_states(); i++) fv_eigen_values_[i] = h_eval[i];
//== 
//==     blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), reduced_apwlo_basis_size, 
//==                     &o_evec(0, num_dependent_apwlo), o_evec.ld(), h_tmp.get_ptr(), h_tmp.ld(), 
//==                     fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());
//== }

void Band::solve_fv(K_point* kp, Periodic_function<double>* effective_potential)
{
    Timer t("sirius::Band::solve_fv");

    if (kp->num_ranks() > 1 && (parameters_.eigen_value_solver() == lapack || parameters_.eigen_value_solver() == magma))
        error_local(__FILE__, __LINE__, "Can't use more than one MPI rank for LAPACK or MAGMA eigen-value solver");

    mdarray<complex16, 2> h(kp->apwlo_basis_size_row(), kp->apwlo_basis_size_col());
    mdarray<complex16, 2> o(kp->apwlo_basis_size_row(), kp->apwlo_basis_size_col());
    
    // Magma requires special allocation
    #ifdef _MAGMA_
    if (parameters_.eigen_value_solver() == magma)
    {
        h.pin_memory();
        o.pin_memory();
    }
    #endif
   
    // setup Hamiltonian and overlap
    switch (parameters_.processing_unit())
    {
        case cpu:
        {
            set_fv_h_o<cpu, basis_type>(kp, effective_potential, h, o);
            break;
        }
        #ifdef _GPU_
        case gpu:
        {
            set_fv_h_o<gpu, basis_type>(kp, effective_potential, h, o);
            break;
        }
        #endif
        default:
        {
            error_local(__FILE__, __LINE__, "wrong processing unit");
        }
    }
    
    // TODO: move debug code to a separate function
    if (debug_level > 0 && parameters_.eigen_value_solver() == lapack)
    {
        Utils::check_hermitian("h", h);
        Utils::check_hermitian("o", o);
    }

    //sirius_io::hdf5_write_matrix("h.h5", h);
    //sirius_io::hdf5_write_matrix("o.h5", o);
    
    //Utils::write_matrix("h.txt", true, h);
    //Utils::write_matrix("o.txt", true, o);

    //** if (verbosity_level > 1)
    //** {
    //**     double h_max = 0;
    //**     double o_max = 0;
    //**     int h_irow = 0;
    //**     int h_icol = 0;
    //**     int o_irow = 0;
    //**     int o_icol = 0;
    //**     std::vector<double> h_diag(apwlo_basis_size(), 0);
    //**     std::vector<double> o_diag(apwlo_basis_size(), 0);
    //**     for (int icol = 0; icol < apwlo_basis_size_col(); icol++)
    //**     {
    //**         int idxglob = apwlo_basis_descriptors_col_[icol].idxglob;
    //**         for (int irow = 0; irow < apwlo_basis_size_row(); irow++)
    //**         {
    //**             if (apwlo_basis_descriptors_row_[irow].idxglob == idxglob)
    //**             {
    //**                 h_diag[idxglob] = abs(h(irow, icol));
    //**                 o_diag[idxglob] = abs(o(irow, icol));
    //**             }
    //**             if (abs(h(irow, icol)) > h_max)
    //**             {
    //**                 h_max = abs(h(irow, icol));
    //**                 h_irow = irow;
    //**                 h_icol = icol;
    //**             }
    //**             if (abs(o(irow, icol)) > o_max)
    //**             {
    //**                 o_max = abs(o(irow, icol));
    //**                 o_irow = irow;
    //**                 o_icol = icol;
    //**             }
    //**         }
    //**     }

    //**     Platform::allreduce(&h_diag[0], apwlo_basis_size(),
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     Platform::allreduce(&o_diag[0], apwlo_basis_size(),
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     if (parameters_.mpi_grid().root(1 << band->dim_row() | 1 << band->dim_col()))
    //**     {
    //**         std::stringstream s;
    //**         s << "h_diag : ";
    //**         for (int i = 0; i < apwlo_basis_size(); i++) s << h_diag[i] << " ";
    //**         s << std::endl;
    //**         s << "o_diag : ";
    //**         for (int i = 0; i < apwlo_basis_size(); i++) s << o_diag[i] << " ";
    //**         warning(__FILE__, __LINE__, s, 0);
    //**     }

    //**     std::stringstream s;
    //**     s << "h_max " << h_max << " irow, icol : " << h_irow << " " << h_icol << std::endl;
    //**     s << " (row) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_row_[h_irow].igk << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].ig << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].ia << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].l << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].lm << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_row_[h_irow].order 
    //**                                                       << std::endl;
    //**     s << " (col) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_col_[h_icol].igk << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].ig << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].ia << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].l << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].lm << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_col_[h_icol].order 
    //**                                                       << std::endl;

    //**     s << "o_max " << o_max << " irow, icol : " << o_irow << " " << o_icol << std::endl;
    //**     s << " (row) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_row_[o_irow].igk << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].ig << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].ia << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].l << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].lm << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_row_[o_irow].order 
    //**                                                       << std::endl;
    //**     s << " (col) igk, ig, ia, l, lm, irdrf, order : " << apwlo_basis_descriptors_col_[o_icol].igk << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].ig << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].ia << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].l << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].lm << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].idxrf << " "  
    //**                                                       << apwlo_basis_descriptors_col_[o_icol].order 
    //**                                                       << std::endl;
    //**     warning(__FILE__, __LINE__, s, 0);
    //** }
    
    assert(kp->apwlo_basis_size() > parameters_.num_fv_states());
    
    // debug scalapack
    //** std::vector<double> fv_eigen_values_glob(parameters_.num_fv_states());
    //** if ((debug_level > 2) && 
    //**     (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa))
    //** {
    //**     mdarray<complex16, 2> h_glob(apwlo_basis_size(), apwlo_basis_size());
    //**     mdarray<complex16, 2> o_glob(apwlo_basis_size(), apwlo_basis_size());
    //**     mdarray<complex16, 2> fv_eigen_vectors_glob(apwlo_basis_size(), parameters_.num_fv_states());

    //**     h_glob.zero();
    //**     o_glob.zero();

    //**     for (int icol = 0; icol < apwlo_basis_size_col(); icol++)
    //**     {
    //**         int j = apwlo_basis_descriptors_col_[icol].idxglob;
    //**         for (int irow = 0; irow < apwlo_basis_size_row(); irow++)
    //**         {
    //**             int i = apwlo_basis_descriptors_row_[irow].idxglob;
    //**             h_glob(i, j) = h(irow, icol);
    //**             o_glob(i, j) = o(irow, icol);
    //**         }
    //**     }
    //**     
    //**     Platform::allreduce(h_glob.get_ptr(), (int)h_glob.size(), 
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));
    //**     
    //**     Platform::allreduce(o_glob.get_ptr(), (int)o_glob.size(), 
    //**                         parameters_.mpi_grid().communicator(1 << band->dim_row() | 1 << band->dim_col()));

    //**     Utils::check_hermitian("h_glob", h_glob);
    //**     Utils::check_hermitian("o_glob", o_glob);
    //**     
    //**     generalized_evp_lapack lapack_solver(-1.0);

    //**     lapack_solver.solve(apwlo_basis_size(), parameters_.num_fv_states(), h_glob.get_ptr(), h_glob.ld(), 
    //**                         o_glob.get_ptr(), o_glob.ld(), &fv_eigen_values_glob[0], fv_eigen_vectors_glob.get_ptr(),
    //**                         fv_eigen_vectors_glob.ld());
    //** }
    
    if (fix_apwlo_linear_dependence)
    {
        //solve_fv_evp_2stage(kp, h, o);
    }
    else
    {
        solve_fv_evp_1stage(kp, h, o, kp->fv_eigen_values(), kp->fv_eigen_vectors());
    }
        
    #ifdef _MAGMA_
    if (parameters_.eigen_value_solver() == magma)
    {
        h.unpin_memory();
        o.unpin_memory();
    }
    #endif
   
    h.deallocate();
    o.deallocate();

    //** if ((debug_level > 2) && (parameters_.eigen_value_solver() == scalapack))
    //** {
    //**     double d = 0.0;
    //**     for (int i = 0; i < parameters_.num_fv_states(); i++) 
    //**         d += fabs(fv_eigen_values_[i] - fv_eigen_values_glob[i]);
    //**     std::stringstream s;
    //**     s << "Totoal eigen-value difference : " << d;
    //**     warning(__FILE__, __LINE__, s, 0);
    //** }
}


//=====================================================================================================================
// GPU code, (L)APW+lo basis
//=====================================================================================================================
//** #ifdef _GPU_
//** template<> void K_point::set_fv_h_o<gpu, apwlo>(Periodic_function<double>* effective_potential, int num_ranks,
//**                                                mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
//** 
//** {
//**     Timer t("sirius::K_point::set_fv_h_o");
//**     
//**     int apw_offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
//**     
//**     mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
//**     mdarray<complex16, 2> halm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
//**     
//**     alm.pin_memory();
//**     alm.allocate_on_device();
//**     halm.pin_memory();
//**     halm.allocate_on_device();
//**     h.allocate_on_device();
//**     h.zero_on_device();
//**     o.allocate_on_device();
//**     o.zero_on_device();
//** 
//**     complex16 zone(1, 0);
//**     
//**     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**     {
//**         Atom* atom = parameters_.atom(ia);
//**         Atom_type* type = atom->type();
//**         
//**         generate_matching_coefficients(num_gkvec_loc(), ia, alm);
//**         
//**         apply_hmt_to_apw(num_gkvec_row(), ia, alm, halm);
//**         
//**         alm.copy_to_device();
//**         halm.copy_to_device();
//**         
//**         blas<gpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &zone, 
//**                         alm.get_ptr_device(), alm.ld(), &(alm.get_ptr_device()[apw_offset_col]), alm.ld(), 
//**                         &zone, o.get_ptr_device(), o.ld()); 
//**         
//**         blas<gpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &zone, 
//**                         halm.get_ptr_device(), halm.ld(), &(alm.get_ptr_device()[apw_offset_col]), alm.ld(),
//**                         &zone, h.get_ptr_device(), h.ld());
//**             
//**         set_fv_h_o_apw_lo(type, atom, ia, apw_offset_col, alm, h, o);
//**     } //ia
//** 
//**     cublas_get_matrix(num_gkvec_row(), num_gkvec_col(), sizeof(complex16), h.get_ptr_device(), h.ld(), 
//**                       h.get_ptr(), h.ld());
//**     
//**     cublas_get_matrix(num_gkvec_row(), num_gkvec_col(), sizeof(complex16), o.get_ptr_device(), o.ld(), 
//**                       o.get_ptr(), o.ld());
//** 
//**     set_fv_h_o_it(effective_potential, h, o);
//** 
//**     set_fv_h_o_lo_lo(h, o);
//** 
//**     h.deallocate_on_device();
//**     o.deallocate_on_device();
//**     alm.deallocate_on_device();
//**     alm.unpin_memory();
//**     alm.deallocate();
//**     halm.deallocate_on_device();
//**     halm.unpin_memory();
//**     halm.deallocate();
//** }
//** 
