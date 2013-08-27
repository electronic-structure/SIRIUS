/** \file band.hpp
    
    \brief Implementation of methods for Band class.
*/

Band::Band(Global& parameters__) : parameters_(parameters__), blacs_context_(-1)
{
    if (!parameters_.initialized()) error_local(__FILE__, __LINE__, "Parameters are not initialized.");

    num_ranks_row_ = parameters_.mpi_grid().dimension_size(_dim_row_);
    num_ranks_col_ = parameters_.mpi_grid().dimension_size(_dim_col_);

    num_ranks_ = num_ranks_row_ * num_ranks_col_;

    rank_row_ = parameters_.mpi_grid().coordinate(_dim_row_);
    rank_col_ = parameters_.mpi_grid().coordinate(_dim_col_);

    init();
    
    #if defined(_SCALAPACK_) || defined(_ELPA_)
    if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
    {
        int rc = (1 << _dim_row_) | (1 << _dim_col_);
        MPI_Comm comm = parameters_.mpi_grid().communicator(rc);
        blacs_context_ = linalg<scalapack>::create_blacs_context(comm);

        mdarray<int, 2> map_ranks(num_ranks_row_, num_ranks_col_);
        for (int i1 = 0; i1 < num_ranks_col_; i1++)
        {
            for (int i0 = 0; i0 < num_ranks_row_; i0++)
            {
                map_ranks(i0, i1) = parameters_.mpi_grid().cart_rank(comm, Utils::intvec(i0, i1));
            }
        }
        linalg<scalapack>::gridmap(&blacs_context_, map_ranks.get_ptr(), map_ranks.ld(), 
                                   num_ranks_row_, num_ranks_col_);

        // check the grid
        int nrow, ncol, irow, icol;
        linalg<scalapack>::gridinfo(blacs_context_, &nrow, &ncol, &irow, &icol);

        if ((rank_row_ != irow) || (rank_col_ != icol) || (num_ranks_row_ != nrow) || (num_ranks_col_ != ncol)) 
        {
            std::stringstream s;
            s << "wrong grid" << std::endl
              << "rank_row : " << rank_row_ << " irow : " << irow << std::endl
              << "rank_col : " << rank_col_ << " icol : " << icol << std::endl
              << "num_ranks_row : " << num_ranks_row_ << " nrow : " << nrow << std::endl
              << "num_ranks_col : " << num_ranks_col_ << " ncol : " << ncol;

            error_local(__FILE__, __LINE__, s);
        }
    }
    #endif
}

Band::~Band()
{
    #if defined(_SCALAPACK_) || defined(_ELPA_)
    if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa) 
        linalg<scalapack>::free_blacs_context(blacs_context_);
    #endif
}

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

    splindex<block> sub_spl_fv_states_col(spl_fv_states_col().local_size(), num_ranks_row(), rank_row());
    
    mdarray<complex16, 3> zm(parameters_.max_mt_basis_size(), parameters_.max_mt_basis_size(), 
                             parameters_.num_mag_dims());
            
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
        blas<cpu>::hemm(0, 0, mt_basis_size, sub_spl_fv_states_col.local_size(), zone, &zm(0, 0, 0), zm.ld(), 
                        &fv_states(offset, sub_spl_fv_states_col.global_offset()), fv_states.ld(), zzero, 
                        &hpsi(offset, sub_spl_fv_states_col.global_offset(), 0), hpsi.ld());
        
        // compute bwf = (B_x - iB_y)|wf_j>
        if (hpsi.size(2) >= 3)
        {
            // reuse first (z) component of zm matrix to store (Bx - iBy)
            for (int j2 = 0; j2 < mt_basis_size; j2++)
            {
                for (int j1 = 0; j1 <= j2; j1++) zm(j1, j2, 0) = zm(j1, j2, 1) - zi * zm(j1, j2, 2);
                
                for (int j1 = j2 + 1; j1 < mt_basis_size; j1++) zm(j1, j2, 0) = conj(zm(j2, j1, 1)) - zi * conj(zm(j2, j1, 2));
            }
              
            blas<cpu>::gemm(0, 0, mt_basis_size, sub_spl_fv_states_col.local_size(), mt_basis_size, &zm(0, 0, 0), 
                            zm.ld(), &fv_states(offset, sub_spl_fv_states_col.global_offset()), fv_states.ld(), 
                            &hpsi(offset, sub_spl_fv_states_col.global_offset(), 2), hpsi.ld());
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
              
            blas<cpu>::gemm(0, 0, mt_basis_size, sub_spl_fv_states_col.local_size(), mt_basis_size, &zm(0, 0, 0), 
                            zm.ld(), &fv_states(offset, sub_spl_fv_states_col.global_offset()), fv_states.ld(), 
                            &hpsi(offset, sub_spl_fv_states_col.global_offset(), 3), hpsi.ld());
        }
    }
    
    Timer *t1 = new Timer("sirius::Band::apply_magnetic_field:it");

    mdarray<complex16, 3> hpsi_pw(num_gkvec, spl_fv_states_col_.local_size(), hpsi.size(2));
    hpsi_pw.zero();

    int num_fft_threads = Platform::num_fft_threads();
    #pragma omp parallel default(shared) num_threads(num_fft_threads)
    {        
        int thread_id = omp_get_thread_num();
        
        std::vector<complex16> psi_it(parameters_.fft().size());
        std::vector<complex16> hpsi_it(parameters_.fft().size());
        
        #pragma omp for
        for (int iloc = 0; iloc < sub_spl_fv_states_col.local_size(); iloc++)
        {
            int i = sub_spl_fv_states_col[iloc];

            parameters_.fft().input(num_gkvec, fft_index, &fv_states(parameters_.mt_basis_size(), i), 
                                    thread_id);
            parameters_.fft().transform(1, thread_id);
            parameters_.fft().output(&psi_it[0], thread_id);
                                        
            for (int ir = 0; ir < parameters_.fft().size(); ir++)
            {
                hpsi_it[ir] = psi_it[ir] * effective_magnetic_field[0]->f_it<global>(ir) * 
                              parameters_.step_function(ir);
            }
            
            parameters_.fft().input(&hpsi_it[0], thread_id);
            parameters_.fft().transform(-1, thread_id);
            parameters_.fft().output(num_gkvec, fft_index, &hpsi_pw(0, i, 0), thread_id); 

            if (hpsi.size(2) >= 3)
            {
                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                {
                    hpsi_it[ir] = psi_it[ir] * (effective_magnetic_field[1]->f_it<global>(ir) - 
                                                zi * effective_magnetic_field[2]->f_it<global>(ir)) * 
                                               parameters_.step_function(ir);
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
                    hpsi_it[ir] = psi_it[ir] * (effective_magnetic_field[1]->f_it<global>(ir) + 
                                                zi * effective_magnetic_field[2]->f_it<global>(ir)) * 
                                               parameters_.step_function(ir);
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
            for (int iloc = 0; iloc < sub_spl_fv_states_col.local_size(); iloc++)
            {
                int i = sub_spl_fv_states_col[iloc];
                memcpy(&hpsi(parameters_.mt_basis_size(), i, n), &hpsi_pw(0, i, n), num_gkvec * sizeof(complex16));
            }
        }
        else
        {
            // copy Bz|\psi> to -Bz|\psi>
            for (int iloc = 0; iloc < sub_spl_fv_states_col.local_size(); iloc++)
            {
                int i = sub_spl_fv_states_col[iloc];
                for (int j = 0; j < mtgk_size; j++) hpsi(j, i, 1) = -hpsi(j, i, 0);
            }
        }
    }

    for (int n = 0; n < hpsi.size(2); n++)
    {
        Platform::allgather(&hpsi(0, 0, n), hpsi.size(0) * sub_spl_fv_states_col.global_offset(),
                            hpsi.size(0) * sub_spl_fv_states_col.local_size(), 
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

                        for (int ist = 0; ist < spl_fv_states_col_.local_size(); ist++)
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
                        
                        for (int ist = 0; ist < spl_fv_states_col_.local_size(); ist++)
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

void Band::init()
{
    // distribue first-variational states along columns
    spl_fv_states_col_.split(parameters_.num_fv_states(), num_ranks_col_, rank_col_, 
                             parameters_.cyclic_block_size());
   
    // distribue first-variational states along rows
    spl_fv_states_row_.split(parameters_.num_fv_states(), num_ranks_row_, rank_row_,
                             parameters_.cyclic_block_size());

    // distribue spinor wave-functions along columns
    spl_spinor_wf_col_.split(parameters_.num_bands(), num_ranks_col_, rank_col_, 
                             parameters_.cyclic_block_size());
    
    // additionally split along rows 
    sub_spl_spinor_wf_.split(spl_spinor_wf_col_.local_size(), num_ranks_row_, rank_row_);
    
    // check if the distribution of fv states is consistent with the distribtion of spinor wave functions
    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
    {
        for (int i = 0; i < spl_fv_states_col_.local_size(); i++)
        {
            if (spl_spinor_wf_col_[i + ispn * spl_fv_states_col_.local_size()] != 
                (spl_fv_states_col_[i] + ispn * parameters_.num_fv_states()))
            {
                error_local(__FILE__, __LINE__, "Wrong distribution of wave-functions");
            }
        }
    }

    if ((verbosity_level > 0) && (Platform::mpi_rank() == 0))
    {
        printf("\n");
        printf("table of column distribution of first-variational states\n");
        printf("(columns of the table correspond to MPI ranks)\n");
        for (int i0 = 0; i0 < spl_fv_states_col_.local_size(0); i0++)
        {
            for (int i1 = 0; i1 < num_ranks_col_; i1++)
                printf("%6i", spl_fv_states_col_.global_index(i0, i1));
            printf("\n");
        }
        
        /*printf("\n");
        printf("table of row distribution of first-variational states\n");
        printf("columns of the table correspond to MPI ranks\n");
        for (int i0 = 0; i0 < fv_states_distribution_row_.size(0); i0++)
        {
            for (int i1 = 0; i1 < fv_states_distribution_row_.size(1); i1++)
                printf("%6i", fv_states_distribution_row_(i0, i1));
            printf("\n");
        }*/

        printf("\n");
        printf("First-variational states index -> (local index, rank) for column distribution\n");
        for (int i = 0; i < parameters_.num_fv_states(); i++)
            printf("%6i -> (%6i %6i)\n", i, spl_fv_states_col_.location(0, i), 
                                            spl_fv_states_col_.location(1, i));
        
        printf("\n");
        printf("table of column distribution of spinor wave functions\n");
        printf("(columns of the table correspond to MPI ranks)\n");
        for (int i0 = 0; i0 < spl_spinor_wf_col_.local_size(0); i0++)
        {
            for (int i1 = 0; i1 < num_ranks_row_; i1++)
                printf("%6i", spl_spinor_wf_col_.global_index(i0, i1));
            printf("\n");
        }
    }
}

void Band::solve_sv(Global& parameters, int mtgk_size, int num_gkvec, int* fft_index, double* evalfv, 
                    mdarray<complex16, 2>& fv_states_row, mdarray<complex16, 2>& fv_states_col, 
                    Periodic_function<double>* effective_magnetic_field[3], double* band_energies,
                    mdarray<complex16, 2>& sv_eigen_vectors)

{
    if (&parameters != &parameters_) error_local(__FILE__, __LINE__, "different set of parameters");

    Timer t("sirius::Band::solve_sv");

    if (!need_sv())
    {
        memcpy(band_energies, evalfv, parameters_.num_fv_states() * sizeof(double));
        sv_eigen_vectors.zero();
        for (int icol = 0; icol < spl_fv_states_col().local_size(); icol++)
        {
            int i = spl_fv_states_col(icol);
            for (int irow = 0; irow < spl_fv_states_row().local_size(); irow++)
            {
                if (spl_fv_states_row(irow) == i) sv_eigen_vectors(irow, icol) = complex16(1, 0);
            }
        }
        return;
    }

    // number of h|\psi> components 
    int nhpsi = parameters_.num_mag_dims() + 1;

    // product of the second-variational Hamiltonian and a wave-function
    mdarray<complex16, 3> hpsi(mtgk_size, spl_fv_states_col_.local_size(), nhpsi);
    hpsi.zero();

    // compute product of magnetic field and wave-function 
    if (parameters_.num_spins() == 2)
        apply_magnetic_field(fv_states_col, mtgk_size, num_gkvec, fft_index, effective_magnetic_field, hpsi);

    if (parameters_.uj_correction())
    {
        apply_uj_correction<uu>(fv_states_col, hpsi);
        if (parameters_.num_mag_dims() != 0) apply_uj_correction<dd>(fv_states_col, hpsi);
        if (parameters_.num_mag_dims() == 3) 
        {
            apply_uj_correction<ud>(fv_states_col, hpsi);
            if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
                apply_uj_correction<du>(fv_states_col, hpsi);
        }
    }

    if (parameters_.so_correction()) apply_so_correction(fv_states_col, hpsi);

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
            solver = new standard_evp_scalapack(parameters_.cyclic_block_size(), num_ranks_row(), 
                                                num_ranks_col(), blacs_context_);
            break;
        }
        case elpa:
        {
            solver = new standard_evp_scalapack(parameters_.cyclic_block_size(), num_ranks_row(), 
                                                num_ranks_col(), blacs_context_);
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
    
    if (parameters.num_mag_dims() == 1)
    {
        mdarray<complex16, 2> h(spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size());
        
        //perform two consecutive diagonalizations
        for (int ispn = 0; ispn < 2; ispn++)
        {
            // compute <wf_i | (h * wf_j)> for up-up or dn-dn block
            blas<cpu>::gemm(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), 
                            mtgk_size, &fv_states_row(0, 0), fv_states_row.ld(), 
                            &hpsi(0, 0, ispn), hpsi.ld(), &h(0, 0), h.ld());

            for (int icol = 0; icol < spl_fv_states_col_.local_size(); icol++)
            {
                int i = spl_fv_states_col_[icol];
                for (int irow = 0; irow < spl_fv_states_row_.local_size(); irow++)
                {
                    if (spl_fv_states_row_[irow] == i) h(irow, icol) += evalfv[i];
                }
            }
        
            t1.start();
            int num_fv_states_col = spl_fv_states_col_.local_size();
            solver->solve(parameters_.num_fv_states(), h.get_ptr(), h.ld(),
                          &band_energies[ispn * parameters_.num_fv_states()],
                          &sv_eigen_vectors(0, ispn * num_fv_states_col),
                          sv_eigen_vectors.ld());
            t1.stop();
        }
    }

    if (parameters.num_mag_dims() == 3)
    {
        mdarray<complex16, 2> h(2 * spl_fv_states_row_.local_size(), 2 * spl_fv_states_col_.local_size());
        h.zero();

        // compute <wf_i | (h * wf_j)> for up-up block
        blas<cpu>::gemm(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), mtgk_size, 
                        &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 0), hpsi.ld(), &h(0, 0), h.ld());

        // compute <wf_i | (h * wf_j)> for up-dn block
        blas<cpu>::gemm(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), mtgk_size, 
                        &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 2), hpsi.ld(), 
                        &h(0, spl_fv_states_col_.local_size()), h.ld());
       
        // compute <wf_i | (h * wf_j)> for dn-dn block
        blas<cpu>::gemm(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), mtgk_size,
                        &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 1), hpsi.ld(), 
                        &h(spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size()), h.ld());

        if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
        {
            // compute <wf_i | (h * wf_j)> for dn-up block
            blas<cpu>::gemm(2, 0, spl_fv_states_row_.local_size(), spl_fv_states_col_.local_size(), mtgk_size, 
                            &fv_states_row(0, 0), fv_states_row.ld(), &hpsi(0, 0, 3), hpsi.ld(), 
                            &h(spl_fv_states_row_.local_size(), 0), h.ld());
        }
      
        for (int ispn = 0; ispn < 2; ispn++)
        {
            for (int icol = 0; icol < spl_fv_states_col_.local_size(); icol++)
            {
                int i = spl_fv_states_col_[icol] + ispn * parameters_.num_fv_states();
                for (int irow = 0; irow < spl_fv_states_row_.local_size(); irow++)
                {
                    int j = spl_fv_states_row_[irow] + ispn * parameters_.num_fv_states();
                    if (j == i) 
                    {
                        h(irow + ispn * spl_fv_states_row_.local_size(), 
                          icol + ispn * spl_fv_states_col_.local_size()) += evalfv[spl_fv_states_col_[icol]];
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
}

