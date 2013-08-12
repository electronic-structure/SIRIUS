template<> void K_point::generate_matching_coefficients_l<1>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                             mdarray<double, 2>& A, mdarray<complex16, 2>& alm)
{
    if ((fabs(A(0, 0)) < 1.0 / sqrt(parameters_.omega())) && (verbosity_level > 0))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0, 0); 
        
        warning(__FILE__, __LINE__, s);
    }
    
    A(0, 0) = 1.0 / A(0, 0);

    complex16 zt;
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt = gkvec_phase_factors_(igkloc, ia) * alm_b_(l, iat, igkloc, 0) * A(0, 0);

        int idxb = type->indexb_by_l_m_order(l, -l, 0);
        for (int m = -l; m <= l; m++)
        {
            // =========================================================================
            // it is more convenient to store conjugated coefficients because then the 
            // overlap matrix is set with single matrix-matrix multiplication without 
            // further conjugation
            // =========================================================================
            alm(igkloc, idxb++) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zt);
        }
    }
}

template<> void K_point::generate_matching_coefficients_l<2>(int ia, int iat, Atom_type* type, int l, int num_gkvec_loc, 
                                                             mdarray<double, 2>& A, mdarray<complex16, 2>& alm)
{
    double det = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0);
    
    if ((fabs(det) < 1.0 / sqrt(parameters_.omega())) && (verbosity_level > 0))
    {   
        std::stringstream s;
        s << "Ill defined plane wave matching problem for atom " << ia << ", l = " << l << std::endl
          << "  radial function value at the MT boundary : " << A(0 ,0); 
        
        warning(__FILE__, __LINE__, s);
    }
    std::swap(A(0, 0), A(1, 1));
    A(0, 0) /= det;
    A(1, 1) /= det;
    A(0, 1) = -A(0, 1) / det;
    A(1, 0) = -A(1, 0) / det;
    
    complex16 zt[2];
    complex16 zb[2];
    for (int igkloc = 0; igkloc < num_gkvec_loc; igkloc++)
    {
        zt[0] = gkvec_phase_factors_(igkloc, ia) * alm_b_(l, iat, igkloc, 0);
        zt[1] = gkvec_phase_factors_(igkloc, ia) * alm_b_(l, iat, igkloc, 1);

        zb[0] = A(0, 0) * zt[0] + A(0, 1) * zt[1];
        zb[1] = A(1, 0) * zt[0] + A(1, 1) * zt[1];

        for (int m = -l; m <= l; m++)
        {
            int idxb0 = type->indexb_by_l_m_order(l, m, 0);
            int idxb1 = type->indexb_by_l_m_order(l, m, 1);
                        
            // ===========================================================================
            // it is more convenient to store conjugated coefficients because then the 
            // overlap matrix is set with single matrix-matrix multiplication without 
            // further conjugation
            // ===========================================================================
            alm(igkloc, idxb0) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zb[0]);
            alm(igkloc, idxb1) = gkvec_ylm_(Utils::lm_by_l_m(l, m), igkloc) * conj(zb[1]);
        }
    }
}

template<> void K_point::set_fv_h_o<cpu, apwlo>(Periodic_function<double>* effective_potential, int num_ranks,
                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o");
    
    int apw_offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    
    mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm( num_gkvec_row(), parameters_.max_mt_aw_basis_size());

    h.zero();
    o.zero();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        Atom_type* type = atom->type();
        
        generate_matching_coefficients(num_gkvec_loc(), ia, alm);
        
        apply_hmt_to_apw(num_gkvec_row(), ia, alm, halm);
        
        blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), zone, &alm(0, 0), alm.ld(), 
                        &alm(apw_offset_col, 0), alm.ld(), zone, &o(0, 0), o.ld()); 
            
        // apw-apw block
        blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), zone, &halm(0, 0), halm.ld(), 
                        &alm(apw_offset_col, 0), alm.ld(), zone, &h(0, 0), h.ld());
        
        set_fv_h_o_apw_lo(type, atom, ia, apw_offset_col, alm, h, o);
    } //ia

    set_fv_h_o_it(effective_potential, h, o);

    set_fv_h_o_lo_lo(h, o);

    alm.deallocate();
    halm.deallocate();
}

template<> void K_point::ibs_force<cpu, apwlo>(Band* band, mdarray<double, 2>& ffac, mdarray<double, 2>& force)
{
    Timer t("sirius::K_point::ibs_force");

    int apw_offset_col = (band->num_ranks() > 1) ? num_gkvec_row() : 0;

    mdarray<double, 2> forcek(3, parameters_.num_atoms());
    forcek.zero();

    mdarray<complex16, 2> ha(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> oa(apwlo_basis_size_row(), apwlo_basis_size_col());
    
    mdarray<complex16, 2> vha(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> voa(apwlo_basis_size_row(), apwlo_basis_size_col());
    
    mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
    
    mdarray<complex16, 2> zf(band->spl_fv_states_row().local_size(), 
                             band->spl_fv_states_col().local_size());
    
    mdarray<complex16, 2> dm(band->spl_fv_states_row().local_size(), 
                             band->spl_fv_states_col().local_size());
    dm.zero();

    // compute the density matrix
    if (!band->need_sv())
    {
        for (int i = 0; i < band->spl_fv_states_col().local_size(); i++)
        {
            int ist = band->spl_fv_states_col(i);
            for (int j = 0; j < band->spl_fv_states_row().local_size(); j++)
            {
                if (band->spl_fv_states_row(j) == ist) dm(j, i) = band_occupancy(ist);
            }
        }
    }
    else
    {
        mdarray<complex16, 2> evq(sv_eigen_vectors_.size(0), band->spl_spinor_wf_col().local_size());
        for (int i = 0; i < band->spl_spinor_wf_col().local_size(); i++)
        {
            int n = band->spl_spinor_wf_col(i);
            for (int j = 0; j < sv_eigen_vectors_.size(0); j++) evq(j, i) = sv_eigen_vectors_(j, i) * band_occupancy(n);
        }
        
        // Important! Obtained with the following zgemm, density matrix is conjugated. 
        if (band->num_ranks() == 1)
        {
            // TODO: this can be combined
            if (parameters_.num_mag_dims() != 3)
            {
                blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
                                &sv_eigen_vectors_(0, 0), sv_eigen_vectors_.ld(), &evq(0, 0), evq.ld(), &dm(0, 0), dm.ld());
            }
            else
            {
                for (int ispn = 0; ispn < 2; ispn++)
                {
                    blas<cpu>::gemm(0, 2, parameters_.num_fv_states(), parameters_.num_fv_states(), parameters_.num_bands(),
                                    complex16(1, 0), &sv_eigen_vectors_(ispn * parameters_.num_fv_states(), 0), 
                                    sv_eigen_vectors_.ld(), &evq(ispn * parameters_.num_fv_states(), 0), evq.ld(), 
                                    complex16(1, 0), &dm(0, 0), dm.ld());
                }


            }
        }

        //// TODO: this is a zgemm or pzgemm
        //for (int n = 0; n < parameters_.num_bands(); n++)
        //{
        //    for (int i = 0; i < band->spl_fv_states_row().global_size(); i++)
        //    {
        //        int ist = i % parameters_.num_fv_states();
        //        int ispn = i / parameters_.num_fv_states();
        //        for (int j = 0; j < band->spl_fv_states_row().global_size(); j++)
        //        {
        //            int jst = j % parameters_.num_fv_states();
        //            int jspn = j / parameters_.num_fv_states();

        //            if (ispn == jspn)
        //            {
        //                dm(ist, jst) += band_occupancy(n) * conj(sv_eigen_vectors_(i, n)) * sv_eigen_vectors_(j, n); 
        //            }
        //        }
        //    }
        //}
    }

    mdarray<complex16, 2> zm1(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        ha.zero();
        oa.zero();
        
        Atom* atom = parameters_.atom(ia);
        Atom_type* type = atom->type();

        int iat = parameters_.atom_type_index_by_id(type->id());
        
        generate_matching_coefficients(num_gkvec_loc(), ia, alm);
        
        apply_hmt_to_apw(num_gkvec_row(), ia, alm, halm);
        
        // apw-apw block of the overlap matrix
        blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &alm(0, 0), alm.ld(), 
                        &alm(apw_offset_col, 0), alm.ld(), &oa(0, 0), oa.ld()); 
            
        // apw-apw block of the Hamiltonian matrix
        blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &halm(0, 0), halm.ld(), 
                        &alm(apw_offset_col, 0), alm.ld(), &ha(0, 0), ha.ld());
        
        // apw-lo and lo-apw blocks of Hamiltonian and overlap
        set_fv_h_o_apw_lo(type, atom, ia, apw_offset_col, alm, ha, oa);

        for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
        {
            double v2c[3];
            parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc2].igk), v2c);

            for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
            {
                int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                                 apwlo_basis_descriptors_col_[igkloc2].ig);
                int igs = parameters_.gvec_shell<global>(ig12);
                double v1c[3];
                parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_row_[igkloc1].igk), v1c);

                complex16 zt = conj(parameters_.gvec_phase_factor<global>(ig12, ia)) * ffac(igs, iat);

                ha(igkloc1, igkloc2) -= 0.5 * Utils::scalar_product(v1c, v2c) * zt;
                oa(igkloc1, igkloc2) -= zt;
            }
        }

        for (int x = 0; x < 3; x++)
        {
            for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
            {
                for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
                {
                    int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                                     apwlo_basis_descriptors_col_[igkloc2].ig);
                    double vg[3];
                    parameters_.gvec_cart(ig12, vg);
                    vha(igkloc1, igkloc2) = complex16(0.0, vg[x]) * ha(igkloc1, igkloc2);
                    voa(igkloc1, igkloc2) = complex16(0.0, vg[x]) * oa(igkloc1, igkloc2);
                }
            }

            for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
            {
                for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++)
                {
                    double vgk[3];
                    parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_row_[igkloc1].igk), vgk);

                    vha(igkloc1, icol) = complex16(0.0, vgk[x]) * ha(igkloc1, icol);
                    voa(igkloc1, icol) = complex16(0.0, vgk[x]) * oa(igkloc1, icol);
                }
            }
                    
            for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
            {
                for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++)
                {
                    double vgk[3];
                    parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc2].igk), vgk);

                    vha(irow, igkloc2) = complex16(0.0, -vgk[x]) * ha(irow, igkloc2);
                    voa(irow, igkloc2) = complex16(0.0, -vgk[x]) * oa(irow, igkloc2);
                }
            }

            if (band->num_ranks() == 1)
            {
                // zm1 = H * V
                blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), apwlo_basis_size(), 
                                &vha(0, 0), vha.ld(), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), &zm1(0, 0), zm1.ld());
                
                // F = V^{+} * zm1 = V^{+} * H * V
                blas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), apwlo_basis_size(),
                                &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), &zm1(0, 0), zm1.ld(), 
                                &zf(0, 0), zf.ld());

                // zm1 = O * V
                blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), apwlo_basis_size(), 
                                &voa(0, 0), voa.ld(), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(),
                                &zm1(0, 0), zm1.ld());

                // multiply by energy
                for (int i = 0; i < parameters_.num_fv_states(); i++)
                {
                    for (int j = 0; j < apwlo_basis_size(); j++) zm1(j, i) = zm1(j, i) * fv_eigen_values_[i];
                }

                // F = F - V^{+} * zm1 = F - V^{+} * O * (E*V)
                blas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), apwlo_basis_size(),
                                complex16(-1, 0), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), &zm1(0, 0), zm1.ld(), 
                                complex16(1, 0), &zf(0, 0), zf.ld());

                for (int i = 0; i < parameters_.num_fv_states(); i++)
                {
                    for (int j = 0; j < parameters_.num_fv_states(); j++) 
                        forcek(x, ia) += weight() * real(conj(dm(j, i)) * zf(j, i));
                }
            }
            else
            {
                #ifdef _SCALAPACK_
                // zm1 = H * V
                pblas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), apwlo_basis_size(), 
                                 complex16(1, 0), &vha(0, 0), vha.ld(), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), 
                                 complex16(0, 0), &zm1(0, 0), zm1.ld(), parameters_.cyclic_block_size(), band->blacs_context());

                // F = V^{+} * zm1 = V^{+} * H * V
                pblas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), apwlo_basis_size(),
                                 complex16(1, 0), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), &zm1(0, 0), zm1.ld(), 
                                 complex16(0, 0), &zf(0, 0), zf.ld(), parameters_.cyclic_block_size(), band->blacs_context());

                // zm1 = O * V
                pblas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), apwlo_basis_size(), 
                                 complex16(1, 0), &voa(0, 0), voa.ld(), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(),
                                 complex16(0, 0), &zm1(0, 0), zm1.ld(), parameters_.cyclic_block_size(), band->blacs_context());

                // multiply by energy
                for (int i = 0; i < band->spl_fv_states_col().local_size(); i++)
                {
                    int ist = band->spl_fv_states_col(i);
                    for (int j = 0; j < apwlo_basis_size_row(); j++) zm1(j, i) = zm1(j, i) * fv_eigen_values_[ist];
                }

                // F = F - V^{+} * zm1 = F - V^{+} * O * (E*V)
                pblas<cpu>::gemm(2, 0, parameters_.num_fv_states(), parameters_.num_fv_states(), apwlo_basis_size(),
                                 complex16(-1, 0), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), &zm1(0, 0), zm1.ld(), 
                                 complex16(1, 0), &zf(0, 0), zf.ld(), parameters_.cyclic_block_size(), band->blacs_context());

                // TODO: this can be combined with the previous code
                for (int i = 0; i < band->spl_fv_states_col().local_size(); i++)
                {
                    for (int j = 0; j < band->spl_fv_states_row().local_size(); j++) 
                        forcek(x, ia) += weight() * real(conj(dm(j, i)) * zf(j, i));
                }

                #else
                error(__FILE__, __LINE__, "not compiled with ScaLAPACK");
                #endif
            }
        }
    } //ia
    
    if (band->num_ranks() > 1)
    {
        Platform::allreduce(&forcek(0, 0), (int)forcek.size(), 
                            parameters_.mpi_grid().communicator((1 << _dim_row_) | (1 << _dim_col_)));
    }
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int x = 0; x < 3; x++) force(x, ia) += forcek(x, ia);
    }
}

template<> void K_point::set_fv_h_o_pw_lo<cpu>(Periodic_function<double>* effective_potential, int num_ranks, 
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_pw_lo");
    
    int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    
    mdarray<Spline<complex16>*, 2> svlo(parameters_.lmmax_pw(), std::max(num_lo_col(), num_lo_row()));

    // first part: compute <G+k|H|lo> and <G+k|lo>

    Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
    Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
    Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);

    // compute V|lo>
    t1.start();
    for (int icol = 0; icol < num_lo_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
        int lm = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].lm;
        int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;
        
        for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
        {
            svlo(lm1, icol) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
                                                    parameters_.atom(ia)->radial_grid());

            for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
            {
                int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
                complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;

                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                {
                    (*svlo(lm1, icol))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
                                               parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
                }
            }

            svlo(lm1, icol)->interpolate();
        }
    }
    t1.stop();
    
    t2.start();
    // compute overlap and kinetic energy
    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

        int l = apwlo_basis_descriptors_col_[icol].l;
        int lm = apwlo_basis_descriptors_col_[icol].lm;
        int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;

        Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
        for (int ir = 0; ir < slo.num_points(); ir++)
            slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
        slo.interpolate();
        
        #pragma omp parallel for default(shared)
        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
        {
            o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
                              Spline<double>::integrate(&slo, (*sbessel_[igkloc])(l, iat)) * 
                              conj(gkvec_phase_factors_(igkloc, ia));

            // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
            h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
        }
    }
    t2.stop();

    t3.start();
    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
    {
        for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
        {
            int ia = apwlo_basis_descriptors_col_[icol].ia;
            int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

            //int l = apwlo_basis_descriptors_col_[icol].l;
            //int lm = apwlo_basis_descriptors_col_[icol].lm;
            //int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;

            //*// compue overlap <G+k|lo>
            //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
            //*for (int ir = 0; ir < s.num_points(); ir++)
            //*{
            //*    s[ir] = (*sbessel_[igkloc])(ir, l, iat) * 
            //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
            //*}
            //*s.interpolate();
            //*    
            //*o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
            //*                  s.integrate(2) * conj(gkvec_phase_factors_(igkloc, ia));

            //*// kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
            //*h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);

            // add <G+k|V|lo>
            complex16 zt1(0, 0);
            for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
            {
                for (int m1 = -l1; m1 <= l1; m1++)
                {
                    int lm1 = Utils::lm_by_l_m(l1, m1);

                    zt1 += Spline<complex16>::integrate(svlo(lm1, icol - num_gkvec_col()), 
                                                        (*sbessel_[igkloc])(l1, iat)) * 
                           conj(zil_[l1]) * gkvec_ylm_(lm1, igkloc);
                }
            }
            zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
            h(igkloc, icol) += zt1;
        }
    }
    t3.stop();
   
    // deallocate V|lo>
    for (int icol = 0; icol < num_lo_col(); icol++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, icol);
    }

    // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
    if (num_ranks == 1)
    {
        for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
        {
            for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
            {
                h(irow, igkloc) = conj(h(igkloc, irow));
                o(irow, igkloc) = conj(o(igkloc, irow));
            }
        }
        return;
    }

    // second part: compute <lo|H|G+k> and <lo|G+k>

    // compute V|lo>
    t1.start();
    for (int irow = 0; irow < num_lo_row(); irow++)
    {
        int ia = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].ia;
        int lm = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].lm;
        int idxrf = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].idxrf;
        
        for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
        {
            svlo(lm1, irow) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
                                                    parameters_.atom(ia)->radial_grid());

            for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
            {
                int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
                complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;

                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                {
                    (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_mt<global>(lm3, ir, ia) * 
                                               parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
                }
            }

            svlo(lm1, irow)->interpolate();
        }
    }
    t1.stop();
   
    t2.start();
    for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
    {
        int ia = apwlo_basis_descriptors_row_[irow].ia;
        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

        int l = apwlo_basis_descriptors_row_[irow].l;
        int lm = apwlo_basis_descriptors_row_[irow].lm;
        int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;

        // compue overlap <lo|G+k>
        Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
        for (int ir = 0; ir < slo.num_points(); ir++)
            slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
        slo.interpolate();
        
        #pragma omp parallel for default(shared)
        for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
        {
            o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
                              conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
                              Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
                              gkvec_phase_factors_(offset_col + igkloc, ia);

            // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
            h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
        }
    }
    t2.stop();

    t3.start();
    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
    {
        for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
        {
            int ia = apwlo_basis_descriptors_row_[irow].ia;
            int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

            //int l = apwlo_basis_descriptors_row_[irow].l;
            //int lm = apwlo_basis_descriptors_row_[irow].lm;
            //int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;

            //*// compue overlap <lo|G+k>
            //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
            //*for (int ir = 0; ir < s.num_points(); ir++)
            //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
            //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
            //*s.interpolate();
            //*    
            //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
            //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
            //*                  gkvec_phase_factors_(offset_col + igkloc, ia);

            //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
            //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);

            // add <lo|V|G+k>
            complex16 zt1(0, 0);
            for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
            {
                for (int m1 = -l1; m1 <= l1; m1++)
                {
                    int lm1 = Utils::lm_by_l_m(l1, m1);

                    zt1 += conj(Spline<complex16>::integrate(svlo(lm1, irow - num_gkvec_row()), 
                                                             (*sbessel_[offset_col + igkloc])(l1, iat))) * 
                           zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
                }
            }
            zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
            h(irow, igkloc) += zt1;
        }
    }
    t3.stop();
    
    for (int irow = 0; irow < num_lo_row(); irow++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
    }
}

template<> void K_point::set_fv_h_o<cpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o");
    
    h.zero();
    o.zero();

    #pragma omp parallel for default(shared)
    for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
    {
        for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
        {
            if (apwlo_basis_descriptors_row_[igkloc1].idxglob == apwlo_basis_descriptors_col_[igkloc2].idxglob) 
            {
                h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
                o(igkloc1, igkloc2) = complex16(1, 0);
            }
                               
            int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                             apwlo_basis_descriptors_col_[igkloc2].ig);
            h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
        }
    }
    
    set_fv_h_o_pw_lo<cpu>(effective_potential, num_ranks, h, o);

    set_fv_h_o_lo_lo(h, o);
}


#ifdef _GPU_
template<> void K_point::set_fv_h_o<gpu, apwlo>(Periodic_function<double>* effective_potential, int num_ranks,
                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)

{
    Timer t("sirius::K_point::set_fv_h_o");
    
    int apw_offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    
    mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
    
    alm.pin_memory();
    alm.allocate_on_device();
    halm.pin_memory();
    halm.allocate_on_device();
    h.allocate_on_device();
    h.zero_on_device();
    o.allocate_on_device();
    o.zero_on_device();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        Atom_type* type = atom->type();
        
        generate_matching_coefficients(num_gkvec_loc(), ia, alm);
        
        apply_hmt_to_apw(num_gkvec_row(), ia, alm, halm);
        
        alm.copy_to_device();
        halm.copy_to_device();
        
        blas<gpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &zone, 
                        alm.get_ptr_device(), alm.ld(), &(alm.get_ptr_device()[apw_offset_col]), alm.ld(), 
                        &zone, o.get_ptr_device(), o.ld()); 
        
        blas<gpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &zone, 
                        halm.get_ptr_device(), halm.ld(), &(alm.get_ptr_device()[apw_offset_col]), alm.ld(),
                        &zone, h.get_ptr_device(), h.ld());
            
        set_fv_h_o_apw_lo(type, atom, ia, apw_offset_col, alm, h, o);
    } //ia

    cublas_get_matrix(num_gkvec_row(), num_gkvec_col(), sizeof(complex16), h.get_ptr_device(), h.ld(), 
                      h.get_ptr(), h.ld());
    
    cublas_get_matrix(num_gkvec_row(), num_gkvec_col(), sizeof(complex16), o.get_ptr_device(), o.ld(), 
                      o.get_ptr(), o.ld());

    set_fv_h_o_it(effective_potential, h, o);

    set_fv_h_o_lo_lo(h, o);

    h.deallocate_on_device();
    o.deallocate_on_device();
    alm.deallocate_on_device();
    alm.unpin_memory();
    alm.deallocate();
    halm.deallocate_on_device();
    halm.unpin_memory();
    halm.deallocate();
}

template<> void K_point::set_fv_h_o_pw_lo<gpu>(Periodic_function<double>* effective_potential, int num_ranks, 
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_pw_lo");
    
    // ===========================================
    // first part: compute <G+k|H|lo> and <G+k|lo>
    // ===========================================

    Timer t1("sirius::K_point::set_fv_h_o_pw_lo:vlo", false);
    Timer t2("sirius::K_point::set_fv_h_o_pw_lo:ohk", false);
    Timer t3("sirius::K_point::set_fv_h_o_pw_lo:hvlo", false);

    mdarray<int, 1> kargs(4);
    kargs(0) = parameters_.num_atom_types();
    kargs(1) = parameters_.max_num_mt_points();
    kargs(2) = parameters_.lmax_pw();
    kargs(3) = parameters_.lmmax_pw();
    kargs.allocate_on_device();
    kargs.copy_to_device();

    // =========================
    // compute V|lo> for columns
    // =========================
    t1.start();
    mdarray<complex16, 3> vlo_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmmax_pw(), num_lo_col());
    #pragma omp parallel for default(shared)
    for (int icol = 0; icol < num_lo_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
        int lm = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].lm;
        int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;
        
        for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
        {
            Spline<complex16> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());

            for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
            {
                int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
                complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;

                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                {
                    s[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
                              parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
                }
            }
            s.interpolate();
            s.get_coefs(&vlo_coefs(0, lm1, icol), parameters_.max_num_mt_points());
        }
    }
    vlo_coefs.pin_memory();
    vlo_coefs.allocate_on_device();
    vlo_coefs.async_copy_to_device(-1);
    t1.stop();
    
    // ===========================================
    // pack Bessel function splines into one array
    // ===========================================
    mdarray<double, 4> sbessel_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmax_pw() + 1, 
                                     parameters_.num_atom_types(), num_gkvec_row());
    for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
    {
        for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
        {
            for (int l = 0; l <= parameters_.lmax_pw(); l++)
            {
                (*sbessel_[igkloc])(l, iat)->get_coefs(&sbessel_coefs(0, l, iat, igkloc), 
                                                       parameters_.max_num_mt_points());
            }
        }
    }
    sbessel_coefs.pin_memory();
    sbessel_coefs.allocate_on_device();
    sbessel_coefs.async_copy_to_device(-1);

    // ==============================
    // pack lo splines into one array
    // ==============================
    mdarray<double, 2> lo_coefs(parameters_.max_num_mt_points() * 4, num_lo_col());
    mdarray<int, 1> l_by_ilo(num_lo_col());
    mdarray<int, 1> iat_by_ilo(num_lo_col());
    for (int icol = 0; icol < num_lo_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].ia;
        l_by_ilo(icol) = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].l;
        iat_by_ilo(icol) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

        int idxrf = apwlo_basis_descriptors_col_[num_gkvec_col() + icol].idxrf;

        Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
            s[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
        s.interpolate();
        s.get_coefs(&lo_coefs(0, icol), parameters_.max_num_mt_points());
    }
    lo_coefs.pin_memory();
    lo_coefs.allocate_on_device();
    lo_coefs.async_copy_to_device(-1);
    l_by_ilo.allocate_on_device();
    l_by_ilo.async_copy_to_device(-1);
    iat_by_ilo.allocate_on_device();
    iat_by_ilo.async_copy_to_device(-1);

    // ============
    // radial grids
    // ============
    mdarray<double, 2> r_dr(parameters_.max_num_mt_points() * 2, parameters_.num_atom_types());
    mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
    for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
    {
        nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
        parameters_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), parameters_.max_num_mt_points());
    }
    r_dr.allocate_on_device();
    r_dr.async_copy_to_device(-1);
    nmtp_by_iat.allocate_on_device();
    nmtp_by_iat.async_copy_to_device(-1);

    mdarray<double, 2> jlo(num_gkvec_row(), num_lo_col());
    jlo.allocate_on_device();

    t2.start();
    sbessel_lo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), l_by_ilo.get_ptr_device(), 
                                 iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), r_dr.get_ptr_device(), 
                                 sbessel_coefs.get_ptr_device(), lo_coefs.get_ptr_device(), jlo.get_ptr_device());
    jlo.copy_to_host();
    // compute overlap and kinetic energy
    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        int l = apwlo_basis_descriptors_col_[icol].l;
        int lm = apwlo_basis_descriptors_col_[icol].lm;

        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
        {
            o(igkloc, icol) = (fourpi / sqrt(parameters_.omega())) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc) * 
                              jlo(igkloc, icol - num_gkvec_col()) * conj(gkvec_phase_factors_(igkloc, ia));

            // kinetic part <G+k| -1/2 \nabla^2 |lo> = 1/2 |G+k|^2 <G+k|lo>
            h(igkloc, icol) = 0.5 * pow(gkvec_len_[igkloc], 2) * o(igkloc, icol);
        }
    }
    t2.stop();

    l_by_lm_.allocate_on_device();
    l_by_lm_.copy_to_device();

    mdarray<complex16, 3> jvlo(parameters_.lmmax_pw(), num_gkvec_row(), num_lo_col());
    jvlo.allocate_on_device();
    
    t3.start();
    sbessel_vlo_inner_product_gpu(kargs.get_ptr_device(), num_gkvec_row(), num_lo_col(), parameters_.lmmax_pw(), 
                                  l_by_lm_.get_ptr_device(), iat_by_ilo.get_ptr_device(), nmtp_by_iat.get_ptr_device(), 
                                  r_dr.get_ptr_device(), sbessel_coefs.get_ptr_device(), vlo_coefs.get_ptr_device(), 
                                  jvlo.get_ptr_device());
    jvlo.copy_to_host();

    l_by_lm_.deallocate_on_device();

    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
    {
        for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
        {
            int ia = apwlo_basis_descriptors_col_[icol].ia;

            // add <G+k|V|lo>
            complex16 zt1(0, 0);
            for (int l = 0; l <= parameters_.lmax_pw(); l++)
            {
                for (int m = -l; m <= l; m++)
                {
                    int lm = Utils::lm_by_l_m(l, m);
                    zt1 += jvlo(lm, igkloc, icol - num_gkvec_col()) * conj(zil_[l]) * gkvec_ylm_(lm, igkloc);
                }
            }
            zt1 *= ((fourpi / sqrt(parameters_.omega())) * conj(gkvec_phase_factors_(igkloc, ia)));
            h(igkloc, icol) += zt1;
        }
    }
    t3.stop();
   
    // restore the <lo|H|G+k> and <lo|G+k> bocks and exit
    if (num_ranks == 1)
    {
        for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
        {
            for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
            {
                h(irow, igkloc) = conj(h(igkloc, irow));
                o(irow, igkloc) = conj(o(igkloc, irow));
            }
        }
        return;
    }

    //** // second part: compute <lo|H|G+k> and <lo|G+k>

    //** int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    //** // compute V|lo>
    //** t1.start();
    //** for (int irow = 0; irow < num_lo_row(); irow++)
    //** {
    //**     int ia = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].ia;
    //**     int lm = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].lm;
    //**     int idxrf = apwlo_basis_descriptors_row_[num_gkvec_row() + irow].idxrf;
    //**     
    //**     for (int lm1 = 0; lm1 < parameters_.lmmax_pw(); lm1++)
    //**     {
    //**         svlo(lm1, irow) = new Spline<complex16>(parameters_.atom(ia)->num_mt_points(), 
    //**                                                 parameters_.atom(ia)->radial_grid());

    //**         for (int k = 0; k < parameters_.gaunt().complex_gaunt_packed_L3_size(lm1, lm); k++)
    //**         {
    //**             int lm3 = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).lm3;
    //**             complex16 cg = parameters_.gaunt().complex_gaunt_packed_L3(lm1, lm, k).cg;

    //**             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
    //**             {
    //**                 (*svlo(lm1, irow))[ir] += (cg * effective_potential->f_rlm(lm3, ir, ia) * 
    //**                                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf));
    //**             }
    //**         }

    //**         svlo(lm1, irow)->interpolate();
    //**     }
    //** }
    //** t1.stop();
   
    //** t2.start();
    //** for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
    //** {
    //**     int ia = apwlo_basis_descriptors_row_[irow].ia;
    //**     int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

    //**     int l = apwlo_basis_descriptors_row_[irow].l;
    //**     int lm = apwlo_basis_descriptors_row_[irow].lm;
    //**     int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;

    //**     // compue overlap <lo|G+k>
    //**     Spline<double> slo(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
    //**     for (int ir = 0; ir < slo.num_points(); ir++)
    //**         slo[ir] = parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
    //**     slo.interpolate();
    //**     
    //**     #pragma omp parallel for default(shared)
    //**     for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
    //**     {
    //**         o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
    //**                           conj(gkvec_ylm_(lm, offset_col + igkloc)) * 
    //**                           Spline<double>::integrate(&slo, (*sbessel_[offset_col + igkloc])(l, iat)) * 
    //**                           gkvec_phase_factors_(offset_col + igkloc, ia);

    //**         // kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
    //**         h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);
    //**     }
    //** }
    //** t2.stop();

    //** t3.start();
    //** #pragma omp parallel for default(shared)
    //** for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
    //** {
    //**     for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
    //**     {
    //**         int ia = apwlo_basis_descriptors_row_[irow].ia;
    //**         int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

    //**         //int l = apwlo_basis_descriptors_row_[irow].l;
    //**         //int lm = apwlo_basis_descriptors_row_[irow].lm;
    //**         //int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;

    //**         //*// compue overlap <lo|G+k>
    //**         //*Spline<double> s(parameters_.atom(ia)->num_mt_points(), parameters_.atom(ia)->radial_grid());
    //**         //*for (int ir = 0; ir < s.num_points(); ir++)
    //**         //*    s[ir] = (*sbessel_[offset_col + igkloc])(ir, l, iat) * 
    //**         //*            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
    //**         //*s.interpolate();
    //**         //*    
    //**         //*o(irow, igkloc) = (fourpi / sqrt(parameters_.omega())) * zil_[l] * 
    //**         //*                  conj(gkvec_ylm_(lm, offset_col + igkloc)) * s.integrate(2) * 
    //**         //*                  gkvec_phase_factors_(offset_col + igkloc, ia);

    //**         //*// kinetic part <li| -1/2 \nabla^2 |G+k> = 1/2 |G+k|^2 <lo|G+k>
    //**         //*h(irow, igkloc) = 0.5 * pow(gkvec_len_[offset_col + igkloc], 2) * o(irow, igkloc);

    //**         // add <lo|V|G+k>
    //**         complex16 zt1(0, 0);
    //**         for (int l1 = 0; l1 <= parameters_.lmax_pw(); l1++)
    //**         {
    //**             for (int m1 = -l1; m1 <= l1; m1++)
    //**             {
    //**                 int lm1 = Utils::lm_by_l_m(l1, m1);

    //**                 zt1 += conj(Spline<complex16>::integrate(svlo(lm1, irow - num_gkvec_row()), 
    //**                                                          (*sbessel_[offset_col + igkloc])(l1, iat))) * 
    //**                        zil_[l1] * conj(gkvec_ylm_(lm1, offset_col + igkloc));
    //**             }
    //**         }
    //**         zt1 *= ((fourpi / sqrt(parameters_.omega())) * gkvec_phase_factors_(offset_col + igkloc, ia));
    //**         h(irow, igkloc) += zt1;
    //**     }
    //** }
    //** t3.stop();
    //** 
    //** for (int irow = 0; irow < num_lo_row(); irow++)
    //** {
    //**     for (int lm = 0; lm < parameters_.lmmax_pw(); lm++) delete svlo(lm, irow);
    //** }
}

template<> void K_point::set_fv_h_o<gpu, pwlo>(Periodic_function<double>* effective_potential, int num_ranks,
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o");
    
    h.zero();
    o.zero();

    #pragma omp parallel for default(shared)
    for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
    {
        for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
        {
            if (apwlo_basis_descriptors_row_[igkloc1].idxglob == apwlo_basis_descriptors_col_[igkloc2].idxglob) 
            {
                h(igkloc1, igkloc2) = 0.5 * pow(gkvec_len_[igkloc1], 2);
                o(igkloc1, igkloc2) = complex16(1, 0);
            }
                               
            int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                             apwlo_basis_descriptors_col_[igkloc2].ig);
            h(igkloc1, igkloc2) += effective_potential->f_pw(ig12);
        }
    }
    
    set_fv_h_o_pw_lo<gpu>(effective_potential, num_ranks, h, o);

    set_fv_h_o_lo_lo(h, o);
}
#endif

void K_point::initialize(Band* band)
{
    Timer t("sirius::K_point::initialize");
    
    zil_.resize(parameters_.lmax() + 1);
    for (int l = 0; l <= parameters_.lmax(); l++) zil_[l] = pow(complex16(0, 1), l);
    
    l_by_lm_.set_dimensions(Utils::lmmax_by_lmax(parameters_.lmax()));
    l_by_lm_.allocate();
    for (int l = 0, lm = 0; l <= parameters_.lmax(); l++)
    {
        for (int m = -l; m <= l; m++, lm++) l_by_lm_(lm) = l;
    }

    generate_gkvec();

    build_apwlo_basis_descriptors();

    distribute_block_cyclic(band);
    
    init_gkvec();
    
    icol_by_atom_.resize(parameters_.num_atoms());
    irow_by_atom_.resize(parameters_.num_atoms());

    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        icol_by_atom_[ia].push_back(icol);
    }
    
    for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
    {
        int ia = apwlo_basis_descriptors_row_[irow].ia;
        irow_by_atom_[ia].push_back(irow);
    }
    
    if (basis_type == pwlo)
    {
        sbessel_.resize(num_gkvec_loc()); 
        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            sbessel_[igkloc] = new sbessel_pw<double>(parameters_, parameters_.lmax_pw());
            sbessel_[igkloc]->interpolate(gkvec_len_[igkloc]);
        }
    }
    
    fv_eigen_values_.resize(parameters_.num_fv_states());

    fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());
    fv_eigen_vectors_.allocate();
    
    fv_states_col_.set_dimensions(mtgk_size(), band->spl_fv_states_col().local_size());
    fv_states_col_.allocate();
    
    if (band->num_ranks() == 1)
    {
        fv_states_row_.set_dimensions(mtgk_size(), parameters_.num_fv_states());
        fv_states_row_.set_ptr(fv_states_col_.get_ptr());
    }
    else
    {
        fv_states_row_.set_dimensions(mtgk_size(), band->spl_fv_states_row().local_size());
        fv_states_row_.allocate();
    }
    
    // in case of collinear magnetism store pure up and pure dn components, otherwise store both up and dn components
    int ns = (parameters_.num_mag_dims() == 3) ? 2 : 1;
    sv_eigen_vectors_.set_dimensions(ns * band->spl_fv_states_row().local_size(), band->spl_spinor_wf_col().local_size());
    sv_eigen_vectors_.allocate();

    band_energies_.resize(parameters_.num_bands());

    spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), band->spl_spinor_wf_col().local_size());

    if (band->need_sv())
    {
        spinor_wave_functions_.allocate();
    }
    else
    {
        spinor_wave_functions_.set_ptr(fv_states_col_.get_ptr());
    }
}

// TODO: add a switch to return conjuagted or normal coefficients
void K_point::generate_matching_coefficients(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm)
{
    Timer t("sirius::K_point::generate_matching_coefficients");

    Atom* atom = parameters_.atom(ia);
    Atom_type* type = atom->type();

    assert(type->max_aw_order() <= 2);

    int iat = parameters_.atom_type_index_by_id(type->id());

    #pragma omp parallel default(shared)
    {
        mdarray<double, 2> A(2, 2);

        #pragma omp for
        for (int l = 0; l <= parameters_.lmax_apw(); l++)
        {
            int num_aw = (int)type->aw_descriptor(l).size();

            for (int order = 0; order < num_aw; order++)
            {
                for (int dm = 0; dm < num_aw; dm++) A(dm, order) = atom->symmetry_class()->aw_surface_dm(l, order, dm);
            }

            switch (num_aw)
            {
                case 1:
                {
                    generate_matching_coefficients_l<1>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                case 2:
                {
                    generate_matching_coefficients_l<2>(ia, iat, type, l, num_gkvec_loc, A, alm);
                    break;
                }
                default:
                {
                    error(__FILE__, __LINE__, "wrong order of augmented wave", fatal_err);
                }
            }
        } //l
    }
    
    // check alm coefficients
    if (debug_level > 1) check_alm(num_gkvec_loc, ia, alm);
}

void K_point::check_alm(int num_gkvec_loc, int ia, mdarray<complex16, 2>& alm)
{
    static SHT* sht = NULL;
    if (!sht)
    {
        sht = new SHT(parameters_.lmax_apw());
    }

    Atom* atom = parameters_.atom(ia);
    Atom_type* type = parameters_.atom(ia)->type();

    mdarray<complex16, 2> z1(sht->num_points(), type->mt_aw_basis_size());
    for (int i = 0; i < type->mt_aw_basis_size(); i++)
    {
        int lm = type->indexb(i).lm;
        int idxrf = type->indexb(i).idxrf;
        double rf = atom->symmetry_class()->radial_function(atom->num_mt_points() - 1, idxrf);
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            z1(itp, i) = sht->ylm_backward(lm, itp) * rf;
        }
    }

    mdarray<complex16, 2> z2(sht->num_points(), num_gkvec_loc);
    blas<cpu>::gemm(0, 2, sht->num_points(), num_gkvec_loc, type->mt_aw_basis_size(), z1.get_ptr(), z1.ld(),
                    alm.get_ptr(), alm.ld(), z2.get_ptr(), z2.ld());

    double vc[3];
    parameters_.get_coordinates<cartesian, direct>(parameters_.atom(ia)->position(), vc);
    
    double tdiff = 0;
    for (int igloc = 0; igloc < num_gkvec_loc; igloc++)
    {
        double gkc[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igkglob(igloc)), gkc);
        for (int itp = 0; itp < sht->num_points(); itp++)
        {
            complex16 aw_value = z2(itp, igloc);
            double r[3];
            for (int x = 0; x < 3; x++) r[x] = vc[x] + sht->coord(x, itp) * type->mt_radius();
            complex16 pw_value = exp(complex16(0, Utils::scalar_product(r, gkc))) / sqrt(parameters_.omega());
            tdiff += abs(pw_value - aw_value);
        }
    }

    printf("atom : %i  absolute alm error : %e  average alm error : %e\n", 
           ia, tdiff, tdiff / (num_gkvec_loc * sht->num_points()));
}

void K_point::apply_hmt_to_apw(int num_gkvec_row, int ia, mdarray<complex16, 2>& alm, mdarray<complex16, 2>& halm)
{
    Timer t("sirius::K_point::apply_hmt_to_apw");
    
    Atom* atom = parameters_.atom(ia);
    Atom_type* type = atom->type();
    
    #pragma omp parallel default(shared)
    {
        std::vector<complex16> zv(num_gkvec_row);
        
        #pragma omp for
        for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
        {
            memset(&zv[0], 0, num_gkvec_row * sizeof(complex16));

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
                    for (int ig = 0; ig < num_gkvec_row; ig++) zv[ig] += zsum * alm(ig, j1); 
                }
            } // j1
             
            int l2 = type->indexb(j2).l;
            int order2 = type->indexb(j2).order;
            
            for (int order1 = 0; order1 < (int)type->aw_descriptor(l2).size(); order1++)
            {
                double t1 = 0.5 * pow(type->mt_radius(), 2) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order1, 0) * 
                            atom->symmetry_class()->aw_surface_dm(l2, order2, 1);
                
                for (int ig = 0; ig < num_gkvec_row; ig++) 
                    zv[ig] += t1 * alm(ig, type->indexb_by_lm_order(lm2, order1));
            }
            
            memcpy(&halm(0, j2), &zv[0], num_gkvec_row * sizeof(complex16));
        } // j2
    }
}

void K_point::set_fv_h_o_apw_lo(Atom_type* type, Atom* atom, int ia, int apw_offset_col, mdarray<complex16, 2>& alm, 
                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_apw_lo");
    
    // apw-lo block
    for (int i = 0; i < (int)icol_by_atom_[ia].size(); i++)
    {
        int icol = icol_by_atom_[ia][i];

        int l = apwlo_basis_descriptors_col_[icol].l;
        int lm = apwlo_basis_descriptors_col_[icol].lm;
        int idxrf = apwlo_basis_descriptors_col_[icol].idxrf;
        int order = apwlo_basis_descriptors_col_[icol].order;
        
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
            }
        }

        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
            {
                o(igkloc, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order) * 
                                   alm(igkloc, type->indexb_by_lm_order(lm, order1));
            }
        }
    }

    std::vector<complex16> ztmp(num_gkvec_col());
    // lo-apw block
    for (int i = 0; i < (int)irow_by_atom_[ia].size(); i++)
    {
        int irow = irow_by_atom_[ia][i];

        int l = apwlo_basis_descriptors_row_[irow].l;
        int lm = apwlo_basis_descriptors_row_[irow].lm;
        int idxrf = apwlo_basis_descriptors_row_[irow].idxrf;
        int order = apwlo_basis_descriptors_row_[irow].order;

        memset(&ztmp[0], 0, num_gkvec_col() * sizeof(complex16));
    
        for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
        {
            int lm1 = type->indexb(j1).lm;
            int idxrf1 = type->indexb(j1).idxrf;
                    
            complex16 zsum = parameters_.gaunt().sum_L3_complex_gaunt(lm, lm1, atom->h_radial_integrals(idxrf, idxrf1));

            if (abs(zsum) > 1e-14)
            {
                for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
                    ztmp[igkloc] += zsum * conj(alm(apw_offset_col + igkloc, j1));
            }
        }

        for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++) h(irow, igkloc) += ztmp[igkloc]; 

        for (int order1 = 0; order1 < (int)type->aw_descriptor(l).size(); order1++)
        {
            for (int igkloc = 0; igkloc < num_gkvec_col(); igkloc++)
            {
                o(irow, igkloc) += atom->symmetry_class()->o_radial_integral(l, order, order1) * 
                                   conj(alm(apw_offset_col + igkloc, type->indexb_by_lm_order(lm, order1)));
            }
        }
    }
}

void K_point::set_fv_h_o_it(Periodic_function<double>* effective_potential, 
                           mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_it");

    #pragma omp parallel for default(shared)
    for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
    {
        double v2c[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc2].igk), v2c);

        for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
        {
            int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                             apwlo_basis_descriptors_col_[igkloc2].ig);
            double v1c[3];
            parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_row_[igkloc1].igk), v1c);
            
            double t1 = 0.5 * Utils::scalar_product(v1c, v2c);
                               
            h(igkloc1, igkloc2) += (effective_potential->f_pw(ig12) + t1 * parameters_.step_function_pw(ig12));
            o(igkloc1, igkloc2) += parameters_.step_function_pw(ig12);
        }
    }
}

void K_point::set_fv_h_o_lo_lo(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::K_point::set_fv_h_o_lo_lo");

    // lo-lo block
    #pragma omp parallel for default(shared)
    for (int icol = num_gkvec_col(); icol < apwlo_basis_size_col(); icol++)
    {
        int ia = apwlo_basis_descriptors_col_[icol].ia;
        int lm2 = apwlo_basis_descriptors_col_[icol].lm; 
        int idxrf2 = apwlo_basis_descriptors_col_[icol].idxrf; 

        for (int irow = num_gkvec_row(); irow < apwlo_basis_size_row(); irow++)
        {
            if (ia == apwlo_basis_descriptors_row_[irow].ia)
            {
                Atom* atom = parameters_.atom(ia);
                int lm1 = apwlo_basis_descriptors_row_[irow].lm; 
                int idxrf1 = apwlo_basis_descriptors_row_[irow].idxrf; 

                h(irow, icol) += parameters_.gaunt().sum_L3_complex_gaunt(lm1, lm2, 
                                                                          atom->h_radial_integrals(idxrf1, idxrf2));

                if (lm1 == lm2)
                {
                    int l = apwlo_basis_descriptors_row_[irow].l;
                    int order1 = apwlo_basis_descriptors_row_[irow].order; 
                    int order2 = apwlo_basis_descriptors_col_[icol].order; 
                    o(irow, icol) += atom->symmetry_class()->o_radial_integral(l, order1, order2);
                }
            }
        }
    }
}

inline void K_point::copy_lo_blocks(const int apwlo_basis_size_row, const int num_gkvec_row, 
                                   const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                   const complex16* z, complex16* vec)
{
    for (int j = num_gkvec_row; j < apwlo_basis_size_row; j++)
    {
        int ia = apwlo_basis_descriptors_row[j].ia;
        int lm = apwlo_basis_descriptors_row[j].lm;
        int order = apwlo_basis_descriptors_row[j].order;
        vec[parameters_.atom(ia)->offset_wf() + parameters_.atom(ia)->type()->indexb_by_lm_order(lm, order)] = z[j];
    }
}

inline void K_point::copy_pw_block(const int num_gkvec, const int num_gkvec_row, 
                                  const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                  const complex16* z, complex16* vec)
{
    memset(vec, 0, num_gkvec * sizeof(complex16));

    for (int j = 0; j < num_gkvec_row; j++) vec[apwlo_basis_descriptors_row[j].igk] = z[j];
}

void K_point::solve_fv_evp_1stage(Band* band, mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer *t1 = new Timer("sirius::K_point::generate_fv_states:genevp");
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
            solver = new generalized_evp_scalapack(parameters_.cyclic_block_size(), band->num_ranks_row(), 
                                                   band->num_ranks_col(), band->blacs_context(), -1.0);
            break;
        }
        case elpa:
        {
            solver = new generalized_evp_elpa(parameters_.cyclic_block_size(), apwlo_basis_size_row(), 
                                              band->num_ranks_row(), band->rank_row(),
                                              apwlo_basis_size_col(), band->num_ranks_col(), 
                                              band->rank_col(), band->blacs_context(), 
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
            error(__FILE__, __LINE__, "eigen value solver is not defined", fatal_err);
        }
    }

    solver->solve(apwlo_basis_size(), parameters_.num_fv_states(), h.get_ptr(), h.ld(), o.get_ptr(), o.ld(), 
                  &fv_eigen_values_[0], fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());

    delete solver;
    delete t1;
}

void K_point::solve_fv_evp_2stage(mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    if (parameters_.eigen_value_solver() != lapack) error(__FILE__, __LINE__, "implemented for LAPACK only");
    
    standard_evp_lapack s;

    std::vector<double> o_eval(apwlo_basis_size());
    
    mdarray<complex16, 2> o_tmp(apwlo_basis_size(), apwlo_basis_size());
    memcpy(o_tmp.get_ptr(), o.get_ptr(), o.size() * sizeof(complex16));
    mdarray<complex16, 2> o_evec(apwlo_basis_size(), apwlo_basis_size());
 
    s.solve(apwlo_basis_size(), o_tmp.get_ptr(), o_tmp.ld(), &o_eval[0], o_evec.get_ptr(), o_evec.ld());

    int num_dependent_apwlo = 0;
    for (int i = 0; i < apwlo_basis_size(); i++) 
    {
        if (fabs(o_eval[i]) < 1e-4) 
        {
            num_dependent_apwlo++;
        }
        else
        {
            o_eval[i] = 1.0 / sqrt(o_eval[i]);
        }
    }

    //std::cout << "num_dependent_apwlo = " << num_dependent_apwlo << std::endl;

    mdarray<complex16, 2> h_tmp(apwlo_basis_size(), apwlo_basis_size());
    // compute h_tmp = Z^{h.c.} * H
    blas<cpu>::gemm(2, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), o_evec.get_ptr(), 
                    o_evec.ld(), h.get_ptr(), h.ld(), h_tmp.get_ptr(), h_tmp.ld());
    // compute \tilda H = Z^{h.c.} * H * Z = h_tmp * Z
    blas<cpu>::gemm(0, 0, apwlo_basis_size(), apwlo_basis_size(), apwlo_basis_size(), h_tmp.get_ptr(), 
                    h_tmp.ld(), o_evec.get_ptr(), o_evec.ld(), h.get_ptr(), h.ld());

    int reduced_apwlo_basis_size = apwlo_basis_size() - num_dependent_apwlo;
    
    for (int i = 0; i < reduced_apwlo_basis_size; i++)
    {
        for (int j = 0; j < reduced_apwlo_basis_size; j++)
        {
            double d = o_eval[num_dependent_apwlo + j] * o_eval[num_dependent_apwlo + i];
            h(num_dependent_apwlo + j, num_dependent_apwlo + i) *= d;
        }
    }

    std::vector<double> h_eval(reduced_apwlo_basis_size);
    s.solve(reduced_apwlo_basis_size, &h(num_dependent_apwlo, num_dependent_apwlo), h.ld(), &h_eval[0], 
            h_tmp.get_ptr(), h_tmp.ld());

    for (int i = 0; i < reduced_apwlo_basis_size; i++)
    {
        for (int j = 0; j < reduced_apwlo_basis_size; j++) h_tmp(j, i) *= o_eval[num_dependent_apwlo + j];
    }

    for (int i = 0; i < parameters_.num_fv_states(); i++) fv_eigen_values_[i] = h_eval[i];

    blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), reduced_apwlo_basis_size, 
                    &o_evec(0, num_dependent_apwlo), o_evec.ld(), h_tmp.get_ptr(), h_tmp.ld(), 
                    fv_eigen_vectors_.get_ptr(), fv_eigen_vectors_.ld());
}

void K_point::generate_fv_states(Band* band, Periodic_function<double>* effective_potential)
{
    Timer t("sirius::K_point::generate_fv_states");

    mdarray<complex16, 2> h(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> o(apwlo_basis_size_row(), apwlo_basis_size_col());
    
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
            set_fv_h_o<cpu, basis_type>(effective_potential, band->num_ranks(), h, o);
            break;
        }
        #ifdef _GPU_
        case gpu:
        {
            set_fv_h_o<gpu, basis_type>(effective_potential, band->num_ranks(), h, o);
            break;
        }
        #endif
        default:
        {
            error(__FILE__, __LINE__, "wrong processing unit");
        }
    }
    
    // TODO: move debug code to a separate function
    if ((debug_level > 0) && (parameters_.eigen_value_solver() == lapack))
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
    
    assert(apwlo_basis_size() > parameters_.num_fv_states());
    
    //== fv_eigen_values_.resize(parameters_.num_fv_states());

    //== fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());
    //== fv_eigen_vectors_.allocate();
   
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
        solve_fv_evp_2stage(h, o);
    }
    else
    {
        solve_fv_evp_1stage(band, h, o);
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
    
    // generate first-variational wave-functions
    fv_states_col_.zero();

    mdarray<complex16, 2> alm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
    
    Timer *t2 = new Timer("sirius::K_point::generate_fv_states:wf");
    if (basis_type == apwlo)
    {
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            Atom* atom = parameters_.atom(ia);
            Atom_type* type = atom->type();
            
            generate_matching_coefficients(num_gkvec_row(), ia, alm);

            blas<cpu>::gemm(2, 0, type->mt_aw_basis_size(), band->spl_fv_states_col().local_size(),
                            num_gkvec_row(), &alm(0, 0), alm.ld(), &fv_eigen_vectors_(0, 0), 
                            fv_eigen_vectors_.ld(), &fv_states_col_(atom->offset_wf(), 0), 
                            fv_states_col_.ld());
        }
    }

    for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
    {
        copy_lo_blocks(apwlo_basis_size_row(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                       &fv_eigen_vectors_(0, j), &fv_states_col_(0, j));

        copy_pw_block(num_gkvec(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                      &fv_eigen_vectors_(0, j), &fv_states_col_(parameters_.mt_basis_size(), j));
    }

    for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
    {
        Platform::allreduce(&fv_states_col_(0, j), mtgk_size(), parameters_.mpi_grid().communicator(1 << _dim_row_));
    }
    delete t2;
}

void K_point::generate_spinor_wave_functions(Band* band)
{
    Timer t("sirius::K_point::generate_spinor_wave_functions");

    spinor_wave_functions_.zero();

    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
    {
        if (parameters_.num_mag_dims() != 3)
        {
            // multiply up block for first half of the bands, dn block for second half of the bands
            blas<cpu>::gemm(0, 0, mtgk_size(), band->spl_fv_states_col().local_size(), 
                            band->spl_fv_states_row().local_size(), 
                            &fv_states_row_(0, 0), fv_states_row_.ld(), 
                            &sv_eigen_vectors_(0, ispn * band->spl_fv_states_col().local_size()), 
                            sv_eigen_vectors_.ld(), 
                            &spinor_wave_functions_(0, ispn, ispn * band->spl_fv_states_col().local_size()), 
                            spinor_wave_functions_.ld() * parameters_.num_spins());
        }
        else
        {
            // multiply up block and then dn block for all bands
            blas<cpu>::gemm(0, 0, mtgk_size(), band->spl_spinor_wf_col().local_size(), 
                            band->spl_fv_states_row().local_size(), 
                            &fv_states_row_(0, 0), fv_states_row_.ld(), 
                            &sv_eigen_vectors_(ispn * band->spl_fv_states_row().local_size(), 0), 
                            sv_eigen_vectors_.ld(), 
                            &spinor_wave_functions_(0, ispn, 0), 
                            spinor_wave_functions_.ld() * parameters_.num_spins());
        }
    }
    
    for (int i = 0; i < band->spl_spinor_wf_col().local_size(); i++)
    {
        Platform::allreduce(&spinor_wave_functions_(0, 0, i), 
                            spinor_wave_functions_.size(0) * spinor_wave_functions_.size(1), 
                            parameters_.mpi_grid().communicator(1 << _dim_row_));
    }
}

void K_point::generate_gkvec()
{
    double gk_cutoff = parameters_.aw_cutoff() / parameters_.min_mt_radius();

    if ((gk_cutoff * parameters_.max_mt_radius() > double(parameters_.lmax_apw())) && basis_type == apwlo)
    {
        std::stringstream s;
        s << "G+k cutoff (" << gk_cutoff << ") is too large for a given lmax (" 
          << parameters_.lmax_apw() << ")" << std::endl
          << "minimum value for lmax : " << int(gk_cutoff * parameters_.max_mt_radius()) + 1;
        error(__FILE__, __LINE__, s);
    }

    if (gk_cutoff * 2 > parameters_.pw_cutoff())
        error(__FILE__, __LINE__, "aw cutoff is too large for a given plane-wave cutoff");

    std::vector< std::pair<double, int> > gkmap;

    // find G-vectors for which |G+k| < cutoff
    for (int ig = 0; ig < parameters_.num_gvec(); ig++)
    {
        double vgk[3];
        for (int x = 0; x < 3; x++) vgk[x] = parameters_.gvec(ig)[x] + vk_[x];

        double v[3];
        parameters_.get_coordinates<cartesian, reciprocal>(vgk, v);
        double gklen = Utils::vector_length(v);

        if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double, int>(gklen, ig));
    }

    std::sort(gkmap.begin(), gkmap.end());

    gkvec_.set_dimensions(3, (int)gkmap.size());
    gkvec_.allocate();

    gvec_index_.resize(gkmap.size());

    for (int ig = 0; ig < (int)gkmap.size(); ig++)
    {
        gvec_index_[ig] = gkmap[ig].second;
        for (int x = 0; x < 3; x++) gkvec_(x, ig) = parameters_.gvec(gkmap[ig].second)[x] + vk_[x];
    }
    
    fft_index_.resize(num_gkvec());
    for (int ig = 0; ig < num_gkvec(); ig++) fft_index_[ig] = parameters_.fft_index(gvec_index_[ig]);
}

void K_point::init_gkvec_phase_factors()
{
    gkvec_phase_factors_.set_dimensions(num_gkvec_loc(), parameters_.num_atoms());
    gkvec_phase_factors_.allocate();

    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
    {
        int igk = igkglob(igkloc);

        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            double phase = twopi * Utils::scalar_product(gkvec(igk), parameters_.atom(ia)->position());

            gkvec_phase_factors_(igkloc, ia) = exp(complex16(0.0, phase));
        }
    }

}

void K_point::init_gkvec()
{
    int lmax = std::max(parameters_.lmax_apw(), parameters_.lmax_pw());

    gkvec_ylm_.set_dimensions(Utils::lmmax_by_lmax(lmax), num_gkvec_loc());
    gkvec_ylm_.allocate();

    #pragma omp parallel for default(shared)
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
    {
        int igk = igkglob(igkloc);
        double v[3];
        double vs[3];

        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
        SHT::spherical_coordinates(v, vs); // vs = {r, theta, phi}

        SHT::spherical_harmonics(lmax, vs[1], vs[2], &gkvec_ylm_(0, igkloc));
    }

    init_gkvec_phase_factors();
    
    gkvec_len_.resize(num_gkvec_loc());
    for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
    {
        int igk = igkglob(igkloc);
        double v[3];
        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
        gkvec_len_[igkloc] = Utils::vector_length(v);
    }
   
    if (basis_type == apwlo)
    {
        alm_b_.set_dimensions(parameters_.lmax_apw() + 1, parameters_.num_atom_types(), num_gkvec_loc(), 2);
        alm_b_.allocate();
        alm_b_.zero();

        // compute values of spherical Bessel functions and first derivative at MT boundary
        mdarray<double, 2> sbessel_mt(parameters_.lmax_apw() + 2, 2);
        sbessel_mt.zero();

        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
        {
            for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
            {
                double R = parameters_.atom_type(iat)->mt_radius();

                double gkR = gkvec_len_[igkloc] * R;

                gsl_sf_bessel_jl_array(parameters_.lmax_apw() + 1, gkR, &sbessel_mt(0, 0));
                
                // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    sbessel_mt(l, 1) = -sbessel_mt(l + 1, 0) * gkvec_len_[igkloc] + (l / R) * sbessel_mt(l, 0);
                
                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    double f = fourpi / sqrt(parameters_.omega());
                    alm_b_(l, iat, igkloc, 0) = zil_[l] * f * sbessel_mt(l, 0); 
                    alm_b_(l, iat, igkloc, 1) = zil_[l] * f * sbessel_mt(l, 1);
                }
            }
        }
    }
}

void K_point::build_apwlo_basis_descriptors()
{
    assert(apwlo_basis_descriptors_.size() == 0);

    apwlo_basis_descriptor apwlobd;

    // G+k basis functions
    for (int igk = 0; igk < num_gkvec(); igk++)
    {
        apwlobd.igk = igk;
        apwlobd.ig = gvec_index(igk);
        apwlobd.ia = -1;
        apwlobd.lm = -1;
        apwlobd.l = -1;
        apwlobd.order = -1;
        apwlobd.idxrf = -1;
        apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();
        apwlo_basis_descriptors_.push_back(apwlobd);
    }

    // local orbital basis functions
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        Atom_type* type = atom->type();
    
        int lo_index_offset = type->mt_aw_basis_size();
        
        for (int j = 0; j < type->mt_lo_basis_size(); j++) 
        {
            int l = type->indexb(lo_index_offset + j).l;
            int lm = type->indexb(lo_index_offset + j).lm;
            int order = type->indexb(lo_index_offset + j).order;
            int idxrf = type->indexb(lo_index_offset + j).idxrf;
            apwlobd.igk = -1;
            apwlobd.ig = -1;
            apwlobd.ia = ia;
            apwlobd.lm = lm;
            apwlobd.l = l;
            apwlobd.order = order;
            apwlobd.idxrf = idxrf;
            apwlobd.idxglob = (int)apwlo_basis_descriptors_.size();
            apwlo_basis_descriptors_.push_back(apwlobd);
        }
    }
    
    // ckeck if we count basis functions correctly
    if ((int)apwlo_basis_descriptors_.size() != (num_gkvec() + parameters_.mt_lo_basis_size()))
    {
        std::stringstream s;
        s << "(L)APW+lo basis descriptors array has a wrong size" << std::endl
          << "size of apwlo_basis_descriptors_ : " << apwlo_basis_descriptors_.size() << std::endl
          << "num_gkvec : " << num_gkvec() << std::endl 
          << "mt_lo_basis_size : " << parameters_.mt_lo_basis_size();
        error(__FILE__, __LINE__, s);
    }
}

/// Block-cyclic distribution of relevant arrays 
void K_point::distribute_block_cyclic(Band* band)
{
    // distribute APW+lo basis between rows
    splindex<block_cyclic> spl_row(apwlo_basis_size(), band->num_ranks_row(), band->rank_row(), 
                                   parameters_.cyclic_block_size());
    apwlo_basis_descriptors_row_.resize(spl_row.local_size());
    for (int i = 0; i < spl_row.local_size(); i++)
        apwlo_basis_descriptors_row_[i] = apwlo_basis_descriptors_[spl_row[i]];

    // distribute APW+lo basis between columns
    splindex<block_cyclic> spl_col(apwlo_basis_size(), band->num_ranks_col(), band->rank_col(), 
                                   parameters_.cyclic_block_size());
    apwlo_basis_descriptors_col_.resize(spl_col.local_size());
    for (int i = 0; i < spl_col.local_size(); i++)
        apwlo_basis_descriptors_col_[i] = apwlo_basis_descriptors_[spl_col[i]];
    
    #if defined(_SCALAPACK) || defined(_ELPA_)
    if (parameters_.eigen_value_solver() == scalapack || parameters_.eigen_value_solver() == elpa)
    {
        int nr = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_row(), 0, band->num_ranks_row());
        
        if (nr != apwlo_basis_size_row()) 
            error(__FILE__, __LINE__, "numroc returned a different local row size");

        int nc = linalg<scalapack>::numroc(apwlo_basis_size(), parameters_.cyclic_block_size(), 
                                           band->rank_col(), 0, band->num_ranks_col());
        
        if (nc != apwlo_basis_size_col()) 
            error(__FILE__, __LINE__, "numroc returned a different local column size");
    }
    #endif

    // get the number of row- and column- G+k-vectors
    num_gkvec_row_ = 0;
    for (int i = 0; i < apwlo_basis_size_row(); i++)
        if (apwlo_basis_descriptors_row_[i].igk != -1) num_gkvec_row_++;
    
    num_gkvec_col_ = 0;
    for (int i = 0; i < apwlo_basis_size_col(); i++)
        if (apwlo_basis_descriptors_col_[i].igk != -1) num_gkvec_col_++;
}

void K_point::find_eigen_states(Band* band, Periodic_function<double>* effective_potential, 
                                Periodic_function<double>* effective_magnetic_field[3])
{
    assert(band != NULL);
    
    Timer t("sirius::K_point::find_eigen_states");

    if (band->num_ranks() > 1 && 
        (parameters_.eigen_value_solver() == lapack || parameters_.eigen_value_solver() == magma))
    {
        error(__FILE__, __LINE__, "Can't use more than one MPI rank for LAPACK or MAGMA eigen-value solver");
    }

    generate_fv_states(band, effective_potential);

    if (band->num_ranks() != 1) distribute_fv_states_row(band);

    if (debug_level > 1) test_fv_states(band, 0);

    band->solve_sv(parameters_, mtgk_size(), num_gkvec(), fft_index(), &fv_eigen_values_[0], 
                   fv_states_row_, fv_states_col_, effective_magnetic_field, &band_energies_[0],
                   sv_eigen_vectors_);

    if (band->need_sv()) generate_spinor_wave_functions(band);

    /*for (int i = 0; i < 3; i++)
        test_spinor_wave_functions(i); */
}

//Periodic_function<complex16>* K_point::spinor_wave_function_component(Band* band, int lmax, int ispn, int jloc)
//{
//    Timer t("sirius::K_point::spinor_wave_function_component");
//
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//
//    Periodic_function<complex16, index_order>* func = 
//        new Periodic_function<complex16, index_order>(parameters_, lmax);
//    func->allocate(ylm_component | it_component);
//    func->zero();
//    
//    if (basis_type == pwlo)
//    {
//        if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");
//
//        double fourpi_omega = fourpi / sqrt(parameters_.omega());
//        
//        for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
//        {
//            int igk = igkglob(igkloc);
//            complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
//
//            // TODO: possilbe optimization with zgemm
//            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//            {
//                int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//                complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia);
//                
//                #pragma omp parallel for default(shared)
//                for (int lm = 0; lm < lmmax; lm++)
//                {
//                    int l = l_by_lm_(lm);
//                    complex16 z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                        func->f_ylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
//                }
//            }
//        }
//
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&func->f_ylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
//                                parameters_.mpi_grid().communicator(1 << band->dim_row()));
//        }
//    }
//
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
//        {
//            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
//            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
//            switch (index_order)
//            {
//                case angular_radial:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(lm, ir, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//                case radial_angular:
//                {
//                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                    {
//                        func->f_ylm(ir, lm, ia) += 
//                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
//                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
//                    }
//                    break;
//                }
//            }
//        }
//    }
//
//    // in principle, wave function must have an overall e^{ikr} phase factor
//    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//                            &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, jloc));
//    parameters_.fft().transform(1);
//    parameters_.fft().output(func->f_it());
//
//    for (int i = 0; i < parameters_.fft().size(); i++) func->f_it(i) /= sqrt(parameters_.omega());
//    
//    return func;
//}

void K_point::spinor_wave_function_component_mt(Band* band, int lmax, int ispn, int jloc, mt_functions<complex16>& psilm)
{
    Timer t("sirius::K_point::spinor_wave_function_component_mt");

    //int lmmax = Utils::lmmax_by_lmax(lmax);

    psilm.zero();
    
    //if (basis_type == pwlo)
    //{
    //    if (index_order != radial_angular) error(__FILE__, __LINE__, "wrong order of indices");

    //    double fourpi_omega = fourpi / sqrt(parameters_.omega());

    //    mdarray<complex16, 2> zm(parameters_.max_num_mt_points(),  num_gkvec_row());

    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    {
    //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
    //        for (int l = 0; l <= lmax; l++)
    //        {
    //            #pragma omp parallel for default(shared)
    //            for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
    //            {
    //                int igk = igkglob(igkloc);
    //                complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;
    //                complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia) * zil_[l];
    //                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
    //                    zm(ir, igkloc) = z2 * (*sbessel_[igkloc])(ir, l, iat);
    //            }
    //            blas<cpu>::gemm(0, 2, parameters_.atom(ia)->num_mt_points(), (2 * l + 1), num_gkvec_row(),
    //                            &zm(0, 0), zm.ld(), &gkvec_ylm_(Utils::lm_by_l_m(l, -l), 0), gkvec_ylm_.ld(), 
    //                            &fylm(0, Utils::lm_by_l_m(l, -l), ia), fylm.ld());
    //        }
    //    }
    //    //for (int igkloc = 0; igkloc < num_gkvec_row(); igkloc++)
    //    //{
    //    //    int igk = igkglob(igkloc);
    //    //    complex16 z1 = spinor_wave_functions_(parameters_.mt_basis_size() + igk, ispn, jloc) * fourpi_omega;

    //    //    // TODO: possilbe optimization with zgemm
    //    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    //    {
    //    //        int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
    //    //        complex16 z2 = z1 * gkvec_phase_factors_(igkloc, ia);
    //    //        
    //    //        #pragma omp parallel for default(shared)
    //    //        for (int lm = 0; lm < lmmax; lm++)
    //    //        {
    //    //            int l = l_by_lm_(lm);
    //    //            complex16 z3 = z2 * zil_[l] * conj(gkvec_ylm_(lm, igkloc)); 
    //    //            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
    //    //                fylm(ir, lm, ia) += z3 * (*sbessel_[igkloc])(ir, l, iat);
    //    //        }
    //    //    }
    //    //}

    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    {
    //        Platform::allreduce(&fylm(0, 0, ia), lmmax * parameters_.max_num_mt_points(),
    //                            parameters_.mpi_grid().communicator(1 << band->dim_row()));
    //    }
    //}

    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
        {
            int lm = parameters_.atom(ia)->type()->indexb(i).lm;
            int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
            for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
            {
                psilm(lm, ir, ia) += 
                    spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, jloc) * 
                    parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
            }
        }
    }
}

void K_point::test_fv_states(Band* band, int use_fft)
{
    std::vector<complex16> v1;
    std::vector<complex16> v2;
    
    if (use_fft == 0) 
    {
        v1.resize(num_gkvec());
        v2.resize(parameters_.fft().size());
    }
    
    if (use_fft == 1) 
    {
        v1.resize(parameters_.fft().size());
        v2.resize(parameters_.fft().size());
    }
    
    double maxerr = 0;

    for (int j1 = 0; j1 < band->spl_fv_states_col().local_size(); j1++)
    {
        if (use_fft == 0)
        {
            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                    &fv_states_col_(parameters_.mt_basis_size(), j1));
            parameters_.fft().transform(1);
            parameters_.fft().output(&v2[0]);

            for (int ir = 0; ir < parameters_.fft().size(); ir++) v2[ir] *= parameters_.step_function(ir);
            
            parameters_.fft().input(&v2[0]);
            parameters_.fft().transform(-1);
            parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[0]); 
        }
        
        if (use_fft == 1)
        {
            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                    &fv_states_col_(parameters_.mt_basis_size(), j1));
            parameters_.fft().transform(1);
            parameters_.fft().output(&v1[0]);
        }
       
        for (int j2 = 0; j2 < band->spl_fv_states_row().local_size(); j2++)
        {
            complex16 zsum(0, 0);
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                int offset_wf = parameters_.atom(ia)->offset_wf();
                Atom_type* type = parameters_.atom(ia)->type();
                Atom_symmetry_class* symmetry_class = parameters_.atom(ia)->symmetry_class();

                for (int l = 0; l <= parameters_.lmax_apw(); l++)
                {
                    int ordmax = type->indexr().num_rf(l);
                    for (int io1 = 0; io1 < ordmax; io1++)
                    {
                        for (int io2 = 0; io2 < ordmax; io2++)
                        {
                            for (int m = -l; m <= l; m++)
                            {
                                zsum += conj(fv_states_col_(offset_wf + type->indexb_by_l_m_order(l, m, io1), j1)) *
                                             fv_states_row_(offset_wf + type->indexb_by_l_m_order(l, m, io2), j2) * 
                                             symmetry_class->o_radial_integral(l, io1, io2);
                            }
                        }
                    }
                }
            }
            
            if (use_fft == 0)
            {
               for (int ig = 0; ig < num_gkvec(); ig++)
                   zsum += conj(v1[ig]) * fv_states_row_(parameters_.mt_basis_size() + ig, j2);
            }
           
            if (use_fft == 1)
            {
                parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                   &fv_states_row_(parameters_.mt_basis_size(), j2));
                parameters_.fft().transform(1);
                parameters_.fft().output(&v2[0]);

                for (int ir = 0; ir < parameters_.fft().size(); ir++)
                    zsum += conj(v1[ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
            }
            
            if (use_fft == 2) 
            {
                for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                {
                    for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                    {
                        int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
                        zsum += conj(fv_states_col_(parameters_.mt_basis_size() + ig1, j1)) * 
                                     fv_states_row_(parameters_.mt_basis_size() + ig2, j2) * 
                                parameters_.step_function_pw(ig3);
                    }
               }
            }

            if (band->spl_fv_states_col(j1) == band->spl_fv_states_row(j2)) zsum = zsum - complex16(1, 0);
           
            maxerr = std::max(maxerr, abs(zsum));
        }
    }

    Platform::allreduce<op_max>(&maxerr, 1, parameters_.mpi_grid().communicator(1 << _dim_row_ | 1 << _dim_col_));

    if (parameters_.mpi_grid().side(1 << _dim_k_)) 
    {
        printf("k-point: %f %f %f, interstitial integration : %i, maximum error : %18.10e\n", 
               vk_[0], vk_[1], vk_[2], use_fft, maxerr);
    }
}

//** void K_point::test_spinor_wave_functions(int use_fft)
//** {
//**     std::vector<complex16> v1[2];
//**     std::vector<complex16> v2;
//** 
//**     if (use_fft == 0 || use_fft == 1)
//**         v2.resize(parameters_.fft().size());
//**     
//**     if (use_fft == 0) 
//**         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             v1[ispn].resize(num_gkvec());
//**     
//**     if (use_fft == 1) 
//**         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             v1[ispn].resize(parameters_.fft().size());
//**     
//**     double maxerr = 0;
//** 
//**     for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
//**     {
//**         if (use_fft == 0)
//**         {
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                    &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
//**                 parameters_.fft().transform(1);
//**                 parameters_.fft().output(&v2[0]);
//** 
//**                 for (int ir = 0; ir < parameters_.fft().size(); ir++)
//**                     v2[ir] *= parameters_.step_function(ir);
//**                 
//**                 parameters_.fft().input(&v2[0]);
//**                 parameters_.fft().transform(-1);
//**                 parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[ispn][0]); 
//**             }
//**         }
//**         
//**         if (use_fft == 1)
//**         {
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                    &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
//**                 parameters_.fft().transform(1);
//**                 parameters_.fft().output(&v1[ispn][0]);
//**             }
//**         }
//**        
//**         for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
//**         {
//**             complex16 zsum(0.0, 0.0);
//**             for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**             {
//**                 for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**                 {
//**                     int offset_wf = parameters_.atom(ia)->offset_wf();
//**                     Atom_type* type = parameters_.atom(ia)->type();
//**                     Atom_symmetry_class* symmetry_class = parameters_.atom(ia)->symmetry_class();
//** 
//**                     for (int l = 0; l <= parameters_.lmax_apw(); l++)
//**                     {
//**                         int ordmax = type->indexr().num_rf(l);
//**                         for (int io1 = 0; io1 < ordmax; io1++)
//**                             for (int io2 = 0; io2 < ordmax; io2++)
//**                                 for (int m = -l; m <= l; m++)
//**                                     zsum += conj(spinor_wave_functions_(offset_wf + 
//**                                                                         type->indexb_by_l_m_order(l, m, io1),
//**                                                                         ispn, j1)) *
//**                                                  spinor_wave_functions_(offset_wf + 
//**                                                                         type->indexb_by_l_m_order(l, m, io2), 
//**                                                                         ispn, j2) * 
//**                                                  symmetry_class->o_radial_integral(l, io1, io2);
//**                     }
//**                 }
//**             }
//**             
//**             if (use_fft == 0)
//**             {
//**                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                {
//**                    for (int ig = 0; ig < num_gkvec(); ig++)
//**                        zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(parameters_.mt_basis_size() + ig, ispn, j2);
//**                }
//**             }
//**            
//**             if (use_fft == 1)
//**             {
//**                 for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                 {
//**                     parameters_.fft().input(num_gkvec(), &fft_index_[0], 
//**                                        &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j2));
//**                     parameters_.fft().transform(1);
//**                     parameters_.fft().output(&v2[0]);
//** 
//**                     for (int ir = 0; ir < parameters_.fft().size(); ir++)
//**                         zsum += conj(v1[ispn][ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
//**                 }
//**             }
//**             
//**             if (use_fft == 2) 
//**             {
//**                 for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
//**                 {
//**                     for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
//**                     {
//**                         int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
//**                         for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
//**                             zsum += conj(spinor_wave_functions_(parameters_.mt_basis_size() + ig1, ispn, j1)) * 
//**                                          spinor_wave_functions_(parameters_.mt_basis_size() + ig2, ispn, j2) * 
//**                                     parameters_.step_function_pw(ig3);
//**                     }
//**                }
//**            }
//** 
//**            zsum = (j1 == j2) ? zsum - complex16(1.0, 0.0) : zsum;
//**            maxerr = std::max(maxerr, abs(zsum));
//**         }
//**     }
//**     std :: cout << "maximum error = " << maxerr << std::endl;
//** }

void K_point::save_wave_functions(int id, Band* band__)
{
    if (parameters_.mpi_grid().root(1 << _dim_col_))
    {
        hdf5_tree fout("sirius.h5", false);

        fout["K_points"].create_node(id);
        fout["K_points"][id].write("coordinates", vk_, 3);
        fout["K_points"][id].write("mtgk_size", mtgk_size());
        fout["K_points"][id].create_node("spinor_wave_functions");
        fout["K_points"][id].write("band_energies", &band_energies_[0], parameters_.num_bands());
        fout["K_points"][id].write("band_occupancies", &band_occupancies_[0], parameters_.num_bands());
    }
    
    Platform::barrier(parameters_.mpi_grid().communicator(1 << _dim_col_));
    
    mdarray<complex16, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
    for (int j = 0; j < parameters_.num_bands(); j++)
    {
        int rank = band__->spl_spinor_wf_col().location(_splindex_rank_, j);
        int offs = band__->spl_spinor_wf_col().location(_splindex_offs_, j);
        if (parameters_.mpi_grid().coordinate(_dim_col_) == rank)
        {
            hdf5_tree fout("sirius.h5", false);
            wfj.set_ptr(&spinor_wave_functions_(0, 0, offs));
            fout["K_points"][id]["spinor_wave_functions"].write(j, wfj);
        }
        Platform::barrier(parameters_.mpi_grid().communicator(_dim_col_));
    }
}

void K_point::load_wave_functions(int id, Band* band__)
{
    hdf5_tree fin("sirius.h5", false);
    
    int mtgk_size_in;
    fin["K_points"][id].read("mtgk_size", &mtgk_size_in);
    if (mtgk_size_in != mtgk_size()) error(__FILE__, __LINE__, "wrong wave-function size");

    band_energies_.resize(parameters_.num_bands());
    fin["K_points"][id].read("band_energies", &band_energies_[0], parameters_.num_bands());

    band_occupancies_.resize(parameters_.num_bands());
    fin["K_points"][id].read("band_occupancies", &band_occupancies_[0], parameters_.num_bands());

    spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
                                          band__->spl_spinor_wf_col().local_size());
    spinor_wave_functions_.allocate();

    mdarray<complex16, 2> wfj(NULL, mtgk_size(), parameters_.num_spins()); 
    for (int jloc = 0; jloc < band__->spl_spinor_wf_col().local_size(); jloc++)
    {
        int j = band__->spl_spinor_wf_col(jloc);
        wfj.set_ptr(&spinor_wave_functions_(0, 0, jloc));
        fin["K_points"][id]["spinor_wave_functions"].read(j, wfj);
    }
}

void K_point::get_fv_eigen_vectors(mdarray<complex16, 2>& fv_evec)
{
    assert(fv_evec.size(0) >= fv_eigen_vectors_.size(0));
    assert(fv_evec.size(1) <= fv_eigen_vectors_.size(1));

    for (int i = 0; i < fv_evec.size(1); i++)
        memcpy(&fv_evec(0, i), &fv_eigen_vectors_(0, i), fv_eigen_vectors_.size(0) * sizeof(complex16));
}

void K_point::get_sv_eigen_vectors(mdarray<complex16, 2>& sv_evec)
{
    assert(sv_evec.size(0) == parameters_.num_bands());
    assert(sv_evec.size(1) == parameters_.num_bands());
    assert(sv_eigen_vectors_.size(1) == parameters_.num_bands());

    sv_evec.zero();

    if (parameters_.num_mag_dims() == 0)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_fv_states());

        for (int i = 0; i < sv_evec.size(1); i++)
            memcpy(&sv_evec(0, i), &sv_eigen_vectors_(0, i), sv_eigen_vectors_.size(0) * sizeof(complex16));
    }
    if (parameters_.num_mag_dims() == 1)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_fv_states());

        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
        {
            for (int i = 0; i < parameters_.num_fv_states(); i++)
            {
                memcpy(&sv_evec(ispn * parameters_.num_fv_states(), ispn * parameters_.num_fv_states() + i), 
                       &sv_eigen_vectors_(0, ispn * parameters_.num_fv_states() + i), 
                       sv_eigen_vectors_.size(0) * sizeof(complex16));
            }
        }
    }
    if (parameters_.num_mag_dims() == 3)
    {
        assert(sv_eigen_vectors_.size(0) == parameters_.num_bands());
        for (int i = 0; i < parameters_.num_bands(); i++)
            memcpy(&sv_evec(0, i), &sv_eigen_vectors_(0, i), sv_eigen_vectors_.size(0) * sizeof(complex16));
    }
}

void K_point::distribute_fv_states_row(Band* band)
{
    for (int i = 0; i < band->spl_fv_states_row().local_size(); i++)
    {
        int ist = band->spl_fv_states_row(i);
        
        // find local column lindex of fv state
        int offset_col = band->spl_fv_states_col().location(_splindex_offs_, ist);
        
        // find column MPI rank which stores this fv state 
        int rank_col = band->spl_fv_states_col().location(_splindex_rank_, ist);

        // copy fv state if this rank stores it
        if (rank_col == band->rank_col())
            memcpy(&fv_states_row_(0, i), &fv_states_col_(0, offset_col), mtgk_size() * sizeof(complex16));

        // send fv state to all column MPI ranks
        Platform::bcast(&fv_states_row_(0, i), mtgk_size(), parameters_.mpi_grid().communicator(1 << _dim_col_), rank_col); 
    }
}
