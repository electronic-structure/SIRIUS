template<> void kpoint::generate_matching_coefficients_l<1>(int ia, int iat, AtomType* type, int l, int num_gkvec_loc, 
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

template<> void kpoint::generate_matching_coefficients_l<2>(int ia, int iat, AtomType* type, int l, int num_gkvec_loc, 
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

template<> void kpoint::set_fv_h_o<cpu, apwlo>(PeriodicFunction<double>* effective_potential, int num_ranks,
                                               mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o");
    
    int apw_offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    
    mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm( num_gkvec_row(), parameters_.max_mt_aw_basis_size());

    h.zero();
    o.zero();

    complex16 zone(1, 0);
    
    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    {
        Atom* atom = parameters_.atom(ia);
        AtomType* type = atom->type();
        
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

template<> void kpoint::ibs_force<cpu, apwlo>(Band* band, mdarray<double, 2>& ffac, mdarray<double, 2>& force)
{
    Timer t("sirius::kpoint::ibs_force");

    int apw_offset_col = (band->num_ranks() > 1) ? num_gkvec_row() : 0;

    mdarray<complex16, 2> ha(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> oa(apwlo_basis_size_row(), apwlo_basis_size_col());
    
    mdarray<complex16, 2> vha(apwlo_basis_size_row(), apwlo_basis_size_col());
    mdarray<complex16, 2> voa(apwlo_basis_size_row(), apwlo_basis_size_col());
    
    mdarray<complex16, 2> alm(num_gkvec_loc(), parameters_.max_mt_aw_basis_size());
    mdarray<complex16, 2> halm(num_gkvec_row(), parameters_.max_mt_aw_basis_size());
    
    mdarray<complex16, 2> dm(parameters_.num_fv_states(), parameters_.num_fv_states());
    dm.zero();
    
    mdarray<complex16, 2> zf(parameters_.num_fv_states(), parameters_.num_fv_states());
    
    if (band->num_ranks() == 1)
    {
        mdarray<complex16, 2> zm1(apwlo_basis_size(), parameters_.num_fv_states());
        
        // compute the density matrix
        // TODO: this is a zgemm or pzgemm
        for (int n = 0; n < parameters_.num_bands(); n++)
        {
            for (int i = 0; i < band->spl_fv_states_row().global_size(); i++)
            {
                int ist = i % parameters_.num_fv_states();
                int ispn = i / parameters_.num_fv_states();
                for (int j = 0; j < band->spl_fv_states_row().global_size(); j++)
                {
                    int jst = j % parameters_.num_fv_states();
                    int jspn = j / parameters_.num_fv_states();

                    if (ispn == jspn)
                    {
                        dm(ist, jst) += band_occupancy(n) * conj(sv_eigen_vectors_(i, n)) * sv_eigen_vectors_(j, n); 
                    }
                }
            }
        }
            
        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
        {
            ha.zero();
            oa.zero();
            
            Atom* atom = parameters_.atom(ia);
            AtomType* type = atom->type();

            int iat = parameters_.atom_type_index_by_id(type->id());
            
            generate_matching_coefficients(num_gkvec_loc(), ia, alm);
            
            apply_hmt_to_apw(num_gkvec_row(), ia, alm, halm);
            
            blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &alm(0, 0), alm.ld(), 
                            &alm(apw_offset_col, 0), alm.ld(), &oa(0, 0), oa.ld()); 
                
            // apw-apw block
            blas<cpu>::gemm(0, 2, num_gkvec_row(), num_gkvec_col(), type->mt_aw_basis_size(), &halm(0, 0), halm.ld(), 
                            &alm(apw_offset_col, 0), alm.ld(), &ha(0, 0), ha.ld());
            
            set_fv_h_o_apw_lo(type, atom, ia, apw_offset_col, alm, ha, oa);

            for (int igkloc2 = 0; igkloc2 < num_gkvec_col(); igkloc2++) // loop over columns
            {
                double v2c[3];
                parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc2].igk), v2c);

                for (int igkloc1 = 0; igkloc1 < num_gkvec_row(); igkloc1++) // for each column loop over rows
                {
                    int ig12 = parameters_.index_g12(apwlo_basis_descriptors_row_[igkloc1].ig,
                                                     apwlo_basis_descriptors_col_[igkloc2].ig);
                    double v1c[3];
                    parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc1].igk), v1c);
                    ha(igkloc1, igkloc2) -= 0.5 * Utils::scalar_product(v1c, v2c) * ffac(ig12, iat) * 
                                            conj(parameters_.gvec_phase_factor<global>(ig12, ia));
                    oa(igkloc1, igkloc2) -= ffac(ig12, iat) * conj(parameters_.gvec_phase_factor<global>(ig12, ia));
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
                        parameters_.get_coordinates<cartesian, reciprocal>(gkvec(apwlo_basis_descriptors_col_[igkloc1].igk), vgk);

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

                // zm1 = H * V
                blas<cpu>::gemm(0, 0, apwlo_basis_size(), parameters_.num_fv_states(), apwlo_basis_size(), 
                                &vha(0, 0), vha.ld(), &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(),
                                &zm1(0, 0), zm1.ld());
                
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
                    for (int j = 0; j < parameters_.num_fv_states(); j++) force(x, ia) += weight() * real(dm(i, j) * zf(i, j));
                }

            }
        } //ia
    }






    //mdarray<double, 2> ffac(parameters_.num_gvec_shells(), parameters_.num_atom_types());
    //parameters_.get_step_function_form_factors(fface);


    //mdarray<complex16, 2> sv_evec(band->spl_fv_states_row().global_size(), band->spl_spinor_wf_col().local_size());
    //sv_evec.zero();


    //for (int jloc = 0; jloc < band->spl_spinor_wf_col().local_size(); jloc++)
    //{
    //    for (int iloc = 0; iloc < band->spl_fv_states_row().local_size(); iloc++)
    //        sv_evec(band->spl_fv_states_row(iloc), jloc) = sv_eigen_vectors_(iloc, jloc);
    //}
    //
    //Platform::allreduce(&sv_evec(0, 0), (int)sv_evec.size(), parameters_.mpi_grid().communicator(1 << band->dim_row()));

    //
    //for (int jloc = 0; jloc < band->spl_spinor_wf_col().local_size(); jloc++)
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
    //                dm(ist, jst) += band_occupancy(band->spl_spinor_wf_col(jloc)) * conj(sv_evec(i, jloc)) * sv_evec(j, jloc);
    //            }
    //        }
    //    }
    //}
    //Platform::allreduce(&dm(0, 0), (int)dm.size(), parameters_.mpi_grid().communicator(1 << band->dim_col()));

    //sv_evec.deallocate();
 
 

               


    


}

template<> void kpoint::set_fv_h_o_pw_lo<cpu>(PeriodicFunction<double>* effective_potential, int num_ranks, 
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o_pw_lo");
    
    int offset_col = (num_ranks > 1) ? num_gkvec_row() : 0;
    
    mdarray<Spline<complex16>*, 2> svlo(parameters_.lmmax_pw(), std::max(num_lo_col(), num_lo_row()));

    // first part: compute <G+k|H|lo> and <G+k|lo>

    Timer t1("sirius::kpoint::set_fv_h_o_pw_lo:vlo", false);
    Timer t2("sirius::kpoint::set_fv_h_o_pw_lo:ohk", false);
    Timer t3("sirius::kpoint::set_fv_h_o_pw_lo:hvlo", false);

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

template<> void kpoint::set_fv_h_o<cpu, pwlo>(PeriodicFunction<double>* effective_potential, int num_ranks,
                                              mdarray<complex16, 2>& h, mdarray<complex16, 2>& o)
{
    Timer t("sirius::kpoint::set_fv_h_o");
    
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
