/** \file band.hpp
    
    \brief Implementation of methods for Band class.

    \todo look at multithreading in apw_lo and lo_apw blocks 

    \todo k-independent L3 sum

    \todo GPU implementation
*/

template <spin_block_t sblock>
void Band::apply_uj_correction(mdarray<double_complex, 2>& fv_states, mdarray<double_complex, 3>& hpsi)
{
    Timer t("sirius::Band::apply_uj_correction");

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        if (parameters_.unit_cell()->atom(ia)->apply_uj_correction())
        {
            Atom_type* type = parameters_.unit_cell()->atom(ia)->type();

            int offset = parameters_.unit_cell()->atom(ia)->offset_wf();

            int l = parameters_.unit_cell()->atom(ia)->uj_correction_l();

            int nrf = type->indexr().num_rf(l);

            for (int order2 = 0; order2 < nrf; order2++)
            {
                for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                {
                    int idx2 = type->indexb_by_lm_order(lm2, order2);
                    for (int order1 = 0; order1 < nrf; order1++)
                    {
                        double ori = parameters_.unit_cell()->atom(ia)->symmetry_class()->o_radial_integral(l, order2, order1);
                        
                        for (int ist = 0; ist < parameters_.spl_fv_states_col().local_size(); ist++)
                        {
                            for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                            {
                                int idx1 = type->indexb_by_lm_order(lm1, order1);
                                double_complex z1 = fv_states(offset + idx1, ist) * ori;

                                if (sblock == uu)
                                {
                                    hpsi(offset + idx2, ist, 0) += z1 * 
                                        parameters_.unit_cell()->atom(ia)->uj_correction_matrix(lm2, lm1, 0, 0);
                                }

                                if (sblock == dd)
                                {
                                    hpsi(offset + idx2, ist, 1) += z1 *
                                        parameters_.unit_cell()->atom(ia)->uj_correction_matrix(lm2, lm1, 1, 1);
                                }

                                if (sblock == ud)
                                {
                                    hpsi(offset + idx2, ist, 2) += z1 *
                                        parameters_.unit_cell()->atom(ia)->uj_correction_matrix(lm2, lm1, 0, 1);
                                }
                                
                                if (sblock == du)
                                {
                                    hpsi(offset + idx2, ist, 3) += z1 *
                                        parameters_.unit_cell()->atom(ia)->uj_correction_matrix(lm2, lm1, 1, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <spin_block_t sblock>
void Band::apply_hmt_to_apw(int num_gkvec, int ia, mdarray<double_complex, 2>& alm, mdarray<double_complex, 2>& halm)
{
    Timer t("sirius::Band::apply_hmt_to_apw");
    
    Atom* atom = parameters_.unit_cell()->atom(ia);
    Atom_type* type = atom->type();
    
    #pragma omp parallel default(shared)
    {
        std::vector<double_complex> zv(num_gkvec);
        
        #pragma omp for
        for (int j2 = 0; j2 < type->mt_aw_basis_size(); j2++)
        {
            memset(&zv[0], 0, num_gkvec * sizeof(double_complex));

            int lm2 = type->indexb(j2).lm;
            int idxrf2 = type->indexb(j2).idxrf;

            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++)
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));

                if (abs(zsum) > 1e-14) 
                {
                    for (int ig = 0; ig < num_gkvec; ig++) zv[ig] += zsum * alm(ig, j1); 
                }
            } // j1
            
            // surface contribution
            if (sblock == nm || sblock == uu || sblock == dd)
            {
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
            }
            
            memcpy(&halm(0, j2), &zv[0], num_gkvec * sizeof(double_complex));
        } // j2
    }
}

template <spin_block_t sblock>
void Band::set_h_apw_lo(K_point* kp, Atom_type* type, Atom* atom, int ia, mdarray<double_complex, 2>& alm, 
                        mdarray<double_complex, 2>& h)
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

            int lm = kp->gklo_basis_descriptor_col(icol).lm;
            int idxrf = kp->gklo_basis_descriptor_col(icol).idxrf;
            
            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                        
                double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm1, lm));
                
                if (abs(zsum) > 1e-14)
                {
                    for (int igkloc = 0; igkloc < kp->num_gkvec_row(); igkloc++) h(igkloc, icol) += zsum * alm(igkloc, j1);
                }
            }
        }
    }
    
    #pragma omp parallel default(shared)
    {
        std::vector<double_complex> ztmp(kp->num_gkvec_col());
        // lo-apw block
        #pragma omp for
        for (int i = 0; i < kp->num_atom_lo_rows(ia); i++)
        {
            int irow = kp->lo_row(ia, i);

            int lm = kp->gklo_basis_descriptor_row(irow).lm;
            int idxrf = kp->gklo_basis_descriptor_row(irow).idxrf;

            memset(&ztmp[0], 0, kp->num_gkvec_col() * sizeof(double_complex));
        
            for (int j1 = 0; j1 < type->mt_aw_basis_size(); j1++) 
            {
                int lm1 = type->indexb(j1).lm;
                int idxrf1 = type->indexb(j1).idxrf;
                        
                double_complex zsum = atom->hb_radial_integrals_sum_L3<sblock>(idxrf, idxrf1, gaunt_coefs_->gaunt_vector(lm, lm1));

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

template <spin_block_t sblock>
void Band::set_h_it(K_point* kp, Periodic_function<double>* effective_potential, 
                    Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h)
{
    Timer t("sirius::Band::set_h_it");

    #pragma omp parallel for default(shared)
    for (int igk_col = 0; igk_col < kp->num_gkvec_col(); igk_col++) // loop over columns
    {
        for (int igk_row = 0; igk_row < kp->num_gkvec_row(); igk_row++) // for each column loop over rows
        {
            int ig12 = parameters_.reciprocal_lattice()->index_g12(kp->gklo_basis_descriptor_row(igk_row).ig,
                                                                   kp->gklo_basis_descriptor_col(igk_col).ig);
            
            // pw kinetic energy
            double t1 = 0.5 * Utils::scalar_product(kp->gklo_basis_descriptor_row(igk_row).gkvec_cart, 
                                                    kp->gklo_basis_descriptor_col(igk_col).gkvec_cart);
                              
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
                                            double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
                case du:
                {
                    h(igk_row, igk_col) += (effective_magnetic_field[1]->f_pw(ig12) + 
                                            double_complex(0, 1) * effective_magnetic_field[2]->f_pw(ig12));
                    break;
                }
            }
        }
    }
}

template <spin_block_t sblock>
void Band::set_h_lo_lo(K_point* kp, mdarray<double_complex, 2>& h)
{
    Timer t("sirius::Band::set_h_lo_lo");

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
                Atom* atom = parameters_.unit_cell()->atom(ia);
                int lm1 = kp->gklo_basis_descriptor_row(irow).lm; 
                int idxrf1 = kp->gklo_basis_descriptor_row(irow).idxrf; 

                h(irow, icol) += atom->hb_radial_integrals_sum_L3<sblock>(idxrf1, idxrf2, gaunt_coefs_->gaunt_vector(lm1, lm2));
            }
        }
    }
}

template <spin_block_t sblock> 
void Band::set_h(K_point* kp, Periodic_function<double>* effective_potential, 
                 Periodic_function<double>* effective_magnetic_field[3], mdarray<double_complex, 2>& h)
{
    Timer t("sirius::Band::set_h");
   
    // index of column apw coefficients in apw array
    int apw_offset_col = kp->apw_offset_col();
    
    mdarray<double_complex, 2> alm(kp->num_gkvec_loc(), parameters_.unit_cell()->max_mt_aw_basis_size());
    mdarray<double_complex, 2> halm(kp->num_gkvec_row(), parameters_.unit_cell()->max_mt_aw_basis_size());

    h.zero();

    double_complex zone(1, 0);
    
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

