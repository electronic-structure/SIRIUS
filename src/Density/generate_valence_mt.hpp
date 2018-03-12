inline void Density::generate_valence_mt(K_point_set& ks)
{
    PROFILE("sirius::Density::generate_valence_mt");

    /* compute occupation matrix */
    if (ctx_.hubbard_correction())
    {
        STOP();

        // TODO: fix the way how occupation matrix is calculated

        //Timer t3("sirius::Density::generate:om");
        //
        //mdarray<double_complex, 4> occupation_matrix(16, 16, 2, 2); 
        //
        //for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++)
        //{
        //    int ia = unit_cell_.spl_num_atoms(ialoc);
        //    Atom_type* type = unit_cell_.atom(ia)->type();
        //    
        //    occupation_matrix.zero();
        //    for (int l = 0; l <= 3; l++)
        //    {
        //        int num_rf = type->indexr().num_rf(l);

        //        for (int j = 0; j < num_zdmat; j++)
        //        {
        //            for (int order2 = 0; order2 < num_rf; order2++)
        //            {
        //            for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
        //            {
        //                for (int order1 = 0; order1 < num_rf; order1++)
        //                {
        //                for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
        //                {
        //                    occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
        //                        mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
        //                                                      type->indexb_by_lm_order(lm2, order2), j, ialoc) *
        //                        unit_cell_.atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
        //                }
        //                }
        //            }
        //            }
        //        }
        //    }
        //
        //    // restore the du block
        //    for (int lm1 = 0; lm1 < 16; lm1++)
        //    {
        //        for (int lm2 = 0; lm2 < 16; lm2++)
        //            occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
        //    }

        //    unit_cell_.atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        //}

        //for (int ia = 0; ia < unit_cell_.num_atoms(); ia++)
        //{
        //    int rank = unit_cell_.spl_num_atoms().local_rank(ia);
        //    unit_cell_.atom(ia)->sync_occupation_matrix(ctx_.comm(), rank);
        //}
    }

    int max_num_rf_pairs = unit_cell_.max_mt_radial_basis_size() * 
                           (unit_cell_.max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 3> mt_density_matrix(ctx_.lmmax_rho(), max_num_rf_pairs, ctx_.num_mag_dims() + 1);
    
    mdarray<double, 2> rf_pairs(unit_cell_.max_num_mt_points(), max_num_rf_pairs);
    mdarray<double, 3> dlm(ctx_.lmmax_rho(), unit_cell_.max_num_mt_points(), 
                           ctx_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& atom_type = unit_cell_.atom(ia).type();

        int nmtp = atom_type.num_mt_points();
        int num_rf_pairs = atom_type.mt_radial_basis_size() * (atom_type.mt_radial_basis_size() + 1) / 2;
        
        sddk::timer t1("sirius::Density::generate|sum_zdens");
        switch (ctx_.num_mag_dims()) {
            case 3: {
                reduce_density_matrix<3>(atom_type, ia, density_matrix_, *gaunt_coefs_, mt_density_matrix);
                break;
            }
            case 1: {
                reduce_density_matrix<1>(atom_type, ia, density_matrix_, *gaunt_coefs_, mt_density_matrix);
                break;
            }
            case 0: {
                reduce_density_matrix<0>(atom_type, ia, density_matrix_, *gaunt_coefs_, mt_density_matrix);
                break;
            }
        }
        t1.stop();
        
        sddk::timer t2("sirius::Density::generate|expand_lm");
        /* collect radial functions */
        for (int idxrf2 = 0; idxrf2 < atom_type.mt_radial_basis_size(); idxrf2++) {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                /* off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2 */
                int n = (idxrf1 == idxrf2) ? 1 : 2; 
                for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                    rf_pairs(ir, offs + idxrf1) = n * unit_cell_.atom(ia).symmetry_class().radial_function(ir, idxrf1) * 
                                                      unit_cell_.atom(ia).symmetry_class().radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
            linalg<CPU>::gemm(0, 1, ctx_.lmmax_rho(), nmtp, num_rf_pairs, 
                              &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), 
                              &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = static_cast<int>(ctx_.lmmax_rho() * nmtp * sizeof(double));
        switch (ctx_.num_mag_dims()) {
            case 3: {
                std::memcpy(&magnetization_[1]->f_mt<index_domain_t::local>(0, 0, ialoc), &dlm(0, 0, 2), sz); 
                std::memcpy(&magnetization_[2]->f_mt<index_domain_t::local>(0, 0, ialoc), &dlm(0, 0, 3), sz);
            }
            case 1: {
                for (int ir = 0; ir < nmtp; ir++) {
                    for (int lm = 0; lm < ctx_.lmmax_rho(); lm++) {
                        rho_->f_mt<index_domain_t::local>(lm, ir, ialoc) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_mt<index_domain_t::local>(lm, ir, ialoc) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0: {
                std::memcpy(&rho_->f_mt<index_domain_t::local>(0, 0, ialoc), &dlm(0, 0, 0), sz);
            }
        }
    }
}
