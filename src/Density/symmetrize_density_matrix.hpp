inline void Density::symmetrize_density_matrix()
{
    PROFILE("sirius::Density::symmetrize_density_matrix");

    auto& sym = unit_cell_.symmetry();

    int ndm = ctx_.num_mag_comp();

    //TODO its just for test
    if (ctx_.use_symmetry()){

        mdarray<double_complex, 4> dm(unit_cell_.max_mt_basis_size(), unit_cell_.max_mt_basis_size(),
                                      ndm, unit_cell_.num_atoms());
        dm.zero();

        int lmax = unit_cell_.lmax();
        int lmmax = Utils::lmmax(lmax);

        mdarray<double, 2> rotm(lmmax, lmmax);

        double alpha = 1.0 / double(sym.num_mag_sym());

        for (int i = 0; i < sym.num_mag_sym(); i++) {
            int pr = sym.magnetic_group_symmetry(i).spg_op.proper;
            auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
            int isym = sym.magnetic_group_symmetry(i).isym;
            SHT::rotation_matrix(lmax, eang, pr, rotm);
            auto spin_rot_su2 = SHT::rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto& atom_type = unit_cell_.atom(ia).type();
                int ja = sym.sym_table(ia, isym);

                for (int xi1 = 0; xi1 < unit_cell_.atom(ia).mt_basis_size(); xi1++) {
                    int l1  = atom_type.indexb(xi1).l;
                    int lm1 = atom_type.indexb(xi1).lm;
                    int o1  = atom_type.indexb(xi1).order;

                    for (int xi2 = 0; xi2 < unit_cell_.atom(ia).mt_basis_size(); xi2++) {
                        int l2  = atom_type.indexb(xi2).l;
                        int lm2 = atom_type.indexb(xi2).lm;
                        int o2  = atom_type.indexb(xi2).order;

                        double_complex dm_loc_spatial[3] = { {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0} };

                        for (int j = 0; j < ndm; j++) {
                            for (int m3 = -l1; m3 <= l1; m3++) {
                                int lm3 = Utils::lm_by_l_m(l1, m3);
                                int xi3 = atom_type.indexb().index_by_lm_order(lm3, o1);
                                for (int m4 = -l2; m4 <= l2; m4++) {
                                    int lm4 = Utils::lm_by_l_m(l2, m4);
                                    int xi4 = atom_type.indexb().index_by_lm_order(lm4, o2);
                                    //dm(xi1, xi2, j, ia) += density_matrix_(xi3, xi4, j, ja) * rotm(lm1, lm3) * rotm(lm2, lm4) * alpha;
                                    dm_loc_spatial[j] += density_matrix_(xi3, xi4, j, ja) * rotm(lm1, lm3) * rotm(lm2, lm4) * alpha;
                                }
                            }
                        }

                        /* magnetic symmetrization */
                        switch (ndm) {
                            case 1: {
                                dm(xi1, xi2, 0, ia) = dm_loc_spatial[0];
                                break;
                            }

                            case 2: {
                                dm(xi1, xi2, 0, ia) += dm_loc_spatial[0] * spin_rot_su2(0, 0) * std::conj( spin_rot_su2(0, 0) ) +
                                        dm_loc_spatial[1] * spin_rot_su2(0, 1) * std::conj( spin_rot_su2(0, 1) ) ;

                                dm(xi1, xi2, 1, ia) += dm_loc_spatial[1] * spin_rot_su2(1, 1) * std::conj( spin_rot_su2(1, 1) ) +
                                        dm_loc_spatial[0] * spin_rot_su2(1, 0) * std::conj( spin_rot_su2(1, 0) ) ;
                                break;
                            }

                            case 3: {
                                double_complex spin_dm[2][2]=
                                {
                                        { dm_loc_spatial[0],              dm_loc_spatial[2] },
                                        { std::conj( dm_loc_spatial[2] ), dm_loc_spatial[1] }
                                };

                                for (int i = 0; i < 2; i++ ){
                                    for (int j = 0; j < 2; j++ ){
                                        dm(xi1, xi2, 0, ia) += spin_dm[i][j] * spin_rot_su2(0, i) * std::conj( spin_rot_su2(0, j) );
                                        dm(xi1, xi2, 1, ia) += spin_dm[i][j] * spin_rot_su2(1, i) * std::conj( spin_rot_su2(1, j) );
                                        dm(xi1, xi2, 2, ia) += spin_dm[i][j] * spin_rot_su2(0, i) * std::conj( spin_rot_su2(1, j) );
                                    }
                                }
                                break;
                            }

                            default: {
                                TERMINATE("FATAL ERROR");
                            }
                        }
                    }
                }
            }
        }

        dm >> density_matrix_;
    }

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = dm.checksum();
        DUMP("checksum(density_matrix): %20.14f %20.14f", cs.real(), cs.imag());
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto cs = mdarray<double_complex, 1>(&dm(0, 0, 0, ia), dm.size(0) * dm.size(1) * dm.size(2)).checksum();
            DUMP("checksum(density_matrix(%i)): %20.14f %20.14f", ia, cs.real(), cs.imag());
        }
    }
    #endif
}

