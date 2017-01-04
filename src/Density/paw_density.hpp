/*
 * paw_density.cpp
 *
 *  Created on: Oct 24, 2016
 *      Author: isivkov
 */

inline void Density::init_paw()
{
    paw_density_data_.clear();

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int ia_paw = unit_cell_.spl_num_paw_atoms(i);
        int ia = unit_cell_.paw_atom_index(ia_paw);
        auto& atom = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int num_mt_points = atom_type.num_mt_points();

        int l_max = 2 * atom_type.indexr().lmax_lo();
        int lm_max_rho = Utils::lmmax(l_max);

        paw_density_data_t pdd;

        pdd.atom_ = &atom;

        pdd.ia = ia;

        // allocate density arrays
        pdd.ae_density_ = mdarray<double, 2>(lm_max_rho, num_mt_points);
        pdd.ps_density_ = mdarray<double, 2>(lm_max_rho, num_mt_points);

        pdd.ae_density_.zero();
        pdd.ps_density_.zero();

        // magnetization arrays
        pdd.ae_magnetization_ = mdarray<double, 3>(lm_max_rho, num_mt_points, 3);
        pdd.ps_magnetization_ = mdarray<double, 3>(lm_max_rho, num_mt_points, 3);

        pdd.ae_magnetization_.zero();
        pdd.ps_magnetization_.zero();

        paw_density_data_.push_back(std::move(pdd));
    }
}

inline void Density::init_density_matrix_for_paw()
{
    density_matrix_.zero();

    for (int ipaw = 0; ipaw < unit_cell_.num_paw_atoms(); ipaw++ ) {
        int ia = unit_cell_.paw_atom_index(ipaw);

        auto& atom = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int nbf = atom_type.mt_basis_size();

        auto& occupations = atom_type.pp_desc().occupations;

        // magnetization vector
        vector3d<double> magn = atom.vector_field();

        double norm = magn.length();

        for (int xi = 0; xi < nbf; xi++)
        {
            basis_function_index_descriptor const& basis_func_index_dsc = atom_type.indexb()[xi];

            int rad_func_index = basis_func_index_dsc.idxrf;

            double occ = occupations[rad_func_index];

            int l = basis_func_index_dsc.l;

            switch (ctx_.num_mag_dims())
            {
                case 0:
                {
                    density_matrix_(xi,xi,0,ia) = occ / (double)( 2 * l + 1 );
                    break;
                }

                case 1:
                {
                    double nm = (norm < 1. ) ? magn[0] : 1.;

                    density_matrix_(xi,xi,0,ia) = 0.5 * (1.0 + nm ) * occ / (double)( 2 * l + 1 );
                    density_matrix_(xi,xi,1,ia) = 0.5 * (1.0 - nm ) * occ / (double)( 2 * l + 1 );
                    break;
                }
            }

        }
    }
}

inline void Density::generate_paw_atom_density(paw_density_data_t &pdd)
{
    int ia = pdd.ia;

    auto& atom_type = pdd.atom_->type();

    auto& pp_desc = atom_type.pp_desc();

    std::vector<int> l_by_lm = Utils::l_by_lm( 2 * atom_type.indexr().lmax_lo() );

    //TODO calculate not for every atom but for every atom type
    Gaunt_coefficients<double> GC(atom_type.indexr().lmax_lo(),
                                  2 * atom_type.indexr().lmax_lo(),
                                  atom_type.indexr().lmax_lo(),
                                  SHT::gaunt_rlm);

    // get density for current atom
    pdd.ae_density_.zero(); //ae_atom_density
    pdd.ps_density_.zero(); //ps_atom_density

    // and magnetization
    pdd.ae_magnetization_.zero(); //ae_atom_magnetization
    pdd.ps_magnetization_.zero(); //ps_atom_magnetization

    // get radial grid to divide density over r^2
    auto &grid = atom_type.radial_grid();

    // iterate over local basis functions (or over lm1 and lm2)
    for (int ib2 = 0; ib2 < atom_type.indexb().size(); ib2++)
    {
        for(int ib1 = 0; ib1 <= ib2; ib1++)
        {
            // get lm quantum numbers (lm index) of the basis functions
            int lm2 = atom_type.indexb(ib2).lm;
            int lm1 = atom_type.indexb(ib1).lm;

            //get radial basis functions indices
            int irb2 = atom_type.indexb(ib2).idxrf;
            int irb1 = atom_type.indexb(ib1).idxrf;

            // index to iterate Qij,
            int iqij = irb2 * (irb2 + 1) / 2 + irb1;

            // get num of non-zero GC
            int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

            double diag_coef = ib1 == ib2 ? 1. : 2. ;

            // add nonzero coefficients
            for(int inz = 0; inz < num_non_zero_gk; inz++)
            {
                auto& lm3coef = GC.gaunt(lm1,lm2,inz);

                // iterate over radial points
                // this part in fortran looks better, is there the same for c++?
                for(int irad = 0; irad < (int)grid.num_points(); irad++)
                {
                    // we need to divide density over r^2 since wave functions are stored multiplied by r
                    double inv_r2 = diag_coef /(grid[irad] * grid[irad]);

                    // TODO for 3 spin dimensions 3th density spin component must be complex
                    // replace order of indices for density from {irad,lm} to {lm,irad}
                    // to be in according with ELK and other SIRIUS code
                    double ae_part = inv_r2 * lm3coef.coef * pp_desc.all_elec_wfc(irad,irb1) * pp_desc.all_elec_wfc(irad,irb2);
                    double ps_part = inv_r2 * lm3coef.coef *
                            (pp_desc.pseudo_wfc(irad,irb1) * pp_desc.pseudo_wfc(irad,irb2) + pp_desc.q_radial_functions_l(irad,iqij,l_by_lm[lm3coef.lm3]));

                    // calculate UP density (or total in case of nonmagnetic)
                    double ae_dens_u = density_matrix_(ib1, ib2, 0, ia).real() * ae_part;
                    double ps_dens_u = density_matrix_(ib1, ib2, 0, ia).real() * ps_part;

                    // add density UP to the total density
                    pdd.ae_density_(lm3coef.lm3, irad) += ae_dens_u;
                    pdd.ps_density_(lm3coef.lm3, irad) += ps_dens_u;

                    switch(ctx_.num_spins()) {
                        case 2: {
                            double ae_dens_d = density_matrix_(ib1, ib2, 1, ia).real() * ae_part;
                            double ps_dens_d = density_matrix_(ib1, ib2, 1, ia).real() * ps_part;

                            // add density DOWN to the total density
                            pdd.ae_density_(lm3coef.lm3, irad) += ae_dens_d;
                            pdd.ps_density_(lm3coef.lm3, irad) += ps_dens_d;

                            // add magnetization to 2nd components (0th and 1st are always zero )
                            pdd.ae_magnetization_(lm3coef.lm3, irad, 0) = ae_dens_u - ae_dens_d;
                            pdd.ps_magnetization_(lm3coef.lm3, irad, 0) = ps_dens_u - ps_dens_d;
                            break;
                        }
                        case 3: {
                            TERMINATE("PAW: non collinear is not implemented");
                        }
                        default: {
                            break;
                        }
                    }
                }
            }
        }
    }
}

inline void Density::generate_paw_loc_density()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        generate_paw_atom_density(paw_density_data_[i]);
    }
}

