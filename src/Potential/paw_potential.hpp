/*
 *  Created on: May 3, 2016
 *      Author: isivkov
 */

inline void Potential::init_PAW()
{
    paw_potential_data_.clear();
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

        paw_potential_data_t ppd;

        ppd.atom_ = &atom;

        ppd.ia = ia;

        ppd.ia_paw = ia_paw;

        /* allocate potential */
        ppd.ae_potential_ = mdarray<double, 3>(lm_max_rho, num_mt_points, ctx_.num_mag_dims() + 1, memory_t::host, "pdd.ae_potential_");
        ppd.ps_potential_ = mdarray<double, 3>(lm_max_rho, num_mt_points, ctx_.num_mag_dims() + 1, memory_t::host, "pdd.ps_potential_");

        ppd.core_energy_ = atom_type.pp_desc().core_energy;

        paw_potential_data_.push_back(std::move(ppd));
    }

    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia = unit_cell_.paw_atom_index(i);
        int bs = unit_cell_.atom(ia).mt_basis_size();
        max_paw_basis_size_ = std::max(max_paw_basis_size_, bs);
    }
    
    /* initialize dij matrix */
    paw_dij_ = mdarray<double_complex, 4>(max_paw_basis_size_, max_paw_basis_size_, ctx_.num_mag_dims() + 1, unit_cell_.num_paw_atoms(),
                                          memory_t::host, "paw_dij_");

    /* allocate PAW energy array */
    paw_hartree_energies_.resize(unit_cell_.num_paw_atoms());
    paw_xc_energies_.resize(unit_cell_.num_paw_atoms());
    paw_core_energies_.resize(unit_cell_.num_paw_atoms());
    paw_one_elec_energies_.resize(unit_cell_.num_paw_atoms());
}

inline void Potential::generate_PAW_effective_potential(Density& density)
{
    PROFILE("sirius::Potential::generate_PAW_effective_potential");

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    /* zero PAW arrays */
    std::fill(paw_one_elec_energies_.begin(), paw_one_elec_energies_.end(), 0.0);
    std::fill(paw_hartree_energies_.begin(), paw_hartree_energies_.end(), 0.0);
    std::fill(paw_xc_energies_.begin(), paw_xc_energies_.end(), 0.0);

    /* zero Dij */
    paw_dij_.zero();

    /* calculate xc and hartree for atoms */
    for(int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_potential(paw_potential_data_[i],
                                 *density.get_ae_paw_atom_density(i),
                                 *density.get_ps_paw_atom_density(i),
                                 *density.get_ae_paw_atom_magn(i),
                                 *density.get_ps_paw_atom_magn(i));
    }


    /* calculate PAW Dij matrix */
    #pragma omp parallel for
    for(int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        calc_PAW_local_Dij(paw_potential_data_[i], paw_dij_);

        calc_PAW_one_elec_energy(paw_potential_data_[i], density.density_matrix(), paw_dij_);
    }

    // collect Dij and add to atom d_mtrx
    comm_.allreduce(&paw_dij_(0, 0, 0, 0), static_cast<int>(paw_dij_.size()));

    // add paw Dij to uspp Dij
    add_paw_Dij_to_atom_Dmtrx();

    // calc total energy
    double energies[] = {0.0, 0.0, 0.0, 0.0};

    for(int ia = 0; ia < unit_cell_.spl_num_paw_atoms().local_size(); ia++) {
        energies[0] += paw_potential_data_[ia].hartree_energy_;
        energies[1] += paw_potential_data_[ia].xc_energy_;
        energies[2] += paw_potential_data_[ia].one_elec_energy_;
        energies[3] += paw_potential_data_[ia].core_energy_;  // it is light operation
    }

    comm_.allreduce(&energies[0], 4);

    paw_hartree_total_energy_ = energies[0];
    paw_xc_total_energy_ = energies[1];
    paw_one_elec_energy_ = energies[2];
    paw_total_core_energy_ = energies[3];
}

inline double Potential::xc_mt_PAW_nonmagnetic(const Radial_grid& rgrid,
                                               mdarray<double, 3>& out_atom_pot,
                                               mdarray<double, 2>& full_rho_lm,
                                               const std::vector<double>& rho_core)
{
    int lmmax = static_cast<int>(full_rho_lm.size(0));

    Spheric_function<spectral,double> out_atom_pot_sf(&out_atom_pot(0,0,0), lmmax, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0), lmmax, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf_new(lmmax, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_rho_lm_sf;

    double invY00 = 1. / y00 ;

    for(int ir = 0; ir < rgrid.num_points(); ir++ )
    {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    Spheric_function<spatial,double> full_rho_tp_sf = transform(sht_.get(), full_rho_lm_sf_new);

    // create potential in theta phi
    Spheric_function<spatial,double> vxc_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

    xc_mt_nonmagnetic(rgrid, xc_func_, full_rho_lm_sf_new, full_rho_tp_sf, vxc_tp_sf, exc_tp_sf);

    out_atom_pot_sf += transform(sht_.get(), vxc_tp_sf);

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_.get(), exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}

inline double Potential::xc_mt_PAW_collinear(const Radial_grid& rgrid,
                                             mdarray<double, 3>& out_atom_pot,
                                             mdarray<double, 2>& full_rho_lm,
                                             mdarray<double, 3>& magnetization_lm,
                                             const std::vector<double>& rho_core)
{
    assert(out_atom_pot.size(2)==2);

    int lmsize_rho = static_cast<int>(full_rho_lm.size(0));

    // make spherical functions for input density
    Spheric_function<spectral,double> full_rho_lm_sf(&full_rho_lm(0,0), lmsize_rho, rgrid);

    Spheric_function<spectral,double> full_rho_lm_sf_new(lmsize_rho, rgrid);

    full_rho_lm_sf_new.zero();
    full_rho_lm_sf_new += full_rho_lm_sf;

    double invY00 = 1. / y00 ;

    for(int ir = 0; ir < rgrid.num_points(); ir++ )
    {
        full_rho_lm_sf_new(0,ir) += invY00 * rho_core[ir];
    }

    // make spherical functions for output potential
    Spheric_function<spectral,double> out_atom_effective_pot_sf(&out_atom_pot(0,0,0),lmsize_rho,rgrid);
    Spheric_function<spectral,double> out_atom_effective_field_sf(&out_atom_pot(0,0,1),lmsize_rho,rgrid);

    // make magnetization from z component in lm components
    Spheric_function<spectral,double> magnetization_Z_lm(&magnetization_lm(0,0,0), lmsize_rho, rgrid );

    // calculate spin up spin down density components in lm components
    // up = 1/2 ( rho + magn );  down = 1/2 ( rho - magn )
    Spheric_function<spectral,double> rho_u_lm_sf =  0.5 * (full_rho_lm_sf_new + magnetization_Z_lm);
    Spheric_function<spectral,double> rho_d_lm_sf =  0.5 * (full_rho_lm_sf_new - magnetization_Z_lm);

    // transform density to theta phi components
    Spheric_function<spatial,double> rho_u_tp_sf = transform(sht_.get(), rho_u_lm_sf );
    Spheric_function<spatial,double> rho_d_tp_sf = transform(sht_.get(), rho_d_lm_sf );

    // create potential in theta phi
    Spheric_function<spatial,double> vxc_u_tp_sf(sht_->num_points(), rgrid);
    Spheric_function<spatial,double> vxc_d_tp_sf(sht_->num_points(), rgrid);

    // create energy in theta phi
    Spheric_function<spatial,double> exc_tp_sf(sht_->num_points(), rgrid);

    // calculate XC
    xc_mt_magnetic(rgrid, xc_func_,
                   rho_u_lm_sf, rho_u_tp_sf,
                   rho_d_lm_sf, rho_d_tp_sf,
                   vxc_u_tp_sf, vxc_d_tp_sf,
                   exc_tp_sf);

    // transform back in lm
    out_atom_effective_pot_sf += transform(sht_.get(), 0.5 * (vxc_u_tp_sf + vxc_u_tp_sf) );
    out_atom_effective_field_sf += transform(sht_.get(), 0.5 * (vxc_u_tp_sf - vxc_u_tp_sf));

    //------------------------
    //--- calculate energy ---
    //------------------------
    Spheric_function<spectral,double> exc_lm_sf = transform(sht_.get(), exc_tp_sf );

    return inner(exc_lm_sf, full_rho_lm_sf_new);
}

inline double Potential::calc_PAW_hartree_potential(Atom& atom,
                                                    const Radial_grid& grid,
                                                    mdarray<double, 2>& full_density,
                                                    mdarray<double, 3>& out_atom_pot)
{
    //---------------------
    //-- calc potential --
    //---------------------
    int lmsize_rho = static_cast<int>(out_atom_pot.size(0));

    Spheric_function<function_domain_t::spectral,double> dens_sf(&full_density(0, 0), lmsize_rho, grid);

    // array passed to poisson solver
    Spheric_function<spectral,double> atom_pot_sf(lmsize_rho, grid);
    atom_pot_sf.zero();

    poisson_vmt<true>(atom, dens_sf, atom_pot_sf);

    // make spher funcs from arrays
    Spheric_function<spectral,double> out_atom_pot_sf(&out_atom_pot(0, 0, 0), lmsize_rho, grid);
    out_atom_pot_sf += atom_pot_sf;


    //---------------------
    //--- calc energy ---
    //---------------------
    std::vector<int> l_by_lm = Utils::l_by_lm( Utils::lmax_by_lmmax(lmsize_rho) );

    // create array for integration
    std::vector<double> intdata(grid.num_points(),0);

    double hartree_energy=0.0;

    for(int lm=0; lm < lmsize_rho; lm++)
    {
        // fill data to integrate
        for(int irad = 0; irad < grid.num_points(); irad++)
        {
            intdata[irad] = full_density(lm,irad) * out_atom_pot(lm, irad, 0) * grid[irad] * grid[irad];
        }

        // create spline from the data
        Spline<double> h_spl(grid,intdata);

        hartree_energy += 0.5 * h_spl.integrate(0);
    }

    return hartree_energy;
}

inline void Potential::calc_PAW_local_potential(paw_potential_data_t& pdd,
                                                mdarray<double, 2>& ae_full_density,
                                                mdarray<double, 2>& ps_full_density,
                                                mdarray<double, 3>& ae_local_magnetization,
                                                mdarray<double, 3>& ps_local_magnetization)
{
    auto& pp_desc = pdd.atom_->type().pp_desc();

    //-----------------------------------------
    //---- Calculation of Hartree potential ---
    //-----------------------------------------

    pdd.ae_potential_.zero();
    pdd.ps_potential_.zero();


    double ae_hartree_energy = calc_PAW_hartree_potential(*pdd.atom_,
                                                          pdd.atom_->radial_grid(),
                                                          ae_full_density,
                                                          pdd.ae_potential_);

    double ps_hartree_energy = calc_PAW_hartree_potential(*pdd.atom_,
                                                          pdd.atom_->radial_grid(),
                                                          ps_full_density,
                                                          pdd.ps_potential_);

    pdd.hartree_energy_ = ae_hartree_energy - ps_hartree_energy;

    //-----------------------------------------
    //---- Calculation of XC potential ---
    //-----------------------------------------
    double ae_xc_energy = 0.0;
    double ps_xc_energy = 0.0;

    switch(ctx_.num_mag_comp())
    {
        case 1:
        {
            ae_xc_energy = xc_mt_PAW_nonmagnetic(pdd.atom_->radial_grid(), pdd.ae_potential_,
                                                 ae_full_density, pp_desc.all_elec_core_charge);

            ps_xc_energy = xc_mt_PAW_nonmagnetic(pdd.atom_->radial_grid(), pdd.ps_potential_,
                                                 ps_full_density, pp_desc.core_charge_density);
        }break;

        case 2:
        {
            ae_xc_energy = xc_mt_PAW_collinear(pdd.atom_->radial_grid(), pdd.ae_potential_, ae_full_density,
                                               ae_local_magnetization, pp_desc.all_elec_core_charge);

            ps_xc_energy = xc_mt_PAW_collinear(pdd.atom_->radial_grid(), pdd.ps_potential_, ps_full_density,
                                               ps_local_magnetization, pp_desc.core_charge_density);

        }break;

        case 3:
        {
            xc_mt_PAW_noncollinear();
            TERMINATE("PAW potential ERROR! Non-collinear is not implemented");
        }break;

        default:
        {
            TERMINATE("PAW local potential error! Wrong number of spins!")
        }break;
    }

    /* save xc energy in pdd structure */
    pdd.xc_energy_ = ae_xc_energy - ps_xc_energy;
}

inline void Potential::calc_PAW_local_Dij(paw_potential_data_t &pdd, mdarray<double_complex,4>& paw_dij)
{
    int paw_ind = pdd.ia_paw;

    auto& atom_type = pdd.atom_->type();

    auto& pp_desc = atom_type.pp_desc();

    /* get lm size for density */
    int lmax = atom_type.indexr().lmax_lo();
    int lmsize_rho = Utils::lmmax(2 * lmax);

    std::vector<int> l_by_lm = Utils::l_by_lm(2 * lmax);

    // TODO: calculate not for every atom but for every atom type
    Gaunt_coefficients<double> GC(lmax, 2 * lmax, lmax, SHT::gaunt_rlm);

    auto &ae_atom_pot = pdd.ae_potential_;
    auto &ps_atom_pot = pdd.ps_potential_;

    mdarray<double, 3> integrals(lmsize_rho, pp_desc.num_beta_radial_functions * (pp_desc.num_beta_radial_functions + 1) / 2,
                                 ctx_.num_mag_comp());

    for(int ispin = 0; ispin < ctx_.num_mag_comp(); ispin++ )
    {
        for(int irb2 = 0; irb2 < pp_desc.num_beta_radial_functions; irb2++)
        {
            for(int irb1 = 0; irb1 <= irb2; irb1++)
            {
                // common index
                int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

                Radial_grid newgrid = atom_type.radial_grid().segment(pp_desc.cutoff_radius_index);

                // create array for integration
                std::vector<double> intdata(newgrid.num_points(),0);

                for (int lm3 = 0; lm3 < lmsize_rho; lm3++) {
                    // fill array
                    for (int irad = 0; irad < newgrid.num_points(); irad++) {
                        double ae_part = pp_desc.all_elec_wfc(irad,irb1) * pp_desc.all_elec_wfc(irad,irb2);
                        double ps_part = pp_desc.pseudo_wfc(irad,irb1) * pp_desc.pseudo_wfc(irad,irb2)  + pp_desc.q_radial_functions_l(irad,iqij,l_by_lm[lm3]);

                        intdata[irad] = ae_atom_pot(lm3,irad,ispin) * ae_part - ps_atom_pot(lm3,irad,ispin) * ps_part;
                    }

                    // create spline from data arrays
                    Spline<double> dij_spl(newgrid,intdata);

                    // integrate
                    integrals(lm3, iqij, ispin) = dij_spl.integrate(0);
                }
            }
        }
    }

    //---- calc Dij ----
    for (int ib2 = 0; ib2 < atom_type.mt_lo_basis_size(); ib2++) {
        for (int ib1 = 0; ib1 <= ib2; ib1++) {
            //int idij = (ib2 * (ib2 + 1)) / 2 + ib1;

            // get lm quantum numbers (lm index) of the basis functions
            int lm1 = atom_type.indexb(ib1).lm;
            int lm2 = atom_type.indexb(ib2).lm;

            //get radial basis functions indices
            int irb1 = atom_type.indexb(ib1).idxrf;
            int irb2 = atom_type.indexb(ib2).idxrf;

            // common index
            int iqij = (irb2 * (irb2 + 1)) / 2 + irb1;

            // get num of non-zero GC
            int num_non_zero_gk = GC.num_gaunt(lm1,lm2);

            for(int ispin = 0; ispin < ctx_.num_mag_comp(); ispin++) {
                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gk; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* add to atom Dij an integral of dij array */
                    paw_dij(ib1, ib2, ispin, paw_ind) += lm3coef.coef * integrals(lm3coef.lm3, iqij, ispin);
                }

                if (ib1 != ib2) {
                    paw_dij(ib2, ib1, ispin, paw_ind) = paw_dij(ib1, ib2, ispin, paw_ind);
                }
            }
        }
    }
}

inline double Potential::calc_PAW_one_elec_energy(paw_potential_data_t& pdd,
                                                  const mdarray<double_complex, 4>& density_matrix,
                                                  const mdarray<double_complex, 4>& paw_dij)
{
    int atom_ind = pdd.ia;
    int paw_ind = pdd.ia_paw;

    double_complex energy = 0.0;

    for (int is = 0; is< ctx_.num_mag_comp(); is++) {
        for (int ib2 = 0; ib2 < pdd.atom_->mt_lo_basis_size(); ib2++ ) {
            for (int ib1 = 0; ib1 < pdd.atom_->mt_lo_basis_size(); ib1++ ) {
                energy += density_matrix(ib1, ib2, is, atom_ind) * paw_dij(ib1, ib2, is, paw_ind);
            }
        }
    }

    if (std::abs(energy.imag()) > 1e-10) {
        std::stringstream s;
        s << "PAW energy is not real: "<< energy;
        TERMINATE(s.str());
    }

    pdd.one_elec_energy_ = energy.real();

    return energy.real();
}

inline void Potential::add_paw_Dij_to_atom_Dmtrx()
{
    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.num_paw_atoms(); i++) {
        int ia = unit_cell_.paw_atom_index(i);
        auto& atom = unit_cell_.atom(ia);

        for (int is = 0; is < ctx_.num_mag_comp(); is++) {
            for (int ib2 = 0; ib2 < atom.mt_lo_basis_size(); ib2++) {
                for (int ib1 = 0; ib1 < atom.mt_lo_basis_size(); ib1++) {
                     atom.d_mtrx(ib1, ib2, is) += paw_dij_(ib1, ib2, is, i);
                }
            }
        }
    }
}

