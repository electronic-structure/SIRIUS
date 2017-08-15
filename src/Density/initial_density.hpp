inline void Density::initial_density()
{
    PROFILE("sirius::Density::initial_density");

    zero();

    if (ctx_.full_potential()) {
        initial_density_full_pot();
    } else {
        initial_density_pseudo();

        init_paw();

        init_density_matrix_for_paw();

        generate_paw_loc_density();
    }
}

inline void Density::initial_density_pseudo()
{
    Radial_integrals_rho_pseudo ri(unit_cell_, ctx_.pw_cutoff(), 20);
    auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri](int iat, double g)
                                                                 {
                                                                     return ri.value(iat, g);
                                                                 });

    if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
        auto z1 = mdarray<double_complex, 1>(&v[0], ctx_.gvec().num_gvec()).checksum();
        DUMP("checksum(rho_pw) : %18.10f %18.10f", z1.real(), z1.imag());
    }

    std::memcpy(&rho_->f_pw_local(0), &v[0], ctx_.gvec().count() * sizeof(double_complex));
    
    double charge = rho_->f_0().real() * unit_cell_.omega();

    if (std::abs(charge - unit_cell_.num_valence_electrons()) > 1e-6) {
        std::stringstream s;
        s << "wrong initial charge density" << std::endl
          << "  integral of the density : " << charge << std::endl
          << "  target number of electrons : " << unit_cell_.num_valence_electrons();
        if (ctx_.comm().rank() == 0) {
            WARNING(s);
        }
        if (ctx_.gvec().comm().rank() == 0) {
            rho_->f_pw_local(0) += (unit_cell_.num_valence_electrons() - charge) / unit_cell_.omega();
        }
    }
    rho_->fft_transform(1);

    if (ctx_.control().print_checksum_) {
        auto cs = rho_->checksum_rg();
        if (ctx_.comm().rank() == 0) {
            DUMP("checksum(rho_rg) : %18.10f", cs);
        }
    }

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        rho_->f_rg(ir) = std::max(rho_->f_rg(ir), 0.0);
    }

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            /* vector field is in Cartesian coordinates */
            vector3d<double> v = unit_cell_.atom(ia).vector_field();

            for (int j0 = 0; j0 < ctx_.fft().grid().size(0); j0++) {
                for (int j1 = 0; j1 < ctx_.fft().grid().size(1); j1++) {
                    for (int j2 = 0; j2 < ctx_.fft().local_size_z(); j2++) {
                        /* get real space fractional coordinate */
                        auto r0 = vector3d<double>(double(j0) / ctx_.fft().grid().size(0),
                                                   double(j1) / ctx_.fft().grid().size(1),
                                                   double(ctx_.fft().offset_z() + j2) / ctx_.fft().grid().size(2));
                        /* index of real space point */
                        int ir = ctx_.fft().grid().index_by_coord(j0, j1, j2);

                        for (int t0 = -1; t0 <= 1; t0++) {
                            for (int t1 = -1; t1 <= 1; t1++) {
                                for (int t2 = -1; t2 <= 1; t2++) {
                                    vector3d<double> r1 = r0 - (unit_cell_.atom(ia).position() + vector3d<double>(t0, t1, t2));
                                    auto r = unit_cell_.get_cartesian_coordinates(r1);
                                    auto a = r.length();

                                    const double R = 2.0;
                                    const double norm = pi * std::pow(R, 3) / 3.0;

                                    if (a <= R) {
                                        magnetization_[0]->f_rg(ir) += v[2] * 1.0 / (std::exp(10 * (a-R)) + 1) / norm;
                                        if (ctx_.num_mag_dims() == 3) {
                                            magnetization_[1]->f_rg(ir) += v[0] * 1.0 / (std::exp(10 * (a-R)) + 1) / norm;
                                            magnetization_[2]->f_rg(ir) += v[1] * 1.0 / (std::exp(10 * (a-R)) + 1) / norm;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            auto cs = rho_vec_[i]->checksum_rg();
            if (ctx_.comm().rank() == 0) {
                DUMP("checksum(rho_vec[%i]_rg) : %18.10f", i, cs);
            }
        }
    }

    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        magnetization_[j]->fft_transform(-1);
    }
    
    /* renormalize charge */
    charge = rho_->f_0().real() * unit_cell_.omega();
    if (ctx_.gvec().comm().rank() == 0) {
        rho_->f_pw_local(0) += (unit_cell_.num_valence_electrons() - charge) / unit_cell_.omega();
    }

    //if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
    //    double_complex cs = mdarray<double_complex, 1>(&rho_->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    //    DUMP("checksum(rho_pw): %20.14f %20.14f", std::real(cs), std::imag(cs));
    //}
}

inline void Density::initial_density_full_pot()
{
    /* initialize smooth density of free atoms */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom(true);
    }
    
    /* compute radial integrals */
    Radial_integrals_rho_free_atom ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 20);
    
    /* compute contribution from free atoms to the interstitial density */
    auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri](int iat, double g)
                                                                 {
                                                                     return ri.value(iat, g);
                                                                 });
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z = mdarray<double_complex, 1>(&v[0], ctx_.gvec().num_gvec()).checksum();
    DUMP("checksum(rho_pw): %18.10f %18.10f", z.real(), z.imag());
    #endif
    
    /* set plane-wave coefficients of the charge density */
    std::memcpy(&rho_->f_pw_local(0), &v[0], ctx_.gvec().count() * sizeof(double_complex));
    /* convert charge deisnty to real space mesh */
    rho_->fft_transform(1);
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    DUMP("checksum(rho_rg): %18.10f", rho_->checksum_rg());
    #endif
    
    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        if (rho_->f_rg(ir) < 0) {
            rho_->f_rg(ir) = 0;
        }
    }
    
    /* mapping between G-shell (global index) and a list of G-vectors (local index) */
    std::map<int, std::vector<int> > gsh_map;
    
    for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
        /* global index of the G-vector */
        int ig = ctx_.gvec().offset() + igloc;
        /* index of the G-vector shell */
        int igsh = ctx_.gvec().shell(ig);
        if (gsh_map.count(igsh) == 0) gsh_map[igsh] = std::vector<int>();
        gsh_map[igsh].push_back(igloc);
    }
    
    /* list of G-shells for the curent MPI rank */
    std::vector<std::pair<int, std::vector<int> > > gsh_list;
    for (auto& i: gsh_map) {
        gsh_list.push_back(std::pair<int, std::vector<int> >(i.first, i.second));
    }
    
    int lmax = 1; //ctx_.lmax_rho();
    int lmmax = Utils::lmmax(lmax); //ctx_.lmmax_rho();
    
    sbessel_approx sba(unit_cell_, lmax, ctx_.gvec().shell_len(1), ctx_.gvec().shell_len(ctx_.gvec().num_shells() - 1), 1e-6);
    
    std::vector<double> gvec_len(gsh_list.size());
    for (int i = 0; i < (int)gsh_list.size(); i++) {
        gvec_len[i] = ctx_.gvec().shell_len(gsh_list[i].first);
    }
    sba.approximate(gvec_len);
    
    auto l_by_lm = Utils::l_by_lm(lmax);
    
    std::vector<double_complex> zil(lmax + 1);
    for (int l = 0; l <= lmax; l++) {
        zil[l] = std::pow(double_complex(0, 1), l);
    }
    
    sddk::timer t3("sirius::Density::initial_density|znulm");
    
    mdarray<double_complex, 3> znulm(sba.nqnu_max(), lmmax, unit_cell_.num_atoms());
    znulm.zero();
    
    auto gvec_ylm = mdarray<double_complex, 2>(lmmax, ctx_.gvec().count());
    for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
        int ig = ctx_.gvec().offset() + igloc;
        auto rtp = SHT::spherical_coordinates(ctx_.gvec().gvec_cart(ig));
        SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_ylm(0, igloc));
    }
    
    #pragma omp parallel for
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        int iat = unit_cell_.atom(ia).type_id();
        
        /* loop over local fraction of G-shells */
        for (int i = 0; i < static_cast<int>(gsh_list.size()); i++) {
            auto& gv = gsh_list[i].second;
            
            /* loop over G-vectors */
            for (int igloc: gv) {
                /* global index of the G-vector */
                int ig = ctx_.gvec().offset() + igloc;
                
                auto z1 = ctx_.gvec_phase_factor(ig, ia) * v[igloc] * fourpi;
                
                for (int lm = 0; lm < lmmax; lm++) {
                    int l = l_by_lm[lm];
                    
                    /* number of expansion coefficients */
                    int nqnu = sba.nqnu(l, iat);
                    
                    auto z2 = z1 * zil[l] * gvec_ylm(lm, igloc);
                    
                    for (int iq = 0; iq < nqnu; iq++) {
                        znulm(iq, lm, ia) += z2 * sba.coeff(iq, i, l, iat);
                    }
                }
            }
        }
    }
    ctx_.comm().allreduce(znulm.at<CPU>(), (int)znulm.size());
    t3.stop();
    
    #ifdef __PRINT_OBJECT_CHECKSUM
    double_complex z3 = znulm.checksum();
    DUMP("checksum(znulm): %18.10f %18.10f", std::real(z3), std::imag(z3));
    #endif
    
    sddk::timer t4("sirius::Density::initial_density|rholm");
    
    SHT sht(lmax);
    
    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        int iat = unit_cell_.atom(ia).type_id();
        
        Spheric_function<spectral, double_complex> rhoylm(lmmax, unit_cell_.atom(ia).radial_grid());
        rhoylm.zero();
        #pragma omp parallel for
        for (int lm = 0; lm < lmmax; lm++) {
            int l = l_by_lm[lm];
            for (int iq = 0; iq < sba.nqnu(l, iat); iq++) {
                double qnu = sba.qnu(iq, l, iat);
                
                for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                    double x = unit_cell_.atom(ia).radial_grid(ir);
                    rhoylm(lm, ir) += znulm(iq, lm, ia) * gsl_sf_bessel_jl(l, x * qnu);
                }
            }
        }
        for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
            double x = unit_cell_.atom(ia).radial_grid(ir);
            rhoylm(0, ir) += (v[0] - unit_cell_.atom(ia).type().free_atom_density(x)) / y00;
        }
        auto rhorlm = convert(rhoylm);
        for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
            for (int lm = 0; lm < lmmax; lm++) {
                rho_->f_mt<index_domain_t::local>(lm, ir, ialoc) = rhorlm(lm, ir);
            }
        }
    }
    
    t4.stop();
    
    /* initialize density of free atoms (not smoothed) */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom(false);
    }
    
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto p = unit_cell_.spl_num_atoms().location(ia);
        
        if (p.rank == ctx_.comm().rank()) {
            /* add density of a free atom */
            for (int ir = 0; ir < unit_cell_.atom(ia).num_mt_points(); ir++) {
                double x = unit_cell_.atom(ia).type().radial_grid(ir);
                rho_->f_mt<index_domain_t::local>(0, ir, p.local_index) += unit_cell_.atom(ia).type().free_atom_density(x) / y00;
            }
        }
    }
    
    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            vector3d<double> v = unit_cell_.atom(ia).vector_field();
            double len = v.length();
            
            int nmtp = unit_cell_.atom(ia).num_mt_points();
            Spline<double> rho(unit_cell_.atom(ia).type().radial_grid());
            double R = unit_cell_.atom(ia).mt_radius();
            for (int ir = 0; ir < nmtp; ir++) {
                double x = unit_cell_.atom(ia).type().radial_grid(ir);
                rho[ir] = rho_->f_mt<index_domain_t::local>(0, ir, ialoc) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
            }
            
            /* maximum magnetization which can be achieved if we smooth density towards MT boundary */
            double q = fourpi * rho.interpolate().integrate(2);
            
            /* if very strong initial magnetization is given */
            if (q < len) {
                /* renormalize starting magnetization */
                for (int x: {0, 1, 2}) {
                    v[x] *= (q / len);
                }
                len = q;
            }
            
            if (len > 1e-8) {
                for (int ir = 0; ir < nmtp; ir++) {
                    magnetization_[0]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho[ir] * v[2] / q / y00;
                }
                
                if (ctx_.num_mag_dims() == 3) {
                    for (int ir = 0; ir < nmtp; ir++) {
                        magnetization_[1]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho[ir] * v[0] / q / y00;
                        magnetization_[2]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho[ir] * v[1] / q / y00;
                    }
                }
            }
        }
    }
}


