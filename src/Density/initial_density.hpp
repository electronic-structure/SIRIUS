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
                                                                    return ri.value<int>(iat, g);
                                                                });

    if (ctx_.control().print_checksum_) {
        auto z1 = mdarray<double_complex, 1>(&v[0], ctx_.gvec().count()).checksum();
        ctx_.comm().allreduce(&z1, 1);
        if (ctx_.comm().rank() == 0) {
            print_checksum("rho_pw_init", z1);
        }
    }
    std::copy(v.begin(), v.end(), &rho_->f_pw_local(0));

    double charge = rho_->f_0().real() * unit_cell_.omega();

    if (std::abs(charge - unit_cell_.num_valence_electrons()) > 1e-6) {
        std::stringstream s;
        s << "wrong initial charge density" << std::endl
          << "  integral of the density : " << std::setprecision(12) << charge << std::endl
          << "  target number of electrons : " << std::setprecision(12) << unit_cell_.num_valence_electrons();
        if (ctx_.comm().rank() == 0) {
            WARNING(s);
        }
        //if (ctx_.gvec().comm().rank() == 0) {
        //    rho_->f_pw_local(0) += (unit_cell_.num_valence_electrons() - charge) / unit_cell_.omega();
        //}
    }
    rho_->fft_transform(1);

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        rho_->f_rg(ir) = std::max(rho_->f_rg(ir), 0.0);
    }

    charge = 0;
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        charge += rho_->f_rg(ir);
    }
    charge *= (ctx_.unit_cell().omega() / ctx_.fft().size());
    ctx_.fft().comm().allreduce(&charge, 1);
    
    /* renormalize charge */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
         rho_->f_rg(ir) *= (unit_cell_.num_valence_electrons() / charge);
    }

    if (ctx_.control().print_checksum_) {
        auto cs = rho_->checksum_rg();
        if (ctx_.comm().rank() == 0) {
            print_checksum("rho_rg", cs);
        }
    }

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        double R = ctx_.av_atom_radius();

        auto w = [R](double x)
        {
            /* the constants are picked in such a way that the volume integral of the
               weight function is equal to the volume of the atomic sphere;
               in this case the starting magnetiation in the atomic spehre
               integrates to the starting magnetization vector */

            /* volume of the sphere */
            const double norm = fourpi * std::pow(R, 3) / 3.0;
            return (35.0 / 8) * std::pow(1 - std::pow(x / R, 2), 2) / norm;
            //return 10 * std::pow(1 - x / R, 2) / norm;
            //const double b = 1.1016992073677703;
            //return b * 1.0 /  (std::exp(10 * (a - R)) + 1) / norm;
            //const double norm = pi * std::pow(R, 3) / 3.0;
            //return 1.0 / (std::exp(10 * (x - R)) + 1) / norm;
       };

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_to_grid_map = ctx_.atoms_to_grid_idx_map()[ia];
            vector3d<double> v = unit_cell_.atom(ia).vector_field();

            for (auto coord: atom_to_grid_map) {
                int ir   = coord.first;
                double a = coord.second;
                magnetization_[0]->f_rg(ir) += v[2] * w(a);
                if (ctx_.num_mag_dims() == 3) {
                    magnetization_[1]->f_rg(ir) += v[0] * w(a);
                    magnetization_[2]->f_rg(ir) += v[1] * w(a);
                }
            }
        }
    }
    
    if (ctx_.control().print_checksum_) {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            auto cs = rho_vec_[i]->checksum_rg();
            if (ctx_.comm().rank() == 0) {
                std::stringstream s;
                s << "rho_vec[" << i << "]";
                print_checksum(s.str(), cs);
            }
        }
    }

    rho_->fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        magnetization_[j]->fft_transform(-1);
    }
    
    //== /* renormalize charge */
    //== charge = rho_->f_0().real() * unit_cell_.omega();
    //== if (ctx_.gvec().comm().rank() == 0) {
    //==     rho_->f_pw_local(0) += (unit_cell_.num_valence_electrons() - charge) / unit_cell_.omega();
    //== }

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
                                                                    return ri.value<int>(iat, g);
                                                                });
    
    double v0{0};
    if (ctx_.comm().rank() == 0) {
        v0 = v[0].real();
    }
    ctx_.comm().bcast(&v0, 1, 0);

    if (ctx_.control().print_checksum_) {
        auto z = mdarray<double_complex, 1>(&v[0], ctx_.gvec().count()).checksum();
        ctx_.comm().allreduce(&z, 1);
        if (ctx_.comm().rank() == 0) {
            print_checksum("rho_pw", z);
        }
    }
    
    /* set plane-wave coefficients of the charge density */
    std::memcpy(&rho_->f_pw_local(0), &v[0], ctx_.gvec().count() * sizeof(double_complex));
    /* convert charge density to real space mesh */
    rho_->fft_transform(1);
    
    if (ctx_.control().print_checksum_) {
        auto cs = rho_->checksum_rg();
        if (ctx_.comm().rank() == 0) {
            print_checksum("rho_rg", cs);
        }
    }
    
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
        if (gsh_map.count(igsh) == 0) {
            gsh_map[igsh] = std::vector<int>();
        }
        gsh_map[igsh].push_back(igloc);
    }
    
    /* list of G-shells for the curent MPI rank */
    std::vector<std::pair<int, std::vector<int>>> gsh_list;
    for (auto& i: gsh_map) {
        gsh_list.push_back(std::pair<int, std::vector<int>>(i.first, i.second));
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
    
    if (ctx_.control().print_checksum_) {
        double_complex z3 = znulm.checksum();
        if (ctx_.comm().rank() == 0) {
            print_checksum("znulm", z3);
        }
    }
    
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
            rhoylm(0, ir) += (v0 - unit_cell_.atom(ia).type().free_atom_density(x)) / y00;
        }
        auto rhorlm = convert(rhoylm);
        if (ctx_.control().print_checksum_) {
            std::stringstream s;
            s << "rhorlm(" << ia << ")";
            auto cs = rhorlm.checksum();
            print_checksum(s.str(), cs);
        }

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
                rho(ir) = rho_->f_mt<index_domain_t::local>(0, ir, ialoc) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
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
                    magnetization_[0]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho(ir) * v[2] / q / y00;
                }
                
                if (ctx_.num_mag_dims() == 3) {
                    for (int ir = 0; ir < nmtp; ir++) {
                        magnetization_[1]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho(ir) * v[0] / q / y00;
                        magnetization_[2]->f_mt<index_domain_t::local>(0, ir, ialoc) = rho(ir) * v[1] / q / y00;
                    }
                }
            }
        }
    }
}


