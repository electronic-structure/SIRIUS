Potential::Potential(Global& parameters__) : parameters_(parameters__), pseudo_density_order(9)
{
    Timer t("sirius::Potential::Potential");
    
    fft_ = parameters_.reciprocal_lattice()->fft();

    if (parameters_.potential_type() == full_potential)
    {
        lmax_ = std::max(parameters_.lmax_rho(), parameters_.lmax_pot());
        sht_ = new SHT(lmax_);

        int ngv = (use_second_variation) ? 0 : parameters_.reciprocal_lattice()->num_gvec();

        effective_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), parameters_.reciprocal_lattice()->num_gvec());
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j] = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), ngv);
        
        // precompute i^l
        zil_.resize(parameters_.lmax_rho() + 1);
        for (int l = 0; l <= parameters_.lmax_rho(); l++) zil_[l] = pow(complex16(0, 1), l);
        
        zilm_.resize(parameters_.lmmax_rho());
        for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
        {
            for (int m = -l; m <= l; m++, lm++) zilm_[lm] = zil_[l];
        }

        l_by_lm_ = Utils::l_by_lm(lmax_);
        
        coulomb_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot(), parameters_.reciprocal_lattice()->num_gvec());
        coulomb_potential_->allocate(false, true);
        
        xc_potential_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
        xc_potential_->allocate(false, false);
        
        xc_energy_density_ = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
        xc_energy_density_->allocate(false, false);

    }

    if (parameters_.potential_type() == ultrasoft_pseudopotential)
    {
        mdarray<double, 2> vloc_radial_integrals(parameters_.unit_cell()->num_atom_types(), 
                                                 parameters_.reciprocal_lattice()->num_gvec_shells_inner());
        
        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            auto atom_type = parameters_.unit_cell()->atom_type(iat);
            Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid());
            for (int igs = 0; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
            {
                if (igs == 0)
                {
                    for (int ir = 0; ir < s.num_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        s[ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn()) * x;
                    }
                    vloc_radial_integrals(iat, igs) = s.interpolate().integrate(0);
                }
                else
                {
                    double g = parameters_.reciprocal_lattice()->gvec_shell_len(igs);
                    double g2 = pow(g, 2);
                    for (int ir = 0; ir < s.num_points(); ir++) 
                    {
                        double x = atom_type->radial_grid(ir);
                        s[ir] = (x * atom_type->uspp().vloc[ir] + atom_type->zn() * gsl_sf_erf(x)) * sin(g * x);
                    }
                    vloc_radial_integrals(iat, igs) = (s.interpolate().integrate(0) / g - atom_type->zn() * exp(-g2 / 4) / g2);
                }
            }
        }
        
        double fourpi_omega = fourpi / parameters_.unit_cell()->omega();

        std::vector<complex16> vloc_pw(parameters_.reciprocal_lattice()->num_gvec());
        memset(&vloc_pw[0], 0, parameters_.reciprocal_lattice()->num_gvec() * sizeof(complex16));
        for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        {
            int igs = parameters_.reciprocal_lattice()->gvec_shell<global>(ig);
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int iat = parameters_.unit_cell()->atom(ia)->type_id();
                vloc_pw[ig] += conj(parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig, ia)) * vloc_radial_integrals(iat, igs) * fourpi_omega;
            }
        }
        std::vector<double> vloc_it(fft_->size());
        fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &vloc_pw[0]);
        fft_->transform(1);
        fft_->output(&vloc_it[0]);
        
        mdarray<double, 3> vloc_3d_map(&vloc_it[0], fft_->size(0), fft_->size(1), fft_->size(2));
        int nx = fft_->size(0);
        int ny = fft_->size(1);
        int nz = fft_->size(2);

        auto p = parameters_.unit_cell()->unit_cell_parameters();

        FILE* fout = fopen("potential.ted", "w");
        fprintf(fout, "%s\n", parameters_.unit_cell()->chemical_formula().c_str());
        fprintf(fout, "%16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f\n", p.a, p.b, p.c, p.alpha, p.beta, p.gamma);
        fprintf(fout, "%i %i %i\n", nx + 1, ny + 1, nz + 1);
        for (int i0 = 0; i0 <= nx; i0++)
        {
            for (int i1 = 0; i1 <= ny; i1++)
            {
                for (int i2 = 0; i2 <= nz; i2++)
                {
                    fprintf(fout, "%14.8f\n", vloc_3d_map(i0 % nx, i1 % ny, i2 % nz));
                }
            }
        }
        fclose(fout);


        stop_here



            
        


    }


    update();
}

Potential::~Potential()
{
    delete effective_potential_; 
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete effective_magnetic_field_[j];
    delete sht_;
    delete coulomb_potential_;
    delete xc_potential_;
    delete xc_energy_density_;
}

void Potential::update()
{
    if (parameters_.potential_type() == full_potential)
    {
        // compute values of spherical Bessel functions at MT boundary
        sbessel_mt_.set_dimensions(lmax_ + pseudo_density_order + 2, parameters_.unit_cell()->num_atom_types(), 
                                   parameters_.reciprocal_lattice()->num_gvec_shells_inner());
        sbessel_mt_.allocate();

        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            for (int igs = 0; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
            {
                gsl_sf_bessel_jl_array(lmax_ + pseudo_density_order + 1, 
                                       parameters_.reciprocal_lattice()->gvec_shell_len(igs) * parameters_.unit_cell()->atom_type(iat)->mt_radius(), 
                                       &sbessel_mt_(0, iat, igs));
            }
        }

        //===============================================================================
        // compute moments of spherical Bessel functions 
        //  
        // Integrate[SphericalBesselJ[l,a*x]*x^(2+l),{x,0,R},Assumptions->{R>0,a>0,l>=0}]
        // and use relation between Bessel and spherical Bessel functions: 
        // Subscript[j, n](z)=Sqrt[\[Pi]/2]/Sqrt[z]Subscript[J, n+1/2](z) 
        //===============================================================================
        sbessel_mom_.set_dimensions(parameters_.lmax_rho() + 1, parameters_.unit_cell()->num_atom_types(), 
                                    parameters_.reciprocal_lattice()->num_gvec_shells_inner());
        sbessel_mom_.allocate();
        sbessel_mom_.zero();

        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            sbessel_mom_(0, iat, 0) = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), 3) / 3.0; // for |G|=0
            for (int igs = 1; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
            {
                for (int l = 0; l <= parameters_.lmax_rho(); l++)
                {
                    sbessel_mom_(l, iat, igs) = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), 2 + l) * 
                                                sbessel_mt_(l + 1, iat, igs) / parameters_.reciprocal_lattice()->gvec_shell_len(igs);
                }
            }
        }
        
        //==================================================
        // compute Gamma[5/2 + n + l] / Gamma[3/2 + l] / R^l
        //
        // use Gamma[1/2 + p] = (2p - 1)!!/2^p Sqrt[Pi]
        //==================================================
        gamma_factors_R_.set_dimensions(parameters_.lmax_rho() + 1, parameters_.unit_cell()->num_atom_types());
        gamma_factors_R_.allocate();
        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            for (int l = 0; l <= parameters_.lmax_rho(); l++)
            {
                //double p = pow(2.0, pseudo_density_order + 1) * pow(parameters_.atom_type(iat)->mt_radius(), l);
                double Rl = pow(parameters_.unit_cell()->atom_type(iat)->mt_radius(), l);

                int n_min = (2 * l + 3);
                int n_max = (2 * l + 1) + (2 * pseudo_density_order + 2);
                // split factorial product into two parts to avoid overflow
                double f1 = 1.0;
                double f2 = 1.0;
                for (int n = n_min; n <= n_max; n += 2) 
                {
                    if (f1 < Rl) 
                    {
                        f1 *= (n / 2.0);
                    }
                    else
                    {
                        f2 *= (n / 2.0);
                    }
                }
                gamma_factors_R_(l, iat) = (f1 / Rl) * f2;
            }
        }
    }
}

void Potential::poisson_vmt(mdarray<Spheric_function<complex16>*, 1>& rho_ylm, mdarray<Spheric_function<complex16>*, 1>& vh_ylm, 
                            mdarray<complex16, 2>& qmt)
{
    Timer t("sirius::Potential::poisson:vmt");

    qmt.zero();
    
    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);

        double R = parameters_.unit_cell()->atom(ia)->mt_radius();
        int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();
       
        #pragma omp parallel default(shared)
        {
            std::vector<complex16> g1;
            std::vector<complex16> g2;

            Spline<complex16> rholm(nmtp, parameters_.unit_cell()->atom(ia)->type()->radial_grid());

            #pragma omp for
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
            {
                int l = l_by_lm_[lm];

                for (int ir = 0; ir < nmtp; ir++) rholm[ir] = (*rho_ylm(ialoc))(lm, ir);
                rholm.interpolate();

                // save multipole moment
                qmt(lm, ia) = rholm.integrate(g1, l + 2);
                
                if (lm < parameters_.lmmax_pot())
                {
                    rholm.integrate(g2, 1 - l);
                    
                    double d1 = 1.0 / pow(R, 2 * l + 1); 
                    double d2 = 1.0 / double(2 * l + 1); 
                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double r = parameters_.unit_cell()->atom(ia)->type()->radial_grid(ir);

                        complex16 vlm = (1.0 - pow(r / R, 2 * l + 1)) * g1[ir] / pow(r, l + 1) +
                                        (g2[nmtp - 1] - g2[ir]) * pow(r, l) - 
                                        (g1[nmtp - 1] - g1[ir]) * pow(r, l) * d1;

                        (*vh_ylm(ialoc))(lm, ir) = fourpi * vlm * d2;
                    }
                }
            }
        }
        
        // nuclear potential
        for (int ir = 0; ir < nmtp; ir++)
        {
            double r = parameters_.unit_cell()->atom(ia)->type()->radial_grid(ir);
            (*vh_ylm(ialoc))(0, ir) -= fourpi * y00 * parameters_.unit_cell()->atom(ia)->type()->zn() / r;
        }

        // nuclear multipole moment
        qmt(0, ia) -= parameters_.unit_cell()->atom(ia)->type()->zn() * y00;
    }

    Platform::allreduce(&qmt(0, 0), (int)qmt.size());
}

void Potential::poisson_sum_G(complex16* fpw, mdarray<double, 3>& fl, mdarray<complex16, 2>& flm)
{
    Timer t("sirius::Potential::poisson_sum_G");
    
    flm.zero();

    mdarray<complex16, 2> zm1(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), parameters_.lmmax_rho());

    #pragma omp parallel for default(shared)
    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
    {
        for (int igloc = 0; igloc < parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
            zm1(igloc, lm) = parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc) * conj(fpw[parameters_.reciprocal_lattice()->spl_num_gvec(igloc)] * zilm_[lm]);
    }

    mdarray<complex16, 2> zm2(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), parameters_.unit_cell()->num_atoms());

    for (int l = 0; l <= parameters_.lmax_rho(); l++)
    {
        #pragma omp parallel for default(shared)
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int iat = parameters_.unit_cell()->atom(ia)->type_id();
            for (int igloc = 0; igloc < parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
            {
                zm2(igloc, ia) = fourpi * parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia) *  
                                 fl(l, iat, parameters_.reciprocal_lattice()->gvec_shell<local>(igloc));
            }
        }

        blas<cpu>::gemm(2, 0, 2 * l + 1, parameters_.unit_cell()->num_atoms(), parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), 
                        &zm1(0, Utils::lm_by_l_m(l, -l)), zm1.ld(), &zm2(0, 0), zm2.ld(), 
                        &flm(Utils::lm_by_l_m(l, -l), 0), parameters_.lmmax_rho());
    }
    
    Platform::allreduce(&flm(0, 0), (int)flm.size());
}

void Potential::poisson_pw(mdarray<complex16, 2>& qmt, mdarray<complex16, 2>& qit, complex16* pseudo_pw)
{
    Timer t("sirius::Potential::poisson_pw");
    memset(pseudo_pw, 0, parameters_.reciprocal_lattice()->num_gvec() * sizeof(complex16));
    
    // The following term is added to the plane-wave coefficients of the charge density:
    // Integrate[SphericalBesselJ[l,a*x]*p[x,R]*x^2,{x,0,R},Assumptions->{l>=0,n>=0,R>0,a>0}] / 
    //   Integrate[p[x,R]*x^(2+l),{x,0,R},Assumptions->{h>=0,n>=0,R>0}]
    // i.e. contributon from pseudodensity to l-th channel of plane wave expansion multiplied by 
    // the difference bethween true and interstitial-in-the-mt multipole moments and divided by the 
    // moment of the pseudodensity
    
    #pragma omp parallel default(shared)
    {
        std::vector<complex16> pseudo_pw_pt(parameters_.reciprocal_lattice()->spl_num_gvec().local_size(), complex16(0, 0));

        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int iat = parameters_.unit_cell()->atom(ia)->type_id();

            double R = parameters_.unit_cell()->atom(ia)->mt_radius();

            // compute G-vector independent prefactor
            std::vector<complex16> zp(parameters_.lmmax_rho());
            for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
            {
                for (int m = -l; m <= l; m++, lm++)
                    zp[lm] = (qmt(lm, ia) - qit(lm, ia)) * conj(zil_[l]) * gamma_factors_R_(l, iat);
            }

            for (int igloc = 0; igloc < parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
            {
                int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);
                
                double gR = parameters_.reciprocal_lattice()->gvec_len(ig) * R;
                
                complex16 zt = fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / parameters_.unit_cell()->omega();

                // TODO: add to documentation
                // (2^(1/2+n) Sqrt[\[Pi]] R^-l (a R)^(-(3/2)-n) BesselJ[3/2+l+n,a R] * 
                //   Gamma[5/2+l+n])/Gamma[3/2+l] and BesselJ is expressed in terms of SphericalBesselJ
                if (ig)
                {
                    complex16 zt2(0, 0);
                    for (int l = 0, lm = 0; l <= parameters_.lmax_rho(); l++)
                    {
                        complex16 zt1(0, 0);
                        for (int m = -l; m <= l; m++, lm++) zt1 += parameters_.reciprocal_lattice()->gvec_ylm(lm, igloc) * zp[lm];

                        zt2 += zt1 * sbessel_mt_(l + pseudo_density_order + 1, iat, parameters_.reciprocal_lattice()->gvec_shell<global>(ig));
                    }

                    pseudo_pw_pt[igloc] += zt * zt2 * pow(2.0 / gR, pseudo_density_order + 1);
                }
                else // for |G|=0
                {
                    pseudo_pw_pt[igloc] += zt * y00 * (qmt(0, ia) - qit(0, ia));
                }
            }
        }
        #pragma omp critical
        for (int igloc = 0; igloc < parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++) 
            pseudo_pw[parameters_.reciprocal_lattice()->spl_num_gvec(igloc)] += pseudo_pw_pt[igloc];
    }

    //Platform::allreduce(&pseudo_pw[0], parameters_.num_gvec());
    Platform::allgather(&pseudo_pw[0], parameters_.reciprocal_lattice()->spl_num_gvec().global_offset(), 
                        parameters_.reciprocal_lattice()->spl_num_gvec().local_size());
}

template<> void Potential::add_mt_contribution_to_pw<cpu>()
{
    Timer t("sirius::Potential::add_mt_contribution_to_pw");

    mdarray<complex16, 1> fpw(parameters_.reciprocal_lattice()->num_gvec());
    fpw.zero();

    mdarray<Spline<double>*, 2> svlm(parameters_.lmmax_pot(), parameters_.unit_cell()->num_atoms());
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
        {
            svlm(lm, ia) = new Spline<double>(parameters_.unit_cell()->atom(ia)->num_mt_points(), 
                                              parameters_.unit_cell()->atom(ia)->type()->radial_grid());
            
            for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->num_mt_points(); ir++)
                (*svlm(lm, ia))[ir] = effective_potential_->f_mt<global>(lm, ir, ia);
            
            svlm(lm, ia)->interpolate();
        }
    }
   
    #pragma omp parallel default(shared)
    {
        mdarray<double, 1> vjlm(parameters_.lmmax_pot());

        sbessel_pw<double> jl(parameters_, parameters_.lmax_pot());
        
        #pragma omp for
        for (int igloc = 0; igloc < parameters_.reciprocal_lattice()->spl_num_gvec().local_size(); igloc++)
        {
            int ig = parameters_.reciprocal_lattice()->spl_num_gvec(igloc);

            jl.interpolate(parameters_.reciprocal_lattice()->gvec_len(ig));

            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int iat = parameters_.unit_cell()->atom(ia)->type_id();

                for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
                {
                    int l = l_by_lm_[lm];
                    vjlm(lm) = Spline<double>::integrate(jl(l, iat), svlm(lm, ia));
                }

                complex16 zt(0, 0);
                for (int l = 0; l <= parameters_.lmax_pot(); l++)
                {
                    for (int m = -l; m <= l; m++)
                    {
                        if (m == 0)
                        {
                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
                                  vjlm(Utils::lm_by_l_m(l, m));

                        }
                        else
                        {
                            zt += conj(zil_[l]) * parameters_.reciprocal_lattice()->gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
                                  (SHT::ylm_dot_rlm(l, m, m) * vjlm(Utils::lm_by_l_m(l, m)) + 
                                   SHT::ylm_dot_rlm(l, m, -m) * vjlm(Utils::lm_by_l_m(l, -m)));
                        }
                    }
                }
                fpw(ig) += zt * fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<local>(igloc, ia)) / parameters_.unit_cell()->omega();
            }
        }
    }
    Platform::allreduce(fpw.get_ptr(), (int)fpw.size());
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++) delete svlm(lm, ia);
    }
}

//== #ifdef _GPU_
//== template <> void Potential::add_mt_contribution_to_pw<gpu>()
//== {
//==     // TODO: couple of things to consider: 1) global array jvlm with G-vector shells may be large; 
//==     //                                     2) MPI reduction over thousands of shell may be slow
//==     Timer t("sirius::Potential::add_mt_contribution_to_pw");
//== 
//==     mdarray<complex16, 1> fpw(parameters_.num_gvec());
//==     fpw.zero();
//==     
//==     mdarray<int, 1> kargs(4);
//==     kargs(0) = parameters_.num_atom_types();
//==     kargs(1) = parameters_.max_num_mt_points();
//==     kargs(2) = parameters_.lmax_pot();
//==     kargs(3) = parameters_.lmmax_pot();
//==     kargs.allocate_on_device();
//==     kargs.copy_to_device();
//== 
//==     mdarray<double, 3> vlm_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmmax_pot(), 
//==                                  parameters_.num_atoms());
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==     {
//==         for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//==         {
//==             Spline<double> s(parameters_.atom(ia)->num_mt_points(), 
//==                              parameters_.atom(ia)->type()->radial_grid());
//==             
//==             for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//==                 s[ir] = effective_potential_->f_rlm(lm, ir, ia);
//==             
//==             s.interpolate();
//==             s.get_coefs(&vlm_coefs(0, lm, ia), parameters_.max_num_mt_points());
//==         }
//==     }
//==     vlm_coefs.allocate_on_device();
//==     vlm_coefs.copy_to_device();
//== 
//==     mdarray<int, 1> iat_by_ia(parameters_.num_atoms());
//==     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==         iat_by_ia(ia) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//==     iat_by_ia.allocate_on_device();
//==     iat_by_ia.copy_to_device();
//== 
//==     l_by_lm_.allocate_on_device();
//==     l_by_lm_.copy_to_device();
//==     
//==     //=============
//==     // radial grids
//==     //=============
//==     mdarray<double, 2> r_dr(parameters_.max_num_mt_points() * 2, parameters_.num_atom_types());
//==     mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
//==     for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//==     {
//==         nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
//==         parameters_.atom_type(iat)->radial_grid().get_r_dr(&r_dr(0, iat), parameters_.max_num_mt_points());
//==     }
//==     r_dr.allocate_on_device();
//==     r_dr.async_copy_to_device(-1);
//==     nmtp_by_iat.allocate_on_device();
//==     nmtp_by_iat.async_copy_to_device(-1);
//== 
//==     splindex<block> spl_num_gvec_shells(parameters_.num_gvec_shells(), Platform::num_mpi_ranks(), Platform::mpi_rank());
//==     mdarray<double, 3> jvlm(parameters_.lmmax_pot(), parameters_.num_atoms(), parameters_.num_gvec_shells());
//==     jvlm.zero();
//== 
//==     cuda_create_streams(Platform::num_threads());
//==     #pragma omp parallel
//==     {
//==         int thread_id = Platform::thread_id();
//== 
//==         mdarray<double, 3> jl_coefs(parameters_.max_num_mt_points() * 4, parameters_.lmax_pot() + 1, 
//==                                     parameters_.num_atom_types());
//==         
//==         mdarray<double, 2> jvlm_loc(parameters_.lmmax_pot(), parameters_.num_atoms());
//== 
//==         jvlm_loc.pin_memory();
//==         jvlm_loc.allocate_on_device();
//==             
//==         jl_coefs.pin_memory();
//==         jl_coefs.allocate_on_device();
//== 
//==         sbessel_pw<double> jl(parameters_, parameters_.lmax_pot());
//==         
//==         #pragma omp for
//==         for (int igsloc = 0; igsloc < spl_num_gvec_shells.local_size(); igsloc++)
//==         {
//==             int igs = spl_num_gvec_shells[igsloc];
//== 
//==             jl.interpolate(parameters_.gvec_shell_len(igs));
//== 
//==             for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//==             {
//==                 for (int l = 0; l <= parameters_.lmax_pot(); l++)
//==                     jl(l, iat)->get_coefs(&jl_coefs(0, l, iat), parameters_.max_num_mt_points());
//==             }
//==             jl_coefs.async_copy_to_device(thread_id);
//== 
//==             sbessel_vlm_inner_product_gpu(kargs.get_ptr_device(), parameters_.lmmax_pot(), parameters_.num_atoms(), 
//==                                           iat_by_ia.get_ptr_device(), l_by_lm_.get_ptr_device(), 
//==                                           nmtp_by_iat.get_ptr_device(), r_dr.get_ptr_device(), 
//==                                           jl_coefs.get_ptr_device(), vlm_coefs.get_ptr_device(), jvlm_loc.get_ptr_device(), 
//==                                           thread_id);
//== 
//==             jvlm_loc.async_copy_to_host(thread_id);
//==             
//==             cuda_stream_synchronize(thread_id);
//== 
//==             memcpy(&jvlm(0, 0, igs), &jvlm_loc(0, 0), parameters_.lmmax_pot() * parameters_.num_atoms() * sizeof(double));
//==         }
//==     }
//==     cuda_destroy_streams(Platform::num_threads());
//==     
//==     for (int igs = 0; igs < parameters_.num_gvec_shells(); igs++)
//==         Platform::allreduce(&jvlm(0, 0, igs), parameters_.lmmax_pot() * parameters_.num_atoms());
//== 
//==     #pragma omp parallel for default(shared)
//==     for (int igloc = 0; igloc < parameters_.spl_num_gvec().local_size(); igloc++)
//==     {
//==         int ig = parameters_.spl_num_gvec(igloc);
//==         int igs = parameters_.gvec_shell<local>(igloc);
//== 
//==         for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//==         {
//==             complex16 zt(0, 0);
//==             for (int l = 0; l <= parameters_.lmax_pot(); l++)
//==             {
//==                 for (int m = -l; m <= l; m++)
//==                 {
//==                     if (m == 0)
//==                     {
//==                         zt += conj(zil_[l]) * parameters_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               jvlm(Utils::lm_by_l_m(l, m), ia, igs);
//== 
//==                     }
//==                     else
//==                     {
//==                         zt += conj(zil_[l]) * parameters_.gvec_ylm(Utils::lm_by_l_m(l, m), igloc) * 
//==                               (SHT::ylm_dot_rlm(l, m, m) * jvlm(Utils::lm_by_l_m(l, m), ia, igs) + 
//==                                SHT::ylm_dot_rlm(l, m, -m) * jvlm(Utils::lm_by_l_m(l, -m), ia, igs));
//==                     }
//==                 }
//==             }
//==             fpw(ig) += zt * fourpi * conj(parameters_.gvec_phase_factor<local>(igloc, ia)) / parameters_.omega();
//==         }
//==     }
//== 
//==     Platform::allreduce(fpw.get_ptr(), (int)fpw.size());
//==     for (int ig = 0; ig < parameters_.num_gvec(); ig++) effective_potential_->f_pw(ig) += fpw(ig);
//== 
//==     l_by_lm_.deallocate_on_device();
//== }
//== #endif

void Potential::generate_pw_coefs()
{
    for (int ir = 0; ir < fft_->size(); ir++)
        fft_->input_buffer(ir) = effective_potential()->f_it<global>(ir) * parameters_.step_function(ir);
    
    fft_->transform(-1);
    fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &effective_potential()->f_pw(0));

    if (!use_second_variation) // for full diagonalization we also need Beff(G)
    {
        for (int i = 0; i < parameters_.num_mag_dims(); i++)
        {
            for (int ir = 0; ir < fft_->size(); ir++)
                fft_->input_buffer(ir) = effective_magnetic_field(i)->f_it<global>(ir) * parameters_.step_function(ir);
    
            fft_->transform(-1);
            fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &effective_magnetic_field(i)->f_pw(0));
        }
    }

    if (parameters_.basis_type() == pwlo) 
    {
        switch (parameters_.processing_unit())
        {
            case cpu:
            {
                add_mt_contribution_to_pw<cpu>();
                break;
            }
            #ifdef _GPU_
            case gpu:
            {
                add_mt_contribution_to_pw<gpu>();
                break;
            }
            #endif
            default:
            {
                error_local(__FILE__, __LINE__, "wrong processing unit");
            }
        }
    }
}

//void Potential::check_potential_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    parameters_.fft().input(&effective_potential_->f_it<global>(0));
//    parameters_.fft().transform(-1);
//    parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), &effective_potential_->f_pw(0));
//    
//    SHT sht(parameters_.lmax_pot());
//
//    double diff = 0.0;
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int itp = 0; itp < sht.num_points(); itp++)
//        {
//            double vc[3];
//            for (int x = 0; x < 3; x++) vc[x] = sht.coord(x, itp) * parameters_.atom(ia)->mt_radius();
//
//            double val_it = 0.0;
//            for (int ig = 0; ig < parameters_.num_gvec(); ig++) 
//            {
//                double vgc[3];
//                parameters_.get_coordinates<cartesian, reciprocal>(parameters_.gvec(ig), vgc);
//                val_it += real(effective_potential_->f_pw(ig) * exp(complex16(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//                val_mt += effective_potential_->f_rlm(lm, parameters_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average potential difference at MT boundary : %.12f %.12f\n", diff, diff / parameters_.num_atoms() / sht.num_points());
//}

void Potential::poisson(Periodic_function<double>* rho, Periodic_function<double>* vh)
{
    Timer t("sirius::Potential::poisson");

    // get plane-wave coefficients of the charge density
    fft_->input(&rho->f_it<global>(0));
    fft_->transform(-1);
    fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho->f_pw(0));
    
    mdarray<Spheric_function<complex16>*, 1> rho_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());
    mdarray<Spheric_function<complex16>*, 1> vh_ylm(parameters_.unit_cell()->spl_num_atoms().local_size());
    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        rho_ylm(ialoc) = new Spheric_function<complex16>(rho->f_mt(ialoc), true);
        vh_ylm(ialoc) = new Spheric_function<complex16>(vh->f_mt(ialoc), false);
    }
    
    // true multipole moments
    mdarray<complex16, 2> qmt(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
    poisson_vmt(rho_ylm, vh_ylm, qmt);
    
    // compute multipoles of interstitial density in MT region
    mdarray<complex16, 2> qit(parameters_.lmmax_rho(), parameters_.unit_cell()->num_atoms());
    poisson_sum_G(&rho->f_pw(0), sbessel_mom_, qit);
    
    // compute contribution from the pseudo-charge
    std::vector<complex16> pseudo_pw(parameters_.reciprocal_lattice()->num_gvec());
    poisson_pw(qmt, qit, &pseudo_pw[0]);

    // add interstitial charge density; now pseudo_pw has the correct multipole moments in the muffin-tins
    for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++) pseudo_pw[ig] += rho->f_pw(ig); 
    
    if (check_pseudo_charge)
    {
        poisson_sum_G(&pseudo_pw[0], sbessel_mom_, qit);

        double d = 0.0;
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++) d += abs(qmt(lm, ia) - qit(lm, ia));
        }

        //** parameters_.rti().pseudo_charge_error = d;
    }
    else
    {
        //** parameters_.rti().pseudo_charge_error = 0.0;
    }

    // compute pw coefficients of Hartree potential
    pseudo_pw[0] = 0.0;
    vh->f_pw(0) = 0.0;
    for (int ig = 1; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        vh->f_pw(ig) = pseudo_pw[ig] * fourpi / pow(parameters_.reciprocal_lattice()->gvec_len(ig), 2);

    // compute V_lm at the MT boundary
    mdarray<complex16, 2> vmtlm(parameters_.lmmax_pot(), parameters_.unit_cell()->num_atoms());
    poisson_sum_G(&vh->f_pw(0), sbessel_mt_, vmtlm);
    
    // add boundary condition and convert to Rlm
    Timer* t1 = new Timer("sirius::Potential::poisson:bc");
    mdarray<double, 2> rRl(parameters_.unit_cell()->max_num_mt_points(), parameters_.lmax_pot() + 1);
    int type_id_prev = -1;

    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
        int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();

        if (parameters_.unit_cell()->atom(ia)->type_id() != type_id_prev)
        {
            type_id_prev = parameters_.unit_cell()->atom(ia)->type_id();
        
            double R = parameters_.unit_cell()->atom(ia)->mt_radius();

            #pragma omp parallel for default(shared)
            for (int l = 0; l <= parameters_.lmax_pot(); l++)
            {
                for (int ir = 0; ir < nmtp; ir++)
                    rRl(ir, l) = pow(parameters_.unit_cell()->atom(ia)->type()->radial_grid(ir) / R, l);
            }
        }

        #pragma omp parallel for default(shared)
        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
        {
            int l = l_by_lm_[lm];

            for (int ir = 0; ir < nmtp; ir++)
                (*vh_ylm(ialoc))(lm, ir) += (vmtlm(lm, ia) - (*vh_ylm(ialoc))(lm, nmtp - 1)) * rRl(ir, l);
        }
        vh_ylm(ialoc)->sh_convert(vh->f_mt(ialoc));
    }
    delete t1;
    
    // transform Hartree potential to real space
    fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &vh->f_pw(0));
    fft_->transform(1);
    for (int irloc = 0; irloc < fft_->local_size(); irloc++)
    {
        int ir = fft_->global_index(irloc);
        vh->f_it<local>(irloc) = real(fft_->output_buffer(ir));
    }
    
    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        delete rho_ylm(ialoc);
        delete vh_ylm(ialoc);
    }
}

void Potential::xc(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3], 
                   Periodic_function<double>* vxc, Periodic_function<double>* bxc[3], Periodic_function<double>* exc)
{
    Timer t("sirius::Potential::xc");

    libxc_interface xci;
    
    int raw_size = sht_->num_points() * parameters_.unit_cell()->max_num_mt_points();
    std::vector<double> rhotp_raw(raw_size);
    std::vector<double> vxctp_raw(raw_size);
    std::vector<double> exctp_raw(raw_size);
    std::vector<double> magtp_raw(raw_size);
    std::vector<double> bxctp_raw(raw_size);
    std::vector<double> vecmagtp_raw(raw_size * parameters_.num_mag_dims());
    std::vector<double> vecbxctp_raw(raw_size * parameters_.num_mag_dims());

    Timer t2("sirius::Potential::xc:mt");
    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);

        auto& rgrid = parameters_.unit_cell()->atom(ia)->radial_grid();

        Spheric_function<double> rhotp(&rhotp_raw[0], *sht_, rgrid);
        Spheric_function<double> vxctp(&vxctp_raw[0], *sht_, rgrid);
        Spheric_function<double> exctp(&exctp_raw[0], *sht_, rgrid);
        Spheric_function<double> magtp(&magtp_raw[0], *sht_, rgrid);
        Spheric_function<double> bxctp(&bxctp_raw[0], *sht_, rgrid);
        Spheric_function_vector<double> vecmagtp(&vecmagtp_raw[0], *sht_, rgrid, parameters_.num_mag_dims());
        Spheric_function_vector<double> vecbxctp(&vecbxctp_raw[0], *sht_, rgrid, parameters_.num_mag_dims());

        int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();

        rho->f_mt(ialoc).sh_transform(rhotp);

        double rhomin = 0.0;
        for (int ir = 0; ir < nmtp; ir++)
        {
            for (int itp = 0; itp < sht_->num_points(); itp++) 
            {
                rhomin = std::min(rhomin, rhotp(itp, ir));
                if (rhotp(itp, ir) < 0.0)  rhotp(itp, ir) = 0.0;
            }
        }

        if (rhomin < 0.0)
        {
            std::stringstream s;
            s << "Charge density for atom " << ia << " has negative values" << std::endl
              << "most negatve value : " << rhomin << std::endl
              << "current Rlm expansion of the charge density may be not sufficient, try to increase lmax_rho";
            warning_local(__FILE__, __LINE__, s);
        }

        if (parameters_.num_spins() == 2)
        {
            
            for (int j = 0; j < parameters_.num_mag_dims(); j++)
                magnetization[j]->f_mt(ialoc).sh_transform(vecmagtp[j]);
            
            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    double t = 0.0;
                    for (int j = 0; j < parameters_.num_mag_dims(); j++) t += pow(vecmagtp[j](itp, ir), 2);
                    magtp(itp, ir) = sqrt(t);
                }
            }
        }
        
        if (parameters_.num_spins() == 1) 
        {
            #pragma omp parallel for default(shared)
            for (int ir = 0; ir < nmtp; ir++)
            {
                xci.getxc(sht_->num_points(), &rhotp(0, ir), &vxctp(0, ir), &exctp(0, ir));
            }
        }
        else
        {
            #pragma omp parallel for default(shared)
            for (int ir = 0; ir < nmtp; ir++)
            {
                xci.getxc(sht_->num_points(), &rhotp(0, ir), &magtp(0, ir), &vxctp(0, ir), &bxctp(0, ir), &exctp(0, ir));
            }
        }
        vxctp.sh_transform(vxc->f_mt(ialoc));
        exctp.sh_transform(exc->f_mt(ialoc));

        if (parameters_.num_spins() == 2)
        {
            for (int ir = 0; ir < nmtp; ir++)
            {
                for (int itp = 0; itp < sht_->num_points(); itp++)
                {
                    if (magtp(itp, ir) > 1e-8)
                    {
                        for (int j = 0; j < parameters_.num_mag_dims(); j++)
                            vecbxctp[j](itp, ir) = bxctp(itp, ir) * vecmagtp[j](itp, ir) / magtp(itp, ir);
                    }
                    else
                    {
                        for (int j = 0; j < parameters_.num_mag_dims(); j++) vecbxctp[j](itp, ir) = 0.0;
                    }
                }       
            }
            for (int j = 0; j < parameters_.num_mag_dims(); j++) vecbxctp[j].sh_transform(bxc[j]->f_mt(ialoc));
        }
    }
    t2.stop();
  
    Timer t3("sirius::Potential::xc:it");

    // TODO: this is unreadable and must be reimplemented
    int irloc_size = fft_->local_size();

    double rhomin = 0.0;
    for (int irloc = 0; irloc < irloc_size; irloc++)
    {
        rhomin = std::min(rhomin, rho->f_it<local>(irloc));
        if (rho->f_it<local>(irloc) < 0.0)  rho->f_it<local>(irloc) = 0.0;
    }
    if (rhomin < 0.0)
    {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        warning_local(__FILE__, __LINE__, s);
    }

    #pragma omp parallel default(shared)
    {
        int id = Platform::thread_id(); 
        int m = irloc_size % Platform::num_threads(); // first m threads get n+1 elements
        int pt_nloc = irloc_size / Platform::num_threads() + (id < m ? 1 : 0); // local number of elements: +1 if id < m
        int pt_offs = (id < m) ? pt_nloc * id : m * (pt_nloc + 1) + (id - m) * pt_nloc; // offset

        if (parameters_.num_spins() == 1)
        {
            xci.getxc(pt_nloc, &rho->f_it<local>(pt_offs), &vxc->f_it<local>(pt_offs), &exc->f_it<local>(pt_offs));
        }
        else
        {
            std::vector<double> magit(pt_nloc);
            std::vector<double> bxcit(pt_nloc);

            for (int irloc = 0; irloc < pt_nloc; irloc++)
            {
                double t = 0.0;
                for (int j = 0; j < parameters_.num_mag_dims(); j++) t += pow(magnetization[j]->f_it<local>(pt_offs + irloc), 2);
                magit[irloc] = sqrt(t);
            }
            xci.getxc(pt_nloc, &rho->f_it<local>(pt_offs), &magit[0], &vxc->f_it<local>(pt_offs), &bxcit[0], &exc->f_it<local>(pt_offs));
            
            for (int irloc = 0; irloc < pt_nloc; irloc++)
            {
                if (magit[irloc] > 1e-8)
                {
                    for (int j = 0; j < parameters_.num_mag_dims(); j++)
                        bxc[j]->f_it<local>(pt_offs + irloc) = (bxcit[irloc] / magit[irloc]) * magnetization[j]->f_it<local>(pt_offs + irloc);
                }
                else
                {
                    for (int j = 0; j < parameters_.num_mag_dims(); j++) bxc[j]->f_it<local>(pt_offs + irloc) = 0.0;
                }
            }
        }
    }
}

void Potential::generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3])
{
    Timer t("sirius::Potential::generate_effective_potential");
    
    // zero effective potential and magnetic field
    zero();

    // solve Poisson equation
    poisson(rho, coulomb_potential_);

    // add Hartree potential to the total potential
    effective_potential_->add(coulomb_potential_);

    //if (debug_level > 1) check_potential_continuity_at_mt();
    
    xc(rho, magnetization, xc_potential_, effective_magnetic_field_, xc_energy_density_);
   
    effective_potential_->add(xc_potential_);

    effective_potential_->sync(true, true);
    for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->sync(true, true);

    //if (debug_level > 1) check_potential_continuity_at_mt();
}

void Potential::set_effective_potential_ptr(double* veffmt, double* veffit)
{
    effective_potential_->set_mt_ptr(veffmt);
    effective_potential_->set_it_ptr(veffit);
}

void Potential::copy_to_global_ptr(double* fmt, double* fit, Periodic_function<double>* src)
{
    Periodic_function<double>* dest = new Periodic_function<double>(parameters_, parameters_.lmmax_pot());
    dest->set_mt_ptr(fmt);
    dest->set_it_ptr(fit);
    dest->copy(src);
    dest->sync(true, true);
    delete dest;
}


//** void Potential::copy_xc_potential(double* vxcmt, double* vxcit)
//** {
//**     // create temporary function
//**     Periodic_function<double>* vxc = 
//**         new Periodic_function<double>(parameters_, Argument(arg_lm, parameters_.lmmax_pot()),
//**                                                    Argument(arg_radial, parameters_.max_num_mt_points()));
//**     // set global pointers
//**     vxc->set_mt_ptr(vxcmt);
//**     vxc->set_it_ptr(vxcit);
//**     
//**     // xc_potential is local, vxc is global so we can sync vxc
//**     vxc->copy(xc_potential_);
//**     vxc->sync();
//** 
//**     delete vxc;
//** }
//** 
//** void Potential::copy_effective_magnetic_field(double* beffmt, double* beffit)
//** {
//**     if (parameters_.num_mag_dims() == 0) return;
//**     assert(parameters_.num_spins() == 2);
//**     
//**     // set temporary array wrapper
//**     mdarray<double,4> beffmt_tmp(beffmt, parameters_.lmmax_pot(), parameters_.max_num_mt_points(), 
//**                                  parameters_.num_atoms(), parameters_.num_mag_dims());
//**     mdarray<double,2> beffit_tmp(beffit, parameters_.fft().size(), parameters_.num_mag_dims());
//**     
//**     Periodic_function<double>* bxc[3];
//**     for (int i = 0; i < parameters_.num_mag_dims(); i++)
//**     {
//**         bxc[i] = new Periodic_function<double>(parameters_, Argument(arg_lm, parameters_.lmmax_pot()),
//**                                                             Argument(arg_radial, parameters_.max_num_mt_points()));
//**     }
//** 
//**     if (parameters_.num_mag_dims() == 1)
//**     {
//**         // z
//**         bxc[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
//**         bxc[0]->set_it_ptr(&beffit_tmp(0, 0));
//**     }
//**     
//**     if (parameters_.num_mag_dims() == 3)
//**     {
//**         // z
//**         bxc[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
//**         bxc[0]->set_it_ptr(&beffit_tmp(0, 2));
//**         // x
//**         bxc[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
//**         bxc[1]->set_it_ptr(&beffit_tmp(0, 0));
//**         // y
//**         bxc[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
//**         bxc[2]->set_it_ptr(&beffit_tmp(0, 1));
//**     }
//** 
//**     for (int i = 0; i < parameters_.num_mag_dims(); i++)
//**     {
//**         bxc[i]->copy(effective_magnetic_field_[i]);
//**         bxc[i]->sync();
//**         delete bxc[i];
//**     }
//** }

void Potential::set_effective_magnetic_field_ptr(double* beffmt, double* beffit)
{
    if (parameters_.num_mag_dims() == 0) return;
    assert(parameters_.num_spins() == 2);
    
    // set temporary array wrapper
    mdarray<double,4> beffmt_tmp(beffmt, parameters_.lmmax_pot(), parameters_.unit_cell()->max_num_mt_points(), 
                                 parameters_.unit_cell()->num_atoms(), parameters_.num_mag_dims());
    mdarray<double,2> beffit_tmp(beffit, fft_->size(), parameters_.num_mag_dims());
    
    if (parameters_.num_mag_dims() == 1)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[0]->set_it_ptr(&beffit_tmp(0, 0));
    }
    
    if (parameters_.num_mag_dims() == 3)
    {
        // z
        effective_magnetic_field_[0]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 2));
        effective_magnetic_field_[0]->set_it_ptr(&beffit_tmp(0, 2));
        // x
        effective_magnetic_field_[1]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 0));
        effective_magnetic_field_[1]->set_it_ptr(&beffit_tmp(0, 0));
        // y
        effective_magnetic_field_[2]->set_mt_ptr(&beffmt_tmp(0, 0, 0, 1));
        effective_magnetic_field_[2]->set_it_ptr(&beffit_tmp(0, 1));
    }
}
         
void Potential::zero()
{
    effective_potential_->zero();
    for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->zero();
}

//double Potential::value(double* vc)
//{
//    int ja, jr;
//    double dr, tp[2];
//
//    if (parameters_.is_point_in_mt(vc, ja, jr, dr, tp)) 
//    {
//        double* rlm = new double[parameters_.lmmax_pot()];
//        SHT::spherical_harmonics(parameters_.lmax_pot(), tp[0], tp[1], rlm);
//        double p = 0.0;
//        for (int lm = 0; lm < parameters_.lmmax_pot(); lm++)
//        {
//            double d = (effective_potential_->f_rlm(lm, jr + 1, ja) - effective_potential_->f_rlm(lm, jr, ja)) / 
//                       (parameters_.atom(ja)->type()->radial_grid(jr + 1) - parameters_.atom(ja)->type()->radial_grid(jr));
//
//            p += rlm[lm] * (effective_potential_->f_rlm(lm, jr, ja) + d * dr);
//        }
//        delete rlm;
//        return p;
//    }
//    else
//    {
//        double p = 0.0;
//        for (int ig = 0; ig < parameters_.num_gvec(); ig++)
//        {
//            double vgc[3];
//            parameters_.get_coordinates<cartesian, reciprocal>(parameters_.gvec(ig), vgc);
//            p += real(effective_potential_->f_pw(ig) * exp(complex16(0.0, Utils::scalar_product(vc, vgc))));
//        }
//        return p;
//    }
//}

void Potential::update_atomic_potential()
{
    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
       int ia = parameters_.unit_cell()->atom_symmetry_class(ic)->atom_id(0);
       int nmtp = parameters_.unit_cell()->atom(ia)->num_mt_points();
       
       std::vector<double> veff(nmtp);
       
       for (int ir = 0; ir < nmtp; ir++) veff[ir] = y00 * effective_potential_->f_mt<global>(0, ir, ia);

       parameters_.unit_cell()->atom_symmetry_class(ic)->set_spherical_potential(veff);
    }
    
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        double* veff = &effective_potential_->f_mt<global>(0, 0, ia);
        
        double* beff[] = {NULL, NULL, NULL};
        for (int i = 0; i < parameters_.num_mag_dims(); i++) beff[i] = &effective_magnetic_field_[i]->f_mt<global>(0, 0, ia);
        
        parameters_.unit_cell()->atom(ia)->set_nonspherical_potential(veff, beff);
    }
}

void Potential::save()
{
    if (Platform::mpi_rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);

        effective_potential_->hdf5_write(fout["effective_potential"]);

        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            effective_magnetic_field_[j]->hdf5_write(fout["effective_magnetic_field"].create_node(j));

        fout["effective_potential"].create_node("free_atom_potential");
        for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
        {
            fout["effective_potential"]["free_atom_potential"].write(iat, parameters_.unit_cell()->atom_type(iat)->free_atom_potential());
        }
    }
    Platform::barrier();
}

void Potential::load()
{
    HDF5_tree fout(storage_file_name, false);
    
    effective_potential_->hdf5_read(fout["effective_potential"]);

    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        effective_magnetic_field_[j]->hdf5_read(fout["effective_magnetic_field"][j]);
    
    for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
    {
        fout["effective_potential"]["free_atom_potential"].read(iat, parameters_.unit_cell()->atom_type(iat)->free_atom_potential());
    }

    update_atomic_potential();
}

