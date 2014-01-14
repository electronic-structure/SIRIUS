/** \file density.hpp
    
    \brief Contains remaining implementation of sirius::Density class.
*/

Density::Density(Global& parameters__) : parameters_(parameters__), gaunt_coefs_(NULL)
{
    fft_ = parameters_.reciprocal_lattice()->fft();

    rho_ = new Periodic_function<double>(parameters_, parameters_.lmmax_rho(), parameters_.reciprocal_lattice()->num_gvec());

    // core density of the pseudopotential method
    if (parameters_.potential_type() == ultrasoft_pseudopotential)
    {
        rho_pseudo_core_ = new Periodic_function<double>(parameters_, 0);
        rho_pseudo_core_->allocate(false, true);
        rho_pseudo_core_->zero();

        generate_pseudo_core_charge_density();
    }

    for (int i = 0; i < parameters_.num_mag_dims(); i++)
    {
        magnetization_[i] = new Periodic_function<double>(parameters_, parameters_.lmmax_rho());
    }

    dmat_spins_.clear();
    dmat_spins_.push_back(std::pair<int, int>(0, 0));
    dmat_spins_.push_back(std::pair<int, int>(1, 1));
    dmat_spins_.push_back(std::pair<int, int>(0, 1));
    
    switch (parameters_.basis_type())
    {
        case apwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<complex16>(parameters_.lmax_apw(), parameters_.lmax_rho(), 
                                                             parameters_.lmax_apw());
            break;
        }
        case pwlo:
        {
            gaunt_coefs_ = new Gaunt_coefficients<complex16>(parameters_.lmax_pw(), parameters_.lmax_rho(), 
                                                             parameters_.lmax_pw());
            break;
        }
        case pw:
        {
            break;
        }
    }

    l_by_lm_ = Utils::l_by_lm(parameters_.lmax_rho());
}

Density::~Density()
{
    delete rho_;
    for (int j = 0; j < parameters_.num_mag_dims(); j++) delete magnetization_[j];
    if (parameters_.potential_type() == ultrasoft_pseudopotential) delete rho_pseudo_core_;
    if (gaunt_coefs_) delete gaunt_coefs_;
}

void Density::set_charge_density_ptr(double* rhomt, double* rhoir)
{
    rho_->set_mt_ptr(rhomt);
    rho_->set_it_ptr(rhoir);
}

void Density::set_magnetization_ptr(double* magmt, double* magir)
{
    if (parameters_.num_mag_dims() == 0) return;
    assert(parameters_.num_spins() == 2);

    // set temporary array wrapper
    mdarray<double, 4> magmt_tmp(magmt, parameters_.lmmax_rho(), parameters_.unit_cell()->max_num_mt_points(), 
                                 parameters_.unit_cell()->num_atoms(), parameters_.num_mag_dims());
    mdarray<double, 2> magir_tmp(magir, fft_->size(), parameters_.num_mag_dims());
    
    if (parameters_.num_mag_dims() == 1)
    {
        // z component is the first and only one
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 0));
    }

    if (parameters_.num_mag_dims() == 3)
    {
        // z component is the first
        magnetization_[0]->set_mt_ptr(&magmt_tmp(0, 0, 0, 2));
        magnetization_[0]->set_it_ptr(&magir_tmp(0, 2));
        // x component is the second
        magnetization_[1]->set_mt_ptr(&magmt_tmp(0, 0, 0, 0));
        magnetization_[1]->set_it_ptr(&magir_tmp(0, 0));
        // y component is the third
        magnetization_[2]->set_mt_ptr(&magmt_tmp(0, 0, 0, 1));
        magnetization_[2]->set_it_ptr(&magir_tmp(0, 1));
    }
}
    
void Density::zero()
{
    rho_->zero();
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->zero();
}

void Density::initial_density()
{
    Timer t("sirius::Density::initial_density");

    zero();
    
    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();

    if (parameters_.potential_type() == full_potential)
    {
        uc->solve_free_atoms();
        
        double mt_charge = 0.0;
        for (int ialoc = 0; ialoc < uc->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = uc->spl_num_atoms(ialoc);
            vector3d<double> v = uc->atom(ia)->vector_field();
            double len = v.length();

            int nmtp = uc->atom(ia)->type()->num_mt_points();
            Spline<double> rho(nmtp, uc->atom(ia)->type()->radial_grid());
            for (int ir = 0; ir < nmtp; ir++)
            {
                rho[ir] = uc->atom(ia)->type()->free_atom_density(ir);
                rho_->f_mt<local>(0, ir, ialoc) = rho[ir] / y00;
            }

            // add charge of the MT sphere
            mt_charge += fourpi * rho.interpolate().integrate(nmtp - 1, 2);

            if (len > 1e-8)
            {
                if (parameters_.num_mag_dims())
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[0]->f_mt<local>(0, ir, ialoc) = 0.2 * rho_->f_mt<local>(0, ir, ialoc) * v[2] / len;
                }

                if (parameters_.num_mag_dims() == 3)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[1]->f_mt<local>(0, ir, ia) = 0.2 * rho_->f_mt<local>(0, ir, ia) * v[0] / len;
                    for (int ir = 0; ir < nmtp; ir++)
                        magnetization_[2]->f_mt<local>(0, ir, ia) = 0.2 * rho_->f_mt<local>(0, ir, ia) * v[1] / len;
                }
            }
        }
        Platform::allreduce(&mt_charge, 1);
        
        // distribute remaining charge
        for (int ir = 0; ir < fft_->size(); ir++)
            rho_->f_it<global>(ir) = (uc->num_electrons() - mt_charge) / uc->volume_it();
    }

    if (parameters_.potential_type() == ultrasoft_pseudopotential)
    {
        mdarray<double, 2> rho_radial_integrals(uc->num_atom_types(), rl->num_gvec_shells_inner());

        sbessel_pw<double> jl(uc, 0);
        for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
        {
            jl.load(rl->gvec_shell_len(igs));

            for (int iat = 0; iat < uc->num_atom_types(); iat++)
            {
                auto atom_type = uc->atom_type(iat);
                Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid()); // not very efficient to create splines 
                                                                                        // for each G-shell, but we do this only once
                for (int ir = 0; ir < s.num_points(); ir++) 
                    s[ir] = jl(ir, 0, iat) * atom_type->uspp().total_charge_density[ir];
                rho_radial_integrals(iat, igs) = s.interpolate().integrate(0) / fourpi; // atomic density from UPF file is multiplied by 4*PI
                                                                                        // we don't need this

                if (igs == 0) 
                {
                    std::cout << "radial_integral : " <<  rho_radial_integrals(iat, 0) * fourpi << std::endl;
                }
            }
        }

        std::vector<complex16> v = rl->make_periodic_function(rho_radial_integrals, rl->num_gvec());

        memcpy(&rho_->f_pw(0), &v[0], rl->num_gvec() * sizeof(complex16));

        if (fabs(rho_->f_pw(0) * uc->omega() - uc->num_valence_electrons()) > 1e-6)
        {
            std::stringstream s;
            s << "wrong initial charge density" << std::endl
              << "  integral of the density : " << real(rho_->f_pw(0) * uc->omega()) << std::endl
              << "  target number of electrons : " << uc->num_valence_electrons();
            warning_local(__FILE__, __LINE__, s);
        }

        fft_->input(rl->num_gvec(), rl->fft_index(), &rho_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_->f_it<global>(0));
        
        // remove possible negative noise
        for (int ir = 0; ir < fft_->size(); ir++)
        {
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }
        
        fft_->input(&rho_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(rl->num_gvec(), rl->fft_index(), &rho_->f_pw(0));
    }

    rho_->sync(true, true);
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->sync(true, true);
}

void Density::add_kpoint_contribution_mt(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                         mdarray<complex16, 4>& mt_complex_density_matrix)
{
    Timer t("sirius::Density::add_kpoint_contribution_mt");
    
    if (occupied_bands.size() == 0) return;
   
    mdarray<complex16, 3> wf1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size(), parameters_.num_spins());
    mdarray<complex16, 3> wf2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size(), parameters_.num_spins());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        int offset_wf = parameters_.unit_cell()->atom(ia)->offset_wf();
        int mt_basis_size = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
        
        for (int i = 0; i < (int)occupied_bands.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                for (int j = 0; j < mt_basis_size; j++)
                {
                    wf1(j, i, ispn) = conj(kp->spinor_wave_function(offset_wf + j, ispn, occupied_bands[i].first));
                    wf2(j, i, ispn) = kp->spinor_wave_function(offset_wf + j, ispn, occupied_bands[i].first) * occupied_bands[i].second;
                }
            }
        }

        for (int j = 0; j < mt_complex_density_matrix.size(2); j++)
        {
            blas<cpu>::gemm(0, 1, mt_basis_size, mt_basis_size, (int)occupied_bands.size(), complex16(1, 0), 
                            &wf1(0, 0, dmat_spins_[j].first), wf1.ld(), 
                            &wf2(0, 0, dmat_spins_[j].second), wf2.ld(), complex16(1, 0), 
                            &mt_complex_density_matrix(0, 0, j, ia), mt_complex_density_matrix.ld());
        }
    }
}

template <int num_mag_dims> 
void Density::reduce_zdens(Atom_type* atom_type, int ialoc, mdarray<complex16, 4>& zdens, mdarray<double, 3>& mt_density_matrix)
{
    mt_density_matrix.zero();
    
    #pragma omp parallel for default(shared)
    for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
    {
        int l2 = atom_type->indexr(idxrf2).l;
        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
            int l1 = atom_type->indexr(idxrf1).l;

            int xi2 = atom_type->indexb().index_by_idxrf(idxrf2);
            for (int lm2 = Utils::lm_by_l_m(l2, -l2); lm2 <= Utils::lm_by_l_m(l2, l2); lm2++, xi2++)
            {
                int xi1 = atom_type->indexb().index_by_idxrf(idxrf1);
                for (int lm1 = Utils::lm_by_l_m(l1, -l1); lm1 <= Utils::lm_by_l_m(l1, l1); lm1++, xi1++)
                {
                    for (int k = 0; k < gaunt_coefs_->num_gaunt(lm1, lm2); k++)
                    {
                        int lm3 = gaunt_coefs_->gaunt(lm1, lm2, k).lm3;
                        complex16 gc = gaunt_coefs_->gaunt(lm1, lm2, k).coef;
                        switch (num_mag_dims)
                        {
                            case 3:
                            {
                                mt_density_matrix(lm3, offs, 2) += 2.0 * real(zdens(xi1, xi2, 2, ialoc) * gc); 
                                mt_density_matrix(lm3, offs, 3) -= 2.0 * imag(zdens(xi1, xi2, 2, ialoc) * gc);
                            }
                            case 1:
                            {
                                mt_density_matrix(lm3, offs, 1) += real(zdens(xi1, xi2, 1, ialoc) * gc);
                            }
                            case 0:
                            {
                                mt_density_matrix(lm3, offs, 0) += real(zdens(xi1, xi2, 0, ialoc) * gc);
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector< std::pair<int, double> > Density::get_occupied_bands_list(Band* band, K_point* kp)
{
    std::vector< std::pair<int, double> > bands;
    for (int jsub = 0; jsub < parameters_.num_sub_bands(); jsub++)
    {
        int j = parameters_.idxbandglob(jsub);
        int jloc = parameters_.idxbandloc(jsub);
        double wo = kp->band_occupancy(j) * kp->weight();
        if (wo > 1e-14) bands.push_back(std::pair<int, double>(jloc, wo));
    }
    return bands;
}

//== // memory-conservative implementation
//== void Density::add_kpoint_contribution_pp(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
//==                                          mdarray<complex16, 4>& pp_complex_density_matrix)
//== {
//==     Timer t("sirius::Density::add_kpoint_contribution_pp");
//== 
//==     if (occupied_bands.size() == 0) return;
//== 
//==     // take only occupied wave-functions
//==     mdarray<complex16, 2> wfs(kp->num_gkvec(), (int)occupied_bands.size());
//==     for (int i = 0; i < (int)occupied_bands.size(); i++)
//==     {
//==         memcpy(&wfs(0, i), &kp->spinor_wave_function(0, 0, occupied_bands[i].first), kp->num_gkvec() * sizeof(complex16));
//==     }
//== 
//==     mdarray<complex16, 2> beta_pw(kp->num_gkvec(), parameters_.unit_cell()->max_mt_basis_size());
//== 
//==     mdarray<complex16, 2> beta_psi(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
//== 
//==     // auxiliary arrays
//==     mdarray<complex16, 2> bp1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
//==     mdarray<complex16, 2> bp2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
//== 
//==     for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
//==     {   
//==         // number of beta functions for a given atom
//==         int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
//== 
//==         kp->generate_beta_pw(&beta_pw(0, 0), ia);
//==         
//==         // compute <beta|Psi>
//==         blas<cpu>::gemm(2, 0, nbf, (int)occupied_bands.size(), kp->num_gkvec(), &beta_pw(0, 0), beta_pw.ld(), 
//==                         &wfs(0, 0), wfs.ld(), &beta_psi(0, 0), beta_psi.ld());
//==         
//==         for (int i = 0; i < (int)occupied_bands.size(); i++)
//==         {
//==             for (int xi = 0; xi < nbf; xi++)
//==             {
//==                 bp1(xi, i) = beta_psi(xi, i);
//==                 bp2(xi, i) = conj(beta_psi(xi, i)) * occupied_bands[i].second;
//==             }
//==         }
//== 
//==         blas<cpu>::gemm(0, 1, nbf, nbf, (int)occupied_bands.size(), complex16(1, 0), &bp1(0, 0), bp1.ld(),
//==                         &bp2(0, 0), bp2.ld(), complex16(1, 0), &pp_complex_density_matrix(0, 0, 0, ia), pp_complex_density_matrix.ld());
//==     }
//== }

// memory-greedy implementation
void Density::add_kpoint_contribution_pp(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                         mdarray<complex16, 4>& pp_complex_density_matrix)
{
    Timer t("sirius::Density::add_kpoint_contribution_pp");

    if (occupied_bands.size() == 0) return;

    // take only occupied wave-functions
    mdarray<complex16, 2> wfs(kp->num_gkvec(), (int)occupied_bands.size());
    for (int i = 0; i < (int)occupied_bands.size(); i++)
    {
        memcpy(&wfs(0, i), &kp->spinor_wave_function(0, 0, occupied_bands[i].first), kp->num_gkvec() * sizeof(complex16));
    }

    int nbf_tot = 0;
    std::vector<int> offsets(parameters_.unit_cell()->num_atoms());
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        offsets[ia] = nbf_tot;
        // add number of beta functions for a given atom
        nbf_tot += parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();
    }

    // <G+k|\beta_{\xi}^{\alpha}>
    mdarray<complex16, 2> beta_pw(kp->num_gkvec(), nbf_tot);

    // <\beta_{\xi}^{\alpha}|\Psi_j>
    mdarray<complex16, 2> beta_psi(nbf_tot, (int)occupied_bands.size());
    
    // collect all |beta>
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        kp->generate_beta_pw(&beta_pw(0, offsets[ia]), ia);
    }
    // compute <beta|Psi>
    blas<cpu>::gemm(2, 0, nbf_tot, (int)occupied_bands.size(), kp->num_gkvec(), &beta_pw(0, 0), beta_pw.ld(), 
                    &wfs(0, 0), wfs.ld(), &beta_psi(0, 0), beta_psi.ld());
    
    #pragma omp parallel
    {
        // auxiliary arrays
        mdarray<complex16, 2> bp1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
        mdarray<complex16, 2> bp2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {   
            // number of beta functions for a given atom
            int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();

            for (int i = 0; i < (int)occupied_bands.size(); i++)
            {
                for (int xi = 0; xi < nbf; xi++)
                {
                    bp1(xi, i) = beta_psi(offsets[ia] + xi, i);
                    bp2(xi, i) = conj(beta_psi(offsets[ia] + xi, i)) * occupied_bands[i].second;
                }
            }

            blas<cpu>::gemm(0, 1, nbf, nbf, (int)occupied_bands.size(), complex16(1, 0), &bp1(0, 0), bp1.ld(),
                            &bp2(0, 0), bp2.ld(), complex16(1, 0), &pp_complex_density_matrix(0, 0, 0, ia), 
                            pp_complex_density_matrix.ld());
        }
    }
}

void Density::add_kpoint_contribution_it(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands)
{
    Timer t("sirius::Density::add_kpoint_contribution_it");
    
    if (occupied_bands.size() == 0) return;

    #pragma omp parallel default(shared) num_threads(Platform::num_fft_threads())
    {
        int thread_id = Platform::thread_id();

        mdarray<double, 2> it_density_matrix(fft_->size(), parameters_.num_mag_dims() + 1);
        it_density_matrix.zero();
        
        mdarray<complex16, 2> wfit(fft_->size(), parameters_.num_spins());

        #pragma omp for
        for (int i = 0; i < (int)occupied_bands.size(); i++)
        {
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
            {
                fft_->input(kp->num_gkvec(), kp->fft_index(), 
                            &kp->spinor_wave_function(parameters_.unit_cell()->mt_basis_size(), ispn, occupied_bands[i].first), 
                            thread_id);
                fft_->transform(1, thread_id);
                fft_->output(&wfit(0, ispn), thread_id);
            }
            
            double w = occupied_bands[i].second / parameters_.unit_cell()->omega();
            
            switch (parameters_.num_mag_dims())
            {
                case 3:
                {
                    for (int ir = 0; ir < fft_->size(); ir++)
                    {
                        complex16 z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
                        it_density_matrix(ir, 2) += 2.0 * real(z);
                        it_density_matrix(ir, 3) -= 2.0 * imag(z);
                    }
                }
                case 1:
                {
                    for (int ir = 0; ir < fft_->size(); ir++)
                        it_density_matrix(ir, 1) += real(wfit(ir, 1) * conj(wfit(ir, 1))) * w;
                }
                case 0:
                {
                    for (int ir = 0; ir < fft_->size(); ir++)
                        it_density_matrix(ir, 0) += real(wfit(ir, 0) * conj(wfit(ir, 0))) * w;
                }
            }
        }

        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                #pragma omp critical
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    magnetization_[1]->f_it<global>(ir) += it_density_matrix(ir, 2);
                    magnetization_[2]->f_it<global>(ir) += it_density_matrix(ir, 3);
                }
            }
            case 1:
            {
                #pragma omp critical
                for (int ir = 0; ir < fft_->size(); ir++)
                {
                    rho_->f_it<global>(ir) += (it_density_matrix(ir, 0) + it_density_matrix(ir, 1));
                    magnetization_[0]->f_it<global>(ir) += (it_density_matrix(ir, 0) - it_density_matrix(ir, 1));
                }
                break;
            }
            case 0:
            {
                #pragma omp critical
                for (int ir = 0; ir < fft_->size(); ir++) 
                    rho_->f_it<global>(ir) += it_density_matrix(ir, 0);
            }
        }
    }
}

void Density::add_q_contribution_to_valence_density(K_set& ks)
{
    Timer t("sirius::Density::add_q_contribution_to_valence_density");

    //========================================================================================
    // if we have ud and du spin blocks, don't compute one of them (du in this implementation)
    // because density matrix is symmetric
    //========================================================================================
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    // complex density matrix
    mdarray<complex16, 4> pp_complex_density_matrix(parameters_.unit_cell()->max_mt_basis_size(), 
                                                    parameters_.unit_cell()->max_mt_basis_size(),
                                                    num_zdmat, parameters_.unit_cell()->num_atoms());
    pp_complex_density_matrix.zero();
    
    //=========================
    // add k-point contribution
    //=========================
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);

        add_kpoint_contribution_pp(ks[ik], occupied_bands, pp_complex_density_matrix);
    }
    Platform::allreduce(pp_complex_density_matrix.get_ptr(), (int)pp_complex_density_matrix.size());

    auto rl = parameters_.reciprocal_lattice();

    std::vector<complex16> f_pw(rl->num_gvec(), complex16(0, 0));
    
    #pragma omp parallel
    {
        std::vector<complex16> f_pw_pt(rl->spl_num_gvec().local_size(), complex16(0, 0));

        #pragma omp for
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            auto atom_type = parameters_.unit_cell()->atom(ia)->type();
            int nbf = atom_type->mt_basis_size();

            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                int idx12 = xi2 * (xi2 + 1) / 2;

                // add diagonal term
                for (int igloc = 0; igloc < rl->spl_num_gvec().local_size(); igloc++)
                {
                    // D_{xi2,xix} * Q(G)_{xi2, xi2}
                    f_pw_pt[igloc] += pp_complex_density_matrix(xi2, xi2, 0, ia) * 
                                      conj(rl->gvec_phase_factor<local>(igloc, ia)) * 
                                      atom_type->uspp().q_pw(igloc, idx12 + xi2);

                }
                // add non-diagonal terms
                for (int xi1 = 0; xi1 < xi2; xi1++, idx12++)
                {
                    for (int igloc = 0; igloc < rl->spl_num_gvec().local_size(); igloc++)
                    {
                        // D_{xi2,xi1} * Q(G)_{xi1, xi2}
                        f_pw_pt[igloc] += 2 * real(pp_complex_density_matrix(xi2, xi1, 0, ia) * 
                                                   conj(rl->gvec_phase_factor<local>(igloc, ia)) * 
                                                   atom_type->uspp().q_pw(igloc, idx12));
                    }
                }
            }
        }
        
        // sum contribution from different atoms
        #pragma omp critical
        for (int igloc = 0; igloc < rl->spl_num_gvec().local_size(); igloc++)
        {
            int ig = rl->spl_num_gvec(igloc);
            f_pw[ig] += f_pw_pt[igloc];
        }
    }
    
    Platform::allgather(&f_pw[0], rl->spl_num_gvec().global_offset(), rl->spl_num_gvec().local_size());

    fft_->input(rl->num_gvec(), rl->fft_index(), &f_pw[0]);
    fft_->transform(1);
    for (int ir = 0; ir < fft_->size(); ir++) rho_->f_it<global>(ir) += real(fft_->output_buffer(ir));
}

void Density::generate_valence_density_mt(K_set& ks)
{
    Timer t("sirius::Density::generate_valence_density_mt");

    //========================================================================================
    // if we have ud and du spin blocks, don't compute one of them (du in this implementation)
    // because density matrix is symmetric
    //========================================================================================
    int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

    // complex density matrix
    mdarray<complex16, 4> mt_complex_density_matrix(parameters_.unit_cell()->max_mt_basis_size(), 
                                                    parameters_.unit_cell()->max_mt_basis_size(),
                                                    num_zdmat, parameters_.unit_cell()->num_atoms());
    mt_complex_density_matrix.zero();
    
    //=========================
    // add k-point contribution
    //=========================
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);
        add_kpoint_contribution_mt(ks[ik], occupied_bands, mt_complex_density_matrix);
    }
    
    mdarray<complex16, 4> mt_complex_density_matrix_loc(parameters_.unit_cell()->max_mt_basis_size(), 
                                                        parameters_.unit_cell()->max_mt_basis_size(),
                                                        num_zdmat, parameters_.unit_cell()->spl_num_atoms().local_size(0));
   
    for (int j = 0; j < num_zdmat; j++)
    {
        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int ialoc = parameters_.unit_cell()->spl_num_atoms().location(_splindex_offs_, ia);
            int rank = parameters_.unit_cell()->spl_num_atoms().location(_splindex_rank_, ia);

            Platform::reduce(&mt_complex_density_matrix(0, 0, j, ia), &mt_complex_density_matrix_loc(0, 0, j, ialoc),
                             parameters_.unit_cell()->max_mt_basis_size() * parameters_.unit_cell()->max_mt_basis_size(),
                             parameters_.mpi_grid().communicator(), rank);
        }
    }
   
    // compute occupation matrix
    if (parameters_.uj_correction())
    {
        Timer* t3 = new Timer("sirius::Density::generate:om");
        
        mdarray<complex16, 4> occupation_matrix(16, 16, 2, 2); 
        
        for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            Atom_type* type = parameters_.unit_cell()->atom(ia)->type();
            
            occupation_matrix.zero();
            for (int l = 0; l <= 3; l++)
            {
                int num_rf = type->indexr().num_rf(l);

                for (int j = 0; j < num_zdmat; j++)
                {
                    for (int order2 = 0; order2 < num_rf; order2++)
                    {
                    for (int lm2 = Utils::lm_by_l_m(l, -l); lm2 <= Utils::lm_by_l_m(l, l); lm2++)
                    {
                        for (int order1 = 0; order1 < num_rf; order1++)
                        {
                        for (int lm1 = Utils::lm_by_l_m(l, -l); lm1 <= Utils::lm_by_l_m(l, l); lm1++)
                        {
                            occupation_matrix(lm1, lm2, dmat_spins_[j].first, dmat_spins_[j].second) +=
                                mt_complex_density_matrix_loc(type->indexb_by_lm_order(lm1, order1),
                                                              type->indexb_by_lm_order(lm2, order2), j, ialoc) *
                                parameters_.unit_cell()->atom(ia)->symmetry_class()->o_radial_integral(l, order1, order2);
                        }
                        }
                    }
                    }
                }
            }
        
            // restore the du block
            for (int lm1 = 0; lm1 < 16; lm1++)
            {
                for (int lm2 = 0; lm2 < 16; lm2++)
                    occupation_matrix(lm2, lm1, 1, 0) = conj(occupation_matrix(lm1, lm2, 0, 1));
            }

            parameters_.unit_cell()->atom(ia)->set_occupation_matrix(&occupation_matrix(0, 0, 0, 0));
        }

        for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
        {
            int rank = parameters_.unit_cell()->spl_num_atoms().location(_splindex_rank_, ia);
            parameters_.unit_cell()->atom(ia)->sync_occupation_matrix(rank);
        }

        delete t3;
    }

    int max_num_rf_pairs = parameters_.unit_cell()->max_mt_radial_basis_size() * 
                           (parameters_.unit_cell()->max_mt_radial_basis_size() + 1) / 2;
    
    // real density matrix
    mdarray<double, 3> mt_density_matrix(parameters_.lmmax_rho(), max_num_rf_pairs, parameters_.num_mag_dims() + 1);
    
    Timer t1("sirius::Density::generate:sum_zdens", false);
    Timer t2("sirius::Density::generate:expand_lm", false);
    mdarray<double, 2> rf_pairs(parameters_.unit_cell()->max_num_mt_points(), max_num_rf_pairs);
    mdarray<double, 3> dlm(parameters_.lmmax_rho(), parameters_.unit_cell()->max_num_mt_points(), 
                           parameters_.num_mag_dims() + 1);
    for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
    {
        int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
        Atom_type* atom_type = parameters_.unit_cell()->atom(ia)->type();

        int nmtp = atom_type->num_mt_points();
        int num_rf_pairs = atom_type->mt_radial_basis_size() * (atom_type->mt_radial_basis_size() + 1) / 2;
        
        t1.start();

        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                reduce_zdens<3>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 1:
            {
                reduce_zdens<1>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
            case 0:
            {
                reduce_zdens<0>(atom_type, ialoc, mt_complex_density_matrix_loc, mt_density_matrix);
                break;
            }
        }
        t1.stop();
        
        t2.start();
        // collect radial functions
        for (int idxrf2 = 0; idxrf2 < atom_type->mt_radial_basis_size(); idxrf2++)
        {
            int offs = idxrf2 * (idxrf2 + 1) / 2;
            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
            {
                // off-diagonal pairs are taken two times: d_{12}*f_1*f_2 + d_{21}*f_2*f_1 = d_{12}*2*f_1*f_2
                int n = (idxrf1 == idxrf2) ? 1 : 2; 
                for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->type()->num_mt_points(); ir++)
                {
                    rf_pairs(ir, offs + idxrf1) = n * parameters_.unit_cell()->atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                                      parameters_.unit_cell()->atom(ia)->symmetry_class()->radial_function(ir, idxrf2); 
                }
            }
        }
        for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
        {
            blas<cpu>::gemm(0, 1, parameters_.lmmax_rho(), nmtp, num_rf_pairs, 
                            &mt_density_matrix(0, 0, j), mt_density_matrix.ld(), 
                            &rf_pairs(0, 0), rf_pairs.ld(), &dlm(0, 0, j), dlm.ld());
        }

        int sz = parameters_.lmmax_rho() * nmtp * (int)sizeof(double);
        switch (parameters_.num_mag_dims())
        {
            case 3:
            {
                memcpy(&magnetization_[1]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 2), sz); 
                memcpy(&magnetization_[2]->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 3), sz);
            }
            case 1:
            {
                for (int ir = 0; ir < nmtp; ir++)
                {
                    for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                    {
                        rho_->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) + dlm(lm, ir, 1);
                        magnetization_[0]->f_mt<local>(lm, ir, ialoc) = dlm(lm, ir, 0) - dlm(lm, ir, 1);
                    }
                }
                break;
            }
            case 0:
            {
                memcpy(&rho_->f_mt<local>(0, 0, ialoc), &dlm(0, 0, 0), sz);
            }
        }
        t2.stop();
    }
}

void Density::generate_valence_density_it(K_set& ks)
{
    Timer t("sirius::Density::generate_valence_density_it");

    //=========================
    // add k-point contribution
    //=========================
    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
    {
        int ik = ks.spl_num_kpoints(ikloc);
        std::vector< std::pair<int, double> > occupied_bands = get_occupied_bands_list(ks.band(), ks[ik]);
        add_kpoint_contribution_it(ks[ik], occupied_bands);
    }
    
    //==========================================================================
    // reduce arrays; assume that each rank did it's own fraction of the density
    //==========================================================================
    Platform::allreduce(&rho_->f_it<global>(0), fft_->size()); 
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        Platform::allreduce(&magnetization_[j]->f_it<global>(0), fft_->size()); 
}

double Density::core_leakage()
{
    double sum = 0.0;
    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        sum += parameters_.unit_cell()->atom_symmetry_class(ic)->core_leakage() * 
               parameters_.unit_cell()->atom_symmetry_class(ic)->num_atoms();
    }
    return sum;
}

double Density::core_leakage(int ic)
{
    return parameters_.unit_cell()->atom_symmetry_class(ic)->core_leakage();
}

void Density::generate_core_charge_density()
{
    Timer t("sirius::Density::generate_core_charge_density");

    for (int icloc = 0; icloc < parameters_.unit_cell()->spl_num_atom_symmetry_classes().local_size(); icloc++)
    {
        int ic = parameters_.unit_cell()->spl_num_atom_symmetry_classes(icloc);
        parameters_.unit_cell()->atom_symmetry_class(ic)->generate_core_charge_density();
    }

    for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++)
    {
        int rank = parameters_.unit_cell()->spl_num_atom_symmetry_classes().location(_splindex_rank_, ic);
        parameters_.unit_cell()->atom_symmetry_class(ic)->sync_core_charge_density(rank);
    }
}

void Density::generate_pseudo_core_charge_density()
{
    Timer t("sirius::Density::generate_pseudo_core_charge_density");

    auto rl = parameters_.reciprocal_lattice();
    auto uc = parameters_.unit_cell();

    mdarray<double, 2> rho_core_radial_integrals(uc->num_atom_types(), rl->num_gvec_shells_inner());

    sbessel_pw<double> jl(uc, 0);
    for (int igs = 0; igs < rl->num_gvec_shells_inner(); igs++)
    {
        jl.load(rl->gvec_shell_len(igs));

        for (int iat = 0; iat < uc->num_atom_types(); iat++)
        {
            auto atom_type = uc->atom_type(iat);
            Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid()); // not very efficient to create splines 
                                                                                    // for each G-shell, but we do this only once
            for (int ir = 0; ir < s.num_points(); ir++) s[ir] = jl(ir, 0, iat) * atom_type->uspp().core_charge_density[ir];
            rho_core_radial_integrals(iat, igs) = s.interpolate().integrate(2);
        }
    }

    std::vector<complex16> v = rl->make_periodic_function(rho_core_radial_integrals, rl->num_gvec());
    
    fft_->input(rl->num_gvec(), rl->fft_index(), &v[0]);
    fft_->transform(1);
    fft_->output(&rho_pseudo_core_->f_it<global>(0));
}

void Density::generate(K_set& ks)
{
    Timer t("sirius::Density::generate");
    
    double wt = 0.0;
    double ot = 0.0;
    for (int ik = 0; ik < ks.num_kpoints(); ik++)
    {
        wt += ks[ik]->weight();
        for (int j = 0; j < parameters_.num_bands(); j++) ot += ks[ik]->weight() * ks[ik]->band_occupancy(j);
    }

    if (fabs(wt - 1.0) > 1e-12) error_local(__FILE__, __LINE__, "K_point weights don't sum to one");

    if (fabs(ot - parameters_.unit_cell()->num_valence_electrons()) > 1e-8)
    {
        std::stringstream s;
        s << "wrong occupancies" << std::endl
          << "  computed : " << ot << std::endl
          << "  required : " << parameters_.unit_cell()->num_valence_electrons() << std::endl
          << "  difference : " << fabs(ot - parameters_.unit_cell()->num_valence_electrons());
        warning_local(__FILE__, __LINE__, s);
    }

    // zero density and magnetization
    zero();

    // interstitial part is independent of basis type
    generate_valence_density_it(ks);
   
    // for muffin-tin part
    switch (parameters_.basis_type())
    {
        case apwlo:
        {
            generate_valence_density_mt(ks);
            break;
        }
        case pwlo:
        {
            switch (parameters_.processing_unit())
            {
                stop_here
                case cpu:
                {
                    break;
                }
                #ifdef _GPU_
                case gpu:
                {
                    break;
                }
                #endif
                default:
                {
                    error_local(__FILE__, __LINE__, "wrong processing unit");
                }
            }
            break;
        }
        case pw:
        {
            add_q_contribution_to_valence_density(ks);
            break;
        }
    }
    
    if (parameters_.basis_type() == apwlo || parameters_.basis_type() == pwlo)
    {
        generate_core_charge_density();

        // add core contribution
        for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
        {
            int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
            for (int ir = 0; ir < parameters_.unit_cell()->atom(ia)->num_mt_points(); ir++)
                rho_->f_mt<local>(0, ir, ialoc) += parameters_.unit_cell()->atom(ia)->symmetry_class()->core_charge_density(ir) / y00;
        }

        // synctronize muffin-tin part (interstitial is already syncronized with allreduce)
        rho_->sync(true, false);
        for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->sync(true, false);
    }

    std::vector<double> nel_mt;
    double nel_it;
    double nel = rho_->integrate(nel_mt, nel_it);
    
    //if (Platform::mpi_rank() == 0)
    //{
    //    printf("\n");
    //    printf("Charges before symmetrization\n");
    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    {
    //        printf("ia : %i  q : %f\n", ia, nel_mt[ia]);
    //    }
    //    printf("interstitial : %f\n", nel_it);
    //}
    
    if (fabs(nel - parameters_.unit_cell()->num_electrons()) > 1e-5)
    {
        std::stringstream s;
        s << "wrong charge density after k-point summation" << std::endl
          << "obtained value : " << nel << std::endl 
          << "target value : " << parameters_.unit_cell()->num_electrons() << std::endl
          << "difference : " << fabs(nel - parameters_.unit_cell()->num_electrons()) << std::endl;
        if (parameters_.potential_type() == full_potential)
        {
            s << "total core leakage : " << core_leakage();
            for (int ic = 0; ic < parameters_.unit_cell()->num_atom_symmetry_classes(); ic++) 
                s << std::endl << "  atom class : " << ic << ", core leakage : " << core_leakage(ic);
        }
        warning_global(__FILE__, __LINE__, s);
    }

    //if (debug_level > 1) check_density_continuity_at_mt();
}

//** void Density::integrate()
//** {
//**     Timer t("sirius::Density::integrate");
//** 
//**     //** parameters_.rti().total_charge = rho_->integrate(parameters_.rti().mt_charge, parameters_.rti().it_charge); 
//** 
//**     //** for (int j = 0; j < parameters_.num_mag_dims(); j++)
//**     //** {
//**     //**     parameters_.rti().total_magnetization[j] = 
//**     //**         magnetization_[j]->integrate(parameters_.rti().mt_magnetization[j], parameters_.rti().it_magnetization[j]);
//**     //** }
//** }

//void Density::check_density_continuity_at_mt()
//{
//    // generate plane-wave coefficients of the potential in the interstitial region
//    parameters_.fft().input(&rho_->f_it<global>(0));
//    parameters_.fft().transform(-1);
//    parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), &rho_->f_pw(0));
//    
//    SHT sht(parameters_.lmax_rho());
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
//                val_it += real(rho_->f_pw(ig) * exp(complex16(0.0, Utils::scalar_product(vc, vgc))));
//            }
//
//            double val_mt = 0.0;
//            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
//                val_mt += rho_->f_rlm(lm, parameters_.atom(ia)->num_mt_points() - 1, ia) * sht.rlm_backward(lm, itp);
//
//            diff += fabs(val_it - val_mt);
//        }
//    }
//    printf("Total and average charge difference at MT boundary : %.12f %.12f\n", diff, diff / parameters_.num_atoms() / sht.num_points());
//}


void Density::save()
{
    if (Platform::mpi_rank() == 0)
    {
        HDF5_tree fout(storage_file_name, false);
        rho_->hdf5_write(fout["density"]);
        for (int j = 0; j < parameters_.num_mag_dims(); j++)
            magnetization_[j]->hdf5_write(fout["magnetization"].create_node(j));
    }
}

void Density::load()
{
    HDF5_tree fout(storage_file_name, false);
    rho_->hdf5_read(fout["density"]);
    for (int j = 0; j < parameters_.num_mag_dims(); j++)
        magnetization_[j]->hdf5_read(fout["magnetization"][j]);
}

