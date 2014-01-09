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
        rho_core_ = new Periodic_function<double>(parameters_, 0, parameters_.reciprocal_lattice()->num_gvec());
        rho_core_->allocate(true, true);
        rho_core_->zero();
        
        mdarray<double, 2> rho_core_radial_integrals(parameters_.unit_cell()->num_atom_types(), 
                                                     parameters_.reciprocal_lattice()->num_gvec_shells_inner());

        sbessel_pw<double> jl(parameters_.unit_cell(), 0);
        for (int igs = 0; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
        {
            jl.load(parameters_.reciprocal_lattice()->gvec_shell_len(igs));

            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                auto atom_type = parameters_.unit_cell()->atom_type(iat);
                Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid()); // not very efficient to create splines 
                                                                                        // for each G-shell, but we do this only once
                for (int ir = 0; ir < s.num_points(); ir++) 
                    s[ir] = jl(ir, 0, iat) * atom_type->uspp().core_charge_density[ir];
                rho_core_radial_integrals(iat, igs) = s.interpolate().integrate(2);
            }
        }
        
        for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        {
            int igs = parameters_.reciprocal_lattice()->gvec_shell<global>(ig);
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int iat = parameters_.unit_cell()->atom(ia)->type_id();
                rho_core_->f_pw(ig) += fourpi * conj(parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig, ia)) * 
                                       rho_core_radial_integrals(iat, igs) / parameters_.unit_cell()->omega();
            }
        }

        fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho_core_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_core_->f_it<global>(0));
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
    if (parameters_.potential_type() == ultrasoft_pseudopotential) delete rho_core_;
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

void Density::initial_density(int type = 0)
{
    zero();

    if (parameters_.potential_type() == full_potential)
    {
        if (type == 0)
        {
            parameters_.unit_cell()->solve_free_atoms();
            
            double mt_charge = 0.0;
            for (int ialoc = 0; ialoc < parameters_.unit_cell()->spl_num_atoms().local_size(); ialoc++)
            {
                int ia = parameters_.unit_cell()->spl_num_atoms(ialoc);
                vector3d<double> v = parameters_.unit_cell()->atom(ia)->vector_field();
                double len = v.length();

                int nmtp = parameters_.unit_cell()->atom(ia)->type()->num_mt_points();
                Spline<double> rho(nmtp, parameters_.unit_cell()->atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++)
                {
                    rho[ir] = parameters_.unit_cell()->atom(ia)->type()->free_atom_density(ir);
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
                rho_->f_it<global>(ir) = (parameters_.unit_cell()->num_electrons() - mt_charge) / parameters_.unit_cell()->volume_it();
        }
    }
    if (parameters_.potential_type() == ultrasoft_pseudopotential)
    {
        mdarray<double, 2> rho_radial_integrals(parameters_.unit_cell()->num_atom_types(), 
                                                parameters_.reciprocal_lattice()->num_gvec_shells_inner());

        sbessel_pw<double> jl(parameters_.unit_cell(), 0);
        for (int igs = 0; igs < parameters_.reciprocal_lattice()->num_gvec_shells_inner(); igs++)
        {
            jl.load(parameters_.reciprocal_lattice()->gvec_shell_len(igs));

            for (int iat = 0; iat < parameters_.unit_cell()->num_atom_types(); iat++)
            {
                auto atom_type = parameters_.unit_cell()->atom_type(iat);
                Spline<double> s(atom_type->num_mt_points(), atom_type->radial_grid()); // not very efficient to create splines 
                                                                                        // for each G-shell, but we do this only once
                for (int ir = 0; ir < s.num_points(); ir++) 
                    s[ir] = jl(ir, 0, iat) * atom_type->uspp().total_charge_density[ir];
                rho_radial_integrals(iat, igs) = s.interpolate().integrate(0);
            }
        }

        for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        {
            int igs = parameters_.reciprocal_lattice()->gvec_shell<global>(ig);
            for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
            {
                int iat = parameters_.unit_cell()->atom(ia)->type_id();
                rho_->f_pw(ig) += conj(parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig, ia)) * 
                                  rho_radial_integrals(iat, igs) / parameters_.unit_cell()->omega(); // atomic density from UPF file is already multiplied by 4*PI
            }
        }
        if (fabs(rho_->f_pw(0) * parameters_.unit_cell()->omega() - parameters_.unit_cell()->num_valence_electrons()) > 1e-6)
            warning_local(__FILE__, __LINE__, "initial charge density is wrong");

        fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho_->f_pw(0));
        fft_->transform(1);
        fft_->output(&rho_->f_it<global>(0));

        for (int ir = 0; ir < fft_->size(); ir++)
        {
            if (rho_->f_it<global>(ir) < 0) rho_->f_it<global>(ir) = 0;
        }
        
        fft_->input(&rho_->f_it<global>(0));
        fft_->transform(-1);
        fft_->output(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &rho_->f_pw(0));
        
        //== mdarray<double, 3> rho_3d_map(&rho_it[0], fft_->size(0), fft_->size(1), fft_->size(2));
        //== int nx = fft_->size(0);
        //== int ny = fft_->size(1);
        //== int nz = fft_->size(2);

        //== auto p = parameters_.unit_cell()->unit_cell_parameters();

        //== FILE* fout = fopen("density.ted", "w");
        //== fprintf(fout, "%s\n", parameters_.unit_cell()->chemical_formula().c_str());
        //== fprintf(fout, "%16.10f %16.10f %16.10f  %16.10f %16.10f %16.10f\n", p.a, p.b, p.c, p.alpha, p.beta, p.gamma);
        //== fprintf(fout, "%i %i %i\n", nx + 1, ny + 1, nz + 1);
        //== for (int i0 = 0; i0 <= nx; i0++)
        //== {
        //==     for (int i1 = 0; i1 <= ny; i1++)
        //==     {
        //==         for (int i2 = 0; i2 <= nz; i2++)
        //==         {
        //==             fprintf(fout, "%14.8f\n", rho_3d_map(i0 % nx, i1 % ny, i2 % nz));
        //==         }
        //==     }
        //== }
        //== fclose(fout);
        //== stop_here

    }

    rho_->sync(true, true);
    for (int i = 0; i < parameters_.num_mag_dims(); i++) magnetization_[i]->sync(true, true);

    //if (type == 1)
    //{
    //    double rho_avg = parameters_.num_electrons() / parameters_.omega();
    //    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
    //    {
    //        int nmtp = parameters_.atom(ia)->num_mt_points();
    //        for (int ir = 0; ir < nmtp; ir++) rho_->f_mt<global>(0, ir, ia) = rho_avg / y00;
    //    }
    //    for (int i = 0; i < parameters_.fft().size(); i++) rho_->f_it<global>(i) = rho_avg;
    //}
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

void Density::add_kpoint_contribution_pp(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                         mdarray<complex16, 4>& pp_complex_density_matrix)
{
    if (occupied_bands.size() == 0) return;

    // take only occupied wave-functions
    mdarray<complex16, 2> wfs(kp->num_gkvec(), (int)occupied_bands.size());
    for (int i = 0; i < (int)occupied_bands.size(); i++)
    {
        memcpy(&wfs(0, i), &kp->spinor_wave_function(0, 0, occupied_bands[i].first), kp->num_gkvec() * sizeof(complex16));
    }

    mdarray<complex16, 2> beta_pw(kp->num_gkvec(), parameters_.unit_cell()->max_mt_basis_size());

    mdarray<complex16, 2> beta_psi(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());

    // auxiliary arrays
    mdarray<complex16, 2> bp1(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());
    mdarray<complex16, 2> bp2(parameters_.unit_cell()->max_mt_basis_size(), (int)occupied_bands.size());

    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {   
        // number of beta functions for a given atom
        int nbf = parameters_.unit_cell()->atom(ia)->type()->mt_basis_size();

        kp->generate_beta_pw(beta_pw, ia);
        
        // compute <beta|Psi>
        blas<cpu>::gemm(2, 0, nbf, (int)occupied_bands.size(), kp->num_gkvec(), &beta_pw(0, 0), beta_pw.ld(), 
                        &wfs(0, 0), wfs.ld(), &beta_psi(0, 0), beta_psi.ld());
        
        for (int i = 0; i < (int)occupied_bands.size(); i++)
        {
            for (int xi = 0; xi < nbf; xi++)
            {
                bp1(xi, i) = beta_psi(xi, i);
                bp2(xi, i) = conj(beta_psi(xi, i)) * occupied_bands[i].second;
            }
        }

        blas<cpu>::gemm(0, 1, nbf, nbf, (int)occupied_bands.size(), complex16(1, 0), &bp1(0, 0), bp1.ld(),
                        &bp2(0, 0), bp2.ld(), complex16(1, 0), &pp_complex_density_matrix(0, 0, 0, ia), pp_complex_density_matrix.ld());
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

    std::vector<complex16> f_pw(parameters_.reciprocal_lattice()->num_gvec(), complex16(0, 0));
    for (int ia = 0; ia < parameters_.unit_cell()->num_atoms(); ia++)
    {
        auto atom_type = parameters_.unit_cell()->atom(ia)->type();
        int iat = atom_type->id();
        int nbf = atom_type->mt_basis_size();

        for (int ig = 0; ig < parameters_.reciprocal_lattice()->num_gvec(); ig++)
        {
            complex16 z1 = parameters_.reciprocal_lattice()->gvec_phase_factor<global>(ig, ia);
            complex16 z2(0, 0);
            for (int xi2 = 0; xi2 < nbf; xi2++)
            {
                for (int xi1 = 0; xi1 <= xi2; xi1++)
                {
                    int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                    z2 += pp_complex_density_matrix(xi2, xi1, 0, ia) * 
                          conj(z1) * parameters_.reciprocal_lattice()->q_pw(ig, idx12, iat);

                    if (xi1 != xi2)
                    {
                        z2 += pp_complex_density_matrix(xi1, xi2, 0, ia) * 
                              z1 * conj(parameters_.reciprocal_lattice()->q_pw(ig, idx12, iat));
                    }
                }
            }
            f_pw[ig] += z2;
        }
    }
    fft_->input(parameters_.reciprocal_lattice()->num_gvec(), parameters_.reciprocal_lattice()->fft_index(), &f_pw[0]);
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

//void Density::add_band_contribution_mt(Band* band, double weight, mdarray<complex16, 3>& fylm, 
//                                       std::vector<Periodic_function<double, radial_angular>*>& dens)
//{
//    splindex<block> spl_num_atoms(parameters_.num_atoms(), band->num_ranks_row(), band->rank_row());
//
//    for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++)
//    {
//        int ia = spl_num_atoms[ialoc];
//        #pragma omp parallel for default(shared)
//        for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
//        {
//            for (int k = 0; k < gaunt12_.complex_gaunt_packed_L1_L2_size(lm3); k++)
//            {
//                int lm1 = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm1;
//                int lm2 = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm2;
//                complex16 cg = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).cg;
//
//                for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//                {
//                    dens[0]->f_rlm(ir, lm3, ia) += weight * real(cg * conj(fylm(ir, lm1, ia)) * fylm(ir, lm2, ia));
//                }
//            }
//        }
//    }
//}

//template<> void Density::generate_valence_density_mt_directly<cpu>(K_set& ks)
//{
//    Timer t("sirius::Density::generate_valence_density_mt_directly");
//    
//    int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
//    int lmmax = Utils::lmmax_by_lmax(lmax);
//    Band* band = ks.band();
//    
//    std::vector<Periodic_function<double, radial_angular>*> dens(1 + parameters_.num_mag_dims());
//    for (int i = 0; i < (int)dens.size(); i++)
//    {
//        dens[i] = new Periodic_function<double, radial_angular>(parameters_, parameters_.lmax_rho());
//        dens[i]->allocate(rlm_component);
//        dens[i]->zero();
//    }
//    
//    mdarray<complex16, 3> fylm(parameters_.max_num_mt_points(), lmmax, parameters_.num_atoms());
//
//    // add k-point contribution
//    for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
//    {
//        int ik = ks.spl_num_kpoints(ikloc);
//        for (int jloc = 0; jloc < band->spl_spinor_wf_col().local_size(); jloc++)
//        {
//            int j = band->spl_spinor_wf_col(jloc);
//
//            double wo = ks[ik]->band_occupancy(j) * ks[ik]->weight();
//
//            if (wo > 1e-14)
//            {
//                int ispn = 0;
//
//                ks[ik]->spinor_wave_function_component_mt<radial_angular>(band, lmax, ispn, jloc, fylm);
//                
//                add_band_contribution_mt(band, wo, fylm, dens);
//            }
//        }
//    }
//
//    for (int i = 0; i < (int)dens.size(); i++)
//    {
//        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//        {
//            Platform::allreduce(&dens[i]->f_rlm(0, 0, ia), 
//                                parameters_.lmmax_rho() * parameters_.max_num_mt_points(), 
//                                parameters_.mpi_grid().communicator());
//        }
//    }
//                                                        
//    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//    {
//        for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//        {
//            for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
//            {
//                rho_->f_rlm(lm, ir, ia) += dens[0]->f_rlm(ir, lm, ia);
//            }
//        }
//    }
//    
//    for (int i = 0; i < (int)dens.size(); i++) 
//    {
//        dens[i]->deallocate(rlm_component);
//        delete dens[i];
//    }
//}

//** void Density::generate_valence_density_mt_sht(K_set& ks)
//** {
//**     Timer t("sirius::Density::generate_valence_density_mt_sht");
//**     
//**     int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
//**     int lmmax = Utils::lmmax(lmax);
//**     
//**     SHT sht(parameters_.lmax_rho());
//** 
//**     mt_functions<complex16> psilm(Argument(arg_lm, lmmax), Argument(arg_radial, parameters_.max_num_mt_points()), 
//**                                   parameters_.num_atoms());
//**     MT_function<complex16> psitp(Argument(arg_tp, sht.num_points()), 
//**                                  Argument(arg_radial, parameters_.max_num_mt_points()));
//**     mt_functions<double> rhotp(Argument(arg_tp, sht.num_points()), 
//**                                Argument(arg_radial, parameters_.max_num_mt_points()), 
//**                                parameters_.num_atoms());
//**     rhotp.zero();
//**     MT_function<double> rholm(Argument(arg_lm, parameters_.lmmax_rho()), 
//**                               Argument(arg_radial, parameters_.max_num_mt_points()));
//** 
//**     
//**     // add k-point contribution
//**     for (int ikloc = 0; ikloc < ks.spl_num_kpoints().local_size(); ikloc++)
//**     {
//**         int ik = ks.spl_num_kpoints(ikloc);
//**         for (int jloc = 0; jloc < parameters_.spl_spinor_wf_col().local_size(); jloc++)
//**         {
//**             int j = parameters_.spl_spinor_wf_col(jloc);
//** 
//**             double wo = ks[ik]->band_occupancy(j) * ks[ik]->weight();
//** 
//**             if (wo > 1e-14)
//**             {
//**                 int ispn = 0;
//** 
//**                 ks[ik]->spinor_wave_function_component_mt(lmax, ispn, jloc, psilm);
//**                 for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**                 {
//**                     psilm(ia)->sh_transform(&sht, &psitp);
//** 
//**                     for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++) 
//**                     {
//**                         for (int itp = 0; itp < sht.num_points(); itp++) 
//**                             rhotp(itp, ir, ia) += wo * pow(abs(psitp(itp, ir)), 2);
//**                     }
//**                 }
//**             }
//**         }
//**     }
//**     
//**     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**     {
//**         rhotp(ia)->sh_transform(&sht, &rholm);
//**      
//**         for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**         {
//**             for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
//**             {
//**                 rho_->f_mt<global>(lm, ir, ia) = rholm(lm, ir);
//**             }
//**         }
//**     }
//** }


//** #ifdef _GPU_
//** template<> void Density::generate_valence_density_mt_directly<gpu>()
//** {
//**     Timer t("sirius::Density::generate_valence_density_mt_directly");
//**     
//**     int lmax = (basis_type == apwlo) ? parameters_.lmax_apw() : parameters_.lmax_pw();
//**     int lmmax = Utils::lmmax_by_lmax(lmax);
//** 
//**     // ==========================
//**     // prepare Gaunt coefficients
//**     // ==========================
//**     int max_num_gaunt = 0;
//**     for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
//**         max_num_gaunt = std::max(max_num_gaunt, gaunt12_.complex_gaunt_packed_L1_L2_size(lm3));
//**    
//**     mdarray<int, 1> gaunt12_size(parameters_.lmmax_rho());
//**     mdarray<int, 2> gaunt12_lm1_by_lm3(max_num_gaunt, parameters_.lmmax_rho());
//**     mdarray<int, 2> gaunt12_lm2_by_lm3(max_num_gaunt, parameters_.lmmax_rho());
//**     mdarray<complex16, 2> gaunt12_cg(max_num_gaunt, parameters_.lmmax_rho());
//** 
//**     for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
//**     {
//**         gaunt12_size(lm3) = gaunt12_.complex_gaunt_packed_L1_L2_size(lm3);
//**         for (int k = 0; k < gaunt12_.complex_gaunt_packed_L1_L2_size(lm3); k++)
//**         {
//**             gaunt12_lm1_by_lm3(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm1;
//**             gaunt12_lm2_by_lm3(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).lm2;
//**             gaunt12_cg(k, lm3) = gaunt12_.complex_gaunt_packed_L1_L2(lm3, k).cg;
//**         }
//**     }
//**     gaunt12_size.allocate_on_device();
//**     gaunt12_size.copy_to_device();
//**     gaunt12_lm1_by_lm3.allocate_on_device();
//**     gaunt12_lm1_by_lm3.copy_to_device();
//**     gaunt12_lm2_by_lm3.allocate_on_device();
//**     gaunt12_lm2_by_lm3.copy_to_device();
//**     gaunt12_cg.allocate_on_device();
//**     gaunt12_cg.copy_to_device();
//** 
//**     mdarray<double, 3> dens_mt(parameters_.max_num_mt_points(), parameters_.lmmax_rho(), parameters_.num_atoms());
//**     dens_mt.zero();
//**     dens_mt.allocate_on_device();
//**     dens_mt.zero_on_device();
//** 
//**     mdarray<complex16, 3> fylm(parameters_.max_num_mt_points(), lmmax, parameters_.num_atoms());
//**     fylm.pin_memory();
//**     fylm.allocate_on_device();
//**     
//**     splindex<block> spl_num_atoms(parameters_.num_atoms(), band_->num_ranks_row(), band_->rank_row());
//**     
//**     mdarray<int, 1> iat_by_ia(parameters_.num_atoms());
//**     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**         iat_by_ia(ia) = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());
//**     iat_by_ia.allocate_on_device();
//**     iat_by_ia.copy_to_device();
//** 
//**     mdarray<int, 1> nmtp_by_iat(parameters_.num_atom_types());
//**     for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
//**         nmtp_by_iat(iat) = parameters_.atom_type(iat)->num_mt_points();
//**     nmtp_by_iat.allocate_on_device();
//**     nmtp_by_iat.copy_to_device();
//** 
//**     mdarray<int, 1> ia_by_ialoc(spl_num_atoms.local_size());
//**     for (int ialoc = 0; ialoc < spl_num_atoms.local_size(); ialoc++)
//**     {
//**         int ia = spl_num_atoms[ialoc];
//**         ia_by_ialoc(ialoc) = ia;
//**     }
//**     ia_by_ialoc.allocate_on_device();
//**     ia_by_ialoc.copy_to_device();
//**     
//**     // add k-point contribution
//**     for (int ikloc = 0; ikloc < kpoint_set_.spl_num_kpoints().local_size(); ikloc++)
//**     {
//**         int ik = kpoint_set_.spl_num_kpoints(ikloc);
//**         for (int jloc = 0; jloc < band_->spl_spinor_wf_col().local_size(); jloc++)
//**         {
//**             int j = band_->spl_spinor_wf_col(jloc);
//** 
//**             double wo = kpoint_set_[ik]->band_occupancy(j) * kpoint_set_[ik]->weight();
//** 
//**             if (wo > 1e-14)
//**             {
//**                 int ispn = 0;
//**                 kpoint_set_[ik]->spinor_wave_function_component_mt<radial_angular>(band_, lmax, ispn, jloc, fylm);
//**                 fylm.copy_to_device();
//**                 
//**                 add_band_density_gpu(parameters_.lmmax_rho(), lmmax, parameters_.max_num_mt_points(), 
//**                                      spl_num_atoms.local_size(), ia_by_ialoc.get_ptr_device(), iat_by_ia.get_ptr_device(),
//**                                      nmtp_by_iat.get_ptr_device(), max_num_gaunt, 
//**                                      gaunt12_size.get_ptr_device(), gaunt12_lm1_by_lm3.get_ptr_device(), 
//**                                      gaunt12_lm2_by_lm3.get_ptr_device(), gaunt12_cg.get_ptr_device(), 
//**                                      fylm.get_ptr_device(), wo, dens_mt.get_ptr_device());
//**             }
//**         }
//**     }
//**     dens_mt.copy_to_host();
//** 
//**     //for (int i = 0; i < (int)dens.size(); i++)
//**     //{
//**         for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**         {
//**             Platform::allreduce(&dens_mt(0, 0, ia), parameters_.lmmax_rho() * parameters_.max_num_mt_points(), 
//**                                 parameters_.mpi_grid().communicator());
//**         }
//**     //}
//** 
//**     for (int ia = 0; ia < parameters_.num_atoms(); ia++)
//**     {
//**         for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
//**         {
//**             for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
//**             {
//**                 rho_->f_rlm(lm, ir, ia) += dens_mt(ir, lm, ia);
//**             }
//**         }
//**     }
//** }
//** #endif

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
            //generate_valence_density_mt_sht(ks);
            break;
        }
        case pwlo:
        {
            switch (parameters_.processing_unit())
            {
                case cpu:
                {
                    //** generate_valence_density_mt_directly<cpu>(ks);
                    break;
                }
                #ifdef _GPU_
                case gpu:
                {
                    //** generate_valence_density_mt_directly<gpu>(ks);
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

