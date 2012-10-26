namespace sirius
{

class Density
{
    private:

        Global& parameters_;
        
        PeriodicFunction<double>* rho_;
        
        PeriodicFunction<double>* magnetization_[3];
        
        std::vector< std::pair<int,int> > dmat_spins_;

        mdarray<complex16,3> complex_gaunt_;

        kpoint_set kpoint_set_;

        template <int num_mag_dims> 
        void reduce_zdens(int ia, mdarray<complex16,3>& zdens, mdarray<double,5>& mt_density_matrix)
        {
            AtomType* type = parameters_.atom(ia)->type();
            int mt_basis_size = type->mt_basis_size();

            for (int lm3 = 0; lm3 < parameters_.lmmax_rho(); lm3++)
            {
                int l3 = l_by_lm(lm3);
                
                for (int j2 = 0; j2 < mt_basis_size; j2++)
                {
                    int l2 = type->indexb(j2).l;
                    int lm2 = type->indexb(j2).lm;
                    int idxrf2 = type->indexb(j2).idxrf;
        
                    int j1 = 0;

                    // compute only upper triangular block and later use the symmetry properties of the density matrix
                    for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                    {
                        int l1 = type->indexr(idxrf1).l;
                        
                        if ((l1 + l2 + l3) % 2 == 0)
                        {
                            for (int lm1 = lm_by_l_m(l1, -l1); lm1 <= lm_by_l_m(l1, l1); lm1++, j1++) 
                            {
                                complex16 gc = complex_gaunt_(lm1, lm2, lm3);

                                switch(num_mag_dims)
                                {
                                    case 3:
                                        mt_density_matrix(idxrf1, idxrf2, lm3, ia, 2) += 2.0 * real(zdens(j1, j2, 2) * gc); 
                                        mt_density_matrix(idxrf1, idxrf2, lm3, ia, 3) -= 2.0 * imag(zdens(j1, j2, 2) * gc);
                                    case 1:
                                        mt_density_matrix(idxrf1, idxrf2, lm3, ia, 1) += real(zdens(j1, j2, 1) * gc);
                                    case 0:
                                        mt_density_matrix(idxrf1, idxrf2, lm3, ia, 0) += real(zdens(j1, j2, 0) * gc);
                                }
                            }
                        } 
                        else
                            j1 += (2 * l1 + 1);
                    }
                } // j2
            } // lm3
        }

        void add_k_contribution(kpoint* kp, mdarray<double,5>& mt_density_matrix)
        {
            Timer t("sirius::Density::add_k_contribution");
            
            std::vector< std::pair<int,double> > bands;
            for (int j = 0; j < parameters_.num_bands(); j++)
            {
                double wo = kp->band_occupancy(j) * kp->weight();
                if (wo > 1e-14)
                    bands.push_back(std::pair<int,double>(j, wo));
            }
           
            // if we have ud and du spin blocks, don't compute one of them (du in this implementation)
            // because density matrix is symmetric
            int num_zdmat = (parameters_.num_mag_dims() == 3) ? 3 : (parameters_.num_mag_dims() + 1);

            mdarray<complex16,3> zdens(parameters_.max_mt_basis_size(), parameters_.max_mt_basis_size(), num_zdmat);
            mdarray<complex16,3> wf1(parameters_.max_mt_basis_size(), bands.size(), parameters_.num_spins());
            mdarray<complex16,3> wf2(parameters_.max_mt_basis_size(), bands.size(), parameters_.num_spins());
       
            Timer t1("sirius::Density::add_k_contribution:zdens", false);
            Timer t2("sirius::Density::add_k_contribution:reduce_zdens", false);
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                t1.start();
                
                int offset_wf = parameters_.atom(ia)->offset_wf();
                int mt_basis_size = parameters_.atom(ia)->type()->mt_basis_size();
                
                for (int i = 0; i < (int)bands.size(); i++)
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        memcpy(&wf1(0, i, ispn), &kp->spinor_wave_function(offset_wf, ispn, bands[i].first), 
                               mt_basis_size * sizeof(complex16));
                        for (int j = 0; j < mt_basis_size; j++) 
                            wf2(j, i, ispn) = wf1(j, i, ispn) * bands[i].second;
                    }

                for (int j = 0; j < num_zdmat; j++)
                    gemm<cpu>(0, 2, mt_basis_size, mt_basis_size, bands.size(), complex16(1.0, 0.0), 
                              &wf1(0, 0, dmat_spins_[j].first), parameters_.max_mt_basis_size(), 
                              &wf2(0, 0, dmat_spins_[j].second), parameters_.max_mt_basis_size(),complex16(0.0, 0.0), 
                              &zdens(0, 0, j), parameters_.max_mt_basis_size());
                
                t1.stop();

                t2.start();
                
                switch(parameters_.num_mag_dims())
                {
                    case 3:
                        reduce_zdens<3>(ia, zdens, mt_density_matrix);
                        break;
                    case 1:
                        reduce_zdens<1>(ia, zdens, mt_density_matrix);
                        break;
                    case 0:
                        reduce_zdens<0>(ia, zdens, mt_density_matrix);
                        break;
                }
                
                t2.stop();

            } // ia
        
            
            Timer t3("sirius::Density::add_k_contribution:it");
            
            #pragma omp parallel default(shared)
            {
                int thread_id = omp_get_thread_num();

                mdarray<double,2> it_density(parameters_.fft().size(), parameters_.num_mag_dims() + 1);
                it_density.zero();
                
                mdarray<complex16,2> wfit(parameters_.fft().size(), parameters_.num_spins());

                #pragma omp for
                for (int i = 0; i < (int)bands.size(); i++)
                {
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        parameters_.fft().input(kp->num_gkvec(), kp->fft_index(), 
                                           &kp->spinor_wave_function(parameters_.mt_basis_size(), ispn, bands[i].first),
                                           thread_id);
                        parameters_.fft().transform(1, thread_id);
                        parameters_.fft().output(&wfit(0, ispn), thread_id);
                    }
                    
                    double w = bands[i].second / parameters_.omega();
                    
                    switch(parameters_.num_mag_dims())
                    {
                        case 3:
                            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            {
                                complex16 z = wfit(ir, 0) * conj(wfit(ir, 1)) * w;
                                it_density(ir, 2) += 2.0 * real(z);
                                it_density(ir, 3) -= 2.0 * imag(z);
                            }
                        case 1:
                            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                                it_density(ir, 1) += real(wfit(ir, 1) * conj(wfit(ir, 1))) * w;
                        case 0:
                            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                                it_density(ir, 0) += real(wfit(ir, 0) * conj(wfit(ir, 0))) * w;
                    }
                }
                switch(parameters_.num_mag_dims())
                {
                    case 3:
                        #pragma omp critical
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        {
                            magnetization_[1]->f_it(ir) += it_density(ir, 2);
                            magnetization_[2]->f_it(ir) += it_density(ir, 3);
                        }
                    case 1:
                        #pragma omp critical
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        {
                            rho_->f_it(ir) += (it_density(ir, 0) + it_density(ir, 1));
                            magnetization_[0]->f_it(ir) += (it_density(ir, 0) - it_density(ir, 1));
                        }
                        break;
                    case 0:
                        #pragma omp critical
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            rho_->f_it(ir) += it_density(ir, 0);
                }
            }
            
            t3.stop();
       }

    public:

        Density(Global& parameters__) : parameters_(parameters__)
        {
            initialize();
        }
        
        void set_charge_density_ptr(double* rhomt, double* rhoir)
        {
            rho_->set_rlm_ptr(rhomt);
            rho_->set_it_ptr(rhoir);
        }
        
        void set_magnetization_ptr(double* magmt, double* magir)
        {
            assert(parameters_.num_spins() == 2);

            // set temporary array wrapper
            mdarray<double,4> magmt_tmp(magmt, parameters_.lmmax_rho(), parameters_.max_num_mt_points(), parameters_.num_atoms(),
                                        parameters_.num_mag_dims());
            mdarray<double,2> magir_tmp(magir, parameters_.fft().size(), parameters_.num_mag_dims());
            
            if (parameters_.num_mag_dims() == 1)
            {
                // z component is the first and only one
                magnetization_[0]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 0));
                magnetization_[0]->set_it_ptr(&magir_tmp(0, 0));
            }

            if (parameters_.num_mag_dims() == 3)
            {
                // z component is the first
                magnetization_[0]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 2));
                magnetization_[0]->set_it_ptr(&magir_tmp(0, 2));
                // x component is the second
                magnetization_[1]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 0));
                magnetization_[1]->set_it_ptr(&magir_tmp(0, 0));
                // y component is the third
                magnetization_[2]->set_rlm_ptr(&magmt_tmp(0, 0, 0, 1));
                magnetization_[2]->set_it_ptr(&magir_tmp(0, 1));
            }
        }
    
        void initialize()
        {
            kpoint_set_.clear();

            rho_ = new PeriodicFunction<double>(parameters_, parameters_.lmax_rho());
            rho_->allocate(pw_component);

            for (int i = 0; i < parameters_.num_mag_dims(); i++)
            {
                magnetization_[i] = new PeriodicFunction<double>(parameters_, parameters_.lmax_rho());
                magnetization_[i]->allocate(pw_component);
            }

            dmat_spins_.clear();
            dmat_spins_.push_back(std::pair<int,int>(0, 0));
            dmat_spins_.push_back(std::pair<int,int>(1, 1));
            dmat_spins_.push_back(std::pair<int,int>(0, 1));
            dmat_spins_.push_back(std::pair<int,int>(1, 0));
            
            complex_gaunt_.set_dimensions(parameters_.lmmax_apw(), parameters_.lmmax_apw(), parameters_.lmmax_rho());
            complex_gaunt_.allocate();

            for (int l1 = 0; l1 <= parameters_.lmax_apw(); l1++) 
            for (int m1 = -l1; m1 <= l1; m1++)
            {
                int lm1 = lm_by_l_m(l1, m1);
                for (int l2 = 0; l2 <= parameters_.lmax_apw(); l2++)
                for (int m2 = -l2; m2 <= l2; m2++)
                {
                    int lm2 = lm_by_l_m(l2, m2);
                    for (int l3 = 0; l3 <= parameters_.lmax_pot(); l3++)
                    for (int m3 = -l3; m3 <= l3; m3++)
                    {
                        int lm3 = lm_by_l_m(l3, m3);
                        complex_gaunt_(lm1, lm2, lm3) = SHT::complex_gaunt(l1, l3, l2, m1, m3, m2);
                    }
                }
            }
        }
        
        void zero()
        {
            rho_->zero();
            for (int i = 0; i < parameters_.num_mag_dims(); i++)
                magnetization_[i]->zero();
        }
      
        void initial_density()
        {
            zero();
            
            std::vector<double> enu;
            for (int i = 0; i < parameters_.num_atom_types(); i++)
                parameters_.atom_type(i)->solve_free_atom(1e-8, 1e-5, 1e-4, enu);

            zero();
            
            double mt_charge = 0.0;
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                double* v = parameters_.atom(ia)->vector_field();
                double len = vector_length(v);

                int nmtp = parameters_.atom(ia)->type()->num_mt_points();
                Spline<double> rho(nmtp, parameters_.atom(ia)->type()->radial_grid());
                for (int ir = 0; ir < nmtp; ir++)
                {
                    rho[ir] = parameters_.atom(ia)->type()->free_atom_density(ir);
                    rho_->f_rlm(0, ir, ia) = rho[ir] / y00;
                }
                rho.interpolate();

                // add charge of the MT sphere
                mt_charge += fourpi * rho.integrate(nmtp - 1, 2);

                if (len > 1e-8)
                {
                    if (parameters_.num_mag_dims())
                        for (int ir = 0; ir < nmtp; ir++)
                            magnetization_[0]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[2] / len;

                    if (parameters_.num_mag_dims() == 3)
                    {
                        for (int ir = 0; ir < nmtp; ir++)
                            magnetization_[1]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[0] / len;
                        for (int ir = 0; ir < nmtp; ir++)
                            magnetization_[2]->f_rlm(0, ir, ia) = 0.2 * rho_->f_rlm(0, ir, ia) * v[1] / len;
                    }
                }
            }
            
            // distribute remaining charge
            for (int i = 0; i < parameters_.fft().size(); i++)
                rho_->f_it(i) = (parameters_.num_electrons() - mt_charge) / parameters_.volume_it();
        }

        void integrate()
        {
            printf("\n");
            printf("Charges and magnetic moments\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n"); 

            double total_charge;
            double it_charge;
            std::vector<double> mt_charge;

            double total_magnetization[3];
            double it_magnetization[3];
            std::vector<double> mt_magnetization[3];

            total_charge = rho_->integrate(rlm_component | it_component, mt_charge, it_charge);
            for (int j = 0; j < parameters_.num_mag_dims(); j++)
                total_magnetization[j] = magnetization_[j]->integrate(rlm_component | it_component, 
                                                                      mt_magnetization[j], it_magnetization[j]);
            printf("muffin-tin\n");
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                printf("atom : %4i   charge : %8.4f", ia, mt_charge[ia]);
                if (parameters_.num_mag_dims())
                {
                    double v[] = {0, 0, 0};
                    v[2] = mt_magnetization[0][ia];
                    if (parameters_.num_mag_dims() == 3)
                    {
                        v[0] = mt_magnetization[1][ia];
                        v[1] = mt_magnetization[2][ia];
                    }
                    printf("   moment : (%8.4f %8.4f %8.4f)   |moment| : %8.4f", v[0], v[1], v[2], vector_length(v));
                }
                printf("\n");
            }
            
            printf("interstitial\n");
            printf("              charge : %8.4f", it_charge);
            if (parameters_.num_mag_dims())
            {
                double v[] = {0, 0, 0};
                v[2] = it_magnetization[0];
                if (parameters_.num_mag_dims() == 3)
                {
                    v[0] = it_magnetization[1];
                    v[1] = it_magnetization[2];
                }
                printf("   moment : (%8.4f %8.4f %8.4f)   |moment| : %8.4f", v[0], v[1], v[2], vector_length(v));
            }
            printf("\n");
            
            printf("total\n");
            printf("              charge : %8.4f", total_charge);
            if (parameters_.num_mag_dims())
            {
                double v[] = {0, 0, 0};
                v[2] = total_magnetization[0];
                if (parameters_.num_mag_dims() == 3)
                {
                    v[0] = total_magnetization[1];
                    v[1] = total_magnetization[2];
                }
                printf("   moment : (%8.4f %8.4f %8.4f)   |moment| : %8.4f", v[0], v[1], v[2], vector_length(v));
            }
            printf("\n");
        }

        void generate()
        {
            Timer t("sirius::Density::generate");
            
            double wt = 0.0;
            double ot = 0.0;
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
            {
                wt += kpoint_set_[ik]->weight();
                for (int j = 0; j < parameters_.num_bands(); j++)
                    ot += kpoint_set_[ik]->weight() * kpoint_set_[ik]->band_occupancy(j);
            }

            if (fabs(wt - 1.0) > 1e-12)
                error(__FILE__, __LINE__, "kpoint weights don't sum to one");

            if (fabs(ot - parameters_.num_valence_electrons()) > 1e-4)
            {
                std::stringstream s;
                s << "wrong occupancies" << std::endl
                  << "  computed : " << ot << std::endl
                  << "  required : " << parameters_.num_valence_electrons() << std::endl
                  << " difference : " << fabs(ot - parameters_.num_valence_electrons());
                error(__FILE__, __LINE__, s);
            }

            // generate radial functions
            potential->set_spherical_potential();
            parameters_.generate_radial_functions();
            
            // generate radial integrals
            potential->set_nonspherical_potential();
            parameters_.generate_radial_integrals();

            // generate plane-wave coefficients of the potential in the interstitial region
            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                 potential->effective_potential()->f_it(ir) *= parameters_.step_function(ir);

            parameters_.fft().input(potential->effective_potential()->f_it());
            parameters_.fft().transform(-1);
            parameters_.fft().output(parameters_.num_gvec(), parameters_.fft_index(), 
                                     potential->effective_potential()->f_pw());

            // auxiliary density matrix
            mdarray<double,5> mt_density_matrix(parameters_.max_mt_radial_basis_size(), 
                                                parameters_.max_mt_radial_basis_size(), 
                                                parameters_.lmmax_rho(), parameters_.num_atoms(), parameters_.num_mag_dims() + 1);
            mt_density_matrix.zero();
            
            // zero density and magnetization
            zero();

            //
            // Main loop over k-points
            //
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
            {
                // solve secular equatiion and generate wave functions
                kpoint_set_[ik]->find_eigen_states(band, potential->effective_potential(),
                                                   potential->effective_magnetic_field());
                // add to charge density and magnetization
                add_k_contribution(kpoint_set_[ik], mt_density_matrix);
            }

            Timer t1("sirius::Density::generate:convert_mt");
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                int nmtp = parameters_.atom(ia)->type()->num_mt_points();
                mdarray<double,2> v(nmtp, parameters_.num_mag_dims() + 1);

                for (int lm = 0; lm < parameters_.lmmax_rho(); lm++)
                {
                    for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
                        for (int idxrf = 0; idxrf < parameters_.atom(ia)->type()->mt_radial_basis_size(); idxrf++)
                            mt_density_matrix(idxrf, idxrf, lm, ia, j) *= 0.5; 

                    v.zero();

                    for (int j = 0; j < parameters_.num_mag_dims() + 1; j++)
                        for (int idxrf2 = 0; idxrf2 < parameters_.atom(ia)->type()->mt_radial_basis_size(); idxrf2++)
                        {
                            for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++)
                            {
                                for (int ir = 0; ir < parameters_.atom(ia)->type()->num_mt_points(); ir++)
                                    v(ir, j) += 2 * mt_density_matrix(idxrf1, idxrf2, lm, ia, j) * 
                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf1) * 
                                                parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf2);
                            }
                        }

                    switch(parameters_.num_mag_dims())
                    {
                        case 3:
                            for (int ir = 0; ir < nmtp; ir++)
                            {
                                magnetization_[1]->f_rlm(lm, ir, ia) = v(ir, 2);
                                magnetization_[2]->f_rlm(lm, ir, ia) = v(ir, 3);
                            }
                        case 1:
                            for (int ir = 0; ir < nmtp; ir++)
                            {
                                rho_->f_rlm(lm, ir, ia) = v(ir, 0) + v(ir, 1);
                                magnetization_[0]->f_rlm(lm, ir, ia) = v(ir, 0) - v(ir, 1);
                            }
                            break;
                        case 0:
                            for (int ir = 0; ir < nmtp; ir++)
                                rho_->f_rlm(lm, ir, ia) = v(ir, 0);
                    }
                }
            }
            t1.stop();
            
            // add core contribution
            for (int ic = 0; ic < parameters_.num_atom_symmetry_classes(); ic++)
            {
                int nmtp = parameters_.atom_symmetry_class(ic)->atom_type()->num_mt_points();
                parameters_.atom_symmetry_class(ic)->generate_core_charge_density();
                for (int i = 0; i < parameters_.atom_symmetry_class(ic)->num_atoms(); i++)
                {
                    int ia = parameters_.atom_symmetry_class(ic)->atom_id(i);
                    for (int ir = 0; ir < nmtp; ir++)
                        rho_->f_rlm(0, ir, ia) += parameters_.atom_symmetry_class(ic)->core_charge_density(ir) / y00;
                }
            }

            parameters_.rti().total_charge_ibz = rho_->integrate(rlm_component | it_component);

            // compute eigen-value sums
            double eval_sum = 0.0;
            for (int ic = 0; ic < parameters_.num_atom_symmetry_classes(); ic++)
                eval_sum += parameters_.atom_symmetry_class(ic)->core_eval_sum() *
                            parameters_.atom_symmetry_class(ic)->num_atoms();
            
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
            {
                double wk = kpoint_set_[ik]->weight();
                for (int j = 0; j < parameters_.num_bands(); j++)
                    eval_sum += wk * kpoint_set_[ik]->band_energy(j) * kpoint_set_[ik]->band_occupancy(j);
            }
            
            parameters_.rti().eval_sum = eval_sum;
        }

        void add_kpoint(int kpoint_id, double* vk, double weight)
        {
            kpoint_set_.add_kpoint(kpoint_id, vk, weight, parameters_);
        }

        void set_band_occupancies(int kpoint_id, double* band_occupancies)
        {
            kpoint_set_.kpoint_by_id(kpoint_id)->set_band_occupancies(band_occupancies);
        }

        void get_band_energies(int kpoint_id, double* band_energies)
        {
            kpoint_set_.kpoint_by_id(kpoint_id)->get_band_energies(band_energies);
        }

        void print_info()
        {
            printf("\n");
            printf("Density\n");
            for (int i = 0; i < 80; i++) printf("-");
            printf("\n");
            printf("number of k-points : %i\n", kpoint_set_.num_kpoints());
            for (int ik = 0; ik < kpoint_set_.num_kpoints(); ik++)
                printf("ik=%4i    vk=%12.6f %12.6f %12.6f    weight=%12.6f   num_gkvec=%6i\n", 
                       ik, kpoint_set_[ik]->vk()[0], kpoint_set_[ik]->vk()[1], kpoint_set_[ik]->vk()[2], 
                       kpoint_set_[ik]->weight(), kpoint_set_[ik]->num_gkvec());
        }

        PeriodicFunction<double>* rho()
        {
            return rho_;
        }
        
        PeriodicFunction<double>** magnetization()
        {
            return magnetization_;
        }

        PeriodicFunction<double>* magnetization(int i)
        {
            return magnetization_[i];
        }
};

Density* density;

};
