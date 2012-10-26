namespace sirius
{

class kpoint
{
    private:

        Global& parameters_;

        mdarray<double,2> gkvec_;

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        mdarray<complex16,2> matching_coefficients_;

        std::vector<double> evalfv_;

        mdarray<complex16,2> evecfv_;
        
        mdarray<complex16,2> evecsv_;

        std::vector<int> fft_index_;
        
        mdarray<complex16,2> scalar_wave_functions_;

        mdarray<complex16,3> spinor_wave_functions_;

        std::vector<double> band_occupancies_;

        std::vector<double> band_energies_; 

        double weight_;

        double vk_[3];

        void generate_matching_coefficients()
        {
            Timer t("sirius::kpoint_data_set::generate_matching_coefficients");

            std::vector<complex16> ylmgk(parameters_.lmmax_apw());
            
            mdarray<double,2> jl(parameters_.lmax_apw() + 2, 2);
            
            std::vector<complex16> zil(parameters_.lmax_apw() + 1);
            for (int l = 0; l <= parameters_.lmax_apw(); l++)
                zil[l] = pow(complex16(0.0, 1.0), l);
      
            matching_coefficients_.set_dimensions(num_gkvec(), parameters_.mt_aw_basis_size());
            matching_coefficients_.allocate();

            // TODO: check if spherical harmonic generation is slowing things: then it can be cached
            // TODO: number of Bessel functions can be considerably decreased (G+k shells, atom types)
            // TODO[?]: G+k shells
            for (int ig = 0; ig < num_gkvec(); ig++)
            {
                double v[3];
                double vs[3];
                parameters_.get_coordinates<cartesian,reciprocal>(gkvec(ig), v);
                SHT::spherical_coordinates(v, vs);
                SHT::spherical_harmonics(parameters_.lmax_apw(), vs[1], vs[2], &ylmgk[0]);

                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    complex16 phase_factor = exp(complex16(0.0, twopi * scalar_product(gkvec(ig), 
                                                                                       parameters_.atom(ia)->position())));

                    assert(parameters_.atom(ia)->type()->max_aw_order() <= 2);

                    double R = parameters_.atom(ia)->type()->mt_radius();
                    double gkR = vs[0] * R; // |G+k|*R

                    // generate spherical Bessel functions
                    gsl_sf_bessel_jl_array(parameters_.lmax_apw() + 1, gkR, &jl(0, 0));
                    // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                        jl(l, 1) = -jl(l + 1, 0) + (l / R) * jl(l, 0);

                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    {
                        int num_aw = parameters_.atom(ia)->type()->aw_descriptor(l).size();

                        complex16 a[2][2];
                        mdarray<complex16,2> b(2, 2 * parameters_.lmax_apw() + 1);

                        for (int order = 0; order < num_aw; order++)
                            for (int order1 = 0; order1 < num_aw; order1++)
                                a[order][order1] = complex16(parameters_.atom(ia)->symmetry_class()->
                                    aw_surface_dm(l, order, order1), 0.0);
                        
                        for (int m = -l; m <= l; m++)
                            for (int order = 0; order < num_aw; order++)
                                b(order, m + l) = (fourpi / sqrt(parameters_.omega())) * zil[l] * jl(l, order) * 
                                    phase_factor * conj(ylmgk[lm_by_l_m(l, m)]) * pow(gkR, order); 
                        
                        double det = (num_aw == 1) ? abs(a[0][0]) : abs(a[0][0] * a[1][1] - a[0][1] * a [1][0]);
                        if (det  < 1e-8)
                            error(__FILE__, __LINE__, "ill defined linear equation problem");
                          
                        int info = gesv<complex16>(num_aw, 2 * l + 1, &a[0][0], 2, &b(0, 0), 2);

                        if (info)
                        {
                            std::stringstream s;
                            s << "gtsv returned " << info;
                            error(__FILE__, __LINE__, s);
                        }

                        for (int order = 0; order < num_aw; order++)
                            for (int m = -l; m <= l; m++)
                                matching_coefficients_(ig, parameters_.atom(ia)->offset_aw() + 
                                                           parameters_.atom(ia)->type()->indexb_by_l_m_order(l, m, order)) = 
                                    conj(b(order, l + m)); // it is more convenient to store conjugated coefficients
                                                           // because then the overlap matrix is set with single
                                                           // matrix-matrix multiplication without further conjugation 
                    }
                }
            }
        }
        
        inline void move_apw_blocks(complex16 *wf)
        {
            for (int ia = parameters_.num_atoms() - 1; ia > 0; ia--)
            {
                int final_block_offset = parameters_.atom(ia)->offset_wf();
                int initial_block_offset = parameters_.atom(ia)->offset_aw();
                int block_size = parameters_.atom(ia)->type()->mt_aw_basis_size();
        
                memmove(&wf[final_block_offset], &wf[initial_block_offset], block_size * sizeof(complex16));
            }
        }
        
        inline void copy_lo_blocks(complex16 *wf, complex16 *evec)
        {
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                int final_block_offset = parameters_.atom(ia)->offset_wf() + parameters_.atom(ia)->type()->mt_aw_basis_size();
                int initial_block_offset = parameters_.atom(ia)->offset_lo();
                int block_size = parameters_.atom(ia)->type()->mt_lo_basis_size();
                
                if (block_size > 0)
                    memcpy(&wf[final_block_offset], &evec[initial_block_offset], block_size * sizeof(complex16));
            }
        }
        
        inline void copy_pw_block(int ngk, complex16 *wf, complex16 *evec)
        {
            memcpy(wf, evec, ngk * sizeof(complex16));
        }

        void generate_scalar_wave_functions()
        {
            Timer t("sirius::kpoint_data_set::generate_scalar_wave_functions");
            
            scalar_wave_functions_.set_dimensions(scalar_wf_size(), parameters_.num_fv_states());
            scalar_wave_functions_.allocate();
            
            gemm<cpu>(2, 0, parameters_.mt_aw_basis_size(), parameters_.num_fv_states(), num_gkvec(), complex16(1.0, 0.0), 
                &matching_coefficients_(0, 0), num_gkvec(), &evecfv_(0, 0), fv_basis_size(), 
                complex16(0.0, 0.0), &scalar_wave_functions_(0, 0), scalar_wf_size());
            
            for (int j = 0; j < parameters_.num_fv_states(); j++)
            {
                move_apw_blocks(&scalar_wave_functions_(0, j));
        
                if (parameters_.mt_lo_basis_size() > 0) 
                    copy_lo_blocks(&scalar_wave_functions_(0, j), &evecfv_(num_gkvec(), j));
        
                copy_pw_block(num_gkvec(), &scalar_wave_functions_(parameters_.mt_basis_size(), j), &evecfv_(0, j));
            }
        }

        void generate_spinor_wave_functions(int flag)
        {
            Timer t("sirius::kpoint_data_set::generate_spinor_wave_functions");

            spinor_wave_functions_.set_dimensions(scalar_wf_size(), parameters_.num_spins(), parameters_.num_bands());
            
            if (flag == -1)
            {
                spinor_wave_functions_.set_ptr(scalar_wave_functions_.get_ptr());
                memcpy(&band_energies_[0], &evalfv_[0], parameters_.num_bands() * sizeof(double));
                return;
            }

            spinor_wave_functions_.allocate();
            
            for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                gemm<cpu>(0, 0, scalar_wf_size(),  parameters_.num_bands(), parameters_.num_fv_states(), complex16(1.0, 0.0), 
                          &scalar_wave_functions_(0, 0), scalar_wf_size(), &evecsv_(ispn * parameters_.num_fv_states(), 0), 
                          parameters_.num_bands(), complex16(0.0, 0.0), &spinor_wave_functions_(0, ispn, 0), 
                          scalar_wf_size() * parameters_.num_spins());
        }

    public:

        kpoint(Global& parameters__, double* vk__, double weight__) : parameters_(parameters__), weight_(weight__)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];

            initialize();
        }

        void initialize()
        {
            if (parameters_.aw_cutoff() > double(parameters_.lmax_apw()))
                error(__FILE__, __LINE__, "aw cutoff is too large for a given lmax");

            double gk_cutoff = parameters_.aw_cutoff() / parameters_.min_mt_radius();
            
            if (gk_cutoff * 2 > parameters_.pw_cutoff())
                error(__FILE__, __LINE__, "aw cutoff is too large for a given plane-wave cutoff");

            std::vector< std::pair<double,int> > gkmap;

            // find G-vectors for which |G+k| < cutoff
            for (int ig = 0; ig < parameters_.num_gvec(); ig++)
            {
                double vgk[3];
                for (int x = 0; x < 3; x++)
                    vgk[x] = parameters_.gvec(ig)[x] + vk_[x];

                double v[3];
                parameters_.get_coordinates<cartesian,reciprocal>(vgk, v);
                double gklen = vector_length(v);

                if (gklen <= gk_cutoff) gkmap.push_back(std::pair<double,int>(gklen, ig));
            }

            std::sort(gkmap.begin(), gkmap.end());

            gkvec_.set_dimensions(3, gkmap.size());
            gkvec_.allocate();

            gvec_index_.resize(gkmap.size());

            for (int ig = 0; ig < (int)gkmap.size(); ig++)
            {
                gvec_index_[ig] = gkmap[ig].second;
                for (int x = 0; x < 3; x++)
                    gkvec_(x, ig) = parameters_.gvec(gkmap[ig].second)[x] + vk_[x];
            }

            fft_index_.resize(num_gkvec());
            for (int ig = 0; ig < num_gkvec(); ig++)
                fft_index_[ig] = parameters_.fft_index(gvec_index_[ig]);
        }

        void find_eigen_states(Band* band, PeriodicFunction<double>* effective_potential, 
                               PeriodicFunction<double>* effective_magnetic_field[3])
        {
            assert(fv_basis_size() > parameters_.num_fv_states());
            
            Timer t("sirius::kpoint::find_eigen_states");
            
            evalfv_.resize(parameters_.num_fv_states());
            evecfv_.set_dimensions(fv_basis_size(), parameters_.num_fv_states());
            evecfv_.allocate();

            generate_matching_coefficients();

            band->solve_fv(parameters_, fv_basis_size(), num_gkvec(), &gvec_index_[0], gkvec_, 
                           matching_coefficients_, effective_potential, effective_magnetic_field, evecfv_,
                           &evalfv_[0]);

            generate_scalar_wave_functions();
            
            /*for (int i = 0; i < 3; i++)
                test_scalar_wave_functions(i);*/

            evecsv_.set_dimensions(parameters_.num_bands(), parameters_.num_bands());
            evecsv_.allocate();
            band_energies_.resize(parameters_.num_bands());
            
            if (parameters_.num_spins() == 2)
            {
                band->solve_sv(parameters_, scalar_wf_size(), num_gkvec(), fft_index(), &evalfv_[0], 
                               scalar_wave_functions_, effective_magnetic_field, &band_energies_[0],
                               evecsv_);

                generate_spinor_wave_functions(1);
            }
            else
                generate_spinor_wave_functions(-1);

            /*for (int i = 0; i < 3; i++)
                test_spinor_wave_functions(i); */
        }

        void test_scalar_wave_functions(int use_fft)
        {
            std::vector<complex16> v1;
            std::vector<complex16> v2;
            
            if (use_fft == 0) 
            {
                v1.resize(num_gkvec());
                v2.resize(parameters_.fft().size());
            }
            
            if (use_fft == 1) 
            {
                v1.resize(parameters_.fft().size());
                v2.resize(parameters_.fft().size());
            }
            
            double maxerr = 0;
        
            for (int j1 = 0; j1 < parameters_.num_fv_states(); j1++)
            {
                if (use_fft == 0)
                {
                    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                            &scalar_wave_functions_(parameters_.mt_basis_size(), j1));
                    parameters_.fft().transform(1);
                    parameters_.fft().output(&v2[0]);

                    for (int ir = 0; ir < parameters_.fft().size(); ir++)
                        v2[ir] *= parameters_.step_function(ir);
                    
                    parameters_.fft().input(&v2[0]);
                    parameters_.fft().transform(-1);
                    parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[0]); 
                }
                
                if (use_fft == 1)
                {
                    parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                            &scalar_wave_functions_(parameters_.mt_basis_size(), j1));
                    parameters_.fft().transform(1);
                    parameters_.fft().output(&v1[0]);
                }
               
                for (int j2 = 0; j2 < parameters_.num_fv_states(); j2++)
                {
                    complex16 zsum(0.0, 0.0);
                    for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                    {
                        int offset_wf = parameters_.atom(ia)->offset_wf();
                        AtomType* type = parameters_.atom(ia)->type();
                        AtomSymmetryClass* symmetry_class = parameters_.atom(ia)->symmetry_class();
        
                        for (int l = 0; l <= parameters_.lmax_apw(); l++)
                        {
                            int ordmax = type->indexr().num_rf(l);
                            for (int io1 = 0; io1 < ordmax; io1++)
                                for (int io2 = 0; io2 < ordmax; io2++)
                                    for (int m = -l; m <= l; m++)
                                        zsum += conj(scalar_wave_functions_(offset_wf + 
                                                                            type->indexb_by_l_m_order(l, m, io1), j1)) *
                                                     scalar_wave_functions_(offset_wf + 
                                                                            type->indexb_by_l_m_order(l, m, io2), j2) * 
                                                     symmetry_class->o_radial_integral(l, io1, io2);
                        }
                    }
                    
                    if (use_fft == 0)
                    {
                       for (int ig = 0; ig < num_gkvec(); ig++)
                           zsum += conj(v1[ig]) * scalar_wave_functions_(parameters_.mt_basis_size() + ig, j2);
                    }
                   
                    if (use_fft == 1)
                    {
                        parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                           &scalar_wave_functions_(parameters_.mt_basis_size(), j2));
                        parameters_.fft().transform(1);
                        parameters_.fft().output(&v2[0]);
        
                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            zsum += conj(v1[ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
                    }
                    
                    if (use_fft == 2) 
                    {
                        for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                        {
                            for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                            {
                                int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
                                zsum += conj(scalar_wave_functions_(parameters_.mt_basis_size() + ig1, j1)) * 
                                             scalar_wave_functions_(parameters_.mt_basis_size() + ig2, j2) * 
                                        parameters_.step_function_pw(ig3);
                            }
                       }
                   }
       
                   zsum = (j1 == j2) ? zsum - complex16(1.0, 0.0) : zsum;
                   maxerr = std::max(maxerr, abs(zsum));
                }
            }
            std :: cout << "maximum error = " << maxerr << std::endl;
        }
        
        void test_spinor_wave_functions(int use_fft)
        {
            std::vector<complex16> v1[2];
            std::vector<complex16> v2;

            if (use_fft == 0 || use_fft == 1)
                v2.resize(parameters_.fft().size());
            
            if (use_fft == 0) 
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    v1[ispn].resize(num_gkvec());
            
            if (use_fft == 1) 
                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    v1[ispn].resize(parameters_.fft().size());
            
            double maxerr = 0;
        
            for (int j1 = 0; j1 < parameters_.num_bands(); j1++)
            {
                if (use_fft == 0)
                {
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                           &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
                        parameters_.fft().transform(1);
                        parameters_.fft().output(&v2[0]);

                        for (int ir = 0; ir < parameters_.fft().size(); ir++)
                            v2[ir] *= parameters_.step_function(ir);
                        
                        parameters_.fft().input(&v2[0]);
                        parameters_.fft().transform(-1);
                        parameters_.fft().output(num_gkvec(), &fft_index_[0], &v1[ispn][0]); 
                    }
                }
                
                if (use_fft == 1)
                {
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                           &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j1));
                        parameters_.fft().transform(1);
                        parameters_.fft().output(&v1[ispn][0]);
                    }
                }
               
                for (int j2 = 0; j2 < parameters_.num_bands(); j2++)
                {
                    complex16 zsum(0.0, 0.0);
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                        {
                            int offset_wf = parameters_.atom(ia)->offset_wf();
                            AtomType* type = parameters_.atom(ia)->type();
                            AtomSymmetryClass* symmetry_class = parameters_.atom(ia)->symmetry_class();
        
                            for (int l = 0; l <= parameters_.lmax_apw(); l++)
                            {
                                int ordmax = type->indexr().num_rf(l);
                                for (int io1 = 0; io1 < ordmax; io1++)
                                    for (int io2 = 0; io2 < ordmax; io2++)
                                        for (int m = -l; m <= l; m++)
                                            zsum += conj(spinor_wave_functions_(offset_wf + 
                                                                                type->indexb_by_l_m_order(l, m, io1),
                                                                                ispn, j1)) *
                                                         spinor_wave_functions_(offset_wf + 
                                                                                type->indexb_by_l_m_order(l, m, io2), 
                                                                                ispn, j2) * 
                                                         symmetry_class->o_radial_integral(l, io1, io2);
                            }
                        }
                    }
                    
                    if (use_fft == 0)
                    {
                       for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                       {
                           for (int ig = 0; ig < num_gkvec(); ig++)
                               zsum += conj(v1[ispn][ig]) * spinor_wave_functions_(parameters_.mt_basis_size() + ig, ispn, j2);
                       }
                    }
                   
                    if (use_fft == 1)
                    {
                        for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                        {
                            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                               &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j2));
                            parameters_.fft().transform(1);
                            parameters_.fft().output(&v2[0]);
        
                            for (int ir = 0; ir < parameters_.fft().size(); ir++)
                                zsum += conj(v1[ispn][ir]) * v2[ir] * parameters_.step_function(ir) / double(parameters_.fft().size());
                        }
                    }
                    
                    if (use_fft == 2) 
                    {
                        for (int ig1 = 0; ig1 < num_gkvec(); ig1++)
                        {
                            for (int ig2 = 0; ig2 < num_gkvec(); ig2++)
                            {
                                int ig3 = parameters_.index_g12(gvec_index(ig1), gvec_index(ig2));
                                for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                                    zsum += conj(spinor_wave_functions_(parameters_.mt_basis_size() + ig1, ispn, j1)) * 
                                                 spinor_wave_functions_(parameters_.mt_basis_size() + ig2, ispn, j2) * 
                                            parameters_.step_function_pw(ig3);
                            }
                       }
                   }
       
                   zsum = (j1 == j2) ? zsum - complex16(1.0, 0.0) : zsum;
                   maxerr = std::max(maxerr, abs(zsum));
                }
            }
            std :: cout << "maximum error = " << maxerr << std::endl;
        }
                
        inline int num_gkvec()
        {
            assert(gkvec_.size(1) == (int)gvec_index_.size());

            return gkvec_.size(1);
        }

        inline double* gkvec(int ig)
        {
            assert(ig >= 0 && ig < gkvec_.size(1));

            return &gkvec_(0, ig);
        }

        inline int gvec_index(int ig) 
        {
            assert(ig >= 0 && ig < (int)gvec_index_.size());
            
            return gvec_index_[ig];
        }

        inline complex16& matching_coefficient(int ig, int i)
        {
            return matching_coefficients_(ig, i);
        }

        /*!
            \brief First-variational basis size
            
            Total number of first-variational functions equals to the sum of the number of augmented 
            plane waves and the number of local orbitals. Number of first-variational functions controls 
            the size of the firt-variational Hamiltonian and overlap matrices and the size of the 
            first-variational eigen-vectors.
        */
        inline int fv_basis_size()
        {
            return (num_gkvec() + parameters_.mt_lo_basis_size());
        }
        
        /*!
            \brief Total size of the scalar wave-function.
        */
        inline int scalar_wf_size()
        {
            return (parameters_.mt_basis_size() + num_gkvec());
        }

        inline void set_band_occupancies(double* band_occupancies)
        {
            band_occupancies_.resize(parameters_.num_bands());
            memcpy(&band_occupancies_[0], band_occupancies, parameters_.num_bands() * sizeof(double));
        }

        inline void get_band_energies(double* band_energies)
        {
            assert((int)band_energies_.size() == parameters_.num_bands());
            
            memcpy(band_energies, &band_energies_[0], parameters_.num_bands() * sizeof(double));
        }

        inline double band_occupancy(int j)
        {
            return band_occupancies_[j];
        }
        
        inline double band_energy(int j)
        {
            return band_energies_[j];
        }

        inline double weight()
        {
            return weight_;
        }

        inline complex16& spinor_wave_function(int idxwf, int ispn, int j)
        {
            return spinor_wave_functions_(idxwf, ispn, j);
        }

        inline int* fft_index()
        {
            return &fft_index_[0];
        }

        inline double* vk()
        {
            return vk_;
        }
};

};

