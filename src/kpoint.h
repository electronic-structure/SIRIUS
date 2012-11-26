namespace sirius
{

class kpoint
{
    private:

        /// global set of parameters
        Global& parameters_;

        /// grid of MPI ranks participating in diagonalization
        MPIGrid* mpi_grid_;

        /// G+k vectors
        mdarray<double, 2> gkvec_;

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        /// plane-wave matching coefficients
        mdarray<complex16, 2> matching_coefficients_;

        std::vector<double> evalfv_;

        mdarray<complex16, 2> evecfv_;
        
        mdarray<complex16, 2> evecsv_;

        std::vector<int> fft_index_;
        
        mdarray<complex16, 2> scalar_wave_functions_;

        mdarray<complex16, 3> spinor_wave_functions_;

        std::vector<double> band_occupancies_;

        std::vector<double> band_energies_; 

        /// weight of k-poitn
        double weight_;

        /// fractional k-point coordinates
        double vk_[3];

        mdarray<complex16, 2> gkvec_phase_factors_;

        mdarray<complex16, 2> gkvec_ylm_;

        mdarray<double, 4> sbessel_mt_;

        std::vector<double> gkvec_len_;

        /// BLACS communication context
        int blacs_context;

        int num_gkvec_row_;
        int num_gkvec_col_;
        /// short information about each APW+lo basis function
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_;
        
        /// in case of ScaLAPACK: local set of basis descriptors: rows then columns
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_loc_;

        int apwlo_row_basis_size_;
        int apwlo_col_basis_size_;

        /// Generate plane-wave matching coefficents for the radial solutions 
        void generate_matching_coefficients()
        {
            Timer t("sirius::kpoint::generate_matching_coefficients");

            std::vector<complex16> zil(parameters_.lmax_apw() + 1);
            for (int l = 0; l <= parameters_.lmax_apw(); l++)
                zil[l] = pow(complex16(0.0, 1.0), l);
      
            matching_coefficients_.set_dimensions(num_gkvec_loc(), parameters_.mt_aw_basis_size());
            matching_coefficients_.allocate();

            #pragma omp parallel default(shared)
            {
                complex16 a[2][2];
                mdarray<complex16,2> b(2, (2 * parameters_.lmax_apw() + 1) * num_gkvec_loc());

                #pragma omp for
                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    assert(parameters_.atom(ia)->type()->max_aw_order() <= 2);

                    int iat = parameters_.atom_type_index_by_id(parameters_.atom(ia)->type_id());

                    double R = parameters_.atom(ia)->type()->mt_radius();

                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                    {
                        int num_aw = (int)parameters_.atom(ia)->type()->aw_descriptor(l).size();

                        for (int order = 0; order < num_aw; order++)
                            for (int order1 = 0; order1 < num_aw; order1++)
                                a[order][order1] = complex16(parameters_.atom(ia)->symmetry_class()->
                                    aw_surface_dm(l, order, order1), 0.0);

                        double det = (num_aw == 1) ? abs(a[0][0]) : abs(a[0][0] * a[1][1] - a[0][1] * a [1][0]);
                        if (det  < 1e-8) 
                            error(__FILE__, __LINE__, "ill defined linear equation problem", fatal_err);
                        
                        int n = 0;
                        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
                        {
                            double gkR = gkvec_len_[igkloc] * R; // |G+k|*R
                            for (int m = -l; m <= l; m++)
                                for (int order = 0; order < num_aw; order++)
                                    b(order, n++) = (fourpi / sqrt(parameters_.omega())) * zil[l] * 
                                                    sbessel_mt_(l, iat, igkloc, order) * 
                                                    gkvec_phase_factors_(igkloc, ia) * 
                                                    conj(gkvec_ylm_(lm_by_l_m(l, m), igkloc)) * pow(gkR, order); 
                        }
                              
                        int info = gesv(num_aw, n, &a[0][0], 2, &b(0, 0), 2);

                        if (info)
                        {
                            std::stringstream s;
                            s << "gtsv returned " << info;
                            error(__FILE__, __LINE__, s, fatal_err);
                        }

                        int offs = parameters_.atom(ia)->offset_aw();
                        n = 0;
                        for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
                            for (int m = -l; m <= l; m++)
                                for (int order = 0; order < num_aw; order++)
                                {
                                    int idxb = parameters_.atom(ia)->type()->indexb_by_l_m_order(l, m, order);
                                    // it is more convenient to store conjugated coefficients because then the 
                                    // overlap matrix is set with single matrix-matrix multiplication without 
                                    // further conjugation 
                                    matching_coefficients_(igkloc, offs + idxb) = conj(b(order, n++));
                                }
                    } // l
                } //ia
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
            Timer t("sirius::kpoint::generate_scalar_wave_functions");
            
            scalar_wave_functions_.set_dimensions(scalar_wf_size(), parameters_.num_fv_states());
            scalar_wave_functions_.allocate();
            
            gemm<cpu>(2, 0, parameters_.mt_aw_basis_size(), parameters_.num_fv_states(), num_gkvec(), complex16(1.0, 0.0), 
                &matching_coefficients_(0, 0), num_gkvec(), &evecfv_(0, 0), apwlo_basis_size(), 
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
            Timer t("sirius::kpoint::generate_spinor_wave_functions");

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

        void generate_gkvec()
        {
            if (parameters_.aw_cutoff() > double(parameters_.lmax_apw()))
                error(__FILE__, __LINE__, "aw cutoff is too large for a given lmax");

            double gk_cutoff = parameters_.aw_cutoff() / parameters_.min_mt_radius();
            
            if (gk_cutoff * 2 > parameters_.pw_cutoff())
                error(__FILE__, __LINE__, "aw cutoff is too large for a given plane-wave cutoff");

            std::vector< std::pair<double, int> > gkmap;

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

            gkvec_.set_dimensions(3, (int)gkmap.size());
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

        void precompute_gk()
        {
            gkvec_phase_factors_.set_dimensions(num_gkvec_loc(), parameters_.num_atoms());
            gkvec_phase_factors_.allocate();

            gkvec_ylm_.set_dimensions(parameters_.lmmax_apw(), num_gkvec_loc());
            gkvec_ylm_.allocate();

            #pragma omp parallel for default(shared)
            for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
            {
                int igk = igkglob(igkloc);
                double v[3];
                double vs[3];

                parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
                SHT::spherical_coordinates(v, vs); // vs = {r, theta, phi}

                SHT::spherical_harmonics(parameters_.lmax_apw(), vs[1], vs[2], &gkvec_ylm_(0, igkloc));

                for (int ia = 0; ia < parameters_.num_atoms(); ia++)
                {
                    double phase = twopi * scalar_product(gkvec(igk), parameters_.atom(ia)->position());

                    gkvec_phase_factors_(igkloc, ia) = exp(complex16(0.0, phase));
                }
            }
            
            // compute values of spherical Bessel functions and first derivative at MT boundary
            sbessel_mt_.set_dimensions(parameters_.lmax_apw() + 2, parameters_.num_atom_types(), num_gkvec_loc(), 2);
            sbessel_mt_.allocate();
            sbessel_mt_.zero();
                    
            gkvec_len_.resize(num_gkvec_loc());

            for (int igkloc = 0; igkloc < num_gkvec_loc(); igkloc++)
            {
                int igk = igkglob(igkloc);

                double v[3];
                parameters_.get_coordinates<cartesian, reciprocal>(gkvec(igk), v);
                gkvec_len_[igkloc] = vector_length(v);
            
                for (int iat = 0; iat < parameters_.num_atom_types(); iat++)
                {
                    double R = parameters_.atom_type(iat)->mt_radius();

                    double gkR = gkvec_len_[igkloc] * R;

                    gsl_sf_bessel_jl_array(parameters_.lmax_apw() + 1, gkR, &sbessel_mt_(0, iat, igkloc, 0));
                    
                    // Bessel function derivative: f_{{n}}^{{\prime}}(z)=-f_{{n+1}}(z)+(n/z)f_{{n}}(z)
                    for (int l = 0; l <= parameters_.lmax_apw(); l++)
                        sbessel_mt_(l, iat, igkloc, 1) = -sbessel_mt_(l + 1, iat, igkloc, 0) * gkvec_len_[igkloc] + 
                                                         (l / R) * sbessel_mt_(l, iat, igkloc, 0);
                }
            }
        }

        void init_blacs_context()
        {
            int nrow = mpi_grid_->size(1 << 0);
            int ncol = mpi_grid_->size(1 << 1);

            mdarray<int, 2> map_ranks(nrow, ncol);
            for (int i1 = 0; i1 < ncol; i1++)
                for (int i0 = 0; i0 < nrow; i0++)
                    map_ranks(i0, i1) = mpi_grid_->cart_rank(mpi_grid_->communicator(), intvec(i0, i1));
 
            // create BLACS context
            blacs_context = Csys2blacs_handle(mpi_grid_->communicator());

            // create grid of MPI ranks 
            Cblacs_gridmap(&blacs_context, map_ranks.get_ptr(), nrow, nrow, ncol);

            // check the grid
            int irow, icol;
            Cblacs_gridinfo(blacs_context, &nrow, &ncol, &irow, &icol);
            std::vector<int> x = mpi_grid_->coordinates();
            if ((x[0] != irow) || (x[1] != icol)) error(__FILE__, __LINE__, "wrong grid", fatal_err);
        }

        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors()
        {
            apwlo_basis_descriptor apwlobd;
            // G+k basis functions
            for (int igk = 0; igk < num_gkvec(); igk++)
            {
                apwlobd.global_index = (int)apwlo_basis_descriptors_.size();
                apwlobd.igk = igk;
                apwlobd.ig = gvec_index_[igk];
                apwlobd.ia = -1;
                apwlobd.lm = -1;
                apwlobd.idxrf  = -1;
                apwlo_basis_descriptors_.push_back(apwlobd);
            }
            // local orbital basis functions
            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                Atom* atom = parameters_.atom(ia);
                AtomType* type = atom->type();
            
                int lo_index_offset = type->mt_aw_basis_size();
                
                for (int j = 0; j < type->mt_lo_basis_size(); j++) 
                {
                    int lm = type->indexb(lo_index_offset + j).lm;
                    int idxrf = type->indexb(lo_index_offset + j).idxrf;
                    apwlobd.global_index = (int)apwlo_basis_descriptors_.size();
                    apwlobd.igk = -1;
                    apwlobd.ig = -1;
                    apwlobd.ia = ia;
                    apwlobd.lm = lm;
                    apwlobd.idxrf  = idxrf;
                    apwlo_basis_descriptors_.push_back(apwlobd);
                }
            }
            
            // ckeck if we count basis functions correctly
            if ((int)apwlo_basis_descriptors_.size() != (num_gkvec() + parameters_.mt_lo_basis_size()))
                error(__FILE__, __LINE__, "APW+lo basis descriptors array has a wrong size");
        }

        void partition_apwlo_basis_descriptors()
        {
            std::vector<int> cart_coord = mpi_grid_->coordinates();
            std::vector<int> cart_dims = mpi_grid_->dimensions();

            // distribute row and column index of basis functions in block-cyclic manner
            // add rows first 
            int irank_row = 0;
            int iblock_row = 0;
            for (int i = 0; i < (int)apwlo_basis_descriptors_.size(); i++)
            {
                if (cart_coord[0] == irank_row) apwlo_basis_descriptors_loc_.push_back(apwlo_basis_descriptors_[i]);
                if ((++iblock_row) == scalapack_nb)
                {
                    iblock_row = 0;
                    irank_row = (irank_row + 1) % cart_dims[0];
                }
            }
            apwlo_row_basis_size_ = (int)apwlo_basis_descriptors_loc_.size();
        
            // then add columns 
            int irank_col = 0;
            int iblock_col = 0;
            for (int i = 0; i < (int)apwlo_basis_descriptors_.size(); i++)
            {
                if (cart_coord[1] == irank_col) apwlo_basis_descriptors_loc_.push_back(apwlo_basis_descriptors_[i]);
                if ((++iblock_col) == scalapack_nb)
                {
                    iblock_col = 0;
                    irank_col = (irank_col + 1) % cart_dims[1];
                }
            }
            apwlo_col_basis_size_ = (int)apwlo_basis_descriptors_loc_.size() - apwlo_row_basis_size_;
#if 0
            int nrow = mpi_grid_->size(1 << 0);
            int ncol = mpi_grid_->size(1 << 1);

            for (int i1 = 0; i1 < ncol; i1++)
                for (int i0 = 0; i0 < nrow; i0++)
                {
                    if (cart_coord[0] == i0 && cart_coord[1] == i1)
                    {
                        std::cout << "row = " << i0 <<" col = " << i1 << std::endl;

                        std::cout << "=== row distribution ===" << std::endl;
                        for (int i = 0; i < (int)fv_basis_descriptors_row.size(); i++)
                            std::cout << fv_basis_descriptors_row[i].igk << " " 
                                      << fv_basis_descriptors_row[i].lm << " " 
                                      << fv_basis_descriptors_row[i].idxrf << " "
                                      << fv_basis_descriptors_row[i].global_index << std::endl;

                        std::cout << "=== column distribution ===" << std::endl;
                        for (int i = 0; i < (int)fv_basis_descriptors_col.size(); i++)
                            std::cout << fv_basis_descriptors_col[i].igk << " " 
                                      << fv_basis_descriptors_col[i].lm << " " 
                                      << fv_basis_descriptors_col[i].idxrf << " "
                                      << fv_basis_descriptors_col[i].global_index << std::endl;
                    }
                    mpi_grid_->barrier();
                }
#endif
            // get the number of row- and column- G+k-vectors
            num_gkvec_row_ = 0;
            for (int i = 0; i < apwlo_row_basis_size_; i++)
                if (apwlo_basis_descriptors_loc_[i].igk != -1) num_gkvec_row_++;
            
            num_gkvec_col_ = 0;
            for (int i = 0; i < apwlo_col_basis_size_; i++)
                if (apwlo_basis_descriptors_loc_[apwlo_row_basis_size_ + i].igk != -1) num_gkvec_col_++;

            std::cout << num_gkvec_row_ << " " << num_gkvec_col_ << std::endl;
        }

    public:

        /// Constructor
        kpoint(Global& parameters__, 
               double* vk__, 
               double weight__, 
               MPIGrid* mpi_grid__) : parameters_(parameters__), 
                                      mpi_grid_(mpi_grid__),
                                      weight_(weight__)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];

            if (eigen_value_solver == scalapack) init_blacs_context();
        }

        ~kpoint()
        {
            if (eigen_value_solver == scalapack) Cfree_blacs_system_handle(blacs_context);
            if (mpi_grid_) delete mpi_grid_;
        }

        void initialize()
        {
            Timer t("sirius::kpoint::initialize");

            generate_gkvec();

            build_apwlo_basis_descriptors();

            if (eigen_value_solver == scalapack) partition_apwlo_basis_descriptors();

            precompute_gk();
        }

        void find_eigen_states(Band* band, PeriodicFunction<double>* effective_potential, 
                               PeriodicFunction<double>* effective_magnetic_field[3])
        {
            assert(apwlo_basis_size() > parameters_.num_fv_states());
            assert(band != NULL);
            
            Timer t("sirius::kpoint::find_eigen_states");
            
            evalfv_.resize(parameters_.num_fv_states());
            evecfv_.set_dimensions(apwlo_basis_size(), parameters_.num_fv_states());
            evecfv_.allocate();

            generate_matching_coefficients();

            error(__FILE__, __LINE__, "stop");

            /*band->solve_fv(parameters_, fv_basis_size(), num_gkvec(), &gvec_index_[0], gkvec_, 
                           matching_coefficients_, effective_potential, effective_magnetic_field, evecfv_,
                           &evalfv_[0])*/;

            generate_scalar_wave_functions();
            
            if (test_scalar_wf)
                for (int i = 0; i < 3; i++)
                    test_scalar_wave_functions(i);

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
        
        PeriodicFunction<complex16>* spinor_wave_function_component(int ispn, int j)
        {
            PeriodicFunction<complex16>* func = new PeriodicFunction<complex16>(parameters_, parameters_.lmax_apw());
            func->allocate(ylm_component | it_component);
            func->zero();

            for (int ia = 0; ia < parameters_.num_atoms(); ia++)
            {
                for (int i = 0; i < parameters_.atom(ia)->type()->mt_basis_size(); i++)
                {
                    int lm = parameters_.atom(ia)->type()->indexb(i).lm;
                    int idxrf = parameters_.atom(ia)->type()->indexb(i).idxrf;
                    for (int ir = 0; ir < parameters_.atom(ia)->num_mt_points(); ir++)
                        func->f_ylm(lm, ir, ia) += 
                            spinor_wave_functions_(parameters_.atom(ia)->offset_wf() + i, ispn, j) * 
                            parameters_.atom(ia)->symmetry_class()->radial_function(ir, idxrf);
                }
            }

            parameters_.fft().input(num_gkvec(), &fft_index_[0], 
                                    &spinor_wave_functions_(parameters_.mt_basis_size(), ispn, j));
            parameters_.fft().transform(1);
            parameters_.fft().output(func->f_it());

            for (int i = 0; i < parameters_.fft().size(); i++)
                func->f_it(i) /= sqrt(parameters_.omega());
            
            return func;
        }
                
        /// Total number of G+k vectors within the cutoff distance
        inline int num_gkvec()
        {
            assert(gkvec_.size(1) == (int)gvec_index_.size());

            return gkvec_.size(1);
        }

        /// In case of scalapck diagonalization: number of G+k vectors along the rows of the matrix
        inline int num_gkvec_row()
        {
            if (eigen_value_solver != scalapack)
                error(__FILE__, __LINE__, "can be called with scalapack eigen-value solver only", fatal_err);

            return num_gkvec_row_;
        }

        /// In case of scalapck diagonalization: number of G+k vectors along the columns of the matrix
        inline int num_gkvec_col()
        {
            if (eigen_value_solver != scalapack)
                error(__FILE__, __LINE__, "can be called with scalapack eigen-value solver only", fatal_err);

            return num_gkvec_col_;
        }

        inline int num_gkvec_loc()
        {
            return (eigen_value_solver == scalapack) ? (num_gkvec_row() + num_gkvec_col()) : num_gkvec();
        } 

        inline int igkglob(int igkloc)
        {
            if (eigen_value_solver == scalapack)
            {
                int igk = (igkloc < num_gkvec_row_) ? apwlo_basis_descriptors_loc_[igkloc].igk : 
                          apwlo_basis_descriptors_loc_[apwlo_row_basis_size_ + igkloc - num_gkvec_row_].igk;
                assert(igk >= 0);
                return igk;
            }
            else 
            {
                return igkloc;
            }
        }
        
        /// Pointer to G+k vector
        inline double* gkvec(int igk)
        {
            assert(igk >= 0 && igk < gkvec_.size(1));

            return &gkvec_(0, igk);
        }

        /// Global index of G-vector by the index of G+k vector
        inline int gvec_index(int igk) 
        {
            assert(igk >= 0 && igk < (int)gvec_index_.size());
            
            return gvec_index_[igk];
        }

        /*inline complex16& matching_coefficient(int ig, int i)
        {
            return matching_coefficients_(ig, i);
        }*/


        /// APW+lo basis size

        /** Total number of APW+lo basis functions is equal to the number of augmented plane-waves plus
            the number of local orbitals. */
        inline int apwlo_basis_size()
        {
            return (int)apwlo_basis_descriptors_.size();
        }
        
        /*!
            \brief Total size of the scalar wave-function.
        */
        inline int scalar_wf_size()
        {
            return (parameters_.mt_basis_size() + num_gkvec());
        }

        inline void get_band_occupancies(double* band_occupancies)
        {
            assert((int)band_occupancies_.size() == parameters_.num_bands());
            
            memcpy(band_occupancies, &band_occupancies_[0], parameters_.num_bands() * sizeof(double));
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

        inline void set_band_energies(double* band_energies)
        {
            band_energies_.resize(parameters_.num_bands()); 
            memcpy(&band_energies_[0], band_energies, parameters_.num_bands() * sizeof(double));
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

