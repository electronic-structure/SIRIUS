namespace sirius
{

// TODO: all ranks along columns of mpi grid have identical fv_states array; utilize this!!

class kpoint
{
    private:

        /// global set of parameters
        Global& parameters_;

        /// weight of k-poitn
        double weight_;

        /// fractional k-point coordinates
        double vk_[3];
        
        /// G+k vectors
        mdarray<double, 2> gkvec_;

        /// global index (in the range [0, N_G - 1]) of G-vector by the index of G+k vector in the range [0, N_Gk - 1]
        std::vector<int> gvec_index_;

        /// plane-wave matching coefficients
        mdarray<complex16, 2> matching_coefficients_;

        /// first-variational eigen values
        std::vector<double> fv_eigen_values_;

        /// first-variational eigen vectors
        mdarray<complex16, 2> fv_eigen_vectors_;
        
        /// second-variational eigen vectors
        mdarray<complex16, 2> sv_eigen_vectors_;

        /// position of the G vector (from the G+k set) inside the FFT buffer 
        std::vector<int> fft_index_;
       
        /// first-variational states, distributed along the columns of the MPI grid
        mdarray<complex16, 2> fv_states_col_;
       
        /// first-variational states, distributed along the rows of the MPI grid
        mdarray<complex16, 2> fv_states_row_;

        /// two-component (spinor) wave functions describing the bands
        mdarray<complex16, 3> spinor_wave_functions_;

        /// band occupation numbers
        std::vector<double> band_occupancies_;

        /// band energies
        std::vector<double> band_energies_; 

        /// phase factors \f$ e^{i ({\bf G+k}) {\bf r}_{\alpha}} \f$
        mdarray<complex16, 2> gkvec_phase_factors_;

        /// spherical harmonics of G+k vectors
        mdarray<complex16, 2> gkvec_ylm_;

        /// spherical Bessel functions and first derivateves at the MT boundary
        mdarray<double, 4> sbessel_mt_;

        /// length of G+k vectors
        std::vector<double> gkvec_len_;

        /// number of G+k vectors distributed along rows of MPI grid
        int num_gkvec_row_;
        
        /// number of G+k vectors distributed along columns of MPI grid
        int num_gkvec_col_;

        /// short information about each APW+lo basis function
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_;

        /// row APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_row_;
        
        /// column APW+lo basis descriptors
        std::vector<apwlo_basis_descriptor> apwlo_basis_descriptors_col_;

        /// Generate plane-wave matching coefficents for the radial solutions 
        void generate_matching_coefficients()
        {
            Timer t("sirius::kpoint::generate_matching_coefficients");

            std::vector<complex16> zil(parameters_.lmax_apw() + 1);
            for (int l = 0; l <= parameters_.lmax_apw(); l++)
                zil[l] = pow(complex16(0, 1), l);
      
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

        inline void move_apw_blocks(complex16 *vec)
        {
            for (int ia = parameters_.num_atoms() - 1; ia >= 0; ia--)
            {
                int final_block_offset = parameters_.atom(ia)->offset_wf();
                int initial_block_offset = parameters_.atom(ia)->offset_aw();
                int block_size = parameters_.atom(ia)->type()->mt_aw_basis_size();
        
                memmove(&vec[final_block_offset], &vec[initial_block_offset], block_size * sizeof(complex16));

                memset(&vec[final_block_offset + block_size], 0, 
                       parameters_.atom(ia)->type()->mt_lo_basis_size() * sizeof(complex16));
            }
        }
        
        inline void copy_lo_blocks(const int apwlo_basis_size_row, const int num_gkvec_row, 
                                   const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                   const complex16* z, complex16 *vec)
        {
            for (int j = num_gkvec_row; j < apwlo_basis_size_row; j++)
            {
                int ia = apwlo_basis_descriptors_row[j].ia;
                int lm = apwlo_basis_descriptors_row[j].lm;
                int order = apwlo_basis_descriptors_row[j].order;
                vec[parameters_.atom(ia)->offset_wf() + parameters_.atom(ia)->type()->indexb_by_lm_order(lm, order)] = z[j];
            }
        }
        
        inline void copy_pw_block(const int num_gkvec, const int num_gkvec_row, 
                                  const std::vector<apwlo_basis_descriptor>& apwlo_basis_descriptors_row, 
                                  const complex16* z, complex16 *vec)
        {
            memset(vec, 0, num_gkvec * sizeof(complex16));

            for (int j = 0; j < num_gkvec_row; j++)
                vec[apwlo_basis_descriptors_row[j].igk] = z[j];
        }

        void generate_fv_states(Band* band)
        {
            Timer t("sirius::kpoint::generate_fv_states");
            
            fv_states_col_.set_dimensions(mtgk_size(), band->spl_fv_states_col().local_size());
            fv_states_col_.allocate();
                
            gemm<cpu>(2, 0, parameters_.mt_aw_basis_size(), band->spl_fv_states_col().local_size(),
                      num_gkvec_row(), complex16(1, 0), &matching_coefficients_(0, 0), matching_coefficients_.ld(),
                      &fv_eigen_vectors_(0, 0), fv_eigen_vectors_.ld(), complex16(0, 0), &fv_states_col_(0, 0), 
                      fv_states_col_.ld());

            for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
            {
                move_apw_blocks(&fv_states_col_(0, j));
        
                copy_lo_blocks(apwlo_basis_size_row(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                               &fv_eigen_vectors_(0, j), &fv_states_col_(0, j));
        
                copy_pw_block(num_gkvec(), num_gkvec_row(), apwlo_basis_descriptors_row_, 
                              &fv_eigen_vectors_(0, j), &fv_states_col_(parameters_.mt_basis_size(), j));
            }

            for (int j = 0; j < band->spl_fv_states_col().local_size(); j++)
                Platform::allreduce(&fv_states_col_(0, j), mtgk_size(), 
                                    parameters_.mpi_grid().communicator(1 << band->dim_row()));
        }
        
        void generate_spinor_wave_functions(Band* band, bool has_sv_evec)
        {
            Timer t("sirius::kpoint::generate_spinor_wave_functions");

            spinor_wave_functions_.set_dimensions(mtgk_size(), parameters_.num_spins(), 
                                                  band->spl_spinor_wf_col().local_size());
            if (has_sv_evec)
            {
                spinor_wave_functions_.allocate();
                spinor_wave_functions_.zero();
                
                if (parameters_.num_mag_dims() != 3)
                {
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                        gemm<cpu>(0, 0, mtgk_size(), band->spl_fv_states_col().local_size(), 
                                  band->spl_fv_states_row().local_size(), complex16(1, 0), 
                                  &fv_states_row_(0, 0), fv_states_row_.ld(), 
                                  &sv_eigen_vectors_(0, ispn * band->spl_fv_states_col().local_size()), 
                                  sv_eigen_vectors_.ld(), complex16(0, 0), 
                                  &spinor_wave_functions_(0, ispn, ispn * band->spl_fv_states_col().local_size()), 
                                  spinor_wave_functions_.ld() * parameters_.num_spins());

                }
                else
                {
                    for (int ispn = 0; ispn < parameters_.num_spins(); ispn++)
                    {
                        gemm<cpu>(0, 0, mtgk_size(), band->spl_spinor_wf_col().local_size(), 
                                  band->num_fv_states_row(ispn), complex16(1, 0), 
                                  &fv_states_row_(0, band->offs_fv_states_row(ispn)), fv_states_row_.ld(), 
                                  &sv_eigen_vectors_(ispn * band->num_fv_states_row_up(), 0), sv_eigen_vectors_.ld(), 
                                  complex16(0, 0), &spinor_wave_functions_(0, ispn, 0), 
                                  spinor_wave_functions_.ld() * parameters_.num_spins());
                    }
                }
                
                for (int i = 0; i < band->spl_spinor_wf_col().local_size(); i++)
                    Platform::allreduce(&spinor_wave_functions_(0, 0, i), 
                                        spinor_wave_functions_.size(0) * spinor_wave_functions_.size(1), 
                                        parameters_.mpi_grid().communicator(1 << band->dim_row()));
            }
            else
            {
                spinor_wave_functions_.set_ptr(fv_states_col_.get_ptr());
                memcpy(&band_energies_[0], &fv_eigen_values_[0], parameters_.num_bands() * sizeof(double));
            }
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
                double gklen = Utils::vector_length(v);

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

        void init_gkvec()
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
                gkvec_len_[igkloc] = Utils::vector_length(v);
            
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

        /// Build APW+lo basis descriptors 
        void build_apwlo_basis_descriptors()
        {
            apwlo_basis_descriptor apwlobd;

            // G+k basis functions
            for (int igk = 0; igk < num_gkvec(); igk++)
            {
                apwlobd.igk = igk;
                apwlobd.ig = gvec_index_[igk];
                apwlobd.ia = -1;
                apwlobd.lm = -1;
                apwlobd.l = -1;
                apwlobd.order = -1;
                apwlobd.idxrf = -1;
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
                    int l = type->indexb(lo_index_offset + j).l;
                    int lm = type->indexb(lo_index_offset + j).lm;
                    int order = type->indexb(lo_index_offset + j).order;
                    int idxrf = type->indexb(lo_index_offset + j).idxrf;
                    apwlobd.igk = -1;
                    apwlobd.ig = -1;
                    apwlobd.ia = ia;
                    apwlobd.lm = lm;
                    apwlobd.l = l;
                    apwlobd.order = order;
                    apwlobd.idxrf = idxrf;
                    apwlo_basis_descriptors_.push_back(apwlobd);
                }
            }
            
            // ckeck if we count basis functions correctly
            if ((int)apwlo_basis_descriptors_.size() != (num_gkvec() + parameters_.mt_lo_basis_size()))
                error(__FILE__, __LINE__, "APW+lo basis descriptors array has a wrong size");
        }

        /// Block-cyclic distribution of relevant arrays 
        void distribute_block_cyclic(Band* band)
        {
            // distribute APW+lo basis between rows
            splindex<block_cyclic, scalapack_nb> spl_row(apwlo_basis_size(), band->num_ranks_row(), band->rank_row());
            apwlo_basis_descriptors_row_.resize(spl_row.local_size());
            for (int i = 0; i < spl_row.local_size(); i++)
                apwlo_basis_descriptors_row_[i] = apwlo_basis_descriptors_[spl_row[i]];

            // distribute APW+lo basis between columns
            splindex<block_cyclic, scalapack_nb> spl_col(apwlo_basis_size(), band->num_ranks_col(), band->rank_col());
            apwlo_basis_descriptors_col_.resize(spl_col.local_size());
            for (int i = 0; i < spl_col.local_size(); i++)
                apwlo_basis_descriptors_col_[i] = apwlo_basis_descriptors_[spl_col[i]];

            // get the number of row- and column- G+k-vectors
            num_gkvec_row_ = 0;
            for (int i = 0; i < apwlo_basis_size_row(); i++)
                if (apwlo_basis_descriptors_row_[i].igk != -1) num_gkvec_row_++;
            
            num_gkvec_col_ = 0;
            for (int i = 0; i < apwlo_basis_size_col(); i++)
                if (apwlo_basis_descriptors_col_[i].igk != -1) num_gkvec_col_++;
        }
        
#if 0        
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
#endif

    public:

        /// Constructor
        kpoint(Global& parameters__, 
               double* vk__, 
               double weight__) : parameters_(parameters__), 
                                  weight_(weight__)
        {
            for (int x = 0; x < 3; x++) vk_[x] = vk__[x];
        }

        ~kpoint()
        {
        }

        void initialize(Band* band)
        {
            Timer t("sirius::kpoint::initialize");

            generate_gkvec();

            build_apwlo_basis_descriptors();

            distribute_block_cyclic(band);
            
            init_gkvec();
        }

        void find_eigen_states(Band* band, PeriodicFunction<double>* effective_potential, 
                               PeriodicFunction<double>* effective_magnetic_field[3])
        {
            assert(apwlo_basis_size() > parameters_.num_fv_states());
            assert(band != NULL);
            
            Timer t("sirius::kpoint::find_eigen_states");

            generate_matching_coefficients();

            fv_eigen_values_.resize(parameters_.num_fv_states());

            fv_eigen_vectors_.set_dimensions(apwlo_basis_size_row(), band->spl_fv_states_col().local_size());
            fv_eigen_vectors_.allocate();
            
            band->solve_fv(parameters_, 
                           apwlo_basis_size(),
                           num_gkvec(),
                           apwlo_basis_descriptors_row_, 
                           apwlo_basis_size_row(), 
                           num_gkvec_row(),
                           apwlo_basis_descriptors_col_, 
                           apwlo_basis_size_col(), 
                           num_gkvec_col(),
                           matching_coefficients_, 
                           gkvec_, 
                           effective_potential, 
                           effective_magnetic_field, 
                           fv_eigen_values_,
                           fv_eigen_vectors_);

            generate_fv_states(band);
            
            // we don't need first-variational eigen-vectors; all information is available in fv_states array
            fv_eigen_vectors_.deallocate();
            
            // distribute fv states along rows of the MPI grid
            if (band->num_ranks() == 1)
            {
                fv_states_row_.set_dimensions(mtgk_size(), parameters_.num_fv_states());
                fv_states_row_.set_ptr(fv_states_col_.get_ptr());
            }
            else
            {
                fv_states_row_.set_dimensions(mtgk_size(), band->spl_fv_states_row().local_size());
                fv_states_row_.allocate();
                fv_states_row_.zero();

                for (int i = 0; i < band->spl_fv_states_row().local_size(); i++)
                {
                    int ist = (band->spl_fv_states_row()[i] % parameters_.num_fv_states());
                    int offset_col = band->spl_fv_states_col().location(0, ist);
                    int rank_col = band->spl_fv_states_col().location(1, ist);
                    if (rank_col == band->rank_col())
                        memcpy(&fv_states_row_(0, i), &fv_states_col_(0, offset_col), mtgk_size() * sizeof(complex16));

                    Platform::allreduce(&fv_states_row_(0, i), mtgk_size(), 
                                        parameters_.mpi_grid().communicator(1 << band->dim_col()));
                }
            }

            /*if (test_scalar_wf)
                for (int i = 0; i < 3; i++)
                    test_scalar_wave_functions(i); */

            sv_eigen_vectors_.set_dimensions(band->spl_fv_states_row().local_size(), 
                                             band->spl_spinor_wf_col().local_size());
            sv_eigen_vectors_.allocate();

            band_energies_.resize(parameters_.num_bands());
            
            if (parameters_.num_spins() == 2) // or some other conditions which require second variation
            {
                band->solve_sv(parameters_, 
                               mtgk_size(), num_gkvec(), fft_index(), &fv_eigen_values_[0], 
                               fv_states_row_, fv_states_col_, effective_magnetic_field, &band_energies_[0],
                               sv_eigen_vectors_);

                generate_spinor_wave_functions(band, true);
            }
            else
            {
                generate_spinor_wave_functions(band, false);
            }

            /*for (int i = 0; i < 3; i++)
                test_spinor_wave_functions(i); */
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

        /// Number of G+k vectors along the rows of the matrix
        inline int num_gkvec_row()
        {
            return num_gkvec_row_;
        }

        /// Number of G+k vectors along the columns of the matrix
        inline int num_gkvec_col()
        {
            return num_gkvec_col_;
        }

        /// Local fraction of G+k vectors for a given MPI rank

        /** In case of ScaLAPACK row and column G+k vector blocks are combined. */
        inline int num_gkvec_loc()
        {
            if ((num_gkvec_row() == num_gkvec()) && (num_gkvec_col() == num_gkvec()))
            {
                return num_gkvec();
            }
            else
            {
                return (num_gkvec_row() + num_gkvec_col());
            }
        } 

        inline int igkglob(int igkloc)
        {
            if ((num_gkvec_row() == num_gkvec()) && (num_gkvec_col() == num_gkvec()))
            {
                return igkloc;
            }
            else
            {
                int igk = (igkloc < num_gkvec_row()) ? apwlo_basis_descriptors_row_[igkloc].igk : 
                                                       apwlo_basis_descriptors_col_[igkloc - num_gkvec_row()].igk;
                assert(igk >= 0);
                return igk;
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

        /// APW+lo basis size

        /** Total number of APW+lo basis functions is equal to the number of augmented plane-waves plus
            the number of local orbitals. */
        inline int apwlo_basis_size()
        {
            return (int)apwlo_basis_descriptors_.size();
        }

        /// number of APW+lo basis functions distributed along rows of MPI grid
        inline int apwlo_basis_size_row()
        {
            return (int)apwlo_basis_descriptors_row_.size();
        }

        /// number of APW+lo basis functions distributed along columns of MPI grid
        inline int apwlo_basis_size_col()
        {
            return (int)apwlo_basis_descriptors_col_.size();
        }

        /// Total number of muffin-tin and plane-wave expansion coefficients for the first-variational state

        /** APW+lo basis \f$ \varphi_{\mu {\bf k}}({\bf r}) = \{ \varphi_{\bf G+k}({\bf r}),
            \varphi_{j{\bf k}}({\bf r}) \} \f$ is used to expand first-variational wave-functions:

            \f[
                \psi_{i{\bf k}}({\bf r}) = \sum_{\mu} c_{\mu i}^{\bf k} \varphi_{\mu \bf k}({\bf r}) = 
                \sum_{{\bf G}}c_{{\bf G} i}^{\bf k} \varphi_{\bf G+k}({\bf r}) + 
                \sum_{j}c_{j i}^{\bf k}\varphi_{j{\bf k}}({\bf r})
            \f]

            Inside muffin-tins the expansion is converted into the following form:
            \f[
                \psi_{i {\bf k}}({\bf r})= \begin{array}{ll} 
                \displaystyle \sum_{L} \sum_{\lambda=1}^{N_{\ell}^{\alpha}} 
                F_{L \lambda}^{i {\bf k},\alpha}f_{\ell \lambda}^{\alpha}(r) 
                Y_{\ell m}(\hat {\bf r}) & {\bf r} \in MT_{\alpha} \end{array}
            \f]

            Thus, the total number of coefficients representing a first-variational state is equal
            to the number of muffi-tin basis functions of the form \f$ f_{\ell \lambda}^{\alpha}(r) 
            Y_{\ell m}(\hat {\bf r}) \f$ plust the number of G+k plane waves. */ 
        inline int mtgk_size()
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

