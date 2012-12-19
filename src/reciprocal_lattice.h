#ifndef __RECIPROCAL_LATTICE_H__
#define __RECIPROCAL_LATTICE_H__

namespace sirius {

class ReciprocalLattice : public UnitCell
{
    private:
        
        /// plane wave cutoff radius (in inverse a.u. of length)
        double pw_cutoff_;
        
        /// FFT wrapper
        FFT3D fft_;

        /// list of G-vector fractional coordinates
        mdarray<int,2> gvec_;

        /// number of G-vectors within plane wave cutoff
        int num_gvec_;

        /// length of G-vectors belonging to the same shell
        std::vector<double> gvec_shell_len_;

        /// mapping between G-vector and shell
        std::vector<int> gvec_shell_;

        /// mapping between linear G-vector index and G-vector coordinates
        mdarray<int,3> index_by_gvec_;

        /// mapping betwee linear G-vector index and position in FFT buffer
        std::vector<int> fft_index_;

    protected:

        void init()
        {
            Timer t("sirius::ReciprocalLattice::init");
            
            int max_frac_coord[3];
            find_translation_limits<reciprocal>(pw_cutoff(), max_frac_coord);
            
            fft_.init(max_frac_coord);
            
            mdarray<int, 2> gvec_tmp(3, fft_.size());
            std::vector< std::pair<double, int> > gvec_tmp_length;

            int ig = 0;
            for (int i0 = fft_.grid_limits(0, 0); i0 <= fft_.grid_limits(0, 1); i0++)
                for (int i1 = fft_.grid_limits(1, 0); i1 <= fft_.grid_limits(1, 1); i1++)
                    for (int i2 = fft_.grid_limits(2, 0); i2 <= fft_.grid_limits(2, 1); i2++)
                    {
                        gvec_tmp(0, ig) = i0;
                        gvec_tmp(1, ig) = i1;
                        gvec_tmp(2, ig) = i2;

                        int fracc[] = {i0, i1, i2};
                        double cartc[3];
                        get_coordinates<cartesian, reciprocal>(fracc, cartc);

                        gvec_tmp_length.push_back(std::pair<double, int>(Utils::vector_length(cartc), ig++));
                    }

            std::sort(gvec_tmp_length.begin(), gvec_tmp_length.end());

            // create sorted list of G-vectors
            gvec_.set_dimensions(3, fft_.size());
            gvec_.allocate();

            num_gvec_ = 0;
            for (int i = 0; i < fft_.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                    gvec_(j, i) = gvec_tmp(j, gvec_tmp_length[i].second);
                
                if (gvec_tmp_length[i].first <= pw_cutoff_)
                    num_gvec_++;
            }
            
            // clean temporary arrays
            gvec_tmp.deallocate();
            gvec_tmp_length.clear();

            index_by_gvec_.set_dimensions(dimension(fft_.grid_limits(0, 0), fft_.grid_limits(0, 1)),
                                          dimension(fft_.grid_limits(1, 0), fft_.grid_limits(1, 1)),
                                          dimension(fft_.grid_limits(2, 0), fft_.grid_limits(2, 1)));
            index_by_gvec_.allocate();

            fft_index_.resize(fft_.size());

            for (int ig = 0; ig < fft_.size(); ig++)
            {
                int i0 = gvec_(0, ig);
                int i1 = gvec_(1, ig);
                int i2 = gvec_(2, ig);

                // mapping from G-vector to it's index
                index_by_gvec_(i0, i1, i2) = ig;

                // mapping of FFT buffer linear index
                fft_index_[ig] = fft_.index(i0, i1, i2);
            }

            // find G-shells
            gvec_shell_.resize(num_gvec_);
            gvec_shell_len_.clear();
            for (int ig = 0; ig < num_gvec_; ig++)
            {
                double cartc[3];
                get_coordinates<cartesian, reciprocal>(&gvec_(0, ig), cartc);
                double t = Utils::vector_length(cartc);

                if (gvec_shell_len_.empty() || fabs(t - gvec_shell_len_.back()) > 1e-8)
                    gvec_shell_len_.push_back(t);
                 
                gvec_shell_[ig] = (int)gvec_shell_len_.size() - 1;
            }
#if 0
            // compute phase factors
            if (num_atoms())
            {
                Timer t1("sirius::ReciprocalLattice::init:pf");

                splindex spl_fft_size(fft().size(), intvec(Platform::num_mpi_ranks()), intvec(Platform::mpi_rank()));
                //std::vector<double> phase_tmp(fft().size());
                //std::vector<complex16> phase_factor_tmp(fft().size());


                //gvec_phase_factor_.set_dimensions(fft_.size(), num_atoms());
                //gvec_phase_factor_.allocate();

                //gvec_phase_factor_by_phase_;

                // TODO: parallelize
                //std::map<double,complex16> phase;
               
                // phase factor map for each MPI rank
                std::map<double, complex16> phase_factor_tmp;
               
                for (int ig = spl_fft_size.begin(); ig <= spl_fft_size.end(); ig++)
                {
                    
                    #pragma omp parallel default(shared)
                    {
                        std::map<double, complex16> phase_factor_tmp_pt;
                        #pragma omp for
                        for (int ia = 0; ia < num_atoms(); ia++)
                        {
                            double t = scalar_product(gvec(ig), atom(ia)->position());
                            t -= floor(t);
                            if (!phase_factor_tmp_pt.count(t))
                                phase_factor_tmp_pt[t] = exp(complex16(0.0, twopi * t));
                        }




                    for (int ig = 0; ig < fft().size(); ig++)
                    {
                        double t = scalar_product(&gvec_(0, ig), atom(ia)->position());
                        t -= floor(t);
                        if (!gvec_phase_factor_by_phase_.count(t)) 
                            gvec_phase_factor_by_phase_[t] = exp(complex16(0.0, twopi * t));
                        //exp(complex16(0.0, twopi * scalar_product(&gvec_(0, ig), atom(ia)->position())));
                        //gvec_phase_factor_(ig, ia) = phase[t];                     
                    }
            }
#endif
        }

        void clear()
        {
            fft_.clear();
            gvec_.deallocate();
            gvec_shell_len_.clear();
            gvec_shell_.clear();
            index_by_gvec_.deallocate();
            fft_index_.clear();
            //gvec_phase_factor_.deallocate();
            //gvec_phase_factor_by_phase_.clear();
        }

    public:
        
        ReciprocalLattice() : pw_cutoff_(pw_cutoff_default),
                              num_gvec_(0)
        {
        }
  
        void set_pw_cutoff(double pw_cutoff__)
        {
            pw_cutoff_ = pw_cutoff__;
        }

        void print_info()
        {
            printf("\n");
            printf("plane wave cutoff : %f\n", pw_cutoff());
            printf("number of G-vectors within the cutoff : %i\n", num_gvec());
            printf("number of G-shells : %i\n", num_gvec_shells());
            printf("FFT grid size : %i %i %i   total : %i\n", fft_.size(0), fft_.size(1), fft_.size(2), fft_.size());
            printf("FFT grid limits : %i %i   %i %i   %i %i\n", fft_.grid_limits(0, 0), fft_.grid_limits(0, 1),
                                                                fft_.grid_limits(1, 0), fft_.grid_limits(1, 1),
                                                                fft_.grid_limits(2, 0), fft_.grid_limits(2, 1));
        }
        
        inline FFT3D& fft()
        {
            return fft_;
        }

        inline int index_by_gvec(int i0, int i1, int i2)
        {
            return index_by_gvec_(i0, i1, i2);
        }

        inline int* index_by_gvec()
        {
            return index_by_gvec_.get_ptr();
        }

        inline int fft_index(int ig)
        {
            return fft_index_[ig];
        }

        inline int* fft_index()
        {
            return &fft_index_[0];
        }
        
        inline int* gvec(int ig)
        {
            return &gvec_(0, ig);
        }
        
        /// Plane-wave cutoff for G-vectors
        inline double pw_cutoff()
        {
            return pw_cutoff_;
        }
        
        /// Number of G-vectors within plane-wave cutoff
        inline int num_gvec()
        {
            return num_gvec_;
        }

        inline int num_gvec_shells()
        {
            return (int)gvec_shell_len_.size();
        }
        
        inline double gvec_shell_len(int igs)
        {
            return gvec_shell_len_[igs];
        }
        
        /// index of G-vector shell
        inline int gvec_shell(int ig)
        {
            assert(ig >= 0 && ig < (int)gvec_shell_.size());
            return gvec_shell_[ig];
        }

        /// length of G-vector
        inline double gvec_len(int ig)
        {
            assert(ig >= 0 && ig < (int)gvec_shell_.size());
            return gvec_shell_len_[gvec_shell_[ig]];
        }

        /// phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        inline complex16 gvec_phase_factor(int ig, int ia)
        {
            return exp(complex16(0.0, twopi * scalar_product(gvec(ig), atom(ia)->position())));
        }

        /// return global index of G1-G2 vector
        inline int index_g12(int ig1, int ig2)
        {
            return index_by_gvec_(gvec_(0, ig1) - gvec_(0, ig2),
                                  gvec_(1, ig1) - gvec_(1, ig2),
                                  gvec_(2, ig1) - gvec_(2, ig2));
        }
};

};

#endif

