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
        mdarray<int, 2> gvec_;

        /// number of G-vectors within plane wave cutoff
        int num_gvec_;

        /// mapping between index of a G-shell and a list of G-vectors belonging to the shell 
        std::vector< std::vector<int> > ig_by_igs_;

        /// length of G-vectors belonging to the same shell
        std::vector<double> gvec_shell_len_;

        /// mapping between G-vector and shell
        std::vector<int> gvec_shell_;

        /// mapping between linear G-vector index and G-vector coordinates
        mdarray<int, 3> index_by_gvec_;

        /// mapping betwee linear G-vector index and position in FFT buffer
        std::vector<int> fft_index_;

        /// split index of G-vectors
        splindex<block> spl_num_gvec_;
        
        /// split index of FFT buffer
        splindex<block> spl_fft_size_;
        
        /// Ylm components of G-vectors
        mdarray<complex16, 2> gvec_ylm_;
        
        /// cached values of G-vector phase factors 
        mdarray<complex16, 2> gvec_phase_factors_;

    protected:

        void init(int lmax)
        {
            Timer t("sirius::ReciprocalLattice::init");
            
            int max_frac_coord[3];
            find_translation_limits<reciprocal>(pw_cutoff(), max_frac_coord);
            
            fft_.init(max_frac_coord);
            
            mdarray<int, 2> gvec_tmp(3, fft_.size());
            std::vector< std::pair<double, int> > gvec_tmp_length;

            int ig = 0;
            for (int i0 = fft_.grid_limits(0, 0); i0 <= fft_.grid_limits(0, 1); i0++)
            {
                for (int i1 = fft_.grid_limits(1, 0); i1 <= fft_.grid_limits(1, 1); i1++)
                {
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
                }
            }

            std::sort(gvec_tmp_length.begin(), gvec_tmp_length.end());

            // create sorted list of G-vectors
            gvec_.set_dimensions(3, fft_.size());
            gvec_.allocate();

            num_gvec_ = 0;
            for (int i = 0; i < fft_.size(); i++)
            {
                for (int j = 0; j < 3; j++) gvec_(j, i) = gvec_tmp(j, gvec_tmp_length[i].second);
                
                if (gvec_tmp_length[i].first <= pw_cutoff_) num_gvec_++;
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

                if (gvec_shell_len_.empty() || fabs(t - gvec_shell_len_.back()) > 1e-8) gvec_shell_len_.push_back(t);
                 
                gvec_shell_[ig] = (int)gvec_shell_len_.size() - 1;
            }

            ig_by_igs_.clear();
            ig_by_igs_.resize(num_gvec_shells());
            for (int ig = 0; ig < num_gvec_; ig++)
            {
                int igs = gvec_shell_[ig];
                ig_by_igs_[igs].push_back(ig);
            }
            
            // create split index
            spl_num_gvec_.split(num_gvec(), Platform::num_mpi_ranks(), Platform::mpi_rank());

            spl_fft_size_.split(fft().size(), Platform::num_mpi_ranks(), Platform::mpi_rank());
            
            // precompute spherical harmonics of G-vectors and G-vector phase factors
            gvec_ylm_.set_dimensions(Utils::lmmax_by_lmax(lmax), spl_num_gvec_.local_size());
            gvec_ylm_.allocate();
            
            gvec_phase_factors_.set_dimensions(spl_num_gvec_.local_size(), num_atoms());
            gvec_phase_factors_.allocate();
            
            for (int igloc = 0; igloc < spl_num_gvec_.local_size(); igloc++)
            {
                int ig = spl_num_gvec_[igloc];
                double xyz[3];
                double rtp[3];
                get_coordinates<cartesian, reciprocal>(gvec(ig), xyz);
                SHT::spherical_coordinates(xyz, rtp);
                SHT::spherical_harmonics(lmax, rtp[1], rtp[2], &gvec_ylm_(0, igloc));

                for (int ia = 0; ia < num_atoms(); ia++) 
                    gvec_phase_factors_(igloc, ia) = gvec_phase_factor<global>(ig, ia);
            }
        }

        void clear()
        {
            fft_.clear();
            gvec_.deallocate();
            gvec_shell_len_.clear();
            gvec_shell_.clear();
            index_by_gvec_.deallocate();
            fft_index_.clear();
        }

    public:
        
        ReciprocalLattice() : pw_cutoff_(pw_cutoff_default), num_gvec_(0)
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

        // TODO: call it everywhere
        inline void gvec_cart(int ig, double vgc[3])
        {
            get_coordinates<cartesian, reciprocal>(gvec(ig), vgc);
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
        template <index_domain_t index_domain>
        inline int gvec_shell(int ig)
        {
            switch (index_domain)
            {
                case global:
                {
                    assert(ig >= 0 && ig < (int)gvec_shell_.size());
                    return gvec_shell_[ig];
                    break;
                }
                case local:
                {
                    return gvec_shell_[spl_num_gvec_[ig]];
                    break;
                }
            }
        }

        /// length of G-vector
        inline double gvec_len(int ig)
        {
            //assert(ig >= 0 && ig < (int)gvec_shell_.size());
            if (ig < (int)gvec_shell_.size())
            {
                return gvec_shell_len_[gvec_shell_[ig]];
            }
            else
            {
                double vgc[3];
                gvec_cart(ig, vgc);
                return Utils::vector_length(vgc);
            }
        }
        
        /// phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        template <index_domain_t index_domain>
        inline complex16 gvec_phase_factor(int ig, int ia)
        {
            switch (index_domain)
            {
                case global:
                {
                    return exp(complex16(0.0, twopi * Utils::scalar_product(gvec(ig), atom(ia)->position())));
                    break;
                }
                case local:
                {
                    return gvec_phase_factors_(ig, ia);
                    break;
                }
            }
        }

        /// Return global index of G1-G2 vector
        inline int index_g12(int ig1, int ig2)
        {
            return index_by_gvec_(gvec_(0, ig1) - gvec_(0, ig2),
                                  gvec_(1, ig1) - gvec_(1, ig2),
                                  gvec_(2, ig1) - gvec_(2, ig2));
        }

        inline splindex<block>& spl_num_gvec()
        {
            return spl_num_gvec_;
        }
        
        inline int spl_num_gvec(int igloc)
        {
            return spl_num_gvec_[igloc];
        }
        
        inline splindex<block>& spl_fft_size()
        {
            return spl_fft_size_;
        }

        inline int spl_fft_size(int i)
        {
            return spl_fft_size_[i];
        }

        inline complex16 gvec_ylm(int lm, int igloc)
        {
            return gvec_ylm_(lm, igloc);
        }

        template <index_domain_t index_domain>
        inline void gvec_ylm_array(int ig, complex16* ylm, int lmax)
        {
            switch (index_domain)
            {
                case local:
                {
                    int lmmax = Utils::lmmax_by_lmax(lmax);
                    assert(lmmax <= gvec_ylm_.size(0));
                    memcpy(ylm, &gvec_ylm_(0, ig), lmmax * sizeof(complex16));
                    return;
                }
                case global:
                {
                    double vgc[3];
                    gvec_cart(ig, vgc);
                    double rtp[3];
                    SHT::spherical_coordinates(vgc, rtp);
                    SHT::spherical_harmonics(lmax, rtp[1], rtp[2], ylm);
                    return;
                }
            }
        }

        inline int igs_size(int igs)
        {
            return (int)ig_by_igs_.size();
        }

        inline std::vector<int>& ig_by_igs(int igs)
        {
            return ig_by_igs_[igs];
        }
};

};

#endif

