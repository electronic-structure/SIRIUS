#ifndef __RECIPROCAL_LATTICE_H__
#define __RECIPROCAL_LATTICE_H__

namespace sirius {

class Reciprocal_lattice : public Unit_cell
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

        void init(int lmax);
        
        void clear();

    public:
        
        Reciprocal_lattice() : pw_cutoff_(pw_cutoff_default), num_gvec_(0)
        {
        }
  
        void print_info();
        
        /// index of G-vector shell
        template <index_domain_t index_domain>
        inline int gvec_shell(int ig);
        
        /// length of G-vector
        inline double gvec_len(int ig);
        
        /// phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        template <index_domain_t index_domain>
        inline complex16 gvec_phase_factor(int ig, int ia);
        
        template <index_domain_t index_domain>
        inline void gvec_ylm_array(int ig, complex16* ylm, int lmax);

        void set_pw_cutoff(double pw_cutoff__)
        {
            pw_cutoff_ = pw_cutoff__;
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

        inline int igs_size(int igs)
        {
            return (int)ig_by_igs_.size();
        }

        inline std::vector<int>& ig_by_igs(int igs)
        {
            return ig_by_igs_[igs];
        }
};

#include "reciprocal_lattice.hpp"

};

#endif

