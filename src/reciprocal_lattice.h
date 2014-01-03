#ifndef __RECIPROCAL_LATTICE_H__
#define __RECIPROCAL_LATTICE_H__

namespace sirius {

class Reciprocal_lattice
{
    private:

        Unit_cell* unit_cell_;
        
        double lattice_vectors_[3][3];

        double reciprocal_lattice_vectors_[3][3];
        
        /// plane wave cutoff radius (in inverse a.u. of length)
        /** Plane-wave cutoff controls the size of the FFT grid used in the interstitial region. */
        double pw_cutoff_;
        
        /// FFT wrapper
        FFT3D<cpu>* fft_;

        /// list of G-vector fractional coordinates
        mdarray<int, 2> gvec_;

        /// number of G-vectors within plane wave cutoff
        int num_gvec_;

        /// mapping between index of a G-shell and a list of G-vectors belonging to the shell 
        //std::vector< std::vector<int> > ig_by_igs_;

        /// mapping between G-vector and shell
        std::vector<int> gvec_shell_;

        /// mapping between linear G-vector index and G-vector coordinates
        mdarray<int, 3> index_by_gvec_;

        /// mapping betwee linear G-vector index and position in FFT buffer
        std::vector<int> fft_index_; // TODO: move to FFT? what to do with sorting of G-vectors?

        /// split index of G-vectors
        splindex<block> spl_num_gvec_;
        
        /// cached Ylm components of G-vectors
        mdarray<complex16, 2> gvec_ylm_;
        
        /// cached values of G-vector phase factors 
        mdarray<complex16, 2> gvec_phase_factors_;

        /// length of G-vectors belonging to the same shell
        std::vector<double> gvec_shell_len_;
        
        void init(int lmax);

    public:
        
        Reciprocal_lattice(Unit_cell* unit_cell__, double pw_cutoff__, int lmax__) 
            : unit_cell_(unit_cell__), 
              pw_cutoff_(pw_cutoff__), 
              num_gvec_(0)
        {
            for (int l = 0; l < 3; l++)
            {
                for (int x = 0; x < 3; x++)
                {
                    lattice_vectors_[l][x] = unit_cell_->lattice_vectors(l, x);
                    reciprocal_lattice_vectors_[l][x] = unit_cell_->reciprocal_lattice_vectors(l, x);
                }
            }

            vector3d<int> max_frac_coord = Utils::find_translation_limits(pw_cutoff_, reciprocal_lattice_vectors_);
            fft_ = new FFT3D<cpu>(max_frac_coord);

            init(lmax__);
        }

        ~Reciprocal_lattice()
        {
            delete fft_;
        }
  
        void update();

        /// Print basic info
        void print_info();
        
        /// Index of G-vector shell
        template <index_domain_t index_domain>
        inline int gvec_shell(int ig);
        
        /// Phase factors \f$ e^{i {\bf G} {\bf r}_{\alpha}} \f$
        template <index_domain_t index_domain>
        inline complex16 gvec_phase_factor(int ig, int ia);
       
        /// Ylm components of G-vector
        template <index_domain_t index_domain>
        inline void gvec_ylm_array(int ig, complex16* ylm, int lmax);

        /// Make periodic function out of form factors
        /** Return vector of plane-wave coefficients */
        std::vector<complex16> make_periodic_function(mdarray<double, 2>& ffac, int ngv);

        inline FFT3D<cpu>* fft()
        {
            return fft_;
        }

        inline int index_by_gvec(int i0, int i1, int i2)
        {
            return index_by_gvec_(i0, i1, i2);
        }

        /// FFT index for a given  G-vector index
        inline int fft_index(int ig)
        {
            return fft_index_[ig];
        }

        /// Pointer to FFT index array
        inline int* fft_index()
        {
            return &fft_index_[0];
        }
        
        /// G-vector in integer fractional coordinates
        inline vector3d<int> gvec(int ig)
        {
            return vector3d<int>(gvec_(0, ig), gvec_(1, ig), gvec_(2, ig));
        }
        
        inline double gvec_shell_len(int igs)
        {
            assert(igs >=0 && igs < (int)gvec_shell_len_.size());
            return gvec_shell_len_[igs];
        }
        
        /// Length of G-vector.
        inline double gvec_len(int ig)
        {
            return gvec_shell_len(gvec_shell_[ig]);
        }
        
        inline vector3d<double> get_fractional_coordinates(vector3d<double> a)
        {
            vector3d<double> b;
            for (int l = 0; l < 3; l++)
            {
                for (int x = 0; x < 3; x++) b[l] += lattice_vectors_[l][x] * a[x] / twopi;
            }
            return b;
        }
        
        template <typename T>
        inline vector3d<double> get_cartesian_coordinates(vector3d<T> a)
        {
            vector3d<double> b;
            for (int x = 0; x < 3; x++)
            {
                for (int l = 0; l < 3; l++) b[x] += a[l] * reciprocal_lattice_vectors_[l][x];
            }
            return b;
        }

        /// G-vector in Cartesian coordinates
        inline vector3d<double> gvec_cart(int ig)
        {
            return get_cartesian_coordinates(gvec(ig));
        }
        
        /// Number of G-vectors within plane-wave cutoff
        inline int num_gvec()
        {
            return num_gvec_;
        }

        /// Number of G-vector shells within plane-wave cutoff
        inline int num_gvec_shells_inner()
        {
            return gvec_shell_[num_gvec_];
        }

        inline int num_gvec_shells_total()
        {
            return (int)gvec_shell_len_.size();
        }
        
        /// Return global index of G1-G2 vector
        inline int index_g12(int ig1, int ig2)
        {
            return index_by_gvec_(gvec_(0, ig1) - gvec_(0, ig2),
                                  gvec_(1, ig1) - gvec_(1, ig2),
                                  gvec_(2, ig1) - gvec_(2, ig2));
        }
        
        inline int index_g12_safe(int ig1, int ig2)
        {
            vector3d<int> v(gvec_(0, ig1) - gvec_(0, ig2), gvec_(1, ig1) - gvec_(1, ig2), gvec_(2, ig1) - gvec_(2, ig2));
            if (v[0] >= fft_->grid_limits(0).first && v[0] <= fft_->grid_limits(0).second &&
                v[1] >= fft_->grid_limits(1).first && v[1] <= fft_->grid_limits(1).second &&
                v[2] >= fft_->grid_limits(2).first && v[2] <= fft_->grid_limits(2).second)
            {
                return index_by_gvec(v[0], v[1], v[2]);
            }
            else
            {
                return -1;
            }
        }

        inline splindex<block>& spl_num_gvec()
        {
            return spl_num_gvec_;
        }
        
        inline int spl_num_gvec(int igloc)
        {
            return spl_num_gvec_[igloc];
        }
        
        inline complex16 gvec_ylm(int lm, int igloc)
        {
            return gvec_ylm_(lm, igloc);
        }

        //inline int igs_size(int igs)
        //{
        //    return (int)ig_by_igs_.size();
        //}

        //inline std::vector<int>& ig_by_igs(int igs)
        //{
        //    return ig_by_igs_[igs];
        //}
};

#include "reciprocal_lattice.hpp"

};

#endif

