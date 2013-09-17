namespace sirius
{

class Density
{
    private:
        
        /// Global set of parameters
        Global& parameters_;

        Periodic_function<double>* rho_;
        
        Periodic_function<double>* magnetization_[3];
        
        std::vector< std::pair<int, int> > dmat_spins_;

        // TODO: clean Gaunt arrays (should be one)
        mdarray< std::vector< std::vector< std::pair<int, complex16> > >, 2> complex_gaunt_;

        GauntCoefficients gaunt12_;

        std::vector<int> l_by_lm_;

        /// Get the local list of occupied bands
        /** Initially bands are distributed over k-points and columns of the MPI grid used for the diagonalization.
            Additionaly bands are sub split over rows of the 2D MPI grid, so each MPI rank in the total MPI grid gets
            it's local fraction of the bands.
        */
        void get_occupied_bands_list(Band* band, K_point* kp, std::vector< std::pair<int, double> >& bands);

        /// Reduce complex density matrix over magnetic quantum numbers
        template <int num_mag_dims> 
        void reduce_zdens(int ia, int ialoc, mdarray<complex16, 4>& zdens, mdarray<double, 4>& mt_density_matrix);
        
        /// Add k-point contribution to the auxiliary muffin-tin density matrix
        /** In case of LDA+U the occupation matrix is also computed. It has the following expression:
            \f[
                n_{\ell,mm'}^{\sigma \sigma'} = \sum_{i {\bf k}}^{occ} \int_{0}^{R_{MT}} r^2 dr 
                          \Psi_{\ell m}^{i{\bf k}\sigma *}({\bf r}) \Psi_{\ell m'}^{i{\bf k}\sigma'}({\bf r})
            \f] 
        */
        void add_kpoint_contribution_mt(Band* band, K_point* kp, mdarray<complex16, 4>& mt_complex_density_matrix);
        
        /// Add k-point contribution to the interstitial density and magnetization
        void add_kpoint_contribution_it(Band* band, K_point* kp);
        
        /// Generate valence density in the muffin-tins 
        void generate_valence_density_mt(K_set& ks);
        
        /// Generate valence density in the muffin-tins using straightforward (slow) approach
        template <processing_unit_t pu> 
        void generate_valence_density_mt_directly(K_set& ks);
        
        void generate_valence_density_mt_sht(K_set& ks);
        
        /// Generate valence density in the interstitial
        void generate_valence_density_it(K_set& ks);
       
        /// Add band contribution to the muffin-tin density
        void add_band_contribution_mt(Band* band, double weight, mdarray<complex16, 3>& fylm, 
                                      std::vector<Periodic_function<double>*>& dens);

    public:

        /// Constructor
        Density(Global& parameters__);
        
        /// Destructor
        ~Density();
       
        /// Set pointers to muffin-tin and interstitial charge density arrays
        void set_charge_density_ptr(double* rhomt, double* rhoir);
        
        /// Set pointers to muffin-tin and interstitial magnetization arrays
        void set_magnetization_ptr(double* magmt, double* magir);
        
        /// Zero density and magnetization
        void zero();
        
        /// Generate initial charge density and magnetization
        void initial_density(int type);

        /// Find the total leakage of the core states out of the muffin-tins
        double core_leakage();
        
        /// Return core leakage for a specific atom symmetry class
        double core_leakage(int ic);

        /// Generate charge density and magnetization from the wave functions
        void generate(K_set& ks);
        
        /// Integrtae charge density to get total and partial charges
        //** void integrate();

        /// Check density at MT boundary
        void check_density_continuity_at_mt();
         
        void save();
        
        void load();
        
        Periodic_function<double>* rho()
        {
            return rho_;
        }
        
        Periodic_function<double>** magnetization()
        {
            return magnetization_;
        }

        Periodic_function<double>* magnetization(int i)
        {
            return magnetization_[i];
        }

        MT_function<double>* density_mt(int ialoc)
        {
            return rho_->f_mt(ialoc);
        }

        void allocate()
        {
            rho_->allocate(true);
            for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->allocate(true);
        }
};

#include "density.hpp"

};
