/** \file density.h
    
    \brief Contains definition and partial implementation of sirius::Density class.
*/

namespace sirius
{


/// Generate charge density and magnetization from occupied spinor wave-functions.
/** Let's start from the definition of the complex density matrix:
    \f[
        \rho_{\sigma' \sigma}({\bf r}) =
         \sum_{j{\bf k}} n_{j{\bf k}} \Psi_{j{\bf k}}^{\sigma*}({\bf r}) \Psi_{j{\bf k}}^{\sigma'}({\bf r}) = 
         \frac{1}{2} \left( \begin{array}{cc} \rho({\bf r})+m_z({\bf r}) & 
                m_x({\bf r})-im_y({\bf r}) \\ m_x({\bf r})+im_y({\bf r}) & \rho({\bf r})-m_z({\bf r}) \end{array} \right)
    \f]
    We notice that the diagonal components of the density matrix are actually real and the off-diagonal components are
    expressed trough two independent functions \f$ m_x({\bf r}) \f$ and \f$ m_y({\bf r}) \f$. Having this in mind we 
    will work with a slightly different object, namely a real density matrix, defined as a 1-, 2- or 4-dimensional 
    (depending on the number of magnetic components) vector with the following elements: 
        - \f$ [ \rho({\bf r}) ] \f$ in case of non-magnetic configuration
        - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}) ]  = 
              [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2} ] \f$ in case of collinear 
           magnetic configuration
        - \f$ [ \rho_{\uparrow \uparrow}({\bf r}), \rho_{\downarrow \downarrow}({\bf r}), 
                2 \Re \rho_{\uparrow \downarrow}({\bf r}), -2 \Im \rho_{\uparrow \downarrow}({\bf r}) ] = 
              [ \frac{\rho({\bf r})+m_z({\bf r})}{2}, \frac{\rho({\bf r})-m_z({\bf r})}{2}, 
                m_x({\bf r}),  m_y({\bf r}) ] \f$ in the general case of non-collinear magnetic configuration
    
    At this point it is straightforward to compute the density and magnetization in the interstitial (see add_kpoint_contribution_it()).
    The muffin-tin part of the density and magnetization is obtained in a slighlty more complicated way. Recall the
    expansion of spinor wave-functions inside the muffin-tin \f$ \alpha \f$
    \f[
        \Psi_{j{\bf k}}^{\sigma}({\bf r}) = \sum_{\xi}^{N_{\xi}^{\alpha}} {S_{\xi}^{\sigma j {\bf k},\alpha}} 
        f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}(\hat {\bf r})
    \f]
    which we insert into expression for the complex density matrix: 
    \f[
        \rho_{\sigma' \sigma}({\bf r}) = \sum_{j{\bf k}} n_{j{\bf k}} \sum_{\xi}^{N_{\xi}^{\alpha}} 
            S_{\xi}^{\sigma j {\bf k},\alpha*} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)
            Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) \sum_{\xi'}^{N_{\xi'}^{\alpha}} S_{\xi'}^{\sigma' j{\bf k},\alpha}
            f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
    \f]
    First, we eliminate a sum over bands and k-points by forming an auxiliary density tensor:
    \f[
        D_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}} S_{\xi}^{\sigma j {\bf k},\alpha*} 
            S_{\xi'}^{\sigma' j {\bf k},\alpha}
    \f]
    The expression for complex density matrix simplifies to:
    \f[
        \rho_{\sigma' \sigma}({\bf r}) =  \sum_{\xi \xi'} D_{\xi \sigma, \xi' \sigma'}^{\alpha} 
            f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) 
            f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r})
    \f]
    Now we can switch to the real density matrix and write its' expansion in real spherical harmonics. Let's take
    non-magnetic case as an example:
    \f[
        \rho({\bf r}) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha} 
            f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r)Y_{\ell_{\xi}m_{\xi}}^{*}(\hat {\bf r}) 
            f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r)Y_{\ell_{\xi'}m_{\xi'}}(\hat {\bf r}) = 
            \sum_{\ell_3 m_3} \rho_{\ell_3 m_3}^{\alpha}(r) R_{\ell_3 m_3}(\hat {\bf r}) 
    \f]
    where
    \f[
        \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\xi \xi'} D_{\xi \xi'}^{\alpha} f_{\ell_{\xi} \lambda_{\xi}}^{\alpha}(r) 
            f_{\ell_{\xi'} \lambda_{\xi'}}^{\alpha}(r) \langle Y_{\ell_{\xi}m_{\xi}} | R_{\ell_3 m_3} | Y_{\ell_{\xi'}m_{\xi'}} \rangle
    \f]
    We are almost done. Now it is time to switch to the full index notation  \f$ \xi \rightarrow \{ \ell \lambda m \} \f$
    and sum over \a m and \a m' indices:
    \f[
         \rho_{\ell_3 m_3}^{\alpha}(r) = \sum_{\ell \lambda, \ell' \lambda'} f_{\ell \lambda}^{\alpha}(r)  
            f_{\ell' \lambda'}^{\alpha}(r) d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} 
    \f]
    where
    \f[
        d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
            \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
            \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
    \f]
    This is our final answer: radial components of density and magnetization are expressed as a linear combination of
    quadratic forms in radial functions. 

    \note density and potential are allocated as global function because it's easier to load and save them.
*/
class Density
{
    private:
        
        /// global set of parameters
        Global& parameters_;

        /// alias for FFT driver
        FFT3D<cpu>* fft_;
        
        /// pointer to charge density
        /** In the case of full-potential calculation this is the full (valence + core) electron charge density.
            In the case of pseudopotential this is the valence charge density. */ 
        Periodic_function<double>* rho_;

        /// pointer to pseudo core charge density
        /** In the case of pseudopotential we need to know the non-linear core correction to the exchange-correlation 
            energy which is introduced trough the pseudo core density: \f$ E_{xc}[\rho_{val} + \rho_{core}] \f$. The 
            'pseudo core' reflects the fact that this density integrated does not reproduce the total number of core 
            elctrons. */
        Periodic_function<double>* rho_pseudo_core_;
        
        Periodic_function<double>* magnetization_[3];
        
        std::vector< std::pair<int, int> > dmat_spins_;

        /// non-zero Gaunt coefficients
        Gaunt_coefficients<complex16>* gaunt_coefs_;
        
        /// fast mapping between composite lm index and corresponding orbital quantum number
        std::vector<int> l_by_lm_;

        /// Get the local list of occupied bands
        /** Initially bands are distributed over k-points and columns of the MPI grid used for the diagonalization.
            Additionaly bands are sub split over rows of the 2D MPI grid, so each MPI rank in the total MPI grid gets
            it's local fraction of the bands.
        */
        std::vector< std::pair<int, double> > get_occupied_bands_list(Band* band, K_point* kp);

        /// Reduce complex density matrix over magnetic quantum numbers
        /** The following operation is performed:
            \f[
                d_{\ell \lambda, \ell' \lambda', \ell_3 m_3}^{\alpha} = 
                    \sum_{mm'} D_{\ell \lambda m, \ell' \lambda' m'}^{\alpha} 
                    \langle Y_{\ell m} | R_{\ell_3 m_3} | Y_{\ell' m'} \rangle
            \f]
        */
        template <int num_mag_dims> 
        void reduce_zdens(Atom_type* atom_type, int ialoc, mdarray<complex16, 4>& zdens, mdarray<double, 3>& mt_density_matrix);
        
        /// Add k-point contribution to the auxiliary muffin-tin density matrix
        /** Complex density matrix has the following expression:
            \f[
                D_{\xi \sigma, \xi' \sigma'}^{\alpha} = \sum_{j{\bf k}} n_{j{\bf k}} S_{\xi}^{\sigma j {\bf k},\alpha*} 
                    S_{\xi'}^{\sigma' j {\bf k},\alpha}
            \f]
        
            In case of LDA+U the occupation matrix is also computed. It has the following expression:
            \f[
                n_{\ell,mm'}^{\sigma \sigma'} = \sum_{i {\bf k}}^{occ} \int_{0}^{R_{MT}} r^2 dr 
                          \Psi_{\ell m}^{i{\bf k}\sigma *}({\bf r}) \Psi_{\ell m'}^{i{\bf k}\sigma'}({\bf r})
            \f] 
        */
        void add_kpoint_contribution_mt(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                        mdarray<complex16, 4>& mt_complex_density_matrix);
        
        /// Add k-point contribution to the interstitial density and magnetization
        void add_kpoint_contribution_it(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands);
        
        /// Add k-point contribution to the density matrix in case of ultrasoft pseudo-potential
        /** The following density matrix has to be computed for each atom:
            \f[
                d_{\xi \xi'}^{\alpha} = \langle \beta_{\xi}^{\alpha} | \hat N | \beta_{\xi'}^{\alpha} \rangle = 
                  \sum_{j {\bf k}} \langle \beta_{\xi}^{\alpha} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}} 
                  \langle \Psi_{j{\bf k}} | \beta_{\xi'}^{\alpha} \rangle
            \f]
            Here \f$ \hat N = \sum_{j{\bf k}} | \Psi_{j{\bf k}} \rangle n_{j{\bf k}} \langle \Psi_{j{\bf k}} | \f$ is 
            the occupancy operator written in spectral representation. 
        */
        void add_kpoint_contribution_pp(K_point* kp, std::vector< std::pair<int, double> >& occupied_bands, 
                                        mdarray<complex16, 4>& pp_complex_density_matrix);
        
        void add_q_contribution_to_valence_density(K_set& kset);

        /// Generate valence density in the muffin-tins 
        void generate_valence_density_mt(K_set& ks);
        
        /// Generate valence density in the muffin-tins using straightforward (slow) approach
        //** template <processing_unit_t pu> 
        //** void generate_valence_density_mt_directly(K_set& ks);
        
        void generate_valence_density_mt_sht(K_set& ks);
        
        /// Generate valence density in the interstitial
        void generate_valence_density_it(K_set& ks);
       
        /// Add band contribution to the muffin-tin density
        void add_band_contribution_mt(Band* band, double weight, mdarray<complex16, 3>& fylm, 
                                      std::vector<Periodic_function<double>*>& dens);
        
        /// Generate charge density of core states
        void generate_core_charge_density();

        void generate_pseudo_core_charge_density();

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
        void initial_density();

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

        inline size_t size()
        {
            size_t s = rho_->size();
            for (int i = 0; i < parameters_.num_mag_dims(); i++) s += magnetization_[i]->size();
            return s;
        }

        inline void pack(double* buffer)
        {
            size_t n = rho_->pack(buffer);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += magnetization_[i]->pack(&buffer[n]);
        }

        inline void unpack(double* buffer)
        {
            size_t n = rho_->unpack(buffer);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += magnetization_[i]->unpack(&buffer[n]);
        }
        
        Periodic_function<double>* rho()
        {
            return rho_;
        }
        
        Periodic_function<double>* rho_pseudo_core()
        {
            return rho_pseudo_core_;
        }
        
        Periodic_function<double>** magnetization()
        {
            return magnetization_;
        }

        Periodic_function<double>* magnetization(int i)
        {
            return magnetization_[i];
        }

        Spheric_function<double>& density_mt(int ialoc)
        {
            return rho_->f_mt(ialoc);
        }

        void allocate()
        {
            rho_->allocate(true, true);
            for (int j = 0; j < parameters_.num_mag_dims(); j++) magnetization_[j]->allocate(true, true);
        }
};

#include "density.hpp"

};
