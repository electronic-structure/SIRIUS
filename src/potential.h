
namespace sirius {

/// Generate effective potential from charge density and magnetization
/** \note At some point we need to update the atomic potential with the new MT potential. This is simple if the 
          effective potential is a global function. Otherwise we need to pass the effective potential between MPI ranks.
          This is also simple, but requires some time. It is also easier to mix the global functions.  */
class Potential 
{
    private:
        
        Global& parameters_;

        /// alias for FFT driver
        FFT3D<cpu>* fft_;

        Periodic_function<double>* effective_potential_;

        Periodic_function<double>* effective_magnetic_field_[3];
 
        Periodic_function<double>* coulomb_potential_;
        Periodic_function<double>* xc_potential_;
        Periodic_function<double>* xc_energy_density_;
        
        /// local part of pseudopotential
        Periodic_function<double>* local_potential_;

        mdarray<double, 3> sbessel_mom_;

        mdarray<double, 3> sbessel_mt_;

        mdarray<double, 2> gamma_factors_R_;

        int lmax_;
        
        SHT* sht_;

        int pseudo_density_order;

        std::vector<complex16> zil_;
        
        std::vector<complex16> zilm_;

        std::vector<int> l_by_lm_;

        /// Compute MT part of the potential and MT multipole moments
        void poisson_vmt(mdarray<Spheric_function<complex16>*, 1>& rho_ylm, mdarray<Spheric_function<complex16>*, 1>& vh, 
                         mdarray<complex16, 2>& qmt);

        /// Compute multipole momenst of the interstitial charge density
        /** Also, compute the MT boundary condition 
        */
        void poisson_sum_G(complex16* fpw, mdarray<double, 3>& fl, mdarray<complex16, 2>& flm);
        
        /// Add contribution from the pseudocharge to the plane-wave expansion
        void poisson_add_pseudo_pw(mdarray<complex16, 2>& qmt, mdarray<complex16, 2>& qit, complex16* rho_pw);

        void generate_local_potential();

    public:

        /// Constructor
        Potential(Global& parameters__);

        ~Potential();

        void update();

        void set_effective_potential_ptr(double* veffmt, double* veffit);
        
        void set_effective_magnetic_field_ptr(double* beffmt, double* beffit);
         
        /// Zero effective potential and magnetic field.
        void zero();

        /// Poisson solver
        /** Plane wave expansion
            \f[
                e^{i{\bf g}{\bf r}}=4\pi e^{i{\bf g}{\bf r}_{\alpha}} \sum_{\ell m} i^\ell 
                    j_{\ell}(g|{\bf r}-{\bf r}_{\alpha}|)
                    Y_{\ell m}^{*}({\bf \hat g}) Y_{\ell m}(\widehat{{\bf r}-{\bf r}_{\alpha}})
            \f]

            Multipole moment:
            \f[
                q_{\ell m} = \int Y_{\ell m}^{*}(\hat {\bf r}) r^l \rho({\bf r}) d {\bf r}

            \f]

            Spherical Bessel function moments
            \f[
                \int_0^R j_{\ell}(a x)x^{2+\ell} dx = \frac{\sqrt{\frac{\pi }{2}} R^{\ell+\frac{3}{2}} 
                    J_{\ell+\frac{3}{2}}(a R)}{a^{3/2}}
            \f]
            for a = 0 the integral is \f$ \frac{R^3}{3} \delta_{\ell,0} \f$

            General solution to the Poisson equation with spherical boundary condition:
            \f[
                V({\bf x}) = \int \rho({\bf x'})G({\bf x},{\bf x'}) d{\bf x'} - \frac{1}{4 \pi} \int_{S} V({\bf x'}) 
                    \frac{\partial G}{\partial n'} d{\bf S'}
            \f]

            Green's function for a sphere
            \f[
                G({\bf x},{\bf x'}) = 4\pi \sum_{\ell m} \frac{Y_{\ell m}^{*}(\hat {\bf x'}) 
                    Y_{\ell m}(\hat {\bf x})}{2\ell + 1}
                    \frac{r_{<}^{\ell}}{r_{>}^{\ell+1}}\Biggl(1 - \Big( \frac{r_{>}}{R} \Big)^{2\ell + 1} \Biggr)
            \f]

            Pseudodensity radial functions:
            \f[
                p_{\ell}(r) = r^{\ell} \left(1-\frac{r^2}{R^2}\right)^n
            \f]
            where n is the order of pseudo density.

        */
        void poisson(Periodic_function<double>* rho, Periodic_function<double>* vh);
        
        /// Generate XC potential and energy density
        void xc(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3], 
                Periodic_function<double>* vxc, Periodic_function<double>* bxc[3], Periodic_function<double>* exc);
        
        /// Generate effective potential and magnetic field from charge density and magnetization.
        void generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* magnetization[3]);
        
        void generate_effective_potential(Periodic_function<double>* rho, Periodic_function<double>* rho_core, 
                                          Periodic_function<double>* magnetization[3]);

        void save();
        
        void load();
        
        void update_atomic_potential();
        
        template <processing_unit_t pu> 
        void add_mt_contribution_to_pw();

        /// Generate plane-wave coefficients of the potential in the interstitial region
        void generate_pw_coefs();

        void generate_d_mtrx();

        #ifdef _GPU_
        void generate_d_mtrx_gpu();
        #endif

        double value(double* vc);

        void check_potential_continuity_at_mt();

        void copy_to_global_ptr(double* fmt, double* fit, Periodic_function<double>* src);
        
        inline size_t size()
        {
            size_t s = effective_potential_->size();
            for (int i = 0; i < parameters_.num_mag_dims(); i++) s += effective_magnetic_field_[i]->size();
            return s;
        }

        inline void pack(double* buffer)
        {
            size_t n = effective_potential_->pack(buffer);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += effective_magnetic_field_[i]->pack(&buffer[n]);
        }

        inline void unpack(double* buffer)
        {
            size_t n = effective_potential_->unpack(buffer);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += effective_magnetic_field_[i]->unpack(&buffer[n]);
        }

        //void copy_xc_potential(double* vxcmt, double* vxcir);

        //void copy_effective_magnetic_field(double* beffmt, double* beffit);
        
        Periodic_function<double>* effective_potential()
        {
            return effective_potential_;
        }

        Spheric_function<double>& effective_potential_mt(int ialoc)
        {
            return effective_potential_->f_mt(ialoc);
        }

        Periodic_function<double>** effective_magnetic_field()
        {
            return effective_magnetic_field_;
        }
        
        Periodic_function<double>* effective_magnetic_field(int i)
        {
            return effective_magnetic_field_[i];
        }

        Periodic_function<double>* coulomb_potential()
        {
            return coulomb_potential_;
        }
        
        Spheric_function<double>& coulomb_potential_mt(int ialoc)
        {
            return coulomb_potential_->f_mt(ialoc);
        }
        
        Periodic_function<double>* xc_potential()
        {
            return xc_potential_;
        }

        Periodic_function<double>* xc_energy_density()
        {
            return xc_energy_density_;
        }
        
        void allocate()
        {
            effective_potential_->allocate(true, true);
            for (int j = 0; j < parameters_.num_mag_dims(); j++) effective_magnetic_field_[j]->allocate(true, true);
        }
};

#include "potential.hpp"

};


