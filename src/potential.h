#ifndef __POTENTIAL_H__
#define __POTENTIAL_H__

#include "global.h"
#include "periodic_function.h"
#include "spheric_function.h"

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

        std::vector<double_complex> zil_;
        
        std::vector<double_complex> zilm_;

        std::vector<int> l_by_lm_;

        /// Compute MT part of the potential and MT multipole moments
        void poisson_vmt(std::vector< Spheric_function<double_complex> >& rho_ylm, 
                         std::vector< Spheric_function<double_complex> >& vh, 
                         mdarray<double_complex, 2>& qmt);

        /// Compute multipole momenst of the interstitial charge density
        /** Also, compute the MT boundary condition 
        */
        void poisson_sum_G(double_complex* fpw, mdarray<double, 3>& fl, mdarray<double_complex, 2>& flm);
        
        /// Add contribution from the pseudocharge to the plane-wave expansion
        void poisson_add_pseudo_pw(mdarray<double_complex, 2>& qmt, mdarray<double_complex, 2>& qit, double_complex* rho_pw);

        void generate_local_potential();
        
        void xc_mt_nonmagnetic(Radial_grid& rgrid,
                               std::vector<XC_functional*>& xc_func,
                               Spheric_function<double>& rho_lm,
                               Spheric_function<double>& rho_tp,
                               Spheric_function<double>& vxc_tp, 
                               Spheric_function<double>& exc_tp);

        void xc_mt_magnetic(Radial_grid& rgrid, 
                            std::vector<XC_functional*>& xc_func,
                            Spheric_function<double>& rho_up_lm, 
                            Spheric_function<double>& rho_up_tp, 
                            Spheric_function<double>& rho_dn_lm, 
                            Spheric_function<double>& rho_dn_tp, 
                            Spheric_function<double>& vxc_up_tp, 
                            Spheric_function<double>& vxc_dn_tp, 
                            Spheric_function<double>& exc_tp);

        void xc_mt(Periodic_function<double>* rho, 
                   Periodic_function<double>* magnetization[3], 
                   std::vector<XC_functional*>& xc_func,
                   Periodic_function<double>* vxc, 
                   Periodic_function<double>* bxc[3], 
                   Periodic_function<double>* exc);
    
        void xc_it_nonmagnetic(Periodic_function<double>* rho, 
                               std::vector<XC_functional*>& xc_func,
                               Periodic_function<double>* vxc, 
                               Periodic_function<double>* exc);

        void xc_it_magnetic(Periodic_function<double>* rho, 
                            Periodic_function<double>* magnetization[3], 
                            std::vector<XC_functional*>& xc_func,
                            Periodic_function<double>* vxc, 
                            Periodic_function<double>* bxc[3], 
                            Periodic_function<double>* exc);
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
        /** In case of spin-unpolarized GGA the XC potential has the following expression:
         *  \f[
         *      V_{XC}({\bf r}) = \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \nabla \rho) - 
         *        \nabla \frac{\partial}{\partial (\nabla \rho)} \varepsilon_{xc}(\rho, \nabla \rho) 
         *  \f]
         *  LibXC packs the gradient information into the so-called \a sigma array:
         *  \f[
         *      \sigma = \nabla \rho \nabla \rho
         *  \f]
         *  Changing variables in \f$ V_{XC} \f$ expression gives:
         *  \f{eqnarray*}{
         *      V_{XC}({\bf r}) &=& \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \sigma) - 
         *        \nabla \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma}
         *        \frac{\partial \sigma}{ \partial (\nabla \rho)} \\
         *                      &=& \frac{\partial}{\partial \rho} \varepsilon_{xc}(\rho, \sigma) - 
         *        2 \nabla \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \nabla \rho - 
         *        2 \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \nabla^2 \rho
         *  \f}
         *  The following sequence of functions must be computed:
         *      - density on the real space grid
         *      - gradient of density (in spectral representation)
         *      - gradient of density on the real space grid
         *      - laplacian of density (in spectral representation)
         *      - laplacian of density on the real space grid
         *      - \a sigma array
         *      - a call to Libxc must be performed \a sigma derivatives must be obtained
         *      - \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ in spectral representation
         *      - gradient of \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ in spectral representation
         *      - gradient of \f$ \frac{\partial \varepsilon_{xc}(\rho, \sigma)}{\partial \sigma} \f$ on the real space grid
         *
         *  Expression for spin-polarized potential has a bit more complicated form:
         *  \f{eqnarray*}
         *      V_{XC}^{\gamma} &=& \frac{\partial \varepsilon_{xc}}{\partial \rho_{\gamma}} - \nabla
         *        \Big( 2 \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla \rho_{\gamma} +
         *        \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla \rho_{\delta} \Big) \\
         *                      &=& \frac{\partial \varepsilon_{xc}}{\partial \rho_{\gamma}} 
         *        -2 \nabla \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla \rho_{\gamma} 
         *        -2 \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \gamma}} \nabla^2 \rho_{\gamma} 
         *        - \nabla \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla \rho_{\delta}
         *        - \frac{\partial \varepsilon_{xc}}{\partial \sigma_{\gamma \delta}} \nabla^2 \rho_{\delta} 
         *  \f}
         *  In magnetic case the "up" and "dn" density and potential decomposition is used. Using the fact that the
         *  effective magnetic field is parallel to magnetization at each point in space, we can write the coupling
         *  of density and magnetization with XC potential and XC magentic field as:
         *  \f[
         *      V_{xc}({\bf r}) \rho({\bf r}) + {\bf B}_{xc}({\bf r}){\bf m}({\bf r}) =
         *        V_{xc}({\bf r}) \rho({\bf r}) + {\rm B}_{xc}({\bf r}) {\rm m}({\bf r}) = 
         *        V^{\uparrow}({\bf r})\rho^{\uparrow}({\bf r}) + V^{\downarrow}({\bf r})\rho^{\downarrow}({\bf r})
         *  \f]
         *  where
         *  \f{eqnarray*}{
         *      \rho^{\uparrow}({\bf r}) &=& \frac{1}{2}\Big( \rho({\bf r}) + {\rm m}({\bf r}) \Big) \\
         *      \rho^{\downarrow}({\bf r}) &=& \frac{1}{2}\Big( \rho({\bf r}) - {\rm m}({\bf r}) \Big)
         *  \f}
         *  and
         *  \f{eqnarray*}{
         *      V^{\uparrow}({\bf r}) &=& V_{xc}({\bf r}) + {\rm B}_{xc}({\bf r}) \\
         *      V^{\downarrow}({\bf r}) &=& V_{xc}({\bf r}) - {\rm B}_{xc}({\bf r}) 
         *  \f}
         */
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

        inline void pack(Mixer* mixer)
        {
            size_t n = effective_potential_->pack(0, mixer);
            for (int i = 0; i < parameters_.num_mag_dims(); i++) n += effective_magnetic_field_[i]->pack(n, mixer);
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

};

#endif // __POTENTIAL_H__

