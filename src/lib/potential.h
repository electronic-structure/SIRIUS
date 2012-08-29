
//extern "C" void FORTRAN(genylm)(int* lmax, double* tp, complex16* ylm); 

namespace sirius {

/*! 
    \brief Generate effective potential from charge density and magnetization
   
*/
class Potential 
{
    private:

        int lmax_pseudo_;

        int lmmax_pseudo_;
        
        mdarray<complex16,2> ylm_gvec_;

        mdarray<double,2> pseudo_mom_;
        mdarray<double,3> sbessel_mom_;

        mdarray<double,3> sbessel_pseudo_prod_;
        
        mdarray<double,3> hartree_potential_mt_;

    public:

        void init()
        {
            Timer t("sirius::Potential::init");

            lmax_pseudo_ = 10;
            lmmax_pseudo_ = (lmax_pseudo_ + 1) * (lmax_pseudo_ + 1);


            // compute moments of pseudodensity
            pseudo_mom_.set_dimensions(lmax_pseudo_ + 1, global.num_atom_types());
            pseudo_mom_.allocate();

            for (int iat = 0; iat < global.num_atom_types(); iat++)
            { 
                Spline s(global.atom_type(iat)->num_mt_points(), global.atom_type(iat)->radial_grid()); 
                
                for (int j = 0; j < global.atom_type(iat)->num_mt_points(); j++)
                    s[j] = 1.0 + cos(pi * global.atom_type(iat)->radial_grid()[j] / global.atom_type(iat)->mt_radius());
                s.interpolate();

                for (int l = 0; l <= lmax_pseudo_; l++)
                    pseudo_mom_(l, iat) = s.integrate(2 + l);
            }

            // compute moments of spherical Bessel functions 
            sbessel_mom_.set_dimensions(lmax_pseudo_ + 1, global.num_atom_types(), global.num_gvec_shells());
            sbessel_mom_.allocate();
            sbessel_mom_.zero();

            for (int igs = 0; igs < global.num_gvec_shells(); igs++)
                for (int iat = 0; iat < global.num_atom_types(); iat++)
                    if (igs == 0) 
                        sbessel_mom_(0, iat, 0) = pow(global.atom_type(iat)->mt_radius(), 3) / 3.0;
                    else
                        for (int l = 0; l <= lmax_pseudo_; l++)
                        {
                            double t = gsl_sf_bessel_Jnu(1.5 + l, global.gvec_shell_len(igs) * global.atom_type(iat)->mt_radius());
                            sbessel_mom_(l, iat, igs) = sqrt(pi / 2) * pow(global.atom_type(iat)->mt_radius(), 1.5 + l) * t / pow(global.gvec_shell_len(igs), 1.5);
                        }
            
            // compute spherical harmonics of G-vectors
            ylm_gvec_.set_dimensions(lmmax_pseudo_, global.num_gvec());
            ylm_gvec_.allocate();
            
            for (int ig = 0; ig < global.num_gvec(); ig++)
            {
                double cartc[3];
                double spc[3];
                global.get_coordinates<cartesian,reciprocal>(global.gvec(ig), cartc);
                spherical_coordinates(cartc, spc);
                spherical_harmonics(lmax_pseudo_, spc[1], spc[2], &ylm_gvec_(0, ig));
            }


            mdarray<double,2> jl(NULL, lmax_pseudo_ + 1, global.max_num_mt_points());
            jl.allocate();

            // compute product of spherical Bessel functions with pseudocharge density
            sbessel_pseudo_prod_.set_dimensions(lmax_pseudo_ + 1, global.num_atom_types(), global.num_gvec_shells());
            sbessel_pseudo_prod_.allocate();

            for (int iat = 0; iat < global.num_atom_types(); iat++)
            { 
                Spline s(global.atom_type(iat)->num_mt_points(), global.atom_type(iat)->radial_grid()); 

                for (int igs = 0; igs < global.num_gvec_shells(); igs++)
                {
                    // compute spherical Bessel functions
                    for (int j = 0; j < global.atom_type(iat)->num_mt_points(); j++)
                        gsl_sf_bessel_jl_array(lmax_pseudo_, global.gvec_shell_len(igs) * global.atom_type(iat)->radial_grid()[j], &jl(0, j));
                
                    for (int l = 0; l <= lmax_pseudo_; l++)
                    {
                        for (int j = 0; j < global.atom_type(iat)->num_mt_points(); j++)
                            s[j] = jl(l, j) * (1.0 + cos(pi * global.atom_type(iat)->radial_grid()[j] / global.atom_type(iat)->mt_radius()));
                        s.interpolate();

                        sbessel_pseudo_prod_(l, iat, igs) = s.integrate(2);

                        //printf("l,iat,|G| = %i  %i  %18.12f   prod = %18.12f\n", l, iat, global.gvec_shell_len(igs), sbessel_pseudo_prod_(l, iat, igs));
                    }
                }
            } 

            hartree_potential_mt_.set_dimensions(global.lmmax_pot(), global.max_num_mt_points(), global.num_atoms());
            hartree_potential_mt_.allocate();

            hartree_potential_mt_.zero();
        }
        
        /*! 
            \brief Poisson solver
            
            plane wave expansion
            \f[
                e^{i{\bf g}{\bf r}}=4\pi e^{i{\bf g}{\bf r}_{\alpha}} \sum_{\ell m} i^\ell j_{\ell}(g|{\bf r}-{\bf r}_{\alpha}|)
                    Y_{\ell m}^{*}({\bf \hat g}) Y_{\ell m}(\widehat{{\bf r}-{\bf r}_{\alpha}})
            \f]

            Spherical Bessel function moments
            \f[
                \int_0^R j_{\ell}(a x)x^{2+\ell} dx = \frac{\sqrt{\frac{\pi }{2}} R^{\ell+\frac{3}{2}} 
                   J_{\ell+\frac{3}{2}}(a R)}{a^{3/2}}
            \f]
            for a = 0 the integral is \f$ \frac{R^3}{3} \delta_{\ell,0} \f$

            General solution to the Poisson equation with spherical boundary condition:
            \f[
                V({\bf x}) = \int \rho({\bf x'})G({\bf x},{\bf x'}) d{\bf x'} - \frac{1}{4 \pi} \int_{S} V({\bf x'}) \frac{\partial G}{\partial n'} d{\bf S'}
            \f]

            Green's function for a sphere
            \f[
                G({\bf x},{\bf x'}) = 4\pi \sum_{\ell m} \frac{Y_{\ell m}^{*}(\hat {\bf x'}) Y_{\ell m}(\hat {\bf x})}{2\ell + 1}
                    \frac{r_{<}^{\ell}}{r_{>}^{\ell+1}}\Biggl(1 - \Big( \frac{r_{>}}{R} \Big)^{2\ell + 1} \Biggr)
            \f]

        */
        void poisson()
        {
            Timer t("sirius::Potential::poisson");
            
            mdarray<double,2> qmt(NULL, global.lmmax_rho(), global.num_atoms());
            qmt.allocate();

            // compute MT multipole moments
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                Spline s(global.atom(ia)->type()->num_mt_points(), global.atom(ia)->type()->radial_grid());
                for (int lm = 0; lm < global.lmmax_rho(); lm++)
                {
                    for (int i = 0; i < global.atom(ia)->type()->num_mt_points(); i++)
                        s[i] = density.charge_density_mt_(lm, i, ia);
                    s.interpolate();

                    qmt(lm, ia) = s.integrate(l_by_lm(lm) + 2);

                    if (lm == 0)
                        qmt(lm, ia) -= global.atom(ia)->type()->zn() * y00;

                    //printf("lm=%i   mom=%f\n", lm, qmt(lm, ia));
                }
            }
            
            // compute multipoles of interstitial density in MT region
            mdarray<double,2> qit(NULL, lmmax_pseudo_, global.num_atoms());
            qit.allocate();
            qit.zero();

            std::vector<complex16> zil(lmax_pseudo_ + 1);
            for (int l = 0; l <= lmax_pseudo_; l++)
                zil[l] = pow(zi, l);

            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int iat = global.atom_type_index_by_id(global.atom(ia)->type_id());
                
                for (int ig = 0; ig < global.num_gvec(); ig++)
                {
                    complex16 zt = fourpi * global.gvec_phase_factor(ig, ia);

                    for (int lm = 0; lm < lmmax_pseudo_; lm++)
                    {
                        int l = l_by_lm(lm);

                        qit(lm, ia) += real(density.charge_density_pw_(ig) * zt * zil[l] * conj(ylm_gvec_(lm, ig)) * sbessel_mom_(l, iat, global.gvec_shell(ig)));
                    }
                }
            }

            int lmmax = std::max(global.lmmax_rho(), lmmax_pseudo_);

            for (int lm = 0; lm < lmmax; lm++)
            {
                double q1 = 0.0;
                double q2 = 0.0;
                if (lm < global.lmmax_rho()) q1 = qmt(lm, 0);
                if (lm < lmmax_pseudo_) q2 = qit(lm, 0);

                printf("lm=%i   qmt=%18.12f   qit=%18.12f \n", lm, q1, q2);
            }
            
            std::cout << "rho(0) = " << density.charge_density_pw_(0) << std::endl;

            // add contribution from pseudocharge density
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int iat = global.atom_type_index_by_id(global.atom(ia)->type_id());
                
                for (int ig = 0; ig < global.num_gvec(); ig++)
                {
                    complex16 zt = fourpi * conj(global.gvec_phase_factor(ig, ia));

                    for (int lm = 0; lm < lmmax_pseudo_; lm++)
                    {
                        int l = l_by_lm(lm);

                        double q1 = 0.0;
                        if (lm < global.lmmax_rho()) q1 = qmt(lm, ia);

                        density.charge_density_pw_(ig) += zt * conj(zil[l]) * ylm_gvec_(lm, ig) * 
                            sbessel_pseudo_prod_(l, iat, global.gvec_shell(ig)) * (q1 - qit(lm, ia)) / pseudo_mom_(l, iat) / global.omega();
                    }
                }
            }

            std::cout << "rho(0) = " << density.charge_density_pw_(0) << std::endl;


            qit.zero();
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int iat = global.atom_type_index_by_id(global.atom(ia)->type_id());
                
                for (int ig = 0; ig < global.num_gvec(); ig++)
                {
                    complex16 zt = fourpi * global.gvec_phase_factor(ig, ia);

                    for (int lm = 0; lm < lmmax_pseudo_; lm++)
                    {
                        int l = l_by_lm(lm);

                        qit(lm, ia) += real(density.charge_density_pw_(ig) * zt * zil[l] * conj(ylm_gvec_(lm, ig)) * sbessel_mom_(l, iat, global.gvec_shell(ig)));
                    }
                }
            }

            for (int lm = 0; lm < lmmax; lm++)
            {
                double q1 = 0.0;
                double q2 = 0.0;
                if (lm < global.lmmax_rho()) q1 = qmt(lm, 0);
                if (lm < lmmax_pseudo_) q2 = qit(lm, 0);

                printf("lm=%i   qmt=%18.12f   qit=%18.12f \n", lm, q1, q2);
            }

            
            
            // compute MT part of the potential
            lmmax = std::min(global.lmmax_pot(), global.lmmax_rho());
            
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                double R = global.atom(ia)->type()->mt_radius();
                int np = global.atom(ia)->type()->num_mt_points();
                
                std::vector<double> g1;
                std::vector<double> g2;
   
                Spline rholm(np, global.atom(ia)->type()->radial_grid());

                for (int lm = 0; lm < lmmax; lm++)
                {
                    int l = l_by_lm(lm);

                    for (int ix = 0; ix < np; ix++)
                        rholm[ix] = density.charge_density_mt_(lm, ix, ia);
                    rholm.interpolate();

                    rholm.integrate(g1, l + 2);
                    rholm.integrate(g2, 1 - l);

                    for (int ix = 0; ix < np; ix++)
                    {
                        double r = global.atom(ia)->type()->radial_grid()[ix];

                        double vlm = (1.0 - pow(r / R, 2 * l + 1)) * g1[ix] / pow(r, l + 1) +
                                     (g2[np - 1] - g2[ix]) * pow(r, l) - 
                                     (g1[np - 1] - g1[ix]) * pow(r, l) / pow(R, 2 * l + 1);

                        hartree_potential_mt_(lm, ix, ia) = fourpi * vlm / (2 * l + 1);
                    }
                }
                
                // nuclear potential
                for (int ix = 0; ix < np; ix++)
                {
                    double r = global.atom(ia)->type()->radial_grid()[ix];
                    hartree_potential_mt_(0, ix, ia) -= fourpi * y00 * global.atom(ia)->type()->zn() * (1.0 / r - 1.0 / R);
                }

            }

            //complex16* fft_buf = global.fft().
            
        
        }
        

};

Potential potential;

};
