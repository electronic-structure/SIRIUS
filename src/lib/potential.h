
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

        mdarray<double,3> sbessel_mt_;
        
        PeriodicFunction<double> hartree_potential_;

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
                int nmtp = global.atom_type(iat)->num_mt_points();
                Spline<double> s(nmtp, global.atom_type(iat)->radial_grid()); 
                
                for (int ir = 0; ir < nmtp; ir++)
                    s[ir] = 1.0 + cos(pi * global.atom_type(iat)->radial_grid()[ir] / global.atom_type(iat)->mt_radius());
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
            
            // compute product of spherical Bessel functions with pseudocharge density
            sbessel_pseudo_prod_.set_dimensions(lmax_pseudo_ + 1, global.num_atom_types(), global.num_gvec_shells());
            sbessel_pseudo_prod_.allocate();
            
            sbessel_mt_.set_dimensions(lmax_pseudo_ + 1, global.num_atom_types(), global.num_gvec_shells());
            sbessel_mt_.allocate();

            mdarray<double,2> jl(NULL, lmax_pseudo_ + 1, global.max_num_mt_points());
            jl.allocate();

            for (int iat = 0; iat < global.num_atom_types(); iat++)
            {
                int nmtp = global.atom_type(iat)->num_mt_points();
                Spline<double> s(nmtp, global.atom_type(iat)->radial_grid()); 

                for (int igs = 0; igs < global.num_gvec_shells(); igs++)
                {
                    // compute spherical Bessel functions
                    for (int ir = 0; ir < nmtp; ir++)
                        gsl_sf_bessel_jl_array(lmax_pseudo_, global.gvec_shell_len(igs) * global.atom_type(iat)->radial_grid()[ir], &jl(0, ir));
                
                    for (int l = 0; l <= lmax_pseudo_; l++)
                    {
                        // save value of the Bessel function at the MT boundary
                        sbessel_mt_(l, iat, igs) = jl(l, nmtp - 1);

                        for (int ir = 0; ir < nmtp; ir++)
                            s[ir] = jl(l, ir) * (1.0 + cos(pi * global.atom_type(iat)->radial_grid()[ir] / global.atom_type(iat)->mt_radius()));
                        s.interpolate();

                        sbessel_pseudo_prod_(l, iat, igs) = s.integrate(2);

                        //printf("l,iat,|G| = %i  %i  %18.12f   prod = %18.12f\n", l, iat, global.gvec_shell_len(igs), sbessel_pseudo_prod_(l, iat, igs));
                    }
                }
            } 

            hartree_potential_.allocate(global.lmax_pot(), global.max_num_mt_points(), global.num_atoms(),
                                        global.fft().size(), global.num_gvec());
            hartree_potential_.allocate_fylm();

            
            std::cout << "number of LL points : " << Lebedev_Laikov_npoint(global.lmax_pot()) << std::endl;
            //exit(0);
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

            hartree_potential_.zero();


            // convert to Ylm expansion
            density.charge_density().convert_to_ylm();
            
            mdarray<complex16,2> qmt(NULL, global.lmmax_rho(), global.num_atoms());
            qmt.allocate();

            // compute MT multipole moments
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int nmtp = global.atom(ia)->type()->num_mt_points(); 
                Spline<complex16> s(nmtp, global.atom(ia)->type()->radial_grid());
                for (int lm = 0; lm < global.lmmax_rho(); lm++)
                {
                    for (int ir = 0; ir < nmtp; ir++)
                        s[ir] = density.charge_density().fylm(lm, ir, ia);
                    s.interpolate();

                    qmt(lm, ia) = s.integrate(l_by_lm(lm) + 2);

                    if (lm == 0)
                        qmt(lm, ia) -= global.atom(ia)->type()->zn() * y00;

                    //printf("lm=%i   mom=%f\n", lm, qmt(lm, ia));
                }
            }
            
            // compute multipoles of interstitial density in MT region
            mdarray<complex16,2> qit(NULL, lmmax_pseudo_, global.num_atoms());
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

                        qit(lm, ia) += density.charge_density().fpw(ig) * zt * zil[l] * conj(ylm_gvec_(lm, ig)) * sbessel_mom_(l, iat, global.gvec_shell(ig));
                    }
                }
            }

            int lmmax = std::max(global.lmmax_rho(), lmmax_pseudo_);

            for (int lm = 0; lm < lmmax; lm++)
            {
                complex16 q1 = 0.0;
                complex16 q2 = 0.0;
                if (lm < global.lmmax_rho()) q1 = qmt(lm, 0);
                if (lm < lmmax_pseudo_) q2 = qit(lm, 0);

                printf("lm=%i   qmt=%18.12f %18.12f   qit=%18.12f %18.12f \n", lm, real(q1), imag(q1), real(q2), imag(q2));
            }
            
            std::cout << "rho(0) = " << density.charge_density().fpw(0) << std::endl;

            std::vector<complex16> pseudo_pw(global.num_gvec());
            memcpy(&pseudo_pw[0], &density.charge_density().fpw(0), global.num_gvec() * sizeof(complex16));

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

                        complex16 q1 = 0.0;
                        if (lm < global.lmmax_rho()) q1 = qmt(lm, ia);

                        pseudo_pw[ig] += zt * conj(zil[l]) * ylm_gvec_(lm, ig) * sbessel_pseudo_prod_(l, iat, global.gvec_shell(ig)) * 
                            (q1 - qit(lm, ia)) / pseudo_mom_(l, iat) / global.omega();
                    }
                }
            }

            std::cout << "rho(0) = " << pseudo_pw[0] << std::endl;

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

                        qit(lm, ia) += pseudo_pw[ig] * zt * zil[l] * conj(ylm_gvec_(lm, ig)) * sbessel_mom_(l, iat, global.gvec_shell(ig));
                    }
                }
            }
            
            for (int lm = 0; lm < lmmax; lm++)
            {
                complex16 q1 = 0.0;
                complex16 q2 = 0.0;
                if (lm < global.lmmax_rho()) q1 = qmt(lm, 0);
                if (lm < lmmax_pseudo_) q2 = qit(lm, 0);

                printf("lm=%i   qmt=%18.12f %18.12f   qit=%18.12f %18.12f \n", lm, real(q1), imag(q1), real(q2), imag(q2));
            }
 
            
            // compute MT part of the potential
            lmmax = std::min(global.lmmax_pot(), global.lmmax_rho());
            
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                double R = global.atom(ia)->type()->mt_radius();
                int nmtp = global.atom(ia)->type()->num_mt_points();
                
                std::vector<complex16> g1;
                std::vector<complex16> g2;
   
                Spline<complex16> rholm(nmtp, global.atom(ia)->type()->radial_grid());

                for (int lm = 0; lm < lmmax; lm++)
                {
                    int l = l_by_lm(lm);

                    for (int ir = 0; ir < nmtp; ir++)
                        rholm[ir] = density.charge_density().fylm(lm, ir, ia);
                    rholm.interpolate();

                    rholm.integrate(g1, l + 2);
                    rholm.integrate(g2, 1 - l);

                    for (int ir = 0; ir < nmtp; ir++)
                    {
                        double r = global.atom(ia)->type()->radial_grid()[ir];

                        complex16 vlm = (1.0 - pow(r / R, 2 * l + 1)) * g1[ir] / pow(r, l + 1) +
                                        (g2[nmtp - 1] - g2[ir]) * pow(r, l) - 
                                        (g1[nmtp - 1] - g1[ir]) * pow(r, l) / pow(R, 2 * l + 1);

                        hartree_potential_.fylm(lm, ir, ia) = fourpi * vlm / double(2 * l + 1);
                    }
                }
                
                // nuclear potential
                for (int ir = 0; ir < nmtp; ir++)
                {
                    double r = global.atom(ia)->type()->radial_grid()[ir];
                    hartree_potential_.fylm(0, ir, ia) -= fourpi * y00 * global.atom(ia)->type()->zn() * (1.0 / r - 1.0 / R);
                }

            }

            // compute pw coefficients of Hartree potential
            pseudo_pw[0] = 0.0;
            for (int ig = 1; ig < global.num_gvec(); ig++)
                pseudo_pw[ig] *= fourpi / pow(global.gvec_shell_len(global.gvec_shell(ig)), 2);

            // compute V_lm at the MT boundary
            mdarray<complex16,2> vmtlm(NULL, global.lmmax_pot(), global.num_atoms());
            vmtlm.allocate();
            vmtlm.zero();
            
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                int iat = global.atom_type_index_by_id(global.atom(ia)->type_id());

                for (int ig = 0; ig < global.num_gvec(); ig++)
                {
                    complex16 zt = fourpi * global.gvec_phase_factor(ig, ia);

                    for (int lm = 0; lm < global.lmmax_pot(); lm++)
                    {
                        int l = l_by_lm(lm);
                        vmtlm(lm, ia) += zt * zil[l] * sbessel_mt_(l, iat, global.gvec_shell(ig)) * conj(ylm_gvec_(lm, ig)) * pseudo_pw[ig];
                    }
                }
            }

            // add boundary condition
            for (int ia = 0; ia < global.num_atoms(); ia++)
            {
                double R = global.atom(ia)->type()->mt_radius();

                for (int lm = 0; lm < global.lmmax_pot(); lm++)
                {
                    int l = l_by_lm(lm);

                    for (int ir = 0; ir < global.atom(ia)->type()->num_mt_points(); ir++)
                        hartree_potential_.fylm(lm, ir, ia) += vmtlm(lm, ia) * pow(global.atom(ia)->type()->radial_grid()[ir] / R, l);
                }
            }

            //complex16* fft_buf = global.fft().
        
        }
        

};

Potential potential;

};
