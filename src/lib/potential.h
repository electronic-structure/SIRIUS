
extern "C" void FORTRAN(genylm)(int* lmax, double* tp, complex16* ylm); 

namespace sirius {

/*! 
    \brief Generate effective potential from charge density and magnetization
   
*/
class Potential 
{

    public:

        void init()
        {
            int lmax = 40;

            for (int i = 0; i < global.num_atom_types(); i++)
            { 
                Spline s(global.atom_type(i)->num_mt_points(), global.atom_type(i)->radial_grid()); 
                for (int j = 0; j < global.atom_type(i)->num_mt_points(); j++)
                s[j] = cos(pi * global.atom_type(i)->radial_grid()[j] / global.atom_type(i)->mt_radius());
                s.interpolate();
                for (int m = 0; m < 10; m++)
                    printf("m=%i   mom=%18.10f\n", m, s.integrate(2 + m));
            }
            
            std::vector<complex16> ylm1((lmax + 1) * (lmax + 1));
            std::vector<complex16> ylm2((lmax + 1) * (lmax + 1));
            
            for (int ig = 0; ig < global.num_gvec(); ig++)
            {
                double cartc[3];
                double spc[3];
                global.get_coordinates<cartesian,reciprocal>(global.gvec(ig), cartc);
                spherical_coordinates(cartc, spc);
                spherical_harmonics(lmax, spc[1], spc[2], &ylm1[0]);
                FORTRAN(genylm)(&lmax, &spc[1], &ylm2[0]);
                
                double t = 0.0;
                for (int lm = 0; lm < (lmax+1)*(lmax+1); lm++)
                {
                    t+=abs(ylm1[lm] - ylm2[lm]);
                }
                //std::cout << "diff=" << t << std::endl;
                if (abs(t) > 1e-10) std::cout << "AAAAAAAAAAAAAAAAAAAAAAA!!!!!!!!!!!!!!!!!!" << std::endl;
            }
            
        }
        
        /*! 
            \brief Poisson solver
            
            plane wave expansion
            \f[
                e^{i{\bf g}{\bf r}}=4\pi e^{i{\bf g}{\bf r}_{\alpha}} \sum_{\ell m} i^\ell j_{\ell}(g|{\bf r}-{\bf r}_{\alpha}|)
                    Y_{\ell m}^{*}({\bf \hat g}) Y_{\ell m}(\widehat{{\bf r}-{\bf r}_{\alpha}})
            \f]
        */
        void poisson()
        {
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

                    printf("lm=%i   mom=%f\n", lm, qmt(lm, ia));
                }
            }
            
            // compute multipoles of interstitial density in MT region
            
        
        }
        

};

Potential potential;

};
