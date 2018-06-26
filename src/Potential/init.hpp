inline void Potential::init()
{
    PROFILE("sirius::Potential::init");

    if (ctx_.electronic_structure_method() == electronic_structure_method_t::full_potential_lapwlo) {
        /* compute values of spherical Bessel functions at MT boundary */
        mdarray<double, 3> sbessel_mt_tmp(lmax_ + pseudo_density_order_ + 2, unit_cell_.num_atom_types(), 
                                          ctx_.gvec().num_shells(), memory_t::host, "sbessel_mt_tmp");

        #pragma omp parallel for schedule(static)
        for (int igs = 0; igs < ctx_.gvec().num_shells(); igs++) {
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                gsl_sf_bessel_jl_array(lmax_ + pseudo_density_order_ + 1, 
                                       ctx_.gvec().shell_len(igs) * unit_cell_.atom_type(iat).mt_radius(), 
                                       &sbessel_mt_tmp(0, iat, igs));
            }
        }

        sbessel_mt_ = mdarray<double, 3>(lmax_ + pseudo_density_order_ + 2,
                                         ctx_.gvec().count(),
                                         unit_cell_.num_atom_types(),
                                         memory_t::host, "sbessel_mt_");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                int ig = ctx_.gvec().offset() + igloc;
                for (int l = 0; l < lmax_ + pseudo_density_order_ + 2; l++) {
                    sbessel_mt_(l, igloc, iat) = sbessel_mt_tmp(l, iat, ctx_.gvec().shell(ig));
                }
            }
        }

        /* compute moments of spherical Bessel functions 
         *
         * In[]:= Integrate[SphericalBesselJ[l,G*x]*x^(2+l),{x,0,R},Assumptions->{R>0,G>0,l>=0}]
         * Out[]= (Sqrt[\[Pi]/2] R^(3/2+l) BesselJ[3/2+l,G R])/G^(3/2)
         *
         * and use relation between Bessel and spherical Bessel functions: 
         * Subscript[j, n](z)=Sqrt[\[Pi]/2]/Sqrt[z]Subscript[J, n+1/2](z) */
        mdarray<double, 3> sbessel_mom_tmp(ctx_.lmax_rho() + 1, unit_cell_.num_atom_types(),
                                           ctx_.gvec().num_shells(), memory_t::host, "sbessel_mom_");
        sbessel_mom_tmp.zero();

        /* for G=0 */
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            sbessel_mom_tmp(0, iat, 0) = std::pow(unit_cell_.atom_type(iat).mt_radius(), 3) / 3.0; // for |G|=0
        }

        #pragma omp parallel for schedule(static)
        for (int igs = 1; igs < ctx_.gvec().num_shells(); igs++) {
            for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
                for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                    sbessel_mom_tmp(l, iat, igs) = std::pow(unit_cell_.atom_type(iat).mt_radius(), l + 2) * 
                                                   sbessel_mt_tmp(l + 1, iat, igs) / ctx_.gvec().shell_len(igs);
                }
            }
        }

        sbessel_mom_ = mdarray<double, 3>(ctx_.lmax_rho() + 1,
                                          ctx_.gvec().count(),
                                          unit_cell_.num_atom_types(),
                                          memory_t::host, "sbessel_mom_");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            for (int igloc = 0; igloc < ctx_.gvec().count(); igloc++) {
                int ig = ctx_.gvec().offset() + igloc;
                for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                    sbessel_mom_(l, igloc, iat) = sbessel_mom_tmp(l, iat, ctx_.gvec().shell(ig));
                }
            }
        }

        /* compute Gamma[5/2 + n + l] / Gamma[3/2 + l] / R^l
         *
         * use Gamma[1/2 + p] = (2p - 1)!!/2^p Sqrt[Pi] */
        gamma_factors_R_ = mdarray<double, 2>(ctx_.lmax_rho() + 1, unit_cell_.num_atom_types(), memory_t::host, "gamma_factors_R_");
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            for (int l = 0; l <= ctx_.lmax_rho(); l++) {
                long double Rl = std::pow(unit_cell_.atom_type(iat).mt_radius(), l);

                int n_min = (2 * l + 3);
                int n_max = (2 * l + 1) + (2 * pseudo_density_order_ + 2);
                /* split factorial product into two parts to avoid overflow */
                long double f1 = 1.0;
                long double f2 = 1.0;
                for (int n = n_min; n <= n_max; n += 2) {
                    if (f1 < Rl) {
                        f1 *= (n / 2.0);
                    } else {
                        f2 *= (n / 2.0);
                    }
                }
                gamma_factors_R_(l, iat) = static_cast<double>((f1 / Rl) * f2);
            }
        }
    }
}

