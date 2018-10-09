// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file initial_density.hpp
 *
 *  \brief Compute initial charge density.
 */

inline void Density::initial_density()
{
    PROFILE("sirius::Density::initial_density");

    zero();

    if (ctx_.full_potential()) {
        initial_density_full_pot();
    } else {
        initial_density_pseudo();

        init_paw();

        init_density_matrix_for_paw();

        generate_paw_loc_density();
    }
}

inline void Density::initial_density_pseudo()
{
    auto v = ctx_.make_periodic_function<index_domain_t::local>([&](int iat, double g)
                                                                {
                                                                    return ctx_.ps_rho_ri().value<int>(iat, g);
                                                                });

    if (ctx_.control().print_checksum_) {
        auto z1 = mdarray<double_complex, 1>(&v[0], ctx_.gvec().count()).checksum();
        ctx_.comm().allreduce(&z1, 1);
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("rho_pw_init", z1);
        }
    }
    std::copy(v.begin(), v.end(), &rho().f_pw_local(0));

    double charge = rho().f_0().real() * unit_cell_.omega();

    if (std::abs(charge - unit_cell_.num_valence_electrons()) > 1e-6) {
        std::stringstream s;
        s << "wrong initial charge density" << std::endl
          << "  integral of the density : " << std::setprecision(12) << charge << std::endl
          << "  target number of electrons : " << std::setprecision(12) << unit_cell_.num_valence_electrons();
        if (ctx_.comm().rank() == 0) {
            WARNING(s);
        }
    }
    rho().fft_transform(1);

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        rho().f_rg(ir) = std::max(rho().f_rg(ir), 0.0);
    }
    /* renormalize charge */
    normalize();

    if (ctx_.control().print_checksum_) {
        auto cs = rho().checksum_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("rho_rg", cs);
        }
    }

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        double R = ctx_.av_atom_radius();

        auto w = [R](double x)
        {
            /* the constants are picked in such a way that the volume integral of the
               weight function is equal to the volume of the atomic sphere;
               in this case the starting magnetiation in the atomic spehre
               integrates to the starting magnetization vector */

            /* volume of the sphere */
            const double norm = fourpi * std::pow(R, 3) / 3.0;
            return (35.0 / 8) * std::pow(1 - std::pow(x / R, 2), 2) / norm;
            //return 10 * std::pow(1 - x / R, 2) / norm;
            //const double b = 1.1016992073677703;
            //return b * 1.0 /  (std::exp(10 * (a - R)) + 1) / norm;
            //const double norm = pi * std::pow(R, 3) / 3.0;
            //return 1.0 / (std::exp(10 * (x - R)) + 1) / norm;
       };

        #pragma omp parallel for
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom_to_grid_map = ctx_.atoms_to_grid_idx_map()[ia];
            vector3d<double> v = unit_cell_.atom(ia).vector_field();

            for (auto coord: atom_to_grid_map) {
                int ir   = coord.first;
                double a = coord.second;
                magnetization(0).f_rg(ir) += v[2] * w(a);
                if (ctx_.num_mag_dims() == 3) {
                    magnetization(1).f_rg(ir) += v[0] * w(a);
                    magnetization(2).f_rg(ir) += v[1] * w(a);
                }
            }
        }
    }

    if (ctx_.control().print_checksum_) {
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            auto cs = component(i).checksum_rg();
            if (ctx_.comm().rank() == 0) {
                std::stringstream s;
                s << "component[" << i << "]";
                utils::print_checksum(s.str(), cs);
            }
        }
    }

    rho().fft_transform(-1);
    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
        magnetization(j).fft_transform(-1);
    }

    //if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
    //    double_complex cs = mdarray<double_complex, 1>(&rho_->f_pw(0), ctx_.gvec().num_gvec()).checksum();
    //    DUMP("checksum(rho_pw): %20.14f %20.14f", std::real(cs), std::imag(cs));
    //}
}

inline void Density::initial_density_full_pot()
{
    /* initialize smooth density of free atoms */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom_density(true);
    }

    /* compute radial integrals */
    Radial_integrals_rho_free_atom ri(ctx_.unit_cell(), ctx_.pw_cutoff(), 40);

    /* compute contribution from free atoms to the interstitial density */
    auto v = ctx_.make_periodic_function<index_domain_t::local>([&ri](int iat, double g)
                                                                {
                                                                    return ri.value(iat, g);
                                                                });

    /* initialize density of free atoms (not smoothed) */
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        unit_cell_.atom_type(iat).init_free_atom_density(false);
    }

    if (ctx_.control().print_checksum_) {
        auto z = mdarray<double_complex, 1>(&v[0], ctx_.gvec().count()).checksum();
        ctx_.comm().allreduce(&z, 1);
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("rho_pw", z);
        }
    }

    /* set plane-wave coefficients of the charge density */
    std::copy(v.begin(), v.end(), &rho().f_pw_local(0));
    /* convert charge density to real space mesh */
    rho().fft_transform(1);

    if (ctx_.control().print_checksum_) {
        auto cs = rho().checksum_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("rho_rg", cs);
        }
    }

    /* remove possible negative noise */
    for (int ir = 0; ir < ctx_.fft().local_size(); ir++) {
        rho().f_rg(ir) = std::max(0.0, rho().f_rg(ir));
    }

    /* set Y00 component of charge density */
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        int nmtp = ctx_.unit_cell().atom(ia).num_mt_points();

        for (int ir = 0; ir < nmtp; ir++) {
            double x = ctx_.unit_cell().atom(ia).radial_grid(ir);
            rho().f_mt<index_domain_t::global>(0, ir, ia) = unit_cell_.atom(ia).type().free_atom_density(x) / y00;
        }
    }

    int lmax = ctx_.lmax_rho();
    int lmmax = utils::lmmax(lmax);

    auto l_by_lm = utils::l_by_lm(lmax);

    std::vector<double_complex> zil(lmax + 1);
    for (int l = 0; l <= lmax; l++) {
        zil[l] = std::pow(double_complex(0, 1), l);
    }

    /* compute boundary value at MT sphere from the plane-wave exapansion */
    auto gvec_ylm = ctx_.generate_gvec_ylm(lmax);

    auto sbessel_mt = ctx_.generate_sbessel_mt(lmax);

    auto flm = ctx_.sum_fg_fl_yg(lmax, v.data(), sbessel_mt, gvec_ylm);

    /* this is the difference between the value of periodic charge density at MT boundary and
       a value of the atom's free density at the boundary */
    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        double R = ctx_.unit_cell().atom(ia).mt_radius();
        double c = unit_cell_.atom(ia).type().free_atom_density(R) / y00;
        flm(0, ia) -= c;
    }

    /* match density at MT */
    for (int iat = 0; iat < ctx_.unit_cell().num_atom_types(); iat++) {
        mdarray<double, 2> rRl(ctx_.unit_cell().max_num_mt_points(), lmax + 1);
        double R = ctx_.unit_cell().atom_type(iat).mt_radius();
        int nmtp = ctx_.unit_cell().atom_type(iat).num_mt_points();

        #pragma omp parallel for default(shared)
        for (int l = 0; l <= lmax; l++) {
            for (int ir = 0; ir < nmtp; ir++) {
                rRl(ir, l) = std::pow(ctx_.unit_cell().atom_type(iat).radial_grid(ir) / R, 2);
            }
        }
        #pragma omp parallel for default(shared)
        for (int i = 0; i < unit_cell_.atom_type(iat).num_atoms(); i++) {
            int ia = unit_cell_.atom_type(iat).atom_id(i);
            std::vector<double> glm(lmmax);
            SHT::convert(lmax, &flm(0, ia), &glm[0]);
            for (int lm = 0; lm < lmmax; lm++) {
                int l = l_by_lm[lm];
                for (int ir = 0; ir < nmtp; ir++) {
                    rho().f_mt<index_domain_t::global>(lm, ir, ia) += glm[lm] * rRl(ir, l);
                }
            }
        }
    }

    /* normalize charge density */
    normalize();

    check_num_electrons();

    //FILE* fout = fopen("rho.dat", "w");
    //for (int i = 0; i <= 10000; i++) {
    //    vector3d<double> v = (i / 10000.0) * vector3d<double>({10.26, 10.26, 10.26});
    //    double val = rho().value(v);
    //    fprintf(fout, "%18.12f %18.12f\n", v.length(), val);
    //}
    //fclose(fout);

    //FILE* fout2 = fopen("rho_rg.dat", "w");
    //for (int i = 0; i <= 10000; i++) {
    //    vector3d<double> v = (i / 10000.0) * vector3d<double>({10.26, 10.26, 10.26});
    //    double val = rho().value_rg(v);
    //    fprintf(fout2, "%18.12f %18.12f\n", v.length(), val);
    //}
    //fclose(fout2);

    /* initialize the magnetization */
    if (ctx_.num_mag_dims()) {
        for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
            int ia = unit_cell_.spl_num_atoms(ialoc);
            vector3d<double> v = unit_cell_.atom(ia).vector_field();
            double len = v.length();

            int nmtp = unit_cell_.atom(ia).num_mt_points();
            Spline<double> rho_s(unit_cell_.atom(ia).type().radial_grid());
            double R = unit_cell_.atom(ia).mt_radius();
            for (int ir = 0; ir < nmtp; ir++) {
                double x = unit_cell_.atom(ia).type().radial_grid(ir);
                rho_s(ir) = this->rho().f_mt<index_domain_t::local>(0, ir, ialoc) * y00 * (1 - 3 * std::pow(x / R, 2) + 2 * std::pow(x / R, 3));
            }

            /* maximum magnetization which can be achieved if we smooth density towards MT boundary */
            double q = fourpi * rho_s.interpolate().integrate(2);

            /* if very strong initial magnetization is given */
            if (q < len) {
                /* renormalize starting magnetization */
                for (int x: {0, 1, 2}) {
                    v[x] *= (q / len);
                }
                len = q;
            }

            if (len > 1e-8) {
                for (int ir = 0; ir < nmtp; ir++) {
                    magnetization(0).f_mt<index_domain_t::local>(0, ir, ialoc) = rho_s(ir) * v[2] / q / y00;
                }
                if (ctx_.num_mag_dims() == 3) {
                    for (int ir = 0; ir < nmtp; ir++) {
                        magnetization(1).f_mt<index_domain_t::local>(0, ir, ialoc) = rho_s(ir) * v[0] / q / y00;
                        magnetization(2).f_mt<index_domain_t::local>(0, ir, ialoc) = rho_s(ir) * v[1] / q / y00;
                    }
                }
            }
        }
    }
}


