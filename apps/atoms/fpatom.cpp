/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

namespace fpatom {

struct radial_function_index_descriptor
{
    int n;
    int l;

    radial_function_index_descriptor(int n, int l)
        : n(n)
        , l(l)
    {
    }
};

struct basis_function_index_descriptor
{
    int n;
    int l;
    int m;
    int lm;
    int idxrf;

    basis_function_index_descriptor(int n, int l, int m, int idxrf)
        : n(n)
        , l(l)
        , m(m)
        , idxrf(idxrf)
    {
        lm = Utils::lm_by_l_m(l, m);
    }
};

} // namespace fpatom

void
generate_vha(sirius::Spheric_function<spectral, double> const& rho, sirius::Spheric_function<spectral, double>& vha)
{
    runtime::Timer t("generate_vha");

    std::vector<double> g1;
    std::vector<double> g2;

    int lmmax   = rho.angular_domain_size();
    auto& rgrid = rho.radial_grid();
    int np      = rgrid.num_points();

    auto l_by_lm = Utils::l_by_lm(Utils::lmax_by_lmmax(lmmax));

    for (int lm = 0; lm < lmmax; lm++) {
        int l = l_by_lm[lm];

        auto rholm = rho.component(lm);

        rholm.integrate(g1, l + 2);
        rholm.integrate(g2, 1 - l);

        double d2 = 1.0 / double(2 * l + 1);
        for (int ir = 0; ir < np; ir++) {
            double r = rgrid[ir];

            double vlm = g1[ir] / std::pow(r, l + 1) + (g2[np - 1] - g2[ir]) * std::pow(r, l);

            vha(lm, ir) = fourpi * vlm * d2;
        }
    }
}

void
generate_xc(int lmmax__, std::array<sirius::Spheric_function<spectral, double>, 2>& rholm__,
            std::array<sirius::Spheric_function<spectral, double>, 2>& vxclm__,
            sirius::Spheric_function<spectral, double>& exclm__)
{
    runtime::Timer t("generate_vxc");

    sirius::SHT sht(Utils::lmax_by_lmmax(lmmax__));

    auto& rgrid = exclm__.radial_grid();

    auto rho_up_tp = sht.transform(rholm__[0]);
    auto rho_dn_tp = sht.transform(rholm__[1]);

    for (int ir = 0; ir < rgrid.num_points(); ir++) {
        for (int i = 0; i < sht.num_points(); i++) {
            if (rho_up_tp(i, ir) < 0)
                rho_up_tp(i, ir) = 0;
            if (rho_dn_tp(i, ir) < 0)
                rho_dn_tp(i, ir) = 0;
        }
    }

    sirius::Spheric_function<spatial, double> vxc_up_tp(sht.num_points(), rgrid);
    sirius::Spheric_function<spatial, double> vxc_dn_tp(sht.num_points(), rgrid);
    sirius::Spheric_function<spatial, double> exc_tp(sht.num_points(), rgrid);

    vxc_up_tp.zero();
    vxc_dn_tp.zero();
    exc_tp.zero();

    sirius::XC_functional_base Ex("XC_LDA_X", 2);
    sirius::XC_functional_base Ec("XC_LDA_C_VWN", 2);

    Ex.add_lda(sht.num_points() * rgrid.num_points(), &rho_up_tp(0, 0), &rho_dn_tp(0, 0), &vxc_up_tp(0, 0),
               &vxc_dn_tp(0, 0), &exc_tp(0, 0));
    Ec.add_lda(sht.num_points() * rgrid.num_points(), &rho_up_tp(0, 0), &rho_dn_tp(0, 0), &vxc_up_tp(0, 0),
               &vxc_dn_tp(0, 0), &exc_tp(0, 0));

    vxclm__[0] = sht.transform(vxc_up_tp);
    vxclm__[1] = sht.transform(vxc_dn_tp);
    exclm__    = sht.transform(exc_tp);
}

void
generate_radial_functions(std::vector<fpatom::radial_function_index_descriptor> const& radial_functions_desc__,
                          sirius::Radial_solver const& rsolver__, sirius::Radial_grid const& rgrid__,
                          std::array<std::vector<double>, 2> const& veff_spherical__, int zn__,
                          mdarray<double, 2>& enu__, mdarray<double, 4>& radial_functions__)
{
    runtime::Timer t("generate_radial_functions");

    #pragma omp parallel
    {
        std::vector<double> p, rdudr;
        #pragma omp for
        for (int i = 0; i < (int)radial_functions_desc__.size(); i++) {
            for (int ispn : {0, 1}) {
                sirius::Bound_state bound_state(relativity_t::none, zn__, radial_functions_desc__[i].n,
                                                radial_functions_desc__[i].l, 0, rgrid__, veff_spherical__[ispn],
                                                enu__(i, ispn));
                enu__(i, ispn) = bound_state.enu();
                STOP();
                // auto& u = bound_state.u();
                // auto& rdudr = bound_state.rdudr();

                ////enu__(i, ispn) = rsolver__.bound_state(radial_functions_desc__[i].n, radial_functions_desc__[i].l,
                ////                                       enu__(i, ispn), veff_spherical__[ispn], p, rdudr);

                // for (int ir = 0; ir < rgrid__.num_points(); ir++)
                //{
                //     radial_functions__(ir, i, 0, ispn) = u[ir]; //p[ir] / rgrid__[ir];
                //     radial_functions__(ir, i, 1, ispn) = rdudr[ir]; //rdudr[ir];
                // }
            }
        }
    }
}

void
generate_radial_integrals(int lmax__, sirius::Radial_grid const& rgrid__,
                          std::vector<fpatom::radial_function_index_descriptor> const& radial_functions_desc__,
                          mdarray<double, 4> const& radial_functions__,
                          std::array<sirius::Spheric_function<spectral, double>, 2> const& vefflm__,
                          mdarray<double, 4>& h_radial_integrals__, mdarray<double, 3>& o_radial_integrals__)
{
    runtime::Timer t("generate_radial_integrals");

    h_radial_integrals__.zero();
    o_radial_integrals__.zero();
    auto l_by_lm = Utils::l_by_lm(lmax__);
    int lmmax    = Utils::lmmax(lmax__);

    runtime::Timer t1("generate_radial_integrals|1");
    for (int ispn : {0, 1}) {
        #pragma omp parallel
        {
            sirius::Spline<double> s(rgrid__);
            #pragma omp for
            for (int i1 = 0; i1 < (int)radial_functions_desc__.size(); i1++) {
                int l1 = radial_functions_desc__[i1].l;
                for (int i2 = 0; i2 <= i1; i2++) {
                    int l2 = radial_functions_desc__[i2].l;
                    if ((l1 + l2) % 2 == 0) {
                        /* for spherical part of potential integrals are diagonal in l */
                        if (radial_functions_desc__[i1].l == radial_functions_desc__[i2].l) {
                            int ll = radial_functions_desc__[i1].l * (radial_functions_desc__[i1].l + 1);
                            for (int ir = 0; ir < rgrid__.num_points(); ir++) {
                                /* u_1(r) * u_2(r) */
                                double t0 = radial_functions__(ir, i1, 0, ispn) * radial_functions__(ir, i2, 0, ispn);
                                /* r*u'_1(r) * r*u'_2(r) */
                                double t1 = radial_functions__(ir, i1, 1, ispn) * radial_functions__(ir, i2, 1, ispn);
                                s[ir]     = 0.5 * t1 +
                                        t0 * (0.5 * ll + vefflm__[ispn](0, ir) * y00 * std::pow(rgrid__[ir], 2));
                            }
                            h_radial_integrals__(0, i1, i2, ispn) = s.interpolate().integrate(0) / y00;
                        }
                    }

                    if (l1 == l2) {
                        for (int ir = 0; ir < rgrid__.num_points(); ir++) {
                            s[ir] = radial_functions__(ir, i1, 0, ispn) * radial_functions__(ir, i2, 0, ispn);
                        }
                        o_radial_integrals__(i1, i2, ispn) = s.interpolate().integrate(2);
                        o_radial_integrals__(i2, i1, ispn) = o_radial_integrals__(i1, i2, ispn);
                    }
                }
            }
        }
    }
    t1.stop();

    mdarray<double, 2> timers(3, omp_get_max_threads());
    timers.zero();

    runtime::Timer t2("generate_radial_integrals|2");
    mdarray<double, 2> buf(196, omp_get_max_threads());
#ifdef SIRIUS_GPU
    buf.allocate_on_device();
#endif
    for (int ispn : {0, 1}) {
        std::vector<sirius::Spline<double>> srf(radial_functions_desc__.size());
        #pragma omp parallel for
        for (int i = 0; i < (int)radial_functions_desc__.size(); i++) {
            int thread_id = omp_get_thread_num();
            double ts     = omp_get_wtime();

            srf[i] = sirius::Spline<double>(rgrid__);
            for (int ir = 0; ir < rgrid__.num_points(); ir++)
                srf[i][ir] = radial_functions__(ir, i, 0, ispn);
            srf[i].interpolate();
#ifdef SIRIUS_GPU
            srf[i].copy_to_device();
#endif

            timers(0, thread_id) += (omp_get_wtime() - ts);
        }

        #pragma omp parallel
        {
            sirius::Spline<double> svrf(rgrid__);
            int thread_id = omp_get_thread_num();
            #pragma omp for
            for (int lm = 1; lm < lmmax; lm++) {
                int l = l_by_lm[lm];
                std::vector<double> vtmp(rgrid__.num_points());
                for (int ir = 0; ir < rgrid__.num_points(); ir++)
                    vtmp[ir] = vefflm__[ispn](lm, ir);

                for (int i1 = 0; i1 < (int)radial_functions_desc__.size(); i1++) {
                    double ts = omp_get_wtime();

                    int l1 = radial_functions_desc__[i1].l;
                    for (int ir = 0; ir < rgrid__.num_points(); ir++)
                        svrf[ir] = radial_functions__(ir, i1, 0, ispn) * vtmp[ir];
                    svrf.interpolate();
#ifdef SIRIUS_GPU
                    svrf.async_copy_to_device(thread_id);
#endif

                    timers(1, thread_id) += (omp_get_wtime() - ts);

                    ts = omp_get_wtime();
                    for (int i2 = 0; i2 <= i1; i2++) {
                        int l2 = radial_functions_desc__[i2].l;
                        if ((l + l1 + l2) % 2 == 0) {
                            // h_radial_integrals__(lm, i1, i2, ispn) += sirius::inner<CPU>(svrf, srf[i2], 2);
                            // h_radial_integrals__(lm, i1, i2, ispn) += sirius::Spline<double>::integrate(&svrf,
                            // &srf[i2], 2);

                            // double r = sirius::spline_inner_product_gpu_v2(rgrid__.num_points(), rgrid__.x().template
                            // at<GPU>(), rgrid__.dx().template at<GPU>(),
                            //                    svrf.coefs().template at<GPU>(), srf[i2].coefs().template at<GPU>(),
                            //                    buf.at<GPU>(0, thread_id), buf.at<CPU>(0, thread_id), thread_id);

                            // h_radial_integrals__(lm, i1, i2, ispn) += r; //sirius::inner<GPU>(svrf, srf[i2], 2);
                            h_radial_integrals__(lm, i1, i2, ispn) += sirius::inner(svrf, srf[i2], 2);
                        }
                        h_radial_integrals__(lm, i2, i1, ispn) = h_radial_integrals__(lm, i1, i2, ispn);
                    }
                    timers(2, thread_id) += (omp_get_wtime() - ts);
                }
            }
        }
    }
    for (int i = 0; i < omp_get_thread_num(); i++) {
        std::printf("thread: %i, timers: %12.6f %12.6f %12.6f\n", i, timers(0, i), timers(1, i), timers(2, i));
    }
    t2.stop();
}

std::pair<std::vector<double>, matrix<double_complex>>
solve_gen_evp(int ispn__, std::vector<fpatom::basis_function_index_descriptor> const& basis_functions_desc__,
              sirius::Gaunt_coefficients<double_complex> const& gaunt__, mdarray<double, 4> const& h_radial_integrals__,
              mdarray<double, 3> const& o_radial_integrals__)
{
    runtime::Timer t("solve_gen_evp");

    Eigenproblem_lapack evp_solver(-1.0);
    std::pair<std::vector<double>, matrix<double_complex>> result;

    int N = (int)basis_functions_desc__.size();

    result.first  = std::vector<double>(N);
    result.second = matrix<double_complex>(N, N);

    matrix<double_complex> h(N, N);
    matrix<double_complex> o(N, N);
    o.zero();
    for (int i1 = 0; i1 < N; i1++) {
        int lm1    = basis_functions_desc__[i1].lm;
        int idxrf1 = basis_functions_desc__[i1].idxrf;
        for (int i2 = 0; i2 < N; i2++) {
            int lm2    = basis_functions_desc__[i2].lm;
            int idxrf2 = basis_functions_desc__[i2].idxrf;
            h(i2, i1)  = gaunt__.sum_L3_gaunt(lm2, lm1, &h_radial_integrals__(0, idxrf2, idxrf1, ispn__));
            if (lm1 == lm2)
                o(i2, i1) = o_radial_integrals__(idxrf2, idxrf1, ispn__);
        }
    }
    if (evp_solver.solve(N, N, h.at<CPU>(), h.ld(), o.at<CPU>(), o.ld(), &result.first[0], result.second.at<CPU>(),
                         result.second.ld())) {
        std::printf("error in evp solver\n");
        exit(0);
    }

    return result;
}

void
generate_density(int lmmax__, int num_states__, int ispn__,
                 std::vector<fpatom::radial_function_index_descriptor> const& radial_functions_desc__,
                 std::vector<fpatom::basis_function_index_descriptor> const& basis_functions_desc__,
                 matrix<double_complex>& evec__, sirius::Gaunt_coefficients<double_complex>& gaunt__,
                 mdarray<double, 4> const& radial_functions__, sirius::Spheric_function<spectral, double>& rholm__)
{
    runtime::Timer t("generate_density");

    int N = (int)basis_functions_desc__.size();

    matrix<double_complex> zdens(N, N);
    mdarray<double, 3> dens(radial_functions_desc__.size(), radial_functions_desc__.size(), lmmax__);

    linalg<device_t::CPU>::gemm(0, 2, N, N, num_states__, double_complex(1, 0), evec__, evec__, double_complex(0, 0),
                                zdens);

    dens.zero();
    for (int i1 = 0; i1 < N; i1++) {
        int lm1    = basis_functions_desc__[i1].lm;
        int idxrf1 = basis_functions_desc__[i1].idxrf;
        for (int i2 = 0; i2 < N; i2++) {
            int lm2    = basis_functions_desc__[i2].lm;
            int idxrf2 = basis_functions_desc__[i2].idxrf;

            for (int k = 0; k < gaunt__.num_gaunt(lm1, lm2); k++) {
                int lm3           = gaunt__.gaunt(lm1, lm2, k).lm3;
                double_complex gc = gaunt__.gaunt(lm1, lm2, k).coef;
                dens(idxrf1, idxrf2, lm3) += std::real(gc * std::conj(zdens(i1, i2)));
            }
        }
    }

    rholm__.zero();
    #pragma omp parallel
    {
        std::vector<double> tmp(rholm__.radial_grid().num_points());
        #pragma omp for
        for (int lm = 0; lm < lmmax__; lm++) {
            memset(&tmp[0], 0, tmp.size() * sizeof(double));
            for (int i1 = 0; i1 < (int)radial_functions_desc__.size(); i1++) {
                for (int i2 = 0; i2 < (int)radial_functions_desc__.size(); i2++) {
                    for (int ir = 0; ir < rholm__.radial_grid().num_points(); ir++) {
                        tmp[ir] += dens(i1, i2, lm) * radial_functions__(ir, i1, 0, ispn__) *
                                   radial_functions__(ir, i2, 0, ispn__);
                    }
                }
            }
            for (int ir = 0; ir < rholm__.radial_grid().num_points(); ir++)
                rholm__(lm, ir) = tmp[ir];
        }
    }
}

void
scf(int zn, int mag_mom, int niter, double alpha, int lmax, int nmax)
{
    int lmax_pot = lmax;
    int lmmax_pot(Utils::lmmax(lmax_pot));

    int occ[2];
    occ[0] = (zn + mag_mom) / 2;
    occ[1] = (zn - mag_mom) / 2;

    auto l_by_lm = Utils::l_by_lm(lmax_pot);

    std::vector<fpatom::radial_function_index_descriptor> radial_functions_desc;
    std::vector<fpatom::basis_function_index_descriptor> basis_functions_desc;

    for (int n = 1; n <= nmax; n++) {
        for (int l = 0; l <= std::min(lmax, n - 1); l++) {
            radial_functions_desc.push_back(fpatom::radial_function_index_descriptor(n, l));
        }
    }

    for (int i = 0; i < (int)radial_functions_desc.size(); i++) {
        int n = radial_functions_desc[i].n;
        int l = radial_functions_desc[i].l;
        for (int m = -l; m <= l; m++) {
            basis_functions_desc.push_back(fpatom::basis_function_index_descriptor(n, l, m, i));
        }
    }

    // for (int i = 0; i < (int)basis_functions_desc.size(); i++)
    //{
    //     std::printf("%i: %i %i %i\n", i, basis_functions_desc[i].n, basis_functions_desc[i].l,
    //     basis_functions_desc[i].m);
    // }

    // sirius::Radial_grid rgrid(pow2_grid, 25000, 1e-7, 100.0);
    sirius::Radial_grid rgrid(sirius::exponential_grid, 20000, 1e-7, 150.0);
#ifdef SIRIUS_GPU
    rgrid.copy_to_device();
#endif
    // sirius::Radial_solver rsolver(relativity_t::none, zn, rgrid);
    // rsolver.set_tolerance(1e-12);

    sirius::Gaunt_coefficients<double_complex> gaunt(lmax, lmax_pot, lmax, sirius::SHT::gaunt_hybrid);

    mdarray<double, 2> enu(radial_functions_desc.size(), 2);
    mdarray<double, 4> radial_functions(rgrid.num_points(), radial_functions_desc.size(), 2, 2);
    mdarray<double, 4> h_radial_integrals(lmmax_pot, radial_functions_desc.size(), radial_functions_desc.size(), 2);
    mdarray<double, 3> o_radial_integrals(radial_functions_desc.size(), radial_functions_desc.size(), 2);

    std::array<std::vector<double>, 2> veff_spherical = {std::vector<double>(rgrid.num_points()),
                                                         std::vector<double>(rgrid.num_points())};

    std::array<sirius::Spheric_function<spectral, double>, 2> vefflm = {
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid),
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid)};

    std::array<sirius::Spheric_function<spectral, double>, 2> rholm = {
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid),
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid)};

    std::array<sirius::Spheric_function<spectral, double>, 2> rholm_new = {
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid),
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid)};

    sirius::Spheric_function<spectral, double> rholm_tot(lmmax_pot, rgrid);

    sirius::Spheric_function<spectral, double> rholm_tot_new(lmmax_pot, rgrid);

    sirius::Spheric_function<spectral, double> vhalm(lmmax_pot, rgrid);

    std::array<sirius::Spheric_function<spectral, double>, 2> vxclm = {
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid),
            sirius::Spheric_function<spectral, double>(lmmax_pot, rgrid)};

    sirius::Spheric_function<spectral, double> exclm(lmmax_pot, rgrid);

    /* initial effective potential */
    for (int ispn = 0; ispn < 2; ispn++) {
        vefflm[ispn].zero();
        for (int ir = 0; ir < rgrid.num_points(); ir++)
            vefflm[ispn](0, ir) = -zn / rgrid[ir] / y00;
    }

    /* starting values for E_{nu} */
    for (int i = 0; i < (int)radial_functions_desc.size(); i++) {
        int n     = radial_functions_desc[i].n;
        enu(i, 0) = enu(i, 1) = -double(zn * zn) / 2 / n / n;
    }

    double Etot_old = 0.0;

    for (int iter = 0; iter < niter; iter++) {
        std::printf("\n");
        std::printf("iteration: %i\n", iter);
        for (int ispn = 0; ispn < 2; ispn++) {
            for (int ir = 0; ir < rgrid.num_points(); ir++)
                veff_spherical[ispn][ir] = vefflm[ispn](0, ir) * y00;
        }

        STOP();
        // generate_radial_functions(radial_functions_desc, rsolver, rgrid, veff_spherical, zn, enu, radial_functions);

        generate_radial_integrals(lmax_pot, rgrid, radial_functions_desc, radial_functions, vefflm, h_radial_integrals,
                                  o_radial_integrals);

        double eband = 0;
        for (int ispn = 0; ispn < 2; ispn++) {
            if (occ[ispn] == 0)
                continue;
            auto evp = solve_gen_evp(ispn, basis_functions_desc, gaunt, h_radial_integrals, o_radial_integrals);

            generate_density(lmmax_pot, occ[ispn], ispn, radial_functions_desc, basis_functions_desc, evp.second, gaunt,
                             radial_functions, rholm_new[ispn]);
            std::printf("eigen-values for spin %i:\n", ispn);
            for (int j = 0; j < occ[ispn]; j++)
                std::printf("%12.6f ", evp.first[j]);
            std::printf("\n");

            for (int j = 0; j < occ[ispn]; j++)
                eband += evp.first[j];
        }

        if (iter) {
            double d = 0;
            for (int ispn = 0; ispn < 2; ispn++) {
                for (int ir = 0; ir < rgrid.num_points(); ir++) {
                    for (int lm = 0; lm < lmmax_pot; lm++) {
                        d += std::abs(rholm[ispn](lm, ir) - rholm_new[ispn](lm, ir));
                        rholm[ispn](lm, ir) = (1 - alpha) * rholm[ispn](lm, ir) + alpha * rholm_new[ispn](lm, ir);
                    }
                }
            }
            std::printf("rho diff: %12.6f\n", d);
        } else {
            for (int ispn = 0; ispn < 2; ispn++)
                memcpy(&rholm[ispn](0, 0), &rholm_new[ispn](0, 0), lmmax_pot * rgrid.num_points() * sizeof(double));
        }

        rholm_tot = rholm[0] + rholm[1];

        std::printf("density: %12.6f\n", rholm_tot.component(0).integrate(2) * y00 * fourpi);
        generate_vha(rholm_tot, vhalm);
        double Ecoul = 0.5 * inner(rholm_tot, vhalm);

        for (int ir = 0; ir < rgrid.num_points(); ir++)
            vhalm(0, ir) -= zn / rgrid[ir] / y00;

        generate_xc(lmmax_pot, rholm, vxclm, exclm);

        vefflm[0] = vxclm[0] + vhalm;
        vefflm[1] = vxclm[1] + vhalm;

        double Ekin  = eband - inner(vefflm[0], rholm[0]) - inner(vefflm[1], rholm[1]);
        double Eenuc = -fourpi * y00 * zn * rholm_tot.component(0).integrate(1);
        double Exc   = inner(rholm_tot, exclm);
        double Etot  = Ekin + Ecoul + Eenuc + Exc;

        std::printf("Etot : %12.6f\n", Etot);
        std::printf("Ekin : %12.6f\n", Ekin);
        std::printf("Ecoul: %12.6f\n", Ecoul);
        std::printf("Eenuc: %12.6f\n", Eenuc);
        std::printf("Exc  : %12.6f\n", Exc);

        std::printf("Etot_diff : %12.10f\n", std::abs(Etot - Etot_old));
        Etot_old = Etot;
    }
}

int
main(int argn, char** argv)
{
    /* handle command line arguments */
    cmd_args args;
    args.register_key("--zn=", "{int} nuclear charge");
    args.register_key("--mag_mom=", "{int} magnetic moment");
    args.register_key("--niter=", "{int} number of iterations");
    args.register_key("--alpha=", "{double} mixing parameter");
    args.register_key("--nmax=", "{int} maximum principal quantum number");
    args.register_key("--lmax=", "{int} maximum orbital quantum number");
    args.parse_args(argn, argv);

    if (argn == 1) {
        std::printf("\n");
        std::printf("Full-potential atom solver.\n");
        std::printf("\n");
        std::printf("Usage: %s [options] \n", argv[0]);
        args.print_help();
        exit(0);
    }

    int zn       = args.value<int>("zn");
    int mag_mom  = args.value<int>("mag_mom");
    int niter    = args.value<int>("niter", 10);
    double alpha = args.value<double>("alpha", 0.5);
    int nmax     = args.value<int>("nmax", 6);
    int lmax     = args.value<int>("lmax", 6);

    sirius::initialize(true);
#ifdef SIRIUS_GPU
    cuda_device_info();
#endif

    scf(zn, mag_mom, niter, alpha, nmax, lmax);

    runtime::Timer::print();
    sirius::finalize();
}
