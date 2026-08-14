/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file lapw_radial_basis.hpp
 *
 *  \brief Radial basis functions of the LAPW method.
 */

#ifndef __LAPW_RADIAL_BASIS_HPP__
#define __LAPW_RADIAL_BASIS_HPP__

#include "unit_cell/atom_type.hpp"

namespace sirius {

struct lapw_radial_basis_t
{
    /// Reference to atom type.
    Atom_type const& atom_type_;

    relativity_t rel_;

    /// Spherical part of the effective potential.
    /** Used by the LAPW radial solver. Actual value is stored, not the Y00 component. */
    std::vector<double> spherical_potential_;

    /// List of radial functions for the LAPW basis.
    /** This array stores all the radial functions (AW and LO) and their derivatives. Radial derivatives of functions
     *  are multiplied by \f$ x \f$.\n
     *  1-st dimension: index of radial point \n
     *  2-nd dimension: index of radial function \n
     *  3-nd dimension: 0 - function itself, 1 - radial derivative r*(du/dr) */
    mdarray<double, 3> radial_functions_;

    /// Surface derivatives of AW radial functions.
    mdarray<double, 2> surface_derivatives_;

    /// Spherical part of radial integral.
    mdarray<double, 2> h_spherical_integrals_;

    /// Overlap integrals.
    mdarray<double, 3> o_radial_integrals_;

    /// Overlap integrals for IORA relativistic treatment.
    mdarray<double, 2> o1_radial_integrals_;

    /// Spin-orbit interaction integrals.
    mdarray<double, 3> so_radial_integrals_;

    std::map<std::pair<int, int>, enu_search_t> enu_search_;

    /// List of radial descriptor sets used to construct augmented waves.
    std::vector<radial_solution_descriptor_set> aw_descriptors_;

    /// List of radial descriptor sets used to construct local orbitals.
    std::vector<local_orbital_descriptor> lo_descriptors_;

    lapw_radial_basis_t(Atom_type const& atom_type__, relativity_t rel__, std::vector<double> spherical_potential__)
      : atom_type_(atom_type__)
      , rel_(rel__)
      , spherical_potential_(spherical_potential__)
    {
        int nl        = atom_type_.indexr().lmax() + 1;
        int max_order = atom_type_.indexr().max_order();
        int nrf       = atom_type_.mt_radial_basis_size();

        radial_functions_ = mdarray<double, 3>({atom_type_.num_mt_points(), nrf, 2});

        surface_derivatives_ = mdarray<double, 2>({3, atom_type_.mt_radial_basis_size() - atom_type_.num_lo_descriptors()});

        h_spherical_integrals_ = mdarray<double, 2>({nrf, nrf});
        h_spherical_integrals_.zero();

        o_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
        o_radial_integrals_.zero();

        so_radial_integrals_ = mdarray<double, 3>({nl, max_order, max_order});
        so_radial_integrals_.zero();

        if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
            o1_radial_integrals_ = mdarray<double, 2>({nrf, nrf});
            o1_radial_integrals_.zero();
        }

        /* copy descriptors because enu is different between atom classes */
        for (int i = 0; i < atom_type_.num_aw_descriptors(); i++) {
            aw_descriptors_.push_back(atom_type_.aw_descriptor(i));
        }

        for (int i = 0; i < atom_type_.num_lo_descriptors(); i++) {
            lo_descriptors_.push_back(atom_type_.lo_descriptor(i));
        }

        /* find which aw functions need auto enu */
        for (int l = 0; l < atom_type_.num_aw_descriptors(); l++) {
            for (auto const& d : aw_descriptors_[l]) {
                if (d.auto_enu) {
                    enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
                }
            }
        }

        /* find which lo functions need auto enu */
        for (int idxlo = 0; idxlo < atom_type_.num_lo_descriptors(); idxlo++) {
            for (auto const& d : lo_descriptors_[idxlo].rsd_set) {
                if (d.auto_enu) {
                    enu_search_[{d.n, d.l}] = enu_search_t{d.enu, d.enu, d.enu, d.auto_enu};
                }
            }
        }
    }

    int
    find_enu()
    {
        PROFILE("sirius::lapw_radial_basis_t::find_enu");

        /* unroll {n,l} -> enu map to enable omp for loop */
        std::vector<std::pair<std::pair<int, int>, enu_search_t>> nl_enu_vec(enu_search_.begin(), enu_search_.end());

        std::vector<std::string> errors(nl_enu_vec.size());
        std::vector<int> status(nl_enu_vec.size(), 0);

        #pragma omp parallel for
        for (size_t i = 0; i < nl_enu_vec.size(); i++) {
            int n = nl_enu_vec[i].first.first;
            int l = nl_enu_vec[i].first.second;
            try {
                nl_enu_vec[i].second = sirius::find_enu(rel_, atom_type_.zn(), n, l, atom_type_.radial_grid(),
                                                        spherical_potential_, nl_enu_vec[i].second);
                status[i]            = 1;
            } catch (std::exception const& e) {
                errors[i] = e.what();
            }
        }
        int ierr{0};
        std::stringstream s;
        s << "lapw_radial_basis_t::find_enu()" << std::endl
          //<< "  atom symmetry class id : " << id_ << std::endl
          << "  atom type label        : " << atom_type_.label() << std::endl
          << "  atom symbol            : " << atom_type_.symbol() << std::endl;
        for (size_t i = 0; i < nl_enu_vec.size(); i++) {
            if (status[i]) {
                enu_search_[nl_enu_vec[i].first] = nl_enu_vec[i].second;
            } else {
                ierr++;
                s << errors[i] << std::endl;
            }
        }
        if (ierr) {
            RTE_WARNING(s);
        }

        /* update the {n,l} -> enu map */
        for (auto& e : nl_enu_vec) {
            enu_search_[e.first] = e.second;
        }

        /* update AW linearization energies */
        for (int l = 0; l < atom_type_.num_aw_descriptors(); l++) {
            for (auto& d : this->aw_descriptors_[l]) {
                if (d.auto_enu) {
                    d.enu = enu_search_[{d.n, d.l}].enu;
                }
            }
        }
        /* update LO linearization energies */
        for (int idxlo = 0; idxlo < atom_type_.num_lo_descriptors(); idxlo++) {
            for (auto& d : this->lo_descriptors_[idxlo].rsd_set) {
                if (d.auto_enu) {
                    d.enu = enu_search_[{d.n, d.l}].enu;
                }
            }
        }

        return ierr;
    }

    int
    generate_aw_radial_functions()
    {
        int nmtp = atom_type_.num_mt_points();

        Radial_solver solver(atom_type_.zn(), spherical_potential_, atom_type_.radial_grid());

        struct compute_all_orders_result
        {
            bool success{false};
            std::string error;
        };

        auto compute_all_orders = [&](int l, double enu_shift) -> compute_all_orders_result {
            Spline<double> s(atom_type_.radial_grid());

            compute_all_orders_result r;
            std::stringstream sinfo;
            sinfo << " l = " << l << ", enu_shift = " << enu_shift << std::endl;

            for (int order = 0; order < (int)atom_type_.aw_descriptor(l).size(); order++) {
                sinfo << "  order = " << order << std::endl;
                auto rsd = this->aw_descriptors_[l][order];

                auto idxrf = atom_type_.indexr().index_of(angular_momentum(l), order);

                try {
                    /* integrate radial equation forward and find radial solution */
                    auto result = solver.solve(rel_, rsd.dme, rsd.l, rsd.enu + enu_shift);
                    for (int ir = 0; ir < nmtp; ir++) {
                        radial_functions_(ir, idxrf, 0) = result.p[ir];
                        radial_functions_(ir, idxrf, 1) = result.rdudr[ir];
                    }
                    for (int i : {0, 1, 2}) {
                        surface_derivatives_(i, idxrf) = result.uderiv[i];
                    }
                } catch (std::exception const& e) {
                    r.success = false;
                    sinfo << e.what();
                    r.error = sinfo.str();
                    return r;
                }

                /* orthogonalize to previous radial functions */
                for (int order1 = 0; order1 < order; order1++) {
                    auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);

                    for (int ir = 0; ir < nmtp; ir++) {
                        s(ir) = radial_functions_(ir, idxrf, 0) * radial_functions_(ir, idxrf1, 0);
                    }

                    /* <u_{\nu'}|u_{\nu}> */
                    double ovlp = s.interpolate().integrate(0);

                    for (int ir = 0; ir < nmtp; ir++) {
                        radial_functions_(ir, idxrf, 0) -= radial_functions_(ir, idxrf1, 0) * ovlp;
                        radial_functions_(ir, idxrf, 1) -= radial_functions_(ir, idxrf1, 1) * ovlp;
                    }
                    for (int i : {0, 1, 2}) {
                        surface_derivatives_(i, idxrf) -= surface_derivatives_(i, idxrf1) * ovlp;
                    }
                }

                /* normalize orthogonal function */
                for (int ir = 0; ir < nmtp; ir++) {
                    s(ir) = std::pow(radial_functions_(ir, idxrf, 0), 2);
                }
                auto norm = s.interpolate().integrate(0);

                /* in case of linear dependency return failure */
                if (std::abs(norm) < 1e-8) {
                    r.success = false;
                    sinfo << "  linear dependent radial function" << std::endl << "  norm : " << norm << std::endl;
                    r.error = sinfo.str();
                    return r;
                }

                norm = 1.0 / std::sqrt(norm);

                for (int ir = 0; ir < nmtp; ir++) {
                    radial_functions_(ir, idxrf, 0) *= norm;
                    radial_functions_(ir, idxrf, 1) *= norm;
                }
                /* aw radial function can't be zero at MT boundary */
                if (std::abs(radial_functions_(nmtp - 1, idxrf, 0)) < 1e-2) {
                    r.success = false;
                    sinfo << "  radial function is zero at the muffin-tin boundary"
                          << "  value : " << radial_functions_(nmtp - 1, idxrf, 0);
                    r.error = sinfo.str();
                    return r;
                }
                for (int i : {0, 1, 2}) {
                    surface_derivatives_(i, idxrf) *= norm;
                }
            } // order
            r.success = true;
            return r;
        };

        std::vector<std::string> errors(atom_type_.num_aw_descriptors());
        std::vector<int> status(atom_type_.num_aw_descriptors(), 0);

        #pragma omp parallel for schedule(dynamic, 1) default(shared)
        for (int l = 0; l < atom_type_.num_aw_descriptors(); l++) {
            /* take first radial function */
            auto rsd = aw_descriptors_[l][0];
            /* try to increase linearisation energy several times in order to find linearly independent
             * radial functions */
            compute_all_orders_result r;
            /* Enu for this level was searched; this should not produce degenerate radial functions */
            double e_shift{0.0};
            if (rsd.auto_enu) {
                r = compute_all_orders(l, e_shift);
            } else {
                /* for high l values, Enu is typically set in the species files and is not searched;
                 * in case of trouble with them we need to increase linearisation energies */
                double de{0.1};
                for (int k = 0; k < 100; k++) {
                    r = compute_all_orders(l, e_shift);
                    if (r.success) {
                        break;
                    } else {
                        e_shift += de;
                        de *= 1.1;
                    }
                }
            }

            if (r.success) {
                status[l] = 1;
                /* divide by r */
                for (int order = 0; order < (int)atom_type_.aw_descriptor(l).size(); order++) {
                    auto idxrf = atom_type_.indexr().index_of(angular_momentum(l), order);
                    for (int ir = 0; ir < nmtp; ir++) {
                        radial_functions_(ir, idxrf, 0) *= atom_type_.radial_grid().x_inv(ir);
                    }
                }
            } else {
                errors[l] = r.error;
            }
        } // l

        int ierr{0};
        std::stringstream s;
        s << "lapw_radial_basis_t::generate_aw_radial_functions()" << std::endl
          //<< "  atom symmetry class id : " << id_ << std::endl
          << "  atom type label        : " << atom_type_.label() << std::endl
          << "  atom symbol            : " << atom_type_.symbol() << std::endl;
        for (int i = 0; i < atom_type_.num_aw_descriptors(); i++) {
            if (!status[i]) {
                ierr++;
                s << errors[i] << std::endl;
            }
        }
        if (ierr) {
            RTE_WARNING(s);
        }
        return ierr;
    }

    int
    generate_lo_radial_functions()
    {
        int nmtp = atom_type_.num_mt_points();

        Radial_solver solver(atom_type_.zn(), spherical_potential_, atom_type_.radial_grid());

        bool found{true};

        #pragma omp parallel for schedule(dynamic, 1)
        for (int idxlo = 0; idxlo < atom_type_.num_lo_descriptors(); idxlo++) {
            Spline<double> s(atom_type_.radial_grid());
            double a[3][3];
            double rderiv[3][3];

            /* number of radial functions */
            int num_rf = static_cast<int>(this->lo_descriptors_[idxlo].rsd_set.size());
            RTE_ASSERT(num_rf <= 3);

            std::vector<std::vector<double>> u(num_rf);
            std::vector<std::vector<double>> rdudr(num_rf);

            for (int irf = 0; irf < num_rf; irf++) {
                auto rsd = this->lo_descriptors_[idxlo].rsd_set[irf];

                auto result = solver.solve(rel_, rsd.dme, rsd.l, rsd.enu);

                u[irf]     = result.p;
                rdudr[irf] = result.rdudr;

                /* divide by r */
                for (int ir = 0; ir < nmtp; ir++) {
                    /* store u(r) = p(r)/r */
                    u[irf][ir] *= atom_type_.radial_grid().x_inv(ir);
                }

                for (int i = 0; i < num_rf; i++) {
                    /* matrix of derivatives */
                    a[irf][i] = rderiv[irf][i] = result.uderiv[i];
                }
            }

            double b[]    = {0, 0, 0};
            b[num_rf - 1] = 1.0;

            int info = la::wrap(la::lib_t::lapack).gesv(num_rf, 1, &a[0][0], 3, b, 3);

            if (info) {
                std::stringstream s;
                s << "a[i][j] = ";
                for (int i = 0; i < num_rf; i++) {
                    for (int j = 0; j < num_rf; j++) {
                        s << rderiv[i][j] << " ";
                    }
                }
                s << std::endl;
                s << "atom: " << atom_type_.label() << std::endl
                  << "zn: " << atom_type_.zn() << std::endl
                  << "l: " << this->lo_descriptors_[idxlo].am.l() << std::endl;
                s << "gesv returned " << info;
                RTE_THROW(s);
            }

            /* index of local orbital radial function */
            auto idxrf = atom_type_.indexr().index_of(rf_lo_index(idxlo));
            /* take linear combination of radial solutions */
            for (int order = 0; order < num_rf; order++) {
                for (int ir = 0; ir < nmtp; ir++) {
                    /* u(r) function */
                    radial_functions_(ir, idxrf, 0) += b[order] * u[order][ir];
                    /* r(du/dr) function */
                    radial_functions_(ir, idxrf, 1) += b[order] * rdudr[order][ir];
                }
            }

            /* find norm of constructed local orbital */
            for (int ir = 0; ir < nmtp; ir++) {
                s(ir) = std::pow(radial_functions_(ir, idxrf, 0), 2);
            }
            double norm = 1.0 / std::sqrt(s.interpolate().integrate(2));

            /* normalize */
            for (int ir = 0; ir < nmtp; ir++) {
                radial_functions_(ir, idxrf, 0) *= norm;
                radial_functions_(ir, idxrf, 1) *= norm;
            }

            if (std::abs(radial_functions_(nmtp - 1, idxrf, 0)) > 1e-10) {
                std::stringstream s;
                s << "local orbital " << idxlo << " is not zero at MT boundary" << std::endl
                  //<< "  atom symmetry class id : " << id() << " (" << atom_type().symbol() << ")" << std::endl
                  << "  value : " << radial_functions_(nmtp - 1, idxrf, 0) << std::endl
                  << "  number of MT points: " << nmtp << std::endl
                  << "  MT radius: " << atom_type_.radial_grid().last() << std::endl
                  << "  matrix of derivatives:" << std::endl;
                for (int i = 0; i < num_rf; i++) {
                    for (int j = 0; j < num_rf; j++) {
                        s << rderiv[i][j] << " ";
                    }
                    s << std::endl;
                }
                s << "  b_coeffs: ";
                for (int j = 0; j < num_rf; j++) {
                    s << b[j] << " ";
                }
                s << std::endl;
                s << "  norm: " << norm << std::endl;
                double d{0};
                for (int i = 0; i < num_rf; i++) {
                    d += b[i] * rderiv[i][0];
                }
                s << "  expected value at MT boundary from the linear equations: " << d << std::endl;
                for (int i = 0; i < num_rf; i++) {
                    s << " rderiv, u: " << rderiv[i][0] << " " << u[i][nmtp - 1] << std::endl;
                }
                RTE_WARNING(s);
            }
        }

        if (found && atom_type_.parameters().cfg().control().verification() > 0 && atom_type_.num_lo_descriptors() > 0) {
            check_lo_linear_independence(0.0001);
        }

        return found ? 0 : 1;
    }

    std::vector<int>
    check_lo_linear_independence(double tol__) const
    {
        int nmtp = atom_type_.num_mt_points();
        int nlo  = atom_type_.num_lo_descriptors();

        Spline<double> s(atom_type_.radial_grid());
        la::dmatrix<double> loprod(nlo, nlo);
        loprod.zero();
        for (int idxlo1 = 0; idxlo1 < nlo; idxlo1++) {

            int idxrf1 = atom_type_.indexr().index_of(rf_lo_index(idxlo1));

            for (int idxlo2 = 0; idxlo2 < nlo; idxlo2++) {

                int idxrf2 = atom_type_.indexr().index_of(rf_lo_index(idxlo2));

                if (lo_descriptors_[idxlo1].am == lo_descriptors_[idxlo2].am) {

                    for (int ir = 0; ir < nmtp; ir++) {
                        s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                    }
                    loprod(idxlo1, idxlo2) = s.interpolate().integrate(2);
                }
            }
        }

        mdarray<double, 2> ovlp({nlo, nlo});
        copy(loprod, ovlp);

        auto stdevp = la::Eigensolver_factory("lapack");

        std::vector<double> loprod_eval(nlo);
        la::dmatrix<double> loprod_evec(nlo, nlo);

        stdevp->solve(nlo, loprod, &loprod_eval[0], loprod_evec);

        if (loprod_eval[0] < tol__) {
            std::cout << "local orbitals are almost linearly dependent" << std::endl
                      << "overlap matrix" << std::endl;
            for (int i = 0; i < nlo; i++) {
                for (int j = 0; j < nlo; j++) {
                    std::cout << ovlp(i, j) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "overlap matrix eigen-values:" << std::endl;
            for (int i = 0; i < nlo; i++) {
                std::cout << loprod_eval[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "smallest eigenvalue: " << loprod_eval[0] << std::endl;
        }

        std::vector<int> inc(nlo, 0);

        /* try all local orbitals */
        for (int i = 0; i < nlo; i++) {
            inc[i] = 1;

            std::vector<int> ilo;
            for (int j = 0; j < nlo; j++) {
                if (inc[j] == 1) {
                    ilo.push_back(j);
                }
            }

            std::vector<double> eval(ilo.size());
            la::dmatrix<double> evec(static_cast<int>(ilo.size()), static_cast<int>(ilo.size()));
            la::dmatrix<double> tmp(static_cast<int>(ilo.size()), static_cast<int>(ilo.size()));
            for (int j1 = 0; j1 < (int)ilo.size(); j1++) {
                for (int j2 = 0; j2 < (int)ilo.size(); j2++) {
                    tmp(j1, j2) = ovlp(ilo[j1], ilo[j2]);
                }
            }

            stdevp->solve(static_cast<int>(ilo.size()), tmp, &eval[0], evec);

            if (eval[0] < tol__) {
                std::cout << "local orbital " << i << " can be removed" << std::endl;
                inc[i] = 0;
            }
        }
        return inc;
    }

    int
    generate_radial_functions()
    {
        PROFILE("sirius::lapw_radial_basis_t::generate_radial_functions");

        //if (update_enu__) {
        //    auto ierr_enu = this->find_enu(rel__);
        //    /* write spherical potential */
        //    if (ierr_enu && atom_type().parameters().cfg().control().save_rf()) {
        //        save_spherical_potential();
        //    }
        //}

        auto ierr_aw = this->generate_aw_radial_functions();
        if (ierr_aw) {
            std::stringstream s;
            s << "generate_aw_radial_functions() failed";
            RTE_WARNING(s);
        }

        auto ierr_lo = this->generate_lo_radial_functions();
        if (ierr_lo) {
            std::stringstream s;
            s << "generate_lo_radial_functions() failed";
            RTE_WARNING(s);
        }

        //if (ierr_aw + ierr_lo == 0) {
        //    copy(rf, radial_functions_);
        //    copy(sd, surface_derivatives_);
        //    if (atom_type().parameters().cfg().control().ortho_rf()) {
        //        orthogonalize_radial_functions();
        //    }
        //} else {
        //    std::stringstream s;
        //    s << "radial functions for atom class " << id_ << " were not found";
        //    RTE_WARNING(s);
        //    if (atom_type().parameters().cfg().control().save_rf()) {
        //        save_spherical_potential();
        //    }
        //}

        //if (atom_type().parameters().cfg().control().save_rf()) {
        //    static int count{0};
        //    std::string fname =
        //            "radial_functions_class_" + std::to_string(id_) + "_step_" + std::to_string(count) + ".json";
        //    save_radial_functions(fname);
        //    count++;
        //}
        return ierr_aw + ierr_lo;
    }

    void generate_radial_integrals()
    {
        PROFILE("sirius::lapw_radial_basis_t::generate_radial_integrals");

        int nmtp = atom_type_.num_mt_points();

        double a2 = sq_alpha_half;
        if (rel_ == relativity_t::none) {
            a2 = 0;
        }

        h_spherical_integrals_.zero();
        #pragma omp parallel default(shared)
        {
            Spline<double> s(atom_type_.radial_grid());
            #pragma omp for
            for (int i1 = 0; i1 < atom_type_.mt_radial_basis_size(); i1++) {
                for (int i2 = 0; i2 < atom_type_.mt_radial_basis_size(); i2++) {
                    /* for spherical part of potential integrals are diagonal in l */
                    if (atom_type_.indexr(i1).am.l() == atom_type_.indexr(i2).am.l()) {
                        int ll = atom_type_.indexr(i1).am.l() * (atom_type_.indexr(i1).am.l() + 1);
                        for (int ir = 0; ir < nmtp; ir++) {
                            double Minv = 1.0 / (1 - spherical_potential_[ir] * a2);
                            /* u_1(r) * u_2(r) */
                            double t0 = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 0);
                            /* r*u'_1(r) * r*u'_2(r) */
                            double t1 = radial_functions_(ir, i1, 1) * radial_functions_(ir, i2, 1);
                            s(ir)     = 0.5 * t1 * Minv +
                                    t0 * (0.5 * ll * Minv +
                                          spherical_potential_[ir] * std::pow(atom_type_.radial_grid(ir), 2));
                        }
                        h_spherical_integrals_(i1, i2) = s.interpolate().integrate(0) / y00;
                    }
                }
            }
        }

        o_radial_integrals_.zero();
        #pragma omp parallel default(shared)
        {
            Spline<double> s(atom_type_.radial_grid());
            #pragma omp for
            for (int l = 0; l <= atom_type_.indexr().lmax(); l++) {
                int nrf = atom_type_.indexr().max_order(l);

                for (int order1 = 0; order1 < nrf; order1++) {
                    auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);
                    for (int order2 = 0; order2 < nrf; order2++) {
                        auto idxrf2 = atom_type_.indexr().index_of(angular_momentum(l), order2);
                        if (order1 == order2) {
                            o_radial_integrals_(l, order1, order2) = 1.0;
                        } else {
                            if (atom_type_.parameters().cfg().settings().simple_lapw_ri()) {
                                for (int ir = 0; ir < nmtp; ir++) {
                                    s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) *
                                            std::pow(atom_type_.radial_grid(ir), 2);
                                }
                                o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(0);
                            } else {
                                for (int ir = 0; ir < nmtp; ir++) {
                                    s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0);
                                }
                                o_radial_integrals_(l, order1, order2) = s.interpolate().integrate(2);
                            }
                        }
                    }
                }
            }
        }
        if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
            o1_radial_integrals_.zero();
            #pragma omp parallel for
            for (int i1 = 0; i1 < atom_type_.mt_radial_basis_size(); i1++) {
                Spline<double> s(atom_type_.radial_grid());
                for (int i2 = 0; i2 < atom_type_.mt_radial_basis_size(); i2++) {
                    /* for spherical part of potential integrals are diagonal in l */
                    if (atom_type_.indexr(i1).am.l() == atom_type_.indexr(i2).am.l()) {
                        int ll = atom_type_.indexr(i1).am.l() * (atom_type_.indexr(i1).am.l() + 1);
                        for (int ir = 0; ir < nmtp; ir++) {
                            double Minv2 = std::pow(1 - spherical_potential_[ir] * a2, -2);
                            /* u_1(r) * u_2(r) */
                            double t0 = radial_functions_(ir, i1, 0) * radial_functions_(ir, i2, 0);
                            /* r*u'_1(r) * r*u'_2(r) */
                            double t1 = radial_functions_(ir, i1, 1) * radial_functions_(ir, i2, 1);
                            s(ir)     = a2 * 0.5 * Minv2 * (t1 + t0 * ll);
                        }
                        o1_radial_integrals_(i1, i2) = s.interpolate().integrate(0);
                    }
                }
            }
        }

        if (atom_type_.parameters().so_correction()) {
            double soc = std::pow(2 * speed_of_light, -2);

            Spline<double> s(atom_type_.radial_grid());
            Spline<double> s1(atom_type_.radial_grid());
            Spline<double> ve(atom_type_.radial_grid());

            for (int i = 0; i < nmtp; i++) {
                ve(i) = spherical_potential_[i] + atom_type_.zn() / atom_type_.radial_grid(i);
            }
            ve.interpolate();

            so_radial_integrals_.zero();
            for (int l = 0; l <= atom_type_.indexr().lmax(); l++) {
                int nrf = atom_type_.indexr().max_order(l);

                for (int order1 = 0; order1 < nrf; order1++) {
                    auto idxrf1 = atom_type_.indexr().index_of(angular_momentum(l), order1);
                    for (int order2 = 0; order2 < nrf; order2++) {
                        auto idxrf2 = atom_type_.indexr().index_of(angular_momentum(l), order2);

                        for (int ir = 0; ir < nmtp; ir++) {
                            double M = 1.0 - 2 * soc * spherical_potential_[ir];
                            /* first part <f| dVe / dr |f'> */
                            s(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * soc *
                                    ve.deriv(1, ir) / pow(M, 2);

                            /* second part <f| d(z/r) / dr |f'> */
                            s1(ir) = radial_functions_(ir, idxrf1, 0) * radial_functions_(ir, idxrf2, 0) * soc *
                                     atom_type_.zn() / pow(M, 2);
                        }
                        s.interpolate();
                        s1.interpolate();

                        so_radial_integrals_(l, order1, order2) = s.integrate(1) + s1.integrate(-1);
                    }
                }
            }
        }
    }

    void
    sync_radial_functions(mpi::Communicator const& comm__, int const rank__)
    {
        /* don't broadcast Hamiltonian radial functions, because they are used locally */
        int size = (int)(radial_functions_.size(0) * radial_functions_.size(1));
        comm__.bcast(radial_functions_.at(memory_t::host), size, rank__);
        comm__.bcast(surface_derivatives_.at(memory_t::host), (int)surface_derivatives_.size(), rank__);
    }

    void
    sync_radial_integrals(mpi::Communicator const& comm__, int const rank__)
    {
        comm__.bcast(h_spherical_integrals_.at(memory_t::host), (int)h_spherical_integrals_.size(), rank__);
        comm__.bcast(o_radial_integrals_.at(memory_t::host), (int)o_radial_integrals_.size(), rank__);
        comm__.bcast(so_radial_integrals_.at(memory_t::host), (int)so_radial_integrals_.size(), rank__);
        if (atom_type_.parameters().valence_relativity() == relativity_t::iora) {
            comm__.bcast(o1_radial_integrals_.at(memory_t::host), (int)o1_radial_integrals_.size(), rank__);
        }
    }
};

class LAPW_radial_basis 
{
  private:
    std::vector<lapw_radial_basis_t> radial_basis_of_symmetry_class;

  public:
    LAPW_radial_basis()
    {
    }
    LAPW_radial_basis(Unit_cell const& unit_cell__, relativity_t rel__, std::vector<std::vector<double>> vs__)
    {
        for (int ic = 0; ic < unit_cell__.num_atom_symmetry_classes(); ic++) {
            radial_basis_of_symmetry_class.emplace_back(unit_cell__.atom_symmetry_class(ic).atom_type(), rel__, vs__[ic]);
        }

        auto spl_num_symcls = splindex_block<atom_symmetry_class_index_t>(
            unit_cell__.num_atom_symmetry_classes(), n_blocks(unit_cell__.comm().size()), block_id(unit_cell__.comm().rank()));

        int ierr{0};
        for (auto it : spl_num_symcls) {
            radial_basis_of_symmetry_class[it.i].find_enu();
            ierr += radial_basis_of_symmetry_class[it.i].generate_radial_functions();
        }
        unit_cell__.comm().allreduce<int, mpi::op_t::max>(&ierr, 1);
        if (ierr) {
            RTE_THROW("lapw radial basis was not generated");
        }
        for (auto it : spl_num_symcls) {
            radial_basis_of_symmetry_class[it.i].generate_radial_integrals();
        }

        for (int ic = 0; ic < unit_cell__.num_atom_symmetry_classes(); ic++) {
            int rank = spl_num_symcls.location(typename atom_symmetry_class_index_t::global(ic)).ib;
            radial_basis_of_symmetry_class[ic].sync_radial_functions(unit_cell__.comm(), rank);
        }
    }

};

}

#endif
