/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file xc_mt.cpp
 *
 *  \brief Generate XC potential in the muffin-tins.
 */

#include <vector>

#include "potential.hpp"
#include "core/typedefs.hpp"
#include "core/omp.hpp"
#include "core/profiler.hpp"
#include "xc_functional.hpp"

namespace sirius {

void
xc_mt_nonmagnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional> const& xc_func__,
                  Flm const& rho_lm__, Ftp& rho_tp__, Flm& vxc_lm__, Flm& exc_lm__, bool use_lapl__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc.is_gga() || ixc.is_vdw()) {
            is_gga = true;
        }
    }

    Ftp exc_tp(sht__.num_points(), rgrid__);
    Ftp vxc_tp(sht__.num_points(), rgrid__);

    RTE_ASSERT(rho_tp__.size() == vxc_tp.size());
    RTE_ASSERT(rho_tp__.size() == exc_tp.size());

    Ftp grad_rho_grad_rho_tp;
    Ftp vsigma_tp;
    Ftp lapl_rho_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm__);
        grad_rho_tp      = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(sht__, grad_rho_lm[x], grad_rho_tp[x]);
        }
        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;
        RTE_ASSERT(rho_tp__.size() == grad_rho_grad_rho_tp.size());

        vsigma_tp = Ftp(sht__.num_points(), rgrid__);
        RTE_ASSERT(rho_tp__.size() == vsigma_tp.size());
        if (use_lapl__) {
            /* backward transform Laplacian from Rlm to (theta, phi) */
            lapl_rho_tp = transform(sht__, laplacian(rho_lm__));
            RTE_ASSERT(lapl_rho_tp.size() == rho_tp__.size());
        }
    }

    for (auto& ixc : xc_func__) {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            ixc.get_lda(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                        vxc_tp.at(memory_t::host), exc_tp.at(memory_t::host));
        }
        /* if this is a GGA functional */
        if (ixc.is_gga()) {

            /* compute vrho and vsigma */
            ixc.get_gga(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                        grad_rho_grad_rho_tp.at(memory_t::host), vxc_tp.at(memory_t::host),
                        vsigma_tp.at(memory_t::host), exc_tp.at(memory_t::host));

            if (use_lapl__) {
                vxc_tp -= 2.0 * vsigma_tp * lapl_rho_tp;

                /* compute gradient of vsgima in spherical harmonics */
                auto grad_vsigma_lm = gradient(transform(sht__, vsigma_tp));

                /* backward transform gradient from Rlm to (theta, phi) */
                Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_tp(sht__.num_points(), rgrid__);
                for (int x = 0; x < 3; x++) {
                    transform(sht__, grad_vsigma_lm[x], grad_vsigma_tp[x]);
                }

                /* compute scalar product of two gradients */
                auto grad_vsigma_grad_rho_tp = grad_vsigma_tp * grad_rho_tp;
                /* add remaining term to Vxc */
                vxc_tp -= 2.0 * grad_vsigma_grad_rho_tp;
            } else {
                Spheric_vector_function<function_domain_t::spectral, double> vsigma_grad_rho_lm(sht__.lmmax(), rgrid__);
                for (int x : {0, 1, 2}) {
                    auto vsigma_grad_rho_tp = vsigma_tp * grad_rho_tp[x];
                    transform(sht__, vsigma_grad_rho_tp, vsigma_grad_rho_lm[x]);
                }
                auto div_vsigma_grad_rho_tp = transform(sht__, divergence(vsigma_grad_rho_lm));
                /* add remaining term to Vxc */
                vxc_tp -= 2.0 * div_vsigma_grad_rho_tp;
            }
        }
        exc_lm__ += transform(sht__, exc_tp);
        vxc_lm__ += transform(sht__, vxc_tp);
    } // ixc
}

void
xc_mt_magnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, int num_mag_dims__,
               std::vector<XC_functional> const& xc_func__, std::vector<Ftp> const& rho_tp__, std::vector<Flm*> vxc__,
               Flm& exc__, bool use_lapl__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc.is_gga() || ixc.is_vdw()) {
            is_gga = true;
        }
    }

    Ftp exc_tp(sht__.num_points(), rgrid__);
    Ftp vxc_tp(sht__.num_points(), rgrid__);

    /* convert to rho_up, rho_dn */
    Ftp rho_dn_tp(sht__.num_points(), rgrid__);
    Ftp rho_up_tp(sht__.num_points(), rgrid__);
    /* loop over radial grid points */
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        /* loop over points on the sphere */
        for (int itp = 0; itp < sht__.num_points(); itp++) {
            r3::vector<double> m;
            for (int j = 0; j < num_mag_dims__; j++) {
                m[j] = rho_tp__[1 + j](itp, ir);
            }
            auto rud = get_rho_up_dn(num_mag_dims__, rho_tp__[0](itp, ir), m);

            /* compute "up" and "dn" components */
            rho_up_tp(itp, ir) = rud.first;
            rho_dn_tp(itp, ir) = rud.second;
        }
    }
    /* transform from (theta, phi) to Rlm */
    auto rho_up_lm = transform(sht__, rho_up_tp);
    auto rho_dn_lm = transform(sht__, rho_dn_tp);

    std::vector<Ftp> bxc_tp(num_mag_dims__);

    Ftp vxc_up_tp(sht__.num_points(), rgrid__);
    Ftp vxc_dn_tp(sht__.num_points(), rgrid__);
    for (int j = 0; j < num_mag_dims__; j++) {
        bxc_tp[j] = Ftp(sht__.num_points(), rgrid__);
    }

    Ftp grad_rho_up_grad_rho_up_tp;
    Ftp grad_rho_up_grad_rho_dn_tp;
    Ftp grad_rho_dn_grad_rho_dn_tp;
    Ftp vsigma_uu_tp;
    Ftp vsigma_ud_tp;
    Ftp vsigma_dd_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_vsigma_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_vsigma_tp;
    Ftp lapl_rho_up_tp;
    Ftp lapl_rho_dn_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);
        grad_rho_up_tp      = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        grad_rho_dn_tp      = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        grad_rho_up_vsigma_tp =
                Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        grad_rho_dn_vsigma_tp =
                Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(sht__, grad_rho_up_lm[x], grad_rho_up_tp[x]);
            transform(sht__, grad_rho_dn_lm[x], grad_rho_dn_tp[x]);
        }
        /* compute density gradient products */
        grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
        grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
        grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;

        vsigma_uu_tp = Ftp(sht__.num_points(), rgrid__);
        vsigma_ud_tp = Ftp(sht__.num_points(), rgrid__);
        vsigma_dd_tp = Ftp(sht__.num_points(), rgrid__);

        if (use_lapl__) {
            /* backward transform Laplacian from Rlm to (theta, phi) */
            lapl_rho_up_tp = transform(sht__, laplacian(rho_up_lm));
            lapl_rho_dn_tp = transform(sht__, laplacian(rho_dn_lm));
        }
    }

    for (auto& ixc : xc_func__) {
        if (ixc.is_lda()) {
            ixc.get_lda(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                        rho_dn_tp.at(memory_t::host), vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host),
                        exc_tp.at(memory_t::host));
        }
        if (ixc.is_gga()) {
            /* get the vrho and vsigma */
            ixc.get_gga(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                        rho_dn_tp.at(memory_t::host), grad_rho_up_grad_rho_up_tp.at(memory_t::host),
                        grad_rho_up_grad_rho_dn_tp.at(memory_t::host), grad_rho_dn_grad_rho_dn_tp.at(memory_t::host),
                        vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host), vsigma_uu_tp.at(memory_t::host),
                        vsigma_ud_tp.at(memory_t::host), vsigma_dd_tp.at(memory_t::host), exc_tp.at(memory_t::host));

            if (use_lapl__) {
                vxc_up_tp -= 2.0 * vsigma_uu_tp * lapl_rho_up_tp + vsigma_ud_tp * lapl_rho_dn_tp;
                vxc_dn_tp -= 2.0 * vsigma_dd_tp * lapl_rho_dn_tp + vsigma_ud_tp * lapl_rho_up_tp;

                /* compute gradients of vsgimas in spherical harmonics */
                auto grad_vsigma_uu_lm = gradient(transform(sht__, vsigma_uu_tp));
                auto grad_vsigma_ud_lm = gradient(transform(sht__, vsigma_ud_tp));
                auto grad_vsigma_dd_lm = gradient(transform(sht__, vsigma_dd_tp));

                /* backward transform gradient from Rlm to (theta, phi) */
                Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_uu_tp(sht__.num_points(),
                                                                                              rgrid__);
                Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_ud_tp(sht__.num_points(),
                                                                                              rgrid__);
                Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_dd_tp(sht__.num_points(),
                                                                                              rgrid__);
                for (int x = 0; x < 3; x++) {
                    transform(sht__, grad_vsigma_uu_lm[x], grad_vsigma_uu_tp[x]);
                    transform(sht__, grad_vsigma_ud_lm[x], grad_vsigma_ud_tp[x]);
                    transform(sht__, grad_vsigma_dd_lm[x], grad_vsigma_dd_tp[x]);
                }

                /* compute scalar product of two gradients */
                auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
                auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;
                auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
                auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;

                vxc_up_tp -= 2.0 * grad_vsigma_uu_grad_rho_up_tp + grad_vsigma_ud_grad_rho_dn_tp;
                vxc_dn_tp -= 2.0 * grad_vsigma_dd_grad_rho_dn_tp + grad_vsigma_ud_grad_rho_up_tp;

            } else {

                for (int x : {0, 1, 2}) {
                    grad_rho_up_vsigma_tp[x] =
                            (2.0 * vsigma_uu_tp * grad_rho_up_tp[x] + vsigma_ud_tp * grad_rho_dn_tp[x]);
                    grad_rho_dn_vsigma_tp[x] =
                            (2.0 * vsigma_dd_tp * grad_rho_dn_tp[x] + vsigma_ud_tp * grad_rho_up_tp[x]);
                }

                Spheric_vector_function<function_domain_t::spectral, double> grad_rho_up_vsigma_lm(sht__.lmmax(),
                                                                                                   rgrid__);
                Spheric_vector_function<function_domain_t::spectral, double> grad_rho_dn_vsigma_lm(sht__.lmmax(),
                                                                                                   rgrid__);

                for (int x : {0, 1, 2}) {
                    grad_rho_up_vsigma_lm[x] = transform(sht__, grad_rho_up_vsigma_tp[x]);
                    grad_rho_dn_vsigma_lm[x] = transform(sht__, grad_rho_dn_vsigma_tp[x]);
                }

                auto div_grad_rho_up_vsigma_lm = divergence(grad_rho_up_vsigma_lm);
                auto div_grad_rho_dn_vsigma_lm = divergence(grad_rho_dn_vsigma_lm);

                /* add remaining terms to Vxc */
                vxc_up_tp -= transform(sht__, div_grad_rho_up_vsigma_lm);
                vxc_dn_tp -= transform(sht__, div_grad_rho_dn_vsigma_lm);
            }
        }
        /* generate magnetic field and effective potential inside MT sphere */
        for (int ir = 0; ir < rgrid__.num_points(); ir++) {
            for (int itp = 0; itp < sht__.num_points(); itp++) {
                /* Vxc = 0.5 * (V_up + V_dn) */
                vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                /* Bxc = 0.5 * (V_up - V_dn) */
                double bxc = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                /* get the sign between mag and B */
                auto s = sign((rho_up_tp(itp, ir) - rho_dn_tp(itp, ir)) * bxc);

                r3::vector<double> m;
                for (int j = 0; j < num_mag_dims__; j++) {
                    m[j] = rho_tp__[1 + j](itp, ir);
                }
                auto m_len = m.length();
                if (m_len > 1e-8) {
                    for (int j = 0; j < num_mag_dims__; j++) {
                        bxc_tp[j](itp, ir) = std::abs(bxc) * s * m[j] / m_len;
                    }
                } else {
                    for (int j = 0; j < num_mag_dims__; j++) {
                        bxc_tp[j](itp, ir) = 0.0;
                    }
                }
            }
        }
        /* convert magnetic field back to Rlm */
        for (int j = 0; j < num_mag_dims__; j++) {
            *vxc__[j + 1] += transform(sht__, bxc_tp[j]);
        }
        /* forward transform from (theta, phi) to Rlm */
        *vxc__[0] += transform(sht__, vxc_tp);
        exc__ += transform(sht__, exc_tp);
    } // ixc
}

double
xc_mt(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional> const& xc_func__,
      int num_mag_dims__, std::vector<Flm const*> rho__, std::vector<Flm*> vxc__, Flm* exc__, bool use_lapl__)
{
    /* zero the fields */
    exc__->zero();
    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        vxc__[j]->zero();
    }

    std::vector<Ftp> rho_tp(num_mag_dims__ + 1);
    for (int j = 0; j < num_mag_dims__ + 1; j++) {
        /* convert density and magnetization to theta, phi */
        rho_tp[j] = transform(sht__, *rho__[j]);
    }

    /* check if density has negative values */
    double rhomin{0};
    for (int ir = 0; ir < rgrid__.num_points(); ir++) {
        for (int itp = 0; itp < sht__.num_points(); itp++) {
            rhomin = std::min(rhomin, rho_tp[0](itp, ir));
            /* fix negative density */
            if (rho_tp[0](itp, ir) < 0.0) {
                for (int j = 0; j < num_mag_dims__ + 1; j++) {
                    rho_tp[j](itp, ir) = 0.0;
                }
            }
        }
    }

    if (num_mag_dims__ == 0) {
        xc_mt_nonmagnetic(rgrid__, sht__, xc_func__, *rho__[0], rho_tp[0], *vxc__[0], *exc__, use_lapl__);
    } else {
        xc_mt_magnetic(rgrid__, sht__, num_mag_dims__, xc_func__, rho_tp, vxc__, *exc__, use_lapl__);
    }

    return rhomin;
}

void
Potential::xc_mt(Density const& density__, bool use_lapl__)
{
    PROFILE("sirius::Potential::xc_mt");

    #pragma omp parallel for
    for (auto it : unit_cell_.spl_num_atoms()) {
        auto& rgrid = unit_cell_.atom(it.i).radial_grid();
        std::vector<Flm const*> rho(ctx_.num_mag_dims() + 1);
        std::vector<Flm*> vxc(ctx_.num_mag_dims() + 1);
        rho[0] = &density__.rho().mt()[it.i];
        vxc[0] = &xc_potential_->mt()[it.i];
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            rho[j + 1] = &density__.mag(j).mt()[it.i];
            vxc[j + 1] = &effective_magnetic_field(j).mt()[it.i];
        }

        auto rhomin = sirius::xc_mt(rgrid, *sht_, xc_func_, ctx_.num_mag_dims(), rho, vxc,
                                    &xc_energy_density_->mt()[it.i], use_lapl__);
        if (rhomin < 0.0) {
            std::stringstream s;
            s << "[xc_mt] negative charge density " << rhomin << " for atom " << it.i << std::endl
              << "  current Rlm expansion of the charge density may be not sufficient, try to increase lmax"
              << std::endl
              << "  sht.lmax       : " << sht_->lmax() << std::endl
              << "  sht.num_points : " << sht_->num_points();
            RTE_WARNING(s);
        }

        /* z, x, y order */
        std::array<int, 3> comp_map = {2, 0, 1};
        /* add auxiliary magnetic field antiparallel to starting magnetization */
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                effective_magnetic_field(j).mt()[it.i](0, ir) -=
                        aux_bf_(j, it.i) * ctx_.unit_cell().atom(it.i).vector_field()[comp_map[j]];
            }
        }
    } // ialoc
}

} // namespace sirius
