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

/** \file xc.cpp
 *
 *  \brief Generate XC potential.
 */

#include <vector>

#include "potential.hpp"
#include "typedefs.hpp"
#include "utils/profiler.hpp"

namespace sirius {

void Potential::xc_mt_nonmagnetic(Radial_grid<double> const& rgrid,
                                  std::vector<XC_functional>& xc_func,
                                  Spheric_function<function_domain_t::spectral, double> const& rho_lm,
                                  Spheric_function<function_domain_t::spatial, double>& rho_tp,
                                  Spheric_function<function_domain_t::spatial, double>& vxc_tp,
                                  Spheric_function<function_domain_t::spatial, double>& exc_tp)
{
    /* use Laplacian (true) or divergence of gradient (false) */
    bool use_lapl{false};

    bool is_gga = is_gradient_correction();

    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_tp(sht_->num_points(), rgrid);
    Spheric_function<function_domain_t::spatial, double> lapl_rho_tp;
    Spheric_function<function_domain_t::spatial, double> grad_rho_grad_rho_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(*sht_, grad_rho_lm[x], grad_rho_tp[x]);
        }

        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;

        if (use_lapl) {
            lapl_rho_tp = Spheric_function<function_domain_t::spatial, double>(sht_->num_points(), rgrid);
            /* compute Laplacian in Rlm spherical harmonics */
            auto lapl_rho_lm = laplacian(rho_lm);

            /* backward transform Laplacian from Rlm to (theta, phi) */
            transform(*sht_, lapl_rho_lm, lapl_rho_tp);
        }
    }

    exc_tp.zero();
    vxc_tp.zero();

    Spheric_function<function_domain_t::spatial, double> vsigma_tp;
    if (is_gga) {
        vsigma_tp = Spheric_function<function_domain_t::spatial, double>(sht_->num_points(), rgrid);
        vsigma_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func) {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            std::vector<double> exc_t(sht_->num_points());
            std::vector<double> vxc_t(sht_->num_points());
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                ixc.get_lda(sht_->num_points(), &rho_tp(0, ir), &vxc_t[0], &exc_t[0]);
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* add Exc contribution */
                    exc_tp(itp, ir) += exc_t[itp];

                    /* directly add to Vxc */
                    vxc_tp(itp, ir) += vxc_t[itp];
                }
            }
        }
        if (ixc.is_gga()) {
            std::vector<double> exc_t(sht_->num_points());
            std::vector<double> vrho_t(sht_->num_points());
            std::vector<double> vsigma_t(sht_->num_points());
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                ixc.get_gga(sht_->num_points(), &rho_tp(0, ir), &grad_rho_grad_rho_tp(0, ir), &vrho_t[0], &vsigma_t[0], &exc_t[0]);
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* add Exc contribution */
                    exc_tp(itp, ir) += exc_t[itp];

                    /* directly add to Vxc available contributions */
                    vxc_tp(itp, ir) += vrho_t[itp];

                    if (use_lapl) {
                        vxc_tp(itp, ir) -= 2 * vsigma_t[itp] * lapl_rho_tp(itp, ir);
                    }

                    /* save the sigma derivative */
                    vsigma_tp(itp, ir) += vsigma_t[itp];
                }
            }
        }
    }

    if (is_gga) {
        if (use_lapl) {
            Spheric_function<function_domain_t::spectral, double> vsigma_lm(ctx_.lmmax_pot(), rgrid);
            /* forward transform vsigma to Rlm */
            transform(*sht_, vsigma_tp, vsigma_lm);

            /* compute gradient of vsgima in spherical harmonics */
            auto grad_vsigma_lm = gradient(vsigma_lm);

            /* backward transform gradient from Rlm to (theta, phi) */
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_tp(sht_->num_points(), rgrid);
            for (int x = 0; x < 3; x++) {
                transform(*sht_, grad_vsigma_lm[x], grad_vsigma_tp[x]);
            }

            /* compute scalar product of two gradients */
            auto grad_vsigma_grad_rho_tp = grad_vsigma_tp * grad_rho_tp;

            /* add remaining term to Vxc */
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    vxc_tp(itp, ir) -= 2 * grad_vsigma_grad_rho_tp(itp, ir);
                }
            }
        } else {
            Spheric_vector_function<function_domain_t::spectral, double> vsigma_grad_rho_lm(ctx_.lmmax_pot(), rgrid);
            for (int x: {0, 1, 2}) {
                auto vsigma_grad_rho_tp = vsigma_tp * grad_rho_tp[x];
                transform(*sht_, vsigma_grad_rho_tp, vsigma_grad_rho_lm[x]);
            }
            auto div_vsigma_grad_rho_lm = divergence(vsigma_grad_rho_lm);
            auto div_vsigma_grad_rho_tp = transform(*sht_, div_vsigma_grad_rho_lm);
            /* add remaining term to Vxc */
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    vxc_tp(itp, ir) -= 2 * div_vsigma_grad_rho_tp(itp, ir);
                }
            }
        }
    }
}

void Potential::xc_mt_magnetic(Radial_grid<double> const& rgrid,
                               std::vector<XC_functional>& xc_func,
                               Spheric_function<function_domain_t::spectral, double>& rho_up_lm,
                               Spheric_function<function_domain_t::spatial, double>& rho_up_tp,
                               Spheric_function<function_domain_t::spectral, double>& rho_dn_lm,
                               Spheric_function<function_domain_t::spatial, double>& rho_dn_tp,
                               Spheric_function<function_domain_t::spatial, double>& vxc_up_tp,
                               Spheric_function<function_domain_t::spatial, double>& vxc_dn_tp,
                               Spheric_function<function_domain_t::spatial, double>& exc_tp)
{
    bool is_gga = is_gradient_correction();

    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_tp(sht_->num_points(), rgrid);
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_tp(sht_->num_points(), rgrid);

    Spheric_function<function_domain_t::spatial, double> lapl_rho_up_tp(ctx_.mem_pool(memory_t::host), sht_->num_points(), rgrid);
    Spheric_function<function_domain_t::spatial, double> lapl_rho_dn_tp(ctx_.mem_pool(memory_t::host), sht_->num_points(), rgrid);

    Spheric_function<function_domain_t::spatial, double> grad_rho_up_grad_rho_up_tp;
    Spheric_function<function_domain_t::spatial, double> grad_rho_dn_grad_rho_dn_tp;
    Spheric_function<function_domain_t::spatial, double> grad_rho_up_grad_rho_dn_tp;

    //assert(rho_up_lm.radial_grid().hash() == rho_dn_lm.radial_grid().hash());

    vxc_up_tp.zero();
    vxc_dn_tp.zero();
    exc_tp.zero();

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            grad_rho_up_tp[x] = transform(*sht_, grad_rho_up_lm[x]);
            grad_rho_dn_tp[x] = transform(*sht_, grad_rho_dn_lm[x]);
        }

        /* compute density gradient products */
        grad_rho_up_grad_rho_up_tp = grad_rho_up_tp * grad_rho_up_tp;
        grad_rho_up_grad_rho_dn_tp = grad_rho_up_tp * grad_rho_dn_tp;
        grad_rho_dn_grad_rho_dn_tp = grad_rho_dn_tp * grad_rho_dn_tp;

        /* compute Laplacians in Rlm spherical harmonics */
        auto lapl_rho_up_lm = laplacian(rho_up_lm);
        auto lapl_rho_dn_lm = laplacian(rho_dn_lm);

        /* backward transform Laplacians from Rlm to (theta, phi) */
        lapl_rho_up_tp = transform(*sht_, lapl_rho_up_lm);
        lapl_rho_dn_tp = transform(*sht_, lapl_rho_dn_lm);
    }

    Spheric_function<function_domain_t::spatial, double> vsigma_uu_tp;
    Spheric_function<function_domain_t::spatial, double> vsigma_ud_tp;
    Spheric_function<function_domain_t::spatial, double> vsigma_dd_tp;
    if (is_gga) {
        vsigma_uu_tp = Spheric_function<function_domain_t::spatial, double>(ctx_.mem_pool(memory_t::host), sht_->num_points(), rgrid);
        vsigma_uu_tp.zero();

        vsigma_ud_tp = Spheric_function<function_domain_t::spatial, double>(ctx_.mem_pool(memory_t::host), sht_->num_points(), rgrid);
        vsigma_ud_tp.zero();

        vsigma_dd_tp = Spheric_function<function_domain_t::spatial, double>(ctx_.mem_pool(memory_t::host), sht_->num_points(), rgrid);
        vsigma_dd_tp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func) {
        /* if this is an LDA functional */
        if (ixc.is_lda()) {
            std::vector<double> exc_t(sht_->num_points());
            std::vector<double> vxc_up_t(sht_->num_points());
            std::vector<double> vxc_dn_t(sht_->num_points());
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                ixc.get_lda(sht_->num_points(), &rho_up_tp(0, ir), &rho_dn_tp(0, ir), &vxc_up_t[0], &vxc_dn_t[0], &exc_t[0]);
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* add Exc contribution */
                    exc_tp(itp, ir) += exc_t[itp];

                    /* directly add to Vxc */
                    vxc_up_tp(itp, ir) += vxc_up_t[itp];
                    vxc_dn_tp(itp, ir) += vxc_dn_t[itp];
                }
            }
        }
        if (ixc.is_gga()) {
            std::vector<double> exc_t(sht_->num_points());
            std::vector<double> vrho_up_t(sht_->num_points());
            std::vector<double> vrho_dn_t(sht_->num_points());
            std::vector<double> vsigma_uu_t(sht_->num_points());
            std::vector<double> vsigma_ud_t(sht_->num_points());
            std::vector<double> vsigma_dd_t(sht_->num_points());
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                ixc.get_gga(sht_->num_points(),
                            &rho_up_tp(0, ir),
                            &rho_dn_tp(0, ir),
                            &grad_rho_up_grad_rho_up_tp(0, ir),
                            &grad_rho_up_grad_rho_dn_tp(0, ir),
                            &grad_rho_dn_grad_rho_dn_tp(0, ir),
                            &vrho_up_t[0],
                            &vrho_dn_t[0],
                            &vsigma_uu_t[0],
                            &vsigma_ud_t[0],
                            &vsigma_dd_t[0],
                            &exc_t[0]);

                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* add Exc contribution */
                    exc_tp(itp, ir) += exc_t[itp];

                    /* directly add to Vxc available contributions */
                    vxc_up_tp(itp, ir) += (vrho_up_t[itp] - 2 * vsigma_uu_t[itp] * lapl_rho_up_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_dn_tp(itp, ir));
                    vxc_dn_tp(itp, ir) += (vrho_dn_t[itp] - 2 * vsigma_dd_t[itp] * lapl_rho_dn_tp(itp, ir) - vsigma_ud_t[itp] * lapl_rho_up_tp(itp, ir));

                    /* save the sigma derivatives */
                    vsigma_uu_tp(itp, ir) += vsigma_uu_t[itp];
                    vsigma_ud_tp(itp, ir) += vsigma_ud_t[itp];
                    vsigma_dd_tp(itp, ir) += vsigma_dd_t[itp];
                }
            }
        }
    }

    if (is_gga) {
        /* forward transform vsigma to Rlm */
        auto vsigma_uu_lm = transform(*sht_, vsigma_uu_tp);
        auto vsigma_ud_lm = transform(*sht_, vsigma_ud_tp);
        auto vsigma_dd_lm = transform(*sht_, vsigma_dd_tp);

        /* compute gradient of vsgima in spherical harmonics */
        auto grad_vsigma_uu_lm = gradient(vsigma_uu_lm);
        auto grad_vsigma_ud_lm = gradient(vsigma_ud_lm);
        auto grad_vsigma_dd_lm = gradient(vsigma_dd_lm);

        /* backward transform gradient from Rlm to (theta, phi) */
        Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_uu_tp(sht_->num_points(), rgrid);
        Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_ud_tp(sht_->num_points(), rgrid);
        Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_dd_tp(sht_->num_points(), rgrid);
        for (int x = 0; x < 3; x++) {
            grad_vsigma_uu_tp[x] = transform(*sht_, grad_vsigma_uu_lm[x]);
            grad_vsigma_ud_tp[x] = transform(*sht_, grad_vsigma_ud_lm[x]);
            grad_vsigma_dd_tp[x] = transform(*sht_, grad_vsigma_dd_lm[x]);
        }

        /* compute scalar product of two gradients */
        auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
        auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
        auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;
        auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;

        /* add remaining terms to Vxc */
        for (int ir = 0; ir < rgrid.num_points(); ir++) {
            for (int itp = 0; itp < sht_->num_points(); itp++) {
                vxc_up_tp(itp, ir) -= (2 * grad_vsigma_uu_grad_rho_up_tp(itp, ir) + grad_vsigma_ud_grad_rho_dn_tp(itp, ir));
                vxc_dn_tp(itp, ir) -= (2 * grad_vsigma_dd_grad_rho_dn_tp(itp, ir) + grad_vsigma_ud_grad_rho_up_tp(itp, ir));
            }
        }
    }
}

void Potential::xc_mt(Density const& density__)
{
    PROFILE("sirius::Potential::xc_mt");

    #pragma omp parallel for
    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& rgrid = unit_cell_.atom(ia).radial_grid();
        int nmtp = unit_cell_.atom(ia).num_mt_points();

        /* backward transform density from Rlm to (theta, phi) */
        auto rho_tp = transform(*sht_, density__.rho().f_mt(ialoc));

        /* backward transform magnetization from Rlm to (theta, phi) */
        std::vector<Spheric_function<function_domain_t::spatial, double> > vecmagtp(ctx_.num_mag_dims());
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            vecmagtp[j] = transform(*sht_, density__.magnetization(j).f_mt(ialoc));
        }

        /* "up" component of the density */
        Spheric_function<function_domain_t::spectral, double> rho_up_lm;
        Spheric_function<function_domain_t::spatial, double> rho_up_tp(sht_->num_points(), rgrid);

        /* "dn" component of the density */
        Spheric_function<function_domain_t::spectral, double> rho_dn_lm;
        Spheric_function<function_domain_t::spatial, double> rho_dn_tp(sht_->num_points(), rgrid);

        /* check if density has negative values */
        double rhomin = 0.0;
        for (int ir = 0; ir < nmtp; ir++) {
            for (int itp = 0; itp < sht_->num_points(); itp++) {
                rhomin = std::min(rhomin, rho_tp(itp, ir));
            }
        }

        if (rhomin < 0.0 && std::abs(rhomin) > 1e-9) {
            std::stringstream s;
            s << "Charge density for atom " << ia << " has negative values" << std::endl
              << "most negatve value : " << rhomin << std::endl
              << "current Rlm expansion of the charge density may be not sufficient, try to increase lmax_rho";
            WARNING(s);
        }

        if (ctx_.num_spins() == 1) {
            for (int ir = 0; ir < nmtp; ir++) {
                /* fix negative density */
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    if (rho_tp(itp, ir) < 0.0) {
                        rho_tp(itp, ir) = 0.0;
                    }
                }
            }
        } else {
            for (int ir = 0; ir < nmtp; ir++) {
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* compute magnitude of the magnetization vector */
                    double mag = 0.0;
                    for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                        mag += std::pow(vecmagtp[j](itp, ir), 2);
                    }
                    mag = std::sqrt(mag);

                    /* in magnetic case fix both density and magnetization */
                    for (int itp = 0; itp < sht_->num_points(); itp++) {
                        if (rho_tp(itp, ir) < 0.0) {
                            rho_tp(itp, ir) = 0.0;
                            mag = 0.0;
                        }
                        /* fix numerical noise at high values of magnetization */
                        mag = std::min(mag, rho_tp(itp, ir));

                        /* compute "up" and "dn" components */
                        rho_up_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) + mag);
                        rho_dn_tp(itp, ir) = 0.5 * (rho_tp(itp, ir) - mag);
                    }
                }
            }

            /* transform from (theta, phi) to Rlm */
            rho_up_lm = transform(*sht_, rho_up_tp);
            rho_dn_lm = transform(*sht_, rho_dn_tp);
        }

        Spheric_function<function_domain_t::spatial, double> exc_tp(sht_->num_points(), rgrid);
        Spheric_function<function_domain_t::spatial, double> vxc_tp(sht_->num_points(), rgrid);

        if (ctx_.num_spins() == 1) {
            xc_mt_nonmagnetic(rgrid, xc_func_, density__.rho().f_mt(ialoc), rho_tp, vxc_tp, exc_tp);
        } else {
            Spheric_function<function_domain_t::spatial, double> vxc_up_tp(sht_->num_points(), rgrid);
            Spheric_function<function_domain_t::spatial, double> vxc_dn_tp(sht_->num_points(), rgrid);

            xc_mt_magnetic(rgrid, xc_func_, rho_up_lm, rho_up_tp, rho_dn_lm, rho_dn_tp, vxc_up_tp, vxc_dn_tp, exc_tp);

            for (int ir = 0; ir < nmtp; ir++) {
                for (int itp = 0; itp < sht_->num_points(); itp++) {
                    /* align magnetic filed parallel to magnetization */
                    /* use vecmagtp as temporary vector */
                    double mag =  rho_up_tp(itp, ir) - rho_dn_tp(itp, ir);
                    if (mag > 1e-8) {
                        /* |Bxc| = 0.5 * (V_up - V_dn) */
                        double b = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                            vecmagtp[j](itp, ir) = b * vecmagtp[j](itp, ir) / mag;
                        }
                    } else {
                        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                            vecmagtp[j](itp, ir) = 0.0;
                        }
                    }
                    /* Vxc = 0.5 * (V_up + V_dn) */
                    vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                }
            }
            /* z, x, y order */
            std::array<int, 3> comp_map = {2, 0, 1};
            /* convert magnetic field back to Rlm */
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                auto bxcrlm = transform(*sht_, vecmagtp[j]);
                for (int ir = 0; ir < nmtp; ir++) {
                    /* add auxiliary magnetic field antiparallel to starting magnetization */
                    bxcrlm(0, ir) -= aux_bf_(j, ia) * ctx_.unit_cell().atom(ia).vector_field()[comp_map[j]];
                    for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                        effective_magnetic_field(j).f_mt<index_domain_t::local>(lm, ir, ialoc) = bxcrlm(lm, ir);
                    }
                }
            }
        }

        /* forward transform from (theta, phi) to Rlm */
        auto vxcrlm = transform(*sht_, vxc_tp);
        auto excrlm = transform(*sht_, exc_tp);
        for (int ir = 0; ir < nmtp; ir++) {
            for (int lm = 0; lm < ctx_.lmmax_pot(); lm++) {
                xc_potential_->f_mt<index_domain_t::local>(lm, ir, ialoc) = vxcrlm(lm, ir);
                xc_energy_density_->f_mt<index_domain_t::local>(lm, ir, ialoc) = excrlm(lm, ir);
            }
        }
    } // ialoc
}

template <bool add_pseudo_core__>
void Potential::xc_rg_nonmagnetic(Density const& density__)
{
    PROFILE("sirius::Potential::xc_rg_nonmagnetic");

    bool const use_2nd_deriv{false};

    bool const use_all_gvec{false};

    std::unique_ptr<Gvec> gv_ptr;
    std::unique_ptr<Gvec_partition> gvp_ptr;
    if (use_all_gvec) {
        STOP();
        ///* this will create a full list of G-vectors of the size of the FFT box */
        //gv_ptr = std::unique_ptr<Gvec>(new Gvec(ctx_.unit_cell().reciprocal_lattice_vectors(),
        //                                        ctx_.pw_cutoff() * 2000, ctx_.fft(), ctx_.comm(), false));
        //gvp_ptr = std::unique_ptr<Gvec_partition>(new Gvec_partition(*gv_ptr, ctx_.comm_fft(), ctx_.comm_ortho_fft()));
    }

    auto& gvp = (use_all_gvec) ? (*gvp_ptr) : ctx_.gvec_partition();

    bool is_gga = is_gradient_correction();

    int num_points = ctx_.spfft().local_slice_size();

    Smooth_periodic_function<double> rho(ctx_.spfft(), gvp);
    Smooth_periodic_function<double> vsigma(ctx_.spfft(), gvp);

    /* we can use this comm for parallelization */
    //auto& comm = ctx_.gvec().comm_ortho_fft();
    /* split real-space points between available ranks */
    //splindex<block> spl_np(num_points, comm.size(), comm.rank());

    /* check for negative values */
    double rhomin{0};
    for (int ir = 0; ir < num_points; ir++) {

        //int ir = spl_np[irloc];
        double d = density__.rho().f_rg(ir);
        if (add_pseudo_core__) {
            d += density__.rho_pseudo_core().f_rg(ir);
        }
        d *= scale_rho_xc_;

        rhomin = std::min(rhomin, d);
        rho.f_rg(ir) = std::max(d, 0.0);
    }
    Communicator(ctx_.spfft().communicator()).allreduce<double, mpi_op_t::min>(&rhomin, 1);
    /* even a small negative density is a sign of something bing wrong; don't remove this check */
    if (rhomin < 0.0 && ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        WARNING(s);
    }

    if (ctx_.control().print_hash_) {
        auto h = rho.hash_f_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("rho", h);
        }
    }

    if (ctx_.control().print_checksum_) {
        auto cs = density__.rho().checksum_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("rho_rg", cs);
        }
    }

    Smooth_periodic_vector_function<double> grad_rho;
    Smooth_periodic_function<double> lapl_rho;
    Smooth_periodic_function<double> grad_rho_grad_rho;

    Smooth_periodic_function<double> div_vsigma_grad_rho;

    if (is_gga) {
        /* use fft_transfrom of the base class (Smooth_periodic_function) */
        rho.fft_transform(-1);

        /* generate pw coeffs of the gradient */
        grad_rho = gradient(rho);
        /* generate pw coeffs of the laplacian */
        if (use_2nd_deriv) {
            lapl_rho = laplacian(rho);
            /* Laplacian in real space */
            lapl_rho.fft_transform(1);
        }

        /* gradient in real space */
        for (int x: {0, 1, 2}) {
            grad_rho[x].fft_transform(1);
        }

        /* product of gradients */
        grad_rho_grad_rho = dot(grad_rho, grad_rho);

        if (ctx_.control().print_hash_) {
            //auto h1 = lapl_rho.hash_f_rg();
            auto h2 = grad_rho_grad_rho.hash_f_rg();
            if (ctx_.comm().rank() == 0) {
                //utils::print_hash("lapl_rho", h1);
                utils::print_hash("grad_rho_grad_rho", h2);
            }
        }
    }

    mdarray<double, 1> exc_tmp(num_points);
    exc_tmp.zero();

    mdarray<double, 1> vxc_tmp(num_points);
    vxc_tmp.zero();

    mdarray<double, 1> vsigma_tmp;
    if (is_gga) {
        vsigma_tmp = mdarray<double, 1>(num_points);
        vsigma_tmp.zero();
    }

    /* loop over XC functionals */
    for (auto& ixc: xc_func_) {

        /*
          I need to split vdw from the parallel section since it involves a fft
          for each grid point
        */
        if (ixc.is_vdw()) {
#if defined(__USE_VDWXC)
            /* Van der Walls correction */
            std::vector<double> exc_t(num_points, 0.0);
            std::vector<double> vrho_t(num_points, 0.0);
            std::vector<double> vsigma_t(num_points, 0.0);

            ixc.get_vdw(&rho.f_rg(0),
                        &grad_rho_grad_rho.f_rg(0),
                        &vrho_t[0],
                        &vsigma_t[0],
                        &exc_t[0]);
            #pragma omp parallel for
            for (int i = 0; i < num_points; i++) {
                /* add Exc contribution */
                exc_tmp(i) += exc_t[i];

                /* directly add to Vxc available contributions */
                //if (use_2nd_deriv) {
                //    vxc_tmp(spl_np_t[i]) += (vrho_t[i] - 2 * vsigma_t[i] * lapl_rho.f_rg(spl_np_t[i]));
                //} else {
                //    vxc_tmp(spl_np_t[i]) += vrho_t[i];
                //}
                vxc_tmp(i) += vrho_t[i];

                /* save the sigma derivative */
                vsigma_tmp(i) += vsigma_t[i];
            }
#else
            TERMINATE("You should not be there since SIRIUS is not compiled with libVDWXC support\n");
#endif
        } else {
            #pragma omp parallel
            {
                /* split local size between threads */
                splindex<splindex_t::block> spl_np_t(num_points, omp_get_num_threads(), omp_get_thread_num());

                std::vector<double> exc_t(spl_np_t.local_size());

                /* if this is an LDA functional */
                if (ixc.is_lda()) {
                    std::vector<double> vxc_t(spl_np_t.local_size());

                    ixc.get_lda(spl_np_t.local_size(),
                                &rho.f_rg(spl_np_t.global_offset()),
                                &vxc_t[0],
                                &exc_t[0]);

                    for (int i = 0; i < spl_np_t.local_size(); i++) {
                        /* add Exc contribution */
                        exc_tmp(spl_np_t[i]) += exc_t[i];

                        /* directly add to Vxc */
                        vxc_tmp(spl_np_t[i]) += vxc_t[i];
                    }
                }

                if (ixc.is_gga()) {
                    std::vector<double> vrho_t(spl_np_t.local_size());
                    std::vector<double> vsigma_t(spl_np_t.local_size());

                    ixc.get_gga(spl_np_t.local_size(),
                                &rho.f_rg(spl_np_t.global_offset()),
                                &grad_rho_grad_rho.f_rg(spl_np_t.global_offset()),
                                &vrho_t[0],
                                &vsigma_t[0],
                                &exc_t[0]);

                    /* this is the same expression between gga and vdw corrections.
                     * The functionals are different that's all */
                    for (int i = 0; i < spl_np_t.local_size(); i++) {
                        /* add Exc contribution */
                        exc_tmp(spl_np_t[i]) += exc_t[i];

                        /* directly add to Vxc available contributions */
                        //if (use_2nd_deriv) {
                        //    vxc_tmp(spl_np_t[i]) += (vrho_t[i] - 2 * vsigma_t[i] * lapl_rho.f_rg(spl_np_t[i]));
                        //} else {
                        //    vxc_tmp(spl_np_t[i]) += vrho_t[i];
                        //}
                        vxc_tmp(spl_np_t[i]) += vrho_t[i];

                        /* save the sigma derivative */
                        vsigma_tmp(spl_np_t[i]) += vsigma_t[i];
                    }
                }
            }
        }
    }

    if (is_gga) { /* generic for gga and vdw */
        /* gather vsigma */
        //comm.allgather(&vsigma_tmp[0], &vsigma_[0]->f_rg(0), spl_np.global_offset(), spl_np.local_size());
        #pragma omp parallel for
        for (int ir = 0; ir < num_points; ir++) {
            vsigma_[0]->f_rg(ir) = vsigma_tmp[ir];
            vsigma.f_rg(ir) = vsigma_tmp[ir];
        }

        if (use_2nd_deriv) {
            /* forward transform vsigma to plane-wave domain */
            vsigma.fft_transform(-1);

            /* gradient of vsigma in plane-wave domain */
            auto grad_vsigma = gradient(vsigma);

            /* backward transform gradient from pw to real space */
            for (int x: {0, 1, 2}) {
                grad_vsigma[x].fft_transform(1);
            }

            /* compute scalar product of two gradients */
            auto grad_vsigma_grad_rho = dot(grad_vsigma, grad_rho);

            /* add remaining term to Vxc */
            #pragma omp parallel for
            for (int ir = 0; ir < num_points; ir++) {
                vxc_tmp(ir) -= 2 * (vsigma.f_rg(ir) * lapl_rho.f_rg(ir) + grad_vsigma_grad_rho.f_rg(ir));
            }
        } else {
            Smooth_periodic_vector_function<double> vsigma_grad_rho(ctx_.spfft(), gvp);

            for (int x: {0, 1, 2}) {
                for (int ir = 0; ir < num_points; ir++) {
                    vsigma_grad_rho[x].f_rg(ir) = grad_rho[x].f_rg(ir) * vsigma.f_rg(ir);
                }
                /* transform to plane wave domain */
                vsigma_grad_rho[x].fft_transform(-1);
            }
            div_vsigma_grad_rho = divergence(vsigma_grad_rho);
            /* transform to real space domain */
            div_vsigma_grad_rho.fft_transform(1);
            for (int ir = 0; ir < num_points; ir++) {
                vxc_tmp(ir) -= 2 * div_vsigma_grad_rho.f_rg(ir);
            }
        }
    }
    //comm.allgather(&vxc_tmp[0], &xc_potential_->f_rg(0), spl_np.global_offset(), spl_np.local_size());
    //comm.allgather(&exc_tmp[0], &xc_energy_density_->f_rg(0), spl_np.global_offset(), spl_np.local_size());

    #pragma omp parallel for
    for (int ir = 0; ir < num_points; ir++) {
        xc_energy_density_->f_rg(ir) = exc_tmp(ir);
        xc_potential_->f_rg(ir) = vxc_tmp(ir);
    }

    /* forward transform vsigma to plane-wave domain */
    vsigma_[0]->fft_transform(-1);

    if (ctx_.control().print_checksum_) {
        auto cs = xc_potential_->checksum_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_checksum("exc", cs);
        }
    }
}

template <bool add_pseudo_core__>
void Potential::xc_rg_magnetic(Density const& density__)
{
    PROFILE("sirius::Potential::xc_rg_magnetic");

    bool is_gga = is_gradient_correction();

    int num_points = ctx_.spfft().local_slice_size();

    Smooth_periodic_function<double> rho_up(ctx_.spfft(), ctx_.gvec_partition());
    Smooth_periodic_function<double> rho_dn(ctx_.spfft(), ctx_.gvec_partition());

    PROFILE_START("sirius::Potential::xc_rg_magnetic|up_dn");
    /* compute "up" and "dn" components and also check for negative values of density */
    double rhomin{0};
    for (int ir = 0; ir < num_points; ir++) {
        double mag{0};
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            mag += std::pow(density__.magnetization(j).f_rg(ir), 2);
        }
        mag = std::sqrt(mag);

        double rho = density__.rho().f_rg(ir);
        if (add_pseudo_core__) {
            rho += density__.rho_pseudo_core().f_rg(ir);
        }
        rho *= scale_rho_xc_;
        mag *= scale_rho_xc_;

        /* remove numerical noise at high values of magnetization */
        mag = std::min(mag, rho);

        rhomin = std::min(rhomin, rho);
        if (rho < 0.0) {
            rho = 0.0;
            mag = 0.0;
        }

        rho_up.f_rg(ir) = 0.5 * (rho + mag);
        rho_dn.f_rg(ir) = 0.5 * (rho - mag);
    }
    PROFILE_STOP("sirius::Potential::xc_rg_magnetic|up_dn");

    Communicator(ctx_.spfft().communicator()).allreduce<double, mpi_op_t::min>(&rhomin, 1);
    if (rhomin < 0.0 && ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        WARNING(s);
    }

    if (ctx_.control().print_hash_) {
        auto h1 = rho_up.hash_f_rg();
        auto h2 = rho_dn.hash_f_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("rho_up", h1);
            utils::print_hash("rho_dn", h2);
        }
    }

    Smooth_periodic_vector_function<double> grad_rho_up;
    Smooth_periodic_vector_function<double> grad_rho_dn;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_up;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_dn;
    Smooth_periodic_function<double> grad_rho_dn_grad_rho_dn;

    if (is_gga) {
        PROFILE("sirius::Potential::xc_rg_magnetic|grad1");
        /* get plane-wave coefficients of densities */
        rho_up.fft_transform(-1);
        rho_dn.fft_transform(-1);

        /* generate pw coeffs of the gradient and laplacian */
        grad_rho_up = gradient(rho_up);
        grad_rho_dn = gradient(rho_dn);

        /* gradient in real space */
        for (int x: {0, 1, 2}) {
            grad_rho_up[x].fft_transform(1);
            grad_rho_dn[x].fft_transform(1);
        }

        /* product of gradients */
        grad_rho_up_grad_rho_up = dot(grad_rho_up, grad_rho_up);
        grad_rho_up_grad_rho_dn = dot(grad_rho_up, grad_rho_dn);
        grad_rho_dn_grad_rho_dn = dot(grad_rho_dn, grad_rho_dn);

        if (ctx_.control().print_hash_) {
            auto h3 = grad_rho_up_grad_rho_up.hash_f_rg();
            auto h4 = grad_rho_up_grad_rho_dn.hash_f_rg();
            auto h5 = grad_rho_dn_grad_rho_dn.hash_f_rg();

            if (ctx_.comm().rank() == 0) {
                utils::print_hash("grad_rho_up_grad_rho_up", h3);
                utils::print_hash("grad_rho_up_grad_rho_dn", h4);
                utils::print_hash("grad_rho_dn_grad_rho_dn", h5);
            }
        }
    }

    mdarray<double, 1> exc_tmp(num_points, memory_t::host, "exc_tmp");
    exc_tmp.zero();

    mdarray<double, 1> vxc_up_tmp(num_points, memory_t::host, "vxc_up_tmp");
    vxc_up_tmp.zero();

    mdarray<double, 1> vxc_dn_tmp(num_points, memory_t::host, "vxc_dn_dmp");
    vxc_dn_tmp.zero();

    mdarray<double, 1> vsigma_uu_tmp;
    mdarray<double, 1> vsigma_ud_tmp;
    mdarray<double, 1> vsigma_dd_tmp;

    if (is_gga) {
        vsigma_uu_tmp = mdarray<double, 1>(num_points, memory_t::host, "vsigma_uu_tmp");
        vsigma_uu_tmp.zero();

        vsigma_ud_tmp = mdarray<double, 1>(num_points, memory_t::host, "vsigma_ud_tmp");
        vsigma_ud_tmp.zero();

        vsigma_dd_tmp = mdarray<double, 1>(num_points, memory_t::host, "vsigma_dd_tmp");
        vsigma_dd_tmp.zero();
    }

    PROFILE_START("sirius::Potential::xc_rg_magnetic|libxc");
    /* loop over XC functionals */
    for (auto& ixc: xc_func_) {
        /* treat vdw correction outside the parallel region because it uses fft
         * internaly */

        if (ixc.is_vdw()) {
#if defined(__USE_VDWXC)
            std::vector<double> vrho_up_t(num_points, 0.0);
            std::vector<double> vrho_dn_t(num_points, 0.0);
            std::vector<double> vsigma_uu_t(num_points, 0.0);
//            std::vector<double> vsigma_ud_t(num_points, 0.0);
            std::vector<double> vsigma_dd_t(num_points, 0.0);
            std::vector<double> exc_t(num_points, 0.0);

            ixc.get_vdw(&rho_up.f_rg(0),
                        &rho_dn.f_rg(0),
                        &grad_rho_up_grad_rho_up.f_rg(0),
                        &grad_rho_dn_grad_rho_dn.f_rg(0),
                        &vrho_up_t[0],
                        &vrho_dn_t[0],
                        &vsigma_uu_t[0],
                        &vsigma_dd_t[0],
                        &exc_t[0]);

            #pragma omp parallel for
            for (int i = 0; i < num_points; i++) {
                /* add Exc contribution */
                exc_tmp(i) += exc_t[i];

                /* directly add to Vxc available contributions */
                vxc_up_tmp(i) += vrho_up_t[i];
                vxc_dn_tmp(i) += vrho_dn_t[i];

                /* save the sigma derivative */
                vsigma_uu_tmp(i) += vsigma_uu_t[i];
                //              vsigma_ud_tmp(i) += vsigma_ud_t[i];
                vsigma_dd_tmp(i) += vsigma_dd_t[i];
            }
#else
            TERMINATE("You should not be there since sirius is not compiled with libVDWXC\n");
#endif
        } else {
#pragma omp parallel
            {
                /* split local size between threads */
                splindex<splindex_t::block> spl_t(num_points, omp_get_num_threads(), omp_get_thread_num());

                std::vector<double> exc_t(spl_t.local_size());

                /* if this is an LDA functional */
                if (ixc.is_lda()) {
                    std::vector<double> vxc_up_t(spl_t.local_size());
                    std::vector<double> vxc_dn_t(spl_t.local_size());


                    ixc.get_lda(spl_t.local_size(),
                                &rho_up.f_rg(spl_t.global_offset()),
                                &rho_dn.f_rg(spl_t.global_offset()),
                                &vxc_up_t[0],
                                &vxc_dn_t[0],
                                &exc_t[0]);

                    for (int i = 0; i < spl_t.local_size(); i++) {
                        /* add Exc contribution */
                        exc_tmp(spl_t[i]) += exc_t[i];

                        /* directly add to Vxc */
                        vxc_up_tmp(spl_t[i]) += vxc_up_t[i];
                        vxc_dn_tmp(spl_t[i]) += vxc_dn_t[i];
                    }
                }

                if (ixc.is_gga()) {
                    std::vector<double> vrho_up_t(spl_t.local_size());
                    std::vector<double> vrho_dn_t(spl_t.local_size());
                    std::vector<double> vsigma_uu_t(spl_t.local_size());
                    std::vector<double> vsigma_ud_t(spl_t.local_size());
                    std::vector<double> vsigma_dd_t(spl_t.local_size());

                    ixc.get_gga(spl_t.local_size(),
                                &rho_up.f_rg(spl_t.global_offset()),
                                &rho_dn.f_rg(spl_t.global_offset()),
                                &grad_rho_up_grad_rho_up.f_rg(spl_t.global_offset()),
                                &grad_rho_up_grad_rho_dn.f_rg(spl_t.global_offset()),
                                &grad_rho_dn_grad_rho_dn.f_rg(spl_t.global_offset()),
                                &vrho_up_t[0],
                                &vrho_dn_t[0],
                                &vsigma_uu_t[0],
                                &vsigma_ud_t[0],
                                &vsigma_dd_t[0],
                                &exc_t[0]);

                    #pragma omp parallel for
                    for (int i = 0; i < spl_t.local_size(); i++) {
                        /* add Exc contribution */
                        exc_tmp(spl_t[i]) += exc_t[i];
                        /* directly add to Vxc available contributions */
                        vxc_up_tmp(spl_t[i]) += vrho_up_t[i];
                        vxc_dn_tmp(spl_t[i]) += vrho_dn_t[i];

                        /* save the sigma derivative */
                        vsigma_uu_tmp(spl_t[i]) += vsigma_uu_t[i];
                        vsigma_ud_tmp(spl_t[i]) += vsigma_ud_t[i];
                        vsigma_dd_tmp(spl_t[i]) += vsigma_dd_t[i];
                    }
                }
            }
        }
    }
    PROFILE_STOP("sirius::Potential::xc_rg_magnetic|libxc");

    if (is_gga) {
        PROFILE("sirius::Potential::xc_rg_magnetic|grad2");
        /* gather vsigma */
        // vsigma_uu: dϵ/dσ↑↑
        Smooth_periodic_function<double> vsigma_uu(ctx_.spfft(), ctx_.gvec_partition());
        // vsigma_ud: dϵ/dσ↑↓
        Smooth_periodic_function<double> vsigma_ud(ctx_.spfft(), ctx_.gvec_partition());
        // vsigma_dd: dϵ/dσ↓↓
        Smooth_periodic_function<double> vsigma_dd(ctx_.spfft(), ctx_.gvec_partition());
        for (int ir = 0; ir < num_points; ir++) {
            vsigma_uu.f_rg(ir) = vsigma_uu_tmp[ir];
            vsigma_ud.f_rg(ir) = vsigma_ud_tmp[ir];
            vsigma_dd.f_rg(ir) = vsigma_dd_tmp[ir];
        }

        Smooth_periodic_vector_function<double> up_gradrho_vsigma(ctx_.spfft(), ctx_.gvec_partition());
        Smooth_periodic_vector_function<double> dn_gradrho_vsigma(ctx_.spfft(), ctx_.gvec_partition());
        for (int x: {0, 1, 2}) {
            for(int ir = 0; ir < num_points; ++ir) {
              up_gradrho_vsigma[x].f_rg(ir) = 2 * grad_rho_up[x].f_rg(ir) * vsigma_uu.f_rg(ir) + grad_rho_dn[x].f_rg(ir) * vsigma_ud.f_rg(ir);
              dn_gradrho_vsigma[x].f_rg(ir) = 2 * grad_rho_dn[x].f_rg(ir) * vsigma_dd.f_rg(ir) + grad_rho_up[x].f_rg(ir) * vsigma_ud.f_rg(ir);
            }
            /* transform to plane wave domain */
            up_gradrho_vsigma[x].fft_transform(-1);
            dn_gradrho_vsigma[x].fft_transform(-1);
        }

        auto div_up_gradrho_vsigma = divergence(up_gradrho_vsigma);
        div_up_gradrho_vsigma.fft_transform(1);
        auto div_dn_gradrho_vsigma = divergence(dn_gradrho_vsigma);
        div_dn_gradrho_vsigma.fft_transform(1);

        /* add remaining term to Vxc */
        #pragma omp parallel for
        for (int ir = 0; ir < num_points; ir++) {
            vxc_up_tmp(ir) -= div_up_gradrho_vsigma.f_rg(ir);
            vxc_dn_tmp(ir) -= div_dn_gradrho_vsigma.f_rg(ir);
        }
    }

    #pragma omp parallel for
    for (int irloc = 0; irloc < num_points; irloc++) {
        xc_energy_density_->f_rg(irloc) = exc_tmp(irloc);
        xc_potential_->f_rg(irloc) = 0.5 * (vxc_up_tmp(irloc) + vxc_dn_tmp(irloc));
        double m = rho_up.f_rg(irloc) - rho_dn.f_rg(irloc);

        if (m > 1e-8) {
            double b = 0.5 * (vxc_up_tmp(irloc) - vxc_dn_tmp(irloc));
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
               effective_magnetic_field(j).f_rg(irloc) = b * density__.magnetization(j).f_rg(irloc) / m;
            }
        } else {
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                effective_magnetic_field(j).f_rg(irloc) = 0.0;
            }
        }
    }
}

template <bool add_pseudo_core__>
void Potential::xc(Density const& density__)
{
    PROFILE("sirius::Potential::xc");

    if (ctx_.xc_functionals().size() == 0) {
        xc_potential_->zero();
        xc_energy_density_->zero();
        for (int i = 0; i < ctx_.num_mag_dims(); i++) {
            effective_magnetic_field(i).zero();
        }
        return;
    }

    if (ctx_.full_potential()) {
        xc_mt(density__);
    }

    if (ctx_.num_spins() == 1) {
        xc_rg_nonmagnetic<add_pseudo_core__>(density__);
    } else {
        xc_rg_magnetic<add_pseudo_core__>(density__);
    }

    if (ctx_.control().print_hash_) {
        auto h = xc_energy_density_->hash_f_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("Exc", h);
        }
    }
}

// explicit instantiation
template void Potential::xc_rg_nonmagnetic<true>(Density const&);
template void Potential::xc_rg_nonmagnetic<false>(Density const&);
template void Potential::xc_rg_magnetic<true>(Density const&);
template void Potential::xc_rg_magnetic<false>(Density const&);
template void Potential::xc<true>(Density const&);
template void Potential::xc<false>(Density const&);

} // namespace sirius
