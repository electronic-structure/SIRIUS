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
#include "SDDK/omp.hpp"
#include "xc_functional.hpp"

namespace sirius {

void xc_mt_nonmagnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional*> xc_func__,
                       Flm const& rho_lm__, Ftp& rho_tp__, Flm& vxc_lm__, Flm& exc_lm__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc->is_gga() || ixc->is_vdw()) {
            is_gga = true;
        }
    }

    Ftp exc_tp(sht__.num_points(), rgrid__);
    Ftp vxc_tp(sht__.num_points(), rgrid__);

    assert(rho_tp__.size() == vxc_tp.size());
    assert(rho_tp__.size() == exc_tp.size());

    Ftp grad_rho_grad_rho_tp;
    Ftp vsigma_tp;
    Ftp lapl_rho_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_tp;

    /* use Laplacian (true) or divergence of gradient (false) */
    bool use_lapl{false};

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_lm = gradient(rho_lm__);
        grad_rho_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        /* backward transform gradient from Rlm to (theta, phi) */
        for (int x = 0; x < 3; x++) {
            transform(sht__, grad_rho_lm[x], grad_rho_tp[x]);
        }
        /* compute density gradient product */
        grad_rho_grad_rho_tp = grad_rho_tp * grad_rho_tp;
        assert(rho_tp__.size() == grad_rho_grad_rho_tp.size());

        vsigma_tp = Ftp(sht__.num_points(), rgrid__);
        assert(rho_tp__.size() == vsigma_tp.size());
        if (use_lapl) {
            /* backward transform Laplacian from Rlm to (theta, phi) */
            lapl_rho_tp = transform(sht__, laplacian(rho_lm__));
            assert(lapl_rho_tp.size() == rho_tp__.size());
        }
    }

    for (auto& ixc: xc_func__) {
        /* if this is an LDA functional */
        if (ixc->is_lda()) {
            ixc->get_lda(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                vxc_tp.at(memory_t::host), exc_tp.at(memory_t::host));
        }
        /* if this is a GGA functional */
        if (ixc->is_gga()) {

            /* compute vrho and vsigma */
            ixc->get_gga(sht__.num_points() * rgrid__.num_points(), rho_tp__.at(memory_t::host),
                grad_rho_grad_rho_tp.at(memory_t::host), vxc_tp.at(memory_t::host), vsigma_tp.at(memory_t::host),
                exc_tp.at(memory_t::host));

            if (use_lapl) {
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
                for (int x: {0, 1, 2}) {
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
    } //ixc
}

void xc_mt_magnetic(Radial_grid<double> const& rgrid__, SHT const& sht__, int num_mag_dims__,
                    std::vector<XC_functional*> xc_func__, std::vector<Ftp> const& rho_tp__,
                    std::vector<Flm*> vxc__, Flm& exc__)
{
    bool is_gga{false};
    for (auto& ixc : xc_func__) {
        if (ixc->is_gga() || ixc->is_vdw()) {
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
            vector3d<double> m;
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
    Ftp lapl_rho_up_tp;
    Ftp lapl_rho_dn_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_up_tp;
    Spheric_vector_function<function_domain_t::spatial, double> grad_rho_dn_tp;

    if (is_gga) {
        /* compute gradient in Rlm spherical harmonics */
        auto grad_rho_up_lm = gradient(rho_up_lm);
        auto grad_rho_dn_lm = gradient(rho_dn_lm);
        grad_rho_up_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
        grad_rho_dn_tp = Spheric_vector_function<function_domain_t::spatial, double>(sht__.num_points(), rgrid__);
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

        /* backward transform Laplacians from Rlm to (theta, phi) */
        lapl_rho_up_tp = transform(sht__, laplacian(rho_up_lm));
        lapl_rho_dn_tp = transform(sht__, laplacian(rho_dn_lm));
    }

    for (auto& ixc: xc_func__) {
        if (ixc->is_lda()) {
            ixc->get_lda(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                rho_dn_tp.at(memory_t::host), vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host),
                exc_tp.at(memory_t::host));
        }
        if (ixc->is_gga()) {
            /* get the vrho and vsigma */
            ixc->get_gga(sht__.num_points() * rgrid__.num_points(), rho_up_tp.at(memory_t::host),
                    rho_dn_tp.at(memory_t::host), grad_rho_up_grad_rho_up_tp.at(memory_t::host),
                    grad_rho_up_grad_rho_dn_tp.at(memory_t::host), grad_rho_dn_grad_rho_dn_tp.at(memory_t::host),
                    vxc_up_tp.at(memory_t::host), vxc_dn_tp.at(memory_t::host), vsigma_uu_tp.at(memory_t::host),
                    vsigma_ud_tp.at(memory_t::host), vsigma_dd_tp.at(memory_t::host), exc_tp.at(memory_t::host));

            /* directly add to Vxc available contributions */
            vxc_up_tp -= (2.0 * vsigma_uu_tp * lapl_rho_up_tp + vsigma_ud_tp * lapl_rho_dn_tp);
            vxc_dn_tp -= (2.0 * vsigma_dd_tp * lapl_rho_dn_tp + vsigma_ud_tp * lapl_rho_up_tp);

            /* forward transform vsigma to Rlm */
            auto vsigma_uu_lm = transform(sht__, vsigma_uu_tp);
            auto vsigma_ud_lm = transform(sht__, vsigma_ud_tp);
            auto vsigma_dd_lm = transform(sht__, vsigma_dd_tp);

            /* compute gradient of vsgima in spherical harmonics */
            auto grad_vsigma_uu_lm = gradient(vsigma_uu_lm);
            auto grad_vsigma_ud_lm = gradient(vsigma_ud_lm);
            auto grad_vsigma_dd_lm = gradient(vsigma_dd_lm);

            /* backward transform gradient from Rlm to (theta, phi) */
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_uu_tp(sht__.num_points(), rgrid__);
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_ud_tp(sht__.num_points(), rgrid__);
            Spheric_vector_function<function_domain_t::spatial, double> grad_vsigma_dd_tp(sht__.num_points(), rgrid__);
            for (int x = 0; x < 3; x++) {
                grad_vsigma_uu_tp[x] = transform(sht__, grad_vsigma_uu_lm[x]);
                grad_vsigma_ud_tp[x] = transform(sht__, grad_vsigma_ud_lm[x]);
                grad_vsigma_dd_tp[x] = transform(sht__, grad_vsigma_dd_lm[x]);
            }

            /* compute scalar product of two gradients */
            auto grad_vsigma_uu_grad_rho_up_tp = grad_vsigma_uu_tp * grad_rho_up_tp;
            auto grad_vsigma_dd_grad_rho_dn_tp = grad_vsigma_dd_tp * grad_rho_dn_tp;
            auto grad_vsigma_ud_grad_rho_up_tp = grad_vsigma_ud_tp * grad_rho_up_tp;
            auto grad_vsigma_ud_grad_rho_dn_tp = grad_vsigma_ud_tp * grad_rho_dn_tp;

            /* add remaining terms to Vxc */
            vxc_up_tp -= (2.0 * grad_vsigma_uu_grad_rho_up_tp + grad_vsigma_ud_grad_rho_dn_tp);
            vxc_dn_tp -= (2.0 * grad_vsigma_dd_grad_rho_dn_tp + grad_vsigma_ud_grad_rho_up_tp);
        }
        /* genertate magnetic filed and effective potential inside MT sphere */
        for (int ir = 0; ir < rgrid__.num_points(); ir++) {
            for (int itp = 0; itp < sht__.num_points(); itp++) {
                /* Vxc = 0.5 * (V_up + V_dn) */
                vxc_tp(itp, ir) = 0.5 * (vxc_up_tp(itp, ir) + vxc_dn_tp(itp, ir));
                /* Bxc = 0.5 * (V_up - V_dn) */
                double bxc = 0.5 * (vxc_up_tp(itp, ir) - vxc_dn_tp(itp, ir));
                /* get the sign between mag and B */
                auto s = utils::sign((rho_up_tp(itp, ir) - rho_dn_tp(itp, ir)) * bxc);

                vector3d<double> m;
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

void xc_mt(Radial_grid<double> const& rgrid__, SHT const& sht__, std::vector<XC_functional*> xc_func__,
        int num_mag_dims__, std::vector<Flm const*> rho__, std::vector<Flm*> vxc__, Flm* exc__)
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
                rho_tp[0](itp, ir) = 0.0;
            }
        }
    }

    if (rhomin < 0.0) {
        std::stringstream s;
        s << "[xc_mt] negative charge density: " << rhomin << std::endl
          << "  current Rlm expansion of the charge density may be not sufficient, try to increase lmax";
        WARNING(s);
    }

    if (num_mag_dims__ == 0) {
        xc_mt_nonmagnetic(rgrid__, sht__, xc_func__, *rho__[0], rho_tp[0], *vxc__[0], *exc__);
    } else {
        xc_mt_magnetic(rgrid__, sht__, num_mag_dims__, xc_func__, rho_tp, vxc__, *exc__);
    }
}

void Potential::xc_mt(Density const& density__)
{
    PROFILE("sirius::Potential::xc_mt");

    #pragma omp parallel for
    for (int ialoc = 0; ialoc < unit_cell_.spl_num_atoms().local_size(); ialoc++) {
        int ia = unit_cell_.spl_num_atoms(ialoc);
        auto& rgrid = unit_cell_.atom(ia).radial_grid();
        std::vector<Flm const*> rho(ctx_.num_mag_dims() + 1);
        std::vector<Flm*> vxc(ctx_.num_mag_dims() + 1);
        rho[0] = &density__.rho().f_mt(ialoc);
        vxc[0] = &xc_potential_->f_mt(ialoc);
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            rho[j + 1] = &density__.magnetization(j).f_mt(ialoc);
            vxc[j + 1] = &effective_magnetic_field(j).f_mt(ialoc);
        }
        sirius::xc_mt(rgrid, *sht_, xc_func_, ctx_.num_mag_dims(), rho, vxc, &xc_energy_density_->f_mt(ialoc));

        /* z, x, y order */
        std::array<int, 3> comp_map = {2, 0, 1};
        /* add auxiliary magnetic field antiparallel to starting magnetization */
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            for (int ir = 0; ir < rgrid.num_points(); ir++) {
                effective_magnetic_field(j).f_mt<index_domain_t::local>(0, ir, ialoc) -=
                    aux_bf_(j, ia) * ctx_.unit_cell().atom(ia).vector_field()[comp_map[j]];
            }
        }
    } // ialoc
}

template <bool add_pseudo_core__>
void Potential::xc_rg_nonmagnetic(Density const& density__)
{
    PROFILE("sirius::Potential::xc_rg_nonmagnetic");

    bool const use_2nd_deriv{false};

    auto& gvp = ctx_.gvec_partition();

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
        d *= (1 + add_delta_rho_xc_);

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

    if (ctx_.cfg().control().print_hash()) {
        auto h = rho.hash_f_rg();
        if (ctx_.comm().rank() == 0) {
            utils::print_hash("rho", h);
        }
    }

    if (ctx_.cfg().control().print_checksum()) {
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

        if (ctx_.cfg().control().print_hash()) {
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

    if (num_points) {
        /* loop over XC functionals */
        for (auto& ixc: xc_func_) {
            /*
              I need to split vdw from the parallel section since it involves a fft
              for each grid point
            */
            if (ixc->is_vdw()) {
#if defined(SIRIUS_USE_VDWXC)
                /* Van der Walls correction */
                std::vector<double> exc_t(num_points, 0.0);
                std::vector<double> vrho_t(num_points, 0.0);
                std::vector<double> vsigma_t(num_points, 0.0);

                ixc->get_vdw(&rho.f_rg(0),
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
                    if (ixc->is_lda()) {
                        std::vector<double> vxc_t(spl_np_t.local_size());

                        ixc->get_lda(spl_np_t.local_size(),
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

                    if (ixc->is_gga()) {
                        std::vector<double> vrho_t(spl_np_t.local_size());
                        std::vector<double> vsigma_t(spl_np_t.local_size());

                        ixc->get_gga(spl_np_t.local_size(),
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
    }

    if (is_gga) { /* generic for gga and vdw */
        /* gather vsigma */
        //comm.allgather(&vsigma_tmp[0], &vsigma_[0]->f_rg(0), spl_np.global_offset(), spl_np.local_size());
        #pragma omp parallel for
        for (int ir = 0; ir < num_points; ir++) {
            /* save for future reuse in XC stress calculation */
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
    //vsigma_[0]->fft_transform(-1);

    if (ctx_.cfg().control().print_checksum()) {
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
    #pragma omp parallel for reduction(min:rhomin)
    for (int ir = 0; ir < num_points; ir++) {
        vector3d<double> m;
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            m[j] = density__.magnetization(j).f_rg(ir) * (1 + add_delta_mag_xc_);
        }

        double rho = density__.rho().f_rg(ir);
        if (add_pseudo_core__) {
            rho += density__.rho_pseudo_core().f_rg(ir);
        }
        rho *= (1 + add_delta_rho_xc_);
        rhomin = std::min(rhomin, rho);
        auto rud = get_rho_up_dn(ctx_.num_mag_dims(), rho, m);

        rho_up.f_rg(ir) = rud.first;
        rho_dn.f_rg(ir) = rud.second;
    }
    PROFILE_STOP("sirius::Potential::xc_rg_magnetic|up_dn");

    Communicator(ctx_.spfft().communicator()).allreduce<double, mpi_op_t::min>(&rhomin, 1);
    if (rhomin < 0.0 && ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl
          << "most negatve value : " << rhomin;
        WARNING(s);
    }

    if (ctx_.cfg().control().print_hash()) {
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

        if (ctx_.cfg().control().print_hash()) {
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

    /* vsigma_uu: dϵ/dσ↑↑ */
    Smooth_periodic_function<double> vsigma_uu;
    /* vsigma_ud: dϵ/dσ↑↓ */
    Smooth_periodic_function<double> vsigma_ud;
    /* vsigma_dd: dϵ/dσ↓↓ */
    Smooth_periodic_function<double> vsigma_dd;

    if (is_gga) {
        vsigma_uu = Smooth_periodic_function<double>(ctx_.spfft(), ctx_.gvec_partition());
        vsigma_ud = Smooth_periodic_function<double>(ctx_.spfft(), ctx_.gvec_partition());
        vsigma_dd = Smooth_periodic_function<double>(ctx_.spfft(), ctx_.gvec_partition());
    }

    sddk::mdarray<double, 1> exc(num_points, memory_t::host, "exc_tmp");
    sddk::mdarray<double, 1> vxc_up(num_points, memory_t::host, "vxc_up_tmp");
    sddk::mdarray<double, 1> vxc_dn(num_points, memory_t::host, "vxc_dn_dmp");

    /* loop over XC functionals */
    for (auto& ixc: xc_func_) {
        PROFILE_START("sirius::Potential::xc_rg_magnetic|libxc");
        if (ixc->is_vdw()) {
#if defined(SIRIUS_USE_VDWXC)
            /* all ranks should make a call because VdW uses FFT internaly */
            if (num_points) {
                ixc->get_vdw(&rho_up.f_rg(0), &rho_dn.f_rg(0), &grad_rho_up_grad_rho_up.f_rg(0),
                             &grad_rho_dn_grad_rho_dn.f_rg(0), vxc_up.at(memory_t::host), vxc_dn.at(memory_t::host),
                             &vsigma_uu.f_rg(0), &vsigma_dd.f_rg(0), exc.at(memory_t::host));
            } else {
                ixc->get_vdw(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            }
#else
            TERMINATE("You should not be there since sirius is not compiled with libVDWXC\n");
#endif
        } else {
            if (num_points) {
            #pragma omp parallel
            {
                /* split local size between threads */
                splindex<splindex_t::block> spl_t(num_points, omp_get_num_threads(), omp_get_thread_num());
                /* if this is an LDA functional */
                if (ixc->is_lda()) {
                    ixc->get_lda(spl_t.local_size(), &rho_up.f_rg(spl_t.global_offset()),
                                 &rho_dn.f_rg(spl_t.global_offset()),
                                 vxc_up.at(memory_t::host, spl_t.global_offset()),
                                 vxc_dn.at(memory_t::host, spl_t.global_offset()),
                                 exc.at(memory_t::host, spl_t.global_offset()));
                }
                /* if this is a GGA functional */
                if (ixc->is_gga()) {
                    ixc->get_gga(spl_t.local_size(), &rho_up.f_rg(spl_t.global_offset()),
                                 &rho_dn.f_rg(spl_t.global_offset()),
                                 &grad_rho_up_grad_rho_up.f_rg(spl_t.global_offset()),
                                 &grad_rho_up_grad_rho_dn.f_rg(spl_t.global_offset()),
                                 &grad_rho_dn_grad_rho_dn.f_rg(spl_t.global_offset()),
                                 vxc_up.at(memory_t::host, spl_t.global_offset()),
                                 vxc_dn.at(memory_t::host, spl_t.global_offset()),
                                 &vsigma_uu.f_rg(spl_t.global_offset()),
                                 &vsigma_ud.f_rg(spl_t.global_offset()),
                                 &vsigma_dd.f_rg(spl_t.global_offset()),
                                 exc.at(memory_t::host, spl_t.global_offset()));
                }
            } // omp parallel region
            } // num_points != 0
        }
        PROFILE_STOP("sirius::Potential::xc_rg_magnetic|libxc");
        if (is_gga) {
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
                vxc_up(ir) -= div_up_gradrho_vsigma.f_rg(ir);
                vxc_dn(ir) -= div_dn_gradrho_vsigma.f_rg(ir);
            }
        }

        #pragma omp parallel for
        for (int irloc = 0; irloc < num_points; irloc++) {
            /* add XC energy density */
            xc_energy_density_->f_rg(irloc) += exc(irloc);
            /* add XC potential */
            xc_potential_->f_rg(irloc) += 0.5 * (vxc_up(irloc) + vxc_dn(irloc));

            double bxc = 0.5 * (vxc_up(irloc) - vxc_dn(irloc));

            /* get the sign between mag and B */
            auto s = utils::sign((rho_up.f_rg(irloc) - rho_dn.f_rg(irloc)) * bxc);

            vector3d<double> m;
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                m[j] = density__.magnetization(j).f_rg(irloc);
            }
            auto m_len = m.length();

            if (m_len > 1e-8) {
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                   effective_magnetic_field(j).f_rg(irloc) += std::abs(bxc) * s * m[j] / m_len;
                }
            } 
        }
    } // for loop over XC functionals
}

template <bool add_pseudo_core__>
void Potential::xc(Density const& density__)
{
    PROFILE("sirius::Potential::xc");

    /* zero all fields */
    xc_potential_->zero();
    xc_energy_density_->zero();
    for (int i = 0; i < ctx_.num_mag_dims(); i++) {
        effective_magnetic_field(i).zero();
    }
    /* quick return */
    if (xc_func_.size() == 0) {
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

    if (ctx_.cfg().control().print_hash()) {
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
