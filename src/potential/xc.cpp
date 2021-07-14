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

template <bool add_pseudo_core__>
void Potential::xc_rg_nonmagnetic(Density const& density__)
{
    PROFILE("sirius::Potential::xc_rg_nonmagnetic");

    bool const use_2nd_deriv{false};

    auto& gvp = ctx_.gvec_partition();

    bool is_gga = is_gradient_correction();

    int num_points = ctx_.spfft().local_slice_size();

    Smooth_periodic_function<double> rho(ctx_.spfft(), gvp);

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

    Smooth_periodic_function<double> vsigma;
    if (is_gga) {
        vsigma = Smooth_periodic_function<double>(ctx_.spfft(), ctx_.gvec_partition());
        vsigma_[0]->zero();
    }

    sddk::mdarray<double, 1> exc(num_points, memory_t::host, "exc_tmp");
    sddk::mdarray<double, 1> vxc(num_points, memory_t::host, "vxc_tmp");

    /* loop over XC functionals */
    for (auto& ixc: xc_func_) {
        PROFILE_START("sirius::Potential::xc_rg_nonmagnetic|libxc");
        if (ixc->is_vdw()) {
#if defined(SIRIUS_USE_VDWXC)
            /* all ranks should make a call because VdW uses FFT internaly */
            if (num_points) {
                /* Van der Walls correction */
                ixc->get_vdw(&rho.f_rg(0), &grad_rho_grad_rho.f_rg(0), vxc.at(memory_t::host), &vsigma.f_rg(0),
                             exc.at(memory_t::host));
            } else {
                ixc->get_vdw(nullptr, nullptr, nullptr, nullptr, nullptr);
            }
#else
            TERMINATE("You should not be there since SIRIUS is not compiled with libVDWXC support\n");
#endif
        } else {
            if (num_points) {
            #pragma omp parallel
            {
                /* split local size between threads */
                splindex<splindex_t::block> spl_t(num_points, omp_get_num_threads(), omp_get_thread_num());
                /* if this is an LDA functional */
                if (ixc->is_lda()) {
                    ixc->get_lda(spl_t.local_size(), &rho.f_rg(spl_t.global_offset()),
                                 vxc.at(memory_t::host, spl_t.global_offset()),
                                 exc.at(memory_t::host, spl_t.global_offset()));
                }
                /* if this is a GGA functional */
                if (ixc->is_gga()) {
                    ixc->get_gga(spl_t.local_size(), &rho.f_rg(spl_t.global_offset()),
                                 &grad_rho_grad_rho.f_rg(spl_t.global_offset()),
                                 vxc.at(memory_t::host, spl_t.global_offset()),
                                 &vsigma.f_rg(spl_t.global_offset()),
                                 exc.at(memory_t::host, spl_t.global_offset()));
                }
            } // omp parallel region
                        ///* this is the same expression between gga and vdw corrections.
                        // * The functionals are different that's all */
                        //for (int i = 0; i < spl_np_t.local_size(); i++) {
                        //    /* add Exc contribution */
                        //    exc_tmp(spl_np_t[i]) += exc_t[i];

                        //    /* directly add to Vxc available contributions */
                        //    //if (use_2nd_deriv) {
                        //    //    vxc_tmp(spl_np_t[i]) += (vrho_t[i] - 2 * vsigma_t[i] * lapl_rho.f_rg(spl_np_t[i]));
                        //    //} else {
                        //    //    vxc_tmp(spl_np_t[i]) += vrho_t[i];
                        //    //}
                        //    vxc_tmp(spl_np_t[i]) += vrho_t[i];

                        //    /* save the sigma derivative */
                        //    vsigma_tmp(spl_np_t[i]) += vsigma_t[i];
            } // num_points != 0
        }
        PROFILE_STOP("sirius::Potential::xc_rg_nonmagnetic|libxc");
        if (ixc->is_gga()) { /* generic for gga and vdw */
            #pragma omp parallel for
            for (int ir = 0; ir < num_points; ir++) {
                /* save for future reuse in XC stress calculation */
                vsigma_[0]->f_rg(ir) += vsigma.f_rg(ir);
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
                    vxc(ir) -= 2 * (vsigma.f_rg(ir) * lapl_rho.f_rg(ir) + grad_vsigma_grad_rho.f_rg(ir));
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
                    vxc(ir) -= 2 * div_vsigma_grad_rho.f_rg(ir);
                }
            }
        }
        #pragma omp parallel for
        for (int ir = 0; ir < num_points; ir++) {
            xc_energy_density_->f_rg(ir) += exc(ir);
            xc_potential_->f_rg(ir) += vxc(ir);
        }
    } // for loop over xc functionals

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

    auto result = get_rho_up_dn<add_pseudo_core__>(density__, add_delta_rho_xc_, add_delta_mag_xc_);

    auto& rho_up = *result[0];
    auto& rho_dn = *result[1];

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
        for (int i = 0; i < 3; i++) {
            vsigma_[i]->zero();
        }
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
        if (ixc->is_gga()) {
            #pragma omp parallel for
            for (int ir = 0; ir < num_points; ir++) {
                /* save for future reuse in XC stress calculation */
                vsigma_[0]->f_rg(ir) += vsigma_uu.f_rg(ir);
                vsigma_[1]->f_rg(ir) += vsigma_ud.f_rg(ir);
                vsigma_[2]->f_rg(ir) += vsigma_dd.f_rg(ir);
            }

            Smooth_periodic_vector_function<double> up_gradrho_vsigma(ctx_.spfft(), ctx_.gvec_partition());
            Smooth_periodic_vector_function<double> dn_gradrho_vsigma(ctx_.spfft(), ctx_.gvec_partition());
            for (int x: {0, 1, 2}) {
                for(int ir = 0; ir < num_points; ir++) {
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
