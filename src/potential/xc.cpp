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
#include "core/typedefs.hpp"
#include "core/omp.hpp"
#include "core/profiler.hpp"
#include "xc_functional.hpp"

namespace sirius {

template <bool add_pseudo_core__>
void
Potential::xc_rg_nonmagnetic(Density const& density__, bool use_lapl__)
{
    PROFILE("sirius::Potential::xc_rg_nonmagnetic");

    auto gvp = ctx_.gvec_fft_sptr();

    bool is_gga = is_gradient_correction();

    int num_points = ctx_.spfft<double>().local_slice_size();

    Smooth_periodic_function<double> rho(ctx_.spfft<double>(), gvp);

    /* we can use this comm for parallelization */
    // auto& comm = ctx_.gvec().comm_ortho_fft();
    /* split real-space points between available ranks */
    // splindex<block> spl_np(num_points, comm.size(), comm.rank());

    /* check for negative values */
    double rhomin{0};
    for (int ir = 0; ir < num_points; ir++) {

        // int ir = spl_np[irloc];
        double d = density__.rho().rg().value(ir);
        if (add_pseudo_core__) {
            d += density__.rho_pseudo_core().value(ir);
        }
        d *= (1 + add_delta_rho_xc_);

        rhomin        = std::min(rhomin, d);
        rho.value(ir) = std::max(d, 0.0);
    }
    mpi::Communicator(ctx_.spfft<double>().communicator()).allreduce<double, mpi::op_t::min>(&rhomin, 1);
    /* even a small negative density is a sign of something bing wrong; don't remove this check */
    if (rhomin < 0.0 && ctx_.comm().rank() == 0) {
        std::stringstream s;
        s << "Interstitial charge density has negative values" << std::endl << "most negatve value : " << rhomin;
        RTE_WARNING(s);
    }

    if (env::print_hash()) {
        auto h = rho.hash_f_rg();
        print_hash("rho", h, ctx_.out());
    }

    if (env::print_checksum()) {
        auto cs = density__.rho().rg().checksum_rg();
        print_checksum("rho_rg", cs, ctx_.out());
    }

    Smooth_periodic_vector_function<double> grad_rho;
    Smooth_periodic_function<double> grad_rho_grad_rho;
    Smooth_periodic_function<double> vsigma;

    if (is_gga) {
        /* transform to reciprocal space */
        rho.fft_transform(-1);

        /* generate pw coeffs of the gradient and transform to real space */
        grad_rho = to_rg(gradient(rho));

        /* product of gradients */
        grad_rho_grad_rho = dot(grad_rho, grad_rho);

        if (env::print_hash()) {
            auto h1 = grad_rho_grad_rho.hash_f_rg();
            print_hash("grad_rho_grad_rho", h1, ctx_.out());
        }

        vsigma = Smooth_periodic_function<double>(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        vsigma_[0]->zero();
    }

    mdarray<double, 1> exc({num_points}, mdarray_label("exc_tmp"));
    mdarray<double, 1> vxc({num_points}, mdarray_label("vxc_tmp"));

    /* loop over XC functionals */
    for (auto& ixc : xc_func_) {
        PROFILE_START("sirius::Potential::xc_rg_nonmagnetic|libxc");
        if (ixc.is_vdw()) {
#if defined(SIRIUS_USE_VDWXC)
            /* all ranks should make a call because VdW uses FFT internaly */
            if (num_points) {
                /* Van der Walls correction */
                ixc.get_vdw(&rho.value(0), &grad_rho_grad_rho.value(0), vxc.at(memory_t::host), &vsigma.value(0),
                            exc.at(memory_t::host));
            } else {
                ixc.get_vdw(nullptr, nullptr, nullptr, nullptr, nullptr);
            }
#else
            RTE_THROW("You should not be there since SIRIUS is not compiled with libVDWXC support\n");
#endif
        } else {
            if (num_points) {
                #pragma omp parallel
                {
                    /* split local size between threads */
                    splindex_block<> spl_t(num_points, n_blocks(omp_get_num_threads()), block_id(omp_get_thread_num()));
                    /* if this is an LDA functional */
                    if (ixc.is_lda()) {
                        ixc.get_lda(spl_t.local_size(), &rho.value(spl_t.global_offset()),
                                    vxc.at(memory_t::host, spl_t.global_offset()),
                                    exc.at(memory_t::host, spl_t.global_offset()));
                    }
                    /* if this is a GGA functional */
                    if (ixc.is_gga()) {
                        ixc.get_gga(spl_t.local_size(), &rho.value(spl_t.global_offset()),
                                    &grad_rho_grad_rho.value(spl_t.global_offset()),
                                    vxc.at(memory_t::host, spl_t.global_offset()), &vsigma.value(spl_t.global_offset()),
                                    exc.at(memory_t::host, spl_t.global_offset()));
                    }
                } // omp parallel region
            }     // num_points != 0
        }
        PROFILE_STOP("sirius::Potential::xc_rg_nonmagnetic|libxc");
        if (ixc.is_gga()) { /* generic for gga and vdw */
            #pragma omp parallel for
            for (int ir = 0; ir < num_points; ir++) {
                /* save for future reuse in XC stress calculation */
                vsigma_[0]->value(ir) += vsigma.value(ir);
            }

            if (use_lapl__) {
                /* generate pw coeffs of the laplacian */
                auto lapl_rho = to_rg(laplacian(rho));

                /* forward transform vsigma to plane-wave domain */
                vsigma.fft_transform(-1);

                /* gradient of vsigma in plane-wave domain */
                auto grad_vsigma = to_rg(gradient(vsigma));

                /* compute scalar product of two gradients */
                auto grad_vsigma_grad_rho = dot(grad_vsigma, grad_rho);

                /* add remaining term to Vxc */
                #pragma omp parallel for
                for (int ir = 0; ir < num_points; ir++) {
                    vxc(ir) -= 2 * (vsigma.value(ir) * lapl_rho.value(ir) + grad_vsigma_grad_rho.value(ir));
                }
            } else {
                Smooth_periodic_vector_function<double> vsigma_grad_rho(ctx_.spfft<double>(), gvp);

                for (int x : {0, 1, 2}) {
                    for (int ir = 0; ir < num_points; ir++) {
                        vsigma_grad_rho[x].value(ir) = grad_rho[x].value(ir) * vsigma.value(ir);
                    }
                    /* transform to plane wave domain */
                    vsigma_grad_rho[x].fft_transform(-1);
                }
                auto div_vsigma_grad_rho = to_rg(divergence(vsigma_grad_rho));
                for (int ir = 0; ir < num_points; ir++) {
                    vxc(ir) -= 2 * div_vsigma_grad_rho.value(ir);
                }
            }
        }

        #pragma omp parallel for
        for (int ir = 0; ir < num_points; ir++) {
            xc_energy_density_->rg().value(ir) += exc(ir);
            xc_potential_->rg().value(ir) += vxc(ir);
        }
    } // for loop over xc functionals

    if (env::print_checksum()) {
        auto cs = xc_potential_->rg().checksum_rg();
        print_checksum("exc", cs, ctx_.out());
    }
}

template <bool add_pseudo_core__>
void
Potential::xc_rg_magnetic(Density const& density__, bool use_lapl__)
{
    PROFILE("sirius::Potential::xc_rg_magnetic");

    bool is_gga = is_gradient_correction();

    int num_points = ctx_.spfft<double>().local_slice_size();

    auto result = get_rho_up_dn<add_pseudo_core__>(density__, add_delta_rho_xc_, add_delta_mag_xc_);

    auto& rho_up = *result[0];
    auto& rho_dn = *result[1];

    if (env::print_hash()) {
        auto h1 = rho_up.hash_f_rg();
        auto h2 = rho_dn.hash_f_rg();
        print_hash("rho_up", h1, ctx_.out());
        print_hash("rho_dn", h2, ctx_.out());
    }

    Smooth_periodic_vector_function<double> grad_rho_up;
    Smooth_periodic_vector_function<double> grad_rho_dn;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_up;
    Smooth_periodic_function<double> grad_rho_up_grad_rho_dn;
    Smooth_periodic_function<double> grad_rho_dn_grad_rho_dn;

    /* vsigma_uu: dϵ/dσ↑↑ */
    Smooth_periodic_function<double> vsigma_uu;
    /* vsigma_ud: dϵ/dσ↑↓ */
    Smooth_periodic_function<double> vsigma_ud;
    /* vsigma_dd: dϵ/dσ↓↓ */
    Smooth_periodic_function<double> vsigma_dd;

    if (is_gga) {
        /* get plane-wave coefficients of densities */
        rho_up.fft_transform(-1);
        rho_dn.fft_transform(-1);

        /* generate pw coeffs of the gradient and laplacian */
        grad_rho_up = to_rg(gradient(rho_up));
        grad_rho_dn = to_rg(gradient(rho_dn));

        /* product of gradients */
        grad_rho_up_grad_rho_up = dot(grad_rho_up, grad_rho_up);
        grad_rho_up_grad_rho_dn = dot(grad_rho_up, grad_rho_dn);
        grad_rho_dn_grad_rho_dn = dot(grad_rho_dn, grad_rho_dn);

        if (env::print_hash()) {
            auto h1 = grad_rho_up_grad_rho_up.hash_f_rg();
            auto h2 = grad_rho_up_grad_rho_dn.hash_f_rg();
            auto h3 = grad_rho_dn_grad_rho_dn.hash_f_rg();

            print_hash("grad_rho_up_grad_rho_up", h1, ctx_.out());
            print_hash("grad_rho_up_grad_rho_dn", h2, ctx_.out());
            print_hash("grad_rho_dn_grad_rho_dn", h3, ctx_.out());
        }

        vsigma_uu = Smooth_periodic_function<double>(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        vsigma_ud = Smooth_periodic_function<double>(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        vsigma_dd = Smooth_periodic_function<double>(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
        for (int i = 0; i < 3; i++) {
            vsigma_[i]->zero();
        }
    }

    mdarray<double, 1> exc({num_points}, mdarray_label("exc_tmp"));
    mdarray<double, 1> vxc_up({num_points}, mdarray_label("vxc_up_tmp"));
    mdarray<double, 1> vxc_dn({num_points}, mdarray_label("vxc_dn_dmp"));

    /* loop over XC functionals */
    for (auto& ixc : xc_func_) {
        PROFILE_START("sirius::Potential::xc_rg_magnetic|libxc");
        if (ixc.is_vdw()) {
#if defined(SIRIUS_USE_VDWXC)
            /* all ranks should make a call because VdW uses FFT internaly */
            if (num_points) {
                ixc.get_vdw(&rho_up.value(0), &rho_dn.value(0), &grad_rho_up_grad_rho_up.value(0),
                            &grad_rho_dn_grad_rho_dn.value(0), vxc_up.at(memory_t::host), vxc_dn.at(memory_t::host),
                            &vsigma_uu.value(0), &vsigma_dd.value(0), exc.at(memory_t::host));
            } else {
                ixc.get_vdw(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
            }
#else
            RTE_THROW("You should not be there since sirius is not compiled with libVDWXC\n");
#endif
        } else {
            if (num_points) {
                #pragma omp parallel
                {
                    /* split local size between threads */
                    splindex_block<> spl_t(num_points, n_blocks(omp_get_num_threads()), block_id(omp_get_thread_num()));
                    /* if this is an LDA functional */
                    if (ixc.is_lda()) {
                        ixc.get_lda(spl_t.local_size(), &rho_up.value(spl_t.global_offset()),
                                    &rho_dn.value(spl_t.global_offset()),
                                    vxc_up.at(memory_t::host, spl_t.global_offset()),
                                    vxc_dn.at(memory_t::host, spl_t.global_offset()),
                                    exc.at(memory_t::host, spl_t.global_offset()));
                    }
                    /* if this is a GGA functional */
                    if (ixc.is_gga()) {
                        ixc.get_gga(spl_t.local_size(), &rho_up.value(spl_t.global_offset()),
                                    &rho_dn.value(spl_t.global_offset()),
                                    &grad_rho_up_grad_rho_up.value(spl_t.global_offset()),
                                    &grad_rho_up_grad_rho_dn.value(spl_t.global_offset()),
                                    &grad_rho_dn_grad_rho_dn.value(spl_t.global_offset()),
                                    vxc_up.at(memory_t::host, spl_t.global_offset()),
                                    vxc_dn.at(memory_t::host, spl_t.global_offset()),
                                    &vsigma_uu.value(spl_t.global_offset()), &vsigma_ud.value(spl_t.global_offset()),
                                    &vsigma_dd.value(spl_t.global_offset()),
                                    exc.at(memory_t::host, spl_t.global_offset()));
                    }
                } // omp parallel region
            }     // num_points != 0
        }
        PROFILE_STOP("sirius::Potential::xc_rg_magnetic|libxc");
        if (ixc.is_gga()) {
            #pragma omp parallel for
            for (int ir = 0; ir < num_points; ir++) {
                /* save for future reuse in XC stress calculation */
                vsigma_[0]->value(ir) += vsigma_uu.value(ir);
                vsigma_[1]->value(ir) += vsigma_ud.value(ir);
                vsigma_[2]->value(ir) += vsigma_dd.value(ir);
            }

            if (use_lapl__) {
                auto lapl_rho_up = to_rg(laplacian(rho_up));
                auto lapl_rho_dn = to_rg(laplacian(rho_dn));
                /* forward transform vsigma to plane-wave domain */
                vsigma_uu.fft_transform(-1);
                vsigma_ud.fft_transform(-1);
                vsigma_dd.fft_transform(-1);

                /* gradients of vsigmas in plane-wave domain */
                auto grad_vsigma_uu = to_rg(gradient(vsigma_uu));
                auto grad_vsigma_ud = to_rg(gradient(vsigma_ud));
                auto grad_vsigma_dd = to_rg(gradient(vsigma_dd));

                auto grad_vsigma_uu_grad_rho_up = dot(grad_vsigma_uu, grad_rho_up);
                auto grad_vsigma_ud_grad_rho_dn = dot(grad_vsigma_ud, grad_rho_dn);

                auto grad_vsigma_dd_grad_rho_dn = dot(grad_vsigma_dd, grad_rho_dn);
                auto grad_vsigma_ud_grad_rho_up = dot(grad_vsigma_ud, grad_rho_up);

                #pragma omp parallel for
                for (int ir = 0; ir < num_points; ir++) {
                    vxc_up(ir) -=
                            2 * (vsigma_uu.value(ir) * lapl_rho_up.value(ir) + grad_vsigma_uu_grad_rho_up.value(ir)) +
                            grad_vsigma_ud_grad_rho_dn.value(ir) + vsigma_ud.value(ir) * lapl_rho_dn.value(ir);

                    vxc_dn(ir) -=
                            2 * (vsigma_dd.value(ir) * lapl_rho_dn.value(ir) + grad_vsigma_dd_grad_rho_dn.value(ir)) +
                            grad_vsigma_ud_grad_rho_up.value(ir) + vsigma_ud.value(ir) * lapl_rho_up.value(ir);
                }
            } else {
                Smooth_periodic_vector_function<double> up_gradrho_vsigma(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                Smooth_periodic_vector_function<double> dn_gradrho_vsigma(ctx_.spfft<double>(), ctx_.gvec_fft_sptr());
                for (int x : {0, 1, 2}) {
                    for (int ir = 0; ir < num_points; ir++) {
                        up_gradrho_vsigma[x].value(ir) = 2 * grad_rho_up[x].value(ir) * vsigma_uu.value(ir) +
                                                         grad_rho_dn[x].value(ir) * vsigma_ud.value(ir);
                        dn_gradrho_vsigma[x].value(ir) = 2 * grad_rho_dn[x].value(ir) * vsigma_dd.value(ir) +
                                                         grad_rho_up[x].value(ir) * vsigma_ud.value(ir);
                    }
                    /* transform to plane wave domain */
                    up_gradrho_vsigma[x].fft_transform(-1);
                    dn_gradrho_vsigma[x].fft_transform(-1);
                }

                auto div_up_gradrho_vsigma = to_rg(divergence(up_gradrho_vsigma));
                auto div_dn_gradrho_vsigma = to_rg(divergence(dn_gradrho_vsigma));

                /* add remaining term to Vxc */
                #pragma omp parallel for
                for (int ir = 0; ir < num_points; ir++) {
                    vxc_up(ir) -= div_up_gradrho_vsigma.value(ir);
                    vxc_dn(ir) -= div_dn_gradrho_vsigma.value(ir);
                }
            }
        }

        #pragma omp parallel for
        for (int irloc = 0; irloc < num_points; irloc++) {
            /* add XC energy density */
            xc_energy_density_->rg().value(irloc) += exc(irloc);
            /* add XC potential */
            xc_potential_->rg().value(irloc) += 0.5 * (vxc_up(irloc) + vxc_dn(irloc));

            double bxc = 0.5 * (vxc_up(irloc) - vxc_dn(irloc));

            /* get the sign between mag and B */
            auto s = sign((rho_up.value(irloc) - rho_dn.value(irloc)) * bxc);

            r3::vector<double> m;
            for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                m[j] = density__.mag(j).rg().value(irloc);
            }
            auto m_len = m.length();

            if (m_len > 1e-8) {
                for (int j = 0; j < ctx_.num_mag_dims(); j++) {
                    effective_magnetic_field(j).rg().value(irloc) += std::abs(bxc) * s * m[j] / m_len;
                }
            }
        }
    } // for loop over XC functionals
}

template <typename T>
inline void
remove_high_pw(Simulation_context const& ctx__, Smooth_periodic_function<T>& f__)
{
    f__.fft_transform(-1);
    #pragma omp parallel for
    for (int ig = 0; ig < ctx__.gvec().count(); ig++) {
        if (ctx__.gvec().gvec_len(gvec_index_t::local(ig)) > ctx__.cfg().parameters().veff_pw_cutoff()) {
            f__.f_pw_local(ig) = 0;
        }
    }
    f__.fft_transform(1);
}

template <bool add_pseudo_core__>
void
Potential::xc(Density const& density__)
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

    auto use_lapl = this->ctx_.cfg().settings().xc_use_lapl();

    if (ctx_.full_potential()) {
        xc_mt(density__, use_lapl);
    }

    if (ctx_.num_spins() == 1) {
        xc_rg_nonmagnetic<add_pseudo_core__>(density__, use_lapl);
    } else {
        xc_rg_magnetic<add_pseudo_core__>(density__, use_lapl);
    }

    if (ctx_.cfg().parameters().veff_pw_cutoff() > 0) {
        for (int j = 0; j < ctx_.num_mag_dims(); j++) {
            remove_high_pw(ctx_, effective_magnetic_field(j).rg());
        }
        remove_high_pw(ctx_, xc_potential_->rg());
        remove_high_pw(ctx_, xc_energy_density_->rg());
    }

    if (env::print_hash()) {
        auto h = xc_energy_density_->rg().hash_f_rg();
        print_hash("Exc", h, ctx_.out());
    }
}

// explicit instantiation
template void
Potential::xc_rg_nonmagnetic<true>(Density const&, bool);
template void
Potential::xc_rg_nonmagnetic<false>(Density const&, bool);
template void
Potential::xc_rg_magnetic<true>(Density const&, bool);
template void
Potential::xc_rg_magnetic<false>(Density const&, bool);
template void
Potential::xc<true>(Density const&);
template void
Potential::xc<false>(Density const&);

} // namespace sirius
