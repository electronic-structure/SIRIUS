/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file beta_projectors_base.cpp
 *
 *  \brief Contains implementation of beta-projectors generator.
 */

#include <stdexcept>
#include "beta_projectors_base.hpp"
#include "core/la/linalg_base.hpp"
#include "core/profiler.hpp"
#include "core/env/env.hpp"
#include "core/wf/wave_functions.hpp"

namespace sirius {

#if defined(SIRIUS_GPU)
void
create_beta_gk_gpu(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<float> const* beta_gk_t,
                   double const* gkvec, double const* atom_pos, std::complex<float>* beta_gk)
{
    create_beta_gk_gpu_float(num_atoms, num_gkvec, beta_desc, beta_gk_t, gkvec, atom_pos, beta_gk);
}

void
create_beta_gk_gpu(int num_atoms, int num_gkvec, int const* beta_desc, std::complex<double> const* beta_gk_t,
                   double const* gkvec, double const* atom_pos, std::complex<double>* beta_gk)
{
    create_beta_gk_gpu_double(num_atoms, num_gkvec, beta_desc, beta_gk_t, gkvec, atom_pos, beta_gk);
}
#endif

/// Internal implementation of beta-projectors generator.
namespace local {

template <class T>
void
beta_projectors_generate_cpu(matrix<std::complex<T>>& pw_coeffs_a, mdarray<std::complex<T>, 3> const& pw_coeffs_t,
                             int ichunk__, int j__, beta_chunk_t const& beta_chunk, Simulation_context const& ctx,
                             fft::Gvec const& gkvec)
{
    PROFILE("beta_projectors_generate_cpu");

    using numeric_t      = std::complex<T>;
    using double_complex = std::complex<double>;

    int num_gkvec_loc = gkvec.count();
    auto& unit_cell   = ctx.unit_cell();

    #pragma omp parallel for
    for (int i = 0; i < beta_chunk.num_atoms_; i++) {
        int ia = beta_chunk.desc_(static_cast<int>(beta_desc_idx::ia), i);

        double phase           = twopi * dot(gkvec.vk(), unit_cell.atom(ia).position());
        double_complex phase_k = std::exp(double_complex(0.0, phase));

        std::vector<double_complex> phase_gk(num_gkvec_loc);
        for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
            auto G = gkvec.gvec(gvec_index_t::local(igk_loc));
            /* total phase e^{-i(G+k)r_{\alpha}} */
            phase_gk[igk_loc] = std::conj(ctx.gvec_phase_factor(G, ia) * phase_k);
        }

        int offset_a = beta_chunk.desc_(beta_desc_idx::offset, i);
        int offset_t = beta_chunk.desc_(beta_desc_idx::offset_t, i);
        int nbeta    = beta_chunk.desc_(beta_desc_idx::nbf, i);
        for (int xi = 0; xi < nbeta; xi++) {
            for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                pw_coeffs_a(igk_loc, offset_a + xi) =
                        pw_coeffs_t(igk_loc, offset_t + xi, j__) * static_cast<numeric_t>(phase_gk[igk_loc]);
            }
        }
    }
}

// explicit instantiation
template void
beta_projectors_generate_cpu<double>(matrix<std::complex<double>>&, mdarray<std::complex<double>, 3> const&, int, int,
                                     beta_chunk_t const&, Simulation_context const&, fft::Gvec const&);
#if defined(SIRIUS_USE_FP32)
// explicit instantiation
template void
beta_projectors_generate_cpu<float>(matrix<std::complex<float>>&, mdarray<std::complex<float>, 3> const&, int, int,
                                    beta_chunk_t const&, Simulation_context const&, fft::Gvec const&);
#endif

template <class T>
void
beta_projectors_generate_gpu(beta_projectors_coeffs_t<T>& out, mdarray<std::complex<T>, 3> const& pw_coeffs_t_device,
                             Simulation_context const& ctx, fft::Gvec const& gkvec,
                             mdarray<double, 2> const& gkvec_coord_, beta_chunk_t const& beta_chunk, int j__)
{
    PROFILE("beta_projectors_generate_gpu");
#if defined(SIRIUS_GPU)
    int num_gkvec_loc = gkvec.count();
    auto& desc        = beta_chunk.desc_;
    create_beta_gk_gpu(beta_chunk.num_atoms_, num_gkvec_loc, desc.at(memory_t::device),
                       pw_coeffs_t_device.at(memory_t::device, 0, 0, j__), gkvec_coord_.at(memory_t::device),
                       beta_chunk.atom_pos_.at(memory_t::device), out.pw_coeffs_a_.at(memory_t::device));
#endif
}

// explicit instantiation
template void
beta_projectors_generate_gpu<double>(beta_projectors_coeffs_t<double>&, mdarray<std::complex<double>, 3> const&,
                                     Simulation_context const&, fft::Gvec const&, mdarray<double, 2> const&,
                                     beta_chunk_t const&, int);
#if defined(SIRIUS_USE_FP32)
// explicit instantiation
template void
beta_projectors_generate_gpu<float>(beta_projectors_coeffs_t<float>&, mdarray<std::complex<float>, 3> const&,
                                    Simulation_context const&, fft::Gvec const&, mdarray<double, 2> const&,
                                    beta_chunk_t const&, int);
#endif
} // namespace local

template <typename T>
void
Beta_projectors_base<T>::split_in_chunks()
{
    auto& uc = ctx_.unit_cell();

    std::vector<int> offset_t(uc.num_atom_types());
    std::generate(offset_t.begin(), offset_t.end(), [n = 0, iat = 0, &uc]() mutable {
        int offs = n;
        n += uc.atom_type(iat++).mt_basis_size();
        return offs;
    });

    if (uc.max_mt_basis_size() == 0) {
        /* no beta projectors at all */
        beta_chunks_ = std::vector<beta_chunk_t>(0);
        num_beta_t_  = 0;
        // max_num_beta_ = 0;
        return;
    }

    /* initial chunk size */
    int chunk_size = std::min(uc.num_atoms(), ctx_.cfg().control().beta_chunk_size());
    /* maximum number of chunks */
    int num_chunks = uc.num_atoms() / chunk_size + std::min(1, uc.num_atoms() % chunk_size);
    /* final maximum chunk size */
    chunk_size = uc.num_atoms() / num_chunks + std::min(1, uc.num_atoms() % num_chunks);

    int offset_in_beta_gk{0};
    beta_chunks_ = std::vector<beta_chunk_t>(num_chunks);

    for (int ib = 0; ib < num_chunks; ib++) {
        /* number of atoms in this chunk */
        int na                      = std::min(uc.num_atoms(), (ib + 1) * chunk_size) - ib * chunk_size;
        beta_chunks_[ib].num_atoms_ = na;
        beta_chunks_[ib].desc_      = mdarray<int, 2>({4, na});
        beta_chunks_[ib].atom_pos_  = mdarray<double, 2>({3, na});

        int num_beta{0};
        for (int i = 0; i < na; i++) {
            /* global index of atom by local index and chunk */
            int ia     = ib * chunk_size + i;
            auto pos   = uc.atom(ia).position();
            auto& type = uc.atom(ia).type();
            /* atom fractional coordinates */
            for (int x : {0, 1, 2}) {
                beta_chunks_[ib].atom_pos_(x, i) = pos[x];
            }
            /* number of beta functions for atom */
            beta_chunks_[ib].desc_(beta_desc_idx::nbf, i) = type.mt_basis_size();
            /* offset in beta_gk*/
            beta_chunks_[ib].desc_(beta_desc_idx::offset, i) = num_beta;
            /* offset in beta_gk_t */
            beta_chunks_[ib].desc_(beta_desc_idx::offset_t, i) = offset_t[type.id()]; // offset_lo();
            /* global index of atom */
            beta_chunks_[ib].desc_(beta_desc_idx::ia, i) = ia;

            num_beta += type.mt_basis_size();
        }
        /* number of beta-projectors in this chunk */
        beta_chunks_[ib].num_beta_ = num_beta;
        beta_chunks_[ib].offset_   = offset_in_beta_gk;
        offset_in_beta_gk += num_beta;

        if (ctx_.processing_unit() == device_t::GPU) {
            beta_chunks_[ib].desc_.allocate(memory_t::device).copy_to(memory_t::device);
            beta_chunks_[ib].atom_pos_.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }
    num_beta_ = offset_in_beta_gk;

    // max_num_beta_ = 0;
    // for (auto& e : beta_chunks_) {
    //     max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
    // }

    num_beta_t_ = 0;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        num_beta_t_ += uc.atom_type(iat).mt_basis_size();
    }
}

template <typename T>
Beta_projectors_base<T>::Beta_projectors_base(Simulation_context& ctx__, fft::Gvec const& gkvec__, int N__)
    : ctx_(ctx__)
    , gkvec_(gkvec__)
    , N_(N__)
{
    split_in_chunks();

    if (!num_beta_t()) {
        return;
    }

    /* allocate memory */
    pw_coeffs_t_ = mdarray<std::complex<T>, 3>({num_gkvec_loc(), num_beta_t(), N__}, mdarray_label("pw_coeffs_t_"));

    if (ctx_.processing_unit() == device_t::GPU) {
        gkvec_coord_ = mdarray<double, 2>({3, num_gkvec_loc()});
        gkvec_coord_.allocate(memory_t::device);
        /* copy G+k vectors */
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            auto vgk = gkvec_.gkvec(gvec_index_t::local(igk_loc));
            for (auto x : {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy_to(memory_t::device);
    }
}

template <class T>
void
Beta_projector_generator<T>::generate(beta_projectors_coeffs_t<T>& out__, int ichunk__, int j__) const
{
    PROFILE("sirius::Beta_projector_generator::generate");
    using numeric_t = std::complex<T>;

    out__.beta_chunk_ = &beta_chunks_[ichunk__];

    auto num_beta = beta_chunks_[ichunk__].num_beta_;
    auto gk_size  = gkvec_.count();

    switch (pu_) {
        case device_t::CPU: {
            if (pw_coeffs_all_atoms_.size()) {
                /* wrap existin buffer */
                out__.pw_coeffs_a_ = matrix<numeric_t>(
                        {gk_size, num_beta}, const_cast<numeric_t*>(pw_coeffs_all_atoms_.at(
                                                     memory_t::host, 0, beta_chunks_[ichunk__].offset_, j__)));
            } else {
                local::beta_projectors_generate_cpu(out__.pw_coeffs_a_, pw_coeffs_t_, ichunk__, j__,
                                                    beta_chunks_[ichunk__], ctx_, gkvec_);
            }
            break;
        }
        case device_t::GPU: {
            if (pw_coeffs_all_atoms_.size() && pw_coeffs_all_atoms_.on_device()) {
                /* wrap existing GPU pointer */
                out__.pw_coeffs_a_ =
                        matrix<numeric_t>({gk_size, num_beta}, nullptr,
                                          const_cast<numeric_t*>(pw_coeffs_all_atoms_.at(
                                                  memory_t::device, 0, beta_chunks_[ichunk__].offset_, j__)));
            } else {
                local::beta_projectors_generate_gpu(out__, pw_coeffs_t_, ctx_, gkvec_, gkvec_coord_,
                                                    beta_chunks_[ichunk__], j__);
            }
            break;
        }
    }
}

template class Beta_projector_generator<double>;
template class Beta_projectors_base<double>;
#if defined(SIRIUS_USE_FP32)
template class Beta_projector_generator<float>;
template class Beta_projectors_base<float>;
#endif

} // namespace sirius
