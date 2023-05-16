// Copyright (c) 2013-2022 Anton Kozhevnikov, Thomas Schulthess
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

/** \file beta_projectors_base.cpp
 *
 *  \brief Contains implementation of sirius::Beta_projectors_base class.
 */

#include "linalg/linalg_base.hpp"
#include "utils/env.hpp"
#include "beta_projectors_base.hpp"
#include "utils/profiler.hpp"
#include "wave_functions.hpp"
#include <stdexcept>

namespace sirius {

using namespace sddk;

// using double_complex = std::complex<double>;

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

namespace local {

template <class T>
void
beta_projectors_generate_cpu(matrix<std::complex<T>>& pw_coeffs_a, const mdarray<std::complex<T>, 3>& pw_coeffs_t,
                             int ichunk__, int j__, const beta_chunk_t& beta_chunk, const Simulation_context& ctx,
                             const fft::Gvec& gkvec)
{
    PROFILE("sirius::Beta_projectors_base::generate");

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
            auto G = gkvec.gvec<index_domain_t::local>(igk_loc);
            /* total phase e^{-i(G+k)r_{\alpha}} */
            phase_gk[igk_loc] = std::conj(ctx.gvec_phase_factor(G, ia) * phase_k);
        }

        int offset_a = beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset), i);
        int offset_t = beta_chunk.desc_(static_cast<int>(beta_desc_idx::offset_t), i);
        int nbeta    = beta_chunk.desc_(static_cast<int>(beta_desc_idx::nbf), i);
        for (int xi = 0; xi < nbeta; xi++) {
            for (int igk_loc = 0; igk_loc < num_gkvec_loc; igk_loc++) {
                pw_coeffs_a(igk_loc, offset_a + xi) =
                    pw_coeffs_t(igk_loc, offset_t + xi, j__) * static_cast<numeric_t>(phase_gk[igk_loc]);
            }
        }
    }
}

// explicit instantiation
template void beta_projectors_generate_cpu<double>(matrix<std::complex<double>>&,
                                                   const mdarray<std::complex<double>, 3>&, int, int,
                                                   const beta_chunk_t&, const Simulation_context&, const fft::Gvec&);
#ifdef USE_FP32
// explicit instantiation
template void beta_projectors_generate_cpu<float>(matrix<std::complex<float>>&, const mdarray<std::complex<float>, 3>&,
                                                  int, int, const beta_chunk_t&, const Simulation_context&,
                                                  const fft::Gvec&);
#endif

template <class T>
void
beta_projectors_generate_gpu(beta_projectors_coeffs_t<T>& out, const mdarray<std::complex<T>, 3>& pw_coeffs_t_device,
                             const mdarray<std::complex<T>, 3>& pw_coeffs_t_host, const Simulation_context& ctx,
                             const fft::Gvec& gkvec, const mdarray<double, 2>& gkvec_coord_,
                             const beta_chunk_t& beta_chunk, int j__)
{
    int num_gkvec_loc = gkvec.count();
    PROFILE("sirius::Beta_projectors_base::generate");
#if defined(SIRIUS_GPU)
    auto& desc = beta_chunk.desc_;
    create_beta_gk_gpu(beta_chunk.num_atoms_, num_gkvec_loc, desc.at(memory_t::device),
                       pw_coeffs_t_device.at(memory_t::device, 0, 0, j__), gkvec_coord_.at(memory_t::device),
                       beta_chunk.atom_pos_.at(memory_t::device), out.pw_coeffs_a.at(memory_t::device));
#endif
}

// explicit instantiation
template void beta_projectors_generate_gpu<double>(beta_projectors_coeffs_t<double>&,
                                                   const mdarray<std::complex<double>, 3>&,
                                                   const mdarray<std::complex<double>, 3>&, const Simulation_context&,
                                                   const fft::Gvec&, const mdarray<double, 2>&, const beta_chunk_t&,
                                                   int);
#ifdef USE_FP32
// explicit instantiation
template void beta_projectors_generate_gpu<float>(beta_projectors_coeffs_t<float>&,
                                                  const mdarray<std::complex<float>, 3>&,
                                                  const mdarray<std::complex<float>, 3>&, const Simulation_context&,
                                                  const fft::Gvec&, const mdarray<double, 2>&, const beta_chunk_t&,
                                                  int);
#endif
} // namespace local

template <typename T>
void
Beta_projectors_base<T>::split_in_chunks()
{
    auto& uc = ctx_.unit_cell();

    if (uc.mt_lo_basis_size() == 0) {
        /* no beta projectors at all */
        beta_chunks_  = std::vector<beta_chunk_t>(0);
        num_beta_t_   = 0;
        max_num_beta_ = 0;
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
        beta_chunks_[ib].desc_      = mdarray<int, 2>(4, na);
        beta_chunks_[ib].atom_pos_  = mdarray<double, 2>(3, na);

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
            beta_chunks_[ib].desc_(beta_desc_idx::offset_t, i) = type.offset_lo();
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
    num_total_beta_ = offset_in_beta_gk;

    max_num_beta_ = 0;
    for (auto& e : beta_chunks_) {
        max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
    }

    num_beta_t_ = 0;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        num_beta_t_ += uc.atom_type(iat).mt_lo_basis_size();
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
    pw_coeffs_t_ = mdarray<std::complex<T>, 3>(num_gkvec_loc(), num_beta_t(), N__, memory_t::host, "pw_coeffs_t_");

    if (ctx_.processing_unit() == device_t::GPU) {
        gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc());
        gkvec_coord_.allocate(memory_t::device);
        /* copy G+k vectors */
        for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
            auto vgk = gkvec_.template gkvec<index_domain_t::local>(igk_loc);
            for (auto x : {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy_to(memory_t::device);
    }
}

template <class T>
void
Beta_projector_generator<T>::generate(beta_projectors_coeffs_t<T>& out, int ichunk__) const
{
    PROFILE("sirius::Beta_projectors_base::generate");
    using numeric_t = std::complex<T>;

    int j__        = 0;
    out.beta_chunk = beta_chunks_.at(ichunk__);

    auto num_beta = out.beta_chunk.num_beta_;
    auto gk_size  = gkvec_.count();

    switch (processing_unit_) {
        case device_t::CPU: {
            out.pw_coeffs_a =
                matrix<numeric_t>(const_cast<numeric_t*>(&beta_pw_all_atoms_(0, beta_chunks_[ichunk__].offset_)),
                                  gk_size, beta_chunks_[ichunk__].num_beta_);
            break;
        }
        case device_t::GPU: {
            out.pw_coeffs_a =
                sddk::matrix<numeric_t>(nullptr, out.__pw_coeffs_a_buffer.device_data(), gk_size, num_beta);
            local::beta_projectors_generate_gpu(out, pw_coeffs_t_device_, pw_coeffs_t_host_, ctx_, gkvec_, gkvec_coord_,
                                                beta_chunks_[ichunk__], j__);
            break;
        }
    }
}

template <class T>
void
Beta_projector_generator<T>::generate(beta_projectors_coeffs_t<T>& out, int ichunk__, int j__) const
{
    PROFILE("sirius::Beta_projectors_base::generate");
    using numeric_t = std::complex<T>;

    out.beta_chunk = beta_chunks_.at(ichunk__);

    auto num_beta = out.beta_chunk.num_beta_;
    auto gk_size  = gkvec_.count();

    switch (processing_unit_) {
        case device_t::CPU: {
            // allocate pw_coeffs_a
            out.pw_coeffs_a = sddk::matrix<numeric_t>(gk_size, num_beta, sddk::get_memory_pool(sddk::memory_t::host));
            local::beta_projectors_generate_cpu(out.pw_coeffs_a, pw_coeffs_t_host_, ichunk__, j__,
                                                beta_chunks_[ichunk__], ctx_, gkvec_);
            break;
        }
        case device_t::GPU: {
            // view of internal buffer with correct number of cols (= num_beta)
            out.pw_coeffs_a =
                sddk::matrix<numeric_t>(nullptr, out.__pw_coeffs_a_buffer.device_data(), gk_size, num_beta);
            // g0 coefficients reside in host memory

            local::beta_projectors_generate_gpu(out, pw_coeffs_t_device_, pw_coeffs_t_host_, ctx_, gkvec_, gkvec_coord_,
                                                beta_chunks_[ichunk__], j__);
            break;
        }
    }
}

template class Beta_projector_generator<double>;
#ifdef USE_FP32
template class Beta_projector_generator<float>;
#endif

template class Beta_projectors_base<double>;
#ifdef USE_FP32
template class Beta_projectors_base<float>;
#endif

} // namespace sirius
