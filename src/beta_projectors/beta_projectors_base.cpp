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

/** \file beta_projectors_base.cpp
 *
 *  \brief Contains implementation of sirius::Beta_projectors_base class.
 */

#include "utils/env.hpp"
#include "beta_projectors_base.hpp"
#include "utils/profiler.hpp"

namespace sirius {

template <typename T>
void Beta_projectors_base<T>::split_in_chunks()
{
    auto& uc = ctx_.unit_cell();

    if (uc.mt_lo_basis_size() == 0) {
        /* no beta projectors at all */
        beta_chunks_ = std::vector<beta_chunk_t>(0);
        num_beta_t_ = 0;
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
        int na = std::min(uc.num_atoms(), (ib + 1) * chunk_size) - ib * chunk_size;
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
            for (int x: {0, 1, 2}) {
                beta_chunks_[ib].atom_pos_(x, i) = pos[x];
            }
            /* number of beta functions for atom */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::nbf), i) = type.mt_basis_size();
            /* offset in beta_gk*/
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::offset), i) = num_beta;
            /* offset in beta_gk_t */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::offset_t), i) = type.offset_lo();
            /* global index of atom */
            beta_chunks_[ib].desc_(static_cast<int>(beta_desc_idx::ia), i) = ia;

            num_beta += type.mt_basis_size();
        }
        /* number of beta-projectors in this chunk */
        beta_chunks_[ib].num_beta_ = num_beta;
        beta_chunks_[ib].offset_ = offset_in_beta_gk;
        offset_in_beta_gk += num_beta;

        if (ctx_.processing_unit() == device_t::GPU) {
            beta_chunks_[ib].desc_.allocate(memory_t::device).copy_to(memory_t::device);
            beta_chunks_[ib].atom_pos_.allocate(memory_t::device).copy_to(memory_t::device);
        }
    }

    max_num_beta_ = 0;
    for (auto& e: beta_chunks_) {
        max_num_beta_ = std::max(max_num_beta_, e.num_beta_);
    }

    num_beta_t_ = 0;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        num_beta_t_ += uc.atom_type(iat).mt_lo_basis_size();
    }
}

template <typename T>
Beta_projectors_base<T>::Beta_projectors_base(Simulation_context& ctx__, Gvec const& gkvec__,
                                              std::vector<int> const& igk__, int N__)
    : ctx_(ctx__)
    , gkvec_(gkvec__)
    //, igk_(igk__)
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
            auto vgk = gkvec_.gkvec<index_domain_t::local>(igk_loc);
            for (auto x: {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy_to(memory_t::device);
    }
}

template <typename T> template <typename F, typename>
matrix<F>
Beta_projectors_base<T>::inner(int chunk__, Wave_functions<T>& phi__, int ispn__, int idx0__, int n__)
{
    PROFILE("sirius::Beta_projectors_base::inner");

    assert(num_gkvec_loc() == phi__.pw_coeffs(ispn__).num_rows_loc());

    int nbeta = chunk(chunk__).num_beta_;

    matrix<F> beta_phi(nbeta, n__, ctx_.mem_pool(ctx_.host_memory_t()));

    /* location of the beta-projectors is always on the memory of the processing unit being used */
    F* pw_coeffs_a_ptr{nullptr};
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            pw_coeffs_a_ptr = reinterpret_cast<F*>(pw_coeffs_a().at(memory_t::host));
            break;
        }
        case device_t::GPU: {
            beta_phi.allocate(ctx_.mem_pool(memory_t::device));
            pw_coeffs_a_ptr = reinterpret_cast<F*>(pw_coeffs_a().at(memory_t::device));
            break;
        }
    }

    local_inner_aux<F>(pw_coeffs_a_ptr, nbeta, phi__, ispn__, idx0__, n__, beta_phi);

    /* copy to host in MPI sequential or parallel case */
    if (is_device_memory(ctx_.preferred_memory_t())) {
        beta_phi.copy_to(memory_t::host);
    }

    /* in parallel case do a reduction */
    if (gkvec_.comm().size() > 1) {
        PROFILE("sirius::Beta_projectors_base::inner|comm");
        /* MPI reduction on the host */
        gkvec_.comm().allreduce(beta_phi.at(memory_t::host), static_cast<int>(beta_phi.size()));
    }

    switch (ctx_.processing_unit()) {
        case device_t::GPU: {
            /* copy back to device */
            if (gkvec_.comm().size() > 1 || is_host_memory(ctx_.preferred_memory_t())) {
                beta_phi.copy_to(memory_t::device);
            }
            break;
        }
        case device_t::CPU: break;
    }

    return beta_phi;
}

#if defined(SIRIUS_GPU)
void create_beta_gk_gpu(int                        num_atoms,
                        int                        num_gkvec,
                        int const*                 beta_desc,
                        std::complex<float> const* beta_gk_t,
                        double const*              gkvec,
                        double const*              atom_pos,
                        std::complex<float>*       beta_gk)
{
    create_beta_gk_gpu_float(num_atoms, num_gkvec, beta_desc, beta_gk_t, gkvec, atom_pos, beta_gk);
}

void create_beta_gk_gpu(int                   num_atoms,
                        int                   num_gkvec,
                        int const*            beta_desc,
                        double_complex const* beta_gk_t,
                        double const*         gkvec,
                        double const*         atom_pos,
                        double_complex*       beta_gk)
{
    create_beta_gk_gpu_double(num_atoms, num_gkvec, beta_desc, beta_gk_t, gkvec, atom_pos, beta_gk);
}
#endif

template <typename T>
void Beta_projectors_base<T>::generate(int ichunk__, int j__)
{
    PROFILE("sirius::Beta_projectors_base::generate");

    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            #pragma omp parallel for
            for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                int ia = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

                double phase = twopi * dot(gkvec_.vk(), ctx_.unit_cell().atom(ia).position());
                auto phase_k = std::exp(std::complex<T>(0.0, phase));

                std::vector<std::complex<double>> phase_gk(num_gkvec_loc());
                for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                    auto G = gkvec_.gvec<index_domain_t::local>(igk_loc);
                    /* total phase e^{-i(G+k)r_{\alpha}} */
                    phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
                }
                int nbeta    = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i);
                int offset_a = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i);
                int offset_t = chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i);
                for (int xi = 0; xi < nbeta; xi++) {
                    for (int igk_loc = 0; igk_loc < num_gkvec_loc(); igk_loc++) {
                        pw_coeffs_a_(igk_loc, offset_a + xi) = pw_coeffs_t_(igk_loc, offset_t + xi, j__) *
                            static_cast<std::complex<T>>(phase_gk[igk_loc]);
                    }
                }
            }
            break;
        }
        case device_t::GPU: {
#if defined(SIRIUS_GPU)
            auto& desc = chunk(ichunk__).desc_;
            create_beta_gk_gpu(chunk(ichunk__).num_atoms_,
                               num_gkvec_loc(),
                               desc.at(memory_t::device),
                               pw_coeffs_t_.at(memory_t::device, 0, 0, j__),
                               gkvec_coord_.at(memory_t::device),
                               chunk(ichunk__).atom_pos_.at(memory_t::device),
                               pw_coeffs_a().at(memory_t::device));
#endif
            /* wave-functions are on CPU but the beta-projectors are on GPU */
            if (gkvec_.comm().rank() == 0 && is_host_memory(ctx_.preferred_memory_t())) {
                /* make beta-projectors for G=0 on the CPU */
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < chunk(ichunk__).num_atoms_; i++) {
                    for (int xi = 0; xi < chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i); xi++) {
                        pw_coeffs_a_g0_(chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset), i) + xi) =
                            pw_coeffs_t_(0, chunk(ichunk__).desc_(static_cast<int>(beta_desc_idx::offset_t), i) + xi, j__);
                    }
                }
            }
            break;
        }
    }
}

template <typename T>
void Beta_projectors_base<T>::prepare()
{
    PROFILE("sirius::Beta_projectors_base::prepare");

    if (max_num_beta() == 0) {
        return;
    }

    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            pw_coeffs_a_ = matrix<std::complex<T>>(num_gkvec_loc(), max_num_beta(), ctx_.mem_pool(ctx_.host_memory_t()),
                "pw_coeffs_a_");
            pw_coeffs_a_g0_ = mdarray<std::complex<T>, 1>(max_num_beta(), ctx_.mem_pool(memory_t::host),
                "pw_coeffs_a_g0_");
            break;
        }
        case device_t::GPU: {
            pw_coeffs_a_ = matrix<std::complex<T>>(num_gkvec_loc(), max_num_beta(), ctx_.mem_pool(memory_t::device),
                "pw_coeffs_a_");
            pw_coeffs_a_g0_ = mdarray<std::complex<T>, 1>(max_num_beta(), ctx_.mem_pool(memory_t::host),
                "pw_coeffs_a_g0_");
            pw_coeffs_a_g0_.allocate(ctx_.mem_pool(memory_t::device));
            break;
        }
    }

    if (ctx_.processing_unit() == device_t::GPU && reallocate_pw_coeffs_t_on_gpu_) {
        pw_coeffs_t_.allocate(ctx_.mem_pool(memory_t::device)).copy_to(memory_t::device);
    }
}

template <typename T>
void Beta_projectors_base<T>::dismiss()
{
    PROFILE("sirius::Beta_projectors_base::dismiss");

    if (ctx_.processing_unit() == device_t::GPU && reallocate_pw_coeffs_t_on_gpu_) {
        pw_coeffs_t_.deallocate(memory_t::device);
    }
    pw_coeffs_a_.deallocate(memory_t::device);
    pw_coeffs_a_g0_.deallocate(memory_t::device);
}


template class Beta_projectors_base<double>;

template
matrix<double>
Beta_projectors_base<double>::inner<double>(int chunk__, Wave_functions<double>& phi__, int ispn__, int idx0__, int n__);

template
matrix<double_complex>
Beta_projectors_base<double>::inner<double_complex>(int chunk__, Wave_functions<double>& phi__, int ispn__, int idx0__, int n__);

#ifdef USE_FP32
template class Beta_projectors_base<float>;


template
matrix<float>
Beta_projectors_base<float>::inner<float>(int chunk__, Wave_functions<float>& phi__, int ispn__, int idx0__, int n__);

template
matrix<std::complex<float>>
Beta_projectors_base<float>::inner<std::complex<float>>(int chunk__, Wave_functions<float>& phi__, int ispn__, int idx0__, int n__);

#endif
} // namespace
