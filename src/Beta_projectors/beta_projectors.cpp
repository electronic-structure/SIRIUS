// Copyright (c) 2013-2016 Anton Kozhevnikov, Thomas Schulthess
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

/** \file beta_projectors.cpp
 *
 *  \brief Contains implementation of sirius::Beta_projectors class.
 */

#include "beta_projectors.h"

namespace sirius {

Beta_projectors::Beta_projectors(Simulation_context const& ctx__,
                                 Communicator const& comm__,
                                 Gvec const& gkvec__)
    : comm_(comm__)
    , unit_cell_(ctx__.unit_cell())
    , gkvec_(gkvec__)
    , lmax_beta_(unit_cell_.lmax())
    , pu_(ctx__.processing_unit())
{
    PROFILE("sirius::Beta_projectors::Beta_projectors");

    num_gkvec_loc_ = gkvec_.gvec_count(comm_.rank());

    split_in_chunks();

    generate_beta_gk_t(ctx__);

    #ifdef __GPU
    if (pu_ == GPU) {
        gkvec_coord_ = mdarray<double, 2>(3, num_gkvec_loc_, memory_t::host | memory_t::device);
        /* copy G+k vectors */
        for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
            int igk  = gkvec_.gvec_offset(comm_.rank()) + igk_loc;
            auto vgk = gkvec_.gkvec(igk);
            for (auto x: {0, 1, 2}) {
                gkvec_coord_(x, igk_loc) = vgk[x];
            }
        }
        gkvec_coord_.copy_to_device();

        beta_gk_t_.allocate(memory_t::device);
        beta_gk_t_.copy_to_device();
    }
    beta_gk_gpu_ = matrix<double_complex>(num_gkvec_loc_, max_num_beta_, memory_t::none);
    #endif

    beta_gk_a_ = matrix<double_complex>(num_gkvec_loc_, unit_cell_.mt_lo_basis_size());

    #pragma omp for
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        double phase = twopi * (gkvec_.vk() * unit_cell_.atom(ia).position());
        double_complex phase_k = std::exp(double_complex(0.0, phase));

        std::vector<double_complex> phase_gk(num_gkvec_loc_);
        for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
            int igk = gkvec_.gvec_offset(comm_.rank()) + igk_loc;
            auto G = gkvec_.gvec(igk);
            phase_gk[igk_loc] = std::conj(ctx__.gvec_phase_factor(G, ia) * phase_k);
        }

        for (int xi = 0; xi < unit_cell_.atom(ia).mt_lo_basis_size(); xi++) {
            for (int igk_loc = 0; igk_loc < num_gkvec_loc_; igk_loc++) {
                beta_gk_a_(igk_loc, unit_cell_.atom(ia).offset_lo() + xi) =
                    beta_gk_t_(igk_loc, unit_cell_.atom(ia).type().offset_lo() + xi) * phase_gk[igk_loc];
            }
        }
    }
}

void Beta_projectors::generate_beta_gk_t(Simulation_context const& ctx__)
{
    PROFILE("sirius::Beta_projectors::generate_beta_gk_t");

    if (!num_beta_t_) {
        return;
    }

    /* allocate array */
    beta_gk_t_ = matrix<double_complex>(gkvec_.gvec_count(comm_.rank()), num_beta_t_);
    
    auto& qgrid = ctx__.qgrid_gkmax();
    auto& beta_radial_integrals = ctx__.beta_radial_integrals();
 
    /* compute <G+k|beta> */
    #pragma omp parallel for
    for (int igkloc = 0; igkloc < gkvec_.gvec_count(comm_.rank()); igkloc++) {
        int igk   = gkvec_.gvec_offset(comm_.rank()) + igkloc;
        double gk = gkvec_.gvec_len(igk);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(gkvec_.gkvec_cart(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> gkvec_rlm(Utils::lmmax(lmax_beta_));
        SHT::spherical_harmonics(lmax_beta_, vs[1], vs[2], &gkvec_rlm[0]);
        /* position in the linear grid of |G| values */
        int iq = static_cast<int>((qgrid.num_points() - 1) * gk / unit_cell_.parameters().gk_cutoff());
        /* delta */
        double dq = gk - qgrid[iq];
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            for (int xi = 0; xi < atom_type.mt_basis_size(); xi++) {
                int l     = atom_type.indexb(xi).l;
                int lm    = atom_type.indexb(xi).lm;
                int idxrf = atom_type.indexb(xi).idxrf;

                double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
                beta_gk_t_(igkloc, atom_type.offset_lo() + xi) = z * gkvec_rlm[lm] * beta_radial_integrals(idxrf, iat)(iq, dq);
            }
        }
    }

    if (unit_cell_.parameters().control().print_checksum_) {
        auto c1 = beta_gk_t_.checksum();
        comm_.allreduce(&c1, 1);
        DUMP("checksum(beta_gk_t) : %18.10f %18.10f", c1.real(), c1.imag())
    }
}

void Beta_projectors::split_in_chunks()
{
    /* split beta-projectors into chunks */
    int num_atoms_in_chunk = (comm_.size() == 1) ? unit_cell_.num_atoms() : std::min(unit_cell_.num_atoms(), 256);
    int num_beta_chunks = unit_cell_.num_atoms() / num_atoms_in_chunk + std::min(1, unit_cell_.num_atoms() % num_atoms_in_chunk);
    splindex<block> spl_beta_chunks(unit_cell_.num_atoms(), num_beta_chunks, 0);

    beta_chunks_ = mdarray<beta_chunk_t, 1>(num_beta_chunks);

    int offset_in_beta_gk = 0;

    for (int ib = 0; ib < num_beta_chunks; ib++) {
        /* number of atoms in chunk */
        int na = spl_beta_chunks.local_size(ib);
        beta_chunks_(ib).num_atoms_ = na;
        beta_chunks_(ib).desc_      = mdarray<int, 2>(4, na);
        beta_chunks_(ib).atom_pos_  = mdarray<double, 2>(3, na);

        int num_beta{0};

        for (int i = 0; i < na; i++) {
            /* global index of atom by local index and chunk */
            int ia = spl_beta_chunks.global_index(i, ib);
            auto pos = unit_cell_.atom(ia).position();
            auto& type = unit_cell_.atom(ia).type();
            /* atom fractional coordinates */
            for (int x: {0, 1, 2}) {
                beta_chunks_(ib).atom_pos_(x, i) = pos[x];
            }
            /* number of beta functions for atom */
            beta_chunks_(ib).desc_(beta_desc_idx::nbf, i) = type.mt_basis_size();
            /* offset in beta_gk*/
            beta_chunks_(ib).desc_(beta_desc_idx::offset, i) = num_beta;
            /* offset in beta_gk_t */
            beta_chunks_(ib).desc_(beta_desc_idx::offset_t, i) = type.offset_lo();
            /* global index of atom */
            beta_chunks_(ib).desc_(beta_desc_idx::ia, i) = ia;

            num_beta += type.mt_basis_size();
        }
        /* number of beta-projectors in this chunk */
        beta_chunks_(ib).num_beta_ = num_beta;
        beta_chunks_(ib).offset_ = offset_in_beta_gk;
        offset_in_beta_gk += num_beta;

        #ifdef __GPU
        if (pu_ == GPU) {
            beta_chunks_[ib].desc_.allocate(memory_t::device);
            beta_chunks_[ib].desc_.copy_to_device();

            beta_chunks_[ib].atom_pos_.allocate(memory_t::device);
            beta_chunks_[ib].atom_pos_.copy_to_device();
        }
        #endif
    }

    max_num_beta_ = 0;
    for (int ib = 0; ib < num_beta_chunks; ib++) {
        max_num_beta_ = std::max(max_num_beta_, beta_chunks_(ib).num_beta_);
    }

    num_beta_t_ = 0;
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        num_beta_t_ += unit_cell_.atom_type(iat).mt_lo_basis_size();
    }
}

void Beta_projectors::generate(int chunk__)
{
    PROFILE("sirius::Beta_projectors::generate");

    if (pu_ == CPU) {
        beta_gk_ = mdarray<double_complex, 2>(&beta_gk_a_(0, beta_chunk(chunk__).offset_),
                                              num_gkvec_loc_, beta_chunk(chunk__).num_beta_);
    }
    #ifdef __GPU
    if (pu_ == GPU) {
        beta_gk_ = mdarray<double_complex, 2>(&beta_gk_a_(0, beta_chunk(chunk__).offset_), beta_gk_gpu_.at<GPU>(),
                                              num_gkvec_loc_, beta_chunk(chunk__).num_beta_);

        auto& desc = beta_chunk(chunk__).desc_;
        create_beta_gk_gpu(beta_chunk(chunk__).num_atoms_,
                           num_gkvec_loc_,
                           desc.at<GPU>(),
                           beta_gk_t_.at<GPU>(),
                           gkvec_coord_.at<GPU>(),
                           beta_chunk(chunk__).atom_pos_.at<GPU>(),
                           beta_gk_.at<GPU>());
    }
    #endif
}

template<>
void Beta_projectors::inner<double_complex>(int chunk__, wave_functions& phi__,
                                            int idx0__, int n__, mdarray<double_complex, 2> &beta_gk, mdarray<double, 1> &beta_phi)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());

    int nbeta = beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi.size()) {
        beta_phi = mdarray<double, 1>(2 * nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) {
            beta_phi.allocate(memory_t::device);
        }
        #endif
    }

    switch (pu_) {
        case CPU: {
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_,
                              beta_gk.at<CPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<CPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi.at<CPU>(), nbeta);
            break;
        }
        case GPU: {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, nbeta, n__, num_gkvec_loc_, beta_gk.at<GPU>(), num_gkvec_loc_,
                              phi__.pw_coeffs().prime().at<GPU>(0, idx0__), phi__.pw_coeffs().prime().ld(),
                              (double_complex*)beta_phi.at<GPU>(), nbeta);
            beta_phi.copy_to_host(2 * nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi.at<CPU>(), 2 * nbeta * n__);

    #ifdef __GPU
    if (pu_ == GPU) {
        beta_phi.copy_to_device(2 * nbeta * n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = mdarray<double, 1>(beta_phi.at<CPU>(), 2 * nbeta * n__).checksum();
        DUMP("checksum(beta_phi) : %18.10f", cs);
    }
    #endif
}

template<>
void Beta_projectors::inner<double>(int chunk__,  wave_functions& phi__,
                                    int idx0__, int n__, mdarray<double_complex, 2> &beta_gk, mdarray<double, 1> &beta_phi)
{
    PROFILE("sirius::Beta_projectors::inner");

    assert(num_gkvec_loc_ == phi__.pw_coeffs().num_rows_loc());

    int nbeta = beta_chunk(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > beta_phi.size())
    {
        beta_phi = mdarray<double, 1>(nbeta * n__);
        #ifdef __GPU
        if (pu_ == GPU) {
            beta_phi.allocate(memory_t::device);
        }
        #endif
    }

    double a = 2;
    double a1 = -1;
    double b = 0;

    switch (pu_) {
        case CPU: {
            /* compute <beta|phi> */
            linalg<CPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                              a,
                              (double*)beta_gk.at<CPU>(), 2 * num_gkvec_loc_,
                              (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                              b,
                              beta_phi.at<CPU>(), nbeta);

            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<CPU>::ger(nbeta, n__, a1, (double*)&beta_gk(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<CPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                &beta_phi[0], nbeta);
            }
            break;
        }
        case GPU: {
            #ifdef __GPU
            linalg<GPU>::gemm(2, 0, nbeta, n__, 2 * num_gkvec_loc_,
                              &a,
                              (double*)beta_gk.at<GPU>(), 2 * num_gkvec_loc_,
                              (double*)phi__.pw_coeffs().prime().at<GPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                              &b,
                              beta_phi.at<GPU>(), nbeta);
            if (comm_.rank() == 0) {
                /* subtract one extra G=0 contribution */
                linalg<GPU>::ger(nbeta, n__, &a1, (double*)beta_gk.at<GPU>(0, 0), 2 * num_gkvec_loc_,
                                (double*)phi__.pw_coeffs().prime().at<GPU>(0, idx0__), 2 * phi__.pw_coeffs().prime().ld(),
                                beta_phi.at<GPU>(), nbeta);
            }
            beta_phi.copy_to_host(nbeta * n__);
            #else
            TERMINATE_NO_GPU
            #endif
            break;
        }
    }

    comm_.allreduce(beta_phi.at<CPU>(), nbeta * n__);

    #ifdef __GPU
    if (pu_ == GPU) {
        beta_phi.copy_to_device(nbeta * n__);
    }
    #endif

    #ifdef __PRINT_OBJECT_CHECKSUM
    {
        auto cs = mdarray<double, 1>(beta_phi.at<CPU>(), nbeta * n__).checksum();
        DUMP("checksum(beta_phi) : %18.10f", cs);
    }
    #endif
}

};
