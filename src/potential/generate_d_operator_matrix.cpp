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

/** \file generate_d_operator_matrix.hpp
 *
 *  \brief Contains implementation of sirius::Potential::generate_D_operator_matrix method.
 */

#include "potential.hpp"

namespace sirius {

#ifdef __GPU
extern "C" void mul_veff_with_phase_factors_gpu(int num_atoms__,
                                                int num_gvec_loc__,
                                                double_complex const* veff__,
                                                int const* gvx__,
                                                int const* gvy__,
                                                int const* gvz__,
                                                double const* atom_pos__,
                                                double* veff_a__,
                                                int ld__,
                                                int stream_id__);
#endif

void Potential::generate_D_operator_matrix()
{
    PROFILE("sirius::Potential::generate_D_operator_matrix");

    auto spl_ngv_loc = ctx_.split_gvec_local();

    if (ctx_.augmentation_op(0)) {
        ctx_.augmentation_op(0)->prepare(stream_id(0), &ctx_.mem_pool(memory_t::device));
    }
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        int nbf         = atom_type.mt_basis_size();

        /* start copy of Q(G) for the next atom type */
        if (ctx_.processing_unit() == device_t::GPU) {
            acc::sync_stream(stream_id(0));
            if (iat + 1 != unit_cell_.num_atom_types() && ctx_.augmentation_op(iat + 1)) {
                ctx_.augmentation_op(iat + 1)->prepare(stream_id(0), &ctx_.mem_pool(memory_t::device));
            }
        }

        /* trivial case */
        if (!atom_type.augment() || atom_type.num_atoms() == 0) {
            for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    int ia     = atom_type.atom_id(i);
                    auto& atom = unit_cell_.atom(ia);

                    for (int xi2 = 0; xi2 < nbf; xi2++) {
                        for (int xi1 = 0; xi1 < nbf; xi1++) {
                            atom.d_mtrx(xi1, xi2, iv) = 0;
                        }
                    }
                }
            }
            continue;
        }
        matrix<double> d_tmp(nbf * (nbf + 1) / 2, atom_type.num_atoms(), ctx_.mem_pool(memory_t::host));
        if (ctx_.processing_unit() == device_t::GPU) {
            d_tmp.allocate(ctx_.mem_pool(memory_t::device));
        }

        ctx_.print_memory_usage(__FILE__, __LINE__);

        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
            matrix<double> veff_a(2 * spl_ngv_loc.local_size(), atom_type.num_atoms(), ctx_.mem_pool(memory_t::host));

            auto la = linalg_t::blas;
            auto mem = memory_t::host;

            d_tmp.zero();
            if (ctx_.processing_unit() == device_t::GPU) {
                la = linalg_t::gpublas;
                mem = memory_t::device;
                d_tmp.zero(memory_t::device);
                veff_a.allocate(ctx_.mem_pool(memory_t::device));
            }

            /* split a large loop over G-vectors into blocks */
            for (int ib = 0; ib < spl_ngv_loc.num_ranks(); ib++) {
                int g_begin = spl_ngv_loc.global_index(0, ib);
                int g_end = g_begin + spl_ngv_loc.local_size(ib);

                switch (ctx_.processing_unit()) {
                    case device_t::CPU: {
                        #pragma omp parallel for schedule(static)
                        for (int i = 0; i < atom_type.num_atoms(); i++) {
                            int ia = atom_type.atom_id(i);

                            for (int igloc = g_begin; igloc < g_end; igloc++) {
                                int ig = ctx_.gvec().offset() + igloc;
                                /* V(G) * exp(i * G * r_{alpha}) */
                                auto z = component(iv).f_pw_local(igloc) * ctx_.gvec_phase_factor(ig, ia);
                                veff_a(2 * (igloc - g_begin),     i) = z.real();
                                veff_a(2 * (igloc - g_begin) + 1, i) = z.imag();
                            }
                        }
                        break;
                    }
                    case device_t::GPU: {
                        /* wait for stream#1 to finish previous zgemm */
                        acc::sync_stream(stream_id(1));
                        /* copy plane wave coefficients of effective potential to GPU */
                        mdarray<double_complex, 1> veff(&component(iv).f_pw_local(g_begin), spl_ngv_loc.local_size(ib));
                        veff.allocate(ctx_.mem_pool(memory_t::device)).copy_to(memory_t::device);
#if defined(__GPU)
                        mul_veff_with_phase_factors_gpu(atom_type.num_atoms(), spl_ngv_loc.local_size(ib),
                                                        veff.at(memory_t::device),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 0),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 1),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 2),
                                                        ctx_.unit_cell().atom_coord(iat).at(memory_t::device),
                                                        veff_a.at(memory_t::device), spl_ngv_loc.local_size(), 1);
#endif
                        break;
                    }
                }
                if (ctx_.control().print_checksum_) {
                    if (ctx_.processing_unit() == device_t::GPU) {
                        veff_a.copy_to(memory_t::host);
                    }
                    auto cs = veff_a.checksum();
                    std::stringstream s;
                    s << "Gvec_block_" << ib << "_veff_a";
                    utils::print_checksum(s.str(), cs);
                }
                linalg(la).gemm('N', 'N', nbf * (nbf + 1) / 2, atom_type.num_atoms(), 2 * spl_ngv_loc.local_size(ib),
                                  &linalg_const<double>::one(),
                                  ctx_.augmentation_op(iat)->q_pw().at(mem, 0, 2 * g_begin),
                                  ctx_.augmentation_op(iat)->q_pw().ld(),
                                  veff_a.at(mem), veff_a.ld(),
                                  &linalg_const<double>::one(),
                                  d_tmp.at(mem), d_tmp.ld(),
                                  stream_id(1));
            } // ib (blocks of G-vectors)

            if (ctx_.processing_unit() == device_t::GPU) {
                d_tmp.copy_to(memory_t::host);
            }

            if (ctx_.gvec().reduced()) {
                if (comm_.rank() == 0) {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nbf * (nbf + 1) / 2; j++) {
                            d_tmp(j, i) = 2 * d_tmp(j, i) - component(iv).f_pw_local(0).real() *
                                ctx_.augmentation_op(iat)->q_pw(j, 0);
                        }
                    }
                } else {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nbf * (nbf + 1) / 2; j++) {
                            d_tmp(j, i) *= 2;
                        }
                    }
                }
            }

            /* sum from all ranks */
            comm_.allreduce(d_tmp.at(memory_t::host), static_cast<int>(d_tmp.size()));

            if (ctx_.control().print_checksum_ && ctx_.comm().rank() == 0) {
                for (int i = 0; i < atom_type.num_atoms(); i++) {
                    std::stringstream s;
                    s << "D_mtrx_val(atom_t" << iat << "_i" << i << "_c" << iv << ")";
                    auto cs = mdarray<double, 1>(&d_tmp(0, i), nbf * (nbf + 1) / 2).checksum();
                    utils::print_checksum(s.str(), cs);
                }
            }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type.num_atoms(); i++) {
                int ia     = atom_type.atom_id(i);
                auto& atom = unit_cell_.atom(ia);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* D-matix is symmetric */
                        atom.d_mtrx(xi1, xi2, iv) = atom.d_mtrx(xi2, xi1, iv) = d_tmp(idx12, i) * unit_cell_.omega();
                    }
                }
            }
        }
        ctx_.augmentation_op(iat)->dismiss();

    }
}

} // namespace sirius
