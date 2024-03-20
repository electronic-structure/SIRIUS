/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_d_operator_matrix.hpp
 *
 *  \brief Contains implementation of sirius::Potential::generate_D_operator_matrix method.
 */

#include "potential.hpp"

namespace sirius {

#ifdef SIRIUS_GPU
extern "C" void
mul_veff_with_phase_factors_gpu(int num_atoms__, int num_gvec_loc__, std::complex<double> const* veff__,
                                int const* gvx__, int const* gvy__, int const* gvz__, double const* atom_pos__,
                                double* veff_a__, int ld__, int stream_id__);
#endif

void
Potential::generate_D_operator_matrix()
{
    PROFILE("sirius::Potential::generate_D_operator_matrix");

    /* local number of G-vectors */
    int gvec_count   = ctx_.gvec().count();
    auto spl_ngv_loc = split_in_blocks(gvec_count, ctx_.cfg().control().gvec_chunk_size());

    auto& mph = get_memory_pool(memory_t::host);
    memory_pool* mpd{nullptr};

    int n_mag_comp{1};

    mdarray<std::complex<double>, 2> veff;
    switch (ctx_.processing_unit()) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
            mpd        = &get_memory_pool(memory_t::device);
            n_mag_comp = ctx_.num_mag_dims() + 1;
            veff       = mdarray<std::complex<double>, 2>({gvec_count, n_mag_comp}, mph);
            for (int j = 0; j < ctx_.num_mag_dims() + 1; j++) {
                std::copy(&component(j).rg().f_pw_local(0), &component(j).rg().f_pw_local(0) + gvec_count, &veff(0, j));
            }
            veff.allocate(*mpd).copy_to(memory_t::device);
            break;
        }
    }

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        /* number of beta-projector functions */
        int nbf = atom_type.mt_basis_size();
        /* number of Q_{xi,xi'} components for each G */
        int nqlm = nbf * (nbf + 1) / 2;

        /* trivial case */
        /* in absence of augmentation charge D-matrix is zero */
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

        mdarray<double, 3> d_tmp({nqlm, atom_type.num_atoms(), ctx_.num_mag_dims() + 1}, mph);
        mdarray<double, 3> veff_a({spl_ngv_loc[0] * 2, atom_type.num_atoms(), n_mag_comp}, mph);
        mdarray<double, 2> qpw;

        switch (ctx_.processing_unit()) {
            case device_t::CPU: {
                d_tmp.zero();
                break;
            }
            case device_t::GPU: {
                d_tmp.allocate(*mpd).zero(memory_t::device);
                veff_a.allocate(*mpd);
                qpw = mdarray<double, 2>({nqlm, 2 * spl_ngv_loc[0]}, *mpd, mdarray_label("qpw"));
                break;
            }
        }

        print_memory_usage(ctx_.out(), FILE_LINE);

        int g_begin{0};
        /* loop over blocks of G-vectors */
        for (auto ng : spl_ngv_loc) {
            /* work on the block of the local G-vectors */
            switch (ctx_.processing_unit()) {
                case device_t::CPU: {
                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                        /* multiply Veff(G) with the phase factors */
                        #pragma omp parallel for
                        for (int i = 0; i < atom_type.num_atoms(); i++) {
                            int ia = atom_type.atom_id(i);

                            for (int g = 0; g < ng; g++) {
                                int ig = ctx_.gvec().offset() + g_begin + g;
                                /* V(G) * exp(i * G * r_{alpha}) */
                                auto z = component(iv).rg().f_pw_local(g_begin + g) * ctx_.gvec_phase_factor(ig, ia);
                                veff_a(2 * g, i, 0)     = z.real();
                                veff_a(2 * g + 1, i, 0) = z.imag();
                            }
                        }
                        la::wrap(la::lib_t::blas)
                                .gemm('N', 'N', nqlm, atom_type.num_atoms(), 2 * ng, &la::constant<double>::one(),
                                      ctx_.augmentation_op(iat).q_pw().at(memory_t::host, 0, 2 * g_begin),
                                      ctx_.augmentation_op(iat).q_pw().ld(), veff_a.at(memory_t::host), veff_a.ld(),
                                      &la::constant<double>::one(), d_tmp.at(memory_t::host, 0, 0, iv), d_tmp.ld());
                    } // iv
                    break;
                }
                case device_t::GPU: {
                    acc::copyin(qpw.at(memory_t::device),
                                ctx_.augmentation_op(iat).q_pw().at(memory_t::host, 0, 2 * g_begin), 2 * ng * nqlm);
                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
#if defined(SIRIUS_GPU)
                        mul_veff_with_phase_factors_gpu(atom_type.num_atoms(), ng, veff.at(memory_t::device, 0, iv),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 0),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 1),
                                                        ctx_.gvec_coord().at(memory_t::device, g_begin, 2),
                                                        ctx_.unit_cell().atom_coord(iat).at(memory_t::device),
                                                        veff_a.at(memory_t::device, 0, 0, iv), ng, 1 + iv);

                        la::wrap(la::lib_t::gpublas)
                                .gemm('N', 'N', nqlm, atom_type.num_atoms(), 2 * ng, &la::constant<double>::one(),
                                      qpw.at(memory_t::device), qpw.ld(), veff_a.at(memory_t::device, 0, 0, iv),
                                      veff_a.ld(), &la::constant<double>::one(), d_tmp.at(memory_t::device, 0, 0, iv),
                                      d_tmp.ld(), acc::stream_id(1 + iv));
#endif
                    } // iv
                    for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
                        acc::sync_stream(acc::stream_id(1 + iv));
                    }
                    break;
                }
            }

            g_begin += ng;
        }

        if (ctx_.processing_unit() == device_t::GPU) {
            d_tmp.copy_to(memory_t::host);
        }

        // if (ctx_.cfg().control().print_checksum()) {
        //     if (ctx_.processing_unit() == device_t::GPU) {
        //         veff_a.copy_to(memory_t::host);
        //     }
        //     auto cs = veff_a.checksum();
        //     std::stringstream s;
        //     s << "Gvec_block_" << ib << "_veff_a";
        //     utils::print_checksum(s.str(), cs, ctx_.out());
        // }

        for (int iv = 0; iv < ctx_.num_mag_dims() + 1; iv++) {
            if (ctx_.gvec().reduced()) {
                if (comm_.rank() == 0) {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nqlm; j++) {
                            d_tmp(j, i, iv) = 2 * d_tmp(j, i, iv) - component(iv).rg().f_pw_local(0).real() *
                                                                            ctx_.augmentation_op(iat).q_pw(j, 0);
                        }
                    }
                } else {
                    for (int i = 0; i < atom_type.num_atoms(); i++) {
                        for (int j = 0; j < nqlm; j++) {
                            d_tmp(j, i, iv) *= 2;
                        }
                    }
                }
            }

            /* sum from all ranks */
            comm_.allreduce(d_tmp.at(memory_t::host, 0, 0, iv), nqlm * atom_type.num_atoms());

            // if (ctx_.cfg().control().print_checksum() && ctx_.comm().rank() == 0) {
            //     for (int i = 0; i < atom_type.num_atoms(); i++) {
            //         std::stringstream s;
            //         s << "D_mtrx_val(atom_t" << iat << "_i" << i << "_c" << iv << ")";
            //         auto cs = mdarray<double, 1>(&d_tmp(0, i), nbf * (nbf + 1) / 2).checksum();
            //         utils::print_checksum(s.str(), cs, ctx_.out());
            //     }
            // }

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < atom_type.num_atoms(); i++) {
                int ia     = atom_type.atom_id(i);
                auto& atom = unit_cell_.atom(ia);

                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 <= xi2; xi1++) {
                        int idx12 = xi2 * (xi2 + 1) / 2 + xi1;
                        /* D-matix is symmetric */
                        atom.d_mtrx(xi1, xi2, iv) = atom.d_mtrx(xi2, xi1, iv) =
                                d_tmp(idx12, i, iv) * unit_cell_.omega();
                    }
                }
            }
        } // iv
    }     // iat
}

} // namespace sirius
