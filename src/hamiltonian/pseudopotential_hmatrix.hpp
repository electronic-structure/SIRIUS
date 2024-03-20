/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __PSEUDOPOTENTIAL_HMATRIX_HPP__
#define __PSEUDOPOTENTIAL_HMATRIX_HPP__

namespace sirius {
template <typename T>
inline dmatrix<T>
pseudopotential_hmatrix(K_point& kp__, int ispn__, Hamiltonian& H__)
{
    PROFILE("sirius::pseudopotential_hmatrix");

    //
    H__.local_op().prepare(H__.potential());
    if (!H__.ctx().gamma_point()) {
        H__.prepare<double_complex>();
    } else {
        H__.prepare<double>();
    }
    H__.local_op().prepare(kp__.gkvec_partition());
    H__.ctx().fft_coarse().prepare(kp__.gkvec_partition());
    kp__.beta_projectors().prepare();

    auto& ctx    = H__.ctx();
    const int bs = ctx.cyclic_block_size();
    dmatrix<T> hmlt(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);
    // dmatrix<T> ovlp(kp__.num_gkvec(), kp__.num_gkvec(), ctx.blacs_grid(), bs, bs);

    hmlt.zero();
    // ovlp.zero();

    auto gen_solver = ctx.gen_evp_solver<T>();

    for (int ig = 0; ig < kp__.num_gkvec(); ig++) {
        hmlt.set(ig, ig, 0.5 * std::pow(kp__.gkvec().gkvec_cart(gvec_index_t::global(ig)).length(), 2));
        // ovlp.set(ig, ig, 1);
    }

    auto veff = H__.potential().effective_potential().gather_f_pw();
    std::vector<double_complex> beff;
    if (ctx.num_mag_dims() == 1) {
        beff = H__.potential().effective_magnetic_field(0).gather_f_pw();
        for (int ig = 0; ig < ctx.gvec().num_gvec(); ig++) {
            auto z1  = veff[ig];
            auto z2  = beff[ig];
            veff[ig] = z1 + z2;
            beff[ig] = z1 - z2;
        }
    }

    #pragma omp parallel for schedule(static)
    for (int igk_col = 0; igk_col < kp__.num_gkvec_col(); igk_col++) {
        int ig_col    = kp__.igk_col(igk_col);
        auto gvec_col = kp__.gkvec().gvec(ig_col);
        for (int igk_row = 0; igk_row < kp__.num_gkvec_row(); igk_row++) {
            int ig_row    = kp__.igk_row(igk_row);
            auto gvec_row = kp__.gkvec().gvec(ig_row);
            auto ig12     = ctx.gvec().index_g12_safe(gvec_row, gvec_col);

            if (ispn__ == 0) {
                if (ig12.second) {
                    hmlt(igk_row, igk_col) += std::conj(veff[ig12.first]);
                } else {
                    hmlt(igk_row, igk_col) += veff[ig12.first];
                }
            } else {
                if (ig12.second) {
                    hmlt(igk_row, igk_col) += std::conj(beff[ig12.first]);
                } else {
                    hmlt(igk_row, igk_col) += beff[ig12.first];
                }
            }
        }
    }

    mdarray<double_complex, 2> dop(ctx.unit_cell().max_mt_basis_size(), ctx.unit_cell().max_mt_basis_size());
    mdarray<double_complex, 2> qop(ctx.unit_cell().max_mt_basis_size(), ctx.unit_cell().max_mt_basis_size());

    mdarray<double_complex, 2> btmp(kp__.num_gkvec_row(), ctx.unit_cell().max_mt_basis_size());

    kp__.beta_projectors_row().prepare();
    kp__.beta_projectors_col().prepare();
    for (int ichunk = 0; ichunk < kp__.beta_projectors_row().num_chunks(); ichunk++) {
        /* generate beta-projectors for a block of atoms */
        kp__.beta_projectors_row().generate(ichunk);
        kp__.beta_projectors_col().generate(ichunk);

        auto& beta_row = kp__.beta_projectors_row().pw_coeffs_a();
        auto& beta_col = kp__.beta_projectors_col().pw_coeffs_a();

        for (int i = 0; i < kp__.beta_projectors_row().chunk(ichunk).num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = kp__.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::nbf, i);
            int offs = kp__.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::offset, i);
            int ia   = kp__.beta_projectors_row().chunk(ichunk).desc_(beta_desc_idx::ia, i);

            const auto& augment_op = ctx.augmentation_op(ctx.unit_cell().atom(ia).type_id());

            for (int xi1 = 0; xi1 < nbf; xi1++) {
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    if (ctx.num_mag_dims() == 1) {
                        if (ispn__ == 0) {
                            dop(xi1, xi2) = ctx.unit_cell().atom(ia).d_mtrx(xi1, xi2, 0) +
                                            ctx.unit_cell().atom(ia).d_mtrx(xi1, xi2, 1);
                        } else {
                            dop(xi1, xi2) = ctx.unit_cell().atom(ia).d_mtrx(xi1, xi2, 0) -
                                            ctx.unit_cell().atom(ia).d_mtrx(xi1, xi2, 1);
                        }
                    } else {
                        dop(xi1, xi2) = ctx.unit_cell().atom(ia).d_mtrx(xi1, xi2, 0);
                    }
                    if (augment_op.atom_type().augment()) {
                        qop(xi1, xi2) = augment_op.q_mtrx(xi1, xi2);
                    }
                }
            }
            linalg<device_t::CPU>::gemm(0, 0, kp__.num_gkvec_row(), nbf, nbf, &beta_row(0, offs), beta_row.ld(),
                                        &dop(0, 0), dop.ld(), &btmp(0, 0), btmp.ld());
            linalg<device_t::CPU>::gemm(0, 2, kp__.num_gkvec_row(), kp__.num_gkvec_col(), nbf,
                                        linalg_const<double_complex>::one(), &btmp(0, 0), btmp.ld(), &beta_col(0, offs),
                                        beta_col.ld(), linalg_const<double_complex>::one(), &hmlt(0, 0), hmlt.ld());
            // if(augment_op.atom_type().augment()) {
            //     linalg<device_t::CPU>::gemm(0, 0, kp__.num_gkvec_row(), nbf, nbf,
            //                       &beta_row(0, offs), beta_row.ld(),
            //                       &qop(0, 0), qop.ld(),
            //                       &btmp(0, 0), btmp.ld());
            // }
            // linalg<device_t::CPU>::gemm(0, 2, kp__.num_gkvec_row(), kp__.num_gkvec_col(), nbf,
            //                   linalg_const<double_complex>::one(),
            //                   &btmp(0, 0), btmp.ld(),
            //                   &beta_col(0, offs), beta_col.ld(),
            //                   linalg_const<double_complex>::one(),
            //                   &ovlp(0, 0), ovlp.ld());
        }
    }
    kp__.beta_projectors_row().dismiss();
    kp__.beta_projectors_col().dismiss();

    //
    kp__.beta_projectors().dismiss();
    H__.local_op().dismiss();
    H__.ctx().fft_coarse().dismiss();

    return hmlt;
}
} // namespace sirius

#endif /* __PSEUDOPOTENTIAL_HMATRIX_HPP__ */
