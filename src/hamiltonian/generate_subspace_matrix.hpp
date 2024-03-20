/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file generate_subspace_matrix.hpp
 *
 *  \brief Generate subspace-matrix in the auxiliary basis |phi>
 */

#ifndef __GENERATE_SUBSPACE_MATRIX_HPP__
#define __GENERATE_SUBSPACE_MATRIX_HPP__

#include "context/simulation_context.hpp"
#include "core/wf/wave_functions.hpp"

namespace sirius {

/// Generate subspace matrix for the iterative diagonalization.
/** Compute \f$ O_{ii'} = \langle \phi_i | \hat O | \phi_{i'} \rangle \f$ operator matrix
 *  for the subspace spanned by the wave-functions \f$ \phi_i \f$. The matrix is always returned
 *  in the CPU pointer because most of the standard math libraries start from the CPU.
 *
 *  \tparam T Precision type of the wave-functions
 *  \tparam F Type of the output subspace matrix.
 *  \param [in] ctx Simulation context
 *  \param [in] N Number of existing (old) states phi.
 *  \param [in] n Number of new states phi.
 *  \param [in] num_locked Number of locked states phi. Locked states are excluded from the subspace basis.
 *  \param [in] phi Subspace basis functions phi.
 *  \param [in] op_phi Operator (H, S or overlap) applied to phi.
 *  \param [out] mtrx New subspace matrix.
 *  \param [inout] mtrx_old Pointer to old subpsace matrix. It is used to store and reuse the subspace matrix
 *                          of the previous step.
 *  */
template <typename T, typename F>
void
generate_subspace_matrix(Simulation_context& ctx__, int N__, int n__, int num_locked__, wf::Wave_functions<T>& phi__,
                         wf::Wave_functions<T>& op_phi__, la::dmatrix<F>& mtrx__, la::dmatrix<F>* mtrx_old__ = nullptr)
{
    PROFILE("sirius::generate_subspace_matrix");

    RTE_ASSERT(n__ != 0);
    if (mtrx_old__ && mtrx_old__->size()) {
        RTE_ASSERT(&mtrx__.blacs_grid() == &mtrx_old__->blacs_grid());
    }

    /* copy old N - num_locked x N - num_locked distributed matrix */
    if (N__ > 0) {
        splindex_block_cyclic<> spl_row(N__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_row()),
                                        block_id(mtrx__.blacs_grid().rank_row()), mtrx__.bs_row());
        splindex_block_cyclic<> spl_col(N__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_col()),
                                        block_id(mtrx__.blacs_grid().rank_col()), mtrx__.bs_col());

        if (mtrx_old__) {
            if (spl_row.local_size()) {
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < spl_col.local_size(); i++) {
                    std::copy(&(*mtrx_old__)(0, i), &(*mtrx_old__)(0, i) + spl_row.local_size(), &mtrx__(0, i));
                }
            }
        }

        if (env::print_checksum()) {
            auto cs = mtrx__.checksum(N__ - num_locked__, N__ - num_locked__);
            if (ctx__.comm_band().rank() == 0) {
                print_checksum("subspace_mtrx_old", cs, RTE_OUT(std::cout));
            }
        }
    }

    /*  [--- num_locked -- | ------ N - num_locked ---- | ---- n ----] */
    /*  [ ------------------- N ------------------------| ---- n ----] */

    auto mem = ctx__.processing_unit() == device_t::CPU ? memory_t::host : memory_t::device;
    /* <{phi,phi_new}|Op|phi_new> */
    inner(ctx__.spla_context(), mem, ctx__.num_mag_dims() == 3 ? wf::spin_range(0, 2) : wf::spin_range(0), phi__,
          wf::band_range(num_locked__, N__ + n__), op_phi__, wf::band_range(N__, N__ + n__), mtrx__, 0,
          N__ - num_locked__);

    /* restore lower part */
    if (N__ > 0) {
        if (mtrx__.blacs_grid().comm().size() == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N__ - num_locked__; i++) {
                for (int j = N__ - num_locked__; j < N__ + n__ - num_locked__; j++) {
                    mtrx__(j, i) = conj(mtrx__(i, j));
                }
            }
        } else {
            la::wrap(la::lib_t::scalapack)
                    .tranc(n__, N__ - num_locked__, mtrx__, 0, N__ - num_locked__, mtrx__, N__ - num_locked__, 0);
        }
    }

    if (env::print_checksum()) {
        splindex_block_cyclic<> spl_row(N__ + n__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_row()),
                                        block_id(mtrx__.blacs_grid().rank_row()), mtrx__.bs_row());
        splindex_block_cyclic<> spl_col(N__ + n__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_col()),
                                        block_id(mtrx__.blacs_grid().rank_col()), mtrx__.bs_col());
        auto cs = mtrx__.checksum(N__ + n__ - num_locked__, N__ + n__ - num_locked__);
        if (ctx__.comm_band().rank() == 0) {
            print_checksum("subspace_mtrx", cs, RTE_OUT(std::cout));
        }
    }

    /* remove any numerical noise */
    mtrx__.make_real_diag(N__ + n__ - num_locked__);

    /* save new matrix */
    if (mtrx_old__) {
        splindex_block_cyclic<> spl_row(N__ + n__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_row()),
                                        block_id(mtrx__.blacs_grid().rank_row()), mtrx__.bs_row());
        splindex_block_cyclic<> spl_col(N__ + n__ - num_locked__, n_blocks(mtrx__.blacs_grid().num_ranks_col()),
                                        block_id(mtrx__.blacs_grid().rank_col()), mtrx__.bs_col());

        if (spl_row.local_size()) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < spl_col.local_size(); i++) {
                std::copy(&mtrx__(0, i), &mtrx__(0, i) + spl_row.local_size(), &(*mtrx_old__)(0, i));
            }
        }
    }
}

} // namespace sirius

#endif
