// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file set_subspace_mtrx.hpp
 *
 *  \brief Set the subspace Hamiltonian and overlap matrices.
 */

template <typename T>
inline void Band::set_subspace_mtrx(int N__,
                                    int n__,
                                    Wave_functions& phi__,
                                    Wave_functions& op_phi__,
                                    dmatrix<T>& mtrx__,
                                    dmatrix<T>* mtrx_old__) const
{
    PROFILE("sirius::Band::set_subspace_mtrx");

    assert(n__ != 0);
    if (mtrx_old__ && mtrx_old__->size()) {
        assert(&mtrx__.blacs_grid() == &mtrx_old__->blacs_grid());
    }

    /* copy old N x N distributed matrix */
    if (N__ > 0) {
        splindex<block_cyclic> spl_row(N__, mtrx__.blacs_grid().num_ranks_row(), mtrx__.blacs_grid().rank_row(),
                                       mtrx__.bs_row());
        splindex<block_cyclic> spl_col(N__, mtrx__.blacs_grid().num_ranks_col(), mtrx__.blacs_grid().rank_col(),
                                       mtrx__.bs_col());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < spl_col.local_size(); i++) {
            std::copy(&(*mtrx_old__)(0, i), &(*mtrx_old__)(0, i) + spl_row.local_size(), &mtrx__(0, i));
        }

        if (ctx_.control().print_checksum_) {
            double_complex cs(0, 0);
            for (int i = 0; i < spl_col.local_size(); i++) {
                for (int j = 0; j < spl_row.local_size(); j++) {
                    cs += mtrx__(j, i);
                }
            }
            mtrx__.blacs_grid().comm().allreduce(&cs, 1);
            if (ctx_.comm_band().rank() == 0) {
                utils::print_checksum("subspace_mtrx_old", cs);
            }
        }
    }

    /* <{phi,phi_new}|Op|phi_new> */
    inner(ctx_.preferred_memory_t(), ctx_.blas_linalg_t(), (ctx_.num_mag_dims() == 3) ? 2 : 0, phi__, 0, N__ + n__,
          op_phi__, N__, n__, mtrx__, 0, N__);

    /* restore lower part */
    if (N__ > 0) {
        if (mtrx__.blacs_grid().comm().size() == 1) {
            #pragma omp parallel for
            for (int i = 0; i < N__; i++) {
                for (int j = N__; j < N__ + n__; j++) {
                    mtrx__(j, i) = utils::conj(mtrx__(i, j));
                }
            }
        } else {
            tranc(n__, N__, mtrx__, 0, N__, mtrx__, N__, 0);
        }
    }

    if (ctx_.control().print_checksum_) {
        splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                       mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
        splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                       mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());
        double_complex cs(0, 0);
        for (int i = 0; i < spl_col.local_size(); i++) {
            for (int j = 0; j < spl_row.local_size(); j++) {
                cs += mtrx__(j, i);
            }
        }
        mtrx__.blacs_grid().comm().allreduce(&cs, 1);
        if (ctx_.comm_band().rank() == 0) {
            utils::print_checksum("subspace_mtrx", cs);
        }
    }

    /* kill any numerical noise */
    mtrx__.make_real_diag(N__ + n__);

    /* save new matrix */
    if (mtrx_old__) {
        splindex<block_cyclic> spl_row(N__ + n__, mtrx__.blacs_grid().num_ranks_row(),
                                       mtrx__.blacs_grid().rank_row(), mtrx__.bs_row());
        splindex<block_cyclic> spl_col(N__ + n__, mtrx__.blacs_grid().num_ranks_col(),
                                       mtrx__.blacs_grid().rank_col(), mtrx__.bs_col());

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < spl_col.local_size(); i++) {
            std::copy(&mtrx__(0, i), &mtrx__(0, i) + spl_row.local_size(), &(*mtrx_old__)(0, i));
        }
    }
}
