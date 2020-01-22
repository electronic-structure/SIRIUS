// Copyright (c) 2013-2019 Anton Kozhevnikov, Thomas Schulthess
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

/** \file non_local_operator.cpp
 *
 *  \brief Contains implementation of sirius::Non_local_operator class.
 */

#include "non_local_operator.hpp"
#include "Beta_projectors/beta_projectors.hpp"

namespace sirius {

Non_local_operator::Non_local_operator(Simulation_context const& ctx__)
    : ctx_(ctx__)
{
    PROFILE("sirius::Non_local_operator");

    pu_                 = this->ctx_.processing_unit();
    auto& uc            = this->ctx_.unit_cell();
    packed_mtrx_offset_ = sddk::mdarray<int, 1>(uc.num_atoms());
    packed_mtrx_size_   = 0;
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        int nbf                 = uc.atom(ia).mt_basis_size();
        packed_mtrx_offset_(ia) = packed_mtrx_size_;
        packed_mtrx_size_ += nbf * nbf;
    }

    switch (pu_) {
        case device_t::GPU: {
            packed_mtrx_offset_.allocate(memory_t::device).copy_to(memory_t::device);
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

template <>
double Non_local_operator::value<double>(int xi1__, int xi2__, int ispn__, int ia__)
{
    int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
    return this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
}

template <>
double_complex Non_local_operator::value<double_complex>(int xi1__, int xi2__, int ispn__, int ia__)
{
    int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
    return double_complex(this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__),
                          this->op_(1, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__));
}

template <>
void Non_local_operator::apply<double_complex>(int chunk__, int ispn_block__, Wave_functions& op_phi__, int idx0__,
                                               int n__, Beta_projectors_base& beta__,
                                               matrix<double_complex>& beta_phi__)
{
    PROFILE("sirius::Non_local_operator::apply");

    if (is_null_) {
        return;
    }

    auto& beta_gk     = beta__.pw_coeffs_a();
    int num_gkvec_loc = beta__.num_gkvec_loc();
    int nbeta         = beta__.chunk(chunk__).num_beta_;

    /* setup linear algebra parameters */
    memory_t mem{memory_t::none};
    linalg_t la{linalg_t::none};
    switch (pu_) {
        case device_t::CPU: {
            mem = memory_t::host;
            la  = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            mem = memory_t::device;
            la  = linalg_t::gpublas;
            break;
        }
    }

    auto work = mdarray<double_complex, 1>(nbeta * n__, ctx_.mem_pool(mem));

    /* compute O * <beta|phi> for atoms in a chunk */
    #pragma omp parallel
    {
        acc::set_device_id(sddk::get_device_id(acc::num_devices())); // avoid cuda mth bugs

        #pragma omp for
        for (int i = 0; i < beta__.chunk(chunk__).num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i);
            int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), i);
            int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

            if (nbf) {
                linalg(la).gemm(
                    'N', 'N', nbf, n__, nbf, &linalg_const<double_complex>::one(),
                    reinterpret_cast<double_complex*>(op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                    beta_phi__.at(mem, offs, 0), beta_phi__.ld(), &linalg_const<double_complex>::zero(),
                    work.at(mem, offs), nbeta, stream_id(omp_get_thread_num()));
            }
        }
    }
    switch (pu_) {
        case device_t::GPU: {
            /* wait for previous zgemms */
            #pragma omp parallel
            acc::sync_stream(stream_id(omp_get_thread_num()));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }

    int jspn = ispn_block__ & 1;

    /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
    linalg(ctx_.blas_linalg_t())
        .gemm('N', 'N', num_gkvec_loc, n__, nbeta, &linalg_const<double_complex>::one(), beta_gk.at(mem), num_gkvec_loc,
              work.at(mem), nbeta, &linalg_const<double_complex>::one(),
              op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__),
              op_phi__.pw_coeffs(jspn).prime().ld());

    switch (pu_) {
        case device_t::GPU: {
            acc::sync_stream(stream_id(-1));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

template <>
void Non_local_operator::apply<double_complex>(int chunk__, int ia__, int ispn_block__, Wave_functions& op_phi__,
                                               int idx0__, int n__, Beta_projectors_base& beta__,
                                               matrix<double_complex>& beta_phi__)
{
    if (is_null_) {
        return;
    }

    auto& beta_gk     = beta__.pw_coeffs_a();
    int num_gkvec_loc = beta__.num_gkvec_loc();

    int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), ia__);
    int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), ia__);
    int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), ia__);

    if (nbf == 0) {
        return;
    }

    /* setup linear algebra parameters */
    memory_t mem{memory_t::none};
    linalg_t la{linalg_t::none};

    switch (pu_) {
        case device_t::CPU: {
            mem = memory_t::host;
            la  = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            mem = memory_t::device;
            la  = linalg_t::gpublas;
            break;
        }
    }

    auto work = mdarray<double_complex, 1>(nbf * n__, ctx_.mem_pool(mem));

    linalg(la).gemm('N', 'N', nbf, n__, nbf, &linalg_const<double_complex>::one(),
                     reinterpret_cast<double_complex*>(op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                     beta_phi__.at(mem, offs, 0), beta_phi__.ld(), &linalg_const<double_complex>::zero(), work.at(mem),
                     nbf);

    int jspn = ispn_block__ & 1;

    linalg(ctx_.blas_linalg_t())
        .gemm('N', 'N', num_gkvec_loc, n__, nbf, &linalg_const<double_complex>::one(), beta_gk.at(mem, 0, offs),
              num_gkvec_loc, work.at(mem), nbf, &linalg_const<double_complex>::one(),
              op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__),
              op_phi__.pw_coeffs(jspn).prime().ld());
    switch (pu_) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
#ifdef __GPU
            acc::sync_stream(stream_id(-1));
#endif
            break;
        }
    }
}

template <>
void Non_local_operator::apply<double>(int chunk__, int ispn_block__, Wave_functions& op_phi__, int idx0__, int n__,
                                       Beta_projectors_base& beta__, matrix<double>& beta_phi__)
{
    PROFILE("sirius::Non_local_operator::apply");

    if (is_null_) {
        return;
    }

    auto& beta_gk     = beta__.pw_coeffs_a();
    int num_gkvec_loc = beta__.num_gkvec_loc();
    int nbeta         = beta__.chunk(chunk__).num_beta_;

    /* setup linear algebra parameters */
    memory_t mem{memory_t::none};
    linalg_t la{linalg_t::none};
    switch (pu_) {
        case device_t::CPU: {
            mem = memory_t::host;
            la  = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            mem = memory_t::device;
            la  = linalg_t::gpublas;
            break;
        }
    }

    auto work = mdarray<double, 1>(nbeta * n__, ctx_.mem_pool(mem));

    /* compute O * <beta|phi> for atoms in a chunk */
    #pragma omp parallel for
    for (int i = 0; i < beta__.chunk(chunk__).num_atoms_; i++) {
        /* number of beta functions for a given atom */
        int nbf  = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::nbf), i);
        int offs = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::offset), i);
        int ia   = beta__.chunk(chunk__).desc_(static_cast<int>(beta_desc_idx::ia), i);

        if (nbf == 0) {
            continue;
        }
        linalg(la).gemm('N', 'N', nbf, n__, nbf, &linalg_const<double>::one(),
                         op_.at(mem, 0, packed_mtrx_offset_(ia), ispn_block__), nbf, beta_phi__.at(mem, offs, 0),
                         beta_phi__.ld(), &linalg_const<double>::zero(), work.at(mem, offs), nbeta,
                         stream_id(omp_get_thread_num()));
    }
    switch (pu_) {
        case device_t::GPU: {
            /* wait for previous zgemms */
            #pragma omp parallel
            acc::sync_stream(stream_id(omp_get_thread_num()));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }

    int jspn = ispn_block__ & 1;

    /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
    linalg(ctx_.blas_linalg_t())
        .gemm('N', 'N', 2 * num_gkvec_loc, n__, nbeta, &linalg_const<double>::one(),
              reinterpret_cast<double*>(beta_gk.at(mem)), 2 * num_gkvec_loc, work.at(mem), nbeta,
              &linalg_const<double>::one(),
              reinterpret_cast<double*>(op_phi__.pw_coeffs(jspn).prime().at(op_phi__.preferred_memory_t(), 0, idx0__)),
              2 * op_phi__.pw_coeffs(jspn).prime().ld());

    switch (pu_) {
        case device_t::GPU: {
            acc::sync_stream(stream_id(-1));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

D_operator::D_operator(Simulation_context const& ctx_)
    : Non_local_operator(ctx_)
{
    if (ctx_.gamma_point()) {
        this->op_ = mdarray<double, 3>(1, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    } else {
        this->op_ = mdarray<double, 3>(2, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    }
    this->op_.zero();
    initialize();
}

void D_operator::initialize()
{
    PROFILE("sirius::D_operator::initialize");

    auto& uc = this->ctx_.unit_cell();

    #pragma omp parallel for
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        auto& atom = uc.atom(ia);
        int nbf    = atom.mt_basis_size();
        auto& dion = atom.type().d_mtrx_ion();

        /* in case of spin orbit coupling */
        if (uc.atom(ia).type().spin_orbit_coupling()) {
            mdarray<double_complex, 3> d_mtrx_so(nbf, nbf, 4);
            d_mtrx_so.zero();

            /* transform the d_mtrx */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {

                    /* first compute \f[A^_\alpha I^{I,\alpha}_{xi,xi}\f] cf Eq.19 in doi:10.1103/PhysRevB.71.115106  */

                    /* note that the `I` integrals are already calculated and stored in atom.d_mtrx */
                    for (int sigma = 0; sigma < 2; sigma++) {
                        for (int sigmap = 0; sigmap < 2; sigmap++) {
                            double_complex result(0, 0);
                            for (auto xi2p = 0; xi2p < nbf; xi2p++) {
                                if (atom.type().compare_index_beta_functions(xi2, xi2p)) {
                                    /* just sum over m2, all other indices are the same */
                                    for (auto xi1p = 0; xi1p < nbf; xi1p++) {
                                        if (atom.type().compare_index_beta_functions(xi1, xi1p)) {
                                            /* just sum over m1, all other indices are the same */

                                            /* loop over the 0, z,x,y coordinates */
                                            for (int alpha = 0; alpha < 4; alpha++) {
                                                for (int sigma1 = 0; sigma1 < 2; sigma1++) {
                                                    for (int sigma2 = 0; sigma2 < 2; sigma2++) {
                                                        result += atom.d_mtrx(xi1p, xi2p, alpha) *
                                                                  pauli_matrix[alpha][sigma1][sigma2] *
                                                                  atom.type().f_coefficients(xi1, xi1p, sigma, sigma1) *
                                                                  atom.type().f_coefficients(xi2p, xi2, sigma2, sigmap);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            const int ind = (sigma == sigmap) * sigma + (1 + 2 * sigma + sigmap) * (sigma != sigmap);
                            d_mtrx_so(xi1, xi2, ind) = result;
                        }
                    }
                }
            }

            /* add ionic contribution */

            /* spin orbit coupling mixes terms */

            /* keep the order of the indices because it is crucial here;
               permuting the indices makes things wrong */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                int l2     = atom.type().indexb(xi2).l;
                double j2  = atom.type().indexb(xi2).j;
                int idxrf2 = atom.type().indexb(xi2).idxrf;
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int l1     = atom.type().indexb(xi1).l;
                    double j1  = atom.type().indexb(xi1).j;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;
                    if ((l1 == l2) && (std::abs(j1 - j2) < 1e-8)) {
                        /* up-up down-down */
                        d_mtrx_so(xi1, xi2, 0) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 0, 0);
                        d_mtrx_so(xi1, xi2, 1) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 1, 1);

                        /* up-down down-up */
                        d_mtrx_so(xi1, xi2, 2) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 0, 1);
                        d_mtrx_so(xi1, xi2, 3) += dion(idxrf1, idxrf2) * atom.type().f_coefficients(xi1, xi2, 1, 0);
                    }
                }
            }

            /* the pseudo potential contains information about
               spin orbit coupling so we use a different formula
               Eq.19 doi:10.1103/PhysRevB.71.115106 for calculating the D matrix

               Note that the D matrices are stored and
               calculated in the up-down basis already not the (Veff,Bx,By,Bz) one */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int idx = xi2 * nbf + xi1;
                    for (int s = 0; s < 4; s++) {
                        this->op_(0, this->packed_mtrx_offset_(ia) + idx, s) = d_mtrx_so(xi1, xi2, s).real();
                        this->op_(1, this->packed_mtrx_offset_(ia) + idx, s) = d_mtrx_so(xi1, xi2, s).imag();
                    }
                }
            }
        } else {
            /* No spin orbit coupling for this atom \f[D = D(V_{eff})
               I + D(B_x) \sigma_x + D(B_y) sigma_y + D(B_z)
               sigma_z\f] since the D matrices are calculated that way */
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                int lm2    = atom.type().indexb(xi2).lm;
                int idxrf2 = atom.type().indexb(xi2).idxrf;
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    int lm1    = atom.type().indexb(xi1).lm;
                    int idxrf1 = atom.type().indexb(xi1).idxrf;

                    int idx = xi2 * nbf + xi1;
                    switch (this->ctx_.num_mag_dims()) {
                        case 3: {
                            double bx = uc.atom(ia).d_mtrx(xi1, xi2, 2);
                            double by = uc.atom(ia).d_mtrx(xi1, xi2, 3);

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 2) = bx;
                            this->op_(1, this->packed_mtrx_offset_(ia) + idx, 2) = -by;

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 3) = bx;
                            this->op_(1, this->packed_mtrx_offset_(ia) + idx, 3) = by;
                        }
                        case 1: {
                            double v  = uc.atom(ia).d_mtrx(xi1, xi2, 0);
                            double bz = uc.atom(ia).d_mtrx(xi1, xi2, 1);

                            /* add ionic part */
                            if (lm1 == lm2) {
                                v += dion(idxrf1, idxrf2);
                            }

                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) = v + bz;
                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 1) = v - bz;
                            break;
                        }
                        case 0: {
                            this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) = uc.atom(ia).d_mtrx(xi1, xi2, 0);
                            /* add ionic part */
                            if (lm1 == lm2) {
                                this->op_(0, this->packed_mtrx_offset_(ia) + idx, 0) += dion(idxrf1, idxrf2);
                            }
                            break;
                        }
                        default: {
                            TERMINATE("wrong number of magnetic dimensions");
                        }
                    }
                }
            }
        }
    }

    if (this->ctx_.control().print_checksum_ && this->ctx_.comm().rank() == 0) {
        auto cs = this->op_.checksum();
        utils::print_checksum("D_operator", cs);
    }

    if (this->pu_ == device_t::GPU) {
        this->op_.allocate(memory_t::device).copy_to(memory_t::device);
    }

    /* D-operator is not diagonal in spin in case of non-collinear magnetism
       (spin-orbit coupling falls into this case) */
    if (ctx_.num_mag_dims() == 3) {
        this->is_diag_ = false;
    }
}

Q_operator::Q_operator(Simulation_context const& ctx__)
    : Non_local_operator(ctx__)
{
    /* Q-operator is independent of spin if there is no spin-orbit; however, it simplifies the apply()
     * method if the Q-operator has a spin index */
    if (ctx_.gamma_point()) {
        this->op_ = mdarray<double, 3>(1, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    } else {
        this->op_ = mdarray<double, 3>(2, this->packed_mtrx_size_, ctx_.num_mag_dims() + 1);
    }
    this->op_.zero();
    initialize();
}

void Q_operator::initialize()
{
    PROFILE("sirius::Q_operator::initialize");

    auto& uc = this->ctx_.unit_cell();
    /* check eigen-values of Q_{xi,xi'} matrix for each atom;
       not sure if it helps, so it's commented for now */
    // if (this->ctx_.control().verification_ >= 1) {
    //    for (int ia = 0; ia < uc.num_atoms(); ia++) {
    //        int iat = uc.atom(ia).type().id();
    //        if (!uc.atom_type(iat).augment()) {
    //            continue;
    //        }
    //        int nbf = uc.atom(ia).mt_basis_size();
    //        Eigensolver_lapack evs;
    //        dmatrix<double> A(nbf, nbf);
    //        dmatrix<double> Z(nbf, nbf);
    //        std::vector<double> ev(nbf);
    //        for (int xi1 = 0; xi1 < nbf; xi1++) {
    //            for (int xi2 = 0; xi2 < nbf; xi2++) {
    //                A(xi1, xi2) = this->ctx_.augmentation_op(iat).q_mtrx(xi1, xi2);
    //            }
    //        }
    //        evs.solve(nbf, A, ev.data(), Z);
    //        if (this->ctx_.control().verbosity_ >= 0 && this->ctx_.comm().rank() == 0) {
    //            printf("eigen-values of the Q-matrix for atom %i\n", ia);
    //            for (int i = 0; i < nbf; i++) {
    //                printf("%18.12f\n", ev[i]);
    //            }
    //        }
    //    }
    //}
    #pragma omp parallel for
    for (int ia = 0; ia < uc.num_atoms(); ia++) {
        int iat = uc.atom(ia).type().id();
        if (!uc.atom_type(iat).augment()) {
            continue;
        }
        int nbf = uc.atom(ia).mt_basis_size();
        for (int xi2 = 0; xi2 < nbf; xi2++) {
            for (int xi1 = 0; xi1 < nbf; xi1++) {
                /* The ultra soft pseudo potential has spin orbit coupling incorporated to it, so we
                   need to rotate the Q matrix */
                if (uc.atom_type(iat).spin_orbit_coupling()) {
                    /* this is nothing else than Eq.18 of doi:10.1103/PhysRevB.71.115106 */
                    for (auto si = 0; si < 2; si++) {
                        for (auto sj = 0; sj < 2; sj++) {

                            double_complex result(0, 0);

                            for (int xi2p = 0; xi2p < nbf; xi2p++) {
                                if (uc.atom(ia).type().compare_index_beta_functions(xi2, xi2p)) {
                                    for (int xi1p = 0; xi1p < nbf; xi1p++) {
                                        /* The F coefficients are already "block diagonal" so we do a full
                                           summation. We actually rotate the q_matrices only */
                                        if (uc.atom(ia).type().compare_index_beta_functions(xi1, xi1p)) {
                                            result += this->ctx_.augmentation_op(iat)->q_mtrx(xi1p, xi2p) *
                                                      (uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 0) *
                                                       uc.atom(ia).type().f_coefficients(xi2p, xi2, 0, si) +
                                                       uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 1) *
                                                       uc.atom(ia).type().f_coefficients(xi2p, xi2, 1, si));
                                        }
                                    }
                                }
                            }

                            /* the order of the index is important */
                            const int ind = (si == sj) ? si : sj + 2;
                            /* this gives
                               ind = 0 if si = up and sj = up
                               ind = 1 if si = sj = down
                               ind = 2 if si = down and sj = up
                               ind = 3 if si = up and sj = down */
                            this->op_(0, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ind) = result.real();
                            this->op_(1, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ind) = result.imag();
                        }
                    }
                } else {
                    for (int ispn = 0; ispn < this->ctx_.num_spins(); ispn++) {
                        this->op_(0, this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ispn) =
                            this->ctx_.augmentation_op(iat)->q_mtrx(xi1, xi2);
                    }
                }
            }
        }
    }
    if (this->ctx_.control().print_checksum_ && this->ctx_.comm().rank() == 0) {
        auto cs = this->op_.checksum();
        utils::print_checksum("Q_operator", cs);
    }

    if (this->pu_ == device_t::GPU) {
        this->op_.allocate(memory_t::device).copy_to(memory_t::device);
    }

    this->is_null_ = true;
    for (int iat = 0; iat < uc.num_atom_types(); iat++) {
        if (uc.atom_type(iat).augment()) {
            this->is_null_ = false;
        }
        /* Q-operator is not diagonal in spin only in the case of spin-orbit coupling */
        if (uc.atom_type(iat).spin_orbit_coupling()) {
            this->is_diag_ = false;
        }
    }
}

template <typename T>
void
apply_non_local_d_q(spin_range spins__, int N__, int n__, Beta_projectors& beta__, Wave_functions& phi__,
                    D_operator* d_op__, Wave_functions* hphi__, Q_operator* q_op__, Wave_functions* sphi__)
{

    for (int i = 0; i < beta__.num_chunks(); i++) {
        /* generate beta-projectors for a block of atoms */
        beta__.generate(i);

        for (int ispn : spins__) {
            auto beta_phi = beta__.inner<T>(i, phi__, ispn, N__, n__);

            if (hphi__ && d_op__) {
                /* apply diagonal spin blocks */
                d_op__->apply(i, ispn, *hphi__, N__, n__, beta__, beta_phi);
                if (!d_op__->is_diag() && hphi__->num_sc() == 2) {
                    /* apply non-diagonal spin blocks */
                    /* xor 3 operator will map 0 to 3 and 1 to 2 */
                    d_op__->apply(i, ispn ^ 3, *hphi__, N__, n__, beta__, beta_phi);
                }
            }

            if (sphi__ && q_op__) {
                /* apply Q operator (diagonal in spin) */
                q_op__->apply(i, ispn, *sphi__, N__, n__, beta__, beta_phi);
                if (!q_op__->is_diag() && sphi__->num_sc() == 2) {
                    q_op__->apply(i, ispn ^ 3, *sphi__, N__, n__, beta__, beta_phi);
                }
            }
        }
    }
}

/// Compute |sphi> = (1 + Q)|phi>
template <typename T>
void
apply_S_operator(device_t pu__, spin_range spins__, int N__, int n__, Beta_projectors& beta__,
                 Wave_functions& phi__, Q_operator* q_op__, Wave_functions& sphi__)
{
    for (auto s: spins__) {
        sphi__.copy_from(pu__, n__, phi__, s, N__, s, N__);
    }

    if (q_op__) {
        beta__.prepare();
        apply_non_local_d_q<T>(spins__, N__, n__, beta__, phi__, nullptr, nullptr, q_op__, &sphi__);
        beta__.dismiss();
    }
}

template
void
apply_non_local_d_q<double>(spin_range spins__, int N__, int n__, Beta_projectors& beta__,
                            Wave_functions& phi__, D_operator* d_op__, Wave_functions* hphi__,
                            Q_operator* q_op__, Wave_functions* sphi__);

template
void
apply_non_local_d_q<double_complex>(spin_range spins__, int N__, int n__, Beta_projectors& beta__,
                                    Wave_functions& phi__, D_operator* d_op__, Wave_functions* hphi__,
                                    Q_operator* q_op__, Wave_functions* sphi__);

template
void
apply_S_operator<double>(device_t pu__, spin_range spins__, int N__, int n__, Beta_projectors& beta__,
                         Wave_functions& phi__, Q_operator* q_op__, Wave_functions& sphi__);

template
void
apply_S_operator<double_complex>(device_t pu__, spin_range spins__, int N__, int n__, Beta_projectors& beta__,
                                 Wave_functions& phi__, Q_operator* q_op__, Wave_functions& sphi__);

} // namespace sirius
