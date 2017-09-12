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

/** \file non_local_operator.h
 *
 *  \brief Contains declaration and implementation of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_H__
#define __NON_LOCAL_OPERATOR_H__

#include "Beta_projectors/beta_projectors.h"
#include "simulation_context.h"

namespace sirius {

template <typename T>
class Non_local_operator
{
  protected:
    Beta_projectors& beta_;

    device_t pu_;

    int packed_mtrx_size_;

    mdarray<int, 1> packed_mtrx_offset_;

    /// Non-local operator matrix.
    mdarray<T, 2> op_;

    mdarray<T, 1> work_;

    bool is_null_{false};

    Non_local_operator& operator=(Non_local_operator const& src) = delete;
    Non_local_operator(Non_local_operator const& src)            = delete;

  public:
    Non_local_operator(Beta_projectors& beta__, device_t pu__)
        : beta_(beta__)
        , pu_(pu__)
    {
        PROFILE("sirius::Non_local_operator::Non_local_operator");

        auto& uc            = beta_.unit_cell();
        packed_mtrx_offset_ = mdarray<int, 1>(uc.num_atoms());
        packed_mtrx_size_   = 0;
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            int nbf                 = uc.atom(ia).mt_basis_size();
            packed_mtrx_offset_(ia) = packed_mtrx_size_;
            packed_mtrx_size_ += nbf * nbf;
        }

        if (pu_ == GPU) {
            packed_mtrx_offset_.allocate(memory_t::device);
            packed_mtrx_offset_.template copy<memory_t::host, memory_t::device>();
        }
    }

    ~Non_local_operator()
    {
    }

    inline void apply(int chunk__, int ispn__, wave_functions& op_phi__, int idx0__, int n__, matrix<T>& beta_phi__);

    inline T operator()(int xi1__, int xi2__, int ia__)
    {
        return (*this)(xi1__, xi2__, 0, ia__);
    }

    inline T operator()(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = beta_.unit_cell().atom(ia__).mt_basis_size();
        return op_(packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
    }
};

template <>
inline void Non_local_operator<double_complex>::apply(
    int chunk__, int ispn__, wave_functions& op_phi__, int idx0__, int n__, matrix<double_complex>& beta_phi__)
{
    PROFILE("sirius::Non_local_operator::apply");

    if (is_null_) {
        return;
    }

    assert(op_phi__.pw_coeffs().num_rows_loc() == beta_.num_gkvec_loc());

    auto& beta_gk     = beta_.pw_coeffs_a();
    int num_gkvec_loc = beta_.num_gkvec_loc();
    auto& bp_chunks   = beta_.beta_projector_chunks();
    int nbeta         = bp_chunks(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > work_.size()) {
        work_ = mdarray<double_complex, 1>(nbeta * n__);
        if (pu_ == GPU) {
            work_.allocate(memory_t::device);
        }
    }
    /* compute O * <beta|phi> for atoms in a chunk */
    #pragma omp parallel for
    for (int i = 0; i < bp_chunks(chunk__).num_atoms_; i++) {
        /* number of beta functions for a given atom */
        int nbf  = bp_chunks(chunk__).desc_(beta_desc_idx::nbf, i);
        int offs = bp_chunks(chunk__).desc_(beta_desc_idx::offset, i);
        int ia   = bp_chunks(chunk__).desc_(beta_desc_idx::ia, i);
        switch (pu_) {
            case CPU: {
                linalg<CPU>::gemm(0, 0, nbf, n__, nbf, op_.at<CPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                                  beta_phi__.at<CPU>(offs, 0), nbeta, work_.at<CPU>(offs), nbeta);

                break;
            }
            case GPU: {
#ifdef __GPU
                linalg<GPU>::gemm(0, 0, nbf, n__, nbf, op_.at<GPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                                  beta_phi__.at<GPU>(offs, 0), nbeta, work_.at<GPU>(offs), nbeta, omp_get_thread_num());
#endif
                break;
            }
        }
    }

    /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
    switch (pu_) {
        case CPU: {
            linalg<CPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, linalg_const<double_complex>::one(), beta_gk.at<CPU>(),
                              num_gkvec_loc, work_.at<CPU>(), nbeta, linalg_const<double_complex>::one(),
                              op_phi__.pw_coeffs().prime().at<CPU>(0, idx0__), op_phi__.pw_coeffs().prime().ld());
            break;
        }
        case GPU: {
#ifdef __GPU
            /* wait for previous zgemms */
            #pragma omp parallel
            acc::sync_stream(omp_get_thread_num());

            linalg<GPU>::gemm(0, 0, num_gkvec_loc, n__, nbeta, &linalg_const<double_complex>::one(), beta_gk.at<GPU>(),
                              beta_gk.ld(), work_.at<GPU>(), nbeta, &linalg_const<double_complex>::one(),
                              op_phi__.pw_coeffs().prime().at<GPU>(0, idx0__), op_phi__.pw_coeffs().prime().ld());
            acc::sync_stream(-1);
#endif
            break;
        }
    }
}

template <>
inline void Non_local_operator<double>::apply(int chunk__,
                                              int ispn__,
                                              wave_functions& op_phi__,
                                              int idx0__,
                                              int n__,
                                              matrix<double>& beta_phi__)
{
    PROFILE("sirius::Non_local_operator::apply");

    if (is_null_) {
        return;
    }

    assert(op_phi__.pw_coeffs().num_rows_loc() == beta_.num_gkvec_loc());

    auto& beta_gk     = beta_.pw_coeffs_a();
    int num_gkvec_loc = beta_.num_gkvec_loc();
    auto& bp_chunks   = beta_.beta_projector_chunks();
    int nbeta         = bp_chunks(chunk__).num_beta_;

    if (static_cast<size_t>(nbeta * n__) > work_.size()) {
        work_ = mdarray<double, 1>(nbeta * n__);
        if (pu_ == GPU) {
            work_.allocate(memory_t::device);
        }
    }

    /* compute O * <beta|phi> */
    #pragma omp parallel for
    for (int i = 0; i < bp_chunks(chunk__).num_atoms_; i++) {
        /* number of beta functions for a given atom */
        int nbf  = bp_chunks(chunk__).desc_(beta_desc_idx::nbf, i);
        int offs = bp_chunks(chunk__).desc_(beta_desc_idx::offset, i);
        int ia   = bp_chunks(chunk__).desc_(beta_desc_idx::ia, i);

        switch (pu_) {
            case CPU: {
                linalg<CPU>::gemm(0, 0, nbf, n__, nbf, op_.at<CPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                                  beta_phi__.at<CPU>(offs, 0), nbeta, work_.at<CPU>(offs), nbeta);
                break;
            }
            case GPU: {
#ifdef __GPU
                linalg<GPU>::gemm(0, 0, nbf, n__, nbf, op_.at<GPU>(packed_mtrx_offset_(ia), ispn__), nbf,
                                  beta_phi__.at<GPU>(offs, 0), nbeta, work_.at<GPU>(offs), nbeta, omp_get_thread_num());
                break;
#endif
            }
        }
    }

    /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
    switch (pu_) {
        case CPU: {
            linalg<CPU>::gemm(0, 0, 2 * num_gkvec_loc, n__, nbeta, 1.0, reinterpret_cast<double*>(beta_gk.at<CPU>()),
                              2 * num_gkvec_loc, work_.at<CPU>(), nbeta, 1.0,
                              reinterpret_cast<double*>(op_phi__.pw_coeffs().prime().at<CPU>(0, idx0__)),
                              2 * op_phi__.pw_coeffs().prime().ld());
            break;
        }
        case GPU: {
#ifdef __GPU
            /* wait for previous zgemms */
            #pragma omp parallel
            acc::sync_stream(omp_get_thread_num());

            linalg<GPU>::gemm(0, 0, 2 * num_gkvec_loc, n__, nbeta, &linalg_const<double>::one(),
                              reinterpret_cast<double*>(beta_gk.at<GPU>()), 2 * num_gkvec_loc, work_.at<GPU>(), nbeta,
                              &linalg_const<double>::one(),
                              reinterpret_cast<double*>(op_phi__.pw_coeffs().prime().at<GPU>(0, idx0__)),
                              2 * num_gkvec_loc);
            acc::sync_stream(-1);
#endif
            break;
        }
    }
}

template <typename T>
class D_operator : public Non_local_operator<T>
{
  public:
    D_operator(Simulation_context const& ctx__, Beta_projectors& beta__)
        : Non_local_operator<T>(beta__, ctx__.processing_unit())
    {
        this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, ctx__.num_mag_dims() + 1);
        this->op_.zero();
        /* D-matrix is complex in non-collinear case */
        if (ctx__.num_mag_dims() == 3) {
            assert((std::is_same<T, double_complex>::value));
        }

        auto& uc = this->beta_.unit_cell();

        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            int nbf = uc.atom(ia).mt_basis_size();
            if (uc.atom(ia).type().pp_desc().spin_orbit_coupling) {

                // the pseudo potential contains information about
                // spin orbit coupling so we use a different formula
                // Eq.19 PRB 71 115106 for calculating the D matrix

                // Note that the D matrices are stored and
                // calculated in the up-down basis already not the
                // (Veff,Bx,By,Bz) one.
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        int idx = xi2 * nbf + xi1;
                        for (int s = 0; s < 4; s++)
                            this->op_(this->packed_mtrx_offset_(ia) + idx, s) =
                                type_wrapper<T>::bypass(uc.atom(ia).d_mtrx_so(xi1, xi2, s));
                    }
                }
            } else {
                // No spin orbit coupling for this atom \f[D = D(V_{eff})
                // I + D(B_x) \sigma_x + D(B_y) sigma_y + D(B_z)
                // sigma_z\f] since the D matrices are calculated that
                // way.
                for (int xi2 = 0; xi2 < nbf; xi2++) {
                    for (int xi1 = 0; xi1 < nbf; xi1++) {
                        int idx = xi2 * nbf + xi1;
                        switch (ctx__.num_mag_dims()) {
                            case 3: {
                                double bx = uc.atom(ia).d_mtrx(xi1, xi2, 2);
                                double by = uc.atom(ia).d_mtrx(xi1, xi2, 3);
                                this->op_(this->packed_mtrx_offset_(ia) + idx, 2) =
                                    type_wrapper<T>::bypass(double_complex(bx, -by));
                                this->op_(this->packed_mtrx_offset_(ia) + idx, 3) =
                                    type_wrapper<T>::bypass(double_complex(bx, by));
                            }
                            case 1: {
                                double v  = uc.atom(ia).d_mtrx(xi1, xi2, 0);
                                double bz = uc.atom(ia).d_mtrx(xi1, xi2, 1);
                                this->op_(this->packed_mtrx_offset_(ia) + idx, 0) = v + bz;
                                this->op_(this->packed_mtrx_offset_(ia) + idx, 1) = v - bz;
                                break;
                            }
                            case 0: {
                                this->op_(this->packed_mtrx_offset_(ia) + idx, 0) =
                                    type_wrapper<T>::bypass(uc.atom(ia).d_mtrx(xi1, xi2, 0));
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

        if (ctx__.control().print_checksum_ && ctx__.comm().rank() == 0) {
            auto cs = this->op_.checksum();
            print_checksum("D_operator", cs);
        }

        if (this->pu_ == GPU) {
            this->op_.allocate(memory_t::device);
            this->op_.template copy<memory_t::host, memory_t::device>();
        }
    }
};

template <typename T>
class Q_operator : public Non_local_operator<T>
{
  public:
    Q_operator(Simulation_context const& ctx__, Beta_projectors& beta__)
        : Non_local_operator<T>(beta__, ctx__.processing_unit())
    {
        if (ctx__.so_correction()) {
            this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, 4);
        } else {
            /* Q-operator is independent of spin */
            this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, 1);
        }
        this->op_.zero();
        auto& uc = this->beta_.unit_cell();
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            int iat = uc.atom(ia).type().id();
            if (!uc.atom_type(iat).pp_desc().augment) {
                continue;
            }
            int nbf = uc.atom(ia).mt_basis_size();
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    // if (ctx__.unit_cell().atom_type(iat).pp_desc().augment) {
                    if (ctx__.unit_cell().atom_type(iat).pp_desc().spin_orbit_coupling) {
                        // the ultra soft pseudo potential has spin
                        // orbit coupling incorporated to it. so we
                        // need to rotate the q matrix

                        // it is nothing else than Eq.18 of Ref PRB 71, 115106
                        for (auto si = 0; si < 2; si++) {
                            for (auto sj = 0; sj < 2; sj++) {

                                double_complex result = double_complex(0.0, 0.0);

                                for (int xi2p = 0; xi2p < nbf; xi2p++) {
                                    if (uc.atom(ia).type().compare_index_beta_functions(xi2, xi2p)) {
                                        for (int xi1p = 0; xi1p < nbf; xi1p++) {
                                            // the F_Coefficients are already "block diagonal" so we do a full
                                            // summation.
                                            // We actually rotate the q_matrices only....
                                            if (uc.atom(ia).type().compare_index_beta_functions(xi1, xi1p)) {
                                                result += ctx__.augmentation_op(iat).q_mtrx(xi1p, xi2p) *
                                                          (uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 0) *
                                                               uc.atom(ia).type().f_coefficients(xi2p, xi2, 0, si) +
                                                           uc.atom(ia).type().f_coefficients(xi1, xi1p, sj, 1) *
                                                               uc.atom(ia).type().f_coefficients(xi2p, xi2, 1, si));
                                            }
                                        }
                                    }
                                }

                                // the order of the index is important
                                const int ind = (si == sj) * si + (1 + 2 * sj + si) * (si != sj);
                                // this formula gives
                                // ind = 0 if si = up and sj = up
                                // ind = 1 if si = sj = down
                                // ind = 2 if si = down and sj = up
                                // ind = 3 if si = up and sj = down
                                assert(ind <= 3);
                                this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, ind) =
                                    type_wrapper<T>::bypass(result);
                            }
                        }
                    } else {
                        this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) =
                            ctx__.augmentation_op(iat).q_mtrx(xi1, xi2);
                        if (ctx__.so_correction()) {
                            // when spin orbit is included the q
                            // matrix is spin depend even for non so
                            // pseudo potentials
                            this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 1) =
                                ctx__.augmentation_op(iat).q_mtrx(xi1, xi2);
                        }
                    }
                    // }
                }
            }
        }
        if (ctx__.control().print_checksum_ && ctx__.comm().rank() == 0) {
            auto cs = this->op_.checksum();
            print_checksum("Q_operator", cs);
        }

        if (this->pu_ == GPU) {
            this->op_.allocate(memory_t::device);
            this->op_.template copy<memory_t::host, memory_t::device>();
        }
    }
};

template <typename T>
class P_operator : public Non_local_operator<T>
{
  public:
    P_operator(Simulation_context const& ctx__, Beta_projectors& beta__, mdarray<double_complex, 3>& p_mtrx__)
        : Non_local_operator<T>(beta__, ctx__.processing_unit())
    {
        /* Q-operator is independent of spin */
        this->op_ = mdarray<T, 2>(this->packed_mtrx_size_, 1);
        this->op_.zero();

        auto& uc = this->beta_.unit_cell();
        for (int ia = 0; ia < uc.num_atoms(); ia++) {
            int iat = uc.atom(ia).type().id();
            if (!uc.atom_type(iat).pp_desc().augment) {
                continue;
            }
            int nbf = uc.atom(ia).mt_basis_size();
            for (int xi2 = 0; xi2 < nbf; xi2++) {
                for (int xi1 = 0; xi1 < nbf; xi1++) {
                    this->op_(this->packed_mtrx_offset_(ia) + xi2 * nbf + xi1, 0) = -p_mtrx__(xi1, xi2, iat).real();
                }
            }
        }
#ifdef __GPU
        if (this->pu_ == GPU) {
            this->op_.allocate(memory_t::device);
            this->op_.copy_to_device();
        }
#endif
    }
};
}

#endif
