/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file non_local_operator_base.hpp
 *
 *  \brief Contains definition of sirius::Non_local_operator class.
 */

#ifndef __NON_LOCAL_OPERATOR_BASE_HPP__
#define __NON_LOCAL_OPERATOR_BASE_HPP__

#include "context/simulation_context.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "core/traits.hpp"

namespace sirius {

/// Non-local part of the Hamiltonian and S-operator in the pseudopotential method.
template <typename T>
class Non_local_operator
{
  protected:
    Simulation_context const& ctx_;

    device_t pu_;

    int packed_mtrx_size_;

    int size_;

    mdarray<int, 1> packed_mtrx_offset_;

    /// Non-local operator matrix.
    mdarray<T, 3> op_;

    bool is_null_{false};

    /// True if the operator is diagonal in spin.
    bool is_diag_{true};

    /* copy assignment operrator is forbidden */
    Non_local_operator<T>&
    operator=(Non_local_operator<T> const& src) = delete;
    /* copy constructor is forbidden */
    Non_local_operator(Non_local_operator<T> const& src) = delete;

  public:
    /// Constructor.
    Non_local_operator(Simulation_context const& ctx__);

    /// Apply chunk of beta-projectors to all wave functions.
    /** \tparam F  Type of the subspace matrix
     */
    template <typename F>
    void
    apply(memory_t mem__, int chunk__, int ispn_block__, wf::Wave_functions<T>& op_phi__, wf::band_range br__,
          beta_projectors_coeffs_t<T> const& beta_coeffs__, matrix<F> const& beta_phi__) const;

    /// Apply beta projectors from one atom in a chunk of beta projectors to all wave-functions.
    template <typename F>
    std::enable_if_t<std::is_same<std::complex<T>, F>::value, void>
    apply(memory_t mem__, int chunk__, atom_index_t::local ia__, int ispn_block__, wf::Wave_functions<T>& op_phi__,
          wf::band_range br__, beta_projectors_coeffs_t<T> const& beta_coeffs__, matrix<F>& beta_phi__);

    /// computes α B*Q + β out
    template <typename F>
    void
    lmatmul(matrix<F>& out, matrix<F> const& B__, int ispn_block__, memory_t mem_t, identity_t<F> alpha = F{1},
            identity_t<F> beta = F{0}) const;

    /// computes α Q*B + β out
    template <typename F>
    void
    rmatmul(matrix<F>& out, matrix<F> const& B__, int ispn_block__, memory_t mem_t, identity_t<F> alpha = F{1},
            identity_t<F> beta = F{0}) const;

    template <typename F, typename = std::enable_if_t<std::is_same<T, real_type<F>>::value>>
    inline F
    value(int xi1__, int xi2__, int ia__)
    {
        return this->value<F>(xi1__, xi2__, 0, ia__);
    }

    template <typename F, std::enable_if_t<std::is_same<T, F>::value, bool> = true>
    F
    value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__);
    }

    template <typename F, std::enable_if_t<std::is_same<std::complex<T>, F>::value, bool> = true>
    F
    value(int xi1__, int xi2__, int ispn__, int ia__)
    {
        int nbf = this->ctx_.unit_cell().atom(ia__).mt_basis_size();
        return std::complex<T>(this->op_(0, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__),
                               this->op_(1, packed_mtrx_offset_(ia__) + xi2__ * nbf + xi1__, ispn__));
    }

    int
    size(int i) const;

    inline bool
    is_diag() const
    {
        return is_diag_;
    }

    template <typename F>
    matrix<F>
    get_matrix(int ispn, memory_t mem) const;
};

template <class T>
template <class F>
void
Non_local_operator<T>::apply(memory_t mem__, int chunk__, int ispn_block__, wf::Wave_functions<T>& op_phi__,
                             wf::band_range br__, beta_projectors_coeffs_t<T> const& beta_coeffs__,
                             matrix<F> const& beta_phi__) const
{
    PROFILE("sirius::Non_local_operator::apply");

    if (is_null_) {
        return;
    }

    auto& beta_gk     = beta_coeffs__.pw_coeffs_a_;
    int num_gkvec_loc = beta_gk.size(0);
    int nbeta         = beta_coeffs__.beta_chunk_->num_beta_;

    /* setup linear algebra parameters */
    la::lib_t la{la::lib_t::blas};
    device_t pu{device_t::CPU};
    if (is_device_memory(mem__)) {
        la = la::lib_t::gpublas;
        pu = device_t::GPU;
    }

    int size_factor = 1;
    if (std::is_same<F, real_type<F>>::value) {
        size_factor = 2;
    }

    auto work = mdarray<F, 2>({nbeta, br__.size()}, get_memory_pool(mem__));

    /* compute O * <beta|phi> for atoms in a chunk */
    #pragma omp parallel
    {
        acc::set_device_id(mpi::get_device_id(acc::num_devices())); // avoid cuda mth bugs

        #pragma omp for
        for (int i = 0; i < beta_coeffs__.beta_chunk_->num_atoms_; i++) {
            /* number of beta functions for a given atom */
            int nbf  = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::nbf, i);
            int offs = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::offset, i);
            int ia   = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::ia, i);

            if (nbf) {
                la::wrap(la).gemm('N', 'N', nbf, br__.size(), nbf, &la::constant<F>::one(),
                                  reinterpret_cast<F const*>(op_.at(mem__, 0, packed_mtrx_offset_(ia), ispn_block__)),
                                  nbf, reinterpret_cast<F const*>(beta_phi__.at(mem__, offs, 0)), beta_phi__.ld(),
                                  &la::constant<F>::zero(), reinterpret_cast<F*>(work.at(mem__, offs, 0)), nbeta,
                                  acc::stream_id(omp_get_thread_num()));
            }
        }
    }

    auto sp = op_phi__.actual_spin_index(wf::spin_index(ispn_block__ & 1));

    /* compute <G+k|beta> * O * <beta|phi> and add to op_phi */
    la::wrap(la).gemm('N', 'N', num_gkvec_loc * size_factor, br__.size(), nbeta, &la::constant<F>::one(),
                      reinterpret_cast<F const*>(beta_gk.at(mem__)), num_gkvec_loc * size_factor, work.at(mem__), nbeta,
                      &la::constant<F>::one(),
                      reinterpret_cast<F*>(op_phi__.at(mem__, 0, sp, wf::band_index(br__.begin()))),
                      op_phi__.ld() * size_factor);

    switch (pu) {
        case device_t::GPU: {
            acc::sync_stream(acc::stream_id(-1));
            break;
        }
        case device_t::CPU: {
            break;
        }
    }
}

template <class T>
template <class F>
std::enable_if_t<std::is_same<std::complex<T>, F>::value, void>
Non_local_operator<T>::apply(memory_t mem__, int chunk__, atom_index_t::local ia__, int ispn_block__,
                             wf::Wave_functions<T>& op_phi__, wf::band_range br__,
                             beta_projectors_coeffs_t<T> const& beta_coeffs__, matrix<F>& beta_phi__)
{
    if (is_null_) {
        return;
    }

    auto& beta_gk     = beta_coeffs__.pw_coeffs_a_;
    int num_gkvec_loc = beta_gk.size(0);

    int nbf  = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::nbf, ia__);
    int offs = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::offset, ia__);
    int ia   = beta_coeffs__.beta_chunk_->desc_(beta_desc_idx::ia, ia__);

    if (nbf == 0) {
        return;
    }

    la::lib_t la{la::lib_t::blas};
    device_t pu{device_t::CPU};
    if (is_device_memory(mem__)) {
        la = la::lib_t::gpublas;
        pu = device_t::GPU;
    }

    auto work = mdarray<std::complex<T>, 1>({nbf * br__.size()}, get_memory_pool(mem__));

    la::wrap(la).gemm('N', 'N', nbf, br__.size(), nbf, &la::constant<std::complex<T>>::one(),
                      reinterpret_cast<std::complex<T>*>(op_.at(mem__, 0, packed_mtrx_offset_(ia), ispn_block__)), nbf,
                      beta_phi__.at(mem__, offs, 0), beta_phi__.ld(), &la::constant<std::complex<T>>::zero(),
                      work.at(mem__), nbf);

    int jspn = ispn_block__ & 1;

    la::wrap(la).gemm('N', 'N', num_gkvec_loc, br__.size(), nbf, &la::constant<std::complex<T>>::one(),
                      beta_gk.at(mem__, 0, offs), num_gkvec_loc, work.at(mem__), nbf,
                      &la::constant<std::complex<T>>::one(),
                      op_phi__.at(mem__, 0, wf::spin_index(jspn), wf::band_index(br__.begin())), op_phi__.ld());

    switch (pu) {
        case device_t::CPU: {
            break;
        }
        case device_t::GPU: {
#ifdef SIRIUS_GPU
            acc::sync_stream(acc::stream_id(-1));
#endif
            break;
        }
    }
}

template <class T>
template <class F>
void
Non_local_operator<T>::lmatmul(matrix<F>& out, const matrix<F>& B__, int ispn_block__, memory_t mem_t,
                               identity_t<F> alpha, identity_t<F> beta) const
{
    /* Computes Cᵢⱼ =∑ₖ Bᵢₖ Qₖⱼ = Bᵢⱼ Qⱼⱼ
     * Note that Q is block-diagonal. */
    auto& uc = this->ctx_.unit_cell();
    std::vector<int> offsets(uc.num_atoms() + 1, 0);
    for (int ia = 0; ia < uc.num_atoms(); ++ia) {
        offsets[ia + 1] = offsets[ia] + uc.atom(ia).mt_basis_size();
    }

    // check shapes
    RTE_ASSERT(out.size(0) == B__.size(0) && static_cast<int>(out.size(1)) == this->size_);
    RTE_ASSERT(static_cast<int>(B__.size(1)) == this->size_);

    int num_atoms = uc.num_atoms();

    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    for (int ja = 0; ja < num_atoms; ++ja) {
        int offset_ja = offsets[ja];
        int size_ja   = offsets[ja + 1] - offsets[ja];
        // std::printf("\tlmatmul: nbf=%d, offs=%d, ia=%d\n", size_ja, offset_ja, ja);
        const F* Bs = B__.at(mem_t, 0, offset_ja);
        // Qjj
        const F* Qs = reinterpret_cast<const F*>(op_.at(mem_t, 0, packed_mtrx_offset_(ja), ispn_block__));
        F* C        = out.at(mem_t, 0, offset_ja);
        int nbf     = size_ja;
        // compute Bij * Qjj
        la::wrap(la).gemm('N', 'N', B__.size(0), size_ja, size_ja, &alpha, Bs, B__.ld(), Qs, nbf, &beta, C, out.ld());
    }
}

template <class T>
template <class F>
void
Non_local_operator<T>::rmatmul(matrix<F>& out, const matrix<F>& B__, int ispn_block__, memory_t mem_t,
                               identity_t<F> alpha, identity_t<F> beta) const
{
    /* Computes Cᵢⱼ =  ∑ₖ Qᵢₖ * Bₖⱼ = Qᵢᵢ * Bᵢⱼ
     * Note that Q is block-diagonal. */
    auto& uc = this->ctx_.unit_cell();
    std::vector<int> offsets(uc.num_atoms() + 1, 0);
    for (int ia = 0; ia < uc.num_atoms(); ++ia) {
        offsets[ia + 1] = offsets[ia] + uc.atom(ia).mt_basis_size();
    }

    // check shapes
    RTE_ASSERT(static_cast<int>(out.size(0)) == this->size_ && out.size(1) == B__.size(1));
    RTE_ASSERT(static_cast<int>(B__.size(0)) == this->size_);

    int num_atoms = uc.num_atoms();

    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(mem_t)) {
        la = la::lib_t::gpublas;
    }

    for (int ia = 0; ia < num_atoms; ++ia) {
        int offset_ia = offsets[ia];
        int size_ia   = offsets[ia + 1] - offsets[ia];
        const F* Bs   = B__.at(mem_t, offset_ia, 0);
        // Qii
        const F* Qs = reinterpret_cast<const F*>(op_.at(mem_t, 0, packed_mtrx_offset_(ia), ispn_block__));
        F* C        = out.at(mem_t, offset_ia, 0);
        // compute Qii * Bij
        la::wrap(la).gemm('N', 'N', size_ia, B__.size(1), size_ia, &alpha, Qs, size_ia, Bs, B__.ld(), &beta, C,
                          out.ld());
    }
}

} // namespace sirius

#endif /* __NON_LOCAL_OPERATOR_BASE_HPP__ */
