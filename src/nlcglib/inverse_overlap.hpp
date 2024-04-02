/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file inverse_overlap.hpp
 *
 *  \brief provides S⁻¹
 */

#ifndef __INVERSE_OVERLAP_HPP__
#define __INVERSE_OVERLAP_HPP__

#include <iostream>
#include <spla/matrix_distribution.hpp>
#include <spla/types.h>
#include <stdexcept>

#include "core/la/linalg_base.hpp"
#include "core/memory.hpp"
#include "core/mpi/communicator.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "context/simulation_context.hpp"
#include "k_point/k_point.hpp"
#include "beta_projectors/beta_projectors.hpp"
#include "memory.h"

namespace sirius {

namespace local {

class Overlap_operator
{
  public:
    Overlap_operator(Simulation_context& simulation_context, int n)
        : ctx_(simulation_context)
        , n_(n)
    {
    }

    const Simulation_context&
    ctx() const
    {
        return ctx_;
    }

    /// global dimension of the operator
    int
    size() const
    {
        return n_;
    }

  protected:
    Simulation_context& ctx_;
    int n_;
};

/// computes C <- A.H x B
template <class T>
void
inner(memory_t mem, spla::Context& ctx, const mdarray<T, 2>& A, const mdarray<T, 2>& B, mdarray<T, 2>& C,
      const mpi::Communicator& comm, int row_offset = 0, int col_offset = 0)
{
    auto spla_mat_dist = spla::MatrixDistribution::create_mirror(comm.native());
    int m              = A.size(1);
    int n              = B.size(1);
    int k              = B.size(0);

    const T* A_ptr{nullptr};
    const T* B_ptr{nullptr};
    T* C_ptr = C.host_data();
    if (is_device_memory(mem)) {
        A_ptr = A.device_data();
        B_ptr = B.device_data();
    } else {
        A_ptr = A.host_data();
        B_ptr = B.host_data();
    }
    int cRowOffset = row_offset;
    int cColOffset = col_offset;
    spla::pgemm_ssb(m, n, k, SPLA_OP_CONJ_TRANSPOSE, T{1.0}, A_ptr, A.ld(), B_ptr, B.ld(), T{0.0}, C_ptr, C.ld(),
                    cRowOffset, cColOffset, spla_mat_dist, ctx);
}
} // namespace local

/// Ref: 10.1016/j.cpc.2005.07.011
/// Electronic energy minimisation with ultrasoft pseudopotentials
/// Hasnip & Pickard
template <class numeric_t>
class InverseS_k : public local::Overlap_operator
{
  public:
    InverseS_k(Simulation_context& simulation_context, const Q_operator<double>& q_op,
               const Beta_projectors_base<double>& bp, int ispn)
        : Overlap_operator(simulation_context, bp.nrows())
        , q_op_(q_op)
        , bp_(bp)
        , ispn_(ispn)
    {
        initialize(bp);
    }

    mdarray<numeric_t, 2>
    apply(const mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);

    void
    apply(mdarray<numeric_t, 2>& Y, const mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);

    const std::string label{"inverse overlap"};

  private:
    void
    initialize(const Beta_projectors_base<double>& bp);
    const Q_operator<double>& q_op_;
    const Beta_projectors_base<double>& bp_;
    const int ispn_;

    mdarray<numeric_t, 2> LU_;
    mdarray<int, 1> ipiv_;
};

template <class numeric_t>
class S_k : public local::Overlap_operator
{
  public:
    S_k(Simulation_context& ctx, const Q_operator<double>& q_op, const Beta_projectors_base<double>& bp, int ispn)
        : Overlap_operator(ctx, bp.nrows())
        , q_op_(q_op)
        , bp_(bp)
        , ispn_(ispn)
    { /* empty */
    }

    mdarray<numeric_t, 2>
    apply(mdarray<numeric_t, 2> const& X, memory_t pu = memory_t::none);
    void
    apply(mdarray<numeric_t, 2>& Y, mdarray<numeric_t, 2> const& X, memory_t pm = memory_t::none);

    const std::string label{"overlap"};

  private:
    Q_operator<double> const& q_op_;
    Beta_projectors_base<double> const& bp_;
    const int ispn_;
};

template <class numeric_t>
void
InverseS_k<numeric_t>::initialize(Beta_projectors_base<double> const& beta_projectors)
{
    using complex_t = std::complex<double>;
    auto mem_t      = ctx_.processing_unit_memory_t();

    auto B = inner_beta(beta_projectors, ctx_); // on preferred memory

    matrix<numeric_t> BQ({B.size(0), q_op_.size(1)}, mem_t);
    // mat * Q
    q_op_.lmatmul(BQ, B, this->ispn_, mem_t);
    int n = BQ.size(0);

    if (is_device_memory(mem_t)) {
        BQ.allocate(memory_t::host);
        BQ.copy_to(memory_t::host);
        BQ.deallocate(memory_t::device);
    }
    // add identity matrix
    std::vector<complex_t> ones(n, complex_t{1, 0});
    la::wrap(la::lib_t::blas).axpy(n, &la::constant<complex_t>::one(), ones.data(), 1, BQ.at(memory_t::host), n + 1);

    LU_ = empty_like(BQ, get_memory_pool(memory_t::host));
    auto_copy(LU_, BQ, device_t::CPU);
    // compute inverse...
    ipiv_ = mdarray<int, 1>({n});
    // compute LU factorization, TODO: use GPU if needed
    la::wrap(la::lib_t::lapack).getrf(n, n, LU_.at(memory_t::host), LU_.ld(), ipiv_.at(memory_t::host));
}

/// apply wfct
/// computes (X + Beta*P*Beta^H*X)
/// where P = -Q*(I + B*Q)⁻¹
template <class numeric_t>
void
InverseS_k<numeric_t>::apply(mdarray<numeric_t, 2>& Y, mdarray<numeric_t, 2> const& X, memory_t pm)
{
    int nbnd = X.size(1);
    assert(static_cast<int>(X.size(0)) == this->size());
    pm          = (pm == memory_t::none) ? ctx_.processing_unit_memory_t() : pm;
    device_t pu = is_host_memory(pm) ? device_t::CPU : device_t::GPU;
    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(pm)) {
        la = la::lib_t::gpublas;
    }

    auto bp_gen      = bp_.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();

    int num_beta = bp_.num_beta();

    mdarray<numeric_t, 2> bphi({num_beta, nbnd});
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp_.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);

        local::inner(pm, ctx_.spla_context(), beta_coeffs.pw_coeffs_a_, X, bphi, beta_coeffs.comm_,
                     beta_coeffs.beta_chunk_->offset_, 0);
    }

    // compute bphi <- (I + B*Q)⁻¹ (B^H X)
    la::wrap(la::lib_t::lapack)
            .getrs('N', num_beta, nbnd, LU_.at(memory_t::host), LU_.ld(), ipiv_.at(memory_t::host),
                   bphi.at(memory_t::host), bphi.ld());

    // compute R <- -Q * Z, where Z = (I + B*Q)⁻¹ (B^H X)
    matrix<numeric_t> R({q_op_.size(0), bphi.size(1)});

    // allocate bphi on gpu if needed
    if (pm == memory_t::device) {
        bphi.allocate(get_memory_pool(memory_t::device));
        bphi.copy_to(memory_t::device);
        R.allocate(memory_t::device);
    }

    // compute -Q*bphi
    q_op_.rmatmul(R, bphi, this->ispn_, pm, -1);

    auto_copy(Y, X, pu);

    for (int ichunk = 0; ichunk < bp_.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a_.size(1);

        la::wrap(la).gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), beta_coeffs.pw_coeffs_a_.at(pm),
                          beta_coeffs.pw_coeffs_a_.ld(), R.at(pm, beta_coeffs.beta_chunk_->offset_, 0), R.ld(),
                          &la::constant<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

/// apply wfct
/// computes (X + Beta*P*Beta^H*X)
/// where P = -Q*(I + B*Q)⁻¹
template <class numeric_t>
mdarray<numeric_t, 2>
InverseS_k<numeric_t>::apply(mdarray<numeric_t, 2> const& X, memory_t pm)
{
    auto Y = empty_like(X, get_memory_pool(pm == memory_t::none ? ctx_.processing_unit_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

template <class numeric_t>
void
S_k<numeric_t>::apply(mdarray<numeric_t, 2>& Y, mdarray<numeric_t, 2> const& X, memory_t pm)
{
    assert(static_cast<int>(X.size(0)) == this->size());

    pm          = (pm == memory_t::none) ? ctx_.processing_unit_memory_t() : pm;
    device_t pu = is_host_memory(pm) ? device_t::CPU : device_t::GPU;
    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(pm)) {
        la = la::lib_t::gpublas;
    }

    int nbnd         = X.size(1);
    auto bp_gen      = bp_.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();
    int num_beta     = bp_.num_beta();

    mdarray<numeric_t, 2> bphi({num_beta, nbnd});
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp_.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        local::inner(pm, ctx_.spla_context(), beta_coeffs.pw_coeffs_a_, X, bphi, beta_coeffs.comm_,
                     beta_coeffs.beta_chunk_->offset_, 0);
    }

    matrix<numeric_t> R({q_op_.size(0), bphi.size(1)});
    // allocate bphi on gpu if needed
    if (pm == memory_t::device) {
        bphi.allocate(get_memory_pool(memory_t::device));
        bphi.copy_to(memory_t::device);
        R.allocate(memory_t::device);
    }

    q_op_.rmatmul(R, bphi, this->ispn_, pm, 1.0, 0.0);

    auto_copy(Y, X, pu);

    for (int ichunk = 0; ichunk < bp_.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a_.size(1);

        la::wrap(la).gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), beta_coeffs.pw_coeffs_a_.at(pm),
                          beta_coeffs.pw_coeffs_a_.ld(), R.at(pm, beta_coeffs.beta_chunk_->offset_, 0), R.ld(),
                          &la::constant<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

template <class numeric_t>
mdarray<numeric_t, 2>
S_k<numeric_t>::apply(mdarray<numeric_t, 2> const& X, memory_t pm)
{
    auto Y = empty_like(X, get_memory_pool(pm == memory_t::none ? ctx_.processing_unit_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

} // namespace sirius

#endif /* __INVERSE_OVERLAP_HPP__ */
