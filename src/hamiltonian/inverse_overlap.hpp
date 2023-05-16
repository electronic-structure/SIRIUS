// Copyright (c) 2023 Simon Pintarelli, Anton Kozhevnikov, Thomas Schulthess
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

#include "linalg/linalg_base.hpp"
#include "memory.hpp"
#include "mpi/communicator.hpp"
#include "non_local_operator.hpp"
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

    const Simulation_context& ctx() const
    {
        return ctx_;
    }

    /// global dimension of the operator
    int size() const
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
inner(sddk::memory_t mem, spla::Context& ctx, const sddk::mdarray<T, 2>& A, const sddk::mdarray<T, 2>& B,
      sddk::mdarray<T, 2>& C, const mpi::Communicator& comm, int row_offset = 0, int col_offset = 0)
{
    auto spla_mat_dist = spla::MatrixDistribution::create_mirror(comm.native());
    int m              = A.size(1);
    int n              = B.size(1);
    int k              = B.size(0);

    const T* A_ptr{nullptr};
    const T* B_ptr{nullptr};
    T* C_ptr = C.host_data();
    if (sddk::is_device_memory(mem)) {
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
        , q_op(q_op)
        , bp(bp)
        , ispn(ispn)
    {
        initialize(bp);
    }

    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm = sddk::memory_t::none);

    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X,
               sddk::memory_t pm = sddk::memory_t::none);

    const std::string label{"inverse overlap"};

  private:
    void initialize(const Beta_projectors_base<double>& bp);
    const Q_operator<double>& q_op;
    const Beta_projectors_base<double>& bp;
    const int ispn;

    sddk::mdarray<numeric_t, 2> LU;
    sddk::mdarray<int, 1> ipiv;
};

template <class numeric_t>
class S_k : public local::Overlap_operator
{
  public:
    S_k(Simulation_context& ctx, const Q_operator<double>& q_op, const Beta_projectors_base<double>& bp, int ispn)
        : Overlap_operator(ctx, bp.nrows())
        , q_op(q_op)
        , bp(bp)
        , ispn(ispn)
    { /* empty */
    }

    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pu = sddk::memory_t::none);
    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X,
               sddk::memory_t pm = sddk::memory_t::none);

    const std::string label{"overlap"};

  private:
    const Q_operator<double>& q_op;
    const Beta_projectors_base<double>& bp;
    const int ispn;
};

template <class numeric_t>
void
InverseS_k<numeric_t>::initialize(const Beta_projectors_base<double>& beta_projectors)
{
    using complex_t = std::complex<double>;
    auto mem_t      = ctx_.processing_unit_memory_t();

    auto B = inner_beta(beta_projectors, ctx_); // on preferred memory

    sddk::matrix<numeric_t> BQ(B.size(0), q_op.size(1), mem_t);
    // mat * Q
    q_op.lmatmul(BQ, B, this->ispn, mem_t);
    int n = BQ.size(0);

    if (is_device_memory(mem_t)) {
        BQ.allocate(sddk::memory_t::host);
        BQ.copy_to(sddk::memory_t::host);
        BQ.deallocate(sddk::memory_t::device);
    }
    // add identity matrix
    std::vector<complex_t> ones(n, complex_t{1, 0});
    la::wrap(la::lib_t::blas)
        .axpy(n, &la::constant<complex_t>::one(), ones.data(), 1, BQ.at(sddk::memory_t::host), n + 1);

    LU = sddk::empty_like(BQ, sddk::get_memory_pool(sddk::memory_t::host));
    sddk::auto_copy(LU, BQ, sddk::device_t::CPU);
    // compute inverse...
    ipiv = sddk::mdarray<int, 1>(n);
    // compute LU factorization, TODO: use GPU if needed
    la::wrap(la::lib_t::lapack).getrf(n, n, LU.at(sddk::memory_t::host), LU.ld(), ipiv.at(sddk::memory_t::host));
}

/// apply wfct
/// computes (X + Beta*P*Beta^H*X)
/// where P = -Q*(I + B*Q)⁻¹
template <class numeric_t>
void
InverseS_k<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    int nbnd = X.size(1);
    assert(X.size(0) == this->size());
    pm                = (pm == sddk::memory_t::none) ? ctx_.processing_unit_memory_t() : pm;
    sddk::device_t pu = is_host_memory(pm) ? sddk::device_t::CPU : sddk::device_t::GPU;
    la::lib_t la{la::lib_t::blas};
    if (sddk::is_device_memory(pm)) {
        la = la::lib_t::gpublas;
    }

    auto bp_gen      = bp.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();

    int num_beta = bp.num_total_beta();

    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd);
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);

        local::inner(pm, ctx_.spla_context(), beta_coeffs.pw_coeffs_a, X, bphi, beta_coeffs.communicator,
                     beta_coeffs.beta_chunk.offset_, 0);
    }

    // compute bphi <- (I + B*Q)⁻¹ (B^H X)
    la::wrap(la::lib_t::lapack)
        .getrs('N', num_beta, nbnd, LU.at(sddk::memory_t::host), LU.ld(), ipiv.at(sddk::memory_t::host),
               bphi.at(sddk::memory_t::host), bphi.ld());

    // compute R <- -Q * Z, where Z = (I + B*Q)⁻¹ (B^H X)
    sddk::matrix<numeric_t> R(q_op.size(0), bphi.size(1));

    // allocate bphi on gpu if needed
    if (pm == sddk::memory_t::device) {
        bphi.allocate(sddk::get_memory_pool(sddk::memory_t::device));
        bphi.copy_to(sddk::memory_t::device);
        R.allocate(sddk::memory_t::device);
    }

    // compute -Q*bphi
    q_op.rmatmul(R, bphi, this->ispn, pm, -1);

    sddk::auto_copy(Y, X, pu);

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a.size(1);

        la::wrap(la).gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), beta_coeffs.pw_coeffs_a.at(pm),
                          beta_coeffs.pw_coeffs_a.ld(), R.at(pm, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &la::constant<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

/// apply wfct
/// computes (X + Beta*P*Beta^H*X)
/// where P = -Q*(I + B*Q)⁻¹
template <class numeric_t>
sddk::mdarray<numeric_t, 2>
InverseS_k<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    auto Y =
        sddk::empty_like(X, sddk::get_memory_pool(pm == sddk::memory_t::none ? ctx_.processing_unit_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

template <class numeric_t>
void
S_k<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    assert(X.size(0) == this->size());

    pm                = (pm == sddk::memory_t::none) ? ctx_.processing_unit_memory_t() : pm;
    sddk::device_t pu = is_host_memory(pm) ? sddk::device_t::CPU : sddk::device_t::GPU;
    la::lib_t la{la::lib_t::blas};
    if (sddk::is_device_memory(pm)) {
        la = la::lib_t::gpublas;
    }

    int nbnd         = X.size(1);
    auto bp_gen      = bp.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();
    int num_beta     = bp.num_total_beta();

    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd);
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        local::inner(pm, ctx_.spla_context(), beta_coeffs.pw_coeffs_a, X, bphi, beta_coeffs.communicator,
                     beta_coeffs.beta_chunk.offset_, 0);
    }

    sddk::matrix<numeric_t> R(q_op.size(0), bphi.size(1));
    // allocate bphi on gpu if needed
    if (pm == sddk::memory_t::device) {
        bphi.allocate(sddk::get_memory_pool(sddk::memory_t::device));
        bphi.copy_to(sddk::memory_t::device);
        R.allocate(sddk::memory_t::device);
    }

    q_op.rmatmul(R, bphi, this->ispn, pm, 1, 0);

    sddk::auto_copy(Y, X, pu);

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a.size(1);

        la::wrap(la).gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), beta_coeffs.pw_coeffs_a.at(pm),
                          beta_coeffs.pw_coeffs_a.ld(), R.at(pm, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &la::constant<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

template <class numeric_t>
sddk::mdarray<numeric_t, 2>
S_k<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    auto Y =
        sddk::empty_like(X, sddk::get_memory_pool(pm == sddk::memory_t::none ? ctx_.processing_unit_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

} // namespace sirius

#endif /* __INVERSE_OVERLAP_HPP__ */
