/** \file inverse_overlap.hpp
 *  \brief provides S⁻¹
 */
#ifndef INVERSE_OVERLAP_H
#define INVERSE_OVERLAP_H

#include <iostream>
#include <stdexcept>

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
    Overlap_operator(const Simulation_context& simulation_context, int n)
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
    const Simulation_context& ctx_;
    int n_;
};
} // namespace local

template <class numeric_t>
class InverseS_k : public local::Overlap_operator
{
  public:
    InverseS_k(const Simulation_context& simulation_context, const Q_operator<double>& q_op,
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
    S_k(const Simulation_context& ctx, const Q_operator<double>& q_op, const Beta_projectors_base<double>& bp, int ispn)
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
    auto linalg_t         = ctx_.blas_linalg_t();
    auto preferred_memory = ctx_.preferred_memory_t();

    // auto& beta_projectors = kp.beta_projectors();
    auto B = inner_beta(beta_projectors, ctx_); // on preferred memory

    if (preferred_memory == sddk::memory_t::device) {
        B.allocate(sddk::memory_t::host).copy_to(sddk::memory_t::host);
    }

    sddk::matrix<numeric_t> BQ(B.size(0), q_op.size(1));

    if (ctx_.processing_unit() == sddk::device_t::GPU) {
        BQ.allocate(sddk::memory_t::device);
    }
    // mat * Q
    q_op.lmatmul(BQ, B, this->ispn, preferred_memory);
    int n = BQ.size(0);

    if (is_device_memory(preferred_memory)) {
        BQ.allocate(sddk::memory_t::host);
        BQ.copy_to(sddk::memory_t::host);
        BQ.deallocate(sddk::memory_t::device);
    }
    // add identity matrix
    std::vector<double_complex> ones(n, double_complex{1, 0});
    sddk::linalg(sddk::linalg_t::blas)
        .axpy(n, &sddk::linalg_const<double_complex>::one(), ones.data(), 1, BQ.at(sddk::memory_t::host), n + 1);

    LU = sddk::empty_like(BQ);
    sddk::copy_auto(LU, BQ, sddk::device_t::CPU);
    // compute inverse...
    ipiv = sddk::mdarray<int, 1>(n);
    // compute LU factorization, TODO: use GPU if needed
    sddk::linalg(sddk::linalg_t::lapack)
        .getrf(n, n, LU.at(sddk::memory_t::host), LU.ld(), ipiv.at(sddk::memory_t::host));

    // copy LU factorization to device if needed
    // LU is always computed on host
    // auto mem = ctx_.preferred_memory_t();
    // if(is_device_memory(mem)) {
    //     ipiv.allocate(mem);
    //     ipiv.copy_to(mem);

    //     LU.allocate(mem);
    //     LU.copy_to(mem);
    // }
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

    pm                = (pm == sddk::memory_t::none) ? ctx_.preferred_memory_t() : pm;
    sddk::device_t pu = is_host_memory(pm) ? sddk::device_t::CPU : sddk::device_t::GPU;

    sddk::linalg_t la{sddk::linalg_t::none};
    switch (pu) {
        case sddk::device_t::CPU: {
            la = sddk::linalg_t::blas;
            break;
        }
        case sddk::device_t::GPU: {
            la = sddk::linalg_t::gpublas;
            break;
        }
    }

    auto bp_gen      = bp.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();

    int num_beta = bp.num_total_beta();

    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd);
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);

        auto bphi_loc = inner<numeric_t>(
            la, pu, pm, [&ctx = ctx_](sddk::device_t dev) -> sddk::memory_pool& { return ctx.mem_pool(dev); },
            beta_coeffs, X, 0, nbnd);

        // copy submatrix to bphi
        int beta_offset = beta_coeffs.beta_chunk.offset_;
#pragma omp parallel for
        for (int lbnd = 0; lbnd < nbnd; ++lbnd) {
            // issue copy operation
            sddk::copy(sddk::memory_t::host, bphi_loc.at(sddk::memory_t::host, 0, lbnd), sddk::memory_t::host,
                       bphi.at(sddk::memory_t::host, beta_offset, lbnd), bphi_loc.size(0));
        }
    }

    // compute bphi <- (I + B*Q)⁻¹ (B^H X)
    sddk::linalg(sddk::linalg_t::lapack)
        .getrs('N', num_beta, nbnd, LU.at(sddk::memory_t::host), LU.ld(), ipiv.at(sddk::memory_t::host),
               bphi.at(sddk::memory_t::host), bphi.ld());

    // compute R <- -Q * Z, where Z = (I + B*Q)⁻¹ (B^H X)
    sddk::matrix<numeric_t> R(q_op.size(0), bphi.size(1));

    // allocate bphi on gpu if needed
    if (pm == sddk::memory_t::device) {
        bphi.allocate(ctx_.mem_pool(sddk::memory_t::device));
        bphi.copy_to(sddk::memory_t::device);
        R.allocate(sddk::memory_t::device);
    }

    // compute -Q*bphi
    q_op.rmatmul(R, bphi, this->ispn, pm, -1);

    sddk::copy_new(Y, X, pu);

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a.size(1);

        sddk::linalg(la).gemm('N', 'N', m, n, k, &sddk::linalg_const<numeric_t>::one(), beta_coeffs.pw_coeffs_a.at(pm),
                              beta_coeffs.pw_coeffs_a.ld(), R.at(pm, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                              &sddk::linalg_const<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

/// apply wfct
/// computes (X + Beta*P*Beta^H*X)
/// where P = -Q*(I + B*Q)⁻¹
template <class numeric_t>
sddk::mdarray<numeric_t, 2>
InverseS_k<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    auto Y = sddk::empty_like(X, ctx_.mem_pool(pm == sddk::memory_t::none ? ctx_.preferred_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

template <class numeric_t>
void
S_k<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    assert(X.size(0) == this->size());

    pm          = (pm == memory_t::none) ? ctx_.preferred_memory_t() : pm;
    device_t pu = is_host_memory(pm) ? device_t::CPU : device_t::GPU;

    int nbnd         = X.size(1);
    auto bp_gen      = bp.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();
    int num_beta     = bp.num_total_beta();

    sddk::linalg_t la{sddk::linalg_t::none};
    switch (pu) {
        case sddk::device_t::CPU: {
            la = sddk::linalg_t::blas;
            break;
        }
        case sddk::device_t::GPU: {
            la = sddk::linalg_t::gpublas;
            break;
        }
    }

    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd);
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);

        auto bphi_loc = inner<numeric_t>(
            la, pu, pm, [&ctx = ctx_](sddk::device_t dev) -> sddk::memory_pool& { return ctx.mem_pool(dev); },
            beta_coeffs, X, 0, nbnd);

        // copy submatrix to bphi
        int beta_offset = beta_coeffs.beta_chunk.offset_;
        // std::printf("* apply_overlap: ichunk=%d,  beta_offset: %d\n", ichunk, beta_offset);
#pragma omp parallel for
        for (int lbnd = 0; lbnd < nbnd; ++lbnd) {
            // issue copy operation
            sddk::copy(sddk::memory_t::host, bphi_loc.at(sddk::memory_t::host, 0, lbnd), sddk::memory_t::host,
                       bphi.at(sddk::memory_t::host, beta_offset, lbnd), bphi_loc.size(0));
        }
    }

    sddk::matrix<numeric_t> R(q_op.size(0), bphi.size(1));
    // allocate bphi on gpu if needed
    if (pm == sddk::memory_t::device) {
        bphi.allocate(ctx_.mem_pool(sddk::memory_t::device));
        bphi.copy_to(sddk::memory_t::device);
        R.allocate(sddk::memory_t::device);
    }

    q_op.rmatmul(R, bphi, this->ispn, pm, 1, 0);

    sddk::copy_new(Y, X, pu);

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        // std::cout << "* ichunk: " << ichunk << "\n";
        bp_gen.generate(beta_coeffs, ichunk);
        int m = Y.size(0);
        int n = Y.size(1);
        int k = beta_coeffs.pw_coeffs_a.size(1);

        sddk::linalg(la).gemm('N', 'N', m, n, k, &sddk::linalg_const<numeric_t>::one(), beta_coeffs.pw_coeffs_a.at(pm),
                              beta_coeffs.pw_coeffs_a.ld(), R.at(pm, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                              &sddk::linalg_const<numeric_t>::one(), Y.at(pm), Y.ld());
    }
}

template <class numeric_t>
sddk::mdarray<numeric_t, 2>
S_k<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, sddk::memory_t pm)
{
    auto Y = sddk::empty_like(X, ctx_.mem_pool(pm == sddk::memory_t::none ? ctx_.preferred_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

} // namespace sirius

#endif /* INVERSE_OVERLAP_H */
