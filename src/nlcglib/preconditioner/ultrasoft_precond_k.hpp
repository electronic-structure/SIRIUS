/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file ultrasoft_precond.hpp
 *
 *  \brief Provides preconditioner for ultrasoft case.
 */

#ifndef __ULTRASOFT_PRECOND_K_HPP__
#define __ULTRASOFT_PRECOND_K_HPP__

#include "context/simulation_context.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "core/fft/gvec.hpp"
#include "diag_mm.hpp"

namespace sirius {

namespace local {

class OperatorBase
{
  public:
    OperatorBase(int n)
        : n(n){};
    OperatorBase() = delete;

    int
    size() const
    {
        return n;
    }

  private:
    int n;
};

} // namespace local

template <class numeric_t>
class DiagonalPreconditioner
{
  public:
    DiagonalPreconditioner(Simulation_context const& ctx)
        // : ctx_(ctx)
    {
    }
    mdarray<numeric_t, 2>
    apply(const mdarray<numeric_t, 2>& X, memory_t pm);
    void
    apply(mdarray<numeric_t, 2>& Y, const mdarray<numeric_t, 2>& X, memory_t pm);

  protected:
    mdarray<numeric_t, 1> d_;
    // Simulation_context& ctx_;
};

template <class numeric_t>
mdarray<numeric_t, 2>
DiagonalPreconditioner<numeric_t>::apply(const mdarray<numeric_t, 2>& X, memory_t pm)
{
    auto Y = empty_like(X, get_memory_pool(pm));
    this->apply(Y, X, pm);
    return Y;
}

/// computes Y <- P*X
template <class numeric_t>
inline void
DiagonalPreconditioner<numeric_t>::apply(mdarray<numeric_t, 2>& Y, const mdarray<numeric_t, 2>& X, memory_t pm)
{
#ifdef SIRIUS_GPU
    // copy d_ to gpu
    if (is_device_memory(pm)) {
        d_.allocate(memory_t::device);
        d_.copy_to(memory_t::device);
        int n = d_.size(0);
        zdiagmm(d_.at(memory_t::device), n, X.at(memory_t::device), X.ld(), X.size(1), Y.at(memory_t::device), Y.ld(),
                std::complex<double>{1});
        return;
    }
#endif /*SIRIUS_GPU*/

    int n = X.size(0);
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        numeric_t d = d_(i);
        for (int j = 0; j < static_cast<int>(X.size(1)); ++j) {
            Y(i, j) = d * X(i, j);
        }
    }
    return;
}

/** Payne, M. C., Teter, M. P., Allan, D. C., Arias, T. A., & Joannopoulos, J.
 *  D., Iterative minimization techniques for ab initio total-energy
 *  calculations: molecular dynamics and conjugate gradients.
 *  https://dx.doi.org/10.1103/RevModPhys.64.1045
 */
template <class numeric_t>
class Teter : DiagonalPreconditioner<numeric_t>, public local::OperatorBase
{
  public:
    Teter(Simulation_context const& ctx, fft::Gvec const& gkvec)
        : DiagonalPreconditioner<numeric_t>(ctx)
        , local::OperatorBase(gkvec.count())
    {
        this->d_ = mdarray<numeric_t, 1>({gkvec.count()});
        for (int i = 0; i < gkvec.count(); ++i) {
            // teter formula
            double T  = gkvec.gkvec_cart(gvec_index_t::global(i)).length2();
            double T2 = T * T;
            double T3 = T2 * T;
            double T4 = T2 * T2;
            double Tp = 16 * T4 / (27 + 18 * T + 12 * T2 + 8 * T3);
            // Eq. (5.16) in Payne et. al
            this->d_(i) = 1 / (1 + Tp);
        }
    }

    using DiagonalPreconditioner<numeric_t>::apply;
};

/** Ultrasoft preconditioner for direct minimization.
 *
 *  (1+T)⁻¹ + G R G⊹
 *  where R = -Q (1 + C Q)⁻¹
 *  and G  are the "preconditioned" beta projectors, C = B⊹ K B
 *  TODO: what is K?
 *
 * Hasnip, P. J., & Pickard, C. J. (). Electronic energy minimisation with
 * ultrasoft pseudopotentials. , 174(1), 24–29.
 * http://dx.doi.org/10.1016/j.cpc.2005.07.011
 */
template <class numeric_t>
class Ultrasoft_preconditioner : public local::OperatorBase
{
  public:
    Ultrasoft_preconditioner(Simulation_context const& simulation_context, std::shared_ptr<Q_operator<double> const> q_op,
                             int ispn, std::shared_ptr<Beta_projectors_base<double> const> bp, const fft::Gvec& gkvec);

    mdarray<numeric_t, 2>
    apply(const mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);
    void
    apply(mdarray<numeric_t, 2>& Y, const mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);

    // const Simulation_context&
    // ctx() const
    // {
    //     return ctx_;
    // }

  private:
    // cannot be const, because memory pool is used
    // Simulation_context& ctx_;
    memory_t pm_;
    Teter<numeric_t> P;
    std::shared_ptr<Q_operator<double> const> q_op;
    int ispn_;
    std::shared_ptr<Beta_projectors_base<double> const> bp_;
    mdarray<int, 1> ipiv_;
    mdarray<numeric_t, 2> LU_;
};

template <class numeric_t>
Ultrasoft_preconditioner<numeric_t>::Ultrasoft_preconditioner(Simulation_context const& simulation_context,
                                                              std::shared_ptr<Q_operator<double> const> q_op, int ispn,
                                                              std::shared_ptr<Beta_projectors_base<double> const> bp,
                                                              const fft::Gvec& gkvec)
    : local::OperatorBase(gkvec.count())
    // , ctx_(simulation_context)
    ,  pm_(simulation_context.processing_unit_memory_t())
    , P(simulation_context, gkvec)
    , q_op(q_op)
    , ispn_(ispn)
    , bp_(bp)
{
    using complex_t = std::complex<double>;
    /* compute C <- <ϐ|P|ϐ> */
    auto C = inner_beta(this->pm_, *bp,
                        [pm=this->pm_, &P = this->P](auto& Y) {
                            return P.apply(Y, pm);
                        });

    matrix<numeric_t> CQ({C.size(0), q_op->size(1)}, memory_t::host);
    if (is_device_memory(this->pm_)) {
        C.allocate(memory_t::host);
        C.copy_to(memory_t::host);
    }
    /* compute C <- C@Q */
    this->q_op->lmatmul(CQ, C, this->ispn_, memory_t::host);
    /* compute C <- 1 + C */
    int n = CQ.size(0);
    // add identiy matrix
    std::vector<complex_t> ones(n, 1);
    // add identity matrix
    la::wrap(la::lib_t::blas).axpy(n, &la::constant<complex_t>::one(), ones.data(), 1, CQ.at(memory_t::host), n + 1);
    // compute LU factorization
    this->LU_ = empty_like(CQ);
    auto_copy(this->LU_, CQ);
    this->ipiv_ = mdarray<int, 1>({n}, memory_t::host);
    // compute LU factorization
    la::wrap(la::lib_t::lapack)
            .getrf(n, n, this->LU_.at(memory_t::host), this->LU_.ld(), this->ipiv_.at(memory_t::host));
    // copy LU factorization to device if needed
    if (is_device_memory(this->pm_)) {
        ipiv_.allocate(this->pm_);
        ipiv_.copy_to(this->pm_);

        LU_.allocate(this->pm_);
        LU_.copy_to(this->pm_);
    }
}

template <class numeric_t>
mdarray<numeric_t, 2>
Ultrasoft_preconditioner<numeric_t>::apply(const mdarray<numeric_t, 2>& X, memory_t pm)
{
    auto Y = empty_like(X, get_memory_pool(pm == memory_t::none ? this->pm_ : pm));
    this->apply(Y, X, pm);
    return Y;
}

template <class numeric_t>
void
Ultrasoft_preconditioner<numeric_t>::apply(mdarray<numeric_t, 2>& Y, const mdarray<numeric_t, 2>& X, memory_t pm)
{
    int num_beta = bp_->num_beta();
    int nbnd     = X.size(1);

    pm          = (pm == memory_t::none) ? this->pm_ : pm;
    device_t pu = is_host_memory(pm) ? device_t::CPU : device_t::GPU;

    la::lib_t la{la::lib_t::blas};
    if (is_device_memory(pm)) {
        la = la::lib_t::gpublas;
    }

    auto bp_gen      = bp_->make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();

    mdarray<numeric_t, 2> bphi({num_beta, nbnd}, get_memory_pool(pm));
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp_->num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors
        auto G         = P.apply(beta_coeffs.pw_coeffs_a_, pm);
        int row_offset = beta_coeffs.beta_chunk_->offset_;

        la::wrap(la).gemm('C', 'N', G.size(1), nbnd, G.size(0), &la::constant<numeric_t>::one(), G.at(pm), G.ld(),
                          X.at(pm), X.ld(), &la::constant<numeric_t>::zero(), bphi.at(pm, row_offset, 0), bphi.ld());
    }
    if (bp_->comm().size() > 1) {
#ifdef SIRIUS_CUDA
        cudaDeviceSynchronize();
#endif
#ifdef SIRIUS_ROCM
        hipDeviceSynchronize();
#endif
        bp_->comm().allreduce(bphi.at(pm), bphi.size());
    }

    assert(num_beta == static_cast<int>(bphi.size(0)) && nbnd == static_cast<int>(bphi.size(1)));

    la::lib_t lapack{la::lib_t::lapack};
    if (pu == device_t::GPU) {
        lapack = la::lib_t::gpublas;
    }
    la::wrap(lapack).getrs('N', num_beta, nbnd, LU_.at(pm), LU_.ld(), ipiv_.at(pm), bphi.at(pm), bphi.ld());

    auto R = empty_like(bphi, get_memory_pool(pm));
    q_op->rmatmul(R, bphi, ispn_, pm, -1);

    // compute Y <- (1+T')^(-1) X
    this->P.apply(Y, X, pm);

    for (int ichunk = 0; ichunk < bp_->num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors in place
        auto G = P.apply(beta_coeffs.pw_coeffs_a_, pm);
        int m  = Y.size(0);
        int n  = Y.size(1);
        int k  = beta_coeffs.pw_coeffs_a_.size(1);

        switch (pu) {
            case device_t::CPU: {
                la::wrap(la::lib_t::blas)
                        .gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), G.at(memory_t::host), G.ld(),
                              R.at(memory_t::host, beta_coeffs.beta_chunk_->offset_, 0), R.ld(),
                              &la::constant<numeric_t>::one(), Y.at(memory_t::host), Y.ld());
                break;
            }
#ifdef SIRIUS_GPU
            case device_t::GPU:
                la::wrap(la::lib_t::gpublas)
                        .gemm('N', 'N', m, n, k, &la::constant<numeric_t>::one(), G.at(memory_t::device), G.ld(),
                              R.at(memory_t::device, beta_coeffs.beta_chunk_->offset_, 0), R.ld(),
                              &la::constant<numeric_t>::one(), Y.at(memory_t::device), Y.ld());

                break;
#endif
            default:
                RTE_THROW("invalid processing unit");
                break;
        }
    }
}
} // namespace sirius

#endif /* __ULTRASOFT_PRECOND_K_HPP__ */
