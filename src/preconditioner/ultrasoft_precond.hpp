/** \file ultrasoft_precond.hpp
    \brief Provides preconditioner for ultrasoft case.
 */

#include "context/simulation_context.hpp"
#include "hamiltonian/non_local_operator.hpp"
#include "beta_projectors/beta_projectors_base.hpp"
#include "SDDK/memory.hpp"
#include "SDDK/gvec.hpp"
#include "diag_mm.hpp"

namespace sirius {

namespace local {

class OperatorBase
{
  public:
    OperatorBase(int n)
        : n(n){};
    OperatorBase() = delete;

    int size() const
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
    DiagonalPreconditioner(Simulation_context& ctx)
        : ctx_(ctx)
    {
    }
    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X, device_t processing_unit);
    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, device_t processing_unit);

  protected:
    sddk::mdarray<numeric_t, 1> d_;
    Simulation_context& ctx_;
};

template <class numeric_t>
sddk::mdarray<numeric_t, 2>
DiagonalPreconditioner<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, device_t processing_unit)
{
    auto Y = empty_like(X, ctx_.mem_pool(processing_unit));
    this->apply(Y, X, processing_unit);
    return Y;
}

/// computes Y <- P*X
template <class numeric_t>
inline void
DiagonalPreconditioner<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X,
                                         device_t processing_unit)
{
    // copy d_ to gpu
    switch (processing_unit) {
        case device_t::CPU: {
            int n = X.size(0);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                numeric_t d = d_(i);
                for (int j = 0; j < static_cast<int>(X.size(1)); ++j) {
                    Y(i, j) = d * X(i, j);
                }
            }
            break;
        }
#ifdef SIRIUS_GPU
        case device_t::GPU: {
            d_.allocate(memory_t::device);
            d_.copy_to(memory_t::device);
            int n = d_.size(0);
            zdiagmm(d_.at(memory_t::device), n, X.at(memory_t::device), X.ld(), X.size(1), Y.at(memory_t::device),
                    Y.ld(), std::complex<double>{1});
            break;
        }
#endif
        default:
            std::cout << "processing_unit: " << int(processing_unit) << "\n";
            throw std::runtime_error("unknown processing unit in DiagonalPreconditioner");
            break;
    }
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
    Teter(Simulation_context& ctx, const Gvec& gkvec)
        : DiagonalPreconditioner<numeric_t>(ctx)
        , local::OperatorBase(gkvec.count())
    {
        this->d_ = mdarray<numeric_t, 1>(gkvec.count());
        for (int i = 0; i < gkvec.count(); ++i) {
            // teter formula
            double T  = gkvec.gkvec_cart<index_domain_t::global>(i).length2();
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
    Ultrasoft_preconditioner(Simulation_context& simulation_context, const Q_operator<double>& q_op, int ispn,
                             const Beta_projectors_base<double>& bp, const Gvec& gkvec);

    sddk::mdarray<numeric_t, 2> apply(const sddk::mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);
    void apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X, memory_t pm = memory_t::none);

    const Simulation_context& ctx() const
    {
        return ctx_;
    }

  private:
    // cannot be const, because memory pool is used
    Simulation_context& ctx_;
    Teter<numeric_t> P;
    const Q_operator<double>& q_op;
    int ispn;
    const Beta_projectors_base<double>& bp;
    sddk::mdarray<int, 1> ipiv;
    sddk::mdarray<numeric_t, 2> LU;
};

template <class numeric_t>
Ultrasoft_preconditioner<numeric_t>::Ultrasoft_preconditioner(Simulation_context& simulation_context,
                                                              const Q_operator<double>& q_op, int ispn,
                                                              const Beta_projectors_base<double>& bp, const Gvec& gkvec)
    : local::OperatorBase(gkvec.count())
    , ctx_(simulation_context)
    , P(simulation_context, gkvec)
    , q_op(q_op)
    , ispn(ispn)
    , bp(bp)
{
    /* compute C <- <ϐ|P|ϐ> */
    auto C = inner_beta(bp, simulation_context, [&simulation_context = this->ctx_, &P = this->P](auto& Y) {
        return P.apply(Y, simulation_context.processing_unit());
    });

    sddk::matrix<numeric_t> CQ(C.size(0), q_op.size(1), memory_t::host);
    if (is_device_memory(ctx_.preferred_memory_t())) {
        C.allocate(memory_t::host);
        C.copy_to(memory_t::host);
    }
    /* compute C <- C@Q */
    this->q_op.lmatmul(CQ, C, this->ispn, memory_t::host);
    /* compute C <- 1 + C */
    int n = CQ.size(0);
    // add identiy matrix
    std::vector<double_complex> ones(n, 1);
    // add identity matrix
    linalg(linalg_t::blas).axpy(n, &linalg_const<double_complex>::one(), ones.data(), 1, CQ.at(memory_t::host), n + 1);
    // compute LU factorization
    this->LU = sddk::empty_like(CQ);
    sddk::copy_new(this->LU, CQ);
    this->ipiv = mdarray<int, 1>(n, memory_t::host);
    // compute LU factorization
    linalg(linalg_t::lapack).getrf(n, n, this->LU.at(memory_t::host), this->LU.ld(), this->ipiv.at(memory_t::host));
    // copy LU factorization to device if needed
    auto mem = ctx_.preferred_memory_t();
    if (is_device_memory(mem)) {
        ipiv.allocate(mem);
        ipiv.copy_to(mem);

        LU.allocate(mem);
        LU.copy_to(mem);
    }
}

template <class numeric_t>
sddk::mdarray<numeric_t, 2>
Ultrasoft_preconditioner<numeric_t>::apply(const sddk::mdarray<numeric_t, 2>& X, memory_t pm)
{
    auto Y = empty_like(X, ctx_.mem_pool(pm == memory_t::none ? ctx_.preferred_memory_t() : pm));
    this->apply(Y, X, pm);
    return Y;
}

template <class numeric_t>
void
Ultrasoft_preconditioner<numeric_t>::apply(sddk::mdarray<numeric_t, 2>& Y, const sddk::mdarray<numeric_t, 2>& X,
                                           memory_t pm)
{
    int num_beta = bp.num_total_beta();
    int nbnd     = X.size(1);

    pm          = (pm == memory_t::none) ? ctx_.preferred_memory_t() : pm;
    device_t pu = is_host_memory(pm) ? device_t::CPU : device_t::GPU;

    linalg_t la{linalg_t::none};
    switch (pu) {
        case device_t::CPU: {
            la = linalg_t::blas;
            break;
        }
        case device_t::GPU: {
            la = linalg_t::gpublas;
            break;
        }
    }

    auto bp_gen      = bp.make_generator(pu);
    auto beta_coeffs = bp_gen.prepare();

    sddk::mdarray<numeric_t, 2> bphi(num_beta, nbnd, ctx_.mem_pool(pm));
    // compute inner Beta^H X -> goes to host memory
    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors
        auto G         = P.apply(beta_coeffs.pw_coeffs_a, pu);
        int row_offset = beta_coeffs.beta_chunk.offset_;

        linalg(la).gemm('C', 'N', G.size(1), nbnd, G.size(0), &linalg_const<numeric_t>::one(), G.at(pm), G.ld(),
                        X.at(pm), X.ld(), &linalg_const<numeric_t>::zero(), bphi.at(pm, row_offset, 0), bphi.ld());
    }
    assert(num_beta == static_cast<int>(bphi.size(0)) && nbnd == static_cast<int>(bphi.size(1)));

    linalg_t lapack;
    switch (pu) {
        case device_t::CPU: {
            lapack = linalg_t::lapack;
            break;
        }
        case device_t::GPU: {
            lapack = linalg_t::gpublas;
            break;
        }
        default:
            throw std::runtime_error("wrong device type");
            break;
    }
    linalg(lapack).getrs('N', num_beta, nbnd, LU.at(pm), LU.ld(), ipiv.at(pm), bphi.at(pm), bphi.ld());

    auto R = empty_like(bphi, ctx_.mem_pool(pm));
    q_op.rmatmul(R, bphi, ispn, pm, -1);

    // compute Y <- (1+T')^(-1) X
    this->P.apply(Y, X, pu);

    for (int ichunk = 0; ichunk < bp.num_chunks(); ++ichunk) {
        bp_gen.generate(beta_coeffs, ichunk);
        // apply preconditioner to beta projectors in place
        auto G = P.apply(beta_coeffs.pw_coeffs_a, pu);
        int m  = Y.size(0);
        int n  = Y.size(1);
        int k  = beta_coeffs.pw_coeffs_a.size(1);

        switch (pu) {
            case device_t::CPU: {
                linalg(linalg_t::blas)
                    .gemm('N', 'N', m, n, k, &linalg_const<numeric_t>::one(), G.at(memory_t::host), G.ld(),
                          R.at(memory_t::host, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &linalg_const<numeric_t>::one(), Y.at(memory_t::host), Y.ld());
                break;
            }
#ifdef SIRIUS_GPU
            case device_t::GPU:
                linalg(linalg_t::gpublas)
                    .gemm('N', 'N', m, n, k, &linalg_const<numeric_t>::one(), G.at(memory_t::device), G.ld(),
                          R.at(memory_t::device, beta_coeffs.beta_chunk.offset_, 0), R.ld(),
                          &linalg_const<numeric_t>::one(), Y.at(memory_t::device), Y.ld());

                break;
#endif
            default:
                throw std::runtime_error("invalid processing unit");
                break;
        }
    }
}
} // namespace sirius
