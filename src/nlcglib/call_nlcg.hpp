#ifndef CALL_NLCG_H
#define CALL_NLCG_H

// #include "context/config.hpp"
#include "context/simulation_context.hpp"
#include "nlcglib/adaptor.hpp"
#include "nlcglib/ultrasoft_precond.hpp"
#include "nlcglib/overlap.hpp"
#include "nlcglib/nlcglib.hpp"
#include "hamiltonian/hamiltonian.hpp"

namespace sirius {

inline void
call_nlcg(Simulation_context& ctx, const config_t::nlcg_t& nlcg_params, Energy& energy, K_point_set& kset, Potential& potential)
{
    using numeric_t = std::complex<double>;

    double temp  = nlcg_params.T();
    double tol   = nlcg_params.tol();
    double kappa = nlcg_params.kappa();
    double tau   = nlcg_params.tau();
    int maxiter  = nlcg_params.maxiter();
    int restart  = nlcg_params.restart();
    auto nlcg_pu = sddk::get_device_t(nlcg_params.processing_unit());

    std::string smear = ctx.cfg().parameters().smearing();

    nlcglib::smearing_type smearing;
    if (smear.compare("fermi_dirac") == 0) {
        smearing = nlcglib::smearing_type::FERMI_DIRAC;
    } else if (smear.compare("gaussian_spline") == 0) {
        smearing = nlcglib::smearing_type::GAUSSIAN_SPLINE;
    } else if (smear.compare("gaussian") == 0) {
        smearing = nlcglib::smearing_type::GAUSS;
    } else if (smear.compare("methfessel_paxton") == 0) {
        smearing = nlcglib::smearing_type::METHFESSEL_PAXTON;
    } else if (smear.compare("cold") == 0) {
        smearing = nlcglib::smearing_type::COLD;
    } else {
        throw std::runtime_error("invalid smearing type given");
    }

    Hamiltonian0<double> H0(potential, false /* precompute laplw */);

    sirius::UltrasoftPrecond us_precond(kset, ctx, H0.Q());
    sirius::Overlap_operators<sirius::S_k<numeric_t>> S(kset, ctx, H0.Q());

    // ultrasoft pp
    switch (nlcg_pu) {
        case sddk::device_t::CPU: {
            nlcglib::nlcg_us_cpu(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
        case sddk::device_t::GPU: {
            nlcglib::nlcg_us_device(energy, us_precond, S, smearing, temp, tol, kappa, tau, maxiter, restart);
            break;
        }
    }
}

} // namespace sirius

#endif /* CALL_NLCG_H */
