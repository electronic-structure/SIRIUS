#ifndef __GENERATE_GVEC_YLM_HPP__
#define __GENERATE_GVEC_YLM_HPP__

namespace sirius {

/// Generate complex spherical harmonics for the local set of G-vectors.
inline auto
generate_gvec_ylm(Simulation_context const& ctx__, int lmax__)
{
    PROFILE("sirius::generate_gvec_ylm");

    sddk::mdarray<std::complex<double>, 2> gvec_ylm(utils::lmmax(lmax__), ctx__.gvec().count(),
            sddk::memory_t::host, "gvec_ylm");
    #pragma omp parallel for schedule(static)
    for (int igloc = 0; igloc < ctx__.gvec().count(); igloc++) {
        auto rtp = r3::spherical_coordinates(ctx__.gvec().gvec_cart<sddk::index_domain_t::local>(igloc));
        sf::spherical_harmonics(lmax__, rtp[1], rtp[2], &gvec_ylm(0, igloc));
    }
    return gvec_ylm;
}

}

#endif
