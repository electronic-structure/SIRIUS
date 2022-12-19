#ifndef __WAVEFUNCTION_STRAIN_DERIV_HPP__
#define __WAVEFUNCTION_STRAIN_DERIV_HPP__

#include "context/simulation_context.hpp"

namespace sirius {

void
wavefunctions_strain_deriv(Simulation_context const& ctx__, K_point<double>& kp__, wf::Wave_functions<double>& dphi__,
                           sddk::mdarray<double, 2> const& rlm_g__, sddk::mdarray<double, 3> const& rlm_dg__,
                           int nu__, int mu__)
{
    auto num_ps_atomic_wf = ctx__.unit_cell().num_ps_atomic_wf();
    PROFILE("sirius::wavefunctions_strain_deriv");
    #pragma omp parallel for schedule(static)
    for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
        /* Cartesian coordinats of G-vector */
        auto gvc = kp__.gkvec().gkvec_cart<sddk::index_domain_t::local>(igkloc);
        /* vs = {r, theta, phi} */
        auto gvs = SHT::spherical_coordinates(gvc);

        std::vector<sddk::mdarray<double, 1>> ri_values(ctx__.unit_cell().num_atom_types());
        for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
            ri_values[iat] = ctx__.ps_atomic_wf_ri().values(iat, gvs[0]);
        }

        std::vector<sddk::mdarray<double, 1>> ridjl_values(ctx__.unit_cell().num_atom_types());
        for (int iat = 0; iat < ctx__.unit_cell().num_atom_types(); iat++) {
            ridjl_values[iat] = ctx__.ps_atomic_wf_ri_djl().values(iat, gvs[0]);
        }

        const double p = (mu__ == nu__) ? 0.5 : 0.0;

        for (int ia = 0; ia < ctx__.unit_cell().num_atoms(); ia++) {
            auto& atom_type = ctx__.unit_cell().atom(ia).type();
            // TODO: this can be optimized, check k_point::generate_atomic_wavefunctions()
            auto phase        = twopi * dot(kp__.gkvec().gkvec<sddk::index_domain_t::local>(igkloc),
                                            ctx__.unit_cell().atom(ia).position());
            auto phase_factor = std::exp(std::complex<double>(0.0, phase));
            for (int xi = 0; xi < atom_type.indexb_wfs().size(); xi++) {
                /*  orbital quantum  number of this atomic orbital */
                int l = atom_type.indexb_wfs().l(xi);
                /*  composite l,m index */
                int lm = atom_type.indexb_wfs().lm(xi);
                /* index of the radial function */
                int idxrf = atom_type.indexb_wfs().idxrf(xi);
                int offset_in_wf = num_ps_atomic_wf.second[ia] + xi;

                auto z = std::pow(std::complex<double>(0, -1), l) * fourpi / std::sqrt(ctx__.unit_cell().omega());

                /* case |G+k| = 0 */
                if (gvs[0] < 1e-10) {
                    if (l == 0) {
                        auto d1 = ri_values[atom_type.id()][idxrf] * p * y00;

                        dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) = -z * d1 * phase_factor;
                    } else {
                        dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) = 0.0;
                    }
                } else {
                    auto d1 = ri_values[atom_type.id()][idxrf] *
                        (gvc[mu__] * rlm_dg__(lm, nu__, igkloc) + p * rlm_g__(lm, igkloc));
                    auto d2 =
                        ridjl_values[atom_type.id()][idxrf] * rlm_g__(lm, igkloc) * gvc[mu__] * gvc[nu__] / gvs[0];

                    dphi__.pw_coeffs(igkloc, wf::spin_index(0), wf::band_index(offset_in_wf)) = -z * (d1 + d2) * std::conj(phase_factor);
                }
            } // xi
        }
    }
}


}

#endif
