/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file get_wave_function_value.hpp
 *
 *  \brief Compute value of the wave-function at a given Cartesian coordinate.
 */

#ifndef __GET_WAVE_FUNCTION_VALUE_HPP__
#define __GET_WAVE_FUNCTION_VALUE_HPP__

// #include ...

namespace sirius {

inline auto
get_wave_function_value(K_point<double> const& kp__, wf::Wave_functions<double> const& wf__, r3::vector<double> r__,
                        wf::band_index band_idx__, wf::spin_index spin_idx__)
{
    int ja{-1}, jr{-1};
    double dr{0}, tp[2];

    auto const& uc = kp__.ctx().unit_cell();

    // returned value
    std::complex<double> val(0, 0);

    if (uc.is_point_in_mt(r__, ja, jr, dr, tp)) {
        // ja - index of the atom where r__ belongs to
        // jr - starting index of the radial grid
        // dr - distance from radial_grid[jr] to r__
        // tp - (theta, phi) angles of the point connecting centre of atom ja with r__

        // get MPI location of atom ja
        auto loc = wf__.spl_num_atoms().location(atom_index_t::global(ja));
        // this rank holds local set of MT coefficients
        if (wf__.comm().rank() == loc.ib) {

            auto& atom = uc.atom(ja);

            // generate Ylm spherical harmonics for the given (theta, phi)
            std::vector<std::complex<double>> ylm(atom.type().lmmax_apw());
            sf::spherical_harmonics(atom.type().lmax_apw(), tp[0], tp[1], &ylm[0]);

            // iterate over all atomic basis functions (apw and lo)
            double re{0}, im{0};
            #pragma omp parallel for reduction(+:re, im)
            for (int xi = 0; xi < atom.mt_basis_size(); xi++) {
                // expansion coefficient
                auto c = wf__.mt_coeffs(xi, loc.index_local, spin_idx__, band_idx__);
                // lm index of spherical harmonic
                int lm = atom.type().indexb(xi).lm;
                // index of radial function
                auto idxrf = atom.type().indexb(xi).idxrf;
                // get derivative of radial function; this is needed for simple linear interpolation
                auto f1 = (atom.symmetry_class().radial_function(jr + 1, idxrf) -
                           atom.symmetry_class().radial_function(jr, idxrf)) /
                          atom.type().radial_grid().dx(jr);

                auto v = c * ylm[lm] * (atom.symmetry_class().radial_function(jr, idxrf) + f1 * dr);
                re += v.real();
                im += v.imag();
            }
            val += std::complex<double>(re, im);
        }
        // broadcast from the rank which computed the sum
        wf__.comm().bcast(&val, 1, loc.ib);

    } else {
        // sum over local set of G+k-vectors
        double re{0}, im{0};
        int ngkloc = kp__.gkvec().count();
        #pragma omp parallel for reduction(+:re, im)
        for (int igloc = 0; igloc < ngkloc; igloc++) {
            // G+k vector in Cartesian coordinates
            auto vgc = kp__.gkvec().gvec_cart(gvec_index_t::local(igloc));
            // plane-wave expansion
            auto v = wf__.pw_coeffs(igloc, spin_idx__, band_idx__) * std::exp(std::complex<double>(0.0, dot(r__, vgc)));
            re += v.real();
            im += v.imag();
        }
        val += std::complex<double>(re, im);
        kp__.gkvec().comm().allreduce(&val, 1);
        val /= std::sqrt(uc.omega());
    }
    return val;
}

} // namespace sirius

#endif
