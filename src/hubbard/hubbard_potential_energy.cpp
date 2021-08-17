// Copyright (c) 2013-2018 Mathieu Taillefumier, Anton Kozhevnikov, Thomas Schulthess
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

/** \file hubbard_potential_energy.hpp
 *
 *  \brief Generate Hubbard potential correction matrix from the occupancy matrix.
 */

#include "hubbard.hpp"

namespace sirius {

/* we can use Ref PRB {\bf 102}, 235159 (2020) as reference for the collinear case.  */

static void
generate_potential_collinear_nonlocal(Simulation_context const& ctx__, const int index__,
                                      sddk::mdarray<double_complex, 3> const& om__,
                                      sddk::mdarray<double_complex, 3>& um__)
{
    auto nl = ctx__.cfg().hubbard().nonlocal(index__);
    int ia  = nl.atom_pair()[0];
    int ja  = nl.atom_pair()[1];
    auto T  = nl.T();

    double v_ij_ = nl.V() / ha2ev;
    int il       = nl.l()[0];
    int jl       = nl.l()[1];
    int ib       = ctx__.unit_cell().atom(ia).type().indexr_hub().subshell_size(il, 0);
    int jb       = ctx__.unit_cell().atom(ja).type().indexr_hub().subshell_size(jl, 0);

    // second term of Eq. 2
    for (int is = 0; is < ctx__.num_spins(); is++) {
        for (int m1 = 0; m1 < 2 * jl + 1; m1++) {
            for (int m2 = 0; m2 < 2 * il + 1; m2++) {
                um__(m2, m1, is) = -v_ij_ * om__(m2, m1, is);
            }
        }
    }
}

static void
generate_potential_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__,
                                   sddk::mdarray<double_complex, 3> const& om__, sddk::mdarray<double_complex, 3>& um__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return;
    }

    um__.zero();

    /* single orbital implementation */
    int idx_hub_wf{0};
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    int const lmax_at = 2 * hub_wf.l + 1;

    if (ctx__.cfg().hubbard().simplified()) {

        if ((hub_wf.Hubbard_U() != 0.0) || (hub_wf.Hubbard_alpha() != 0.0)) {

            double U_effective = hub_wf.Hubbard_U();

            if (std::abs(hub_wf.Hubbard_J0()) > 1e-8) {
                U_effective -= hub_wf.Hubbard_J0();
            }

            for (int is = 0; is < ctx__.num_spins(); is++) {

                // is = 0 up-up
                // is = 1 down-down

                /* Expression Eq.7 without the P_IJ which is applied when we
                 * actually apply the potential to the wave functions. Also
                 * note the presence of alpha  */
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    um__(m1, m1, is) = hub_wf.Hubbard_alpha() + 0.5 * U_effective;

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        um__(m2, m1, is) -= U_effective * om__(m2, m1, is);
                    }
                }
            }
        }

        if (std::abs(hub_wf.Hubbard_J0() > 1e-8) || std::abs(hub_wf.Hubbard_beta()) > 1e-8) {
            for (int is = 0; is < ctx__.num_spins(); is++) {

                // s = 0 -> s_opposite = 1
                // s = 1 -> s_opposite = 0
                int const s_opposite = (is + 1) % 2;

                double const sign = (is == 0) - (is == 1);

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    um__(m1, m1, is) += sign * hub_wf.Hubbard_beta();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        um__(m1, m2, is) += hub_wf.Hubbard_J0() * om__(m2, m1, s_opposite);
                    }
                }
            }
        }
    } else {
        /* full hubbard correction */

        // total N and M^2 for the double counting problem

        double n_total{0};
        // n_up and n_down spins
        double n_updown[2] = {0.0, 0.0};

        for (int s = 0; s < ctx__.num_spins(); s++) {
            for (int m = 0; m < lmax_at; m++) {
                n_total += om__(m, m, s).real();
            }

            for (int m = 0; m < lmax_at; m++) {
                n_updown[s] += om__(m, m, s).real();
            }
        }

        // now hubbard contribution

        for (int is = 0; is < ctx__.num_spins(); is++) {
            for (int m1 = 0; m1 < lmax_at; m1++) {

                /* dc contribution */
                um__(m1, m1, is) += hub_wf.Hubbard_J() * n_updown[is] +
                                    0.5 * (hub_wf.Hubbard_U() - hub_wf.Hubbard_J()) - hub_wf.Hubbard_U() * n_total;

                // the u contributions
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m3 = 0; m3 < lmax_at; m3++) {
                        for (int m4 = 0; m4 < lmax_at; m4++) {

                            /* non-magnetic case */
                            if (ctx__.num_mag_dims() == 0) {
                                um__(m1, m2, is) += 2.0 * hub_wf.hubbard_matrix(m1, m3, m2, m4) * om__(m3, m4, is);
                            } else {
                                /* collinear case */
                                for (int is2 = 0; is2 < ctx__.num_spins(); is2++) {
                                    um__(m1, m2, is) += hub_wf.hubbard_matrix(m1, m3, m2, m4) * om__(m3, m4, is2);
                                }
                            }

                            um__(m1, m2, is) -= hub_wf.hubbard_matrix(m1, m3, m4, m2) * om__(m3, m4, is);
                        }
                    }
                }
            }
        }
    }
}

static double
calculate_energy_collinear_nonlocal(Simulation_context const& ctx__, const int index__,
                                    sddk::mdarray<double_complex, 3> const& om__)
{
    auto nl = ctx__.cfg().hubbard().nonlocal(index__);
    int ia  = nl.atom_pair()[0];
    int ja  = nl.atom_pair()[1];
    auto T  = nl.T();
    double hubbard_energy{0.0};
    double v_ij_ = nl.V() / ha2ev;
    int il       = nl.l()[0];
    int jl       = nl.l()[1];
    int ib_      = ctx__.unit_cell().atom(ia).type().indexr_hub().subshell_size(il, 0);
    int jb_      = ctx__.unit_cell().atom(ja).type().indexr_hub().subshell_size(jl, 0);

    // second term of Eq. 2
    for (int is = 0; is < ctx__.num_spins(); is++) {
        for (int m1 = 0; m1 < jb_; m1++) {
            for (int m2 = 0; m2 < ib_; m2++) {
                hubbard_energy += v_ij_ * std::real(om__(m2, m1, is) * conj(om__(m2, m1, is)));
            }
        }
    }

    return -0.5 * hubbard_energy;
}

static double
calculate_energy_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__,
                                 sddk::mdarray<double_complex, 3> const& om__)
{
    double hubbard_energy{0};
    double hubbard_energy_u{0};
    double hubbard_energy_dc_contribution{0};

    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return 0.0;
    }

    /* single orbital implementation */
    int idx_hub_wf{0};
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    int const lmax_at = 2 * hub_wf.l + 1;

    if (ctx__.cfg().hubbard().simplified()) {
        if ((hub_wf.Hubbard_U() != 0.0) || (hub_wf.Hubbard_alpha() != 0.0)) {

            double U_effective = hub_wf.Hubbard_U();

            if (std::abs(hub_wf.Hubbard_J0()) > 1e-8) {
                U_effective -= hub_wf.Hubbard_J0();
            }

            for (int is = 0; is < ctx__.num_spins(); is++) {

                // is = 0 up-up
                // is = 1 down-down

                for (int m1 = 0; m1 < lmax_at; m1++) {
                    hubbard_energy += (hub_wf.Hubbard_alpha() + 0.5 * U_effective) * om__(m1, m1, is).real();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        hubbard_energy -= 0.5 * U_effective * (om__(m1, m2, is) * om__(m2, m1, is)).real();
                    }
                }
            }
        }
        if (std::abs(hub_wf.Hubbard_J0() > 1e-8) || std::abs(hub_wf.Hubbard_beta()) > 1e-8) {
            for (int is = 0; is < ctx__.num_spins(); is++) {
                // s = 0 -> s_opposite = 1
                // s= 1 -> s_opposite = 0
                const int s_opposite = (is + 1) % 2;

                const double sign = (is == 0) - (is == 1);

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    hubbard_energy += sign * hub_wf.Hubbard_beta() * om__(m1, m1, is).real();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        hubbard_energy +=
                            0.5 * hub_wf.Hubbard_J0() * (om__(m2, m1, is) * om__(m1, m2, s_opposite)).real();
                    }
                }
            }
        }

        if (ctx__.num_spins() == 1) {
            hubbard_energy *= 2.0;
        }

    } else {
        /* full hubbard correction */

        // total N and M^2 for the double counting problem

        double n_total{0};
        // n_up and n_down spins
        double n_updown[2] = {0.0, 0.0};

        for (int s = 0; s < ctx__.num_spins(); s++) {
            for (int m = 0; m < lmax_at; m++) {
                n_total += om__(m, m, s).real();
            }

            for (int m = 0; m < lmax_at; m++) {
                n_updown[s] += om__(m, m, s).real();
            }
        }
        double magnetization{0};

        if (ctx__.num_mag_dims() == 0) {
            n_total *= 2.0; /* factor two here because the occupations are <= 1 */
        } else {
            for (int m = 0; m < lmax_at; m++) {
                magnetization += (om__(m, m, 0) - om__(m, m, 1)).real();
            }
            magnetization *= magnetization;
        }

        hubbard_energy_dc_contribution +=
            0.5 * (hub_wf.Hubbard_U() * n_total * (n_total - 1.0) -
                   hub_wf.Hubbard_J() * n_total * (0.5 * n_total - 1.0) - hub_wf.Hubbard_J() * magnetization * 0.5);

        /* now hubbard contribution */

        for (int is = 0; is < ctx__.num_spins(); is++) {
            for (int m1 = 0; m1 < lmax_at; m1++) {

                /* the u contributions */
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m3 = 0; m3 < lmax_at; m3++) {
                        for (int m4 = 0; m4 < lmax_at; m4++) {

                            hubbard_energy_u +=
                                0.5 * ((hub_wf.hubbard_matrix(m1, m2, m3, m4) - hub_wf.hubbard_matrix(m1, m2, m4, m3)) *
                                           om__(m1, m3, is) * om__(m2, m4, is) +
                                       hub_wf.hubbard_matrix(m1, m2, m3, m4) * om__(m1, m3, is) *
                                           om__(m2, m4, (ctx__.num_mag_dims() == 1) ? ((is + 1) % 2) : (0)))
                                          .real();
                        }
                    }
                }
            }
        }

        if (ctx__.num_mag_dims() == 0) {
            hubbard_energy_u *= 2.0;
        }
        hubbard_energy = hubbard_energy_u - hubbard_energy_dc_contribution;
    }

    //// TODO: move the printout to proper place
    // if ((ctx.verbosity() >= 1) && (ctx.comm().rank() == 0)) {
    //    std::printf("hub Energy (total) %.5lf  (dc) %.5lf\n", hubbard_energy, hubbard_energy_dc_contribution);
    //}
    return hubbard_energy;
}

static void
generate_potential_non_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__,
                                       sddk::mdarray<double_complex, 3> const& om__,
                                       sddk::mdarray<double_complex, 3>& um__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return;
    }

    /* single orbital implementation */
    int idx_hub_wf{0};
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    int const lmax_at = 2 * hub_wf.l + 1;

    um__.zero();

    // compute the charge and magnetization of the hubbard bands for
    // calculation of the double counting term in the hubbard correction

    double_complex n_total{0};
    for (int m = 0; m < lmax_at; m++) {
        n_total += om__(m, m, 0) + om__(m, m, 1);
    }

    for (int is = 0; is < 4; is++) {

        // diagonal elements of n^{\sigma\sigma'}

        int is1 = -1;

        switch (is) {
            case 2:
                is1 = 3;
                break;
            case 3:
                is1 = 2;
                break;
            default:
                is1 = is;
                break;
        }

        // same thing for the hubbard potential
        if (is1 == is) {
            // non spin flip
            for (int m1 = 0; m1 < lmax_at; ++m1) {
                for (int m2 = 0; m2 < lmax_at; ++m2) {
                    for (int m3 = 0; m3 < lmax_at; ++m3) {
                        for (int m4 = 0; m4 < lmax_at; ++m4) {
                            um__(m1, m2, is) +=
                                hub_wf.hubbard_matrix(m1, m3, m2, m4) * (om__(m3, m4, 0) + om__(m3, m4, 1));
                        }
                    }
                }
            }
        }

        // double counting contribution

        double_complex n_aux{0};
        for (int m1 = 0; m1 < lmax_at; m1++) {
            n_aux += om__(m1, m1, is1);
        }

        for (int m1 = 0; m1 < lmax_at; m1++) {

            // hubbard potential : dc contribution

            um__(m1, m1, is) += hub_wf.Hubbard_J() * n_aux;

            if (is1 == is) {
                um__(m1, m1, is) += 0.5 * (hub_wf.Hubbard_U() - hub_wf.Hubbard_J()) - hub_wf.Hubbard_U() * n_total;
            }

            // spin flip contributions
            for (int m2 = 0; m2 < lmax_at; m2++) {
                for (int m3 = 0; m3 < lmax_at; m3++) {
                    for (int m4 = 0; m4 < lmax_at; m4++) {
                        um__(m1, m2, is) -= hub_wf.hubbard_matrix(m1, m3, m4, m2) * om__(m3, m4, is1);
                    }
                }
            }
        }
    }
}

static double
calculate_energy_non_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__,
                                     sddk::mdarray<double_complex, 3> const& om__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return 0.0;
    }

    /* single orbital implementation */
    int idx_hub_wf{0};
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    int const lmax_at = 2 * hub_wf.l + 1;

    double hubbard_energy_dc_contribution{0};
    double hubbard_energy_noflip{0};
    double hubbard_energy_flip{0};
    double hubbard_energy{0};

    // compute the charge and magnetization of the hubbard bands for
    // calculation of the double counting term in the hubbard correction

    double n_total{0};
    double mx{0};
    double my{0};
    double mz{0};

    for (int m = 0; m < lmax_at; m++) {
        n_total += (om__(m, m, 0) + om__(m, m, 1)).real();
        mz += (om__(m, m, 0) - om__(m, m, 1)).real();
        mx += (om__(m, m, 2) + om__(m, m, 3)).real();
        my += (om__(m, m, 2) - om__(m, m, 3)).imag();
    }

    double magnetization = mz * mz + mx * mx + my * my;

    hubbard_energy_dc_contribution +=
        0.5 * (hub_wf.Hubbard_U() * n_total * (n_total - 1.0) - hub_wf.Hubbard_J() * n_total * (0.5 * n_total - 1.0) -
               0.5 * hub_wf.Hubbard_J() * magnetization);

    for (int is = 0; is < 4; is++) {

        // diagonal elements of n^{\sigma\sigma'}

        int is1 = -1;

        switch (is) {
            case 2:
                is1 = 3;
                break;
            case 3:
                is1 = 2;
                break;
            default:
                is1 = is;
                break;
        }

        if (is1 == is) {
            // non spin flip contributions for the hubbard energy
            for (int m1 = 0; m1 < lmax_at; ++m1) {
                for (int m2 = 0; m2 < lmax_at; ++m2) {
                    for (int m3 = 0; m3 < lmax_at; ++m3) {
                        for (int m4 = 0; m4 < lmax_at; ++m4) {
                            // 2 - is - 1 = 0 if is = 1
                            //            = 1 if is = 0

                            hubbard_energy_noflip +=
                                0.5 *
                                ((hub_wf.hubbard_matrix(m1, m2, m3, m4) - hub_wf.hubbard_matrix(m1, m2, m4, m3)) *
                                     om__(m1, m3, is) * om__(m2, m4, is) +
                                 hub_wf.hubbard_matrix(m1, m2, m3, m4) * om__(m1, m3, is) * om__(m2, m4, (is + 1) % 2))
                                    .real();
                        }
                    }
                }
            }
        } else {
            // spin flip contributions
            for (int m1 = 0; m1 < lmax_at; ++m1) {
                for (int m2 = 0; m2 < lmax_at; ++m2) {
                    for (int m3 = 0; m3 < lmax_at; ++m3) {
                        for (int m4 = 0; m4 < lmax_at; ++m4) {
                            hubbard_energy_flip -=
                                0.5 *
                                (hub_wf.hubbard_matrix(m1, m2, m4, m3) * om__(m1, m3, is) * om__(m2, m4, is1)).real();
                        }
                    }
                }
            }
        }
    }

    hubbard_energy = hubbard_energy_noflip + hubbard_energy_flip - hubbard_energy_dc_contribution;

    // if ((ctx_.verbosity() >= 1) && (ctx_.comm().rank() == 0)) {
    //    std::printf("\n hub Energy (total) %.5lf (no-flip) %.5lf (flip) %.5lf (dc) %.5lf\n",
    //        hubbard_energy, hubbard_energy_noflip, hubbard_energy_flip, hubbard_energy_dc_contribution);
    //}

    return hubbard_energy;
}

void
generate_potential(Hubbard_matrix const& om__, Hubbard_matrix& um__)
{
    auto& ctx = om__.ctx();

    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto& atype = ctx.unit_cell().atom(ia).type();
        if (atype.hubbard_correction()) {
            if (ctx.num_mag_dims() != 3) {
                ::sirius::generate_potential_collinear_local(ctx, atype, om__.local(ia), um__.local(ia));
            } else {
                ::sirius::generate_potential_non_collinear_local(ctx, atype, om__.local(ia), um__.local(ia));
            }
        }
    }
    for (int i = 0; i < ctx.cfg().hubbard().nonlocal().size(); i++) {
        if (ctx.num_mag_dims() != 3) {
            ::sirius::generate_potential_collinear_nonlocal(ctx, i, om__.nonlocal(i), um__.nonlocal(i));
        } // else {
          //    ::sirius::generate_potential_non_collinear_nonlocal(ctx, nl, om__.nonlocal(i), um__.nonlocal(i));
          //}
    }
}

double
energy(Hubbard_matrix const& om__)
{
    double energy{0};

    auto& ctx = om__.ctx();

    for (int ia = 0; ia < ctx.unit_cell().num_atoms(); ia++) {
        auto& atype = ctx.unit_cell().atom(ia).type();
        if (atype.hubbard_correction()) {
            if (ctx.num_mag_dims() != 3) {
                energy += ::sirius::calculate_energy_collinear_local(ctx, atype, om__.local(ia));
            } else {
                energy += ::sirius::calculate_energy_non_collinear_local(ctx, atype, om__.local(ia));
            }
        }
    }

    double energy_u_v_{0};
    for (int i = 0; i < ctx.cfg().hubbard().nonlocal().size(); i++) {
        if (ctx.num_mag_dims() != 3) {
            energy_u_v_ += ::sirius::calculate_energy_collinear_nonlocal(ctx, i, om__.nonlocal(i));
        }
        // else {
        //    energy += ::sirius::calculate_energy_noncollinear_nonlocal(ctx, nl, om__.nonlocal(i));
        //}
    }
    return energy + energy_u_v_;
}

} // namespace sirius
