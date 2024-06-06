/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hubbard_potential_energy.hpp
 *
 *  \brief Generate Hubbard potential correction matrix from the occupancy matrix.
 */

#include "hubbard.hpp"

namespace sirius {

/* we can use Ref PRB {\bf 102}, 235159 (2020) as reference for the collinear case.  */

static void
generate_constraint_potential(Simulation_context const& ctx__, Atom_type const& atom_type__, const int idx_hub_wf,
                              const int at_lvl, mdarray<std::complex<double>, 3> const& om__, // __unused__
                              mdarray<std::complex<double>, 3> const& om_ref__,               // __unused__
                              mdarray<std::complex<double>, 3> const& lagrange_multiplier__,
                              mdarray<std::complex<double>, 3>& um__)
{
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    int const lmax_at = 2 * hub_wf.l() + 1;
    for (int is = 0; is < ctx__.num_spins(); is++) {
        for (int m1 = 0; m1 < lmax_at; m1++) {
            for (int m2 = 0; m2 < lmax_at; m2++) {
                um__(m2, m1, is) -= ctx__.cfg().hubbard().constraint_strength() * lagrange_multiplier__(m2, m1, is);
            }
        }
    }
}

static void
generate_potential_collinear_nonlocal(Simulation_context const& ctx__, const int index__,
                                      mdarray<std::complex<double>, 3> const& om__,
                                      mdarray<std::complex<double>, 3>& um__)
{
    auto nl = ctx__.cfg().hubbard().nonlocal(index__);
    um__.zero();
    const int il = nl.l()[0];
    const int jl = nl.l()[1];
    um__.zero();
    // second term of Eq. 2
    for (int is = 0; is < ctx__.num_spins(); is++) {
        for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
            for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                um__(m1, m2, is) = -nl.V() * om__(m1, m2, is);
            }
        }
    }
}

static void
generate_potential_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__, const int idx_hub_wf,
                                   mdarray<std::complex<double>, 3> const& om__, mdarray<std::complex<double>, 3>& um__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return;
    }

    um__.zero();

    /* single orbital implementation */
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    if (!hub_wf.use_for_calculation()) {
        return;
    }

    int const lmax_at = 2 * hub_wf.l() + 1;

    if (ctx__.cfg().hubbard().simplified()) {

        if ((hub_wf.U() != 0.0) || (hub_wf.alpha() != 0.0)) {

            double U_effective = hub_wf.U();

            if (std::abs(hub_wf.J0()) > 1e-8) {
                U_effective -= hub_wf.J0();
            }

            for (int is = 0; is < ctx__.num_spins(); is++) {

                // is = 0 up-up
                // is = 1 down-down

                /* Expression Eq.7 without the P_IJ which is applied when we
                 * actually apply the potential to the wave functions. Also
                 * note the presence of alpha  */
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    um__(m1, m1, is) = hub_wf.alpha() + 0.5 * U_effective;

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        um__(m2, m1, is) -= U_effective * om__(m2, m1, is);
                    }
                }
            }
        }

        if (std::abs(hub_wf.J0()) > 1e-8 || std::abs(hub_wf.beta()) > 1e-8) {
            for (int is = 0; is < ctx__.num_spins(); is++) {

                // s = 0 -> s_opposite = 1
                // s = 1 -> s_opposite = 0
                int const s_opposite = (is + 1) % 2;

                double const sign = (is == 0) - (is == 1);

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    um__(m1, m1, is) += sign * hub_wf.beta();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        um__(m1, m2, is) += hub_wf.J0() * om__(m2, m1, s_opposite);
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
                um__(m1, m1, is) += hub_wf.J() * n_updown[is] + 0.5 * (hub_wf.U() - hub_wf.J()) - hub_wf.U() * n_total;

                // the u contributions
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m3 = 0; m3 < lmax_at; m3++) {
                        for (int m4 = 0; m4 < lmax_at; m4++) {
                            for (int is2 = 0; is2 < ctx__.num_spins(); is2++) {
                                um__(m1, m2, is) += hub_wf.hubbard_matrix(m1, m3, m2, m4) * om__(m3, m4, is2);
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
calculate_energy_constraint_contribution(Simulation_context const& ctx__, Atom_type const& atom_type__,
                                         const int idx_hub_wf, mdarray<std::complex<double>, 3> const& om__,
                                         mdarray<std::complex<double>, 3> const& om_constraints__,
                                         mdarray<std::complex<double>, 3> const& lagrange_multipliers__)
{
    std::complex<double> hubbard_energy_constraint{0};
    /* quick exit */

    /* single orbital implementation */
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    if (!hub_wf.use_for_calculation())
        return 0.0;

    int const lmax_at = 2 * hub_wf.l() + 1;
    for (int is = 0; is < ctx__.num_spins(); is++) {

        // is = 0 up-up
        // is = 1 down-down

        for (int m1 = 0; m1 < lmax_at; m1++) {
            for (int m2 = 0; m2 < lmax_at; m2++) {
                hubbard_energy_constraint += ctx__.cfg().hubbard().constraint_strength() *
                                             (om__(m2, m1, is) - om_constraints__(m2, m1, is)) *
                                             lagrange_multipliers__(m2, m1, is);
            }
        }
    }

    return std::real(hubbard_energy_constraint);
}

static double
calculate_energy_collinear_nonlocal(Simulation_context const& ctx__, const int index__,
                                    mdarray<std::complex<double>, 3> const& om__)
{
    auto nl = ctx__.cfg().hubbard().nonlocal(index__);
    double hubbard_energy{0.0};

    const int il = nl.l()[0];
    const int jl = nl.l()[1];

    // second term of Eq. 2
    for (int is = 0; is < ctx__.num_spins(); is++) {
        for (int m1 = 0; m1 < 2 * jl + 1; m1++) {
            for (int m2 = 0; m2 < 2 * il + 1; m2++) {
                hubbard_energy += nl.V() * std::real(om__(m2, m1, is) * conj(om__(m2, m1, is)));
            }
        }
    }

    // non magnetic case
    if (ctx__.num_spins() == 1) {
        hubbard_energy *= 2.0;
    }

    return -0.5 * hubbard_energy;
}

static double
calculate_energy_collinear_local(Simulation_context const& ctx__, Atom_type const& atom_type__, const int idx_hub_wf,
                                 mdarray<std::complex<double>, 3> const& om__)
{
    double hubbard_energy{0};
    double hubbard_energy_u{0};
    double hubbard_energy_dc_contribution{0};

    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return 0.0;
    }

    /* single orbital implementation */
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    if (!hub_wf.use_for_calculation())
        return 0.0;

    int const lmax_at = 2 * hub_wf.l() + 1;

    if (ctx__.cfg().hubbard().simplified()) {
        if ((hub_wf.U() != 0.0) || (hub_wf.alpha() != 0.0)) {

            double U_effective = hub_wf.U();

            if (std::abs(hub_wf.J0()) > 1e-8) {
                U_effective -= hub_wf.J0();
            }

            for (int is = 0; is < ctx__.num_spins(); is++) {

                // is = 0 up-up
                // is = 1 down-down

                for (int m1 = 0; m1 < lmax_at; m1++) {
                    hubbard_energy += (hub_wf.alpha() + 0.5 * U_effective) * om__(m1, m1, is).real();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        hubbard_energy -= 0.5 * U_effective * (om__(m1, m2, is) * om__(m2, m1, is)).real();
                    }
                }
            }
        }
        if (std::abs(hub_wf.J0()) > 1e-8 || std::abs(hub_wf.beta()) > 1e-8) {
            for (int is = 0; is < ctx__.num_spins(); is++) {
                // s = 0 -> s_opposite = 1
                // s= 1 -> s_opposite = 0
                const int s_opposite = (is + 1) % 2;

                const double sign = (is == 0) - (is == 1);

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    hubbard_energy += sign * hub_wf.beta() * om__(m1, m1, is).real();

                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        hubbard_energy += 0.5 * hub_wf.J0() * (om__(m2, m1, is) * om__(m1, m2, s_opposite)).real();
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
                0.5 * (hub_wf.U() * n_total * (n_total - 1.0) - hub_wf.J() * n_total * (0.5 * n_total - 1.0) -
                       hub_wf.J() * magnetization * 0.5);

        /* now hubbard contribution */

        for (int is = 0; is < ctx__.num_spins(); is++) {
            for (int m1 = 0; m1 < lmax_at; m1++) {

                /* the u contributions */
                for (int m2 = 0; m2 < lmax_at; m2++) {
                    for (int m3 = 0; m3 < lmax_at; m3++) {
                        for (int m4 = 0; m4 < lmax_at; m4++) {

                            hubbard_energy_u +=
                                    0.5 *
                                    ((hub_wf.hubbard_matrix(m1, m2, m3, m4) - hub_wf.hubbard_matrix(m1, m2, m4, m3)) *
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
                                       const int idx_hub_wf, mdarray<std::complex<double>, 3> const& om__,
                                       mdarray<std::complex<double>, 3>& um__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return;
    }

    um__.zero();

    /* single orbital implementation */
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    if (!hub_wf.use_for_calculation())
        return;

    int const lmax_at = 2 * hub_wf.l() + 1;

    // compute the charge and magnetization of the hubbard bands for
    // calculation of the double counting term in the hubbard correction

    std::complex<double> n_total{0};
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

        std::complex<double> n_aux{0};
        for (int m1 = 0; m1 < lmax_at; m1++) {
            n_aux += om__(m1, m1, is1);
        }

        for (int m1 = 0; m1 < lmax_at; m1++) {

            // hubbard potential : dc contribution

            um__(m1, m1, is) += hub_wf.J() * n_aux;

            if (is1 == is) {
                um__(m1, m1, is) += 0.5 * (hub_wf.U() - hub_wf.J()) - hub_wf.U() * n_total;
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
                                     const int idx_hub_wf, mdarray<std::complex<double>, 3> const& om__)
{
    /* quick exit */
    if (!atom_type__.hubbard_correction()) {
        return 0.0;
    }

    /* single orbital implementation */
    auto& hub_wf = atom_type__.lo_descriptor_hub(idx_hub_wf);

    if (!hub_wf.use_for_calculation())
        return 0.0;

    int const lmax_at = 2 * hub_wf.l() + 1;

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
            0.5 * (hub_wf.U() * n_total * (n_total - 1.0) - hub_wf.J() * n_total * (0.5 * n_total - 1.0) -
                   0.5 * hub_wf.J() * magnetization);

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
                                     hub_wf.hubbard_matrix(m1, m2, m3, m4) * om__(m1, m3, is) *
                                             om__(m2, m4, (is + 1) % 2))
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
                                    0.5 * (hub_wf.hubbard_matrix(m1, m2, m4, m3) * om__(m1, m3, is) * om__(m2, m4, is1))
                                                  .real();
                        }
                    }
                }
            }
        }
    }

    hubbard_energy = hubbard_energy_noflip + hubbard_energy_flip - hubbard_energy_dc_contribution;

    return hubbard_energy;
}

void
generate_potential(Hubbard_matrix const& om__, Hubbard_matrix& um__)
{
    auto& ctx = om__.ctx();

    for (int at_lvl = 0; at_lvl < static_cast<int>(om__.local().size()); at_lvl++) {
        const int ia = om__.atomic_orbitals(at_lvl).first;
        auto& atype  = ctx.unit_cell().atom(ia).type();
        int lo_ind   = om__.atomic_orbitals(at_lvl).second;

        if (ctx.num_mag_dims() != 3) {
            ::sirius::generate_potential_collinear_local(ctx, atype, lo_ind, om__.local(at_lvl), um__.local(at_lvl));
        } else {
            ::sirius::generate_potential_non_collinear_local(ctx, atype, lo_ind, om__.local(at_lvl),
                                                             um__.local(at_lvl));
        }

        if (om__.apply_constraint() && ctx.cfg().hubbard().constraint_method() == "energy") {
            if (om__.apply_constraints(at_lvl)) {
                ::sirius::generate_constraint_potential(ctx, atype, lo_ind, at_lvl, om__.local(at_lvl),
                                                        om__.local_constraints(at_lvl),
                                                        om__.multipliers_constraints(at_lvl), um__.local(at_lvl));
            }
        }

        if (om__.apply_constraint() && ctx.cfg().hubbard().constraint_method() == "occupantion") {
            if (om__.apply_constraints(at_lvl)) {
                ::sirius::generate_constraint_potential(ctx, atype, lo_ind, at_lvl, om__.local(at_lvl),
                                                        om__.local_constraints(at_lvl), om__.local_constraints(at_lvl),
                                                        um__.local(at_lvl));
            }
        }
    }
    for (int i = 0; i < static_cast<int>(ctx.cfg().hubbard().nonlocal().size()); i++) {
        if (ctx.num_mag_dims() != 3) {
            ::sirius::generate_potential_collinear_nonlocal(ctx, i, om__.nonlocal(i), um__.nonlocal(i));
        }
    }
}

double
energy(Hubbard_matrix const& om__)
{
    double energy{0};

    auto& ctx = om__.ctx();

    for (int at_lvl = 0; at_lvl < static_cast<int>(om__.local().size()); at_lvl++) {
        const int ia = om__.atomic_orbitals(at_lvl).first;
        auto& atype  = ctx.unit_cell().atom(ia).type();
        int lo_ind   = om__.atomic_orbitals(at_lvl).second;
        if (ctx.num_mag_dims() != 3) {
            energy += ::sirius::calculate_energy_collinear_local(ctx, atype, lo_ind, om__.local(at_lvl));
        } else {
            energy += ::sirius::calculate_energy_non_collinear_local(ctx, atype, lo_ind, om__.local(at_lvl));
        }

        if (om__.apply_constraint() && ctx.cfg().hubbard().constraint_method() == "energy") {
            if (om__.apply_constraints(at_lvl)) {
                double tmp_ = ::sirius::calculate_energy_constraint_contribution(ctx, atype, lo_ind, om__.local(at_lvl),
                                                                                 om__.local_constraints(at_lvl),
                                                                                 om__.multipliers_constraints(at_lvl));
                energy += tmp_;
            }
        }
    }
    for (int i = 0; i < static_cast<int>(ctx.cfg().hubbard().nonlocal().size()); i++) {
        if (ctx.num_mag_dims() != 3) {
            energy += ::sirius::calculate_energy_collinear_nonlocal(ctx, i, om__.nonlocal(i));
        }
    }
    return energy;
}

// this function is used when we want to calculate the kinetic energy. The
// kinetic energy is calculated from the self consistent hamiltonian not from
// the direct calculation of the gradient of the wave functions.
//
// E_kin = \sum_{i,k} \epsilon_ik n_{ik} - <\psi_{ik} | V | \psi_{ik}>
//
//
// where V is the potential (that include all contributions). V = V0 + VHub and
//
// <\psi_{ik} | V_Hub | \psi_{ik}> = \sum_{m_1,m_2} U^I (\delta_m1m2 - n^I_{m_1, m_2}) conj(n^I_{m1, m2}) - V^{ij} |
// n^{ij}_{m1,m2}| ^ 2
//
// it is a real number

double
one_electron_energy_hubbard(Hubbard_matrix const& om__, Hubbard_matrix const& pm__)
{
    auto& ctx = om__.ctx();
    if (ctx.hubbard_correction()) {
        std::complex<double> tmp{0.0, 0.0};
        for (int at_lvl = 0; at_lvl < static_cast<int>(om__.local().size()); at_lvl++) {
            const int ia = om__.atomic_orbitals(at_lvl).first;
            int lo_ind   = om__.atomic_orbitals(at_lvl).second;
            auto& atype  = ctx.unit_cell().atom(ia).type();
            auto& hub_wf = atype.lo_descriptor_hub(lo_ind);

            if (hub_wf.use_for_calculation()) {
                auto src1 = om__.local(at_lvl).at(memory_t::host);
                auto src2 = pm__.local(at_lvl).at(memory_t::host);

                for (int i = 0; i < static_cast<int>(om__.local(at_lvl).size()); i++) {
                    tmp += src1[i] * std::conj(src2[i]);
                }
            }
        }

        for (int i = 0; i < static_cast<int>(ctx.cfg().hubbard().nonlocal().size()); i++) {
            auto nl = ctx.cfg().hubbard().nonlocal(i);
            int il  = nl.l()[0];
            int jl  = nl.l()[1];

            const auto& n1 = om__.nonlocal(i);
            const auto& n2 = pm__.nonlocal(i);

            for (int is = 0; is < ctx.num_spins(); is++) {
                for (int m2 = 0; m2 < 2 * jl + 1; m2++) {
                    for (int m1 = 0; m1 < 2 * il + 1; m1++) {
                        tmp += std::conj(n2(m1, m2, is)) * n1(m1, m2, is);
                    }
                }
            }
        }

        if (ctx.num_spins() == 1) {
            tmp *= 2.0;
        }
        return std::real(tmp);
    }
    return 0.0;
}

} // namespace sirius
