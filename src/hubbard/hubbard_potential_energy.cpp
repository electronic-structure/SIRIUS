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

  /* we can use Ref PRB {\bf 102}, 235159 (2020) as reference for the colinear
   * case.
   */
void
Hubbard::generate_potential_collinear(sddk::mdarray<double_complex, 4> const& om__)
{
    this->hubbard_potential_.zero();

    if (this->approximation_ == 1) {
        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
            auto const& atom = this->unit_cell_.atom(ia);
            if (!atom.type().hubbard_correction()) {
                continue;
            }
            double U_effective = 0.0;

            int const lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;

            if ((atom.type().lo_descriptor_hub(0).Hubbard_U() != 0.0) ||
                (atom.type().lo_descriptor_hub(0).Hubbard_alpha() != 0.0)) {

                U_effective = atom.type().lo_descriptor_hub(0).Hubbard_U();

                if (std::abs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8) {
                    U_effective -= atom.type().lo_descriptor_hub(0).Hubbard_J0();
                }

                for (int is = 0; is < ctx_.num_spins(); is++) {

                    // is = 0 up-up
                    // is = 1 down-down

                  /* Expression Eq.7 without the P_IJ which is applied when we
                   * actually apply the potential to the wave functions. Also
                   * note the presence of alpha  */
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        this->hubbard_potential_(m1, m1, is, ia) =
                            (atom.type().lo_descriptor_hub(0).Hubbard_alpha() + 0.5 * U_effective);

                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            this->hubbard_potential_(m2, m1, is, ia) -= U_effective * om__(m2, m1, is, ia);
                        }
                    }
                }
            }

            if ((std::abs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8) ||
                (std::abs(atom.type().lo_descriptor_hub(0).Hubbard_beta()) > 1e-8)) {
                for (int is = 0; is < ctx_.num_spins(); is++) {

                    // s = 0 -> s_opposite = 1
                    // s = 1 -> s_opposite = 0
                    int const s_opposite = (is + 1) % 2;

                    double const sign = (is == 0) - (is == 1);

                    for (int m1 = 0; m1 < lmax_at; m1++) {

                        this->U(m1, m1, is, ia) += sign * atom.type().lo_descriptor_hub(0).Hubbard_beta();

                        for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
                            this->U(m1, m2, is, ia) += atom.type().lo_descriptor_hub(0).Hubbard_J0() *
                                om__(m2, m1, s_opposite, ia);
                        }
                    }
                }
            }
        }
    } else {
        /* full hubbard correction */
        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {

            auto const& atom = this->unit_cell_.atom(ia);

            if (!atom.type().hubbard_correction()) {
                continue;
            }

            // total N and M^2 for the double counting problem

            double n_total = 0.0;
            // n_up and n_down spins
            double n_updown[2] = {0.0, 0.0};

            int const lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;

            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int m = 0; m < lmax_at; m++) {
                    n_total += om__(m, m, s, ia).real();
                }

                for (int m = 0; m < lmax_at; m++) {
                    n_updown[s] += om__(m, m, s, ia).real();
                }
            }
            double magnetization = 0.0;

            if (ctx_.num_mag_dims() == 0) {
                n_total *= 2.0; // factor two here because the occupations are <= 1
            } else {
                for (int m = 0; m < lmax_at; m++) {
                    magnetization += (om__(m, m, 0, ia) - om__(m, m, 1, ia)).real();
                }
                magnetization *= magnetization;
            }


            // now hubbard contribution

            for (int is = 0; is < ctx_.num_spins(); is++) {
                for (int m1 = 0; m1 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m1++) {

                    // dc contribution
                    this->U(m1, m1, is, ia) += atom.type().lo_descriptor_hub(0).Hubbard_J() * n_updown[is] +
                                               0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() -
                                                      atom.type().lo_descriptor_hub(0).Hubbard_J()) -
                                               atom.type().lo_descriptor_hub(0).Hubbard_U() * n_total;

                    // the u contributions
                    for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
                        for (int m3 = 0; m3 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m3++) {
                            for (int m4 = 0; m4 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m4++) {

                                // why should we consider the spinless case

                                if (ctx_.num_mag_dims() == 0) {
                                    this->U(m1, m2, is, ia) +=
                                        2.0 * atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m2, m4) *
                                        om__(m3, m4, is, ia);
                                } else {
                                    // colinear case
                                    for (int is2 = 0; is2 < 2; is2++) {
                                        this->U(m1, m2, is, ia) +=
                                            atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m2, m4) *
                                            om__(m3, m4, is2, ia);
                                    }
                                }

                                this->U(m1, m2, is, ia) -=
                                    atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m4, m2) * om__(m3, m4, is, ia);
                            }
                        }
                    }
                }
            }
        }
    }
}

double
Hubbard::calculate_energy_collinear(sddk::mdarray<double_complex, 4> const& om__) const
{
    double hubbard_energy{0};
    double hubbard_energy_u{0};
    double hubbard_energy_dc_contribution{0};

    if (this->approximation_ == 1) {
        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
            const auto& atom = this->unit_cell_.atom(ia);
            if (atom.type().hubbard_correction()) {
                double    U_effective = 0.0;
                const int lmax_at     = 2 * atom.type().lo_descriptor_hub(0).l + 1;
                if ((atom.type().lo_descriptor_hub(0).Hubbard_U() != 0.0) || (atom.type().lo_descriptor_hub(0).Hubbard_alpha() != 0.0)) {

                    U_effective = atom.type().lo_descriptor_hub(0).Hubbard_U();

                    if (std::abs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8) {
                        U_effective -= atom.type().lo_descriptor_hub(0).Hubbard_J0();
                    }

                    for (int is = 0; is < ctx_.num_spins(); is++) {

                        // is = 0 up-up
                        // is = 1 down-down

                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            hubbard_energy +=
                                (atom.type().lo_descriptor_hub(0).Hubbard_alpha() + 0.5 * U_effective) * om__(m1, m1, is, ia).real();

                            for (int m2 = 0; m2 < lmax_at; m2++) {

                                hubbard_energy -=
                                    0.5 * U_effective * (om__(m1, m2, is, ia) * om__(m2, m1, is, ia)).real();
                            }
                        }
                    }
                }

                if ((std::abs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8) ||
                    (std::abs(atom.type().lo_descriptor_hub(0).Hubbard_beta()) > 1e-8)) {
                    for (int is = 0; is < ctx_.num_spins(); is++) {

                        // s = 0 -> s_opposite = 1
                        // s= 1 -> s_opposite = 0
                        const int s_opposite = (is + 1) % 2;

                        const double sign = (is == 0) - (is == 1);

                        for (int m1 = 0; m1 < lmax_at; m1++) {

                            hubbard_energy += sign * atom.type().lo_descriptor_hub(0).Hubbard_beta() *
                                                     om__(m1, m1, is, ia).real();

                            for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
                                hubbard_energy +=
                                    0.5 * atom.type().lo_descriptor_hub(0).Hubbard_J0() *
                                    (om__(m2, m1, is, ia) *
                                     om__(m1, m2, s_opposite, ia)).real();
                            }
                        }
                    }
                }
            }
        }

        if (ctx_.num_mag_dims() != 1) {
            hubbard_energy *= 2.0;
        }

    } else {
        // full hubbard correction
        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {

            auto& atom = this->unit_cell_.atom(ia);

            if (!atom.type().hubbard_correction()) {
                continue;
            }

            // total N and M^2 for the double counting problem

            double n_total = 0.0;
            // n_up and n_down spins
            double n_updown[2] = {0.0, 0.0};

            const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;

            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int m = 0; m < lmax_at; m++) {
                    n_total += om__(m, m, s, ia).real();
                }

                for (int m = 0; m < lmax_at; m++) {
                    n_updown[s] += om__(m, m, s, ia).real();
                }
            }
            double magnetization = 0.0;

            if (ctx_.num_mag_dims() == 0) {
                n_total *= 2.0; // factor two here because the occupations are <= 1
            } else {
                for (int m = 0; m < lmax_at; m++) {
                    magnetization += (om__(m, m, 0, ia) - om__(m, m, 1, ia)).real();
                }
                magnetization *= magnetization;
            }

            hubbard_energy_dc_contribution +=
                0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() * n_total * (n_total - 1.0) - atom.type().lo_descriptor_hub(0).Hubbard_J() * n_total * (0.5 * n_total - 1.0) -
                       0.5 * atom.type().lo_descriptor_hub(0).Hubbard_J() * magnetization);

            // now hubbard contribution

            for (int is = 0; is < ctx_.num_spins(); is++) {
                for (int m1 = 0; m1 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m1++) {

                    // the u contributions
                    for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
                        for (int m3 = 0; m3 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m3++) {
                            for (int m4 = 0; m4 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m4++) {

                                // why should we consider the spinless case

                                hubbard_energy_u +=
                                    0.5 * ((atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) - atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m4, m3)) *
                                               om__(m1, m3, is, ia) * om__(m2, m4, is, ia) +
                                           atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) * om__(m1, m3, is, ia) *
                                               om__(m2, m4, (ctx_.num_mag_dims() == 1) ? ((is + 1) % 2) : (0), ia))
                                              .real();
                            }
                        }
                    }
                }
            }
        }

        if (ctx_.num_mag_dims() == 0) {
            hubbard_energy_u *= 2.0;
        }
        hubbard_energy = hubbard_energy_u - hubbard_energy_dc_contribution;
    }

    if ((ctx_.verbosity() >= 1) && (ctx_.comm().rank() == 0)) {
        std::printf("hub Energy (total) %.5lf  (dc) %.5lf\n", hubbard_energy, hubbard_energy_dc_contribution);
    }
    return hubbard_energy;
}

//void
//Hubbard::calculate_hubbard_potential_and_energy_colinear_case(sddk::mdarray<double_complex, 4> const& om__)
//{
//    this->hubbard_energy_u_               = 0.0;
//    this->hubbard_energy_dc_contribution_ = 0.0;
//    this->hubbard_potential_.zero();
//    if (this->approximation_ == 1) {
//        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
//            const auto& atom = this->unit_cell_.atom(ia);
//            if (atom.type().hubbard_correction()) {
//                double    U_effective = 0.0;
//                const int lmax_at     = 2 * atom.type().lo_descriptor_hub(0).l + 1;
//                if ((atom.type().lo_descriptor_hub(0).Hubbard_U() != 0.0) || (atom.type().lo_descriptor_hub(0).Hubbard_alpha() != 0.0)) {
//
//                    U_effective = atom.type().lo_descriptor_hub(0).Hubbard_U();
//
//                    if (fabs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8)
//                        U_effective -= atom.type().lo_descriptor_hub(0).Hubbard_J0();
//
//                    for (int is = 0; is < ctx_.num_spins(); is++) {
//
//                        // is = 0 up-up
//                        // is = 1 down-down
//
//                        for (int m1 = 0; m1 < lmax_at; m1++) {
//                            this->hubbard_energy_ +=
//                                (atom.type().lo_descriptor_hub(0).Hubbard_alpha() + 0.5 * U_effective) * om__(m1, m1, is, ia).real();
//                            this->hubbard_potential_(m1, m1, is, ia) += (atom.type().lo_descriptor_hub(0).Hubbard_alpha() + 0.5 * U_effective);
//
//                            for (int m2 = 0; m2 < lmax_at; m2++) {
//
//                                this->hubbard_energy_ -=
//                                    0.5 * U_effective *
//                                    (om__(m1, m2, is, ia) * om__(m2, m1, is, ia)).real();
//
//                                // POTENTIAL
//                                this->hubbard_potential_(m1, m2, is, ia) -= U_effective * om__(m2, m1, is, ia);
//                            }
//                        }
//                    }
//                }
//
//                if ((std::abs(atom.type().lo_descriptor_hub(0).Hubbard_J0()) > 1e-8) ||
//                    (std::abs(atom.type().lo_descriptor_hub(0).Hubbard_beta()) > 1e-8)) {
//                    for (int is = 0; is < ctx_.num_spins(); is++) {
//
//                        // s = 0 -> s_opposite = 1
//                        // s= 1 -> s_opposite = 0
//                        const int s_opposite = (is + 1) % 2;
//
//                        const double sign = (is == 0) - (is == 1);
//
//                        for (int m1 = 0; m1 < lmax_at; m1++) {
//
//                            this->hubbard_energy_ += sign * atom.type().lo_descriptor_hub(0).Hubbard_beta() *
//                                                     om__(m1, m1, is, ia).real();
//
//                            this->U(m1, m1, is, ia) += sign * atom.type().lo_descriptor_hub(0).Hubbard_beta();
//
//                            for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
//                                this->hubbard_energy_ +=
//                                    0.5 * atom.type().lo_descriptor_hub(0).Hubbard_J0() *
//                                    (om__(m2, m1, is, ia) *
//                                     om__(m1, m2, s_opposite, ia)).real();
//
//                                this->U(m1, m2, is, ia) += atom.type().lo_descriptor_hub(0).Hubbard_J0() *
//                                    om__(m2, m1, s_opposite, ia);
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        // boring DFT.
//        if (ctx_.num_mag_dims() != 1)
//            this->hubbard_energy_ *= 2.0;
//
//    } else {
//        // full hubbard correction
//        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
//
//            auto& atom = this->unit_cell_.atom(ia);
//
//            if (!atom.type().hubbard_correction()) {
//                continue;
//            }
//
//            // total N and M^2 for the double counting problem
//
//            double n_total = 0.0;
//            // n_up and n_down spins
//            double n_updown[2] = {0.0, 0.0};
//
//            const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;
//
//            for (int s = 0; s < ctx_.num_spins(); s++) {
//                for (int m = 0; m < lmax_at; m++) {
//                    n_total += om__(m, m, s, ia).real();
//                }
//
//                for (int m = 0; m < lmax_at; m++) {
//                    n_updown[s] += om__(m, m, s, ia).real();
//                }
//            }
//            double magnetization = 0.0;
//
//            if (ctx_.num_mag_dims() == 0) {
//                n_total *= 2.0; // factor two here because the occupations are <= 1
//            } else {
//                for (int m = 0; m < lmax_at; m++) {
//                    magnetization += (om__(m, m, 0, ia) - om__(m, m, 1, ia)).real();
//                }
//                magnetization *= magnetization;
//            }
//
//            this->hubbard_energy_dc_contribution_ +=
//                0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() * n_total * (n_total - 1.0) - atom.type().lo_descriptor_hub(0).Hubbard_J() * n_total * (0.5 * n_total - 1.0) -
//                       0.5 * atom.type().lo_descriptor_hub(0).Hubbard_J() * magnetization);
//
//            // now hubbard contribution
//
//            for (int is = 0; is < ctx_.num_spins(); is++) {
//                for (int m1 = 0; m1 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m1++) {
//
//                    // dc contribution
//                    this->U(m1, m1, is, ia) += atom.type().lo_descriptor_hub(0).Hubbard_J() * n_updown[is] +
//                                               0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() - atom.type().lo_descriptor_hub(0).Hubbard_J()) -
//                                               atom.type().lo_descriptor_hub(0).Hubbard_U() * n_total;
//
//                    // the u contributions
//                    for (int m2 = 0; m2 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m2++) {
//                        for (int m3 = 0; m3 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m3++) {
//                            for (int m4 = 0; m4 < 2 * atom.type().lo_descriptor_hub(0).l + 1; m4++) {
//
//                                // why should we consider the spinless case
//
//                                if (ctx_.num_mag_dims() == 0) {
//                                    this->U(m1, m2, is, ia) +=
//                                        2.0 * atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m2, m4) *
//                                        om__(m3, m4, is, ia);
//                                } else {
//                                    // colinear case
//                                    for (int is2 = 0; is2 < 2; is2++) {
//                                        this->U(m1, m2, is, ia) +=
//                                            atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m2, m4) *
//                                            om__(m3, m4, is2, ia);
//                                    }
//                                }
//
//                                this->U(m1, m2, is, ia) -=
//                                    atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m4, m2) * om__(m3, m4, is, ia);
//
//                                this->hubbard_energy_u_ +=
//                                    0.5 * ((atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) - atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m4, m3)) *
//                                               om__(m1, m3, is, ia) * om__(m2, m4, is, ia) +
//                                           atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) * om__(m1, m3, is, ia) *
//                                               om__(m2, m4, (ctx_.num_mag_dims() == 1) ? ((is + 1) % 2) : (0), ia))
//                                              .real();
//                            }
//                        }
//                    }
//                }
//            }
//        }
//
//        // boring DFT
//        if (ctx_.num_mag_dims() == 0) {
//            this->hubbard_energy_u_ *= 2.0;
//        }
//        this->hubbard_energy_ = this->hubbard_energy_u_ - this->hubbard_energy_dc_contribution_;
//    }
//
//    if ((ctx_.control().verbosity_ >= 1) && (ctx_.comm().rank() == 0)) {
//        std::printf("\n hub Energy (total) %.5lf  (dc) %.5lf\n", this->hubbard_energy_, this->hubbard_energy_dc_contribution_);
//    }
//}

void
Hubbard::generate_potential_non_collinear(sddk::mdarray<double_complex, 4> const& om__)
{
    this->hubbard_potential_.zero();
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        if (atom.type().hubbard_correction()) {

            const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;

            // compute the charge and magnetization of the hubbard bands for
            // calculation of the double counting term in the hubbard correction

            double_complex n_total;
            double         mx;
            double         my;
            double         mz;

            n_total = om__(0, 0, 0, ia) + om__(0, 0, 1, ia);
            mz      = (om__(0, 0, 0, ia) - om__(0, 0, 1, ia)).real();
            mx      = (om__(0, 0, 2, ia) + om__(0, 0, 3, ia)).real();
            my      = (om__(0, 0, 2, ia) - om__(0, 0, 3, ia)).imag();

            for (int m = 1; m < lmax_at; m++) {
                n_total += om__(m, m, 0, ia) + om__(m, m, 1, ia);
                mz += (om__(m, m, 0, ia) - om__(m, m, 1, ia)).real();
                mx += (om__(m, m, 2, ia) + om__(m, m, 3, ia)).real();
                my += (om__(m, m, 2, ia) - om__(m, m, 3, ia)).imag();
            }

            mx = n_total.real();

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
                                    this->U(m1, m2, is, ia) +=
                                        atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m2, m4) *
                                        (om__(m3, m4, 0, ia) + om__(m3, m4, 1, ia));
                                }
                            }
                        }
                    }
                }

                // double counting contribution

                double_complex n_aux = 0.0;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    n_aux += om__(m1, m1, is1, ia);
                }

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    // hubbard potential : dc contribution

                    this->U(m1, m1, is, ia) += atom.type().lo_descriptor_hub(0).Hubbard_J() * n_aux;

                    if (is1 == is) {
                        this->U(m1, m1, is, ia) +=
                            0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() - atom.type().lo_descriptor_hub(0).Hubbard_J()) - atom.type().lo_descriptor_hub(0).Hubbard_U() * n_total;
                    }

                    // spin flip contributions
                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        for (int m3 = 0; m3 < lmax_at; m3++) {
                            for (int m4 = 0; m4 < lmax_at; m4++) {
                                this->U(m1, m2, is, ia) -=
                                    atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m3, m4, m2) * om__(m3, m4, is1, ia);
                            }
                        }
                    }
                }
            }
        }
    }
}

double
Hubbard::calculate_energy_non_collinear(sddk::mdarray<double_complex, 4> const& om__) const
{
    double hubbard_energy_dc_contribution{0};
    double hubbard_energy_noflip{0};
    double hubbard_energy_flip{0};
    double hubbard_energy{0};

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        if (atom.type().hubbard_correction()) {

            const int lmax_at = 2 * atom.type().lo_descriptor_hub(0).l + 1;

            // compute the charge and magnetization of the hubbard bands for
            // calculation of the double counting term in the hubbard correction

            double_complex n_total;
            double         mx;
            double         my;
            double         mz;

            n_total = om__(0, 0, 0, ia) + om__(0, 0, 1, ia);
            mz      = (om__(0, 0, 0, ia) - om__(0, 0, 1, ia)).real();
            mx      = (om__(0, 0, 2, ia) + om__(0, 0, 3, ia)).real();
            my      = (om__(0, 0, 2, ia) - om__(0, 0, 3, ia)).imag();

            for (int m = 1; m < lmax_at; m++) {
                n_total += om__(m, m, 0, ia) + om__(m, m, 1, ia);
                mz += (om__(m, m, 0, ia) - om__(m, m, 1, ia)).real();
                mx += (om__(m, m, 2, ia) + om__(m, m, 3, ia)).real();
                my += (om__(m, m, 2, ia) - om__(m, m, 3, ia)).imag();
            }

            double magnetization = mz * mz + mx * mx + my * my;

            mx = n_total.real();
            hubbard_energy_dc_contribution +=
                0.5 * (atom.type().lo_descriptor_hub(0).Hubbard_U() * mx * (mx - 1.0) - atom.type().lo_descriptor_hub(0).Hubbard_J() * mx * (0.5 * mx - 1.0) -
                       0.5 * atom.type().lo_descriptor_hub(0).Hubbard_J() * magnetization);

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
                                        0.5 * ((atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) -
                                                atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m4, m3)) *
                                                   om__(m1, m3, is, ia) * om__(m2, m4, is, ia) +
                                               atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m3, m4) *
                                                   om__(m1, m3, is, ia) * om__(m2, m4, (is + 1) % 2, ia)).real();
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
                                        0.5 * (atom.type().lo_descriptor_hub(0).hubbard_matrix(m1, m2, m4, m3) *
                                            om__(m1, m3, is, ia) * om__(m2, m4, is1, ia)).real();
                                }
                            }
                        }
                    }
                }

                // double counting contribution

                double_complex n_aux = 0.0;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    n_aux += om__(m1, m1, is1, ia);
                }
            }
        }
    }

    hubbard_energy = hubbard_energy_noflip + hubbard_energy_flip - hubbard_energy_dc_contribution;

    if ((ctx_.verbosity() >= 1) && (ctx_.comm().rank() == 0)) {
        std::printf("\n hub Energy (total) %.5lf (no-flip) %.5lf (flip) %.5lf (dc) %.5lf\n",
            hubbard_energy, hubbard_energy_noflip, hubbard_energy_flip, hubbard_energy_dc_contribution);
    }

    return hubbard_energy;
}

/**
 * retrieve or initialize the hubbard potential
 *
 * this functions helps retrieving or setting up the hubbard potential
 * tensors from an external tensor. Retrieving it is done by specifying
 * "get" in the first argument of the method while setting it is done
 * with the parameter set up to "set". The second parameter is the
 * output pointer and the last parameter is the leading dimension of the
 * tensor.
 *
 * The returned result has the same layout than SIRIUS layout, * i.e.,
 * the harmonic orbitals are stored from m_z = -l..l. The occupancy
 * matrix can also be accessed through the method occupation_matrix()
 *
 *
 * @param what__ string to set to "set" for initializing sirius
 * occupancy tensor and "get" for retrieving it
 * @param pointer to external potential tensor
 * @param leading dimension of the outside tensor
 * @return
 * return the potential if the first parameter is set to "get"
 */

void
Hubbard::access_hubbard_potential(std::string const& what__, double_complex* occ__, int ld__)
{
    /* this implementation is QE-specific at the moment */

    if (!(what__ == "get" || what__ == "set")) {
        std::stringstream s;
        s << "wrong access label: " << what__;
        TERMINATE(s);
    }

    mdarray<double_complex, 4> pot_mtrx;

    if (ctx_.num_mag_dims() == 3) {
        pot_mtrx = mdarray<double_complex, 4>(occ__, ld__, ld__, 4, ctx_.unit_cell().num_atoms());
    } else {
        pot_mtrx = mdarray<double_complex, 4>(occ__, ld__, ld__, ctx_.num_spins(), ctx_.unit_cell().num_atoms());
    }
    if (what__ == "get") {
        pot_mtrx.zero();
    }

    auto& potential_matrix = this->potential_matrix();

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {
            const int l = ctx_.unit_cell().atom(ia).type().lo_descriptor_hub(0).l;
            for (int m1 = -l; m1 <= l; m1++) {
                for (int m2 = -l; m2 <= l; m2++) {
                    if (what__ == "get") {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            pot_mtrx(l + m1, l + m2, j, ia) = potential_matrix(l + m1, l + m2, j, ia);
                        }
                    } else {
                        for (int j = 0; j < ((ctx_.num_mag_dims() == 3) ? 4 : ctx_.num_spins()); j++) {
                            potential_matrix(l + m1, l + m2, j, ia) = pot_mtrx(l + m1, l + m2, j, ia);
                        }
                    }
                }
            }
        }
    }
}
}
