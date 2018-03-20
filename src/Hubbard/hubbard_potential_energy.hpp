void calculate_hubbard_potential_and_energy_colinear_case()
{
    this->hubbard_energy_u_               = 0.0;
    this->hubbard_energy_dc_contribution_ = 0.0;
    this->hubbard_potential_.zero();
    if (this->approximation_ == 1) {
        for (int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
            const auto& atom = this->unit_cell_.atom(ia);
            if (atom.type().hubbard_correction()) {
                double U_effective = 0.0;

                if ((atom.type().Hubbard_U() != 0.0) || (atom.type().Hubbard_alpha() != 0.0)) {

                    U_effective = atom.type().Hubbard_U();

                    if (fabs(atom.type().Hubbard_J0()) > 1e-8)
                        U_effective -= atom.type().Hubbard_J0();

                    for (int is = 0; is < ctx_.num_spins(); is++) {

                        // is = 0 up-up
                        // is = 1 down-down

                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                            this->hubbard_energy_ += (atom.type().Hubbard_alpha() + 0.5 * U_effective) *
                                this->occupancy_number_(m1, m1, is, ia, 0).real();
                            this->U(m1, m1, is, ia) += (atom.type().Hubbard_alpha() + 0.5 * U_effective);

                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {

                                this->hubbard_energy_ -=
                                    0.5 * U_effective *
                                    (this->occupancy_number_(m1, m2, is, ia, 0) * this->occupancy_number_(m2, m1, is, ia, 0))
                                        .real();

                                // POTENTIAL

                                this->U(m1, m2, is, ia) -= U_effective * this->occupancy_number_(m2, m1, is, ia, 0);
                            }
                        }
                    }
                }

                if ((atom.type().Hubbard_J0() != 0.0) || (atom.type().Hubbard_beta() != 0.0)) {
                    for (int is = 0; is < ctx_.num_spins(); is++) {

                        // s = 0 -> s_opposite = 1
                        // s= 1 -> s_opposite = 0
                        int s_opposite = (is + 1) % 2;

                        double sign = (is == 0) - (is == 1);

                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {

                            this->hubbard_energy_ +=
                                sign * atom.type().Hubbard_beta() * this->occupancy_number_(m1, m1, is, ia, 0).real();

                            this->U(m1, m1, is, ia) += sign * atom.type().Hubbard_beta();

                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                this->hubbard_energy_ += 0.5 * atom.type().Hubbard_J0() *
                                    (this->occupancy_number_(m2, m1, is, ia, 0) *
                                     this->occupancy_number_(m1, m2, s_opposite, ia, 0)).real();

                                this->U(m1, m2, is, ia) +=
                                    atom.type().Hubbard_J0() * this->occupancy_number_(m2, m1, s_opposite, ia, 0);
                            }
                        }
                    }
                }
            }
        }

        // boring DFT.
        if (ctx_.num_mag_dims() != 1)
            this->hubbard_energy_ *= 2.0;

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

            for (int s = 0; s < ctx_.num_spins(); s++) {
                for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                    n_total += this->occupancy_number_(m, m, s, ia, 0).real();
                }

                for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                    n_updown[s] += this->occupancy_number_(m, m, s, ia, 0).real();
                }
            }
            double magnetization = 0.0;

            if (ctx_.num_mag_dims() != 1) {
                n_total *= 2.0; // factor two here because the occupations are <= 1
            } else {
                for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                    magnetization +=
                        (this->occupancy_number_(m, m, 0, ia, 0) - this->occupancy_number_(m, m, 1, ia, 0)).real();
                }
                magnetization *= magnetization;
            }

            this->hubbard_energy_dc_contribution_ += 0.5 * (atom.type().Hubbard_U() * n_total * (n_total - 1.0) -
                                                            atom.type().Hubbard_J() * n_total * (0.5 * n_total - 1.0) -
                                                            0.5 * atom.type().Hubbard_J() * magnetization);

            // now hubbard contribution

            for (int is = 0; is < ctx_.num_spins(); is++) {
                for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                    this->U(m1, m1, is, ia) += atom.type().Hubbard_J() * n_updown[is] +
                                               0.5 * (atom.type().Hubbard_U() - atom.type().Hubbard_J()) -
                                               atom.type().Hubbard_U() * n_total;

                    // the u contributions
                    for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                        for (int m3 = 0; m3 < 2 * atom.type().hubbard_l() + 1; m3++) {
                            for (int m4 = 0; m4 < 2 * atom.type().hubbard_l() + 1; m4++) {

                                // why should we consider the spinless case

                                if (ctx_.num_mag_dims() != 1) {
                                    this->U(m1, m2, is, ia) += 2.0 * atom.type().hubbard_matrix(m1, m3, m2, m4) *
                                        this->occupancy_number_(m3, m4, is, ia, 0);
                                } else {
                                    // colinear case
                                    for (int is2 = 0; is2 < 2; is2++) {
                                        this->U(m1, m2, is, ia) += atom.type().hubbard_matrix(m1, m3, m2, m4) *
                                            this->occupancy_number_(m3, m4, is2, ia, 0);
                                    }
                                }

                                this->U(m1, m2, is, ia) -= atom.type().hubbard_matrix(m1, m3, m4, m2) *
                                    this->occupancy_number_(m3, m4, is, ia, 0);

                                this->hubbard_energy_u_ += 0.5 *
                                    ((atom.type().hubbard_matrix(m1, m2, m3, m4) -
                                      atom.type().hubbard_matrix(m1, m2, m4, m3)) *
                                     this->occupancy_number_(m1, m3, is, ia, 0) *
                                     this->occupancy_number_(m2, m4, is, ia, 0) +
                                     atom.type().hubbard_matrix(m1, m2, m3, m4) *
                                     this->occupancy_number_(m1, m3, is, ia, 0) *
                                     this->occupancy_number_(m2, m4, (ctx_.num_mag_dims() == 1) ? ((is + 1) % 2) : (0), ia, 0))
                                    .real();
                            }
                        }
                    }
                }
            }
        }

        // boring DFT
        if (ctx_.num_mag_dims() != 1) {
            this->hubbard_energy_u_ *= 2.0;
        }

        this->hubbard_energy_ = this->hubbard_energy_u_ - this->hubbard_energy_dc_contribution_;
        if ((ctx_.control().verbosity_ >= 1) && (ctx_.comm().rank() == 0)) {
            printf("\n hub Energy (total) %.5lf (no-flip) %.5lf (flip) %.5lf (dc) %.5lf\n", this->hubbard_energy_,
                   this->hubbard_energy_noflip_, this->hubbard_energy_flip_, this->hubbard_energy_dc_contribution_);
        }
    }
}

void calculate_hubbard_potential_and_energy_non_colinear_case()
{
    this->hubbard_potential_.zero();
    this->hubbard_energy_dc_contribution_ = 0.0;
    this->hubbard_energy_noflip_          = 0.0;
    this->hubbard_energy_flip_            = 0.0;
    this->hubbard_potential_.zero();
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom = unit_cell_.atom(ia);
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
        if (atom.type().hubbard_correction()) {
            // compute the charge and magnetization of the hubbard bands for
            // calculation of the double counting term in the hubbard correction

            double_complex n_total;
            double mx;
            double my;
            double mz;

            n_total = this->occupancy_number_(0, 0, 0, ia, 0) + this->occupancy_number_(0, 0, 1, ia, 0);
            mz      = (this->occupancy_number_(0, 0, 0, ia, 0) - this->occupancy_number_(0, 0, 1, ia, 0)).real();
            mx      = (this->occupancy_number_(0, 0, 2, ia, 0) + this->occupancy_number_(0, 0, 3, ia, 0)).real();
            my      = (this->occupancy_number_(0, 0, 2, ia, 0) - this->occupancy_number_(0, 0, 3, ia, 0)).imag();

            for (int m = 1; m < lmax_at; m++) {
                n_total += this->occupancy_number_(m, m, 0, ia, 0) + this->occupancy_number_(m, m, 1, ia, 0);
                mz += (this->occupancy_number_(m, m, 0, ia, 0) - this->occupancy_number_(m, m, 1, ia, 0)).real();
                mx += (this->occupancy_number_(m, m, 2, ia, 0) + this->occupancy_number_(m, m, 3, ia, 0)).real();
                my += (this->occupancy_number_(m, m, 2, ia, 0) - this->occupancy_number_(m, m, 3, ia, 0)).imag();
            }

            double magnetization = mz * mz + mx * mx + my * my;

            mx = n_total.real();
            this->hubbard_energy_dc_contribution_ += 0.5 *
                (atom.type().Hubbard_U() * mx * (mx - 1.0) - atom.type().Hubbard_J() * mx * (0.5 * mx - 1.0) -
                 0.5 * atom.type().Hubbard_J() * magnetization);

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

                                    this->hubbard_energy_noflip_ +=
                                        0.5 * ((atom.type().hubbard_matrix(m1, m2, m3, m4) -
                                                atom.type().hubbard_matrix(m1, m2, m4, m3)) *
                                               this->occupancy_number_(m1, m3, is, ia, 0) *
                                               this->occupancy_number_(m2, m4, is, ia, 0) +
                                               atom.type().hubbard_matrix(m1, m2, m3, m4) *
                                               this->occupancy_number_(m1, m3, is, ia, 0) *
                                               this->occupancy_number_(m2, m4, (is + 1) % 2, ia, 0))
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
                                    this->hubbard_energy_flip_ -= 0.5 * (atom.type().hubbard_matrix(m1, m2, m4, m3) *
                                                                         this->occupancy_number_(m1, m3, is, ia, 0) *
                                                                         this->occupancy_number_(m2, m4, is1, ia, 0))
                                        .real();
                                }
                            }
                        }
                    }
                }

                // same thing for the hubbard potential
                if (is1 == is) {
                    // non spin flip
                    for (int m1 = 0; m1 < lmax_at; ++m1) {
                        for (int m2 = 0; m2 < lmax_at; ++m2) {
                            for (int m3 = 0; m3 < lmax_at; ++m3) {
                                for (int m4 = 0; m4 < lmax_at; ++m4) {
                                    this->U(m1, m2, is, ia) += atom.type().hubbard_matrix(m1, m3, m2, m4) *
                                                               (this->occupancy_number_(m3, m4, 0, ia, 0) +
                                                                this->occupancy_number_(m3, m4, 1, ia, 0));
                                }
                            }
                        }
                    }
                }

                // double counting contribution

                double_complex n_aux = 0.0;
                for (int m1 = 0; m1 < lmax_at; m1++) {
                    n_aux += this->occupancy_number_(m1, m1, is1, ia, 0);
                }

                for (int m1 = 0; m1 < lmax_at; m1++) {

                    // hubbard potential : dc contribution

                    this->U(m1, m1, is, ia) += atom.type().Hubbard_J() * n_aux;

                    if (is1 == is) {
                        this->U(m1, m1, is, ia) += 0.5 * (atom.type().Hubbard_U() - atom.type().Hubbard_J()) -
                            atom.type().Hubbard_U() * n_total;
                    }

                    // spin flip contributions
                    for (int m2 = 0; m2 < lmax_at; m2++) {
                        for (int m3 = 0; m3 < lmax_at; m3++) {
                            for (int m4 = 0; m4 < lmax_at; m4++) {
                                this->U(m1, m2, is, ia) -= atom.type().hubbard_matrix(m1, m3, m4, m2) *
                                    this->occupancy_number_(m3, m4, is1, ia, 0);
                            }
                        }
                    }
                }
            }
        }
    }

    // this->hubbard_energy_noflip_ *= 0.5;
    // this->hubbard_energy_flip_ *= 0.5;
    // this->hubbard_energy_dc_contribution_ *= 0.5;
    this->hubbard_energy_ =
        this->hubbard_energy_noflip_ + this->hubbard_energy_flip_ - this->hubbard_energy_dc_contribution_;

    //    if ((ctx_.control().verbosity_ >= 1) && (ctx_.comm().rank() == 0)) {
        printf("\n hub Energy (total) %.5lf (no-flip) %.5lf (flip) %.5lf (dc) %.5lf\n", this->hubbard_energy_,
               this->hubbard_energy_noflip_, this->hubbard_energy_flip_, this->hubbard_energy_dc_contribution_);
        // }
}

inline void set_hubbard_potential(double *occ, int ld)
{
    mdarray<double, 4> occupation_(occ, ld, ld, ctx_.num_spins(), ctx_.unit_cell().num_atoms());

    this->hubbard_potential_.zero();
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l ; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++)
                    this->hubbard_potential_(l + m1, l + m2, ispn, ia, 0) = 0.5 * occupation_(mm1, mm2, ispn, ia);
            }
        }
    }
}

inline void set_hubbard_potential_nc(double_complex *occ, int ld)
{
    mdarray<double_complex, 4> occupation_(occ, ld, ld, 4, ctx_.unit_cell().num_atoms());
    //this->calculate_hubbard_potential_and_energy();
    this->hubbard_potential_.zero();
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                this->hubbard_potential_(l + m1, l + m2, 0, ia, 0) = 0.5 * occupation_(mm1, mm2, 0, ia);
                this->hubbard_potential_(l + m1, l + m2, 1, ia, 0) = 0.5 * occupation_(mm1, mm2, 3, ia);
                this->hubbard_potential_(l + m1, l + m2, 2, ia, 0) = 0.5 * occupation_(mm1, mm2, 1, ia);
                this->hubbard_potential_(l + m1, l + m2, 3, ia, 0) = 0.5 * occupation_(mm1, mm2, 2, ia);
            }
        }
    }
}
