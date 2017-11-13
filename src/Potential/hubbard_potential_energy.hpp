template<typename T> void hubbard::calculate_hubbard_potential_and_energy_colinear_case()
{
  if(this->approximation_ == NAIVE_HUBBARD_CORRECTION) {
#pragma omp parallel for reduction(+ :this->hubbard_energy_)
    for(int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {
      auto atom = this->unit_cell.atoms(ia);
      if(atom.atom_type().hubbard_correction()) {
        double U_effective = 0.0;

        if((atom.atom_type().hubbard_U() != 0.0) || (atom.atom_type().hubbard_alpha() != 0.0)) {

          U_effective = atom.atom_type().hubbard_U();

          if(fabs(atom.atom_type().hubbard_J0()) > 1e-8)
            U_effective -= atom.atom_type().hubbard_J0();

          for(int is=0; is<ctx_.num_spins(); is++) {

            // is = 0 up-up
            // is = 1 down-down

            for(int m1=0; m1 < 2 * atom.atom_type()->hubbard_lmax() + 1; m1++) {
              this->hubbard_energy_ += (atom.atom_type()->hubbard_alpha + 0.5 * U_effective) *
                this->occupancy_number_(m1, m1, is, ia);
            }

            this->hubbard_potential(m1, m1, is, ia) += (atom.atom_type()->hubbard_alpha() + 0.5 * U_effective);

            for(int m2 = 0; m2 < 2 * atom.atom_type()->hubbard_lmax() + 1; m2++) {

              this->hubbard_energy_ -= 0.5 * U_effective *
                this->occupancy_number_(m1, m2, is, ia) *
                this->occupancy_number_(m2, m1, is, ia);

              // POTENTIAL

              this->hubbard_potential(m1, m2, is, ia) -= U_effective *
                this->occupancy_number_(m2, m1, is, ia);
            }
          }
        }

        if((atom.atom_type().hubbard_J0() != 0.0) || (atom.atom_type().hubbard_beta() != 0.0)) {
          for(int is=0; is < ctx.num_spins(); is++) {

            int s_opposite = ctx.num_spins() - 1 - is;
            double sign = (is == 0) - (is == 1);

            for(int m1=0; m1 < 2 * atom.atom_type()->hubbard_lmax() + 1; m1++) {

              this->hubbard_energy_ += sign *
                atom.atom_type().hubbard_beta() *
                this->occupancy_number_(m1, m1, is, ia);

              this->hubbard_potential(m1, m1, is, ia) += sign *
                atom.atom_type().hubbard_beta();

              for(int m2=0; m2 < 2 * atom.atom_type()->hubbard_lmax() + 1; m2++) {
                this->hubbard_energy_ += 0.5 * atom.atom_type()->hubbard_J0() *
                  this->occupancy_number_(m2, m1, is, ia) *
                  this->occupancy_number_(m1, m2, s_opposite, ia);

                this->hubbard_potential(m1, m2, is, ia) += atom.atom_type()->hubbard_J0() *
                  this->occupancy_number_(m2, m1, s_opposite, ia);
              }
            }
          }
        }
      }
    }
    if(ctx_num_spins() == 1)
      this->hubbard_energy_ *= 2.0;

  }  else {

    // full hubbard correction
#pragma omp parallel for reduction(+ :this->hubbard_energy_dc_contribution_) reduction(+: this->hubbard_energy_u_)
    for(int ia = 0; ia < this->unit_cell_.num_atoms(); ia++) {

      auto atom = this->unit_cell.atoms(ia);

      if(!atom.atom_type().hubbard_correction()) {
        continue;
      }

      // total N and M^2 for the double counting problem

      double n_total = 0.0;
      // n_up and n_down spins
      double n_updown[2] = {0.0, 0.0};

      for (int s = 0; s < ctx_.ctx_.num_spins(); s++) {
        for(int m = 0 ; m < 2 * atom.atom_type().hubbard_lmax() + 1; m++) {
          n_total += double_complex::real(this->occupancy_number_(m, m, s, ia));
        }

        for(int m = 0 ; m < 2 * atom.atom_type().hubbard_lmax() + 1; m++) {
          n_updown[s] += double_complex::real(this->occupancy_number_(m, m, s, ia));
        }
      }
      double magnetization = 0.0;

      if(ctx_.ctx_.num_spins() == 1) {
        n_total *= 2.0;
      } else {
        for(int m = 0 ; m < 2 * atom.atom_type().hubbard_lmax() + 1; m++) {
          magnetization += double_complex::real(this->occupancy_number_(m, m, 1, ia) - this->occupancy_number_(m, m, 2, ia));
        }
        magnetization *= magnetization;
      }

      this->hubbard_energy_dc_contribution_ += 0.5 * (atom.atom_type().hubbard_U() * n_total*(n_total - 1.0) -
                                                      atom.atom_type().hubbard_J() * n_total*(0.5 * n_total -1.0)
                                                      - 0.5 * atom.atom_type().hubbard_J()*magnetization);

      // now hubbard contribution

      for (int is = 0; is < ctx_.ctx_.num_spins(); is++) {


        for (int m1 = 0; m1 < 2 * atom.atom_type().hubbard_lmax() + 1; m1++) {
          this->hubbard_potential(m1, m2, is, ia) += atom.atom_type().hubbard_J() * n_updown[s] +
            0.5 * atom.atom_type().hubbard_U_minus_J() - atom.atom_type().hubbard_U() * n_total;

          // the u contributions
          for (int m2 = 0; m2 < 2 * atom.atom_type().hubbard_lmax() + 1; m2++) {
            for (int m3 = 0; m3 < 2 * atom.atom_type().hubbard_lmax() + 1; m3++) {
              for (int m4 = 0; m4 < 2 * atom.atom_type().hubbard_lmax() + 1; m4++) {

                // why should we consider the spinless case

                if ( ctx_.ctx_.num_spins() == 1) {
                  this->hubbard_potential(m1, m2, is, ia) += 2.0 * this->hubbard_matrix_(m1, m3, m2, m4) *
                    this->occupancy_number_(m3, m4, is, ia);
                } else {
                  // colinear case
                  for (int is2 = 0; is2 < 2; is2++) {
                    this->hubbard_potential(m1, m2, is, ia) += this->hubbard_matrix_(m1, m3, m2, m4) *
                      this->occupancy_number_(m3, m4, is2, ia);
                  }
                }

                this->hubbard_potential(m1, m2, is, ia) -= this->hubbard_matrix_(m1, m3, m4, m2) *
                  this->occupancy_number_(m3, m4, is, ia);

                this->hubbard_energy_u_ += 0.5 * double_complex::real((
                                                  ( this->hubbard_matrix_(m1, m2, m3, m4) - this->hubbard_matrix_(m1, m2, m4, m3)) *
                                                  this->occupancy_number_(m1, m3, is, ia) *
                                                  this->occupancy_number_(m2, m4, is, ia) +
                                                  this->hubbard_matrix_(m1, m2, m3, m4) *
                                                  this->occupancy_number_(m1, m3, is, ia) *
                                                  this->occupancy_number_(m2, m4, ctx.num_spins() - 1 - is, ia)));
              }
            }
          }
        }
      }
    }
    if(ctx.num_spins() == 1) {
      this->hubbard_energy_u_ *= 2.0;
    }

    this->hubbard_energy_ = this->hubbard_energy_u_ - this->hubbard_energy_dc_contribution_;
  }
}

void calculate_hubbard_potential_and_energy_non_colinear_case()
{
  for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
    auto atom = unit_cell_.atoms(ia);
    if(atom.hubbard_correction()) {
      // compute the charge and magnetization of the hubbard bands for
      // calculation of the double counting term in the hubbard correction

      double n_total = 0.0;
      double mx = 0.0;
      double my = 0.0;
      double mz = 0.0;

      for(int m = 0; m < 2 * atom.hubbard_lmax() + 1; m++) {
        n_total += double_complex::real(this->occupancy_number_(m, m, 0, ia) + this->occupancy_number_(m, m, 1, ia));
        mz += this->occupancy_number_(m, m, 0, ia) - this->occupancy_number_(m, m, 1, ia);
        mx += double_complex::real(this->occupancy_number_(m, m, 2, ia) + this->occupancy_number_(m, m, 3, ia));
        my += double_complex::imag(this->occupancy_number_(m, m, 2, ia) - this->occupancy_number_(m, m, 3, ia));
      }

      double magnetization = mz * mz + mx * mx + my * my;

      this->hubbard_energy_dc_contribution_ += 0.5 * ( atom.hubbard_U() * mx * (mx - 1.0) -
                                                       atom.hubbard_J() * mx * (0.5 * mx - 1.0) -
                                                       0.5 * atom.hubbard_J() * magnetization);

      for(int is = 0; is < 4; is++) {

        // diagonal elements of n^{\sigma\sigma'}
        int is1 = is;

        // if off-diagonal
        if(is == 2)
          is1 = 3;

        if( is == 3)
          is1 = 2;

        if(is1 == is) {
          // non spin flip contributions for the hubbard energy
          for (int m1 = 0; m1 < 2 * atom.hubbard_lmax() + 1; ++m1) {
            for (int m2 = 0; m2 < 2 * atom.hubbard_lmax() + 1; ++m2) {
              for (int m3 = 0; m3 < 2 * atom.hubbard_lmax() + 1; ++m3) {
                for (int m4 = 0; m4 < 2 * atom.hubbard_lmax() + 1; ++m4) {

                  this->hubbard_energy_noflip_ += 0.5 * (
                                                         ( atom.atom_type().hubbard_matrix(m1, m2, m3, m4) - atom.atom_type().hubbard_matrix(m1, m2, m4, m3)) *
                                                         this->occupancy_number_(m1, m3, is, ia) *
                                                         this->occupancy_number_(m2, m4, is, ia) +
                                                         atom.atom_type().hubbard_matrix(m1, m2, m3, m4) *
                                                         this->occupancy_number_(m1, m3, is, ia) *
                                                         this->occupancy_number_(m2, m4, 2 - is - 1, ia)
                                                         );
                }
              }
            }
          }
        } else {
          for (int m1 = 0; m1 < 2 * atom.hubbard_lmax() + 1; ++m1) {
            for (int m2 = 0; m2 < 2 * atom.hubbard_lmax() + 1; ++m2) {
              for (int m3 = 0; m3 < 2 * atom.hubbard_lmax() + 1; ++m3) {
                for (int m4 = 0; m4 < 2 * atom.hubbard_lmax() + 1; ++m4) {
                  this->hubbard_energy_flip_ -= 0.5 * atom.atom_type().hubbard_matrix(m1, m2, m4, m3) *
                    this->occupancy_number_(m1, m3, is, ia) *
                    this->occupancy_number_(m2, m4, is1, ia);
                }
              }
            }
          }
        }

        // same thing for the hubbard potential
        if (is1 == is) {
          for (int m1 = 0; m1 < 2 * atom.hubbard_lmax() + 1; ++m1) {
            for (int m2 = 0; m2 < 2 * atom.hubbard_lmax() + 1; ++m2) {
              for (int m3 = 0; m3 < 2 * atom.hubbard_lmax() + 1; ++m3) {
                for (int m4 = 0; m4 < 2 * atom.hubbard_lmax() + 1; ++m4) {
                  this->hubbard_potential_(m1, m2, is, ia) += atom.atom_type().hubbard_matrix(m1, m3, m2, m4) *
                    (this->occupancy_number_(m3, m4, 0, ia) + this->occupancy_number_(m3, m4, 1, ia));
                }
              }
            }
          }
        }

        // double counting contribution

        double n_aux = 0.0;
        for (int m1 = 0; m1 < 2 * atom.hubbard_lmax() + 1; ++m1) {
          n_aux += this->occupancy_number_(m1, m1, is1, ia));
        }

      for (int m1 = 0; m1 < 2 * atom.hubbard_lmax() + 1; ++m1) {

        // hubbard potential : dc contribution

        this->hubbard_potential_(m1, m1, is, ia) += atom.atom_type().hubbard_J() * n_aux;

        if(is1 == is) {
          this->hubbard_potential_(m1, m1, is, ia) += 0.5*(atom.atom_type().hubbard_U() - atom.atom_type().hubbard_J()) -
            atom.atom_type().hubbard_U() * n_total;
        }

        // spin flip contributions
        for (int m2 = 0; m2 < 2 * atom.hubbard_lmax() + 1; ++m2) {
          for (int m3 = 0; m3 < 2 * atom.hubbard_lmax() + 1; ++m3) {
            for (int m4 = 0; m4 < 2 * atom.hubbard_lmax() + 1; ++m4) {
              this->hubbard_potential_(m1, m2, is, ia) -= atom.atom_type().hubbard_matrix(m1, m3, m4, m2) *
                this->occupancy_number_(m3, m4, is1, ia);
            }
          }
        }
      }
    }
  }

  this->hubbard_energy_ = this->hubbard_energy_noflip_ +
    this->hubbard_energy_flip_ -
    this->hubbard_energy_dc_contribution_;
}
