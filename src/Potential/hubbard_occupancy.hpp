/// as the name says. We compute the occupation numbers associated to
/// the hubbard wavefunctions (locally centered orbitals, wannier
/// functions, etc) that are relevant for the hubbard correction.

/// These quantities are defined by
///    \f[ n_{m,m'}^I \sigma = \sum_{kv} f(\varepsilon_{kv}) |<\psi_{kv}| phi_I_m>|^2 \f]
/// where \f[m=-l\cdot l$ (same for m')\f], I is the atom.

/// We need to symmetrize them

template <typename T> void hubbard::compute_occupation_numbers_hubbard_orbitals(Kpoint_set &kset_, Q_operator<T>& q_op)
{
  mdarray<T, 4> occupancy_number_(2*this->hubbard_lmax+1, // per atom
                                  2*this->hubbard_lmax+1,
                                  4,
                                  ctx_->unit_cell()->num_atoms());

  T alpha = (std::is_same<T, double_complex>::value) ? 1 : 2;
  T beta = 0;

  for(auto kp_=0; kp_<kset_->num_kpoints(); kp_++) {
    auto kp = kset[kp_];
    const int hubbard_dim = this->number_of_hubbard_orbitals();
    std::vector<mdarray<T, 2>(this->number_of_hubbard_orbitals(), ctx_.num_bands())> oc_num_(4);

    Wavefunctions ophi__(ctx_.processing_unit(), kp->gkvec(), ctx_.num_bands(), 2);
    // need to apply the Overlap operator Copied from apply_h_o since we
    // do not have a dedicated function calculating S|\phi>

    // Compute S|phi_{nk}> and return it in ophi
    for (int i = 0; i < ctx_.beta_projector_chunks().num_chunks(); i++) {
      /* generate beta-projectors for a block of atoms */
      kp__->beta_projectors().generate(i);
      /* non-collinear case */
      for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        auto beta_phi = kp__->beta_projectors().inner<T>(i, kp->spinor_wave_functions(is2).component(ispn), 0, ctx_.num_bands());
        if (ctx_.so_correction()) {
          q_op.apply(i, ispn, ophi__.component(ispn), N__, n__, beta_phi);
          /* apply non-diagonal spin blocks */
          q_op.apply(i, (ispn == 0) ? 3 : 2, ophi__.component((ispn == 0) ? 1 : 0), N__, n__, beta_phi);
        } else {
          /* apply Q operator (diagonal in spin) */
          q_op.apply(i, 0, ophi__.component(ispn), N__, n__, beta_phi);
        }
      }
    }

    // generate the centered atomic orbitals \phi_ik

    generate_local_atomic_orbitals(kset_);

    // now for each spin components and each atom we need to calculate
    // <psi_{nk}|phi^I_m'><phi^I_m|psi_{nk}>

    // split the bands over the different procs of a given kp communicator
    splindex<block> spl_nbnd(nbnd, kp__->comm().size(), kp__->comm().rank());

    // How many bands do I have
    int nbnd_loc = spl_nbnd.local_size();

    if(nbnd_loc) {
      for(int s1 = 0; s1 < ctx_.num_spins(); s1++) {
        for(int s2 = 0; s2 < ctx_.num_spins(); s2++) {
          mdarray<T, 2> dm(hubbard_dim, nbnd_loc);
          // compute <phi_i|psi_nk> the bands been distributed over the pool
          linalg<CPU>::gemm(2,
                            0,
                            hubbard_dim,
                            nbnd_loc,
                            this->phi_.components(s1).pw_coeffs().num_rows_loc(),
                            *reinterpret_cast<double_complex*>(&alpha),
                            this->phi_.components(s1).pw_coeffs().prime().at<CPU>(0, 0), this->phi_.components(s1).pw_coeffs().prime().ld(),
                            ophi__.components(s2).pw_coeffs().prime().at<CPU>(0, spl_nbnd[0]), ophi__.components(s2).pw_coeffs().prime().ld(),
                            *reinterpret_cast<double_complex*>(&beta),
                            reinterpret_cast<double_complex*>(dm.at<CPU>()),
                            dm.ld());

          int s = (s1 == s2) + (s1 != s2) * (2 * s1 + 3 * s2);

#pragma omp parallel for schedule(static)
          for(int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            const atom = unit_cell_.atoms(ia);
            if(atoms.atom_type().hubbard_correction()) {
              for(int nband = 0; nband < nbnd_loc; nband++) {
                for(int m=0;m<2*this->hubbard_l[ia];m++) {
                  for(int mp=m;mp<2*this->hubbard_l[ia];mp++) {
                    this->occupancy_number_(m, mp, s, ia) += conj(dm(this->hubbard_orbital_starting_index[ia] + m, nband)) *
                      dm(this->hubbard_orbital_starting_index[ia] + mp, nband) *
                      kp->occupancies(nband);
                    this->occupancy_number_(mp, m, s, ia) = occupancy_number_(m, mp, s, ia);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  // global reduction
  ctx_.comm().allreduce<T, mpi_op_t::sum>(occupation_number_.at<CPU>(), occupation_number_.size());

  // impose hermiticity of occupation_number_
#pragma omp parallel for schedule(static)
  for(int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
    const atom = unit_cell_.atoms(ia);
    if(atoms.atom_type().hubbard_correction()) {
      for(int s1 = 0; s1 < ctx_.num_spins(); s1++) {
        for(int s2 = 0; s2 < ctx_.num_spins(); s2++) {
          for(int m=0;m<2*this->hubbard_l[ia];m++) {
            for(int mp=m+1;mp<2*this->hubbard_l[ia];mp++) {
              this->occupancy_number_(mp, m, 2 * s1 + s2, ia) = occupation_number_(m, mp, 2 * s1 + s2, ia);
            }
          }
        }
      }
    }
  }

  // Now symmetrization procedure

  auto& sym = unit_cell_.symmetry();

  int lmax = unit_cell_.lmax();
  int lmmax = Utils::lmmax(lmax);

  mdarray<double, 2> rotm(lmmax, lmmax);

  double alpha = 1.0 / double(sym.num_mag_sym());

  for (int i = 0; i < sym.num_mag_sym(); i++) {
    int pr = sym.magnetic_group_symmetry(i).spg_op.proper;
    auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
    int isym = sym.magnetic_group_symmetry(i).isym;
    SHT::rotation_matrix(lmax, eang, pr, rotm);
    auto spin_rot_su2 = SHT::rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

#pragma omp parallel for schedule(static)
    for(int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
      const atom = unit_cell_.atoms(ia);
      if(atoms.atom_type().hubbard_correction()) {

        mdarray<T, 3> rotated_oc(lmmax, lmmax, 4);

        int ja = sym.sym_table(ia, isym);
        rotated_oc.zero();
        for(int s1 = 0; s1 < ctx.num_spins(); s1++) {
          for(int s2 = 0; s2 < ctx.num_spins(); s2++) {
          // symmetrization procedure
            for(int ii=0; ii < 2 * this->hubbard_l[ia] + 1; ii++) {
              for(int ll=0; ll < 2 * this->hubbard_l[ia] + 1; ll++) {
                // A_ij B_jk C_kl
                for(int jj = 0; jj < 2 * this->hubbard_l[ia] + 1; jj++) {
                  for(int kk = 0; kk < 2 * this->hubbard_l[ia] + 1; kk++) {
                    rotated_oc(ii, jj, 2 * s1 + s2) += std::conj(rotm(ll, jj)) * rotm(kk, ll) * occupancy_number_(jj, kk, 2 * s1 + s2, ia) * alpha;
                  }
              }
            }
          }
        }

          if(ctx_.num_mag_dims() == 3) {
          // magnetic symmetrization
          for(int il = 0; il < 2 *this->hubbard_l[ia] + 1; il++) {
            for(int jl = 0; jl < 2 *this->hubbard_l[ia] + 1; jl++) {
              double_complex spin_dm[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
              for(int ii=0;ii<2;ii++) {
                for(int ll=0;ll<2;ll++) {
                  for(int jj = 0; jj < ctx_.num_spins(); jj++) {
                    for(int kk = 0; kk < ctx_.num_spins(); kk++) {
                      int s_index = (jj == kk) + (jj != kk)*(2 * jj + 3 * kk);
                      spin_dm[ii][ll] += rotated_oc(il, jl, s_index) * spin_rot_su2(ii, jj) * std::conj(spin_rot_su2(kk, ll));
                    }
                  }
                }
              }

              for(int ii = 0; ii < 2 *this->hubbard_l[ia] + 1; ii++) {
                for(int jj = 0; jj < 2 *this->hubbard_l[ia] + 1; jj++) {
                  this->occupancy_number_(jj, jj, 0, ia) = spin_dm[0][0]; // up up
                  this->occupancy_number_(jj, jj, 1, ia) = spin_dm[1][1]; // down down
                  this->occupancy_number_(jj, jj, 2, ia) = spin_dm[0][1]; // up down
                  this->occupancy_number_(jj, jj, 3, ia) = spin_dm[1][0]; // down up
                }
              }
            }
          }
        } else {
          // colinear magnetism
            for(int s = 0; s < ctx_.num_spins(); s++) {
              for(int ii = 0; ii < 2 *this->hubbard_l[ia] + 1; ii++) {
                for(int jj = 0; jj < 2 *this->hubbard_l[ia] + 1; jj++) {
                  this->occupancy_number_(jj, jj, s, ia) = rotate_oc(ii, jj, s);
                }
              }
            }
          }
        }
      }
    }
  }
}


// The initial occupancy is calculated following Hund rules

void initialize_hubbard_occupancies()
{

}
