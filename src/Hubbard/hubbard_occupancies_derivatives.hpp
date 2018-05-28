// compute the forces for the simplex LDA+U method not the fully
// rotationally invariant one. It can not be used for LDA+U+SO either

// It is based on this reference : PRB 84, 161102(R) (2011)

// gradient of beta projectors. Needed for the computations of the forces

void Hubbard_potential::compute_occupancies_derivatives(K_point &kp,
                                                        Wave_functions &phi, // hubbard derivatives
                                                        Beta_projectors_gradient &bp_grad_,
                                                        Q_operator<double_complex>& q_op, // Compensnation operator or overlap operator
                                                        mdarray<double_complex, 5> &dn_, // derivative of the occupation number compared to displacement of atom aton_id
                                                        const int atom_id) // Atom we shift
{
  dn_.zero();
  // check if we have a norm conserving pseudo potential only. OOnly
  // derivatives of the hubbard wave functions are needed.

  bool augment = false;

  for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
    augment = ctx_.unit_cell().atom_type(ia).augment();
  }

  if ((ctx_.full_potential() || !augment) && (!ctx_.unit_cell().atom(atom_id).type().hubbard_correction())) {
    // return immediatly if the atom has no hubbard correction and is norm conserving pp.
    return;
  }

  // Compute the derivatives of the occupancies in two cases.

  //- the atom is pp norm conserving or

  // - the atom is ppus (in that case the derivative the beta projectors
  // compared to the atomic displacements gives a non zero contribution)

  // temporary wave functions
  Wave_functions phitmp(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);
  Wave_functions dphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);

  int HowManyBands = kp.num_occupied_bands(0);
  if (ctx_.num_spins() == 2)
    HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));

  // d_phitmp contains the derivatives of the hubbard wave functions
  // corresponding to the displacement r^I_a.

  dmatrix<double_complex> dPhi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
  dmatrix<double_complex> Phi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());

  Phi_S_Psi.zero();
  dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phi, 0, 0, 0, 0);

  // computes the S|phi^I_ia>
  if (!ctx_.full_potential() && augment) {
    for (int i = 0; i < kp.beta_projectors().num_chunks(); i++) {
      /* generate beta-projectors for a block of atoms */
      kp.beta_projectors().generate(i);
      /* non-collinear case */
      auto beta_phi = kp.beta_projectors().inner<double_complex>(i, phi, 0, 0, this->number_of_hubbard_orbitals());
      /* apply Q operator (diagonal in spin) */
      q_op.apply(i,
                 0,
                 dphi,
                 0,
                 this->number_of_hubbard_orbitals(),
                 kp.beta_projectors(),
                 beta_phi);
    }
  }

  // compute <phi^I_m| S | psi_{nk}>
  for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    inner(ctx_.processing_unit(),
          ispn,
          kp.spinor_wave_functions(),
          0,
          kp.num_occupied_bands(ispn),
          dphi,
          0,
          this->number_of_hubbard_orbitals(),
          Phi_S_Psi,
          0,
          ispn * this->number_of_hubbard_orbitals());
  }

  for (int dir = 0; dir < 3; dir++) {
    // reset dphi
    dphi.pw_coeffs(0).prime().zero();

    if (ctx_.unit_cell().atom(atom_id).type().hubbard_correction()) {
      // atom atom_id has hubbard correction so we need to compute the
      // derivatives of the hubbard orbitals associated to the atom
      // atom_id, the derivatives of the others hubbard orbitals been
      // zero compared to the displacement of atom atom_id

      // compute the derivative of |phi> corresponding to the
      // atom atom_id
      const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().hubbard_l() + 1;

      // compute the derivatives of the hubbard wave functions
      // |phi_m^J> (J = atom_id) compared to a displacement of atom J.

      kp.compute_gradient_wavefunctions(phi,
                                        this->offset[atom_id],
                                        lmax_at,
                                        dphi,
                                        this->offset[atom_id],
                                        dir);
      // For norm conserving pp, it is enough to have the derivatives
      // of |phi^J_m> (J = atom_id)

      if (!ctx_.full_potential() && augment) {
        phitmp.copy_from(ctx_.processing_unit(),
                         0,
                         dphi,
                         this->offset[atom_id],
                         lmax_at,
                         0,
                         0);
            // for the ppus potential we have an additional term coming from the
      // derivatives of the overlap matrix.
      // need to apply S on dphi^I

      for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
        kp.beta_projectors().generate(chunk__);
        // S| dphi> for this chunk
        const int lmax_at = 2 * ctx_.unit_cell().atom(atom_id).type().hubbard_l() + 1;
        auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__,
                                                                   phi,
                                                                   0,
                                                                   this->number_of_hubbard_orbitals(),
                                                                   lmax_at);
        /* apply Q operator (diagonal in spin) */
        q_op.apply(chunk__,
                   0,
                   dphi,
                   0,
                   this->number_of_hubbard_orbitals(),
                   kp.beta_projectors(),
                   beta_phi);
      }
    }

    // compute d S/ dr^I_a |phi>
    if (!ctx_.full_potential() && augment) {
      // it is equal to
      // \sum Q^I_ij <d \beta^I_i|phi> \beta^I_j + < \beta^I_i|phi> |d\beta^I_j>
      for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
        for (int i = 0; i < kp.beta_projectors().chunk(chunk__).num_atoms_; i++) {
          // need to find the right atom in the chunks.
          if  (kp.beta_projectors().chunk(chunk__).desc_(beta_desc_idx::ia, i) == atom_id) {
            kp.beta_projectors().generate(chunk__);
            bp_grad_.generate(chunk__, dir);

            // < beta | phi> for this chunk
            auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

            /* apply Q operator (diagonal in spin) */
            /* compute Q_ij <d beta_i|phi> |beta_j> */
            q_op.apply_one_atom(chunk__,
                                0,
                                dphi,
                                0,
                                this->number_of_hubbard_orbitals(),
                                bp_grad_,
                                beta_phi,
                                i);
          }
        }
      }

      for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
        for (int i = 0; i < kp.beta_projectors().chunk(chunk__).num_atoms_; i++) {
          // need to find the right atom in the chunks.
          if  (kp.beta_projectors().chunk(chunk__).desc_(beta_desc_idx::ia, i) == atom_id) {
            kp.beta_projectors().generate(chunk__);
            bp_grad_.generate(chunk__, dir);
            // < dbeta | phi> for this chunk
            auto dbeta_phi = bp_grad_.inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

            /* apply Q operator (diagonal in spin) */
            /* Effectively compute Q_ij <dbeta_i| phi> |beta_j>*/
            q_op.apply_one_atom(chunk__, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                                dbeta_phi, i);

          }
        }
      }
    }
    // it is actually <psi | d(S|phi>)
    dPhi_S_Psi.zero();

    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
      inner(ctx_.processing_unit(),
            ispn,
            kp.spinor_wave_functions(),
            0,
            kp.num_occupied_bands(ispn),
            dphi, //   S d |phi>
            0,
            this->number_of_hubbard_orbitals(),
            dPhi_S_Psi,
            0,
            ispn * this->number_of_hubbard_orbitals());
    }

    #pragma omp parallel for
    for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
      const auto& atom = ctx_.unit_cell().atom(ia1);
      if (atom.type().hubbard_correction()) {
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
          const size_t ispn_offset = ispn * this->number_of_hubbard_orbitals() + this->offset[ia1];
          for (int m1 = 0; m1 < lmax_at; m1++) {
            for (int m2 = 0; m2 < lmax_at; m2++) {
              for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                // d n_{m,m'}^I = \sum_{nk} <psi_{nk}|d phi_m><phi_m'|psi_nk> + <psi_{nk}|phi_m><d phi_m'|psi_nk>
                dn_(m1, m2, ispn, ia1, dir) +=  (
                                                 Phi_S_Psi(nbnd, ispn_offset + m1) * std::conj(dPhi_S_Psi(nbnd, ispn_offset + m2)) +
                                                 dPhi_S_Psi(nbnd, ispn_offset + m1) * std::conj(Phi_S_Psi(nbnd, ispn_offset + m2))
                                                 ) * kp.weight() * kp.band_occupancy(nbnd, ispn);
              }
            }
          }
        }
      }
    }
  }
}

void Hubbard_potential::compute_occupancies_stress_derivatives(K_point &kp,
                                                               Beta_projectors_strain_deriv &bp_grad_,
                                                               Q_operator<double_complex>& q_op, // Compensnation operator or overlap operator
                                                               mdarray<double_complex, 5> &dn_) // derivative of the occupation number compared to displacement of atom aton_id
{
  dn_.zero();
  // check if we have a norm conserving pseudo potential only. OOnly
  // derivatives of the hubbard wave functions are needed.

  Wave_functions phitmp(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);
  Wave_functions dphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);
  Wave_functions phi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), 1);

  kp.generate_atomic_centered_wavefunctions_(this->number_of_hubbard_orbitals(),
                                             phi,
                                             this->offset,
                                             true);


  bool augment = false;

  for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
    augment = ctx_.unit_cell().atom_type(ia).augment();
  }

  int lmax = ctx_.unit_cell().lmax();
  int lmmax = Utils::lmmax(lmax);

  mdarray<double, 2> rlm_g(lmmax, kp.num_gkvec_loc());
  mdarray<double, 3> rlm_dg(lmmax, 3, kp.num_gkvec_loc());

  /* array of real spherical harmonics and derivatives for each G-vector */
  #pragma omp parallel for schedule(static)
  for (int igkloc = 0; igkloc < kp.num_gkvec_loc(); igkloc++) {
    auto gvc = kp.gkvec().gkvec_cart(kp.idxgk(igkloc));
    auto rtp = SHT::spherical_coordinates(gvc);

    double theta = rtp[1];
    double phi   = rtp[2];

    SHT::spherical_harmonics(lmax, theta, phi, &rlm_g(0, igkloc));
    mdarray<double, 2> rlm_dg_tmp(&rlm_dg(0, 0, igkloc), lmmax, 3);
    SHT::dRlm_dr(lmax, gvc, rlm_dg_tmp);
  }

  // Compute the derivatives of the occupancies in two cases.

  //- the atom is pp norm conserving or

  // - the atom is ppus (in that case the derivative the beta projectors
  // compared to the atomic displacements gives a non zero contribution)

  // temporary wave functions

  int HowManyBands = kp.num_occupied_bands(0);
  if (ctx_.num_spins() == 2)
    HowManyBands = std::max(kp.num_occupied_bands(1), kp.num_occupied_bands(0));

  // d_phitmp contains the derivatives of the hubbard wave functions
  // corresponding to the distortion epsilon_alphabeta.

  dmatrix<double_complex> dPhi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());
  dmatrix<double_complex> Phi_S_Psi(HowManyBands, this->number_of_hubbard_orbitals() * ctx_.num_spins());

  Phi_S_Psi.zero();
  dphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), phi, 0, 0, 0, 0);

  // computes the S|phi^I_ia>
  if (!ctx_.full_potential() && augment) {
    for (int i = 0; i < kp.beta_projectors().num_chunks(); i++) {
      /* generate beta-projectors for a block of atoms */
      kp.beta_projectors().generate(i);
      /* non-collinear case */
      auto beta_phi = kp.beta_projectors().inner<double_complex>(i, phi, 0, 0, this->number_of_hubbard_orbitals());
      /* apply Q operator (diagonal in spin) */
      q_op.apply(i,
                 0,
                 dphi,
                 0,
                 this->number_of_hubbard_orbitals(),
                 kp.beta_projectors(),
                 beta_phi);
    }
  }

  // compute <phi^I_m| S | psi_{nk}>
  for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
    inner(ctx_.processing_unit(),
          ispn,
          kp.spinor_wave_functions(),
          0,
          kp.num_occupied_bands(ispn),
          dphi,
          0,
          this->number_of_hubbard_orbitals(),
          Phi_S_Psi,
          0,
          ispn * this->number_of_hubbard_orbitals());
  }

  for (int mu = 0; mu < 3; mu++) {
    for (int nu = 0; nu < 3; nu++) {

      // atom atom_id has hubbard correction so we need to compute the
      // derivatives of the hubbard orbitals associated to the atom
      // atom_id, the derivatives of the others hubbard orbitals been
      // zero compared to the displacement of atom atom_id

      // compute the derivatives of all hubbard wave functions
      // |phi_m^J> compared to the strain

      compute_gradient_strain_wavefunctions(kp,
                                            dphi,
                                            rlm_g,
                                            rlm_dg,
                                            mu,
                                            nu);

      if (!ctx_.full_potential() && augment) {
        phitmp.copy_from(ctx_.processing_unit(),
                         this->number_of_hubbard_orbitals(),
                         dphi,
                         0,
                         0,
                         0,
                         0);
        // for the ppus potential we have an additional term coming from the
        // derivatives of the overlap matrix.
        // need to apply S on dphi^I

        for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
          kp.beta_projectors().generate(chunk__);
          // S| dphi> for this chunk
          auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__,
                                                                     phitmp,
                                                                     0,
                                                                     0,
                                                                     this->number_of_hubbard_orbitals());
          /* apply Q operator (diagonal in spin) */
          q_op.apply(chunk__,
                     0,
                     dphi,
                     0,
                     this->number_of_hubbard_orbitals(),
                     kp.beta_projectors(),
                     beta_phi);
        }
      }

      // compute d S/ dr^I_a |phi>
      if (!ctx_.full_potential() && augment) {
        // it is equal to
        // \sum Q^I_ij <d \beta^I_i|phi> \beta^I_j + < \beta^I_i|phi> |d\beta^I_j>
        for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
          kp.beta_projectors().generate(chunk__);
          bp_grad_.generate(chunk__, 3 * mu + nu);

          // < beta | phi> for this chunk
          auto beta_phi = kp.beta_projectors().inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

          /* apply Q operator (diagonal in spin) */
          /* compute Q_ij <d beta_i|phi> |beta_j> */
          q_op.apply(chunk__,
                     0,
                     dphi,
                     0,
                     this->number_of_hubbard_orbitals(),
                     bp_grad_,
                     beta_phi);
        }
      }

      for (int chunk__ = 0; chunk__ < kp.beta_projectors().num_chunks(); chunk__++) {
        kp.beta_projectors().generate(chunk__);
        bp_grad_.generate(chunk__, 3 * mu + nu);
        // < dbeta | phi> for this chunk
        auto dbeta_phi = bp_grad_.inner<double_complex>(chunk__, phi, 0, 0, this->number_of_hubbard_orbitals());

        /* apply Q operator (diagonal in spin) */
        q_op.apply(chunk__, 0, dphi, 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                   dbeta_phi);
      }

      // it is actually <psi | d(S|phi>)
      dPhi_S_Psi.zero();

      for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
        inner(ctx_.processing_unit(),
              ispn,
              kp.spinor_wave_functions(),
              0,
              kp.num_occupied_bands(ispn),
              dphi, //   S d |phi>
              0,
              this->number_of_hubbard_orbitals(),
              dPhi_S_Psi,
              0,
              ispn * this->number_of_hubbard_orbitals());
      }

      #pragma omp parallel for
      for (int ia1 = 0; ia1 < ctx_.unit_cell().num_atoms(); ++ia1) {
        const auto& atom = ctx_.unit_cell().atom(ia1);
        if (atom.type().hubbard_correction()) {
          const int lmax_at = 2 * atom.type().hubbard_l() + 1;
          for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            const size_t ispn_offset = ispn * this->number_of_hubbard_orbitals() + this->offset[ia1];
            // todo : use zgemm
            for (int m1 = 0; m1 < lmax_at; m1++) {
              for (int m2 = 0; m2 < lmax_at; m2++) {
                for (int nbnd = 0; nbnd < kp.num_occupied_bands(ispn); nbnd++) {
                  // d n_{m,m'}^I = \sum_{nk} <psi_{nk}|d phi_m><phi_m'|psi_nk> + <psi_{nk}|phi_m><d phi_m'|psi_nk>
                  dn_(m1, m2, ispn, ia1, 3 * mu + nu) +=  (
                                                           Phi_S_Psi(nbnd, ispn_offset + m1) * std::conj(dPhi_S_Psi(nbnd, ispn_offset + m2)) +
                                                           dPhi_S_Psi(nbnd, ispn_offset + m1) * std::conj(Phi_S_Psi(nbnd, ispn_offset + m2))
                                                           ) * kp.weight() * kp.band_occupancy(nbnd, ispn);
                }
              }
            }
          }
        }
      }
    } // nu
  } // mu
}

void Hubbard_potential::compute_gradient_strain_wavefunctions(K_point &kp__,
                                                              Wave_functions &dphi,
                                                              const mdarray<double, 2> &rlm_g,
                                                              const mdarray<double, 3> &rlm_dg,
                                                              const int mu,
                                                              const int nu)
{
  #pragma omp parallel for schedule(static)
  for (int igkloc = 0; igkloc < kp__.num_gkvec_loc(); igkloc++) {
    auto gvc = kp__.gkvec().gkvec_cart(kp__.idxgk(igkloc));
    /* vs = {r, theta, phi} */
    auto gvs = SHT::spherical_coordinates(gvc);
    std::vector<mdarray<double, 1>> ri_values(unit_cell_.num_atom_types());
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
      ri_values[iat] = ctx_.atomic_wf_ri().values(iat, gvs[0]);
    }

    std::vector<mdarray<double, 1>> ridjl_values(unit_cell_.num_atom_types());
    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
      ridjl_values[iat] = ctx_.atomic_wf_djl().values(iat, gvs[0]);
    }

    const auto p_g = (gvs[0] < 1e-10) ? 0.0 : 1.0 / gvs[0];
    double p = (mu == nu) ? 0.5 : 0.0;
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
      auto &atom_type = ctx_.unit_cell().atom(ia).type();
      if (atom_type.hubbard_correction()) {
        for (int i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
          auto l = std::abs(atom_type.ps_atomic_wf(i).first);
          if (l == atom_type.hubbard_l()) {
            for (int m = -l; m <= l; m++) {
              int lm = Utils::lm_by_l_m(l, m);
              auto z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
              auto d1 = ri_values[atom_type.id()][i] * (-gvc[mu] * rlm_dg(lm, nu, igkloc) - p * rlm_g(lm, igkloc));
              auto d2 = ridjl_values[atom_type.id()][i] * rlm_g(lm, igkloc) * -gvc[mu] * gvc[nu] * p_g;
              dphi.pw_coeffs(0).prime(igkloc, this->offset[ia] + l + m) =z * (d1 + d2);
            }
          }
        }
      }
    }
  }
}
