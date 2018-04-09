void generate_atomic_orbitals(K_point& kp, Q_operator<double>& q_op)
{
    TERMINATE("Not implemented for gamma point only");
}

void generate_atomic_orbitals(K_point& kp, Q_operator<double_complex>& q_op)
{

  const int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

  // return immediately if the wave functions are already allocated
    if (kp.hubbard_wave_functions_calculated()) {

       // the hubbard orbitals are already calculated but are stored on the CPU memory.
       // when the GPU is used, we need to do an explicit copy of them after allocation
       #ifdef __GPU
       if (ctx_.processing_unit() == GPU) {
          for (int ispn = 0; ispn < num_sc; ispn++) {
              /* allocate GPU memory */
              kp.hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
	          kp.hubbard_wave_functions().pw_coeffs(ispn).copy_to_device(0, this->number_of_hubbard_orbitals());
          }
       }
       #endif
       return;
    }

    kp.allocate_hubbard_wave_functions(this->number_of_hubbard_orbitals());

    // temporary wave functions
    Wave_functions sphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), num_sc);

    kp.generate_atomic_centered_wavefunctions_(this->number_of_hubbard_orbitals(), sphi, this->offset, true);

    // check if we have a norm conserving pseudo potential only
    bool augment = false;
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        for (int ispn = 0; ispn < num_sc; ispn++) {
            /* allocate GPU memory */
            sphi.pw_coeffs(ispn).prime().allocate(memory_t::device);
            // can do async copy
            sphi.pw_coeffs(ispn).copy_to_device(0, this->number_of_hubbard_orbitals());
            kp.hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
        }
    }
#endif


    for (int s = 0; s < num_sc; s++) {
        // I need to consider the case where all atoms are norm
        // conserving. In that case the S operator is diagonal in orbital space
        kp.hubbard_wave_functions().copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), sphi, s, 0, s, 0);
    }

    if (!ctx_.full_potential() && augment) {
        kp.beta_projectors().prepare();
        /* need to apply the matrix here on the orbitals (ultra soft pseudo potential) */
        for (int i = 0; i < kp.beta_projectors().num_chunks(); i++) {
            /* generate beta-projectors for a block of atoms */
            kp.beta_projectors().generate(i);
            /* non-collinear case */
            for (int ispn = 0; ispn < num_sc; ispn++) {

              auto beta_phi = kp.beta_projectors().inner<double_complex>(i, sphi, ispn, 0,
                                                                         this->number_of_hubbard_orbitals());
              /* apply Q operator (diagonal in spin) */
              q_op.apply(i, ispn, kp.hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                         beta_phi);
              /* apply non-diagonal spin blocks */
              if (ctx_.so_correction()) {
                q_op.apply(i, ispn ^ 3, kp.hubbard_wave_functions(), 0, this->number_of_hubbard_orbitals(), kp.beta_projectors(),
                           beta_phi);
              }
            }
        }
        kp.beta_projectors().dismiss();
    }

    orthogonalize_atomic_orbitals(kp, sphi);

#ifdef __GPU
    // All calculations on GPU then we need to copy the final result back to the cpus
    if (ctx_.processing_unit() == GPU) {
       for (int ispn = 0; ispn < num_sc; ispn++) {
           kp.hubbard_wave_functions().pw_coeffs(ispn).copy_to_host(0, this->number_of_hubbard_orbitals());
       }
    }
#endif
}

void orthogonalize_atomic_orbitals(K_point& kp, Wave_functions &sphi)
{
    // do we orthogonalize the all thing

  const int num_sc = (ctx_.num_mag_dims() == 3) ? 2 : 1;

  if (this->orthogonalize_hubbard_orbitals_ || this->normalize_hubbard_orbitals()) {
    bool augment = false;
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
      augment = ctx_.unit_cell().atom_type(ia).augment();
    }


    dmatrix<double_complex> S(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());
    S.zero();

    #ifdef __GPU
    if (ctx_.processing_unit()) {
      S.allocate(memory_t::device);
    }
    #endif

    if(ctx_.num_mag_dims() == 3) {
      inner<double_complex>(ctx_.processing_unit(),
                            2,
                            sphi,
                            0,
                            this->number_of_hubbard_orbitals(),
                            kp.hubbard_wave_functions(),
                            0,
                            this->number_of_hubbard_orbitals(),
                            S, 0, 0);
    } else {
      // we do not need to treat both up and down spins for the
      // colinear case because the up and down components are
      // identical
      inner<double_complex>(ctx_.processing_unit(),
                            0,
                            sphi,
                            0,
                            this->number_of_hubbard_orbitals(),
                            kp.hubbard_wave_functions(),
                            0,
                            this->number_of_hubbard_orbitals(),
                            S, 0, 0);
    }
    #ifdef __GPU
    if (ctx_.processing_unit()) {
      S.copy<memory_t::device, memory_t::host>();
    }
    #endif

    // diagonalize the all stuff

    if (this->orthogonalize_hubbard_orbitals_) {
      dmatrix<double_complex> Z(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());

      auto ev_solver = Eigensolver_factory<double_complex>(ev_solver_t::lapack);

      std::vector<double> eigenvalues(this->number_of_hubbard_orbitals(), 0.0);

      ev_solver->solve(number_of_hubbard_orbitals(), S, &eigenvalues[0], Z);

      // build the O^{-1/2} operator
      for (int i = 0; i < static_cast<int>(eigenvalues.size()); i++) {
        eigenvalues[i] = 1.0 / std::sqrt(eigenvalues[i]);
      }

      // // First compute S_{nm} = E_m Z_{nm}
      S.zero();
      for (int l = 0; l < this->number_of_hubbard_orbitals(); l++) {
        for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
          for (int n = 0; n < this->number_of_hubbard_orbitals(); n++) {
            S(n, m) += eigenvalues[l] * Z(n, l) * std::conj(Z(m, l));
          }
        }
      }
    } else {
      for (int l = 0; l < this->number_of_hubbard_orbitals(); l++) {
        for (int m = 0; m < this->number_of_hubbard_orbitals(); m++) {
          if (l == m) {
            S(l, m) = 1.0 / sqrt(S(l, l).real());
          } else {
            S(l, m) = 0.0;
          }
        }
      }
    }

    #ifdef __GPU
    if (ctx_.processing_unit()) {
      S.copy<memory_t::host, memory_t::device>();
    }
    #endif

    // only need to do that when in the ultra soft case
    if (augment) {
      // now apply the overlap matrix
      for (int s = 0; (s < num_sc) && augment; s++) {
        sphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), kp.hubbard_wave_functions(),
                       s, 0, s, 0);
      }
    }

    // Apply the transform on the wave functions
    transform<double_complex>(ctx_.processing_unit(),
                              (ctx_.num_mag_dims() == 3) ? 2 : 0,
                              sphi,
                              0,
                              this->number_of_hubbard_orbitals(),
                              S,
                              0,
                              0,
                              kp.hubbard_wave_functions(),
                              0,
                              this->number_of_hubbard_orbitals());

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
      S.deallocate(memory_t::device);
    }
    #endif
  }
}

void ComputeDerivatives(K_point& kp, Wave_functions &phi, Wave_functions &dphi, const int direction)
{
#pragma omp parallel for schedule(static)
  for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
    const auto& atom             = unit_cell_.atom(ia);
    std::vector<double_complex> qalpha(kp.num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
      int igk           = kp.idxgk(igk_loc);
      auto G            = kp.gkvec().gvec(igk);
      qalpha[igk_loc] = G[direction];
    }

    const auto& atom_type = atom.type();
    if (atom_type.hubbard_correction()) {
      const int l      = atom_type.hubbard_l();
      if (ctx_.num_mag_dims() == 3) {
        for (int m = -l; m <= l; m++) {
          for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            dphi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m) = qalpha[igk_loc] * phi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m);
            dphi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + l + m + 2 * l + 1) = qalpha[igk_loc] * phi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + l + m + 2 * l + 1);
          }
        }
      } else {
        for (int m = -l; m <= l; m++) {
          for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            dphi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m) = qalpha[igk_loc] * phi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m);
          }
        }
      }
    }
  }
}
