void generate_atomic_orbitals(K_point& kp, Q_operator<double>& q_op)
{
    TERMINATE("Not implemented for gamma point only");
}

void generate_atomic_orbitals(K_point& kp, Q_operator<double_complex>& q_op)
{
    int lmax{0};
    // return immediately if the wave functions are already allocated
    if (kp.hubbard_wave_functions_calculated()) {

       // the hubbard orbitals are already calculated but are stored on the CPU memory.
       // when the GPU is used, we need to do an explicit copy of them after allocation
       #ifdef __GPU
       if (ctx_.processing_unit() == GPU) {
          for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
              /* allocate GPU memory */
              kp.hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
	          kp.hubbard_wave_functions().pw_coeffs(ispn).copy_to_device(0, this->number_of_hubbard_orbitals());
          }
       }
       #endif
       return;
    }

    kp.allocate_hubbard_wave_functions(this->number_of_hubbard_orbitals());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        lmax = std::max(lmax, unit_cell_.atom_type(iat).lmax_ps_atomic_wf());
    }
    // we need the complex spherical harmonics for the spin orbit case
    // mdarray<double_complex, 2> ylm_gk;
    // if (ctx_.so_correction())
    //   ylm_gk = mdarray<double_complex, 2>(this->num_gkvec_loc(), Utils::lmmax(lmax));

    mdarray<double, 2> rlm_gk(kp.num_gkvec_loc(), Utils::lmmax(lmax));
    mdarray<std::pair<int, double>, 1> idx_gk(kp.num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
      int igk = kp.idxgk(igk_loc);
      /* vs = {r, theta, phi} */
      auto vs = SHT::spherical_coordinates(kp.gkvec().gkvec_cart(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(Utils::lmmax(lmax));
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        for (int lm = 0; lm < Utils::lmmax(lmax); lm++) {
            rlm_gk(igk_loc, lm) = rlm[lm];
        }

        idx_gk(igk_loc) = ctx_.atomic_wf_ri().iqdq(vs[0]);
    }

    // temporary wave functions
    Wave_functions sphi(kp.gkvec_partition(), this->number_of_hubbard_orbitals(), ctx_.num_spins());

    this->GenerateAtomicOrbitals(kp, sphi);

    // check if we have a norm conserving pseudo potential only
    bool augment = false;
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment); ia++) {
        augment = ctx_.unit_cell().atom_type(ia).augment();
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            /* allocate GPU memory */
            sphi.pw_coeffs(ispn).prime().allocate(memory_t::device);
            // can do async copy
            sphi.pw_coeffs(ispn).copy_to_device(0, this->number_of_hubbard_orbitals());
            kp.hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
        }
    }
#endif


    for (int s = 0; s < ctx_.num_spins(); s++) {
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
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {

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
       for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
           kp.hubbard_wave_functions().pw_coeffs(ispn).copy_to_host(0, this->number_of_hubbard_orbitals());
       }
    }
#endif
}

void orthogonalize_atomic_orbitals(K_point& kp, Wave_functions &sphi)
{
    // do we orthogonalize the all thing



    if (this->orthogonalize_hubbard_orbitals_ || this->normalize_hubbard_orbitals()) {

        // check if we have a norm conserving pseudo potential only
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

           // First compute S_{nm} = E_m Z_{nm}
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
                     S(l, m) = 1.0 / sqrt((S(l, l) * conj(S(l, l))).real());
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

        // now apply the overlap matrix
        for (int s = 0; (s < ctx_.num_spins()) && augment; s++) {
          sphi.copy_from(ctx_.processing_unit(), this->number_of_hubbard_orbitals(), kp.hubbard_wave_functions(),
                         s, 0, s, 0);
        }

        if(ctx_.num_mag_dims() == 3) {
          transform<double_complex>(ctx_.processing_unit(),
                                    2,
                                    sphi,
                                    0,
                                    this->number_of_hubbard_orbitals(),
                                    S,
                                    0,
                                    0,
                                    kp.hubbard_wave_functions(),
                                    0,
                                    this->number_of_hubbard_orbitals());
        } else {
          for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            transform<double_complex>(ctx_.processing_unit(),
                                      ispn,
                                      sphi,
                                      0,
                                      this->number_of_hubbard_orbitals(),
                                      S,
                                      0,
                                      0,
                                      kp.hubbard_wave_functions(),
                                      0,
                                      this->number_of_hubbard_orbitals());
          }
        }

#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            S.deallocate(memory_t::device);
        }
#endif
    }
}

void GenerateAtomicOrbitals(K_point& kp, Wave_functions &phi)
{
    int lmax{0};

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        lmax = std::max(lmax, unit_cell_.atom_type(iat).lmax_ps_atomic_wf());
    }
    // we need the complex spherical harmonics for the spin orbit case
    // mdarray<double_complex, 2> ylm_gk;
    // if (ctx_.so_correction())
    //   ylm_gk = mdarray<double_complex, 2>(this->num_gkvec_loc(), Utils::lmmax(lmax));

    mdarray<double, 2> rlm_gk(kp.num_gkvec_loc(), Utils::lmmax(lmax));
    mdarray<std::pair<int, double>, 1> idx_gk(kp.num_gkvec_loc());

    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
        int igk = kp.idxgk(igk_loc);
        /* vs = {r, theta, phi} */
        auto vs = SHT::spherical_coordinates(kp.gkvec().gkvec_cart(igk));
        /* compute real spherical harmonics for G+k vector */
        std::vector<double> rlm(Utils::lmmax(lmax));
        SHT::spherical_harmonics(lmax, vs[1], vs[2], &rlm[0]);

        for (int lm = 0; lm < Utils::lmmax(lmax); lm++) {
            rlm_gk(igk_loc, lm) = rlm[lm];
        }

        idx_gk(igk_loc) = ctx_.atomic_wf_ri().iqdq(vs[0]);
    }

    #pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom             = unit_cell_.atom(ia);
        const double phase           = twopi * geometry3d::dot(kp.gkvec().vk(), unit_cell_.atom(ia).position());
        const double_complex phase_k = double_complex(cos(phase), sin(phase));

        std::vector<double_complex> phase_gk(kp.num_gkvec_loc());
        for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            int igk           = kp.idxgk(igk_loc);
            auto G            = kp.gkvec().gvec(igk);
            phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
        }
        const auto& atom_type = atom.type();
        if (atom_type.hubbard_correction()) {
            const int l      = atom_type.hubbard_l();
            const double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
            if (atom_type.spin_orbit_coupling()) {
                int orb[2];
                int s = 0;
                for (auto i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                  // this is very ugly but we encode the j = l +- 1/2
                  // directly in the sign of l.

                  // Ideally we should also check the orbital level
                  if (std::abs(atom_type.ps_atomic_wf(i).first) == atom_type.hubbard_l()) {
                        orb[s] = i;
                        s++;
                    }
                }

                for (int m = -l; m <= l; m++) {
                  int lm = Utils::lm_by_l_m(l, m);
                  for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                    const double_complex temp = (ctx_.atomic_wf_ri().values(orb[0], atom_type.id())(
                                                     idx_gk[igk_loc].first, idx_gk[igk_loc].second) +
                                                     ctx_.atomic_wf_ri().values(orb[1], atom_type.id())(
                                                     idx_gk[igk_loc].first, idx_gk[igk_loc].second)) * 0.5;
                        phi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;

                        phi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + l + m + 2 * l + 1) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;
                    }
                }
            } else {
                // find the right hubbard orbital
                int orb = -1;
                for (auto i = 0; i < atom_type.num_ps_atomic_wf(); i++) {
                    if (atom_type.ps_atomic_wf(i).first == atom_type.hubbard_l()) {
                        orb = i;
                        break;
                    }
                }

                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    if (ctx_.num_mag_dims() == 3) {
                      for (int s = 0; s < ctx_.num_spins(); s++) {
                        for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                          phi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m + s * (2 * l + 1)) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) *
                            ctx_.atomic_wf_ri().values(orb, atom_type.id())(idx_gk[igk_loc].first, idx_gk[igk_loc].second);
                        }
                      }
                    } else {
                      for (int s = 0; s < ctx_.num_spins(); s++) {
                        for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                          phi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) *
                            ctx_.atomic_wf_ri().values(orb, atom_type.id())(idx_gk[igk_loc].first, idx_gk[igk_loc].second);
                        }
                      }
                    }
                }
            }
        }
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
      if (atom_type.spin_orbit_coupling()) {

        for (int m = -l; m <= l; m++) {
          for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            dphi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m) = qalpha[igk_loc] * phi.pw_coeffs(0).prime(igk_loc, this->offset[ia] + l + m);
            dphi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + l + m + 2 * l + 1) = qalpha[igk_loc] * phi.pw_coeffs(1).prime(igk_loc, this->offset[ia] + l + m + 2 * l + 1);
          }
        }
      } else {
        if (ctx_.num_mag_dims() == 3) {
          for (int s = 0; s < ctx_.num_spins(); s++) {
              for (int m = -l; m <= l; m++) {
                  for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                      dphi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m + s * (2 * l + 1)) = qalpha[igk_loc] * phi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m + s * (2 * l + 1));
                  }
              }
          }
        } else {
          for (int s = 0; s < ctx_.num_spins(); s++) {
              for (int m = -l; m <= l; m++) {
                  for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                      dphi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m) = qalpha[igk_loc] * phi.pw_coeffs(s).prime(igk_loc, this->offset[ia] + l + m);
                  }
              }
          }
        }
      }
    }
  }
}
