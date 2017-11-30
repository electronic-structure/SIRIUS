void generate_atomic_orbitals(K_point& kp, Q_operator<double>& q_op)
{
    TERMINATE("Not implemented for gamma point only");
}

void generate_atomic_orbitals(K_point& kp, Q_operator<double_complex>& q_op)
{
    int lmax{0};
    // return immediately if the wave functions are already allocated
    if (kp.hubbard_wave_functions_calculated())
        return;
    // printf("test\n");
    kp.allocate_hubbard_wave_functions(this->number_of_hubbard_orbitals());

    for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
        auto& atom_type = unit_cell_.atom_type(iat);
        for (auto& wf : atom_type.pp_desc().atomic_pseudo_wfs_) {
            lmax = std::max(lmax, wf.first);
        }
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

        int i = static_cast<int>((vs[0] / ctx_.gk_cutoff()) * (ctx_.centered_atm_wfc().orbital(0, 0).num_points() - 1));
        double dgk      = vs[0] - ctx_.centered_atm_wfc().orbital(0, 0).radial_grid()[i];
        idx_gk(igk_loc) = std::pair<int, double>(i, dgk);
    }

    // temporary wave functions
    Wave_functions sphi(ctx_.processing_unit(), kp.gkvec(), this->number_of_hubbard_orbitals(), ctx_.num_spins());

#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        auto& atom             = unit_cell_.atom(ia);
        double phase           = twopi * geometry3d::dot(kp.gkvec().vk(), unit_cell_.atom(ia).position());
        double_complex phase_k = double_complex(cos(phase), sin(phase));

        std::vector<double_complex> phase_gk(kp.num_gkvec_loc());
        for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
            int igk           = kp.idxgk(igk_loc);
            auto G            = kp.gkvec().gvec(igk);
            phase_gk[igk_loc] = std::conj(ctx_.gvec_phase_factor(G, ia) * phase_k);
        }
        auto& atom_type = atom.type();
        int n{0};
        if (atom_type.hubbard_correction()) {
            const int l      = atom_type.hubbard_l();
            double_complex z = std::pow(double_complex(0, -1), l) * fourpi / std::sqrt(unit_cell_.omega());
            if (atom_type.pp_desc().spin_orbit_coupling) {
                int orb[2];
                int s = 0;
                for (auto i = 0; i < static_cast<int>(atom_type.pp_desc().atomic_pseudo_wfs_.size()); i++) {
                    if (atom_type.pp_desc().atomic_pseudo_wfs_[i].first == atom_type.hubbard_l()) {
                        orb[s] = i;
                        s++;
                    }
                }
                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                        double_complex temp = (ctx_.centered_atm_wfc().orbital(orb[0], atom_type.id())(
                                                   idx_gk[igk_loc].first, idx_gk[igk_loc].second) +
                                               ctx_.centered_atm_wfc().orbital(orb[1], atom_type.id())(
                                                   idx_gk[igk_loc].first, idx_gk[igk_loc].second)) *
                                              0.5;
                        sphi.component(0).pw_coeffs().prime(igk_loc, this->offset[ia] + n) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;

                        sphi.component(1).pw_coeffs().prime(igk_loc, this->offset[ia] + n + 2 * l + 1) =
                            z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) * temp;
                    }
                    n++;
                }
            } else {
                // find the right hubbard orbital
                int orb = -1;
                for (auto i = 0; i < static_cast<int>(atom_type.pp_desc().atomic_pseudo_wfs_.size()); i++) {
                    if (atom_type.pp_desc().atomic_pseudo_wfs_[i].first == atom_type.hubbard_l()) {
                        orb = i;
                        break;
                    }
                }

                for (int m = -l; m <= l; m++) {
                    int lm = Utils::lm_by_l_m(l, m);
                    for (int igk_loc = 0; igk_loc < kp.num_gkvec_loc(); igk_loc++) {
                        for (int s = 0; s < ctx_.num_spins(); s++) {
                            sphi.component(s).pw_coeffs().prime(igk_loc, this->offset[ia] + n + s * (2 * l + 1)) =
                                z * phase_gk[igk_loc] * rlm_gk(igk_loc, lm) *
                                ctx_.centered_atm_wfc().orbital(orb, atom_type.id())(idx_gk[igk_loc].first,
                                                                                     idx_gk[igk_loc].second);
                        }
                    }
                }
            }
        }
    }

    // check if we have a norm conserving pseudo potential only
    bool augment_ = false;
    for (auto ia = 0; (ia < ctx_.unit_cell().num_atom_types()) && (!augment_); ia++) {
        augment_ = ctx_.unit_cell().atom_type(ia).pp_desc().augment;
    }

    for (int s = 0; s < ctx_.num_spins(); s++) {
        // I need to consider the case where all atoms are norm
        // conserving. In that case the S operator is diagonal in orbital space

        kp.hubbard_wave_functions(s).copy_from(sphi.component(s),
                                                    0,
                                                    this->number_of_hubbard_orbitals(),
                                                    ctx_.processing_unit());
    }

    if (!ctx_.full_potential() && augment_) {
        //     // need to apply the matrix here on the orbitals (ultra soft pseudo potential)
        //     Q_operator<double_complex> q_op(ctx_, kp.beta_projectors());
        for (int i = 0; i < ctx_.beta_projector_chunks().num_chunks(); i++) {
            /* generate beta-projectors for a block of atoms */
            kp.beta_projectors().generate(i);
            /* non-collinear case */
            if (ctx_.num_mag_dims() == 3) {
                for (int ispn = 0; ispn < 2; ispn++) {

                    auto beta_phi = kp.beta_projectors().inner<double_complex>(i, sphi.component(ispn), 0,
                                                                               this->number_of_hubbard_orbitals());

                    if (ctx_.so_correction()) {
                        q_op.apply(i, ispn, kp.hubbard_wave_functions(ispn), 0, this->number_of_hubbard_orbitals(),
                                   beta_phi);
                        /* apply non-diagonal spin blocks */
                        q_op.apply(i, (ispn == 0) ? 3 : 2, kp.hubbard_wave_functions((ispn == 0) ? 1 : 0), 0,
                                   this->number_of_hubbard_orbitals(), beta_phi);
                    } else {
                        /* apply Q operator (diagonal in spin) */
                        q_op.apply(i, 0, kp.hubbard_wave_functions(0), 0, this->number_of_hubbard_orbitals(),
                                   beta_phi);
                    }
                }
            } else { /* non-magnetic or collinear case */
                auto beta_phi = kp.beta_projectors().inner<double_complex>(i, kp.hubbard_wave_functions(0), 0,
                                                                           this->number_of_hubbard_orbitals());

                q_op.apply(i, 0, kp.hubbard_wave_functions(0), 0, this->number_of_hubbard_orbitals(), beta_phi);
            }
        }
        kp.beta_projectors().dismiss();
    }

    // do we orthogonalize the all thing

    if (this->orthogonalize_hubbard_orbitals_ || this->normalize_hubbard_orbitals_only()) {
        mdarray<double_complex, 2> S(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());
        S.zero();
        linalg<CPU>::gemm(2, 0, this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals(),
                          sphi.component(0).pw_coeffs().num_rows_loc(),
                          sphi.component(0).pw_coeffs().prime().at<CPU>(0, 0),
                          sphi.component(0).pw_coeffs().prime().ld(),
                          kp.hubbard_wave_functions(0).pw_coeffs().prime().at<CPU>(0, 0),
                          kp.hubbard_wave_functions(0).pw_coeffs().prime().ld(), S.at<CPU>(0, 0), S.ld());

        if (ctx_.num_spins() == 2)
            linalg<CPU>::gemm(2, 0, this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals(),
                              sphi.component(1).pw_coeffs().num_rows_loc(), linalg_const<double_complex>::one(),
                              sphi.component(1).pw_coeffs().prime().at<CPU>(0, 0),
                              sphi.component(1).pw_coeffs().prime().ld(),
                              kp.hubbard_wave_functions(1).pw_coeffs().prime().at<CPU>(0, 0),
                              kp.hubbard_wave_functions(1).pw_coeffs().prime().ld(),
                              linalg_const<double_complex>::one(), S.at<CPU>(0, 0), S.ld());

        kp.comm().allreduce<double_complex, mpi_op_t::sum>(S.at<CPU>(), static_cast<int>(S.size()));

        // diagonalize the all stuff

        if (this->orthogonalize_hubbard_orbitals_) {
            mdarray<double_complex, 2> Z(this->number_of_hubbard_orbitals(), this->number_of_hubbard_orbitals());

            Eigenproblem_lapack ei;
            std::vector<double> eigenvalues(this->number_of_hubbard_orbitals(), 0.0);
            ei.solve(this->number_of_hubbard_orbitals(),
                     S.at<CPU>(0, 0),
                     S.ld(),
                     &eigenvalues[0],
                     Z.at<CPU>(0, 0),
                     Z.ld());

            // build the O^{-1/2} operator
            for (int i = 0; i < static_cast<int>(eigenvalues.size()); i++) {
                eigenvalues[i] = 1.0 / sqrt(eigenvalues[i]);
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
                    if (l == m)
                        S(l, m) = 1.0 / sqrt(S(l, l));
                    else
                        S(l, m) = 0.0;
                }
            }
        }

        // now apply the overlap matrix
        for (int s = 0; s < ctx_.num_spins(); s++) {
            sphi.component(s).copy_from(kp.hubbard_wave_functions(s), 0, this->number_of_hubbard_orbitals(),
                                        ctx_.processing_unit());
            linalg<CPU>::gemm(0, 2, sphi.component(s).pw_coeffs().num_rows_loc(), this->number_of_hubbard_orbitals(),
                              this->number_of_hubbard_orbitals(), sphi.component(s).pw_coeffs().prime().at<CPU>(0, 0),
                              sphi.component(s).pw_coeffs().prime().ld(), S.at<CPU>(0, 0), S.ld(),
                              kp.hubbard_wave_functions(s).pw_coeffs().prime().at<CPU>(0, 0),
                              kp.hubbard_wave_functions(s).pw_coeffs().prime().ld());
        }
    }
}
