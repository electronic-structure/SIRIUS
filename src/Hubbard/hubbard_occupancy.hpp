/// as the name says. We compute the occupation numbers associated to
/// the hubbard wavefunctions (locally centered orbitals, wannier
/// functions, etc) that are relevant for the hubbard correction.

/// These quantities are defined by
///    \f[ n_{m,m'}^I \sigma = \sum_{kv} f(\varepsilon_{kv}) |<\psi_{kv}| phi_I_m>|^2 \f]
/// where \f[m=-l\cdot l$ (same for m')\f], I is the a.

/// We need to symmetrize them
void hubbard_compute_occupation_numbers(K_point_set& kset_)
{
    if (!ctx_.hubbard_correction())
        return;

    this->occupancy_number_.zero();

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {

        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        // now for each spin components and each atom we need to calculate
        // <psi_{nk}|phi^I_m'><phi^I_m|psi_{nk}>

        // split the bands over the different procs of a given kp communicator
        // splindex<block> spl_nbnd(kp->num_occupied_bands(), kp->comm().size(), kp->comm().rank());

        // How many bands do I have locally
        // int nbnd_loc = spl_nbnd.local_size();

        // if (nbnd_loc) {
        mdarray<double_complex, 2> dm(this->number_of_hubbard_orbitals(), kp->num_occupied_bands());
        dm.zero();

        linalg<CPU>::gemm(2, 0, this->number_of_hubbard_orbitals(), kp->num_occupied_bands(),
                          kp->hubbard_wave_functions_ppus(0).pw_coeffs().num_rows_loc(),
                          kp->hubbard_wave_functions_ppus(0).pw_coeffs().prime().at<CPU>(0, 0),
                          kp->hubbard_wave_functions_ppus(0).pw_coeffs().prime().ld(),
                          kp->spinor_wave_functions(0).pw_coeffs().prime().at<CPU>(0, 0),
                          kp->spinor_wave_functions(0).pw_coeffs().prime().ld(), dm.at<CPU>(0, 0), dm.ld());

        for (int s1 = 1; s1 < ctx_.num_spins(); s1++) {
            // compute <phi_{i,\sigma}|psi_nk> the bands been distributed over the pool

            linalg<CPU>::gemm(2, 0, this->number_of_hubbard_orbitals(), kp->num_occupied_bands(),
                              kp->hubbard_wave_functions_ppus(s1).pw_coeffs().num_rows_loc(),
                              linalg_const<double_complex>::one(),
                              kp->hubbard_wave_functions_ppus(s1).pw_coeffs().prime().at<CPU>(0, 0),
                              kp->hubbard_wave_functions_ppus(s1).pw_coeffs().prime().ld(),
                              kp->spinor_wave_functions(s1).pw_coeffs().prime().at<CPU>(0, 0),
                              kp->spinor_wave_functions(s1).pw_coeffs().prime().ld(),
                              linalg_const<double_complex>::one(), dm.at<CPU>(0, 0), dm.ld());
        }

        // now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk}
        
        // there must be a way to do that with matrix multiplication
        #pragma omp parallel for schedule(static)
        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            const auto& atom = unit_cell_.atom(ia);
            if (atom.type().hubbard_correction()) {
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        int s = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                        for (int nband = 0; nband < kp->num_occupied_bands(); nband++) {
                            for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                                for (int mp = 0; mp < 2 * atom.type().hubbard_l() + 1; mp++) {
                                    this->occupancy_number_(m, mp, s, ia, 0) +=
                                        std::conj(
                                            dm(this->offset[ia] + m + s1 * (2 * atom.type().hubbard_l() + 1), nband)) *
                                        dm(this->offset[ia] + mp + s2 * (2 * atom.type().hubbard_l() + 1), nband) *
                                        kp->band_occupancy(nband) * kp->weight();
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // global reduction
    ctx_.comm().allreduce<double_complex, mpi_op_t::sum>(this->occupancy_number_.at<CPU>(),
                                                         static_cast<int>(this->occupancy_number_.size()));

    // Now symmetrization procedure
    if (ctx_.use_symmetry()) {
        auto& sym = unit_cell_.symmetry();

        // check if we have some symmetries
        if (sym.num_mag_sym()) {
            int lmax  = unit_cell_.lmax();
            int lmmax = Utils::lmmax(lmax);

            mdarray<double_complex, 2> rotm(lmmax, lmmax);
            mdarray<double_complex, 4> rotated_oc(lmmax, lmmax, ctx_.num_spins() * ctx_.num_spins(),
                                                  unit_cell_.num_atoms());

            double alpha = 1.0 / static_cast<double>(sym.num_mag_sym());
            rotated_oc.zero();

            for (int i = 0; i < sym.num_mag_sym(); i++) {
                int pr    = sym.magnetic_group_symmetry(i).spg_op.proper;
                auto eang = sym.magnetic_group_symmetry(i).spg_op.euler_angles;
                //int isym  = sym.magnetic_group_symmetry(i).isym;
                SHT::rotation_matrix(lmax, eang, pr, rotm);
                auto spin_rot_su2 = SHT::rotation_matrix_su2(sym.magnetic_group_symmetry(i).spin_rotation);

                #pragma omp parallel for schedule(static)
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    const auto& atom = unit_cell_.atom(ia);
                    if (atom.type().hubbard_correction()) {
                        for (int ii = 0; ii < 2 * atom.type().hubbard_l() + 1; ii++) {
                            int l1 = Utils::lm_by_l_m(atom.type().hubbard_l(), ii - atom.type().hubbard_l());
                            for (int ll = 0; ll < 2 * atom.type().hubbard_l() + 1; ll++) {
                                int l2 = Utils::lm_by_l_m(atom.type().hubbard_l(), ll - atom.type().hubbard_l());
                                mdarray<double_complex, 1> rot_spa(ctx_.num_spins() * ctx_.num_spins());
                                rot_spa.zero();
                                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                        // symmetrization procedure
                                        // A_ij B_jk C_kl

                                        for (int jj = 0; jj < 2 * atom.type().hubbard_l() + 1; jj++) {
                                            int l3 =
                                                Utils::lm_by_l_m(atom.type().hubbard_l(), jj - atom.type().hubbard_l());
                                            for (int kk = 0; kk < 2 * atom.type().hubbard_l() + 1; kk++) {
                                                int l4 = Utils::lm_by_l_m(atom.type().hubbard_l(),
                                                                          kk - atom.type().hubbard_l());
                                                rot_spa(2 * s1 + s2) +=
                                                    std::conj(rotm(l1, l3)) *
                                                    occupancy_number_(
                                                                      jj, kk, (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s1 + s2), ia, 0) *
                                                    rotm(l2, l4) * alpha;
                                            }
                                        }
                                    }
                                }

                                if (ctx_.num_mag_dims() == 3) {
                                    double_complex spin_dm[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
                                    for (int iii = 0; iii < ctx_.num_spins(); iii++) {
                                        for (int lll = 0; lll < ctx_.num_spins(); lll++) {
                                            // A_ij B_jk C_kl
                                            for (int jj = 0; jj < ctx_.num_spins(); jj++) {
                                                for (int kk = 0; kk < ctx_.num_spins(); kk++) {
                                                    spin_dm[iii][lll] += spin_rot_su2(iii, jj) * rot_spa(jj + 2 * kk) *
                                                                         std::conj(spin_rot_su2(kk, lll));
                                                }
                                            }
                                        }
                                    }

                                    rotated_oc(ii, ll, 0, ia) += spin_dm[0][0];
                                    rotated_oc(ii, ll, 1, ia) += spin_dm[1][1];
                                    rotated_oc(ii, ll, 2, ia) += spin_dm[0][1];
                                    rotated_oc(ii, ll, 3, ia) += spin_dm[1][0];
                                } else {
                                    for (int s = 0; s < ctx_.num_spins(); s++) {
                                        rotated_oc(ii, ll, s, ia) += rot_spa(s);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                auto& atom = unit_cell_.atom(ia);
                if (atom.type().hubbard_correction()) {
                    for (int ii = 0; ii < 2 * atom.type().hubbard_l() + 1; ii++) {
                        for (int ll = 0; ll < 2 * atom.type().hubbard_l() + 1; ll++) {
                            for (int s = 0; s < ctx_.num_spins() * ctx_.num_spins(); s++) {
                                this->occupancy_number_(ii, ll, s, ia, 0) = rotated_oc(ii, ll, s, ia);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom = unit_cell_.atom(ia);
        if (atom.type().hubbard_correction()) {
            for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                // diagonal blocks
                for (int m = 0; m < (2 * atom.type().hubbard_l() + 1); m++) {
                    for (int mp = m + 1; mp < (2 * atom.type().hubbard_l() + 1); mp++) {
                        this->occupancy_number_(mp, m, s1, ia, 0) = std::conj(this->occupancy_number_(m, mp, s1, ia, 0));
                    }
                }
            }

            if (ctx_.num_mag_dims() == 3) {
                // off diagonal blocks
                for (int m = 0; m < (2 * atom.type().hubbard_l() + 1); m++) {
                    for (int mp = m + 1; mp < (2 * atom.type().hubbard_l() + 1); mp++) {
                        this->occupancy_number_(mp, m, 2, ia, 0) = std::conj(this->occupancy_number_(m, mp, 3, ia, 0));
                    }
                }
            }
        }
    }

    if (ctx_.control().verbosity_ > 1) {
        if (ctx_.comm().rank() == 0) {
            printf("\n");
            printf("hubbard occupancies\n");
            printf("----------------------------------------------------------------------\n");
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                printf("Atom : %d\n", ia);
                const auto& atom = unit_cell_.atom(ia);
                if (atom.type().hubbard_correction()) {
                    for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                        for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                            printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 0, ia, 0).real(),
                                   this->occupancy_number_(m1, m2, 0, ia, 0).imag());
                        }

                        if (ctx_.num_mag_dims() == 3) {
                            printf(" ");
                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 2, ia, 0).real(),
                                       this->occupancy_number_(m1, m2, 2, ia, 0).imag());
                            }
                        }
                        printf("\n");
                    }
                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 3, ia, 0).real(),
                                       this->occupancy_number_(m1, m2, 3, ia, 0).imag());
                            }
                            if (ctx_.num_mag_dims() == 3) {
                                printf(" ");
                                for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                    printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 1, ia, 0).real(),
                                           this->occupancy_number_(m1, m2, 1, ia, 0).imag());
                                }
                            }
                            printf("\n");
                        }
                    }

                    double n_up, n_down, n_total;
                    n_up   = 0.0;
                    n_down = 0.0;
                    for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                        n_up += this->occupancy_number_(m1, m1, 0, ia, 0).real();
                    }

                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                            n_down += this->occupancy_number_(m1, m1, 1, ia, 0).real();
                        }
                    }
                    printf("\n");
                    n_total = n_up + n_down;
                    if (ctx_.num_spins() == 2) {
                        printf("Atom charge (total) %.5lf (n_up) %.5lf (n_down) %.5lf (mz) %.5lf\n", n_total, n_up,
                               n_down, n_up - n_down);
                    } else {
                        printf("Atom charge (total) %.5lf\n", n_total);
                    }
                }
            }
            printf("-------------------------------------------------------------\n");
        }
    }
}

// The initial occupancy is calculated following Hund rules. We first
// fill the d (f) states according to the hund's rules and with majority
// spin first and the remaining electrons distributed among the minority
// states.
void calculate_initial_occupation_numbers()
{
    this->occupancy_number_.zero();

#pragma omp parallel for schedule(static)
    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom = unit_cell_.atom(ia);
        if (atom.type().hubbard_correction()) {

            // compute the total charge for the hubbard orbitals
            double charge = atom.type().get_occupancy_hubbard_orbital();
            bool nm       = true; // true if the atom is non magnetic
            int majs, mins;

            if(ctx_.num_spins() != 1) {
                if (atom.type().starting_magnetization() > 0.0) {
                    nm   = false;
                    majs = 0;
                    mins = 1;
                } else if (atom.type().starting_magnetization() < 0.0) {
                    nm   = false;
                    majs = 1;
                    mins = 0;
                }
            }

            if (!nm) {
                if (ctx_.num_mag_dims() != 3) {
                    // colinear case

                    if (charge > (2 * atom.type().hubbard_l() + 1)) {
                        for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                            this->occupancy_number_(m, m, majs, ia, 0) = 1.0;
                            this->occupancy_number_(m, m, mins, ia, 0) =
                                (charge - static_cast<double>(2 * atom.type().hubbard_l() + 1)) /
                                static_cast<double>(2 * atom.type().hubbard_l() + 1);
                        }
                    } else {
                        for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                            this->occupancy_number_(m, m, majs, ia, 0) =
                                charge / static_cast<double>(2 * atom.type().hubbard_l() + 1);
                        }
                    }
                } else {
                    //double c1, s1;
                    //sincos(atom.type().starting_magnetization_theta(), &s1, &c1);
                    double c1 = std::cos(atom.type().starting_magnetization_theta());
                    double_complex cs =
                        double_complex(cos(atom.type().starting_magnetization_phi()), sin(atom.type().starting_magnetization_phi()));
                    double_complex ns[4];

                    if (charge > (2 * atom.type().hubbard_l() + 1)) {
                        ns[majs] = 1.0;
                        ns[mins] = (charge - static_cast<double>(2 * atom.type().hubbard_l() + 1)) /
                                   static_cast<double>(2 * atom.type().hubbard_l() + 1);
                    } else {
                        ns[majs] = charge / static_cast<double>(2 * atom.type().hubbard_l() + 1);
                        ns[mins] = 0.0;
                    }

                    // charge and moment
                    double nc  = ns[majs].real() + ns[mins].real();
                    double mag = ns[majs].real() - ns[mins].real();

                    // rotate the occ matrix
                    ns[0] = (nc + mag * c1) * 0.5;
                    ns[1] = (nc - mag * c1) * 0.5;
                    ns[2] = mag * std::conj(cs) * 0.5;
                    ns[3] = mag * cs * 0.5;

                    for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                        this->occupancy_number_(m, m, 0, ia, 0) = ns[0];
                        this->occupancy_number_(m, m, 1, ia, 0) = ns[1];
                        this->occupancy_number_(m, m, 2, ia, 0) = ns[2];
                        this->occupancy_number_(m, m, 3, ia, 0) = ns[3];
                    }
                }
            } else {
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    for (int m = 0; m < 2 * atom.type().hubbard_l() + 1; m++) {
                        this->occupancy_number_(m, m, s, ia, 0) =
                            charge * 0.5 / static_cast<double>(2 * atom.type().hubbard_l() + 1);
                    }
                }
            }
        }
    }

    if (ctx_.control().verbosity_ > 1) {
        if (ctx_.comm().rank() == 0) {
            printf("Initial occupancy\n");
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                printf("Atom : %d\n", ia);
                const auto& atom = unit_cell_.atom(ia);
                if (atom.type().hubbard_correction()) {
                    for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                        for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                            printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 0, ia, 0).real(),
                                   this->occupancy_number_(m1, m2, 0, ia, 0).imag());
                        }

                        if (ctx_.num_mag_dims() == 3) {
                            printf(" ");
                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 2, ia, 0).real(),
                                       this->occupancy_number_(m1, m2, 2, ia, 0).imag());
                            }
                        }
                        printf("\n");
                    }
                    printf("\n");
                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                            for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 3, ia, 0).real(),
                                       this->occupancy_number_(m1, m2, 3, ia, 0).imag());
                            }
                            if (ctx_.num_mag_dims() == 3) {
                                printf(" ");
                                for (int m2 = 0; m2 < 2 * atom.type().hubbard_l() + 1; m2++) {
                                    printf("%.3lf %.3lf ", this->occupancy_number_(m1, m2, 1, ia, 0).real(),
                                           this->occupancy_number_(m1, m2, 1, ia, 0).imag());
                                }
                            }
                            printf("\n");
                        }
                    }

                    printf("\n");
                    double n_up, n_down, n_total;
                    n_up   = 0.0;
                    n_down = 0.0;
                    for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                        n_up += this->occupancy_number_(m1, m1, 0, ia, 0).real();
                    }

                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < 2 * atom.type().hubbard_l() + 1; m1++) {
                            n_down += this->occupancy_number_(m1, m1, 1, ia, 0).real();
                        }
                    }

                    n_total = n_up + n_down;
                    if (ctx_.num_spins() == 2) {
                        printf("Atom charge (total) %.5lf (n_up) %.5lf (n_down) %.5lf (mz) %.5lf\n", n_total, n_up,
                               n_down, n_up - n_down);
                    } else {
                        printf("Atom charge (total) %.5lf\n", n_total);
                    }
                }
            }
            printf("\n");
        }
    }
}
