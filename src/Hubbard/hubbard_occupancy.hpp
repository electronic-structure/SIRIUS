/// as the name says. We compute the occupation numbers associated to
/// the hubbard wavefunctions (locally centered orbitals, wannier
/// functions, etc) that are relevant for the hubbard correction.

/// These quantities are defined by
///    \f[ n_{m,m'}^I \sigma = \sum_{kv} f(\varepsilon_{kv}) |<\psi_{kv}| phi_I_m>|^2 \f]
/// where \f[m=-l\cdot l$ (same for m')\f], I is the a.

/// We need to symmetrize them
void hubbard_compute_occupation_numbers(K_point_set& kset_)
{
    if (!ctx_.hubbard_correction()) {
        return;
    }

    this->occupancy_number_.zero();

    int HowManyBands = -1;
    // if we are doing calculations for non colinear magnetism or
    // simple LDA then do not change the number of bands. the factor
    // two is important for colinear magnetism since the up-up and
    // down-down blocks are decoupled but the wave-functions are up
    // and down are still stored as a spinor to conserve space.

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {
        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        int tmp = kp->num_occupied_bands(0);
        if (ctx_.num_mag_dims() == 1) {
            tmp += kp->num_occupied_bands(1);
        }

        HowManyBands = std::max(HowManyBands, tmp);
    }

    // now for each spin components and each atom we need to calculate
    // <psi_{nk}|phi^I_m'><phi^I_m|psi_{nk}>

    int Ncf = 1;
    if (ctx_.num_mag_dims() == 1) {
        // !!! colinear magnetism. We need to have 2 times more space.
        Ncf = 2;
    }

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals() * Ncf, HowManyBands);

    dm.zero();

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        // the communicator is always of size 1.  I need to allocate memory
        // on the device manually
        dm.allocate(memory_t::device);
    }
#endif

    for (int ikloc = 0; ikloc < kset_.spl_num_kpoints().local_size(); ikloc++) {

        int ik  = kset_.spl_num_kpoints(ikloc);
        auto kp = kset_[ik];

        #ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* allocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                kp->spinor_wave_functions().pw_coeffs(ispn).copy_to_device(0, kp->num_occupied_bands(ispn));

                kp->hubbard_wave_functions().pw_coeffs(ispn).prime().allocate(memory_t::device);
                kp->hubbard_wave_functions().pw_coeffs(ispn).copy_to_device(0, this->number_of_hubbard_orbitals());
            }
        }
        #endif

        if (ctx_.num_mag_dims() == 3) {
            inner(ctx_.processing_unit(),
                  2,
                  kp->hubbard_wave_functions(),
                  0,
                  this->number_of_hubbard_orbitals(),
                  kp->spinor_wave_functions(),
                  0,
                  kp->num_occupied_bands(),
                  dm,
                  0,
                  0);
        } else {
            // This is annoying to have to keep track of this factor 2

            for (int ispn_ = 0; ispn_ < ctx_.num_spins(); ispn_++) {
                inner(ctx_.processing_unit(),
                      ispn_,
                      kp->hubbard_wave_functions(),
                      0,
                      this->number_of_hubbard_orbitals(),
                      kp->spinor_wave_functions(),
                      0,
                      kp->num_occupied_bands(ispn_),
                      dm,
                      ispn_ * this->number_of_hubbard_orbitals(),
                      0);
            }
        }

#ifdef __GPU
        if (ctx_.processing_unit() == GPU) {
            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                /* deallocate GPU memory */
                kp->spinor_wave_functions().pw_coeffs(ispn).deallocate_on_device();
                kp->hubbard_wave_functions().pw_coeffs(ispn).deallocate_on_device();
            }
            dm.copy<memory_t::device, memory_t::host>();
        }
#endif
        // now compute O_{ij}^{sigma,sigma'} = \sum_{nk} <psi_nk|phi_{i,sigma}><phi_{j,sigma^'}|psi_nk> f_{nk}

        if (ctx_.num_mag_dims() == 3) {
            // there must be a way to do that with matrix multiplication
            #pragma omp parallel for schedule(static)
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                const auto& atom = unit_cell_.atom(ia);
                const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                if (atom.type().hubbard_correction()) {
                    for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                        for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                            int s = (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s2 + s1);
                            for (int nband = 0; nband < kp->num_occupied_bands(0); nband++) {
                                for (int m = 0; m < lmax_at; m++) {
                                    for (int mp = 0; mp < lmax_at; mp++) {
                                        this->occupancy_number_(m, mp, s, ia, 0) +=
                                            std::conj(dm(this->offset[ia] + m + s1 * lmax_at, nband)) *
                                            dm(this->offset[ia] + mp + s2 * lmax_at, nband) *
                                            kp->band_occupancy(nband, 0) * kp->weight();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {

            // Well we need to apply a factor 1/2 when we compute the
            // occupancies for the boring LDA + U. It is because the
            // calculations of E and U consider occupancies <= 1.
            // Sirius for the boring lda+U has a factor 2 in the kp band
            // occupancies. We need to compenssate for it.

            const double scal = 1.0 - 0.5 * (ctx_.num_mag_dims() != 1);


            for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                #pragma omp parallel for schedule(static)
                for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                    const auto& atom = unit_cell_.atom(ia);
                    const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                    if (atom.type().hubbard_correction()) {
                        for (int nband = 0; nband < kp->num_occupied_bands(ispn); nband++) {
                            for (int m = 0; m < lmax_at; m++) {
                                for (int mp = 0; mp < lmax_at; mp++) {
                                    this->occupancy_number_(m, mp, ispn, ia, 0) += scal *
                                        std::conj(dm(this->offset[ia] + m + ispn * this->number_of_hubbard_orbitals(), nband)) *
                                        dm(this->offset[ia] + mp + ispn * this->number_of_hubbard_orbitals(), nband) *
                                        kp->band_occupancy(nband, ispn) * kp->weight();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

#ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        dm.deallocate(memory_t::device);
    }
#endif

    // global reduction
    ctx_.comm().allreduce<double_complex, mpi_op_t::sum>(this->occupancy_number_.at<CPU>(),
                                                         static_cast<int>(this->occupancy_number_.size()));

    // Now symmetrization procedure. We need to review that

    if (0) {
        if (ctx_.num_mag_dims() == 3) {
            symmetrize_occupancy_matrix_noncolinear_case();
        } else {
            symmetrize_occupancy_matrix();
        }
    }

    for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
        const auto& atom = unit_cell_.atom(ia);
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
        if (atom.type().hubbard_correction()) {
            for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                // diagonal blocks
                for (int m = 0; m < lmax_at; m++) {
                    for (int mp = 0; mp < lmax_at; mp++) {
                        this->occupancy_number_(mp, m, s1, ia, 0) = std::conj(this->occupancy_number_(m, mp, s1, ia, 0));
                    }
                }
            }

            if (ctx_.num_mag_dims() == 3) {
                // off diagonal blocks
                for (int m = 0; m < lmax_at; m++) {
                    for (int mp = m; mp < lmax_at; mp++) {
                        this->occupancy_number_(mp, m, 2, ia, 0) = std::conj(this->occupancy_number_(m, mp, 3, ia, 0));
                    }
                }
            }
        }
    }

    print_occupancies();
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
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
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

                    if (charge > (lmax_at)) {
                        for (int m = 0; m < lmax_at; m++) {
                            this->occupancy_number_(m, m, majs, ia, 0) = 1.0;
                            this->occupancy_number_(m, m, mins, ia, 0) =
                                (charge - static_cast<double>(lmax_at)) /
                                static_cast<double>(lmax_at);
                        }
                    } else {
                        for (int m = 0; m < lmax_at; m++) {
                            this->occupancy_number_(m, m, majs, ia, 0) =
                                charge / static_cast<double>(lmax_at);
                        }
                    }
                } else {
                    //double c1, s1;
                    //sincos(atom.type().starting_magnetization_theta(), &s1, &c1);
                    double c1 = std::cos(atom.type().starting_magnetization_theta());
                    double_complex cs =
                        double_complex(cos(atom.type().starting_magnetization_phi()), sin(atom.type().starting_magnetization_phi()));
                    double_complex ns[4];

                    if (charge > (lmax_at)) {
                        ns[majs] = 1.0;
                        ns[mins] = (charge - static_cast<double>(lmax_at)) /
                                   static_cast<double>(lmax_at);
                    } else {
                        ns[majs] = charge / static_cast<double>(lmax_at);
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

                    for (int m = 0; m < lmax_at; m++) {
                        this->occupancy_number_(m, m, 0, ia, 0) = ns[0];
                        this->occupancy_number_(m, m, 1, ia, 0) = ns[1];
                        this->occupancy_number_(m, m, 2, ia, 0) = ns[2];
                        this->occupancy_number_(m, m, 3, ia, 0) = ns[3];
                    }
                }
            } else {
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    for (int m = 0; m < lmax_at; m++) {
                        this->occupancy_number_(m, m, s, ia, 0) =
                            charge * 0.5 / static_cast<double>(lmax_at);
                    }
                }
            }
        }
    }

    print_occupancies();
}


inline int natural_lm_to_qe(int m, int l)
{
    return (m > 0) ? 2 * m - 1 : -2 * m;
}

inline void set_hubbard_occupancies_matrix(double *occ, int ld)
{
    mdarray<double, 4> occupation_(occ, ld, ld, ctx_.num_spins(), ctx_.unit_cell().num_atoms());
    this->occupancy_number_.zero();
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l ; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                for(int s = 0; s < ctx_.num_spins(); s++) {
                    this->occupancy_number_(l + m1, l + m2, s, ia, 0) = occupation_(mm1, mm2, s, ia);
                }
            }
        }
    }
}

inline void set_hubbard_occupancies_matrix_nc(double_complex *occ, int ld)
{
    mdarray<double_complex, 4> occupation_(occ, ld, ld, 4, ctx_.unit_cell().num_atoms());
    this->occupancy_number_.zero();
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                this->occupancy_number_(l + m1, l + m2, 0, ia, 0) = occupation_(mm1, mm2, 0, ia);
                this->occupancy_number_(l + m1, l + m2, 1, ia, 0) = occupation_(mm1, mm2, 3, ia);
                this->occupancy_number_(l + m1, l + m2, 2, ia, 0) = occupation_(mm1, mm2, 1, ia);
                this->occupancy_number_(l + m1, l + m2, 3, ia, 0) = occupation_(mm1, mm2, 2, ia);
            }
        }
    }
}

inline void get_hubbard_occupancies_matrix(double *occ, int ld)
{
    mdarray<double, 4> occupation_(occ, ld, ld, ctx_.num_spins(), ctx_.unit_cell().num_atoms());

    // we have a factor 1/2 to apply because sirius in the basic LDA
    // case add up the spin degenary to the band occupation
    assert(ctx_.num_mag_dims() != 3);
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                for (int s = 0; s < ctx_.num_spins(); s++) {
                    occupation_(mm1, mm2, s, ia) = this->occupancy_number_(l + m1, l + m2, s, ia, 0).real();
                }
            }
        }
    }
}

inline void get_hubbard_occupancies_matrix_nc(double_complex *occ, int ld)
{
    mdarray<double_complex, 4> occupation_(occ, ld, ld, 4, ctx_.unit_cell().num_atoms());

    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ia++) {
        const int l = ctx_.unit_cell().atom(ia).type().hubbard_l();
        for (int m1 = -l; m1 <= l; m1++) {
            const int mm1 = natural_lm_to_qe(m1, l);
            for (int m2 = -l; m2 <= l; m2++) {
                const int mm2 = natural_lm_to_qe(m2, l);
                occupation_(mm1, mm2, 0, ia) = this->occupancy_number_(l + m1, l + m2, 0, ia, 0);
                occupation_(mm1, mm2, 3, ia) = this->occupancy_number_(l + m1, l + m2, 1, ia, 0);
                occupation_(mm1, mm2, 1, ia) = this->occupancy_number_(l + m1, l + m2, 2, ia, 0);
                occupation_(mm1, mm2, 2, ia) = this->occupancy_number_(l + m1, l + m2, 3, ia, 0);
            }
        }
    }
}

inline void print_occupancies()
{
    if (ctx_.control().verbosity_ > 1) {
        if (ctx_.comm().rank() == 0) {
            printf("\n");
            for(int ci = 0; ci < 10; ci++)
                printf("--------");
            printf("\n");
            printf("hubbard occupancies\n");
            for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
                printf("Atom : %d\n", ia);
                printf("Mag Dim : %d\n", ctx_.num_mag_dims());
                const auto& atom = unit_cell_.atom(ia);
                const int lmax_at = 2 * atom.type().hubbard_l() + 1;

                if (atom.type().hubbard_correction()) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 0, ia, 0)));
                        }

                        if (ctx_.num_mag_dims() == 3) {
                            printf(" ");
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 2, ia, 0)));
                            }
                        }
                        printf("\n");
                    }

                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            if (ctx_.num_mag_dims() == 3) {
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 3, ia, 0)));
                                }
                                printf(" ");
                            }
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                printf("%.3lf ", std::abs(this->occupancy_number_(m1, m2, 1, ia, 0)));
                            }
                            printf("\n");
                        }
                    }

                    double n_up, n_down, n_total;
                    n_up   = 0.0;
                    n_down = 0.0;
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        n_up += this->occupancy_number_(m1, m1, 0, ia, 0).real();
                    }

                    if (ctx_.num_spins() == 2) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            n_down += this->occupancy_number_(m1, m1, 1, ia, 0).real();
                        }
                    }
                    printf("\n");
                    n_total = n_up + n_down;
                    if (ctx_.num_spins() == 2) {
                        printf("Atom charge (total) %.5lf (n_up) %.5lf (n_down) %.5lf (mz) %.5lf\n", n_total, n_up,
                               n_down, n_up - n_down);
                    } else {
                        printf("Atom charge (total) %.5lf\n", 2.0 * n_total);
                    }

                    printf("\n");
                    for(int ci = 0; ci < 10; ci++)
                        printf("--------");
                    printf("\n");

                }
            }
        }
    }
}

inline void symmetrize_occupancy_matrix_noncolinear_case()
{
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
                const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                if (atom.type().hubbard_correction()) {
                    for (int ii = 0; ii < lmax_at; ii++) {
                        int l1 = Utils::lm_by_l_m(atom.type().hubbard_l(), ii - atom.type().hubbard_l());
                        for (int ll = 0; ll < lmax_at; ll++) {
                            int l2 = Utils::lm_by_l_m(atom.type().hubbard_l(), ll - atom.type().hubbard_l());
                            mdarray<double_complex, 1> rot_spa(ctx_.num_spins() * ctx_.num_spins());
                            rot_spa.zero();
                            for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                                for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                    // symmetrization procedure
                                    // A_ij B_jk C_kl

                                    for (int jj = 0; jj < lmax_at; jj++) {
                                        int l3 =
                                            Utils::lm_by_l_m(atom.type().hubbard_l(), jj - atom.type().hubbard_l());
                                        for (int kk = 0; kk < lmax_at; kk++) {
                                            int l4 = Utils::lm_by_l_m(atom.type().hubbard_l(),
                                                                      kk - atom.type().hubbard_l());
                                            rot_spa(2 * s1 + s2) +=
                                                std::conj(rotm(l1, l3)) *
                                                occupancy_number_(jj, kk, (s1 == s2) * s1 + (s1 != s2) * (1 + 2 * s1 + s2), ia, 0) *
                                                rotm(l2, l4) * alpha;
                                        }
                                    }
                                }
                            }

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
                        }
                    }
                }
            }
        }

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            const int lmax_at = 2 * atom.type().hubbard_l() + 1;
            if (atom.type().hubbard_correction()) {
                for (int ii = 0; ii < lmax_at; ii++) {
                    for (int ll = 0; ll < lmax_at; ll++) {
                        for (int s = 0; s < ctx_.num_spins() * ctx_.num_spins(); s++) {
                            this->occupancy_number_(ii, ll, s, ia, 0) = rotated_oc(ii, ll, s, ia);
                        }
                    }
                }
            }
        }
    }
}

inline void symmetrize_occupancy_matrix()
{
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
                const int lmax_at = 2 * atom.type().hubbard_l() + 1;
                if (atom.type().hubbard_correction()) {
                    dmatrix<double_complex> rot_spa(lmax_at, lmax_at);
                    rot_spa.zero();
                    for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                        for (int ii = 0; ii < lmax_at; ii++) {
                            int l1 = Utils::lm_by_l_m(atom.type().hubbard_l(), ii - atom.type().hubbard_l());
                            for (int ll = 0; ll < lmax_at; ll++) {
                                int l2 = Utils::lm_by_l_m(atom.type().hubbard_l(), ll - atom.type().hubbard_l());
                                // symmetrization procedure
                                // A_ij B_jk C_kl
                                for (int kk = 0; kk < lmax_at; kk++) {
                                    int l4 = Utils::lm_by_l_m(atom.type().hubbard_l(),
                                                              kk - atom.type().hubbard_l());
                                    for (int jj = 0; jj < lmax_at; jj++) {
                                        int l3 =
                                            Utils::lm_by_l_m(atom.type().hubbard_l(), jj - atom.type().hubbard_l());
                                        rot_spa(ii, kk) +=
                                            std::conj(rotm(l1, l3)) *
                                            occupancy_number_(jj, kk, ispn, ia, 0) *
                                            rotm(l2, l4) * alpha;
                                    }
                                }
                            }
                        }

                        // we have the rotated occupancy matrix
                        for (int ii = 0; ii < lmax_at; ii++) {
                            for (int jj = 0; jj < lmax_at; jj++) {
                                occupancy_number_(ii, jj, ispn, ia, 0) = rot_spa(ii, jj);
                            }
                        }
                        rot_spa.zero();
                    }

                }
            }
        }

        for (int ia = 0; ia < unit_cell_.num_atoms(); ia++) {
            auto& atom = unit_cell_.atom(ia);
            const int lmax_at = 2 * atom.type().hubbard_l() + 1;
            if (atom.type().hubbard_correction()) {
                for (int ii = 0; ii < lmax_at; ii++) {
                    for (int ll = 0; ll < lmax_at; ll++) {
                        for (int s = 0; s < ctx_.num_spins(); s++) {
                            this->occupancy_number_(ii, ll, s, ia, 0) = rotated_oc(ii, ll, s, ia);
                        }
                    }
                }
            }
        }
    }
}
