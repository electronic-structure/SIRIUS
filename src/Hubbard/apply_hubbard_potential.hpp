// this function computes the hubbard contribution to the hamiltonian
// and add it to ophi.

// the S matrix is already applied to phi_i

void apply_hubbard_potential(K_point& kp,
                             const int idx__,
                             const int n__,
                             Wave_functions& phi,
                             Wave_functions& ophi)
{
    auto &hub_wf = kp.hubbard_wave_functions();
    // First calculate the local part of the projections
    // dm(i, n)  = <phi_i | psi_{nk}>

    int Nfc = 1;
    if (ctx_.num_mag_dims() == 1)
        Nfc *= 2;

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals(),
                               n__ * Nfc);

    dm.zero();

    if (ctx_.num_mag_dims() == 3) {
        inner(ctx_.processing_unit(),
              2,
              hub_wf,
              0,
              this->number_of_hubbard_orbitals(),
              phi,
              idx__,
              n__,
              dm,
              0,
              0);
    } else {
        inner(ctx_.processing_unit(),
              0,
              hub_wf,
              0,
              this->number_of_hubbard_orbitals(),
              phi,
              idx__,
              n__,
              dm,
              0,
              0);

        // colinear case
        if (ctx_.num_spins() == 2) {
            inner(ctx_.processing_unit(),
                  1,
                  hub_wf,
                  0,
                  this->number_of_hubbard_orbitals(),
                  phi,
                  idx__,
                  n__,
                  dm,
                  0,
                  n__);
        }
    }


    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
        const auto& atom = ctx_.unit_cell().atom(ia);
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
        if (atom.type().hubbard_correction()) {
            // we apply the hubbard correction. For now I have no papers
            // giving me the formula for the SO case so I rely on QE for it
            // but I do not like it at all
            if (ctx_.num_mag_dims() == 3) {
                dmatrix<double_complex> Up(lmax_at * ctx_.num_spins(), n__);
                Up.zero();
#pragma omp parallel for
                for (int nbnd = 0; nbnd < n__; nbnd++) {
                    for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            double_complex temp = linalg_const<double_complex>::zero();

                            // computes \f[
                            // \sum_{\sigma,m} V^{\sigma\sigma'}_{m,m'} \left<\phi_m^\sigma|\Psi_{nk}\right>
                            // \f]
                            for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                                const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
                                for (int m2 = 0; m2 < lmax_at; m2++) {
                                    temp += this->hubbard_potential_(m1, m2, ind, ia, 0) *
                                        dm(this->offset[ia] + s2 * lmax_at + m2, nbnd);
                                }
                            }
                            Up(s1 * lmax_at + m1, nbnd) = temp;
                        }
                    }
                }

                transform<double_complex>(ctx_.processing_unit(),
                                          2,
                                          1.0,
                                          hub_wf,
                                          this->offset[ia],
                                          2 * lmax_at,
                                          Up,
                                          0,
                                          0,
                                          1.0,
                                          ophi,
                                          idx__,
                                          n__);
            } else {
                //Conventional LDA or colinear magnetism
                dmatrix<double_complex> Up(lmax_at, n__);
                Up.zero();
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
#pragma omp parallel for
                    for (int nbnd = 0; nbnd < n__; nbnd++) {
                        for (int m1 = 0; m1 < lmax_at; m1++) {
                            double_complex temp = linalg_const<double_complex>::zero();

                            // computes \f[
                            // \sum_{\sigma,m} V^{\sigma\sigma'}_{m,m'} \left<\phi_m^\sigma|\Psi_{nk}\right>
                            // \f]
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                    temp += this->hubbard_potential_(m1, m2, ispn, ia, 0) *
                                        dm(this->offset[ia] + m2, nbnd + ispn * n__);
                            }
                            Up(m1, nbnd) = temp;
                        }
                    }
                    transform<double_complex>(ctx_.processing_unit(),
                                              ispn,
                                              1.0,
                                              hub_wf,
                                              this->offset[ia],
                                              lmax_at,
                                              Up,
                                              0,
                                              0,
                                              1.0,
                                              ophi,
                                              idx__,
                                              n__);
                }
            }
        }
    }
}
