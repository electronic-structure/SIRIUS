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


    // for colinear magnetism the wavefunctions of the up and down block
    // are stored in the up and down part of the "spinor" so we
    // calculate Overlap ^ up = <phi_i^up | phi_j^up> stored in the
    // first half of dm the down part in the other half

    // the overlaps are then contiguous
    int Nfc = 1;
    if (ctx_.num_mag_dims() == 1)
        Nfc *= 2;

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals(),
                               n__ * Nfc);

    dm.zero();

    if (ctx_.processing_unit() == GPU) {
        dm.allocate(memory_t::device);
    }

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
        // compute the overlaps
        for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
            inner(ctx_.processing_unit(),
                  ispn,
                  hub_wf,
                  0,
                  this->number_of_hubbard_orbitals(),
                  phi,
                  idx__,
                  n__,
                  dm,
                  0,
                  ispn * n__);
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
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);
                        linalg<CPU>::gemm(0, 0,
                                          lmax_at,
                                          n__,
                                          lmax_at,
                                          linalg_const<double_complex>::one(),
                                          this->hubbard_potential_.template at<CPU>(0, 0, ind, ia, 0),
                                          this->hubbard_potential_.ld(),
                                          dm.template at<CPU>(this->offset[ia] + s2 * lmax_at, 0),
                                          dm.ld(),
                                          linalg_const<double_complex>::one(),
                                          Up.template at<CPU>(s1 * lmax_at, 0), Up.ld());
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
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    Up.zero();
                    linalg<CPU>::gemm(0, 0,
                                      lmax_at,
                                      n__,
                                      lmax_at,
                                      linalg_const<double_complex>::one(),
                                      this->hubbard_potential_.template at<CPU>(0, 0, ispn, ia, 0),
                                      this->hubbard_potential_.ld(),
                                      dm.template at<CPU>(this->offset[ia], ispn * n__),
                                      dm.ld(),
                                      linalg_const<double_complex>::one(),
                                      Up.template at<CPU>(0, 0),
                                      Up.ld());

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
