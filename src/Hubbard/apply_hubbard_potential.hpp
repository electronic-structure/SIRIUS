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


    // this stuff is a little odd. When we are doing calculation with
    // colinear magnetism, the up and down components are independent
    // but still stored in the same wavefunction. The same is true for
    // the hubbard wave functions to minimize the amount of memory used.

    // so for colinear magnetism, the spin index of the hubbard
    // wavefunctions should treated differently from the non colinear
    // magnetic case since the up and down component are stored in the
    // same wavefunction.  So the overlap matrix is twice as large as
    // the number of hubbard orbitals.

    dmatrix<double_complex> dm(Nfc * this->number_of_hubbard_orbitals(), n__);
    dm.zero();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        dm.allocate(memory_t::device);
    }
    #endif

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
                  ispn * this->number_of_hubbard_orbitals(),
                  0);
        }
    }

    // free memory on GPU
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        dm.copy<memory_t::device, memory_t::host>();
    }
    #endif

    dmatrix<double_complex> Up;

    if (ctx_.num_mag_dims() == 3)
        Up = dmatrix<double_complex>((2 * this->lmax_ + 1) * ctx_.num_spins(), n__);
    else
        Up = dmatrix<double_complex>((2 * this->lmax_ + 1), n__);
    Up.zero();

    #ifdef __GPU
    // the communicator is always of size 1.  I need to allocate memory
    // on the device manually

    if (ctx_.processing_unit() == GPU) {
        Up.allocate(memory_t::device);
    }
    #endif

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
        const auto& atom = ctx_.unit_cell().atom(ia);
        const int lmax_at = 2 * atom.type().hubbard_l() + 1;
        if (atom.type().hubbard_correction()) {
            // we apply the hubbard correction. For now I have no papers
            // giving me the formula for the SO case so I rely on QE for it
            // but I do not like it at all
            if (ctx_.num_mag_dims() == 3) {
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    for (int s2 = 0; s2 < ctx_.num_spins(); s2++) {
                        const int ind = (s1 == s2) * s1 + (1 + 2 * s2 + s1) * (s1 != s2);

                        // !!! Replace this with matrix matrix multiplication

                        for (int nbd = 0; nbd < n__; nbd++) {
                            for (int m2 = 0; m2 < lmax_at; m2++) {
                                for (int m1 = 0; m1 < lmax_at; m1++) {
                                    Up(s1 * lmax_at + m1, nbd) += this->hubbard_potential_(m1, m2, ind, ia, 0) *
                                        dm(this->offset[ia] + s2 * lmax_at + m2, nbd);
                                }
                            }
                        }
                    }
                }

                #ifdef __GPU
                if (ctx_.processing_unit() == GPU) {
                    Up.copy<memory_t::host, memory_t::device>();
                }
                #endif

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
                for (int ispn = 0; ispn < ctx_.num_spins(); ispn++) {
                    Up.zero();
                    for (int nbd = 0; nbd < n__; nbd++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            for (int m1 = 0; m1 < lmax_at; m1++) {
                                Up(m1, nbd) += this->hubbard_potential_(m1, m2, ispn, ia, 0) *
                                    dm(this->offset[ia] + m2 + ispn * this->number_of_hubbard_orbitals(), nbd);
                            }
                        }
                    }

                    #ifdef __GPU
                    if (ctx_.processing_unit() == GPU) {
                        Up.copy<memory_t::host, memory_t::device>();
                    }
                    #endif

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

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        Up.deallocate(memory_t::device);
    }
    #endif
}
