// this function computes the hubbard contribution to the hamiltonian
// and add it to ophi.

// the S matrix is already applied to phi_i

void apply_hubbard_potential(K_point& kp,
                             const int ispn_,
                             const int idx__,
                             const int n__,
                             Wave_functions& phi,
                             Wave_functions& ophi)
{
    auto &hub_wf = kp.hubbard_wave_functions();

    dmatrix<double_complex> dm(this->number_of_hubbard_orbitals(), n__);
    dm.zero();

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        dm.allocate(memory_t::device);
    }
    #endif

    // First calculate the local part of the projections
    // dm(i, n)  = <phi_i | psi_{nk}>

    inner(ctx_.processing_unit(),
          ispn_,
          hub_wf,
          0,
          this->number_of_hubbard_orbitals(),
          phi,
          idx__,
          n__,
          dm,
          0,
          0);

    // free memory on GPU
    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        dm.copy<memory_t::device, memory_t::host>();
    }
    #endif

    dmatrix<double_complex> Up;

    Up = dmatrix<double_complex>(this->number_of_hubbard_orbitals(), n__);
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
                                    Up(this->offset[ia] + s1 * lmax_at + m1, nbd) += this->hubbard_potential_(m2, m1, ind, ia, 0) *
                                        dm(this->offset[ia] + s2 * lmax_at + m2, nbd);
                                }
                            }
                        }
                    }
                }
            } else {
                // Conventional LDA or colinear magnetism
                for (int nbd = 0; nbd < n__; nbd++) {
                    for (int m1 = 0; m1 < lmax_at; m1++) {
                        for (int m2 = 0; m2 < lmax_at; m2++) {
                            Up(this->offset[ia] + m1, nbd) += this->hubbard_potential_(m2, m1, ispn_, ia, 0) *
                                dm(this->offset[ia] + m2, nbd);
                        }
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
                              ispn_,
                              1.0,
                              hub_wf,
                              0,
                              this->number_of_hubbard_orbitals(),
                              Up,
                              0,
                              0,
                              1.0,
                              ophi,
                              idx__,
                              n__);

    #ifdef __GPU
    if (ctx_.processing_unit() == GPU) {
        Up.deallocate(memory_t::device);
    }
    #endif
}
