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

    auto dm = hub_wf.overlap<double_complex>(ctx_.processing_unit(),
                                             phi,
                                             0,
                                             number_of_hubbard_orbitals(),
                                             idx__,
                                             n__);

    // Need a reduction over the pool
    kp.comm().allreduce<double_complex, mpi_op_t::sum>(dm.at<CPU>(), static_cast<int>(dm.size()));

    for (int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
        const auto& atom = ctx_.unit_cell().atom(ia);
        if (atom.type().hubbard_correction()) {

            // we apply the hubbard correction. For now I have no papers
            // giving me the formula for the SO case so I rely on QE for it
            // but I do not like it at all
            #pragma omp parallel for
            for (int nbnd = 0; nbnd < n__; nbnd++) {
                for (int s1 = 0; s1 < ctx_.num_spins(); s1++) {
                    const int lmax_at = 2 * atom.type().hubbard_l() + 1;
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

                        if (ctx_.processing_unit() == CPU) {
                            for (int s = 0; s < ctx_.num_spins(); s++) {
                                for (int l = 0; l < hub_wf.pw_coeffs(s).num_rows_loc(); l++) {
                                    ophi.pw_coeffs(s).prime(l, idx__ + nbnd) += temp *
                                        hub_wf.pw_coeffs(s).prime(l, this->offset[ia] + s1 * lmax_at + m1);
                                }
                            }
                        } else {
#ifdef __GPU
                            for (int s = 0; s < ctx_.num_spins(); s++) {
                                linalg<GPU>::axpy(hub_wf.pw_coeffs(s).num_rows_loc(),
                                                  temp,
                                                  hub_wf.pw_coeffs(s).prime().at(0,
                                                                                 this->offset[ia] +
                                                                                 s1 * lmax_at + m1),
                                                  1,
                                                  ophi.pw_coeffs(s).prime().at(0, idx__ + nbnd),
                                                  1);
                            }
#endif
                        }
                    }
                }
            }
        }
    }
}
