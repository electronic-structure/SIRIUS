/// this function computes the hubbard contribution to the hamiltonian
/// and add it to ophi.

/// Note that when applied in combination with ultra soft pp, phi should
/// be S|phi> not phi.
template<typename T> void hubbard::apply_hubbard_correction(const K_point &kp, Wavefunctions &phi, Wavefunctions &ophi)
{

  mdarray<T, 4> dm(hubbard_dim, // per atom
                   phi.conponents(0).num_wf(),
                   2,
                   2);

  // should be the number of occupied bands
  mdarray<T, 2> tmp(2 * this->hubbard_lmax_ + 1, phi.components(0).num_wf());
  // First calculate the projections
  // dm(i,n,sigma,sigma')  = <phi_i^\sigma | psi_{nk}>

  // compute the atomic orbitals. I do not store them.
  generate_centered_atomic_orbitals(kp);

  for(int s1 = 0; s1 < ctx_.num_spins(); s1++) {
    for(int s2 = 0; s2 < ctx_.num_spins(); s2++) {
      // compute <phi_i|psi_nk> the bands been distributed over the pool
      linalg<CPU>::gemm(2,
                        0,
                        this->hubbard_dim,
                        this->phi_.components(s1).num_wf(),
                        this->phi_.components(s1).pw_coeffs().num_rows_loc(),
                        *reinterpret_cast<double_complex*>(&alpha),
                        this->phi_.components(s1).pw_coeffs().prime().at<CPU>(0, 0),
                        this->phi_.components(s1).pw_coeffs().prime().ld(),
                        phi.components(s2).pw_coeffs().prime().at<CPU>(0, 0),
                        phi.components(s2).pw_coeffs().prime().ld(),
                        *reinterpret_cast<double_complex*>(&beta),
                        reinterpret_cast<double_complex*>(dm.at<CPU>(0, 0, s1, s2)),
                        dm.ld());

      int s = (s1 == s2) + (s1 != s2) * (2 * s1 + 3 * s2);
    }
  }

  kp.comm().allreduce<T, mpi_op_t::sum>(dm.at<CPU>(), dm.size());

  if(ctx_.num_mag_dims() != 3) {
    // colinear case

  } else {
    for(int ia = 0; ia < ctx_.unit_cell().num_atoms(); ++ia) {
      const auto atom = ctx_.unit_cell().atoms(ia);
      if(atom.atom_type().hubbard_correction()) {
        for(int s1 = 0; s1 < ctx_.num_spins(); s1++) {
          for(int s2 = 0; s2 < ctx_.num_spins(); s2++) {
            for(int s3 = 0; s3 < ctx_.num_spins(); s3++) {
              linalg<CPU>::gemm(0,
                                0,
                                2 * atom.atom_type().lmax() + 1,
                                this->phi_.components(s1).num_wf(),
                                2 * atom.atom_type().lmax() + 1,
                                *reinterpret_cast<double_complex*>(&alpha),
                                this->hubbard_potential_.at<CPU>(0, 0, s1, s3, ia),
                                this->hubbard_potential_.ld(),
                                dm.at<CPU>(this->offset[ia], 0, s3, s2),
                                dm.ld(),
                                *reinterpret_cast<double_complex*>(&beta),
                                reinterpret_cast<double_complex*>(tmp.at<CPU>(0, 0)),
                                tmp.ld());
              linalg<CPU>::gemm(1,
                                0,
                                2 * atom.atom_type().lmax() + 1,
                                this->phi_.components(s1).pw_coeffs().num_rows_loc(),
                                2 * atom.atom_type().lmax() + 1,
                                *reinterpret_cast<double_complex*>(&alpha),
                                tmp.at<CPU>(),
                                tmp.ld(),
                                this->phi_.components(s1).pw_coeffs().prime().at<CPU>(0, this->offset[ia]),
                                this->phi_.components(s1).pw_coeffs().prime().ld(),
                                *reinterpret_cast<double_complex*>(&beta),
                                reinterpret_cast<double_complex*>(ophi.components(s3).pw_coeffs().prime().at<CPU>(0, 0)),
                                ophi.components(s3).pw_coeffs().prime().ld());
            }
          }
        }
      }
    }
  }
}
