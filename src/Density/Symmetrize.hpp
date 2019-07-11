/// Apply a given rotation in angular momentum subspace l and spin space s =
/// 1/2. note that we only store three components uu, dd, ud

/* we compute \f[ Oc = (R_l\cross R_s) . O . (R_l\cross R_s)^\dagger \f] */

inline void apply_symmetry(const mdarray<double_complex, 3> &dm_,
                           const mdarray<double, 2> &rot,
                           const mdarray<double_complex, 2> &spin_rot_su2,
                           const int num_mag_dims_,
                           const int l,
                           mdarray<double_complex, 3> &res_)
{
    res_.zero();
    for (int lm1 = 0; lm1 <= 2 * l + 1; lm1++) {
        for (int lm2 = 0; lm2 <= 2 * l + 1; lm2++) {
            res_.zero();
            double_complex dm_rot_spatial[3];

            for (int j = 0; j < num_mag_dims_; j++) {
                // this is a matrix-matrix multiplication P A P ^-1
                dm_rot_spatial[j] = 0.0;
                for (int lm3 = 0; lm3 <= 2 * l + 1; lm3++) {
                    for (int lm4 = 0; lm4 <= 2 * l + 1; lm4++) {
                        dm_rot_spatial[j] += dm_(lm3, lm4, j) * rot(l * l + lm1, l * l + lm3) * rot(l * l + lm2, l * l + lm4);
                    }
                }
            }
            if (num_mag_dims_ != 3)
                for (int j = 0; j < num_mag_dims_; j++) {
                    res_(lm1, lm2, j) += dm_rot_spatial[j];
                }
            else {
                // full non colinear magnetism
                double_complex spin_dm[2][2] = {
                    {dm_rot_spatial[0], dm_rot_spatial[2]},
                    {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                /* spin blocks of density matrix are: uu, dd, ud
                   the mapping from linear index (0, 1, 2) of density matrix components is:
                   for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                   for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                */
                for (int k = 0; k < num_mag_dims_; k++) {
                    for (int is = 0; is < 2; is++) {
                        for (int js = 0; js < 2; js++) {
                            res_(lm1, lm2, k) += spin_rot_su2(k & 1, is) * spin_dm[is][js] * std::conj(spin_rot_su2(std::min(k, 1), js));
                        }
                    }
                }
            }
        }
    }
}


/*
  symmetrize the occupation matrix according to a given list of beta or wfc
  list. the symmetrization does depend explicitly on the beta or wfc the last
  parameter is on when the atom has spin-orbit coupling and hubbard correction in
  that case, we must skip half of the indices because of the averaging of the
  radial integrals over the total angular momentum
*/

void Symmetrize(const mdarray<double_complex, 4> &ns_,
                const basis_functions_index &indexb,
                const int ia,
                const int ja,
                const int ndm,
                const mdarray<double, 2> &rotm,
                const mdarray<double_complex, 2> &spin_rot_su2,
                mdarray<double_complex, 4> &dm_,
                const bool hubbard_)
{
    for (int xi1 = 0; xi1 < indexb.size(); xi1++) {
        int l1  = indexb[xi1].l;
        int lm1 = indexb[xi1].lm;
        int o1  = indexb[xi1].order;

        if ((hubbard_)&&(xi1 >= (2 * l1 + 1))) {
            break;
        }

        for (int xi2 = 0; xi2 < indexb.size(); xi2++) {
            int l2  = indexb[xi2].l;
            int lm2 = indexb[xi2].lm;
            int o2  = indexb[xi2].order;
            std::array<double_complex, 3> dm_rot_spatial = {0, 0, 0};

            //} the hubbard treatment when spin orbit coupling is present is
            // foundamentally wrong since we consider the full hubbard
            // coorection with an averaged wave function (meaning we neglect the
            // L.S correction within hubbard). A better option (although still
            // wrong from physics pov) would be to consider a multi orbital case.
            if ((hubbard_)&&(xi2 >= (2 * l2 + 1))) {
                break;
            }

            //      if (l1 == l2) {
            // the rotation matrix of the angular momentum is block
            // diagonal and does not couple different l.
            for (int j = 0; j < ndm; j++) {
                for (int m3 = -l1; m3 <= l1; m3++) {
                    int lm3 = utils::lm(l1, m3);
                    int xi3 = indexb.index_by_lm_order(lm3, o1);
                    for (int m4 = -l2; m4 <= l2; m4++) {
                        int lm4 = utils::lm(l2, m4);
                        int xi4 = indexb.index_by_lm_order(lm4, o2);
                        dm_rot_spatial[j] += ns_(xi3, xi4, j, ja) * rotm(lm1, lm3) * rotm(lm2, lm4);
                    }
                }
            }

            /* magnetic symmetrization */
            if (ndm == 1) {
                dm_(xi1, xi2, 0, ia) += dm_rot_spatial[0];
            } else {
                double_complex spin_dm[2][2] = {
                    {dm_rot_spatial[0], dm_rot_spatial[2]},
                    {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                /* spin blocks of density matrix are: uu, dd, ud
                   the mapping from linear index (0, 1, 2) of density matrix components is:
                   for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                   for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                */
                for (int k = 0; k < ndm; k++) {
                    for (int is = 0; is < 2; is++) {
                        for (int js = 0; js < 2; js++) {
                            dm_(xi1, xi2, k, ia) += spin_rot_su2(k & 1, is) * spin_dm[is][js] * std::conj(spin_rot_su2(std::min(k, 1), js));
                        }
                    }
                }
            }
        }
    }
}
