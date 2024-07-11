/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file symmetrize_density_matrix.hpp
 *
 *  \brief Symmetrize density matrix of LAPW and PW methods.
 */

#ifndef __SYMMETRIZE_DENSITY_MATRIX_HPP__
#define __SYMMETRIZE_DENSITY_MATRIX_HPP__

#include "density/density_matrix.hpp"

namespace sirius {

inline void
apply_symmetry_to_density_matrix(mdarray<std::complex<double>, 3> const& dm_ia__, basis_functions_index const& indexb__,
                                 const int num_mag_comp__, std::vector<mdarray<double, 2>> const& rotm__,
                                 mdarray<std::complex<double>, 2> const& spin_rot_su2__,
                                 mdarray<std::complex<double>, 3>& dm_ja__)
{
    auto& indexr = indexb__.indexr();

    /* loop over radial functions */
    for (auto e1 : indexr) {
        /* angular momentum of radial function */
        auto am1     = e1.am;
        auto ss1     = am1.subshell_size();
        auto offset1 = indexb__.index_of(e1.idxrf);
        for (auto e2 : indexr) {
            /* angular momentum of radial function */
            auto am2     = e2.am;
            auto ss2     = am2.subshell_size();
            auto offset2 = indexb__.index_of(e2.idxrf);

            /* apply spatial rotation */
            for (int m1 = 0; m1 < ss1; m1++) {
                for (int m2 = 0; m2 < ss2; m2++) {
                    std::array<std::complex<double>, 3> dm_rot_spatial = {0, 0, 0};
                    for (int j = 0; j < num_mag_comp__; j++) {
                        for (int m1p = 0; m1p < ss1; m1p++) {
                            for (int m2p = 0; m2p < ss2; m2p++) {
                                dm_rot_spatial[j] += rotm__[am1.l()](m1, m1p) *
                                                     dm_ia__(offset1 + m1p, offset2 + m2p, j) *
                                                     rotm__[am2.l()](m2, m2p);
                            }
                        }
                    }
                    /* non-magnetic case */
                    if (num_mag_comp__ == 1) {
                        dm_ja__(offset1 + m1, offset2 + m2, 0) += dm_rot_spatial[0];
                    } else { /* magnetic symmetrization */
                        std::complex<double> spin_dm[2][2] = {{dm_rot_spatial[0], dm_rot_spatial[2]},
                                                              {std::conj(dm_rot_spatial[2]), dm_rot_spatial[1]}};

                        /* spin blocks of density matrix are: uu, dd, ud
                           the mapping from linear index (0, 1, 2) of density matrix components is:
                           for the first spin index: k & 1, i.e. (0, 1, 2) -> (0, 1, 0)
                           for the second spin index: min(k, 1), i.e. (0, 1, 2) -> (0, 1, 1)
                        */
                        for (int k = 0; k < num_mag_comp__; k++) {
                            for (int is = 0; is < 2; is++) {
                                for (int js = 0; js < 2; js++) {
                                    dm_ja__(offset1 + m1, offset2 + m2, k) +=
                                            spin_rot_su2__(k & 1, is) * spin_dm[is][js] *
                                            std::conj(spin_rot_su2__(std::min(k, 1), js));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Symmetrize density matrix.
/** Density matrix arises in LAPW or PW methods. In PW it is computed in the basis of beta-projectors. Occupancy
 *  matrix is computed for the Hubbard-U correction. In both cases the matrix has the same structure and is
 *  symmetrized in the same way The symmetrization does depend explicitly on the beta or wfc. The last
 *  parameter is on when the atom has spin-orbit coupling and hubbard correction in
 *  that case, we must skip half of the indices because of the averaging of the
 *  radial integrals over the total angular momentum
 *
 *  We start from the spectral represntation of the occupancy operator defined for the irreducible Brillouin
 *  zone:
 *  \f[
 *    \hat N_{IBZ} = \sum_{j} \sum_{{\bf k}}^{IBZ} | \Psi_{j{\bf k}}^{\sigma} \rangle w_{\bf k} n_{j{\bf k}}
 *          \langle \Psi_{j{\bf k}}^{\sigma'} |
 *  \f]
 *  and a set of localized orbitals with the pure angular character (this can be LAPW functions,
 *  beta-projectors or localized Hubbard orbitals) :
 *  \f[
 *    |\phi_{\ell m}^{\alpha {\bf T}}\rangle
 *  \f]
 *  The orbitals are labeled by the angular and azimuthal quantum numbers (\f$ \ell m \f$),
 *  atom index (\f$ \alpha \f$) and a lattice translation vector (\f$ {\bf T} \f$) such that:
 *  \f[
 *    \langle {\bf r} | \phi_{\ell m}^{\alpha {\bf T}} \rangle = \phi_{\ell m}({\bf r - r_{\alpha} - T})
 *  \f]
 *
 *  There might be several localized orbitals per atom. We wish to compute the symmetrized occupation matrix:
 *  \f[
 *    n_{\ell m \alpha {\bf T} \sigma, \ell' m' \alpha' {\bf T}' \sigma'} =
 *      \langle \phi_{\ell m}^{\alpha {\bf T}} | \hat N | \phi_{\ell' m'}^{\alpha' {\bf T'}} \rangle =
 *       \sum_{\bf P} \sum_{j} \sum_{{\bf k}}^{IBZ}
 *       \langle \phi_{\ell m}^{\alpha {\bf T}} | \hat{\bf P} \Psi_{j{\bf k}}^{\sigma} \rangle
 *       w_{\bf k} n_{j{\bf k}}
 *       \langle \hat{\bf P} \Psi_{j{\bf k}}^{\sigma'} | \phi_{\ell' m'}^{\alpha' {\bf T'}} \rangle
 *  \f]
 *
 *  Let's now label the overlap integrals between localized orbitals and KS wave-functions:
 *  \f[
 *    A_{\ell m j{\bf k}}^{\alpha {\bf T} \sigma} = \langle \Psi_{j{\bf k}}^{\sigma} |
 *       \phi_{\ell m}^{\alpha {\bf T}} \rangle =
 *      \int \Psi_{j{\bf k}}^{\sigma *}({\bf r})
 *        \phi_{\ell m}({\bf r} - {\bf r}_{\alpha} - {\bf T}) d{\bf r}
 *  \f]
 *  and check how it transforms under the symmetry operation \f$ \hat {\bf P} = \{ {\bf R} | {\bf t} \} \f$
 *  applied to the KS states.
 *  \f[
 *   \int \big( \hat {\bf P}\Psi_{j {\bf k}}^{\sigma *}({\bf r}) \big)
 *        \phi_{\ell m}({\bf r} - {\bf r}_{\alpha} - {\bf T}) d{\bf r} =
 *   \int \Psi_{j {\bf k}}^{\sigma *}({\bf r})
 *     \big( \hat {\bf P}^{-1} \phi_{\ell m}({\bf r} - {\bf r}_{\alpha} - {\bf T}) \big) d{\bf r}
 *  \f]
 *
 *  Let's first derive how the inverse symmetry operation acts on the localized orbital centered on the
 *  atom inside a unit cell (no <b>T</b>):
 *  \f[
 *   \hat {\bf P}^{-1} \phi\big( {\bf r} -  {\bf r}_{\alpha} \big) =
 *     \phi\big( {\bf R} {\bf r} + {\bf t} - {\bf r}_{\alpha} \big) = \\
 *     \phi\big( {\bf R}({\bf r} - {\bf R}^{-1}({\bf r}_{\alpha} - {\bf t})) \big) =
 *     \tilde \phi \big( {\bf r} - {\bf R}^{-1}({\bf r}_{\alpha} - {\bf t}) \big)
 *  \f]
 *  This operation rotates the orbital and centers it at the position
 *  \f[
 *   {\bf r}_{\beta} = {\bf R}^{-1}{\bf r}_{\alpha} - {\bf R}^{-1}{\bf t} = \hat {\bf P}^{-1}{\bf r}_{\alpha}
 *  \f]
 *
 *
 *  For example, suppose thar we have y-orbital centered at \f$ {\bf r}_{\alpha} = [1, 0] \f$ (black dot)
 *  and a symmetry operation \f$ \hat {\bf P}  = \{ {\bf R} | {\bf t} \} \f$ that rotates by
 *  \f$ \pi/2 \f$ counterclockwise and translates by [1/2, 1/2]:
 *
 *  \image html sym_orbital1.png width=400px
 *
 *  Under this symmetry operation the atom coordinate will transform into [1/2, 3/2] (red dot), but
 *  this is not(!) how the orbital is transformed. The origin of the atom will transform according to
 *  the inverse of \f$ \hat {\bf P} \f$ into \f$ {\bf r}_{\beta} = [-1/2, -1/2] \f$ (blue dot) such that
 *  \f$ \hat {\bf P} {\bf r}_{\beta} = {\bf r}_{\alpha} \f$:
 *
 *  \image html sym_orbital2.png width=400px
 *
 *  To be more precise, we should highlight that the transformed atom coordinate can go out of the original
 *  unit cell and can be brought back with a translation vector:
 *  \f[
 *   \hat {\bf P}^{-1}{\bf r}_{\alpha} = {\bf r}_{\beta} + {\bf T}_{P\alpha\beta}
 *  \f]
 *
 *  Now let's derive how the inverse symmetry operation acts on the localized orbital \f$ \phi({\bf r}) \f$
 *  centered on atom in the arbitrary unit cell:
 *  \f[
 *   \hat {\bf P}^{-1} \phi\big( {\bf r} - {\bf r}_{\alpha} - {\bf T} \big) =
 *     \phi\big( {\bf R} {\bf r} + {\bf t} - {\bf r}_{\alpha} - {\bf T} \big) = \\
 *     \phi\big( {\bf R}({\bf r} - {\bf R}^{-1}({\bf r}_{\alpha} + {\bf T} - {\bf t})) \big) =
 *     \tilde \phi\big( {\bf r} - {\bf R}^{-1}({\bf r}_{\alpha} + {\bf T} - {\bf t}) \big) =
 *     \tilde \phi\big( {\bf r} - {\bf r}_{\beta} - {\bf T}_{P\alpha\beta} - {\bf R}^{-1}{\bf T} \big)
 *  \f]
 *
 *  Now let's check how the atomic orbitals transfrom under the rotational part of the symmetry operation.
 *  The atomic functions of (\f$ \ell m \f$) character is expressed as a product of radial function and a
 *  spherical harmonic:
 *  \f[
 *    \phi_{\ell m}({\bf r}) = \phi_{\ell}(r) Y_{\ell m}(\theta, \phi)
 *  \f]
 *
 *  Under rotation the spherical harmonic is transformed as:
 *  \f[
 *    Y_{\ell m}({\bf P} \hat{\bf r}) = {\bf P}^{-1}Y_{\ell m}(\hat {\bf r}) =
 *       \sum_{m'} D_{m'm}^{\ell}({\bf P}^{-1}) Y_{\ell m'}(\hat {\bf r}) =
 *       \sum_{m'} D_{mm'}^{\ell}({\bf P}) Y_{\ell m'}(\hat {\bf r})
 *  \f]
 *  so
 *  \f[
 *    \tilde \phi_{\ell m}({\bf r}) =\hat {\bf P}^{-1} \phi_{\ell}(r) Y_{\ell m}(\theta, \phi) =
 *      \sum_{m'} D_{mm'}^{\ell}({\bf P}) \phi_{\ell}(r) Y_{\ell m'}(\theta, \phi)
 *  \f]
 *
 *  We will use Bloch theorem to get rid of the translations in the argument of \f$ \tilde \phi \f$:
 *  \f[
 *   \int \Psi_{j {\bf k}}^{\sigma *}({\bf r})
 *    \tilde \phi_{\ell m} \big( {\bf r} - {\bf r}_{\beta} - {\bf T}_{P\alpha\beta} - {\bf R}^{-1}{\bf T} \big)
 *    d{\bf r} =
 *    e^{-i{\bf k}({\bf T}_{P\alpha\beta} + {\bf R}^{-1}{\bf T})} \int \Psi_{j {\bf k}}^{\sigma *}({\bf r})
 *     \tilde \phi_{\ell m} \big( {\bf r} - {\bf r}_{\beta} \big) d{\bf r}
 *  \f]
 *  (the "-" in the phase factor appears because KS wave-functions are complex conjugate) and now we can write
 *  \f[
 *    A_{\ell m j\hat {\bf P}{\bf k}}^{\alpha {\bf T} \sigma} =
 *      e^{-i{\bf k}({\bf T}_{P\alpha\beta} + {\bf R}^{-1}{\bf T})}
 *      \sum_{m'} D_{mm'}^{\ell}({\bf P}) A_{\ell m' j{\bf k}}^{\beta \sigma}
 *  \f]
 *
 *  The final expression for the symmetrized matrix is then
 *  \f[
 *    n_{\ell m \alpha {\bf T} \sigma, \ell' m' \alpha' {\bf T}' \sigma'} =
 *       \sum_{\bf P} \sum_{j} \sum_{{\bf k}}^{IBZ}
 *        A_{\ell m j\hat {\bf P}{\bf k}}^{\alpha {\bf T} \sigma *}
 *       w_{\bf k} n_{j{\bf k}}
 *        A_{\ell' m' j\hat {\bf P}{\bf k}}^{\alpha' {\bf T'} \sigma'} = \\ = \sum_{\bf P} \sum_{j}
 *        \sum_{{\bf k}}^{IBZ}
 *      e^{i{\bf k}({\bf T}_{P\alpha\beta} + {\bf R}^{-1}{\bf T})}
 *      e^{-i{\bf k}({\bf T}_{P\alpha'\beta'} + {\bf R}^{-1}{\bf T'})}
 *      \sum_{m_1 m_2}  D_{mm_1}^{\ell *}({\bf P})  D_{m'm_2}^{\ell'}({\bf P})
 *        A_{\ell m_1 j{\bf k}}^{\beta \sigma *} A_{\ell' m_2 j{\bf k}}^{\beta' \sigma'} w_{\bf k} n_{j{\bf k}}
 *  \f]
 *
 *  In the case of \f$ \alpha = \alpha' \f$ and \f$ {\bf T}={\bf T}' \f$ all the phase-factor exponents disappear
 *  and we get an expression for the "on-site" occupation matrix:
 *
 *  \f[
 *    n_{\ell m \sigma, \ell' m' \sigma'}^{\alpha} =
 *        \sum_{\bf P} \sum_{j} \sum_{{\bf k}}^{IBZ}
 *      \sum_{m_1 m_2}  D_{mm_1}^{\ell *}({\bf P})  D_{m'm_2}^{\ell'}({\bf P})
 *        A_{\ell m_1 j{\bf k}}^{\beta \sigma *} A_{\ell' m_2 j{\bf k}}^{\beta \sigma'}
 *        w_{\bf k} n_{j{\bf k}} = \\ =
 *        \sum_{\bf P} \sum_{m_1 m_2}  D_{mm_1}^{\ell *}({\bf P})  D_{m'm_2}^{\ell'}({\bf P})
 *        \tilde n_{\ell m_1 \sigma, \ell' m_2 \sigma'}^{\beta}
 *  \f]
 *
 *
 *  To compute the overlap integrals between KS wave-functions and localized Hubbard orbitals we insert
 *  resolution of identity (in \f$ {\bf G+k} \f$ planve-waves) between bra and ket:
 *  \f[
 *    \langle  \phi_{\ell m}^{\alpha} | \Psi_{j{\bf k}}^{\sigma} \rangle = \sum_{\bf G}
 *      \phi_{\ell m}^{\alpha *}({\bf G+k}) \Psi_{j}^{\sigma}({\bf G+k})
 *  \f]
 *
 */
inline void
symmetrize_density_matrix(Unit_cell const& uc__, std::vector<std::vector<mdarray<double, 2>>> const& rotm__,
                          density_matrix_t& dm__, int num_mag_comp__)
{
    PROFILE("sirius::symmetrize_density_matrix");

    auto& sym = uc__.symmetry();

    /* quick exit */
    if (sym.size() == 1) {
        return;
    }

    density_matrix_t dm_sym(uc__, num_mag_comp__);

    for (int i = 0; i < sym.size(); i++) {
        auto& spin_rot_su2 = sym[i].spin_rotation_su2;

        for (int ia = 0; ia < uc__.num_atoms(); ia++) {
            int ja = sym[i].spg_op.sym_atom[ia];

            sirius::apply_symmetry_to_density_matrix(dm__[ia], uc__.atom(ia).type().indexb(), num_mag_comp__, rotm__[i],
                                                     spin_rot_su2, dm_sym[ja]);
        }
    }

    double alpha = 1.0 / double(sym.size());
    /* multiply by alpha which is the inverse of the number of symmetries */
    for (int ia = 0; ia < uc__.num_atoms(); ia++) {
        for (size_t i = 0; i < dm__[ia].size(); i++) {
            dm__[ia][i] = dm_sym[ia][i] * alpha;
        }
    }
}

} // namespace sirius

#endif
