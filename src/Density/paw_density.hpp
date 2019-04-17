// Copyright (c) 2013-2018 Anton Kozhevnikov, Ilia Sivkov, Thomas Schulthess
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are permitted provided that
// the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the
//    following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
//    and the following disclaimer in the documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED
// WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

/** \file paw_density.hpp
 *
 *  \brief Generate PAW density.
 */

inline void Density::init_paw()
{
    paw_density_data_.clear();

    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        int   ia_paw    = unit_cell_.spl_num_paw_atoms(i);
        int   ia        = unit_cell_.paw_atom_index(ia_paw);
        auto& atom      = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int l_max      = 2 * atom_type.indexr().lmax_lo();
        int lm_max_rho = utils::lmmax(l_max);

        paw_density_data_t pdd;

        pdd.atom_ = &atom;

        pdd.ia = ia;

        // allocate density arrays
        for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
            pdd.ae_density_.push_back(Spheric_function<function_domain_t::spectral, double>(lm_max_rho, pdd.atom_->radial_grid()));
            pdd.ps_density_.push_back(Spheric_function<function_domain_t::spectral, double>(lm_max_rho, pdd.atom_->radial_grid()));
        }

        paw_density_data_.push_back(std::move(pdd));
    }
}

inline void Density::init_density_matrix_for_paw()
{
    density_matrix_.zero();

    for (int ipaw = 0; ipaw < unit_cell_.num_paw_atoms(); ipaw++) {
        int ia = unit_cell_.paw_atom_index(ipaw);

        auto& atom      = unit_cell_.atom(ia);
        auto& atom_type = atom.type();

        int nbf = atom_type.mt_basis_size();

        auto& occupations = atom_type.paw_wf_occ();

        /* magnetization vector */
        auto magn = atom.vector_field();

        for (int xi = 0; xi < nbf; xi++) {
            auto& basis_func_index_dsc = atom_type.indexb()[xi];

            int rad_func_index = basis_func_index_dsc.idxrf;

            double occ = occupations[rad_func_index];

            int l = basis_func_index_dsc.l;

            switch (ctx_.num_mag_dims()) {
                case 0: {
                    density_matrix_(xi, xi, 0, ia) = occ / double(2 * l + 1);
                    break;
                }

                case 3:
                case 1: {
                    double nm                      = (std::abs(magn[2]) < 1.0) ? magn[2] : std::copysign(1, magn[2]);
                    density_matrix_(xi, xi, 0, ia) = 0.5 * (1.0 + nm) * occ / double(2 * l + 1);
                    density_matrix_(xi, xi, 1, ia) = 0.5 * (1.0 - nm) * occ / double(2 * l + 1);
                    break;
                }
            }
        }
    }
}

inline void Density::generate_paw_atom_density(paw_density_data_t& pdd)
{
    int ia = pdd.ia;

    auto& atom_type = pdd.atom_->type();

    auto l_by_lm = utils::l_by_lm(2 * atom_type.indexr().lmax_lo());

    /* get gaunt coefficients */
    Gaunt_coefficients<double> GC(atom_type.indexr().lmax_lo(), 2 * atom_type.indexr().lmax_lo(),
                                  atom_type.indexr().lmax_lo(), SHT::gaunt_rlm);

    for (int i = 0; i < ctx_.num_mag_dims() + 1; i++) {
        pdd.ae_density_[i].zero();
        pdd.ps_density_[i].zero();
    }

    /* get radial grid to divide density over r^2 */
    auto& grid = atom_type.radial_grid();

    auto& paw_ae_wfs = atom_type.ae_paw_wfs_array();
    auto& paw_ps_wfs = atom_type.ps_paw_wfs_array();

    /* iterate over local basis functions (or over lm1 and lm2) */
    for (int xi2 = 0; xi2 < atom_type.indexb().size(); xi2++) {
        int lm2  = atom_type.indexb(xi2).lm;
        int irb2 = atom_type.indexb(xi2).idxrf;

        for (int xi1 = 0; xi1 <= xi2; xi1++) {
            int lm1  = atom_type.indexb(xi1).lm;
            int irb1 = atom_type.indexb(xi1).idxrf;

            /* get num of non-zero GC */
            int num_non_zero_gk = GC.num_gaunt(lm1, lm2);

            double diag_coef = (xi1 == xi2) ? 1.0 : 2.0;

            /* store density matrix in aux form */
            double dm[4] = {0, 0, 0, 0};
            switch (ctx_.num_mag_dims()) {
                case 3: {
                    dm[2] = 2 * std::real(density_matrix_(xi1, xi2, 2, ia));
                    dm[3] = -2 * std::imag(density_matrix_(xi1, xi2, 2, ia));
                }
                case 1: {
                    dm[0] = std::real(density_matrix_(xi1, xi2, 0, ia) + density_matrix_(xi1, xi2, 1, ia));
                    dm[1] = std::real(density_matrix_(xi1, xi2, 0, ia) - density_matrix_(xi1, xi2, 1, ia));
                    break;
                }
                case 0: {
                    dm[0] = std::real(density_matrix_(xi1, xi2, 0, ia));
                }
            }

            for (int imagn = 0; imagn < ctx_.num_mag_dims() + 1; imagn++) {
                auto& ae_dens = pdd.ae_density_[imagn];
                auto& ps_dens = pdd.ps_density_[imagn];

                /* add nonzero coefficients */
                for (int inz = 0; inz < num_non_zero_gk; inz++) {
                    auto& lm3coef = GC.gaunt(lm1, lm2, inz);

                    /* iterate over radial points */
                    for (int irad = 0; irad < grid.num_points(); irad++) {

                        /* we need to divide density over r^2 since wave functions are stored multiplied by r */
                        double inv_r2 = diag_coef / (grid[irad] * grid[irad]);

                        /* calculate unified density/magnetization
                         * dm_ij * GauntCoef * ( phi_i phi_j  +  Q_ij) */
                        ae_dens(lm3coef.lm3, irad) += dm[imagn] * inv_r2 * lm3coef.coef * paw_ae_wfs(irad, irb1) * paw_ae_wfs(irad, irb2);
                        ps_dens(lm3coef.lm3, irad) += dm[imagn] * inv_r2 * lm3coef.coef *
                                                      (paw_ps_wfs(irad, irb1) * paw_ps_wfs(irad, irb2) + atom_type.q_radial_function(irb1, irb2, l_by_lm[lm3coef.lm3])(irad));
                    }
                }
            }
        }
    }
}

inline void Density::generate_paw_loc_density()
{
    if (!unit_cell_.num_paw_atoms()) {
        return;
    }

    #pragma omp parallel for
    for (int i = 0; i < unit_cell_.spl_num_paw_atoms().local_size(); i++) {
        generate_paw_atom_density(paw_density_data_[i]);
    }
}
