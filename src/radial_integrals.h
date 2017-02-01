// Copyright (c) 2013-2017 Anton Kozhevnikov, Thomas Schulthess
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

/** \file radial_integrals.h
 *
 *  \brief Representation of various radial integrals.
 */

#ifndef __RADIAL_INTEGRALS_H__
#define __RADIAL_INTEGRALS_H__

#include "sbessel.h"

namespace sirius {

class Radial_integrals
{
  private:
    /// Basic parameters.
    Simulation_parameters const& param_;

    /// Unit cell.
    Unit_cell const& unit_cell_;

    /// Linear grid up to |G+k|_{max}
    Radial_grid grid_gkmax_;

    /// Linear grid up to |G|_{max}
    Radial_grid grid_gmax_;

    mdarray<Spline<double>, 3> aug_radial_integrals_;

    /// Beta-projector radial integrals.
    mdarray<Spline<double>, 2> beta_radial_integrals_;

    mdarray<Spline<double>, 1> pseudo_core_radial_integrals_;

    inline void generate_pseudo_core_radial_integrals()
    {
        PROFILE("sirius::Radial_integrals::generate_pseudo_core_radial_integrals");
        pseudo_core_radial_integrals_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            pseudo_core_radial_integrals_(iat) = Spline<double>(grid_gmax_);

            Spline<double> ps_core(atom_type.radial_grid());
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                ps_core[ir] = atom_type.pp_desc().core_charge_density[ir];
            }
            ps_core.interpolate();

            #pragma omp parallel for
            for (int iq = 0; iq < grid_gmax_.num_points(); iq++) {
                Spherical_Bessel_functions jl(0, atom_type.radial_grid(), grid_gmax_[iq]);

                pseudo_core_radial_integrals_(iat)[iq] = sirius::inner(jl[0], ps_core, 2, atom_type.num_mt_points());
            }
            pseudo_core_radial_integrals_(iat).interpolate();
        }
    }

    inline void generate_aug_radial_integrals()
    {
        PROFILE("sirius::Radial_integrals::generate_aug_radial_integrals");

        int nmax = unit_cell_.max_mt_radial_basis_size();
        int lmax = unit_cell_.lmax(); 

        /* interpolate <j_{l_n}(q*x) | Q_{xi,xi'}^{l}(x) > with splines */
        aug_radial_integrals_ = mdarray<Spline<double>, 3>(nmax * (nmax + 1) / 2, 2 * lmax + 1, unit_cell_.num_atom_types());
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);

            if (!atom_type.pp_desc().augment) {
                continue;
            }

            /* number of radial beta-functions */
            int nbrf = atom_type.mt_radial_basis_size();
            /* maximum l of beta-projectors */
            int lmax_beta = atom_type.indexr().lmax();

            for (int l = 0; l <= 2 * lmax_beta; l++) {
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                    aug_radial_integrals_(idx, l, iat) = Spline<double>(grid_gmax_);
                }
            }

            /* interpolate Q-operator radial functions */
            mdarray<Spline<double>, 2> qrf_spline(nbrf * (nbrf + 1) / 2, 2 * lmax_beta + 1);
            #pragma omp parallel for
            for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                for (int l3 = 0; l3 <= 2 * lmax_beta; l3++) {
                    qrf_spline(idx, l3) = Spline<double>(atom_type.radial_grid());
                    for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                        qrf_spline(idx, l3)[ir] = atom_type.pp_desc().q_radial_functions_l(ir, idx, l3);
                    }
                    qrf_spline(idx, l3).interpolate();
                }
            }
            
            #pragma omp parallel for
            for (int iq = 0; iq < grid_gmax_.num_points(); iq++) {
                Spherical_Bessel_functions jl(2 * lmax_beta, atom_type.radial_grid(), grid_gmax_[iq]);

                for (int l3 = 0; l3 <= 2 * lmax_beta; l3++) {
                    for (int idxrf2 = 0; idxrf2 < nbrf; idxrf2++) {
                        int l2 = atom_type.indexr(idxrf2).l;
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                            int l1 = atom_type.indexr(idxrf1).l;

                            int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                            
                            if (l3 >= std::abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0) {
                                aug_radial_integrals_(idx, l3, iat)[iq] = sirius::inner(jl[l3], qrf_spline(idx, l3), 0,
                                                                                        atom_type.num_mt_points());
                            }
                        }
                    }
                }
            }

            for (int l = 0; l <= 2 * lmax_beta; l++) {
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                    aug_radial_integrals_(idx, l, iat).interpolate();
                }
            }
        }
    }

    inline void generate_beta_radial_integrals()
    {
        PROFILE("sirius::Radial_integrals::generate_beta_radial_integrals");
    
        /* create space for <j_l(qr)|beta> radial integrals */
        beta_radial_integrals_ = mdarray<Spline<double>, 2>(unit_cell_.max_mt_radial_basis_size(), unit_cell_.num_atom_types());
        
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int nrb = atom_type.mt_radial_basis_size();

            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                beta_radial_integrals_(idxrf, iat) = Spline<double>(grid_gkmax_);
            }
    
            /* interpolate beta radial functions */
            std::vector<Spline<double>> beta_rf(nrb);
            for (int idxrf = 0; idxrf < nrb; idxrf++) {
                beta_rf[idxrf] = Spline<double>(atom_type.radial_grid());
                int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                for (int ir = 0; ir < nr; ir++) {
                    beta_rf[idxrf][ir] = atom_type.pp_desc().beta_radial_functions(ir, idxrf);
                }
                beta_rf[idxrf].interpolate();
            }
    
            #pragma omp parallel for
            for (int iq = 0; iq < grid_gkmax_.num_points(); iq++) {
                Spherical_Bessel_functions jl(unit_cell_.lmax(), atom_type.radial_grid(), grid_gkmax_[iq]);
                for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                    int l  = atom_type.indexr(idxrf).l;
                    int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                    /* compute \int j_l(|G+k|r) beta_l(r) r^2 dr */
                    /* remeber that beta(r) are defined as miltiplied by r */
                    beta_radial_integrals_(idxrf, iat)[iq] = sirius::inner(jl[l], beta_rf[idxrf], 1, nr);
                }
            }
            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                beta_radial_integrals_(idxrf, iat).interpolate();
            }
        }
    }

    inline std::pair<int, double> iqdq_gkmax(double q__) const
    {
        std::pair<int, double> result;
        result.first = static_cast<int>((grid_gkmax_.num_points() - 1) * q__ / param_.gk_cutoff());
        /* delta q = q - q_i */
        result.second = q__ - grid_gkmax_[result.first];
        return std::move(result);
    }

    inline std::pair<int, double> iqdq_gmax(double q__) const
    {
        std::pair<int, double> result;
        result.first = static_cast<int>((grid_gmax_.num_points() - 1) * q__ / param_.pw_cutoff());
        /* delta q = q - q_i */
        result.second = q__ - grid_gmax_[result.first];
        return std::move(result);
    }

  public:
    /// Constructor.
    Radial_integrals(Simulation_parameters const& param__,
                     Unit_cell const& unit_cell__)
        : param_(param__)
        , unit_cell_(unit_cell__)
    {
        grid_gmax_  = Radial_grid(linear_grid, static_cast<int>(12 * param_.pw_cutoff()), 0, param_.pw_cutoff());
        grid_gkmax_ = Radial_grid(linear_grid, static_cast<int>(12 * param_.gk_cutoff()), 0, param_.gk_cutoff());

        if (param_.esm_type() == electronic_structure_method_t::pseudopotential) {
            generate_beta_radial_integrals();
            generate_aug_radial_integrals();
            generate_pseudo_core_radial_integrals();
        }
    }

    inline double beta_radial_integral(int idxrf__, int iat__, double q__) const
    {
        auto iqdq = iqdq_gkmax(q__);
        return beta_radial_integrals_(idxrf__, iat__)(iqdq.first, iqdq.second);
    }

    inline double aug_radial_integral(int idx__, int l__, int iat__, double q__) const
    {
        auto iqdq = iqdq_gmax(q__);
        return aug_radial_integrals_(idx__, l__, iat__)(iqdq.first, iqdq.second);
    }

    inline double pseudo_core_radial_integral(int iat__, double q__) const
    {
        auto iqdq = iqdq_gmax(q__);
        return pseudo_core_radial_integrals_(iat__)(iqdq.first, iqdq.second);
    }
};

}

#endif // __RADIAL_INTEGRALS_H__


