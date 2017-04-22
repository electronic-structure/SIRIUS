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

#include "unit_cell.h"
#include "sbessel.h"

namespace sirius {

/// Base class for all kinds of radial integrals.
template <int N>
class Radial_integrals_base
{
  protected:
    /// Unit cell.
    Unit_cell const& unit_cell_;
    
    /// Linear grid of q-points on which the interpolation of radial integrals is done.
    Radial_grid<double> grid_q_;
   
    /// Array with integrals.
    mdarray<Spline<double>, N> values_;
    
  public:
    /// Constructor.
    Radial_integrals_base(Unit_cell const& unit_cell__, double qmax__, int np__)
        : unit_cell_(unit_cell__)
    {
        grid_q_ = Radial_grid_lin<double>(static_cast<int>(np__ * qmax__), 0, qmax__);
    }
    
    inline std::pair<int, double> iqdq(double q__) const
    {
        std::pair<int, double> result;
        /* find index of q-point */
        result.first = static_cast<int>((grid_q_.num_points() - 1) * q__ / grid_q_.last());
        /* delta q = q - q_i */
        result.second = q__ - grid_q_[result.first];
        return std::move(result);
    }
};

/// Radial integrals of the augmentation operator.
template <bool jl_deriv>
class Radial_integrals_aug: public Radial_integrals_base<3>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|aug");

        /* interpolate <j_{l_n}(q*x) | Q_{xi,xi'}^{l}(x) > with splines */
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
                    values_(idx, l, iat) = Spline<double>(grid_q_);
                }
            }

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spherical_Bessel_functions jl(2 * lmax_beta, atom_type.radial_grid(), grid_q_[iq]);

                for (int l3 = 0; l3 <= 2 * lmax_beta; l3++) {
                    for (int idxrf2 = 0; idxrf2 < nbrf; idxrf2++) {
                        int l2 = atom_type.indexr(idxrf2).l;
                        for (int idxrf1 = 0; idxrf1 <= idxrf2; idxrf1++) {
                            int l1 = atom_type.indexr(idxrf1).l;

                            int idx = idxrf2 * (idxrf2 + 1) / 2 + idxrf1;
                            
                            if (l3 >= std::abs(l1 - l2) && l3 <= (l1 + l2) && (l1 + l2 + l3) % 2 == 0) {
                                if (jl_deriv) {
                                    auto s = jl.deriv_q(l3);
                                    values_(idx, l3, iat)[iq] = sirius::inner(s, atom_type.q_rf(idx, l3), 0,
                                                                              atom_type.num_mt_points());
                                } else {
                                    values_(idx, l3, iat)[iq] = sirius::inner(jl[l3], atom_type.q_rf(idx, l3), 0,
                                                                              atom_type.num_mt_points());
                                }
                            }
                        }
                    }
                }
            }

            #pragma omp parallel for
            for (int l = 0; l <= 2 * lmax_beta; l++) {
                for (int idx = 0; idx < nbrf * (nbrf + 1) / 2; idx++) {
                    values_(idx, l, iat).interpolate();
                }
            }
        }
    }

  public:
    Radial_integrals_aug(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<3>(unit_cell__, qmax__, np__)
    {
        int nmax = unit_cell_.max_mt_radial_basis_size();
        int lmax = unit_cell_.lmax(); 

        values_ = mdarray<Spline<double>, 3>(nmax * (nmax + 1) / 2, 2 * lmax + 1, unit_cell_.num_atom_types());

        generate();
    }

    inline double value(int idx__, int l__, int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(idx__, l__, iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_rho_pseudo: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|rho_pseudo");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            values_(iat) = Spline<double>(grid_q_);

            Spline<double> rho(atom_type.radial_grid());
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                rho[ir] = atom_type.pp_desc().total_charge_density[ir];
            }
            rho.interpolate();

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spherical_Bessel_functions jl(0, atom_type.radial_grid(), grid_q_[iq]);

                values_(iat)[iq] = sirius::inner(jl[0], rho, 0, atom_type.num_mt_points()) / fourpi;
            }
            values_(iat).interpolate();
        }
    }

  public:
    Radial_integrals_rho_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_rho_core_pseudo: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|rho_core_pseudo");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            values_(iat) = Spline<double>(grid_q_);

            Spline<double> ps_core(atom_type.radial_grid());
            for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                ps_core[ir] = atom_type.pp_desc().core_charge_density[ir];
            }
            ps_core.interpolate();

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spherical_Bessel_functions jl(0, atom_type.radial_grid(), grid_q_[iq]);

                values_(iat)[iq] = sirius::inner(jl[0], ps_core, 2, atom_type.num_mt_points());
            }
            values_(iat).interpolate();
        }
    }

  public:
    Radial_integrals_rho_core_pseudo(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

template <bool jl_deriv>
class Radial_integrals_beta: public Radial_integrals_base<2>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|beta");
    
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int nrb = atom_type.mt_radial_basis_size();

            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                values_(idxrf, iat) = Spline<double>(grid_q_);
            }
    
            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spherical_Bessel_functions jl(unit_cell_.lmax(), atom_type.radial_grid(), grid_q_[iq]);
                for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                    int l  = atom_type.indexr(idxrf).l;
                    int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                    /* compute \int j_l(q * r) beta_l(r) r^2 dr or \int d (j_l(q*r) / dq) beta_l(r) r^2  */
                    /* remeber that beta(r) are defined as miltiplied by r */
                    if (jl_deriv) {
                        auto s = jl.deriv_q(l);
                        sirius::inner(s, atom_type.beta_rf(idxrf), 1, nr);
                    } else {
                        values_(idxrf, iat)[iq] = sirius::inner(jl[l], atom_type.beta_rf(idxrf), 1, nr);
                    }
                }
            }

            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                values_(idxrf, iat).interpolate();
            }
        }
    }

  public:
    Radial_integrals_beta(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<2>(unit_cell__, qmax__, np__)
    {
        /* create space for <j_l(qr)|beta> or <d j_l(qr) / dq|beta> radial integrals */
        values_ = mdarray<Spline<double>, 2>(unit_cell_.max_mt_radial_basis_size(), unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int idxrf__, int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(idxrf__, iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_beta_jl: public Radial_integrals_base<3>
{
  private:
    int lmax_;

    void generate()
    {
        PROFILE("sirius::Radial_integrals|beta");
    
        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            int nrb = atom_type.mt_radial_basis_size();

            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                for (int l1 = 0; l1 <= lmax_; l1++) {
                    values_(idxrf, l1, iat) = Spline<double>(grid_q_);
                }
            }

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spherical_Bessel_functions jl(lmax_, atom_type.radial_grid(), grid_q_[iq]);
                for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                    int l  = atom_type.indexr(idxrf).l;
                    int nr = atom_type.pp_desc().num_beta_radial_points[idxrf];
                    for (int l1 = 0; l1 <= lmax_; l1++) {
                        /* compute \int j_{l'}(q * r) beta_l(r) r^2 * r * dr */
                        /* remeber that beta(r) are defined as miltiplied by r */
                        values_(idxrf, l1, iat)[iq] = sirius::inner(jl[l1], atom_type.beta_rf(idxrf), 2, nr);
                    }
                }
            }

            for (int idxrf = 0; idxrf < atom_type.mt_radial_basis_size(); idxrf++) {
                for (int l1 = 0; l1 <= lmax_; l1++) {
                    values_(idxrf, l1, iat).interpolate();
                }
            }
        }
    }

  public:
    Radial_integrals_beta_jl(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<3>(unit_cell__, qmax__, np__)
    {
        lmax_ = unit_cell__.lmax() + 2;
        /* create space for <j_l(qr)|beta> radial integrals */
        values_ = mdarray<Spline<double>, 3>(unit_cell_.max_mt_radial_basis_size(), lmax_ + 1, unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int idxrf__, int l__, int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(idxrf__, l__, iat__)(idx.first, idx.second);
    }
};


class Radial_integrals_theta: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|theta");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            auto R = atom_type.mt_radius();
            values_(iat) = Spline<double>(grid_q_);

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                if (iq == 0) {
                    values_(iat)[iq] = std::pow(R, 3) / 3.0;
                } else {
                    double g = grid_q_[iq];
                    values_(iat)[iq] = (std::sin(g * R) - g * R * std::cos(g * R)) / std::pow(g, 3);
                }
            }
            values_(iat).interpolate();
        }
    }

  public:
    Radial_integrals_theta(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_vloc: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|vloc");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            values_(iat) = Spline<double>(grid_q_);

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                Spline<double> s(atom_type.radial_grid());
                if (iq == 0) {
                    for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                        double x = atom_type.radial_grid(ir);
                        s[ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn()) * x;
                    }
                    values_(iat)[iq] = s.interpolate().integrate(0);
                } else {
                    double g = grid_q_[iq];
                    double g2 = g * g;
                    for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                        double x = atom_type.radial_grid(ir);
                        s[ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn() * gsl_sf_erf(x)) * std::sin(g * x);
                    }
                    values_(iat)[iq] = (s.interpolate().integrate(0) / g - atom_type.zn() * std::exp(-g2 / 4) / g2);
                }
            }
            values_(iat).interpolate();
        }
    }
  
  public:
    Radial_integrals_vloc(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_vloc_dg: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|vloc_dg");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            values_(iat) = Spline<double>(grid_q_);

            #pragma omp parallel for
            for (int iq = 1; iq < grid_q_.num_points(); iq++) {
                Spline<double> s1(atom_type.radial_grid());
                Spline<double> s2(atom_type.radial_grid());
                double g = grid_q_[iq];
                double g2 = g * g;
                
                for (int ir = 0; ir < atom_type.num_mt_points(); ir++) {
                    double x = atom_type.radial_grid(ir);
                    s1[ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn() * gsl_sf_erf(x)) * std::sin(g * x);
                    s2[ir] = (x * atom_type.pp_desc().vloc[ir] + atom_type.zn() * gsl_sf_erf(x)) * std::cos(g * x);
                }
                values_(iat)[iq] = (s1.interpolate().integrate(0) / g - s2.interpolate().integrate(1)) / g2;
                values_(iat)[iq] -= atom_type.zn() * std::exp(-g2 / 4) * (4 + g2) / 2 / g2 / g2;
            }
            /* V(0) is not used; this is done just to make the interpolation easy */
            values_(iat)[0] = values_(iat)[1];
            values_(iat).interpolate();
        }
    }
  
  public:
    Radial_integrals_vloc_dg(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

class Radial_integrals_rho_free_atom: public Radial_integrals_base<1>
{
  private:
    void generate()
    {
        PROFILE("sirius::Radial_integrals|rho_free_atom");

        for (int iat = 0; iat < unit_cell_.num_atom_types(); iat++) {
            auto& atom_type = unit_cell_.atom_type(iat);
            values_(iat) = Spline<double>(grid_q_);

            #pragma omp parallel for
            for (int iq = 0; iq < grid_q_.num_points(); iq++) {
                double g = grid_q_[iq];
                Spline<double> s(unit_cell_.atom_type(iat).free_atom_radial_grid());
                if (iq == 0) {
                    for (int ir = 0; ir < s.num_points(); ir++) {
                        s[ir] = atom_type.free_atom_density(ir);
                    }
                    values_(iat)[iq] = s.interpolate().integrate(2);
                } else {
                    for (int ir = 0; ir < s.num_points(); ir++) {
                        s[ir] = atom_type.free_atom_density(ir) * std::sin(g * atom_type.free_atom_radial_grid(ir)) / g;
                    }
                    values_(iat)[iq] = s.interpolate().integrate(1);
                }
            }
            values_(iat).interpolate();
        }
    }

  public:
    Radial_integrals_rho_free_atom(Unit_cell const& unit_cell__, double qmax__, int np__)
        : Radial_integrals_base<1>(unit_cell__, qmax__, np__)
    {
        values_ = mdarray<Spline<double>, 1>(unit_cell_.num_atom_types());
        generate();
    }

    inline double value(int iat__, double q__) const
    {
        auto idx = iqdq(q__);
        return values_(iat__)(idx.first, idx.second);
    }
};

} // namespace

#endif // __RADIAL_INTEGRALS_H__


