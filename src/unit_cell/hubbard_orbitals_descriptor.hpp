/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file hubbard_orbitals_descriptor.hpp
 *
 *  \brief Contains a descriptor class for Hubbard orbitals.
 */

#ifndef __HUBBARD_ORBITALS_DESCRIPTOR_HPP__
#define __HUBBARD_ORBITALS_DESCRIPTOR_HPP__

#include "core/sht/sht.hpp"

namespace sirius {

/// Structure containing all information about a specific hubbard orbital (including the radial function).
class hubbard_orbital_descriptor
{
  private:
    /// Principal quantum number of atomic orbital.
    int n_{-1};
    /// Orbital quantum number of atomic orbital.
    int l_{-1};
    /// Set to true if this orbital is part the Hubbard subspace.
    bool use_for_calculation_{true};

    /// Orbital occupancy.
    double occupancy_{-1.0};

    Spline<double> f_;

    /// Hubbard U parameter (on-site repulsion).
    double U_{0.0};
    /// Hubbard J parameter (exchange).
    double J_{0.0};

    /// Different hubbard coefficients.
    /** s: U = hubbard_coefficients_[0]
        p: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1]
        d: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1],  B  = hubbard_coefficients_[2]
        f: U = hubbard_coefficients_[0], J = hubbard_coefficients_[1],  E2 = hubbard_coefficients_[2], E3 =
        hubbard_coefficients_[3]
        hubbard_coefficients[4] = U_alpha
        hubbard_coefficients[5] = U_beta */
    std::array<double, 4> hubbard_coefficients_ = {0.0, 0.0, 0.0, 0.0};

    mdarray<double, 4> hubbard_matrix_;

    /* simplifed hubbard theory */
    double alpha_{0.0};

    double beta_{0.0};

    double J0_{0.0};

    std::vector<double> initial_occupancy_;

    /// Index of the corresponding atomic wave-function.
    int idx_wf_{-1}; // TODO: better name

    inline auto
    hubbard_F_coefficients() const
    {
        std::vector<double> F(4);
        F[0] = U();

        switch (this->l()) {
            case 0: {
                F[1] = J();
                break;
            }
            case 1: {
                F[1] = 5.0 * J();
                break;
            }
            case 2: {
                F[1] = 5.0 * J() + 31.5 * B();
                F[2] = 9.0 * J() - 31.5 * B();
                break;
            }
            case 3: {
                F[1] = (225.0 / 54.0) * J() + (32175.0 / 42.0) * E2() + (2475.0 / 42.0) * E3();
                F[2] = 11.0 * J() - (141570.0 / 77.0) * E2() + (4356.0 / 77.0) * E3();
                F[3] = (7361.640 / 594.0) * J() + (36808.20 / 66.0) * E2() - 111.54 * E3();
                break;
            }
            default: {
                std::stringstream s;
                s << "Hubbard correction not implemented for l > 3\n"
                  << "  current l: " << this->l();
                RTE_THROW(s);
                break;
            }
        }
        return F;
    }

    inline void
    calculate_ak_coefficients(mdarray<double, 5>& ak)
    {
        // compute the ak coefficients appearing in the general treatment of
        // hubbard corrections.  expression taken from Liechtenstein {\it et
        // al}, PRB 52, R5467 (1995)

        // Note that for consistency, the ak are calculated with complex
        // harmonics in the gaunt coefficients <R_lm|Y_l'm'|R_l''m''>.
        // we need to keep it that way because of the hubbard potential.
        // With a spherical one it does not really matter-
        ak.zero();

        int l = this->l();

        for (int m1 = -l; m1 <= l; m1++) {
            for (int m2 = -l; m2 <= l; m2++) {
                for (int m3 = -l; m3 <= l; m3++) {
                    for (int m4 = -l; m4 <= l; m4++) {
                        for (int k = 0; k < 2 * l; k += 2) {
                            double sum = 0.0;
                            for (int q = -k; q <= k; q++) {
                                sum += SHT::gaunt_rlm_ylm_rlm(l, k, l, m1, q, m2) *
                                       SHT::gaunt_rlm_ylm_rlm(l, k, l, m3, q, m4);
                            }
                            /* according to PRB 52, R5467 it is 4 \pi/(2 k + 1) -> 4 \pi / (4 * k + 1) because
                               only a_{k=0} a_{k=2}, a_{k=4} are considered */
                            ak(k / 2, m1 + l, m2 + l, m3 + l, m4 + l) = 4.0 * sum * pi / static_cast<double>(2 * k + 1);
                        }
                    }
                }
            }
        }
    }

    /// this function computes the matrix elements of the orbital part of
    /// the electron-electron interactions. we effectively compute

    /// \f[ u(m,m'',m',m''') = \left<m,m''|V_{e-e}|m',m'''\right> \sum_k
    /// a_k(m,m',m'',m''') F_k \f] where the F_k are calculated for real
    /// spherical harmonics

    inline void
    compute_hubbard_matrix()
    {
        int l                 = this->l();
        this->hubbard_matrix_ = mdarray<double, 4>({2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1});
        mdarray<double, 5> ak({l, 2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1});
        auto F = hubbard_F_coefficients();
        calculate_ak_coefficients(ak);

        // the indices are rotated around

        // <m, m |vee| m'', m'''> = hubbard_matrix(m, m'', m', m''')
        this->hubbard_matrix_.zero();
        for (int m1 = 0; m1 < 2 * l + 1; m1++) {
            for (int m2 = 0; m2 < 2 * l + 1; m2++) {
                for (int m3 = 0; m3 < 2 * l + 1; m3++) {
                    for (int m4 = 0; m4 < 2 * l + 1; m4++) {
                        for (int k = 0; k < l; k++) {
                            this->hubbard_matrix(m1, m2, m3, m4) += ak(k, m1, m3, m2, m4) * F[k];
                        }
                    }
                }
            }
        }
    }

    void
    initialize_hubbard_matrix()
    {
        int l = this->l();
        mdarray<double, 5> ak({l, 2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1});
        auto F = hubbard_F_coefficients();
        calculate_ak_coefficients(ak);

        this->hubbard_matrix_ = mdarray<double, 4>({2 * l + 1, 2 * l + 1, 2 * l + 1, 2 * l + 1});
        // the indices are rotated around

        // <m, m |vee| m'', m'''> = hubbard_matrix(m, m'', m', m''')
        this->hubbard_matrix_.zero();
        for (int m1 = 0; m1 < 2 * l + 1; m1++) {
            for (int m2 = 0; m2 < 2 * l + 1; m2++) {
                for (int m3 = 0; m3 < 2 * l + 1; m3++) {
                    for (int m4 = 0; m4 < 2 * l + 1; m4++) {
                        for (int k = 0; k < l; k++) {
                            this->hubbard_matrix(m1, m2, m3, m4) += ak(k, m1, m3, m2, m4) * F[k];
                        }
                    }
                }
            }
        }
    }

  public:
    /// Constructor.
    hubbard_orbital_descriptor()
    {
    }

    /// Constructor.
    hubbard_orbital_descriptor(const int n__, const int l__, const int orbital_index__, const double occ__,
                               const double J__, const double U__, const double* hub_coef__, const double alpha__,
                               const double beta__, const double J0__, std::vector<double> initial_occupancy__,
                               Spline<double> f__, bool use_for_calculations__, int idx_wf__)
        : n_(n__)
        , l_(l__)
        , use_for_calculation_(use_for_calculations__)
        , occupancy_(occ__)
        , f_(std::move(f__))
        , U_(U__)
        , J_(J__)
        , alpha_(alpha__)
        , beta_(beta__)
        , J0_(J0__)
        , initial_occupancy_(initial_occupancy__)
        , idx_wf_(idx_wf__)
    {
        if (hub_coef__) {
            for (int s = 0; s < 4; s++) {
                hubbard_coefficients_[s] = hub_coef__[s];
            }

            initialize_hubbard_matrix();
        }
    }

    ~hubbard_orbital_descriptor()
    {
    }

    /// Move constructor
    hubbard_orbital_descriptor(hubbard_orbital_descriptor&& src)
        : n_(src.n_)
        , l_(src.l_)
        , use_for_calculation_(src.use_for_calculation_)
        , occupancy_(src.occupancy_)
        , U_(src.U_)
        , J_(src.J_)
        , alpha_(src.alpha_)
        , beta_(src.beta_)
        , J0_(src.J0_)
        , initial_occupancy_(src.initial_occupancy_)
        , idx_wf_(src.idx_wf_)
    {
        hubbard_matrix_ = std::move(src.hubbard_matrix_);
        for (int s = 0; s < 4; s++) {
            hubbard_coefficients_[s] = src.hubbard_coefficients_[s];
        }
        f_ = std::move(src.f_);
    }

    inline int
    n() const
    {
        return n_;
    }

    inline int
    l() const
    {
        return l_;
    }

    inline double
    hubbard_matrix(const int m1, const int m2, const int m3, const int m4) const
    {
        return hubbard_matrix_(m1, m2, m3, m4);
    }

    inline double&
    hubbard_matrix(const int m1, const int m2, const int m3, const int m4)
    {
        return hubbard_matrix_(m1, m2, m3, m4);
    }

    inline double
    J0() const
    {
        return J0_;
    }

    inline double
    U() const
    {
        return U_;
    }

    inline double
    J() const
    {
        return J_;
    }

    inline double
    U_minus_J() const
    {
        return this->U() - this->J();
    }

    inline double
    B() const
    {
        return hubbard_coefficients_[2];
    }

    inline double
    E2() const
    {
        return hubbard_coefficients_[2];
    }

    inline double
    E3() const
    {
        return hubbard_coefficients_[3];
    }

    inline double
    alpha() const
    {
        return alpha_;
    }

    inline double
    beta() const
    {
        return beta_;
    }

    inline double
    occupancy() const
    {
        return occupancy_;
    }

    Spline<double> const&
    f() const
    {
        return f_;
    }

    bool
    use_for_calculation() const
    {
        return use_for_calculation_;
    }

    auto const&
    initial_occupancy() const
    {
        return initial_occupancy_;
    }

    auto
    idx_wf() const
    {
        return idx_wf_;
    }
};

inline std::ostream&
operator<<(std::ostream& out, hubbard_orbital_descriptor const& ho)
{
    out << "{n: " << ho.n() << ", l: " << ho.l() << "}";
    return out;
}

} // namespace sirius

#endif
