/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file gaunt.hpp
 *
 *  \brief Contains definition and implementation of sirius::Gaunt class.
 */

#ifndef __GAUNT_HPP__
#define __GAUNT_HPP__

#include "core/memory.hpp"
#include "core/typedefs.hpp"
#include "core/sf/specfunc.hpp"

namespace sirius {

/// Used in the {lm1, lm2} : {lm3, coefficient} way of grouping non-zero Gaunt coefficients
template <typename T>
struct gaunt_L3
{
    int lm3;
    int l3;
    T coef;
};

/// Used in the {lm1, lm2, coefficient} : {lm3} way of grouping non-zero Gaunt coefficients
template <typename T>
struct gaunt_L1_L2
{
    int lm1;
    int lm2;
    T coef;
};

/// Compact storage of non-zero Gaunt coefficients \f$ \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle \f$.
/** Very important! The following notation is adopted and used everywhere: lm1 and lm2 represent 'bra' and 'ket'
 *  spherical harmonics of the Gaunt integral and lm3 represent the inner spherical harmonic.
 */
template <typename T>
class Gaunt_coefficients
{
  private:
    /// lmax of <lm1|
    int lmax1_;
    /// lmmax of <lm1|
    int lmmax1_;

    /// lmax of inner real or complex spherical harmonic
    int lmax3_;
    /// lmmax of inner real or complex spherical harmonic
    int lmmax3_;

    /// lmax of |lm2>
    int lmax2_;
    /// lmmax of |lm2>
    int lmmax2_;

    /// List of non-zero Gaunt coefficients for each lm3.
    mdarray<std::vector<gaunt_L1_L2<T>>, 1> gaunt_packed_L1_L2_;

    /// List of non-zero Gaunt coefficients for each combination of lm1, lm2.
    mdarray<std::vector<gaunt_L3<T>>, 2> gaunt_packed_L3_;

  public:
    /// Class constructor.
    Gaunt_coefficients(int lmax1__, int lmax3__, int lmax2__, std::function<T(int, int, int, int, int, int)> get__)
        : lmax1_(lmax1__)
        , lmax3_(lmax3__)
        , lmax2_(lmax2__)
    {
        lmmax1_ = sf::lmmax(lmax1_);
        lmmax3_ = sf::lmmax(lmax3_);
        lmmax2_ = sf::lmmax(lmax2_);

        gaunt_packed_L1_L2_ = mdarray<std::vector<gaunt_L1_L2<T>>, 1>({lmmax3_});
        gaunt_L1_L2<T> g12;

        gaunt_packed_L3_ = mdarray<std::vector<gaunt_L3<T>>, 2>({lmmax1_, lmmax2_});
        gaunt_L3<T> g3;

        for (int l1 = 0, lm1 = 0; l1 <= lmax1_; l1++) {
            for (int m1 = -l1; m1 <= l1; m1++, lm1++) {
                for (int l2 = 0, lm2 = 0; l2 <= lmax2_; l2++) {
                    for (int m2 = -l2; m2 <= l2; m2++, lm2++) {
                        for (int l3 = 0, lm3 = 0; l3 <= lmax3_; l3++) {
                            for (int m3 = -l3; m3 <= l3; m3++, lm3++) {

                                T gc = get__(l1, l3, l2, m1, m3, m2);
                                if (std::abs(gc) > 1e-12) {
                                    g12.lm1  = lm1;
                                    g12.lm2  = lm2;
                                    g12.coef = gc;
                                    gaunt_packed_L1_L2_[lm3].push_back(g12);

                                    g3.lm3  = lm3;
                                    g3.l3   = l3;
                                    g3.coef = gc;
                                    gaunt_packed_L3_(lm1, lm2).push_back(g3);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Return number of non-zero Gaunt coefficients for a given lm3.
    inline int
    num_gaunt(int lm3) const
    {
        assert(lm3 >= 0 && lm3 < lmmax3_);
        return static_cast<int>(gaunt_packed_L1_L2_[lm3].size());
    }

    /// Return a structure containing {lm1, lm2, coef} for a given lm3 and index.
    /** Example:
     *  \code{.cpp}
     *  for (int lm3 = 0; lm3 < lmmax3; lm3++)
     *  {
     *      for (int i = 0; i < gaunt_coefs.num_gaunt(lm3); i++) {
     *          int lm1 = gaunt_coefs.gaunt(lm3, i).lm1;
     *          int lm2 = gaunt_coefs.gaunt(lm3, i).lm2;
     *          double coef = gaunt_coefs.gaunt(lm3, i).coef;
     *
     *          // do something with lm1,lm2,lm3 and coef
     *      }
     *  }
     *  \endcode
     */
    inline gaunt_L1_L2<T> const&
    gaunt(int lm3, int idx) const
    {
        assert(lm3 >= 0 && lm3 < lmmax3_);
        assert(idx >= 0 && idx < (int)gaunt_packed_L1_L2_[lm3].size());
        return gaunt_packed_L1_L2_[lm3][idx];
    }

    /// Return number of non-zero Gaunt coefficients for a combination of lm1 and lm2.
    inline int
    num_gaunt(int lm1, int lm2) const
    {
        return static_cast<int>(gaunt_packed_L3_(lm1, lm2).size());
    }

    /// Return a structure containing {lm3, coef} for a given lm1, lm2 and index
    inline gaunt_L3<T> const&
    gaunt(int lm1, int lm2, int idx) const
    {
        return gaunt_packed_L3_(lm1, lm2)[idx];
    }

    /// Return a sum over L3 (lm3) index of Gaunt coefficients and a complex vector.
    /** The following operation is performed:
     *  \f[
     *      \sum_{\ell_3 m_3} \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle v_{\ell_3 m_3}
     *  \f]
     *  Result is assumed to be complex.
     */
    inline auto
    sum_L3_gaunt(int lm1, int lm2, std::complex<double> const* v) const
    {
        std::complex<double> zsum(0, 0);
        for (int k = 0; k < (int)gaunt_packed_L3_(lm1, lm2).size(); k++) {
            zsum += gaunt_packed_L3_(lm1, lm2)[k].coef * v[gaunt_packed_L3_(lm1, lm2)[k].lm3];
        }
        return zsum;
    }

    /// Return a sum over L3 (lm3) index of Gaunt coefficients and a real vector.
    /** The following operation is performed:
     *  \f[
     *      \sum_{\ell_3 m_3} \langle \ell_1 m_1 | \ell_3 m_3 | \ell_2 m_2 \rangle v_{\ell_3 m_3}
     *  \f]
     *  Result is assumed to be of the same type as Gaunt coefficients.
     */
    inline T
    sum_L3_gaunt(int lm1, int lm2, double const* v) const
    {
        T sum = 0;
        for (int k = 0; k < (int)gaunt_packed_L3_(lm1, lm2).size(); k++) {
            sum += gaunt_packed_L3_(lm1, lm2)[k].coef * v[gaunt_packed_L3_(lm1, lm2)[k].lm3];
        }
        return sum;
    }

    /// Return vector of non-zero Gaunt coefficients for a given combination of lm1 and lm2
    inline std::vector<gaunt_L3<T>> const&
    gaunt_vector(int lm1, int lm2) const
    {
        return gaunt_packed_L3_(lm1, lm2);
    }

    /// Return the full tensor of Gaunt coefficients <R_{L1}|R_{L3}|R_{L2}> with a (L3, L1, L2) order of indices.
    inline auto
    get_full_set_L3() const
    {
        mdarray<T, 3> gc(lmmax3_, lmmax1_, lmmax2_);
        gc.zero();
        for (int lm2 = 0; lm2 < lmmax2_; lm2++) {
            for (int lm1 = 0; lm1 < lmmax1_; lm1++) {
                for (int k = 0; k < (int)gaunt_packed_L3_(lm1, lm2).size(); k++) {
                    int lm3           = gaunt_packed_L3_(lm1, lm2)[k].lm3;
                    gc(lm3, lm1, lm2) = gaunt_packed_L3_(lm1, lm2)[k].coef;
                }
            }
        }
        return gc;
    }
};

}; // namespace sirius

#endif
