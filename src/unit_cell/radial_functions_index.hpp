/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file radial_functions_index.hpp
 *
 *  \brief Contains definition and implementation of sirius::radial_functions_index class.
 */

#ifndef __RADIAL_FUNCTIONS_INDEX_HPP__
#define __RADIAL_FUNCTIONS_INDEX_HPP__

#include "core/rte/rte.hpp"
#include "core/strong_type.hpp"

namespace sirius {

/// Radial function index.
using rf_index = strong_type<int, struct __rf_index_tag>;
/// Augmented wave radial function index.
using rf_aw_index = strong_type<int, struct __rf_aw_index_tag>;
/// Local orbital radial function index.
using rf_lo_index = strong_type<int, struct __rf_lo_index_tag>;
/// Basis function index.
using bf_index = strong_type<int, struct __bf_index_tag>;
/// Augmented wave basis function index.
using bf_aw_index = strong_type<int, struct __bf_aw_index_tag>;
/// Local orbital basis function index.
using bf_lo_index = strong_type<int, struct __bf_lo_index_tag>;

/// Angular momentum quantum number.
/** This class handles orbital or total angluar momentum quantum number. */
class angular_momentum
{
  private:
    /// Orbital quantum number l.
    int l_;

    /// Spin quantum number in the units of 1/2.
    /** This variable can have only three values: -1,0,1. It is used to consruct the total angular
     *  momentum j = l + s/2. In case s = 0 total agular momentum j = l (no level splitting). */
    int s_{0};

  public:
    /// Constructor.
    explicit angular_momentum(int l__)
        : l_(l__)
    {
        if (l__ < 0) {
            RTE_THROW("l can't be negative");
        }
    }

    /// Constructor.
    explicit angular_momentum(int l__, int s__)
        : l_(l__)
        , s_(s__)
    {
        if (l__ < 0) {
            RTE_THROW("l can't be negative");
        }
        if (s__ != -1 && s__ != 0 && s__ != 1) {
            RTE_THROW("wrong value of s");
        }
        if (l__ == 0 && s__ == -1) {
            RTE_THROW("incompatible combination of l and s quantum numbers");
        }
    }

    /// Get orbital quantum number l.
    inline auto
    l() const
    {
        return l_;
    }

    /// Get total angular momentum j = l +/- 1/2
    inline auto
    j() const
    {
        return l_ + s_ / 2.0;
    }

    /// Get twice the total angular momentum 2j = 2l +/- 1
    inline auto
    two_j() const
    {
        return 2 * l_ + s_;
    }

    /// The size of the subshell for the angular momentum l or j.
    /** This is the number of m_l values in the range [-l, l] or the number of
     *  m_j values in the range [-j, j] */
    inline auto
    subshell_size() const
    {
        return two_j() + 1;
    }

    /// Get spin quantum number s.
    inline auto
    s() const
    {
        return s_;
    }
};

inline bool
operator==(angular_momentum lhs__, angular_momentum rhs__)
{
    return (lhs__.l() == rhs__.l()) && (lhs__.s() == rhs__.s());
}

inline bool
operator!=(angular_momentum lhs__, angular_momentum rhs__)
{
    return !(lhs__ == rhs__);
}

/// Output angular momentum to a stream.
inline std::ostream&
operator<<(std::ostream& out, angular_momentum am)
{
    if (am.s() == 0) {
        out << "{l: " << am.l() << "}";
    } else {
        out << "{l: " << am.l() << ", j: " << am.j() << "}";
    }
    return out;
}

/// Descriptor for the atomic radial functions.
/** The radial functions \f$ f_{\ell \nu}(r) \f$ are labeled by two indices: orbital quantum number \f$ \ell \f$ and
 *  an order \f$ \nu \f$ for a given \f$ \ell \f$. Radial functions can be any of augmented waves or local orbitals
 *  (in case of FP-LAPW) or beta projectors, atomic or Hubbard wave functions in case of PP-PW.
 */
struct radial_function_index_descriptor
{
    /// Total angular momentum
    angular_momentum am;

    /// Order of a function for a given \f$ \ell \f$.
    int order;

    /// If this is a local orbital radial function, idxlo is it's index in the list of local orbital descriptors.
    rf_lo_index idxlo;

    rf_index idxrf{-1};

    radial_function_index_descriptor(angular_momentum am__, int order__, rf_index idxrf__,
                                     rf_lo_index idxlo__ = rf_lo_index(-1))
        : am{am__}
        , order{order__}
        , idxlo{idxlo__}
        , idxrf{idxrf__}
    {
        RTE_ASSERT(order >= 0);
    }
};

/// Radial basis function index.
/** Radial functions can have a repeating orbital quantum number, for example {2s, 2s, 3p, 3p, 4d} configuration
 *  corresponds to {l=0, l=0, l=1, l=1, l=2} radial functions index. */
class radial_functions_index
{
  private:
    /// Store index of the radial function by angular momentum j and order of the function for a given j. */
    std::vector<std::vector<std::array<int, 2>>> index_by_j_order_;

    /// List of radial function index descriptors.
    /** This list establishes a mapping \f$ f_{\mu}(r) \leftrightarrow  f_{j \nu}(r) \f$ between a
     *  composite index \f$ \mu \f$ of radial functions and
     *  corresponding \f$ j \nu \f$ indices, where \f$ j \f$ is the total orbital quantum number and
     *  \f$ \nu \f$ is the order of radial function for a given \f$ j \f$. */
    std::vector<radial_function_index_descriptor> vrd_;

    /// Starting index of local orbitals (if added in LAPW case).
    int offset_lo_{-1};

  public:
    /// Default constructor.
    radial_functions_index()
    {
    }

    /// Add a single radial function with a given angular momentum.
    void
    add(angular_momentum am__)
    {
        /* current l */
        auto l = am__.l();
        /* current s */
        auto s = am__.s();

        if (s != 0 && l > 0) {
            RTE_THROW("for l > 0 full-j radial functions are added in pairs");
        }

        /* make sure that the space is available */
        if (static_cast<int>(index_by_j_order_.size()) < l + 1) {
            index_by_j_order_.resize(l + 1);
        }

        /* size of array is equal to current index */
        auto size = this->size();

        std::array<int, 2> idx({-1, -1});
        /* std::max(s, 0) maps s = -1 -> 0, s = 0 -> 0, s = 1 -> 1 */
        idx[std::max(s, 0)] = size;
        /* current order */
        auto o = static_cast<int>(index_by_j_order_[l].size());
        /* for the reverse mapping */
        index_by_j_order_[l].push_back(idx);
        /* add descriptor to the list */
        vrd_.push_back(radial_function_index_descriptor(am__, o, rf_index(size)));
    }

    /// Add local-orbital type of radial function.
    /** Local orbitals are only used in FP-LAPW, where the distinction between APW and local orbitals
     *  must be made. For PP-PW this is not used. */
    void
    add_lo(angular_momentum am__)
    {
        /* mark the start of the local orbital block of radial functions */
        if (offset_lo_ < 0) {
            offset_lo_ = this->size();
        }
        /* add current index of radial function for reverese mapping from local orbital index */
        this->add(am__);
        /* set index of the local orbital */
        vrd_.back().idxlo = rf_lo_index(this->size() - offset_lo_ - 1);
    }

    /// Add two component of the spinor radial function.
    void
    add(angular_momentum am1__, angular_momentum am2__)
    {
        /* current l */
        auto l = am1__.l();
        /* check second l */
        if (l != am2__.l()) {
            RTE_THROW("orbital quantum numbers are different");
        }

        /* current s */
        auto s1 = am1__.s();
        /* current s */
        auto s2 = am2__.s();

        if (s1 == s2) {
            RTE_THROW("spin quantum numbers are the same");
        }

        if (s1 * s2 == 0) {
            RTE_THROW("spin quantum numbers can't be zero in case of full orbital momentum");
        }

        /* make sure that the space is available */
        if (static_cast<int>(index_by_j_order_.size()) < l + 1) {
            index_by_j_order_.resize(l + 1);
        }

        /* for the total angular momantum, the radial functions are stored in pairs and
         * have a contiguous index from s=-1 to s=1, for example:
         *                        | l = 0 | l = 1 |
         *   +-----------+--------+-------+-------+---
         *   |           | s = -1 |  n/a  | idx=1 |
         *   | order = 0 +--------+-------+-------+---
         *   |           ! s =  1 | idx=0 | idx=2 |
         *   +-----------+--------+-------+-------+---
         *   |           | s = -1 |  n/a  | idx=4 |
         *   | order = 1 +--------+-------+-------+---
         *   |           ! s =  1 | idx=3 | idx=5 |
         *   +-----------+--------+-------+-------+---
         *   |........................................
         *   |
         */

        /* current order */
        auto o = static_cast<int>(index_by_j_order_[l].size());

        auto size = this->size();

        std::array<int, 2> idx({-1, -1});
        /* std::max(s, 0) maps s = -1 -> 0, s = 0 -> 0, s = 1 -> 1 */
        idx[std::max(s1, 0)] = size;
        idx[std::max(s2, 0)] = size + 1;

        vrd_.push_back(radial_function_index_descriptor(am1__, o, rf_index(size)));
        vrd_.push_back(radial_function_index_descriptor(am2__, o, rf_index(size + 1)));

        index_by_j_order_[l].push_back(idx);
    }

    /// Return angular momentum of the radial function.
    inline auto
    am(rf_index idx__) const
    {
        return vrd_[idx__].am;
    }

    /// Return order of the radial function.
    inline auto
    order(rf_index idx__) const
    {
        return vrd_[idx__].order;
    }

    /// Return maximum angular momentum quantum number.
    inline auto
    lmax() const
    {
        return static_cast<int>(index_by_j_order_.size()) - 1;
    }

    /// Maximum angular momentum quantum number for local orbitals.
    inline auto
    lmax_lo() const
    {
        int result{-1};
        if (offset_lo_ >= 0) {
            for (int i = offset_lo_; i < this->size(); i++) {
                result = std::max(result, vrd_[i].am.l());
            }
        }
        return result;
    }

    /// Number of local orbitals for a given l.
    inline auto
    num_lo(int l__) const
    {
        int result{-1};
        if (offset_lo_ >= 0) {
            for (int i = offset_lo_; i < this->size(); i++) {
                if (vrd_[i].am.l() == l__) {
                    result++;
                }
            }
        }
        return result;
    }

    /// Return maximum order of the radial functions for a given angular momentum.
    inline auto
    max_order(int l__) const
    {
        return static_cast<int>(index_by_j_order_[l__].size());
    }

    /// Return maximum order of the radial functions across all angular momentums.
    inline auto
    max_order() const
    {
        int result{0};
        for (int l = 0; l <= this->lmax(); l++) {
            result = std::max(result, this->max_order(l));
        }
        return result;
    }

    /// Return index of radial function.
    inline auto
    index_of(angular_momentum am__, int order__) const
    {
        /* std::max(s, 0) maps s = -1 -> 0, s = 0 -> 0, s = 1 -> 1 */
        return rf_index(index_by_j_order_[am__.l()][order__][std::max(am__.s(), 0)]);
    }

    /// Return index of local orbital.
    inline auto
    index_of(rf_lo_index idxlo__) const
    {
        RTE_ASSERT(idxlo__ >= 0 && idxlo__ + offset_lo_ < this->size());
        return rf_index(offset_lo_ + idxlo__);
    }

    /// Check if the angular momentum is treated as full (j = l +/- 1/2).
    auto
    full_j(int l__, int o__) const
    {
        /* look at index of l + s */
        if (index_by_j_order_[l__][o__][1] >= 0) {
            return true;
        } else {
            return false;
        }
    }

    /// Return the angular mementum(s) of the subshell with given l and order.
    auto
    subshell(int l__, int o__) const
    {
        if (full_j(l__, o__)) {
            if (l__ == 0) {
                return std::vector<angular_momentum>({angular_momentum(l__, 1)});
            } else {
                return std::vector<angular_momentum>({angular_momentum(l__, -1), angular_momentum(l__, 1)});
            }
        } else {
            return std::vector<angular_momentum>({angular_momentum(l__)});
        }
    }

    /// Return total number of radial functions.
    inline int
    size() const
    {
        return static_cast<int>(vrd_.size());
    }

    /// Return the subshell size for a given l and order.
    /** In case of orbital quantum number l the size is 2l+1; in case of full
     *  angular momentum j the size is 2*(2l+1) and consists of 2j_{+} + 1 and 2j_{-} + 1
     *  contributions */
    inline auto
    subshell_size(int l__, int o__) const
    {
        int size{0};
        for (auto j : subshell(l__, o__)) {
            size += j.subshell_size();
        }
        return size;
    }

    /// Return radial function descriptor for a given index.
    inline auto const&
    operator[](rf_index i__) const
    {
        return vrd_[i__];
    }

    /// Begin iterator of radial function descriptor list.
    auto
    begin() const
    {
        return vrd_.begin();
    }

    /// End iterator of radial function descriptor list.
    auto
    end() const
    {
        return vrd_.end();
    }
};

inline auto
begin(radial_functions_index const& idx__)
{
    return idx__.begin();
}

inline auto
end(radial_functions_index const& idx__)
{
    return idx__.end();
}

} // namespace sirius

#endif
