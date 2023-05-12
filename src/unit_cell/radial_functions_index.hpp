// Copyright (c) 2013-2018 Anton Kozhevnikov, Thomas Schulthess
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

/** \file radial_functions_index.hpp
 *
 *  \brief Contains definition and implementation of sirius::radial_functions_index class.
 */

#ifndef __RADIAL_FUNCTIONS_INDEX_HPP__
#define __RADIAL_FUNCTIONS_INDEX_HPP__

#include "utils/rte.hpp"
#include "strong_type.hpp"

namespace sirius {

/// Radial function index.
using rf_index = strong_type<int, struct __rf_index_tag>;
/// Augmented wave radial function index.
using rf_aw_index = strong_type<int, struct __rf_aw_index_tag>;
/// Local-orbital radial function index.
using rf_lo_index = strong_type<int, struct __rf_lo_index_tag>;
/// Basis function index.
using bf_index = strong_type<int, struct __bf_index_tag>;
using bf_aw_index = strong_type<int, struct __bf_aw_index_tag>;
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
    inline auto l() const
    {
        return l_;
    }

    /// Get total angular momentum j = l +/- 1/2
    inline auto j() const
    {
        return l_ + s_ / 2.0;
    }

    /// Get twice the total angular momentum 2j = 2l +/- 1
    inline auto two_j() const
    {
        return 2 * l_ + s_;
    }

    /// The size of the subshell for the angular momentum l or j.
    /** This is the number of m_l values in the range [-l, l] or the number of
     *  m_j values in the range [-j, j] */
    inline auto subshell_size() const
    {
        return two_j() + 1;
    }

    /// Get spin quantum number s.
    inline auto s() const
    {
        return s_;
    }
};

inline bool operator==(angular_momentum lhs__, angular_momentum rhs__)
{
    return (lhs__.l() == rhs__.l()) && (lhs__.s() == rhs__.s());
}

inline bool operator!=(angular_momentum lhs__, angular_momentum rhs__)
{
    return !(lhs__ == rhs__);
}

/// Output angular momentum to a stream.
inline std::ostream& operator<<(std::ostream& out, angular_momentum am)
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
    /// Orbital quantum number \f$ \ell \f$.
    int l;

    /// Total angular momentum
    double j;

    /// Order of a function for a given \f$ \ell \f$.
    int order;

    /// If this is a local orbital radial function, idxlo is it's index in the list of local orbital descriptors.
    int idxlo;

    rf_index idxrf{-1};

    angular_momentum am;

    /// Constructor.
    radial_function_index_descriptor(int l, int order, int idxlo = -1)
        : l{l}
        , order{order}
        , idxlo{idxlo}
        , am(l)
    {
        RTE_ASSERT(l >= 0);
        RTE_ASSERT(order >= 0);
    }

    radial_function_index_descriptor(int l, double j, int order, int idxlo = -1)
        : l{l}
        , j{j}
        , order{order}
        , idxlo{idxlo}
        , am(l)
    {
        RTE_ASSERT(l >= 0);
        RTE_ASSERT(order >= 0);
    }

    radial_function_index_descriptor(angular_momentum am__, int order__, rf_index idxrf__, int idxlo__ = -1)
        : l(am__.l())
        , j(am__.j())
        , order{order__}
        , idxlo{idxlo__}
        , idxrf{idxrf__}
        , am{am__}
    {
        RTE_ASSERT(order >= 0);
    }
};


namespace experimental {

/// Radial basis function index.
/** Radial functions can have a repeating orbital quantum number, for example {2s, 2s, 3p, 3p, 4d} configuration
 *  corresponds to {l=0, l=0, l=1, l=1, l=2} radial functions index. */
class radial_functions_index
{
  private:
    /// Store index of the radial function by angular momentum j and order of the function for a given j. */
    std::vector<std::vector<std::array<int, 2>>> index_by_j_order_;

    /// Vector of radial function index descriptors.
    std::vector<radial_function_index_descriptor> vrd_;

    int offset_lo_{-1};
  public:
    /// Default constructor.
    radial_functions_index()
    {
    }

    /// Add a single radial function with a given angular momentum.
    void add(angular_momentum am__)
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
    void add_lo(angular_momentum am__)
    {
        /* mark the start of the local orbital block of radial functions */
        if (offset_lo_ < 0) {
            offset_lo_ = this->size();
        }
        /* add current index of radial function for reverese mapping from local orbital index */
        this->add(am__);
        /* set index of the local orbital */
        vrd_.back().idxlo = this->size() - offset_lo_ - 1;
    }

    /// Add two component of the spinor radial function.
    void add(angular_momentum am1__, angular_momentum am2__)
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
    inline auto am(rf_index idx__) const
    {
        return vrd_[idx__].am;
    }

    /// Return order of the radial function.
    inline auto order(rf_index idx__) const
    {
        return vrd_[idx__].order;
    }

    /// Return maximum angular momentum quantum number.
    inline auto lmax() const
    {
        return static_cast<int>(index_by_j_order_.size()) - 1;
    }

    /// Return maximum order of the radial functions for a given angular momentum.
    inline auto max_order(int l__) const
    {
        return static_cast<int>(index_by_j_order_[l__].size());
    }

    /// Return maximum order of the radial functions across all angular momentums.
    inline auto max_order() const
    {
        int result{0};
        for (int l = 0; l <= this->lmax(); l++) {
            result = std::max(result, this->max_order(l));
        }
        return result;
    }

    /// Return index of radial function.
    inline auto index_of(angular_momentum am__, int order__) const
    {
        /* std::max(s, 0) maps s = -1 -> 0, s = 0 -> 0, s = 1 -> 1 */
        return rf_index(index_by_j_order_[am__.l()][order__][std::max(am__.s(), 0)]);
    }

    /// Return index of local orbital.
    inline auto index_of(rf_lo_index idxlo__) const
    {
        RTE_ASSERT(idxlo__ >= 0 && idxlo__ + offset_lo_ < this->size());
        return rf_index(offset_lo_ + idxlo__);
    }

    auto full_j(int l__, int o__) const
    {
        /* look at index of l + s */
        if (index_by_j_order_[l__][o__][1] >= 0) {
            return true;
        } else {
            return false;
        }
    }

    auto subshell(int l__, int o__) const
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

    inline int size() const
    {
        return static_cast<int>(vrd_.size());
    }

    inline auto subshell_size(int l__, int o__) const
    {
        int size{0};
        for (auto j: subshell(l__, o__)) {
            size += j.subshell_size();
        }
        return size;
    }

    inline auto const& operator[](rf_index i__) const
    {
        return vrd_[i__];
    }

    auto begin() const
    {
        return vrd_.begin();
    }

    auto end() const
    {
        return vrd_.end();
    }
};

inline auto begin(radial_functions_index const& idx__)
{
    return idx__.begin();
}

inline auto end(radial_functions_index const& idx__)
{
    return idx__.end();
}

class basis_functions_index
{
  private:
    radial_functions_index indexr_;

    int size_{0};

    int offset_lo_{-1};

    sddk::mdarray<int, 2> index_by_lm_order_;
//
//    //mdarray<int, 1> index_by_idxrf_; // TODO: rename to first_lm_index_by_idxrf_ or similar
//
//    ///// Number of augmented wave basis functions.
//    //int size_aw_{0};
//
//    ///// Number of local orbital basis functions.
//    //int size_lo_{0};
//
//    ///// Maximum l of the radial basis functions.
//    //int lmax_{-1};
//    //
//    std::vector<std::vector<int>> index_by_lm_order_; // TODO: subject to change to index_by_lm_idxrf
//
    std::vector<rf_index> idxrf_;
    std::vector<int> lm_;
    /// Offset in the basis function index for the given radial function index.
    std::vector<int> offset_;
  public:
    basis_functions_index()
    {
    }
    basis_functions_index(radial_functions_index const& indexr__, bool expand_full_j__)
        : indexr_(indexr__)
    {
        if (expand_full_j__) {
            RTE_THROW("j,mj expansion of the full angular momentum index is not implemented");
        }

        if (!expand_full_j__) {
            index_by_lm_order_ = sddk::mdarray<int, 2>(utils::lmmax(indexr_.lmax()), indexr_.max_order());
            std::fill(index_by_lm_order_.begin(), index_by_lm_order_.end(), -1);
            //for (auto idxrf = indexr_.begin(); idxrf != indexr_.end(); idxrf++) {
            for (auto e : indexr_) {
                if (e.idxrf == indexr_.index_of(rf_lo_index(0))) {
                    offset_lo_ = size_;
                }
                /* angular momentum */
                auto am = e.am;

                /* order of radial function */
                auto o = e.order;

                offset_.push_back(size_);

                for (int m = -am.l(); m <= am.l(); m++) {
                    idxrf_.push_back(e.idxrf);
                    auto lm = utils::lm(am.l(), m);
                    lm_.push_back(lm);
                    /* reverse mapping */
                    index_by_lm_order_(lm, o) = size_;
                    size_++;

    
//                    basis_function_index_descriptor b;
//                    b.idxrf = idxrf;
//                    b.m = m;
//                    b.lm = utils::lm(l, m);
//                    if (static_cast<int>(index_by_lm_order_[b.lm].size()) < indexr_.max_order(l)) {
//                        index_by_lm_order_[b.lm].resize(indexr_.max_order(l));
//                    }
//                    int idx = static_cast<int>(this->size());
//                    this->push_back(b);
//                    index_by_lm_order_[b.lm][o] = idx;
                }
            }
            //for (int l = 0; l <= indexr_.lmax(); l++) {
            //    for (int o = 0; o < indexr_.max_order(l); o++) {
            //        for (auto s: indexr_.spins(l, o)) {
            //            int idxrf = indexr_.index_of(angular_momentum(l, s), o);
            //            for (int m = -l; m <= l; m++) {
            //            }
            //        }
            //    }
            //}
        }
    }

    inline auto size() const
    {
        return size_;
    }

    inline auto size_aw() const
    {
        if (offset_lo_ == -1) {
            return size_;
        } else {
            return offset_lo_;
        }
    }

    inline auto size_lo() const
    {
        if (offset_lo_ == -1) {
            return 0;
        } else {
            return size_ - offset_lo_;
        }
    }

    inline auto idxrf(bf_index xi__) const
    {
        return idxrf_[xi__];
    }

    inline auto l(bf_index xi__) const
    {
        return indexr_.am(idxrf(xi__)).l();
    }

    inline auto order(bf_index xi__) const
    {
        return indexr_.order(idxrf(xi__));
    }

    /// Return total angular momentum.
    inline auto am(bf_index xi__) const
    {
        return indexr_.am(idxrf(xi__));
    }

    inline auto lm(bf_index xi__) const
    {
        return lm_[xi__];
    }

    inline auto& indexr() const
    {
        return indexr_;
    }

    inline auto index_of(rf_index idxrf__) const
    {
        return bf_index(offset_[idxrf__]);
    }

    auto begin() const
    {
        return bf_index(0);
    }

    auto end() const
    {
        return bf_index(size_);
    }

    auto begin_aw() const
    {
        return bf_index(0);
    }

    auto end_aw() const
    {
        if (offset_lo_ < 0) {
            return bf_index(size_);
        } else {
            return bf_index(offset_lo_);
        }
    }

    auto begin_lo() const
    {
        return bf_index(offset_lo_);
    }

    auto end_lo() const
    {
        if (offset_lo_ < 0) {
            return bf_index(offset_lo_);
        } else {
            return bf_index(size_);
        }
    }

//    inline int order(int xi__) const
//    {
//        return indexr_[(*this)[xi__].idxrf].order;
//    }
//
    inline auto index_by_lm_order(int lm__, int order__) const
    {
        return bf_index(index_by_lm_order_(lm__, order__));
    }

    /* this is needed for full angular momentum indexing
    inline auto index_by_jmj_order(...) const
    {
    }
    */
//
//
//    //void init(radial_functions_index& indexr__)
//    //{
//    //    basis_function_index_descriptors_.clear();
//
//    //    index_by_idxrf_ = mdarray<int, 1>(indexr__.size());
//
//    //    for (int idxrf = 0; idxrf < indexr__.size(); idxrf++) {
//    //        int l     = indexr__[idxrf].l;
//    //        int order = indexr__[idxrf].order;
//    //        int idxlo = indexr__[idxrf].idxlo;
//
//    //        index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();
//
//    //        for (int m = -l; m <= l; m++) {
//    //            basis_function_index_descriptors_.push_back(
//    //                basis_function_index_descriptor(l, m, indexr__[idxrf].j, order, idxlo, idxrf));
//    //        }
//    //    }
//    //    index_by_lm_order_ = mdarray<int, 2>(utils::lmmax(indexr__.lmax()), indexr__.max_num_rf());
//
//    //    for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++) {
//    //        int lm    = basis_function_index_descriptors_[i].lm;
//    //        int order = basis_function_index_descriptors_[i].order;
//    //        index_by_lm_order_(lm, order) = i;
//
//    //        /* get number of aw basis functions */
//    //        if (basis_function_index_descriptors_[i].idxlo < 0) {
//    //            size_aw_ = i + 1;
//    //        }
//    //    }
//
//    //    size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;
//
//    //    lmax_ = indexr__.lmax();
//
//    //    assert(size_aw_ >= 0);
//    //    assert(size_lo_ >= 0);
//    //}
//
//    ///// Return total number of MT basis functions.
//    //inline int size() const
//    //{
//    //    return static_cast<int>(basis_function_index_descriptors_.size());
//    //}
//
//    //inline int size_aw() const
//    //{
//    //    return size_aw_;
//    //}
//
//    //inline int size_lo() const
//    //{
//    //    return size_lo_;
//    //}
//
//    //inline int index_by_l_m_order(int l, int m, int order) const
//    //{
//    //    return index_by_lm_order_(utils::lm(l, m), order);
//    //}
//
//    //inline int index_by_lm_order(int lm, int order) const
//    //{
//    //    return index_by_lm_order_(lm, order);
//    //}
//
//    //inline int index_by_idxrf(int idxrf) const
//    //{
//    //    return index_by_idxrf_(idxrf);
//    //}
//
//    ///// Return descriptor of the given basis function.
//    //inline basis_function_index_descriptor const& operator[](int i) const
//    //{
//    //    assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
//    //    return basis_function_index_descriptors_[i];
//    //}
};



} // namespace "experimental"

/// A helper class to establish various index mappings for the atomic radial functions.
class radial_functions_index
{
  private:
    /// A list of radial function index descriptors.
    /** This list establishes a mapping \f$ f_{\mu}(r) \leftrightarrow  f_{\ell \nu}(r) \f$ between a
     *  composite index \f$ \mu \f$ of radial functions and
     *  corresponding \f$ \ell \nu \f$ indices, where \f$ \ell \f$ is the orbital quantum number and
     *  \f$ \nu \f$ is the order of radial function for a given \f$ \ell \f$. */
    std::vector<radial_function_index_descriptor> radial_function_index_descriptors_;

    sddk::mdarray<int, 2> index_by_l_order_;

    sddk::mdarray<int, 1> index_by_idxlo_;

    /// Number of radial functions for each angular momentum quantum number.
    std::vector<int> num_rf_;

    /// Number of local orbitals for each angular momentum quantum number.
    std::vector<int> num_lo_;

    // Maximum number of radial functions across all angular momentums.
    int max_num_rf_;

    /// Maximum orbital quantum number of augmented-wave radial functions.
    int lmax_aw_;

    /// Maximum orbital quantum number of local orbital radial functions.
    int lmax_lo_;

    /// Maximum orbital quantum number of radial functions.
    int lmax_;

  public:
    /// Initialize a list of radial functions from the list of local orbitals.
    template <typename T>
    void init(std::vector<T> const& lo_descriptors__)
    {
        /* create an empty descriptor */
        std::vector<radial_solution_descriptor_set> aw_descriptors;
        this->init(aw_descriptors, lo_descriptors__);
    }

    /// Initialize a list of radial functions from the list of APW radial functions and the list of local orbitals.
    template <typename T>
    void init(std::vector<radial_solution_descriptor_set> const& aw_descriptors,
              std::vector<T> const& lo_descriptors)
    {
        lmax_aw_ = static_cast<int>(aw_descriptors.size()) - 1;
        lmax_lo_ = -1;
        for (size_t idxlo = 0; idxlo < lo_descriptors.size(); idxlo++) {
            int l    = lo_descriptors[idxlo].l;
            lmax_lo_ = std::max(lmax_lo_, l);
        }

        lmax_ = std::max(lmax_aw_, lmax_lo_);

        num_rf_ = std::vector<int>(lmax_ + 1, 0);
        num_lo_ = std::vector<int>(lmax_ + 1, 0);

        max_num_rf_ = 0;

        radial_function_index_descriptors_.clear();

        for (int l = 0; l <= lmax_aw_; l++) {
            assert(aw_descriptors[l].size() <= 3);

            for (size_t order = 0; order < aw_descriptors[l].size(); order++) {
                radial_function_index_descriptors_.push_back(radial_function_index_descriptor(l, num_rf_[l]));
                num_rf_[l]++;
            }
        }

        for (int idxlo = 0; idxlo < static_cast<int>(lo_descriptors.size()); idxlo++) {
            int l = lo_descriptors[idxlo].l;
            radial_function_index_descriptors_.push_back(
                radial_function_index_descriptor(l, lo_descriptors[idxlo].total_angular_momentum, num_rf_[l], idxlo));
            num_rf_[l]++;
            num_lo_[l]++;
        }

        for (int l = 0; l <= lmax_; l++) {
            max_num_rf_ = std::max(max_num_rf_, num_rf_[l]);
        }

        index_by_l_order_ = sddk::mdarray<int, 2>(lmax_ + 1, max_num_rf_);

        if (lo_descriptors.size()) {
            index_by_idxlo_ = sddk::mdarray<int, 1>(lo_descriptors.size());
        }

        for (int i = 0; i < (int)radial_function_index_descriptors_.size(); i++) {
            int l     = radial_function_index_descriptors_[i].l;
            int order = radial_function_index_descriptors_[i].order;
            int idxlo = radial_function_index_descriptors_[i].idxlo;
            index_by_l_order_(l, order) = i;
            if (idxlo >= 0) {
                index_by_idxlo_(idxlo) = i;
            }
        }
    }

    inline int size() const
    {
        return static_cast<int>(radial_function_index_descriptors_.size());
    }

    inline radial_function_index_descriptor const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)radial_function_index_descriptors_.size());
        return radial_function_index_descriptors_[i];
    }

    inline int index_by_l_order(int l, int order) const
    {
        return index_by_l_order_(l, order);
    }

    inline int index_by_idxlo(int idxlo) const
    {
        return index_by_idxlo_(idxlo);
    }

    /// Number of radial functions for a given orbital quantum number.
    inline int num_rf(int l) const
    {
        assert(l >= 0 && l < (int)num_rf_.size());
        return num_rf_[l];
    }

    /// Number of local orbitals for a given orbital quantum number.
    inline int num_lo(int l) const
    {
        assert(l >= 0 && l < (int)num_lo_.size());
        return num_lo_[l];
    }

    /// Maximum possible number of radial functions for an orbital quantum number.
    inline int max_num_rf() const
    {
        return max_num_rf_;
    }

    inline int lmax() const
    {
        return lmax_;
    }

    inline int lmax_lo() const
    {
        return lmax_lo_;
    }
};

}

#endif
