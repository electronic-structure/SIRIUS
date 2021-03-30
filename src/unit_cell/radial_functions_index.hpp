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

namespace sirius {

namespace experimental {

#define THROW(msg)                                                                                    \
{                                                                                                     \
    do {                                                                                              \
        throw std::runtime_error(std::string("[") + std::string(__func__) + std::string("] ") + msg); \
    } while(0);                                                                                       \
}



/// Angular momentum quantum number.
/** This class handles orbital or total angluar momentum quantum number. */
class angular_momentum_quantum_number
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
    explicit angular_momentum_quantum_number(int l__)
      : l_(l__)
    {
        if (l__ < 0) {
            THROW("l can't be negative");
        }
    }

    /// Constructor.
    explicit angular_momentum_quantum_number(int l__, int s__)
      : l_(l__)
      , s_(s__)
    {
        if (l__ < 0) {
            THROW("l can't be negative");
        }
        if (s__ != -1 && s__ != 0 && s__ != 1) {
            THROW("wrong value of s");
        }
        if (l__ == 0 && s__ == -1) {
            THROW("incompatible combination of l and s quantum numbers");
        }
    }

    /// Get orbital quantum number l.
    inline int l() const
    {
        return l_;
    }

    /// Get total angular momentum j = l +/- 1/2
    inline double j() const
    {
        return l_ + s_ / 2.0;
    }

    /// Get spin quantum number s.
    inline int s() const
    {
        return s_;
    }
};

/// Descriptor of the radial function.
/** Several radial functions can exist for the same angular quantum number. In order to distinguish them, they
 *  are labeled by the additional index \f$ \nu \f$ : \f$ f_{\ell \nu}(r) \f$ or \f$ f_{j \nu}(r) \f$
 *  which is called the "order" of the radial function for a given orbital quantum number. */
struct radial_function_index_descriptor
{
    /// Angular momentum quantum number \f$ \ell \f$ or total momentum j.
    angular_momentum_quantum_number aqn;

    /// Order of a function for a given \f$ \ell \f$.
    int order{-1};

    /// If this is a local orbital radial function, idxlo is it's index in the list of local orbital descriptors.
    int idxlo{-1}; // TODO: check if this is strictly necessary

    /// Constructor.
    radial_function_index_descriptor(angular_momentum_quantum_number aqn__, int order__, int idxlo__)
        : aqn(aqn__)
        , order(order__)
        , idxlo(idxlo__)
    {
        if (order < 0) {
            THROW("wrong order of radial function");
        }
    }
};

class radial_functions_index : public std::vector<radial_function_index_descriptor>
{
  private:
    std::vector<std::vector<int>> index_by_l_order_;
    std::vector<int> index_by_idxlo_;
    int full_j_{-1};
  public:

    /// Add index of radial function to the list.
    void add(angular_momentum_quantum_number aqn__, bool is_lo__ = false)
    {
        if (is_lo__ && aqn__.s()) {
            THROW("local orbitals can only be of pure l character");
        }
        /* perform immediate check of j */
        if (full_j_ == -1) { /* checking first time */
            full_j_ = std::abs(aqn__.s()); /* set if this will be an index for the full j orbitals or for pure l */
        } else {
            /* this is not the first time */
            //if (full_j_ == 0 && aqn__.s()) {
            //    throw std::runtime_error("radial orbital index is set to count pure-l radial functions");
            //}
            //if (full_j_ && aqn__.s() == 0) {
            //    throw std::runtime_error("radial orbital index is set to count full-j radial functions");
            //}
        }
        /* current l */
        int l = aqn__.l();
        /* make sure that the space is available */
        if (static_cast<int>(index_by_l_order_.size()) < l + 1) {
            index_by_l_order_.resize(l + 1);
        }
        /* current size of the index */
        auto i = static_cast<int>(this->size());
        /* current order of the radial function for l */
        int o = static_cast<int>(index_by_l_order_[l].size());

        int idxlo = is_lo__ ? static_cast<int>(index_by_idxlo_.size()) : -1;

        /* add radial function descriptor */
        this->emplace_back(aqn__, o, idxlo);

        /* add reverse index */
        index_by_l_order_[aqn__.l()].push_back(i);
        if (is_lo__) {
            index_by_idxlo_.push_back(i);
        }
    }

    inline int index_by_l_order(angular_momentum_quantum_number aqn__, int order__) const
    {
        int idx = index_by_l_order_[aqn__.l()][order__];
        if ((*this)[idx].aqn.s() != aqn__.s()) {
            THROW("wrong value of spin");
        }
        if ((*this)[idx].order != order__) {
            THROW("wrong order");
        }
        return idx;
    }

    inline int index_by_l_order(int l__, int order__) const
    {
        int idx = index_by_l_order_[l__][order__];
        if ((*this)[idx].order != order__) {
            THROW("wrong order");
        }
        return idx;
    }

    inline bool full_j() const
    {
        return (full_j_ == 1);
    }

    inline int lmax() const
    {
        return static_cast<int>(index_by_l_order_.size()) - 1;
    }

    inline int max_order(int l__) const
    {
        return static_cast<int>(index_by_l_order_[l__].size());
    }
};

struct basis_function_index_descriptor
{
    /// Projection of the angular momentum.
    int m;
    /// Composite index.
    int lm;
    /// Index of the radial function or beta projector in the case of pseudo potential.
    int idxrf;

    //basis_function_index_descriptor(int l, int m, int order, int idxlo, int idxrf)
    //    : l(l)
    //    , m(m)
    //    , lm(utils::lm(l, m))
    //    , order(order)
    //    , idxlo(idxlo)
    //    , idxrf(idxrf)
    //{
    //    assert(l >= 0);
    //    assert(m >= -l && m <= l);
    //    assert(order >= 0);
    //    assert(idxrf >= 0);
    //}

    //basis_function_index_descriptor(int l, int m, double j, int order, int idxlo, int idxrf)
    //    : l(l)
    //    , m(m)
    //    , lm(utils::lm(l, m))
    //    , j(j)
    //    , order(order)
    //    , idxlo(idxlo)
    //    , idxrf(idxrf)
    //{
    //    assert(l >= 0);
    //    assert(m >= -l && m <= l);
    //    assert(order >= 0);
    //    assert(idxrf >= 0);
    //}
};


/// A helper class to establish various index mappings for the atomic basis functions.
/** Atomic basis function is a radial function multiplied by a spherical harmonic:
    \f[
      \phi_{\ell m \nu}({\bf r}) = f_{\ell \nu}(r) Y_{\ell m}(\hat {\bf r})
    \f]
    Multiple radial functions for each \f$ \ell \f$ channel are allowed. This is reflected by
    the \f$ \nu \f$ index and called "order".
  */
class basis_functions_index : public std::vector<basis_function_index_descriptor>
{
  private:
    radial_functions_index indexr_;

    //mdarray<int, 2> index_by_lm_order_;

    //mdarray<int, 1> index_by_idxrf_; // TODO: rename to first_lm_index_by_idxrf_ or similar

    ///// Number of augmented wave basis functions.
    //int size_aw_{0};

    ///// Number of local orbital basis functions.
    //int size_lo_{0};

    ///// Maximum l of the radial basis functions.
    //int lmax_{-1};
    //
    std::vector<std::vector<int>> index_by_lm_order_; // TODO: subject to change to index_by_lm_idxrf

  public:
    basis_functions_index()
    {
    }
    basis_functions_index(radial_functions_index const& indexr__, bool expand_full_j__)
        : indexr_(indexr__)
    {
        if (expand_full_j__) {
            throw std::runtime_error("j,mj expansion of the full angular momentum index is not implemented");
        }
        /* check radial index here */
        if (indexr_.full_j()) {
            for (int l = 1; l <= indexr_.lmax(); l++) { /* skip s-states, they don't have +/- 1/2 splitting */
                if (indexr_.max_order(l) % 2) {
                    throw std::runtime_error("number of radial functions should be even");
                }
                for (int o = 0; o < indexr_.max_order(l); o += 2) {
                    auto i1 = indexr_.index_by_l_order(l, o);
                    auto i2 = indexr_.index_by_l_order(l, o + 1);
                    if (i2 != i1 + 1) {
                        throw std::runtime_error("wrong order of radial functions");
                    }
                    if (indexr_[i1].aqn.s() * indexr_[i2].aqn.s() != -1) {
                        throw std::runtime_error("wrong j of radial functions");
                    }
                }
            }
        }

        if (!expand_full_j__) {
            int lmmax = utils::lmmax(indexr_.lmax());
            index_by_lm_order_.resize(lmmax);
            for (int idxrf = 0; idxrf < static_cast<int>(indexr_.size()); idxrf++) {
                int l = indexr_[idxrf].aqn.l();
                int o = indexr_[idxrf].order;
                for (int m = -l; m <= l; m++) {
                    sirius::experimental::basis_function_index_descriptor b;
                    b.idxrf = idxrf;
                    b.m = m;
                    b.lm = utils::lm(l, m);
                    if (static_cast<int>(index_by_lm_order_[b.lm].size()) < indexr_.max_order(l)) {
                        index_by_lm_order_[b.lm].resize(indexr_.max_order(l));
                    }
                    int idx = static_cast<int>(this->size());
                    this->push_back(b);
                    index_by_lm_order_[b.lm][o] = idx;
                }
            }
        }
    }

    inline int l(int xi__) const
    {
        return indexr_[(*this)[xi__].idxrf].aqn.l();
    }

    inline int lm(int xi__) const
    {
        return (*this)[xi__].lm;
    }

    inline int order(int xi__) const
    {
        return indexr_[(*this)[xi__].idxrf].order;
    }

    inline int index_by_lm_order(int lm__, int order__) const // TODO: subject to change
    {
        return index_by_lm_order_[lm__][order__];
    }


    //void init(radial_functions_index& indexr__)
    //{
    //    basis_function_index_descriptors_.clear();

    //    index_by_idxrf_ = mdarray<int, 1>(indexr__.size());

    //    for (int idxrf = 0; idxrf < indexr__.size(); idxrf++) {
    //        int l     = indexr__[idxrf].l;
    //        int order = indexr__[idxrf].order;
    //        int idxlo = indexr__[idxrf].idxlo;

    //        index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

    //        for (int m = -l; m <= l; m++) {
    //            basis_function_index_descriptors_.push_back(
    //                basis_function_index_descriptor(l, m, indexr__[idxrf].j, order, idxlo, idxrf));
    //        }
    //    }
    //    index_by_lm_order_ = mdarray<int, 2>(utils::lmmax(indexr__.lmax()), indexr__.max_num_rf());

    //    for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++) {
    //        int lm    = basis_function_index_descriptors_[i].lm;
    //        int order = basis_function_index_descriptors_[i].order;
    //        index_by_lm_order_(lm, order) = i;

    //        /* get number of aw basis functions */
    //        if (basis_function_index_descriptors_[i].idxlo < 0) {
    //            size_aw_ = i + 1;
    //        }
    //    }

    //    size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

    //    lmax_ = indexr__.lmax();

    //    assert(size_aw_ >= 0);
    //    assert(size_lo_ >= 0);
    //}

    ///// Return total number of MT basis functions.
    //inline int size() const
    //{
    //    return static_cast<int>(basis_function_index_descriptors_.size());
    //}

    //inline int size_aw() const
    //{
    //    return size_aw_;
    //}

    //inline int size_lo() const
    //{
    //    return size_lo_;
    //}

    //inline int index_by_l_m_order(int l, int m, int order) const
    //{
    //    return index_by_lm_order_(utils::lm(l, m), order);
    //}

    //inline int index_by_lm_order(int lm, int order) const
    //{
    //    return index_by_lm_order_(lm, order);
    //}

    //inline int index_by_idxrf(int idxrf) const
    //{
    //    return index_by_idxrf_(idxrf);
    //}

    ///// Return descriptor of the given basis function.
    //inline basis_function_index_descriptor const& operator[](int i) const
    //{
    //    assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
    //    return basis_function_index_descriptors_[i];
    //}
};
}

/// Descriptor for the atomic radial functions.
/** The radial functions \f$ f_{\ell \nu}(r) \f$ are labeled by two indices: orbital quantum number \f$ \ell \f$ and
 *  an order \f$ \nu \f$ for a given \f$ \ell \f$. Radial functions can be any of augmented waves of local orbitals
 *  (in case of FP-LAPW) or bete projectors, atomic or Hubbard wave functions in case of PP-PW.
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

    /// Constructor.
    radial_function_index_descriptor(int l, int order, int idxlo = -1)
        : l(l)
        , order(order)
        , idxlo(idxlo)
    {
        assert(l >= 0);
        assert(order >= 0);
    }

    radial_function_index_descriptor(int l, double j, int order, int idxlo = -1)
        : l(l)
        , j(j)
        , order(order)
        , idxlo(idxlo)
    {
        assert(l >= 0);
        assert(order >= 0);
    }
};

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

    mdarray<int, 2> index_by_l_order_;

    mdarray<int, 1> index_by_idxlo_;

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

        index_by_l_order_ = mdarray<int, 2>(lmax_ + 1, max_num_rf_);

        if (lo_descriptors.size()) {
            index_by_idxlo_ = mdarray<int, 1>(lo_descriptors.size());
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
