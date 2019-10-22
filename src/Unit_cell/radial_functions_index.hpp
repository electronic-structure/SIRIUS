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
