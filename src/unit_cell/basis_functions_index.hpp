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

/** \file basis_functions_index.hpp
 *
 *  \brief Contains definitiona and implementation of sirius::basis_functions_index class.
 */

#ifndef __BASIS_FUNCTIONS_INDEX_HPP__
#define __BASIS_FUNCTIONS_INDEX_HPP__

namespace sirius {

struct basis_function_index_descriptor
{
    /// Angular momentum.
    int l;
    /// Projection of the angular momentum.
    int m;
    /// Composite index.
    int lm;
    /// Total angular momemtum. // TODO: replace with positive and negative integer in l
    double j;
    /// Order of the radial function for a given l (j).
    int order;
    /// Index of local orbital.
    int idxlo;
    /// Index of the radial function or beta projector in the case of pseudo potential.
    int idxrf;

    basis_function_index_descriptor(int l, int m, int order, int idxlo, int idxrf)
        : l(l)
        , m(m)
        , lm(utils::lm(l, m))
        , order(order)
        , idxlo(idxlo)
        , idxrf(idxrf)
    {
        assert(l >= 0);
        assert(m >= -l && m <= l);
        assert(order >= 0);
        assert(idxrf >= 0);
    }

    basis_function_index_descriptor(int l, int m, double j, int order, int idxlo, int idxrf)
        : l(l)
        , m(m)
        , lm(utils::lm(l, m))
        , j(j)
        , order(order)
        , idxlo(idxlo)
        , idxrf(idxrf)
    {
        assert(l >= 0);
        assert(m >= -l && m <= l);
        assert(order >= 0);
        assert(idxrf >= 0);
    }
};

/// A helper class to establish various index mappings for the atomic basis functions.
/** Atomic basis function is a radial function multiplied by a spherical harmonic:
    \f[
      \phi_{\ell m \nu}({\bf r}) = f_{\ell \nu}(r) Y_{\ell m}(\hat {\bf r})
    \f]
    Multiple radial functions for each \f$ \ell \f$ channel are allowed. This is reflected by
    the \f$ \nu \f$ index and called "order".
  */
class basis_functions_index
{
  private:
    std::vector<basis_function_index_descriptor> basis_function_index_descriptors_;

    mdarray<int, 2> index_by_lm_order_;

    mdarray<int, 1> index_by_idxrf_; // TODO: rename to first_lm_index_by_idxrf_ or similar

    /// Number of augmented wave basis functions.
    int size_aw_{0};

    /// Number of local orbital basis functions.
    int size_lo_{0};

    /// Maximum l of the radial basis functions.
    int lmax_{-1};

  public:

    void init(radial_functions_index& indexr__)
    {
        basis_function_index_descriptors_.clear();

        index_by_idxrf_ = mdarray<int, 1>(indexr__.size());

        for (int idxrf = 0; idxrf < indexr__.size(); idxrf++) {
            int l     = indexr__[idxrf].l;
            int order = indexr__[idxrf].order;
            int idxlo = indexr__[idxrf].idxlo;

            index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

            for (int m = -l; m <= l; m++) {
                basis_function_index_descriptors_.push_back(
                    basis_function_index_descriptor(l, m, indexr__[idxrf].j, order, idxlo, idxrf));
            }
        }
        index_by_lm_order_ = mdarray<int, 2>(utils::lmmax(indexr__.lmax()), indexr__.max_num_rf());

        for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++) {
            int lm    = basis_function_index_descriptors_[i].lm;
            int order = basis_function_index_descriptors_[i].order;
            index_by_lm_order_(lm, order) = i;

            /* get number of aw basis functions */
            if (basis_function_index_descriptors_[i].idxlo < 0) {
                size_aw_ = i + 1;
            }
        }

        size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

        lmax_ = indexr__.lmax();

        assert(size_aw_ >= 0);
        assert(size_lo_ >= 0);
    }

    /// Return total number of MT basis functions.
    inline int size() const
    {
        return static_cast<int>(basis_function_index_descriptors_.size());
    }

    inline int size_aw() const
    {
        return size_aw_;
    }

    inline int size_lo() const
    {
        return size_lo_;
    }

    inline int index_by_l_m_order(int l, int m, int order) const
    {
        return index_by_lm_order_(utils::lm(l, m), order);
    }

    inline int index_by_lm_order(int lm, int order) const
    {
        return index_by_lm_order_(lm, order);
    }

    inline int index_by_idxrf(int idxrf) const
    {
        return index_by_idxrf_(idxrf);
    }

    /// Return descriptor of the given basis function.
    inline basis_function_index_descriptor const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
        return basis_function_index_descriptors_[i];
    }
};

}

#endif
