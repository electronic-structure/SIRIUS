/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file basis_functions_index.hpp
 *
 *  \brief Contains definitiona and implementation of sirius::basis_functions_index class.
 */

#ifndef __BASIS_FUNCTIONS_INDEX_HPP__
#define __BASIS_FUNCTIONS_INDEX_HPP__

namespace sirius {

struct basis_function_index_descriptor
{
    /// Total angular momemtum.
    angular_momentum am;
    /// Projection of the angular momentum.
    int m;
    /// Composite index.
    int lm;
    /// Order of the radial function for a given l (j).
    int order;
    /// Index of local orbital.
    rf_lo_index idxlo;
    /// Index of the radial function or beta projector in the case of pseudo potential.
    rf_index idxrf;
    bf_index xi{-1};

    basis_function_index_descriptor(angular_momentum am, int m, int order, rf_lo_index idxlo, rf_index idxrf)
        : am(am)
        , m(m)
        , lm(sf::lm(am.l(), m))
        , order(order)
        , idxlo(idxlo)
        , idxrf(idxrf)
    {
        RTE_ASSERT(m >= -am.l() && m <= am.l());
        RTE_ASSERT(order >= 0);
        RTE_ASSERT(idxrf >= 0);
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
    std::vector<basis_function_index_descriptor> vbd_;

    mdarray<int, 2> index_by_lm_order_;

    int offset_lo_{-1};

    std::vector<int> offset_;

    radial_functions_index indexr_;

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
            index_by_lm_order_ = mdarray<int, 2>({sf::lmmax(indexr_.lmax()), indexr_.max_order()});
            std::fill(index_by_lm_order_.begin(), index_by_lm_order_.end(), -1);
            /* loop over radial functions */
            for (auto e : indexr_) {

                /* index of this block starts from the current size of basis functions descriptor */
                auto size = this->size();

                // if (e.am.s() != 0) {
                //     RTE_THROW("full-j radial function index is not allowed here");
                // }
                if (e.idxrf == indexr_.index_of(rf_lo_index(0))) {
                    offset_lo_ = size;
                }
                /* angular momentum */
                auto am = e.am;

                offset_.push_back(size);

                for (int m = -am.l(); m <= am.l(); m++) {
                    vbd_.push_back(basis_function_index_descriptor(am, m, e.order, e.idxlo, e.idxrf));
                    vbd_.back().xi = bf_index(size);
                    /* reverse mapping */
                    index_by_lm_order_(sf::lm(am.l(), m), e.order) = size;
                    size++;
                }
            }
        } else { /* for the full-j expansion */
            /* several things have to be done here:
             *  - packing of jmj index for l+1/2 and l-1/2 subshells has to be introduced
             *    like existing sf::lm(l, m) function
             *  - indexing within l-shell has to be implemented; l shell now contains 2(2l+1) spin orbitals
             *  - order of s=-1 and s=1 components has to be agreed upon and respected
             */
            RTE_THROW("full j is not yet implemented");
        }
    }
    /// Return total number of MT basis functions.
    inline int
    size() const
    {
        return static_cast<int>(vbd_.size());
    }

    /// Return size of AW part of basis functions in case of LAPW.
    inline auto
    size_aw() const
    {
        if (offset_lo_ == -1) {
            return this->size();
        } else {
            return offset_lo_;
        }
    }

    /// Return size of local-orbital part of basis functions in case of LAPW.
    inline auto
    size_lo() const
    {
        if (offset_lo_ == -1) {
            return 0;
        } else {
            return this->size() - offset_lo_;
        }
    }

    inline int
    index_by_l_m_order(int l, int m, int order) const
    {
        return index_by_lm_order_(sf::lm(l, m), order);
    }

    inline int
    index_by_lm_order(int lm, int order) const
    {
        return index_by_lm_order_(lm, order);
    }

    inline int
    index_of(rf_index idxrf__) const
    {
        return offset_[idxrf__];
    }

    /// Return descriptor of the given basis function.
    inline auto const&
    operator[](int i) const
    {
        RTE_ASSERT(i >= 0 && i < this->size());
        return vbd_[i];
    }

    inline auto
    begin() const
    {
        return vbd_.begin();
    }

    inline auto
    end() const
    {
        return vbd_.end();
    }

    auto const&
    indexr() const
    {
        return indexr_;
    }
};

inline auto
begin(basis_functions_index const& idx__)
{
    return idx__.begin();
}

inline auto
end(basis_functions_index const& idx__)
{
    return idx__.end();
}

} // namespace sirius

#endif
