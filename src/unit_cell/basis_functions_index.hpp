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
    angular_momentum am;
    /// Order of the radial function for a given l (j).
    int order;
    /// Index of local orbital.
    int idxlo;
    /// Index of the radial function or beta projector in the case of pseudo potential.
    int idxrf;
    bf_index xi{-1};

    basis_function_index_descriptor(int l, int m, int order, int idxlo, int idxrf)
        : l(l)
        , m(m)
        , lm(utils::lm(l, m))
        , am(l)
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
        , am(l)
        , order(order)
        , idxlo(idxlo)
        , idxrf(idxrf)
    {
        assert(l >= 0);
        assert(m >= -l && m <= l);
        assert(order >= 0);
        assert(idxrf >= 0);
    }

    basis_function_index_descriptor(angular_momentum am, int m, int order, int idxlo, int idxrf)
        : l(am.l())
        , m(m)
        , lm(utils::lm(l, m))
        , j(am.j())
        , am(am)
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
    std::vector<basis_function_index_descriptor> basis_function_index_descriptors_; // TODO: rename to vbd_

    sddk::mdarray<int, 2> index_by_lm_order_;

    sddk::mdarray<int, 1> index_by_idxrf_; // TODO: rename to first_lm_index_by_idxrf_ or similar

    /// Number of augmented wave basis functions.
    int size_aw_{0};

    /// Number of local orbital basis functions.
    int size_lo_{0};

    /// Maximum l of the radial basis functions.
    int lmax_{-1};

    int offset_lo_{-1};

    std::vector<int> offset_;

    experimental::radial_functions_index indexr_;

  public:
    basis_functions_index()
    {
    }

    basis_functions_index(experimental::radial_functions_index const& indexr__, bool expand_full_j__)
        : indexr_(indexr__)
    {
        if (expand_full_j__) {
            RTE_THROW("j,mj expansion of the full angular momentum index is not implemented");
        }

        if (!expand_full_j__) {
            index_by_lm_order_ = sddk::mdarray<int, 2>(utils::lmmax(indexr_.lmax()), indexr_.max_order());
            std::fill(index_by_lm_order_.begin(), index_by_lm_order_.end(), -1);
            /* loop over radial functions */
            for (auto e : indexr_) {

                /* index of this block starts from the current size of basis functions descriptor */
                auto size = static_cast<int>(basis_function_index_descriptors_.size());

                if (e.am.s() != 0) {
                    RTE_THROW("full-j radial function index is not allowed here");
                }
                std::cout << "e.idxrf = " << e.idxrf << ", indexr_.index_of(rf_lo_index(0)=" << indexr_.index_of(rf_lo_index(0)) << std::endl;
                if (e.idxrf == indexr_.index_of(rf_lo_index(0))) {
                    offset_lo_ = size;
                }
                std::cout << "offset_lo_ = " << offset_lo_ << std::endl;
                /* angular momentum */
                auto am = e.am;

                offset_.push_back(size);

                for (int m = -am.l(); m <= am.l(); m++) {
                    basis_function_index_descriptors_.push_back(basis_function_index_descriptor(am, m, e.order, e.idxlo, e.idxrf));
                    basis_function_index_descriptors_.back().xi = bf_index(size);
                    /* reverse mapping */
                    index_by_lm_order_(utils::lm(am.l(), m), e.order) = size;
                    size++;
                }
            }
        } else { /* for the full-j expansion */
            /* several things have to be done here:
             *  - packing of jmj index for l+1/2 and l-1/2 subshells has to be introduced
             *    like existing utils::lm(l, m) function
             *  - indexing within l-shell has to be implemented; l shell now contains 2(2l+1) spin orbitals
             *  - order of s=-1 and s=1 components has to be agreed upon and respected
             */
            RTE_THROW("full j is not yet implemented");
        }
    }


    void init(radial_functions_index& indexr__)
    {
        basis_function_index_descriptors_.clear();

        index_by_idxrf_ = sddk::mdarray<int, 1>(indexr__.size());

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
        index_by_lm_order_ = sddk::mdarray<int, 2>(utils::lmmax(indexr__.lmax()), indexr__.max_num_rf());

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

    //inline auto size_aw1() const
    //{
    //    if (offset_lo_ == -1) {
    //        return this->size();
    //    } else {
    //        return offset_lo_;
    //    }
    //}

    //inline auto size_lo1() const
    //{
    //    if (offset_lo_ == -1) {
    //        return 0;
    //    } else {
    //        return this->size() - offset_lo_;
    //    }
    //}

    /// Return size of AW part of basis functions in case of LAPW.
    inline int size_aw() const
    {
        return size_aw_;
    }

    /// Return size of local-orbital part of basis functions in case of LAPW.
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
        //return offset_[idxrf];
    }

    /// Return descriptor of the given basis function.
    inline auto const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
        return basis_function_index_descriptors_[i];
    }
};

namespace experimental {

class basis_functions_index1
{
  private:
    std::vector<basis_function_index_descriptor> vbd_; // TODO: rename to vbd_

    sddk::mdarray<int, 2> index_by_lm_order_;

    /// Maximum l of the radial basis functions.
    int lmax_{-1};

    int offset_lo_{-1};

    std::vector<int> offset_;

    radial_functions_index indexr_;

  public:
    basis_functions_index1()
    {
    }

    basis_functions_index1(experimental::radial_functions_index const& indexr__, bool expand_full_j__)
        : indexr_(indexr__)
    {
        if (expand_full_j__) {
            RTE_THROW("j,mj expansion of the full angular momentum index is not implemented");
        }

        if (!expand_full_j__) {
            index_by_lm_order_ = sddk::mdarray<int, 2>(utils::lmmax(indexr_.lmax()), indexr_.max_order());
            std::fill(index_by_lm_order_.begin(), index_by_lm_order_.end(), -1);
            /* loop over radial functions */
            for (auto e : indexr_) {

                /* index of this block starts from the current size of basis functions descriptor */
                auto size = this->size();

                //if (e.am.s() != 0) {
                //    RTE_THROW("full-j radial function index is not allowed here");
                //}
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
                    index_by_lm_order_(utils::lm(am.l(), m), e.order) = size;
                    size++;
                }
            }
        } else { /* for the full-j expansion */
            /* several things have to be done here:
             *  - packing of jmj index for l+1/2 and l-1/2 subshells has to be introduced
             *    like existing utils::lm(l, m) function
             *  - indexing within l-shell has to be implemented; l shell now contains 2(2l+1) spin orbitals
             *  - order of s=-1 and s=1 components has to be agreed upon and respected
             */
            RTE_THROW("full j is not yet implemented");
        }
    }
    /// Return total number of MT basis functions.
    inline int size() const
    {
        return static_cast<int>(vbd_.size());
    }

    /// Return size of AW part of basis functions in case of LAPW.
    inline auto size_aw() const
    {
        if (offset_lo_ == -1) {
            return this->size();
        } else {
            return offset_lo_;
        }
    }

    /// Return size of local-orbital part of basis functions in case of LAPW.
    inline auto size_lo() const
    {
        if (offset_lo_ == -1) {
            return 0;
        } else {
            return this->size() - offset_lo_;
        }
    }

    inline int index_by_l_m_order(int l, int m, int order) const
    {
        return index_by_lm_order_(utils::lm(l, m), order);
    }

    inline int index_by_lm_order(int lm, int order) const
    {
        return index_by_lm_order_(lm, order);
    }

    inline int offset(rf_index idxrf__) const
    {
        return offset_[idxrf__];
    }

    /// Return descriptor of the given basis function.
    inline auto const& operator[](int i) const
    {
        RTE_ASSERT(i >= 0 && i < this->size());
        return vbd_[i];
    }

    inline auto begin() const
    {
        return vbd_.begin();
    }

    inline auto end() const
    {
        return vbd_.end();
    }
};

inline auto begin(basis_functions_index1 const& idx__)
{
    return idx__.begin();
}

inline auto end(basis_functions_index1 const& idx__)
{
    return idx__.end();
}

}

}

#endif
