#ifndef __BASIS_FUNCTIONS_INDEX_HPP__
#define __BASIS_FUNCTIONS_INDEX_HPP__

namespace sirius {

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

    mdarray<int, 1> index_by_idxrf_;

    /// Number of augmented wave basis functions.
    int size_aw_{0};

    /// Number of local orbital basis functions.
    int size_lo_{0};

  public:

    void init(radial_functions_index& indexr)
    {
        basis_function_index_descriptors_.clear();

        index_by_idxrf_ = mdarray<int, 1>(indexr.size());

        for (int idxrf = 0; idxrf < indexr.size(); idxrf++) {
            int l     = indexr[idxrf].l;
            int order = indexr[idxrf].order;
            int idxlo = indexr[idxrf].idxlo;

            index_by_idxrf_(idxrf) = (int)basis_function_index_descriptors_.size();

            for (int m = -l; m <= l; m++)
                basis_function_index_descriptors_.push_back(
                    basis_function_index_descriptor(l, m, indexr[idxrf].j, order, idxlo, idxrf));
        }
        index_by_lm_order_ = mdarray<int, 2>(Utils::lmmax(indexr.lmax()), indexr.max_num_rf());

        for (int i = 0; i < (int)basis_function_index_descriptors_.size(); i++) {
            int lm    = basis_function_index_descriptors_[i].lm;
            int order = basis_function_index_descriptors_[i].order;
            index_by_lm_order_(lm, order) = i;

            // get number of aw basis functions
            if (basis_function_index_descriptors_[i].idxlo < 0)
                size_aw_ = i + 1;
        }

        size_lo_ = (int)basis_function_index_descriptors_.size() - size_aw_;

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
        return index_by_lm_order_(Utils::lm_by_l_m(l, m), order);
    }

    inline int index_by_lm_order(int lm, int order) const
    {
        return index_by_lm_order_(lm, order);
    }

    inline int index_by_idxrf(int idxrf) const
    {
        return index_by_idxrf_(idxrf);
    }

    inline basis_function_index_descriptor const& operator[](int i) const
    {
        assert(i >= 0 && i < (int)basis_function_index_descriptors_.size());
        return basis_function_index_descriptors_[i];
    }
};

}

#endif
