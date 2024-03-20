/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef __DENSITY_MATRIX_HPP__
#define __DENSITY_MATRIX_HPP__

namespace sirius {

class density_matrix_t
{
  private:
    std::vector<mdarray<std::complex<double>, 3>> data_;

  public:
    density_matrix_t(Unit_cell const& uc__, int num_mag_comp__)
    {
        data_ = std::vector<mdarray<std::complex<double>, 3>>(uc__.num_atoms());
        for (int ia = 0; ia < uc__.num_atoms(); ia++) {
            auto& atom = uc__.atom(ia);
            data_[ia]  = mdarray<std::complex<double>, 3>({atom.mt_basis_size(), atom.mt_basis_size(), num_mag_comp__});
        }
        this->zero();
    }
    void
    zero()
    {
        for (auto& e : data_) {
            e.zero();
        }
    }
    auto
    size() const
    {
        return data_.size();
    }
    auto&
    operator[](int ia__)
    {
        return data_[ia__];
    }
    auto const&
    operator[](int ia__) const
    {
        return data_[ia__];
    }
    auto const
    begin() const
    {
        return data_.begin();
    }
    auto const
    end() const
    {
        return data_.end();
    }
};

inline void
copy(density_matrix_t const& src__, density_matrix_t& dest__)
{
    for (size_t i = 0; i < src__.size(); i++) {
        copy(src__[i], dest__[i]);
    }
}

} // namespace sirius

#endif
