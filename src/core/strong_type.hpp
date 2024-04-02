/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file strong_type.hpp
 *
 *  \brief A wrapper class to create strong types.
 */

#ifndef __STRONG_TYPE_HPP__
#define __STRONG_TYPE_HPP__

namespace sirius {

template <typename T, typename Tag>
class strong_type
{
  private:
    T val_;

  public:
    explicit strong_type(T const& val__)
        : val_{val__}
    {
    }

    explicit strong_type(T&& val__)
        : val_{std::move(val__)}
    {
    }

    inline T
    get() const
    {
        return val_;
    }

    operator T() const
    {
        return val_;
    }

    inline bool
    operator!=(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ != rhs__.val_;
    }

    inline bool
    operator==(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ == rhs__.val_;
    }

    inline strong_type<T, Tag>&
    operator++(int)
    {
        this->val_++;
        return *this;
    }
    inline strong_type<T, Tag>&
    operator+=(T rhs__)
    {
        this->val_ += rhs__;
        return *this;
    }
};

} // namespace sirius

#endif
