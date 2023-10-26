// Copyright (c) 2013-2023 Anton Kozhevnikov, Thomas Schulthess
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

    inline T get() const
    {
        return val_;
    }

    operator T() const
    {
        return val_;
    }

    inline bool operator!=(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ != rhs__.val_;
    }

    inline bool operator==(strong_type<T, Tag> const& rhs__)
    {
        return this->val_ == rhs__.val_;
    }

    inline strong_type<T, Tag>& operator++(int)
    {
        this->val_++;
        return *this;
    }
    inline strong_type<T, Tag>& operator+=(T rhs__)
    {
        this->val_ += rhs__;
        return *this;
    }
};

}

#endif
