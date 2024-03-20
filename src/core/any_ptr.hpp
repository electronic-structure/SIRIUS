/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file any_ptr.hpp
 *
 *  \brief Implementation of pointer to any object.
 */

#ifndef __ANY_PTR_HPP__
#define __ANY_PTR_HPP__

#include <functional>

namespace sirius {

/// Handle deallocation of a poiniter to an object of any type.
/** Example:
    \code{.cpp}
    class Foo
    {
      private:
        int n_{0};
      public:
        Foo(int i) : n_(i)
        {
            std::cout << "in Foo() constructor\n";
        }

        ~Foo()
        {
            std::cout << "in Foo() destructor\n";
        }

        void print()
        {
            std::cout << "the number is: " << n_ << "\n";
        }
    };

    int main(int argn, char** argv)
    {
        void* ptr = new any_ptr(new Foo(42));
        auto& foo = static_cast<any_ptr*>(ptr)->get<Foo>();
        foo.print();
        delete static_cast<any_ptr*>(ptr);

        return 0;
    }
    \endcode
    And the output is:
    \verbatim
    in Foo() constructor
    the number is: 42
    in Foo() destructor
    \endverbatim
*/
class any_ptr
{
  private:
    /// Untyped pointer to a stored object.
    void* ptr_;
    /// Deleter for the stored object.
    std::function<void(void*)> deleter_;

  public:
    /// Constructor.
    template <typename T>
    any_ptr(T* ptr__)
        : ptr_(ptr__)
    {
        deleter_ = [](void* p) { delete static_cast<T*>(p); };
    }
    /// Destructor.
    ~any_ptr()
    {
        deleter_(ptr_);
    }
    /// Cast pointer to a given type and return a reference.
    template <typename T>
    T&
    get() const
    {
        return *static_cast<T*>(ptr_);
    }
};

} // namespace sirius

#endif
