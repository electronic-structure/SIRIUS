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

/** \file any_ptr.hpp
 *
 *  \brief Implementation of pointer to any object.
 */

#ifndef __ANY_PTR_HPP__
#define __ANY_PTR_HPP__

namespace utils {

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
    T& get() const
    {
        return *static_cast<T*>(ptr_);
    }
};

} // namespace utils

#endif
