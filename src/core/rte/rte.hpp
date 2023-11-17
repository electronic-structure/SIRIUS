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

/** \file rte.hpp
 *
 *  \brief Eror and warning handling during run-time execution.
 *
 */

#ifndef __RTE_HPP__
#define __RTE_HPP__

#include <stdexcept>
#include <sstream>
#include <ostream>
#include <vector>
#include <iostream>
#include "core/string_tools.hpp"

namespace sirius {

/// Run-time error and warning handling.
namespace rte {

inline void
message_impl(bool fatal__, const char* func__, const char* file__, int line__, std::string const& msg__)
{
    std::stringstream s;

    if (!fatal__) {
        s << "Warning";
    } else {
        s << "Exception";
    }
    s << " in function \"" << func__ << "\" at " << file__ << ":" << line__ << std::endl;
    s << msg__;

    if (fatal__) {
        throw std::runtime_error(s.str());
    } else {
        std::cout << s.str() << std::endl;
    }
}

inline void
message_impl(bool fatal__, const char* func__, const char* file__, int line__, std::stringstream const& msg__)
{
    message_impl(fatal__, func__, file__, line__, msg__.str());
}

class ostream : public std::ostringstream
{
  private:
    std::ostream* out_{nullptr};
    std::string prefix_;

  public:
    ostream()
    {
    }
    ostream(std::ostream& out__, std::string prefix__)
        : out_(&out__)
        , prefix_(prefix__)
    {
    }
    ostream(ostream&& src__)
        : std::ostringstream(std::move(src__))
    {
        out_       = src__.out_;
        src__.out_ = nullptr;
        prefix_    = src__.prefix_;
    }
    ~ostream()
    {
        if (out_) {
            auto strings = split(this->str(), '\n');
            for (size_t i = 0; i < strings.size(); i++) {
                if (!(i == strings.size() - 1 && strings[i].size() == 0)) {
                    (*out_) << "[" << prefix_ << "] " << strings[i];
                }
                if (i != strings.size() - 1) {
                    (*out_) << std::endl;
                }
            }
        }
    }
};

#define FILE_LINE std::string(__FILE__) + ":" + std::to_string(__LINE__)

#define RTE_THROW(...)                                                                                                 \
    {                                                                                                                  \
        ::sirius::rte::message_impl(true, __func__, __FILE__, __LINE__, __VA_ARGS__);                                  \
    }

#define RTE_WARNING(...)                                                                                               \
    {                                                                                                                  \
        ::sirius::rte::message_impl(false, __func__, __FILE__, __LINE__, __VA_ARGS__);                                 \
    }

#ifdef NDEBUG
#define RTE_ASSERT(condition__)
#else
#define RTE_ASSERT(condition__)                                                                                        \
    {                                                                                                                  \
        if (!(condition__)) {                                                                                          \
            std::stringstream _s;                                                                                      \
            _s << "Assertion (" << #condition__ << ") failed "                                                         \
               << "at " << __FILE__ << ":" << __LINE__;                                                                \
            RTE_THROW(_s);                                                                                             \
        }                                                                                                              \
    }
#endif

#define RTE_OUT(_out) rte::ostream(_out, std::string(__func__))

} // namespace rte

} // namespace sirius

#endif
