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

/** \file env.hpp
 *
 *  \brief Get the environment variables
 */

#ifndef __ENV_HPP__
#define __ENV_HPP__

#include <cstdlib>
#include <string>
#include <algorithm>
#include <map>
#include <sstream>

namespace env {

/// Check for environment variable and return a pointer to a stored value if found or a null-pointer if not.
template <typename T>
inline T const* get_value_ptr(std::string const& name__)
{
    static std::map<std::string, T*> map_name;
    if (map_name.count(name__) == 0) {
        /* first time the function is called */
        const char* raw_str = std::getenv(name__.c_str());
        if (raw_str == NULL) {
            map_name[name__] = nullptr;
        } else {
            map_name[name__] = new T;
            std::istringstream(std::string(raw_str)) >> (*map_name[name__]);
        }
    }
    return map_name[name__];
}

inline bool
print_performance()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_PERFORMANCE");
    return val && *val;
}

inline bool
print_checksum()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_CHECKSUM");
    return val && *val;
}

} // namespace env

#endif
