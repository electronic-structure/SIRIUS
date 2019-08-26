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

namespace utils {

/// Check for environment variable and return a pointer to a stored value if found or a null-pointer if not.
template <typename T>
inline T const* get_env(std::string const& name__)
{
    static std::map<std::string, std::pair<bool, T>> map_name;
    if (map_name.count(name__) == 0) {
        /* first time the function is called */
        const char* raw_str = std::getenv(name__.c_str());
        if (raw_str == NULL) {
            map_name[name__] = std::make_pair(false, T());
        } else {
            T var;
            std::istringstream(std::string(raw_str)) >> var;
            map_name[name__] = std::make_pair(true, var);
        }
    }
    if (map_name[name__].first == false) {
        return nullptr;
    } else {
        return &map_name[name__].second;
    }
}

} // namespace utils

#endif
