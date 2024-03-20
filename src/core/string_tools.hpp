/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file string_tools.hpp
 *
 *  \brief Extra functions to work with std::strings
 */

#ifndef __STRING_TOOLS_HPP__
#define __STRING_TOOLS_HPP__

namespace sirius {

/// Split multi-line string into a list of strings.
inline auto
split(std::string const str__, char delim__)
{
    std::istringstream iss(str__);
    std::vector<std::string> result;

    while (iss.good()) {
        std::string s;
        std::getline(iss, s, delim__);
        result.push_back(s);
    }
    return result;
}

inline std::string&
ltrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(0, str.find_first_not_of(chars));
    return str;
}

inline std::string&
rtrim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    str.erase(str.find_last_not_of(chars) + 1);
    return str;
}

inline std::string&
trim(std::string& str, const std::string& chars = "\t\n\v\f\r ")
{
    return ltrim(rtrim(str, chars), chars);
}

} // namespace sirius
#endif
