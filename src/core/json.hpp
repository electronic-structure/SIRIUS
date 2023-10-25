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

/** \file json.hpp
 *
 *  \brief Interface to nlohmann::json library and helper functions.
 */

#ifndef __JSON_HPP__
#define __JSON_HPP__

#include <fstream>
#define JSON_USE_IMPLICIT_CONVERSIONS 0
#include "nlohmann_json.hpp"
#include "core/rte/rte.hpp"

/// Read json dictionary from file or string.
/** Terminate if file doesn't exist. */
inline nlohmann::json try_parse(std::istream &is) {
    nlohmann::json dict;

    try {
        is >> dict;
    } catch (std::exception& e) {
        std::stringstream s;
        s << "cannot parse input JSON" << std::endl << e.what();
        RTE_THROW(s);
    }

    return dict;
}

inline nlohmann::json read_json_from_file(std::string const &filename) {
    std::ifstream file{filename};
    if (!file.is_open()) {
        std::stringstream s;
        s << "file " << filename << " can't be opened";
        RTE_THROW(s);
    }

    return try_parse(file);
}

inline nlohmann::json read_json_from_string(std::string const &str) {
    if (str.empty()) {
        return {};
    }
    std::istringstream input{str};
    return try_parse(input);
}

inline nlohmann::json read_json_from_file_or_string(std::string const& str__)
{
    if (str__.empty()) {
        return {};
    }
    // Detect JSON
    if (str__.find("{") == std::string::npos) {
        return read_json_from_file(str__);
    } else {
        return read_json_from_string(str__);
    }
}

#endif /* __JSON_HPP__ */
