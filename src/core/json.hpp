/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
inline nlohmann::json
try_parse(std::istream& is)
{
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

inline nlohmann::json
read_json_from_file(std::string const& filename)
{
    std::ifstream file{filename};
    if (!file.is_open()) {
        std::stringstream s;
        s << "file " << filename << " can't be opened";
        RTE_THROW(s);
    }

    return try_parse(file);
}

inline nlohmann::json
read_json_from_string(std::string const& str)
{
    if (str.empty()) {
        return {};
    }
    std::istringstream input{str};
    return try_parse(input);
}

inline nlohmann::json
read_json_from_file_or_string(std::string const& str__)
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
