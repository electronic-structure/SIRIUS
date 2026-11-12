/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file json.cpp
 *
 *  \brief Implementation of the JSON helper functions.
 */

#include <fstream>
#include <sstream>
#include "core/json.hpp"
#include "core/rte/rte.hpp"

namespace sirius {

nlohmann::json
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

bool
is_json_string(std::string const& str__)
{
    try {
        auto json = nlohmann::json::parse(str__);
        return true;
    } catch (nlohmann::json::parse_error const&) {
    }
    return false;
}

nlohmann::json
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

nlohmann::json
read_json_from_string(std::string const& str)
{
    if (str.empty()) {
        return {};
    }
    std::istringstream input{str};
    return try_parse(input);
}

nlohmann::json
read_json_from_file_or_string(std::string const& str__)
{
    if (str__.empty()) {
        return {};
    }
    /* detect JSON */
    if (str__.find("{") == std::string::npos) {
        return read_json_from_file(str__);
    } else {
        return read_json_from_string(str__);
    }
}

void
write_json_to_file(nlohmann::json const& dict__, std::string const& file_name__)
{
    std::ofstream ofs(file_name__, std::ofstream::out | std::ofstream::trunc);
    ofs << dict__.dump(4);
}

} // namespace sirius
