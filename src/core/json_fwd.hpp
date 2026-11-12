/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file json_fwd.hpp
 *
 *  \brief Forward declaration of nlohmann::json and declarations of the JSON helper functions.
 *
 *  Include this file in headers that only need to name nlohmann::json. Include "core/json.hpp" in the
 *  source files that construct, access or serialize json objects.
 */

#ifndef __JSON_FWD_HPP__
#define __JSON_FWD_HPP__

#include <iosfwd>
#include <string>
#include "core/nlohmann_json_fwd.hpp"

namespace sirius {

/// Read json dictionary from the input stream.
/** Throw if the stream can't be parsed. */
nlohmann::json
try_parse(std::istream& is);

/// Check if the string contains a valid json document.
bool
is_json_string(std::string const& str__);

/// Read json dictionary from file. Throw if the file can't be opened.
nlohmann::json
read_json_from_file(std::string const& filename);

/// Read json dictionary from string. Return an empty dictionary if the string is empty.
nlohmann::json
read_json_from_string(std::string const& str);

/// Read json dictionary from file or from a string, depending on whether the input contains a "{".
nlohmann::json
read_json_from_file_or_string(std::string const& str__);

/// Write json dictionary to a file.
void
write_json_to_file(nlohmann::json const& dict__, std::string const& file_name__);

} // namespace sirius

#endif /* __JSON_FWD_HPP__ */
