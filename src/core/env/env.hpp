/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
#include <memory>
#include <sstream>

namespace sirius {

/// Get environment variables.
namespace env {

/// Check for environment variable and return a pointer to a stored value if found or a null-pointer if not.
template <typename T>
inline T const*
get_value_ptr(std::string const& name__)
{
    static std::map<std::string, std::unique_ptr<T>> map_name;
    if (map_name.count(name__) == 0) {
        /* first time the function is called */
        const char* raw_str = std::getenv(name__.c_str());
        if (raw_str == NULL) {
            map_name[name__] = nullptr;
        } else {
            map_name[name__] = std::make_unique<T>();
            std::istringstream(std::string(raw_str)) >> (*map_name[name__]);
        }
    }
    return map_name[name__].get();
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

inline bool
print_hash()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_HASH");
    return val && *val;
}

inline bool
print_mpi_layout()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_MPI_LAYOUT");
    return val && *val;
}

inline bool
print_memory_usage()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_MEMORY_USAGE");
    return val && *val;
}

inline int
print_timing()
{
    auto val = get_value_ptr<int>("SIRIUS_PRINT_TIMING");
    if (val) {
        return *val;
    } else {
        return 0;
    }
}

inline std::string
save_config()
{
    auto val = get_value_ptr<std::string>("SIRIUS_SAVE_CONFIG");
    if (val) {
        return *val;
    } else {
        return "";
    }
}

inline std::string
config_file()
{
    auto val = get_value_ptr<std::string>("SIRIUS_CONFIG");
    if (val) {
        return *val;
    } else {
        return "";
    }
}

inline std::string
get_ev_solver()
{
    auto val = get_value_ptr<std::string>("SIRIUS_EV_SOLVER");
    if (val) {
        return *val;
    } else {
        return "";
    }
}

inline int
get_verbosity()
{
    auto verb_lvl = env::get_value_ptr<int>("SIRIUS_VERBOSITY");
    if (verb_lvl) {
        return *verb_lvl;
    } else {
        return 0;
    }
}

inline bool
check_scf_density()
{
    auto val = get_value_ptr<int>("SIRIUS_CHECK_SCF_DENSITY");
    return val && *val;
}

} // namespace env

} // namespace sirius
#endif
