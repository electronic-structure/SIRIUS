/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file serialize_mdarray.hpp
 *
 *  \brief Serialize madarray to json.
 */

#ifndef __SERIALIZE_MDARRAY_HPP__
#define __SERIALIZE_MDARRAY_HPP__

#include "core/memory.hpp"
#include "core/json.hpp"

namespace sirius {

template <typename T, int N>
inline nlohmann::json
serialize(mdarray<T, N> const& a__)
{
    nlohmann::json dict;
    std::array<index_range::index_type, N> begin;
    std::array<index_range::index_type, N> end;
    for (int i = 0; i < N; i++) {
        begin[i] = a__.dim(i).begin();
        end[i]   = a__.dim(i).end();
    }
    dict["begin"] = begin;
    dict["end"]   = end;
    dict["data"]  = std::vector<T>(a__.size());
    for (size_t i = 0; i < a__.size(); i++) {
        dict["data"][i] = a__[i];
    }
    return dict;
}

template <typename T, int N>
inline nlohmann::json
serialize(mdarray<std::complex<T>, N> const& a__)
{
    nlohmann::json dict;
    std::array<index_range::index_type, N> begin;
    std::array<index_range::index_type, N> end;
    for (int i = 0; i < N; i++) {
        begin[i] = a__.dim(i).begin();
        end[i]   = a__.dim(i).end();
    }
    dict["begin"] = begin;
    dict["end"]   = end;
    dict["data"]  = std::vector<T>(2 * a__.size());
    for (size_t i = 0; i < a__.size(); i++) {
        dict["data"][2 * i]     = a__[i].real();
        dict["data"][2 * i + 1] = a__[i].imag();
    }
    return dict;
}

template <typename T, int N>
void
write_to_json_file(mdarray<T, N> const& a__, std::string const& fname__)
{
    try {
        auto dict = serialize(a__);
        std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    } catch (...) {
        std::stringstream s;
        s << "Error writing mdarray to file " << fname__;
        printf("%s\n", s.str().c_str());
    }
}

} // namespace sirius

#endif
