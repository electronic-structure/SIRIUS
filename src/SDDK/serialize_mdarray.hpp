// Copyright (c) 2013-2021 Anton Kozhevnikov, Thomas Schulthess
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

/** \file serialize_mdarray.hpp
 *
 *  \brief Serialize madarray to json.
 */

#ifndef __SERIALIZE_MDARRAY_HPP__
#define __SERIALIZE_MDARRAY_HPP__

#include "memory.hpp"
#include "utils/json.hpp"

namespace sirius {

template <typename T, int N>
inline nlohmann::json
serialize(sddk::mdarray<T, N> const& a__)
{
    nlohmann::json dict;
    std::array<sddk::mdarray_index_descriptor::index_type, N> begin;
    std::array<sddk::mdarray_index_descriptor::index_type, N> end;
    for (int i = 0; i < N; i++) {
        begin[i] = a__.dim(i).begin();
        end[i] = a__.dim(i).begin();
    }
    dict["begin"] = begin;
    dict["end"] = end;
    dict["data"] = std::vector<T>(a__.size());
    for (size_t i = 0; i < a__.size(); i++) {
        dict["data"][i] = a__[i];
    }
    return dict;
}

template <int N>
inline nlohmann::json
serialize(sddk::mdarray<double_complex, N> const& a__)
{
    nlohmann::json dict;
    std::array<sddk::mdarray_index_descriptor::index_type, N> begin;
    std::array<sddk::mdarray_index_descriptor::index_type, N> end;
    for (int i = 0; i < N; i++) {
        begin[i] = a__.dim(i).begin();
        end[i] = a__.dim(i).begin();
    }
    dict["begin"] = begin;
    dict["end"] = end;
    dict["data"] = std::vector<double>(2 * a__.size());
    for (size_t i = 0; i < a__.size(); i++) {
        dict["data"][2 * i] = a__[i].real();
        dict["data"][2 * i + 1] = a__[i].imag();
    }
    return dict;
}

template <typename T, int N>
void
write_to_json_file(sddk::mdarray<T, N> const& a__, std::string const& fname__)
{
    try {
        auto dict = serialize(a__);
        std::ofstream ofs(fname__, std::ofstream::out | std::ofstream::trunc);
        ofs << dict.dump(4);
    } catch(...) {
        std::stringstream s;
        s << "Error writing mdarray to file " << fname__;
        printf("%s\n", s.str().c_str());
    }
}

}

#endif

