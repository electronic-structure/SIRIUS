// Copyright (c) 2013-2020 Anton Kozhevnikov, Thomas Schulthess
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

/** \file cmd_args.hpp
 *
 *  \brief Contains definition and implementation of cmd_args class.
 */

#ifndef __CMD_ARGS_HPP__
#define __CMD_ARGS_HPP__

#include <vector>
#include <string>
#include <map>
#include <sstream>
#include <algorithm>
#include <iostream>

namespace sirius {

/// Simple command line arguments handler.
class cmd_args
{
  private:
    /// Helper string for each key.
    std::vector<std::pair<std::string, std::string>> key_desc_;

    /// Mapping between a key and its kind (with or without value).
    std::map<std::string, int> known_keys_;

    /// Key to value mapping.
    std::map<std::string, std::string> keys_;

    template <typename T>
    std::vector<T>
    get_vector(std::string const key__) const
    {
        auto s = keys_.at(key__);
        std::replace(s.begin(), s.end(), ':', ' ');
        std::istringstream iss(s);
        std::vector<T> v;
        while (!iss.eof()) {
            T k;
            iss >> k;
            v.push_back(k);
        }
        return v;
    }

    template <typename T, std::size_t N>
    std::array<T, N>
    get_array(std::string const key__) const
    {
        auto v = this->get_vector<T>(key__);
        if (v.size() != N) {
            std::stringstream s;
            s << "[cmd_args::get_array] wrong size of array for the command-line argument " << key__ << std::endl
              << "[cmd_args::get_array] expected size : " << N << ", provided size : " << v.size();
            throw std::runtime_error(s.str());
        } else {
            std::array<T, N> out;
            out.fill(T{}); // prevent compiler warning about uninitialized array
            std::copy(v.begin(), v.end(), out.begin());
            return out;
        }
    }

    void
    check_for_key(std::string const key__) const;

  public:
    /// Constructor.
    cmd_args();

    /// Constructor with the list of keys.
    /** The following example shows how to initialize arguments:
        \code{.cpp}
         cmd_args args(argn, argv, {
             {"device=", "(string) CPU or GPU"},
             {"pw_cutoff=", "(double) plane-wave cutoff for density and potential"},
             {"N=", "(int) cell multiplicity"}
        });
        \endcode
     */
    cmd_args(int argn__, char** argv__, std::initializer_list<std::pair<std::string, std::string>> keys__);

    void
    register_key(std::string const key__, std::string const description__);

    void
    parse_args(int argn__, char** argv__);

    void
    print_help();

    inline bool
    exist(const std::string key__) const
    {
        return keys_.count(key__);
    }

    /// Get a value or terminate if key is not found.
    template <typename T>
    inline T
    value(std::string const key__) const
    {
        check_for_key(key__);
        T v;
        std::istringstream(keys_.at(key__)) >> v;
        return v;
    }

    ///// Get a vector of values or terminate if key is not found.
    // template <typename T>
    // inline std::vector<T> value(std::string const key__) const
    //{
    //     check_for_key(key__);
    //     return get_vector<T>(key__);
    // }

    // template <typename T, std::size_t N>
    // inline std::array<T, N> value(std::string const key__) const
    //{
    //     check_for_key(key__);
    //     return get_array<T, N>(key__);
    // }

    /// Get a value if key exists or return a default value.
    template <typename T>
    inline T
    value(std::string const key__, T default_val__) const
    {
        if (!exist(key__)) {
            return default_val__;
        }
        T v;
        std::istringstream(keys_.at(key__)) >> v;
        return v;
    }

    template <typename T>
    inline std::vector<T>
    value(std::string const key__, std::vector<T> default_val__) const
    {
        if (!exist(key__)) {
            return default_val__;
        }
        return get_vector<T>(key__);
    }

    template <typename T, std::size_t N>
    inline std::array<T, N>
    value(std::string const key__, std::array<T, N> default_val__) const
    {
        if (!exist(key__)) {
            return default_val__;
        }
        return get_array<T, N>(key__);
    }

    std::string
    operator[](const std::string key__) const
    {
        return keys_.at(key__);
    }

    std::map<std::string, std::string>
    keys() const
    {
        return keys_;
    }
};

template <>
inline std::string
cmd_args::value<std::string>(const std::string key__) const
{
    return keys_.at(key__);
}

template <>
inline std::string
cmd_args::value<std::string>(const std::string key__, const std::string default_val__) const
{
    if (!exist(key__)) {
        return default_val__;
    }
    return keys_.at(key__);
}

template <>
inline std::vector<double>
cmd_args::value<std::vector<double>>(const std::string key__) const
{
    check_for_key(key__);
    return get_vector<double>(key__);
}

template <>
inline std::vector<int>
cmd_args::value<std::vector<int>>(const std::string key__) const
{
    check_for_key(key__);
    return get_vector<int>(key__);
}

} // namespace sirius

#endif // __CMD_ARGS_HPP__
