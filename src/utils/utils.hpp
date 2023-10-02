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

/** \file utils.hpp
 *
 *  \brief A collection of utility functions.
 *
 *  General purpose header file containing various helper utility functions. This file should only include
 *  standard headers without any code-specific headers.
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <cassert>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include <complex>
#include <chrono>
#include "json.hpp"
#include "string_tools.hpp"
#include "ostream_tools.hpp"
#include "rte.hpp"

/// Namespace for simple utility functions.
namespace utils {

/// Pack two indices into one for symmetric matrices.
inline int packed_index(int i__, int j__)
{
    /* suppose we have a symmetric matrix: M_{ij} = M_{ji}
           j
       +-------+
       | + + + |
      i|   + + |   -> idx = j * (j + 1) / 2 + i  for  i <= j
       |     + |
       +-------+

       i, j are row and column indices 
    */

    if (i__ > j__) {
        std::swap(i__, j__);
    }
    return j__ * (j__ + 1) / 2 + i__;
}

/// Return the maximum number of blocks (with size 'block_size') needed to split the 'length' elements.
inline int num_blocks(int length__, int block_size__)
{
    return (length__ / block_size__) + std::min(length__ % block_size__, 1);
}

/// Split the 'length' elements into blocks with the initial block size.
/** Return vector of block sizes that sum up to the initial 'length'. */
inline auto split_in_blocks(int length__, int block_size__)
{
    int nb = num_blocks(length__, block_size__);
    /* adjust the block size; this is done to prevent very unequal block sizes */
    /* Take, for example, 21 elements and initial block size of 15. Number of blocks equals 2.
     * Final block size is 21 / 2 + min(1, 21 % 2) = 11. Thus 21 elements will be split in two blocks
     * of 11 and 10 elements. */
    block_size__ = length__ / nb + std::min(1, length__ % nb);

    std::vector<int> result(nb);

    for (int i = 0; i < nb; i++) {
        result[i] = std::min(length__, (i + 1) * block_size__) - i * block_size__;
    }
    /* check for correctness */
    if (std::accumulate(result.begin(), result.end(), 0) != length__) {
        throw std::runtime_error("error in utils::split_in_blocks()");
    }

    return result;
}

///// Check if lambda F(Args) is of type T.
//template <typename T, typename F, typename ...Args>
//constexpr bool check_lambda_type()
//{
//    return std::is_same<typedef std::result_of<F(Args...)>::type, T>::value;
//}

template <typename T, typename OUT>
inline void print_checksum(std::string label__, T value__, OUT&& out__)
{
    out__ << "checksum(" << label__ << ") : " << ffmt(16, 8) << value__ << std::endl;
}

inline void print_hash(std::string label__, unsigned long long int hash__)
{
    std::printf("hash(%s): %llx\n", label__.c_str(), hash__);
}

} // namespace

#endif
