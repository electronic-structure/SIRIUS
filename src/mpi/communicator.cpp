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

/** \file communicator.cpp
 *
 *  \brief Definitions.
 *
 */

#include "communicator.hpp"

namespace sddk {

int num_ranks_per_node()
{
    static int num_ranks{-1};
    if (num_ranks == -1) {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        CALL_MPI(MPI_Get_processor_name, (name, &len));
        std::vector<size_t> hash(Communicator::world().size());
        hash[Communicator::world().rank()] = std::hash<std::string>{}(std::string(name, len));
        Communicator::world().allgather(hash.data(), Communicator::world().rank(), 1);
        std::sort(hash.begin(), hash.end());

        int n{1};
        for (int i = 1; i < (int)hash.size(); i++) {
            if (hash[i] == hash.front()) {
                n++;
            } else {
                break;
            }
        }
        int m{1};
        for (int i = (int)hash.size() - 2; i >= 0; i--) {
            if (hash[i] == hash.back()) {
                m++;
            } else {
                break;
            }
        }
        num_ranks = std::max(n, m);
    }

    return num_ranks;
}

int get_device_id(int num_devices__)
{
    static int id{-1};
    if (num_devices__ == 0) {
        return id;
    }
    if (id == -1) {
        #pragma omp single
        {
            int r = Communicator::world().rank();
            char name[MPI_MAX_PROCESSOR_NAME];
            int len;
            CALL_MPI(MPI_Get_processor_name, (name, &len));
            std::vector<size_t> hash(Communicator::world().size());
            hash[r] = std::hash<std::string>{}(std::string(name, len));
            Communicator::world().allgather(hash.data(), r, 1);
            std::map<size_t, std::vector<int>> rank_map;
            for (int i = 0; i < Communicator::world().size(); i++) {
                rank_map[hash[i]].push_back(i);
            }
            for (int i = 0; i < (int)rank_map[hash[r]].size(); i++) {
                if (rank_map[hash[r]][i] == r) {
                    id = i % num_devices__;
                    break;
                }
            }
        }
        assert(id >= 0);
    }
    return id;
}

void sddk::pstdout::printf(const char* fmt, ...)
{
    std::vector<char> str(1024); // assume that one printf will not output more than this

    std::va_list arg;
    va_start(arg, fmt);
    int n = vsnprintf(&str[0], str.size(), fmt, arg);
    va_end(arg);

    n = std::min(n, (int)str.size());

    if ((int)buffer_.size() - count_ < n) {
        buffer_.resize(buffer_.size() + str.size());
    }
    std::memcpy(&buffer_[count_], &str[0], n);
    count_ += n;
}

void sddk::pstdout::flush()
{
    std::vector<int> counts(comm_.size());
    comm_.allgather(&count_, counts.data(), comm_.rank(), 1);

    int offset{0};
    for (int i = 0; i < comm_.rank(); i++) {
        offset += counts[i];
    }

    /* total size of the output buffer */
    int sz = count_;
    comm_.allreduce(&sz, 1);

    if (sz != 0) {
        std::vector<char> outb(sz + 1);
        comm_.allgather(&buffer_[0], &outb[0], offset, count_);
        outb[sz] = 0;

        if (comm_.rank() == 0) {
            std::printf("%s", &outb[0]);
        }
    }
    count_ = 0;
}
} // namespace sddk
