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
        Communicator::world().allgather(hash.data(), 1, Communicator::world().rank());
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
            Communicator::world().allgather(hash.data(), 1, r);
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

} // namespace sddk
