/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

/** \file communicator.cpp
 *
 *  \brief Definitions.
 *
 */

#include "communicator.hpp"

namespace sirius {

namespace mpi {

int
num_ranks_per_node()
{
    static int num_ranks{-1};
    if (num_ranks == -1) {
        char name[MPI_MAX_PROCESSOR_NAME];
        int len;
        CALL_MPI(MPI_Get_processor_name, (name, &len));
        std::vector<size_t> hash(mpi::Communicator::world().size());
        hash[mpi::Communicator::world().rank()] = std::hash<std::string>{}(std::string(name, len));
        mpi::Communicator::world().allgather(hash.data(), 1, mpi::Communicator::world().rank());
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

int
get_device_id(int num_devices__)
{
    static int id{-1};
    if (num_devices__ == 0) {
        return id;
    }
    if (id == -1) {
        #pragma omp single
        {
            int r = mpi::Communicator::world().rank();
            char name[MPI_MAX_PROCESSOR_NAME];
            int len;
            CALL_MPI(MPI_Get_processor_name, (name, &len));
            std::vector<size_t> hash(mpi::Communicator::world().size());
            hash[r] = std::hash<std::string>{}(std::string(name, len));
            mpi::Communicator::world().allgather(hash.data(), 1, r);
            std::map<size_t, std::vector<int>> rank_map;
            for (int i = 0; i < mpi::Communicator::world().size(); i++) {
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

} // namespace mpi

} // namespace sirius
