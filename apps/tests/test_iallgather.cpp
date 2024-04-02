/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

void
f()
{
    std::vector<int> v(Platform::num_mpi_ranks(), -1);

    int r = Platform::mpi_rank();

    // std::vector<MPI_Request> send_requests(Platform::num_mpi_ranks());

    for (int i = 0; i < Platform::num_mpi_ranks(); i++) {
        /* send rank id to all other ranks */
        // MPI_Isend(&r, 1, type_wrapper<int>::mpi_type_id(), i, tag, MPI_COMM_WORLD, &send_requests[i]);
        MPI_Send(&r, 1, type_wrapper<int>::mpi_type_id(), i, r, MPI_COMM_WORLD);
    }

    // for (int i = 0; i < Platform::num_mpi_ranks(); i++)
    //{
    //     MPI_Status status;
    //     MPI_Wait(&send_requests[i], &status);
    // }

    std::vector<MPI_Request> recv_requests(Platform::num_mpi_ranks());

    for (int i = 0; i < Platform::num_mpi_ranks(); i++) {
        int tag = i;
        MPI_Irecv(&v[i], 1, type_wrapper<int>::mpi_type_id(), i, tag, MPI_COMM_WORLD, &recv_requests[i]);
    }

    Timer::delay(1);

    for (int i = 0; i < Platform::num_mpi_ranks(); i++) {
        int flg;
        MPI_Status status;
        MPI_Test(&recv_requests[i], &flg, &status);
    }

    //== MPI_Request request;
    //== std::cout << "calling Iallgather" << std::endl;
    //== MPI_Iallgather(&r, 1, type_wrapper<int>::mpi_type_id(), &v[0], 1, type_wrapper<int>::mpi_type_id(),
    //==                MPI_COMM_WORLD, &request);

    //== MPI_Status status;
    //== std::cout << "calling Wait" << std::endl;
    //== MPI_Wait(&request, &status);
    //==
    std::stringstream s;
    s << "Rank : " << Platform::mpi_rank() << ", other ranks : ";
    for (int i = 0; i < Platform::num_mpi_ranks(); i++)
        s << " " << v[i];
    std::cout << s.str() << std::endl;
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);

    if (Platform::mpi_rank() > 2)
        f();

    Platform::barrier();
    Platform::finalize();
}
