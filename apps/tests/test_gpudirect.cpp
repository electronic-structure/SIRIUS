/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.h>

using namespace sirius;

extern "C" void
randomize_on_gpu(double* ptr, size_t size);

void
test1()
{
    int N = 1000;
    int K = 10000;
    mdarray<double_complex, 2> A(N, K);
    mdarray<double_complex, 2> B(K, N);
    mdarray<double_complex, 2> C(N, N);

    for (int j = 0; j < K; j++) {
        for (int i = 0; i < N; i++) {
            A(i, j) = double_complex(1, 1);
            B(j, i) = double_complex(1, 1);
        }
    }
    A.allocate_on_device();
    A.copy_to_device();
    B.allocate_on_device();
    B.copy_to_device();
    C.allocate_on_device();

    blas<gpu>::gemm(0, 0, N, N, K, A.at<gpu>(), A.ld(), B.at<gpu>(), B.ld(), C.at<gpu>(), C.ld());
    Platform::comm_world().allreduce(C.at<gpu>(), (int)C.size());
    C.copy_to_host();

    int nerr = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (std::abs(C(i, j) - double_complex(0, 2 * K * Platform::comm_world().size())) > 1e-12)
                nerr++;
        }
    }

    if (nerr) {
        printf("test1, number of errors: %i\n", nerr);
    } else {
        printf("test1 passed!\n");
    }
}

void
test2()
{
    int N = 1000000;

    mdarray<double_complex, 1> A(N);
    A.allocate_on_device();
    for (int i = 0; i < N; i++)
        A(i) = double_complex(1, 1);

    A.copy_to_device();
    Platform::comm_world().allreduce(A.at<gpu>(), (int)A.size());
    A.copy_to_host();

    int nerr = 0;
    for (int i = 0; i < N; i++) {
        double d = std::abs(A(i) - double_complex(Platform::comm_world().size(), Platform::comm_world().size()));
        if (d > 1e-12)
            nerr++;
    }

    if (nerr) {
        printf("test2, number of errors: %i\n", nerr);
    } else {
        printf("test2 passed!\n");
    }
}

void
test3()
{
    mdarray<double_complex, 1> A(1000000);
    A.allocate_on_device();
    randomize_on_gpu((double*)A.at<gpu>(), A.size() * 2);

    A.copy_to_host();
    Platform::comm_world().allreduce(A.at<cpu>(), (int)A.size());

    mdarray<double_complex, 1> A_ref(1000000);
    A >> A_ref;

    Platform::comm_world().allreduce(A.at<gpu>(), (int)A.size());
    A.copy_to_host();

    for (int i = 0; i < 1000000; i++) {
        double d = std::abs(A(i) - A_ref(i));
        if (d > 1e-8)
            INFO << "i=" << i << " diff=" << d << std::endl;
    }
}

int
main(int argn, char** argv)
{
    Platform::initialize(1);
    test1();
    test2();
    test3();
    test1();
    test2();
    test3();
    Platform::finalize();
}
