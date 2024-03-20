/* This file is part of SIRIUS electronic structure library.
 *
 * Copyright (c), ETH Zurich.  All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <sirius.hpp>

using namespace sirius;

void
test1()
{
    int N = 400;
    matrix<std::complex<double>> A({N, N});
    matrix<std::complex<double>> B({N, N});
    matrix<std::complex<double>> C({N, N});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            B(j, i) = random<std::complex<double>>();
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            A(i, j) = B(i, j) + std::conj(B(j, i));
    }
    copy(A, B);

    la::wrap(la::lib_t::lapack).syinv(N, A);
    la::wrap(la::lib_t::blas)
            .hemm('L', 'U', N, N, &la::constant<std::complex<double>>::one(), &A(0, 0), A.ld(), &B(0, 0), B.ld(),
                  &la::constant<std::complex<double>>::zero(), &C(0, 0), C.ld());

    int err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> z = C(i, j);
            if (i == j)
                z -= 1.0;
            if (std::abs(z) > 1e-10)
                err++;
        }
    }

    la::wrap(la::lib_t::blas)
            .hemm('L', 'U', N, N, &la::constant<std::complex<double>>::one(), &A(0, 0), A.ld(), &B(0, 0), B.ld(),
                  &la::constant<std::complex<double>>::zero(), &C(0, 0), C.ld());
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::complex<double> z = C(i, j);
            if (i == j)
                z -= 1.0;
            if (std::abs(z) > 1e-10)
                err++;
        }
    }

    if (err) {
        printf("test1 failed!\n");
        exit(1);
    } else {
        printf("test1 passed!\n");
    }
}

template <typename T>
void
test2()
{
    int N = 400;
    matrix<T> A({N, N});
    matrix<T> B({N, N});
    matrix<T> C({N, N});
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A(j, i) = random<T>();
        }
    }
    copy(A, B);

    la::wrap(la::lib_t::lapack).geinv(N, A);
    la::wrap(la::lib_t::blas)
            .gemm('N', 'N', N, N, N, &la::constant<T>::one(), A.at(memory_t::host), A.ld(), B.at(memory_t::host),
                  B.ld(), &la::constant<T>::zero(), C.at(memory_t::host), C.ld());

    int err = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            T c = C(i, j);
            if (i == j) {
                c -= 1.0;
            }
            if (std::abs(c) > 1e-10) {
                err++;
            }
        }
    }

    if (err) {
        printf("test2 failed!\n");
        exit(1);
    } else {
        printf("test2 passed!\n");
    }
}
// #ifdef SIRIUS_SCALAPACK
// template <typename T>
// void test3()
//{
//     int bs = 32;
//
//     int num_ranks = Communicator::world().size();
//     int nrc = (int)std::sqrt(0.1 + num_ranks);
//     if (nrc * nrc != num_ranks)
//     {
//         printf("wrong mpi grid\n");
//         exit(-1);
//     }
//
//     int N = 400;
//     BLACS_grid blacs_grid(Communicator::world(), nrc, nrc);
//
//     dmatrix<T> A(N, N, blacs_grid, bs, bs);
//     dmatrix<T> B(N, N, blacs_grid, bs, bs);
//     dmatrix<T> C(N, N, blacs_grid, bs, bs);
//     for (int i = 0; i < N; i++)
//     {
//         for (int j = 0; j < N; j++) A.set(j, i, utils::random<T>());
//     }
//     A >> B;
//
//     T alpha = 1.0;
//     T beta = 0.0;
//
//     linalg(lib_t::scalapack).geinv(N, A);
//
//     linalg(lib_t::scalapack).gemm('N', 'N', N, N, N, &alpha, A, 0, 0, B, 0, 0, &beta, C, 0, 0);
//
//     int err = 0;
//     for (int i = 0; i < C.num_cols_local(); i++)
//     {
//         for (int j = 0; j < C.num_rows_local(); j++)
//         {
//             T c = C(j, i);
//             if (C.icol(i) == C.irow(j)) c -= 1.0;
//             if (std::abs(c) > 1e-10) err++;
//         }
//     }
//
//     if (err)
//     {
//         printf("test3 failed!\n");
//         exit(1);
//     }
//     else
//     {
//         printf("test3 passed!\n");
//     }
// }
// #endif

int
main(int argn, char** argv)
{
    sirius::initialize(1);
    test1();
    test2<double>();
    test2<std::complex<double>>();
    // #ifdef SIRIUS_SCALAPACK
    // test3<std::complex<double>>();
    // #endif
    sirius::finalize();
    return 0;
}
